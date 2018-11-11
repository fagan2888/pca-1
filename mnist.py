import click
import requests
from scipy.io import loadmat

@click.group()
def cli():
    pass


def fetch_mnist(url="https://github.com/noahgift/pca/blob/master/data/mnist-original.mat?raw=true"):
    """Fetch MNIST from Github and Assert Correct Shape"""
    
    response = requests.get(url)
    mnist_local = "mnist-original.mat"
    with open(mnist_local, "wb") as my_file:
        my_file.write(response.content)
    mnist_raw = loadmat(mnist_local)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    #Assert this is the correct shape
    X = mnist["data"]
    y = mnist["target"]
    assert X.shape == (70000, 784)
    assert y.shape == (70000,)
    return X,y 

@cli.command()
@click.option("--url", 
    default="https://github.com/noahgift/pca/blob/master/data/mnist-original.mat?raw=true",
    help="Download mnist.mat, can take a custom url")
def download(url):
    """Fetches mnist.mat
    
    Run it like this:


    `python mnist.py download`


    Downloading MNIST from: https://github.com/noahgift/pca/blob/master/data/mnist-original.mat?raw=true
    X Shape: (70000, 784)
    y Shape: (70000,)
    """

    click.echo(f"Downloading MNIST from: {url}")
    X,y = fetch_mnist(url)
    click.echo(f"X Shape: {X.shape}")
    click.echo(f"y Shape: {y.shape}")

if __name__ == "__main__":
    cli()