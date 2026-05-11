//@only-target: linux # We only support tokio on Linux
//@compile-flags: -Zmiri-disable-isolation

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const TEST_BYTES: &[u8] = b"these are some test bytes!";

#[tokio::main]
async fn main() {
    test_accept_and_connect().await;
    test_read_write().await;
}

/// Test connecting and accepting a connection.
async fn test_accept_and_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    // Start server thread.
    tokio::spawn(async move {
        let (_stream, _addr) = listener.accept().await.unwrap();
    });

    let _stream = TcpStream::connect(address).await.unwrap();
}

/// Test writing bytes into and reading bytes from a connected stream.
async fn test_read_write() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    // Start server thread.
    tokio::spawn(async move {
        let (mut stream, _addr) = listener.accept().await.unwrap();

        stream.write_all(TEST_BYTES).await.unwrap();

        let mut buffer = [0; TEST_BYTES.len()];
        stream.read_exact(&mut buffer).await.unwrap();

        assert_eq!(&buffer, TEST_BYTES);
    });

    let mut stream = TcpStream::connect(address).await.unwrap();

    let mut buffer = [0; TEST_BYTES.len()];
    stream.read_exact(&mut buffer).await.unwrap();
    assert_eq!(&buffer, TEST_BYTES);

    stream.write_all(TEST_BYTES).await.unwrap();
}
