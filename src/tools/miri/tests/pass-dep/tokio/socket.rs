//@only-target: linux # We only support tokio on Linux
//@compile-flags: -Zmiri-disable-isolation

use tokio::io::{self, AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const TEST_BYTES: &[u8] = b"these are some test bytes!";

#[tokio::main]
async fn main() {
    test_accept_and_connect().await;
    test_read_write().await;
    test_echo_server().await;
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

// Test that copying every socket input directly into it's output works.
// By this, we test whether Miri correctly removes read readiness when we
// have a short read (meaning the buffer is empty).
// This test case is heavily inspired by the `tcp_echo` test case of the tokio
// test suite.
async fn test_echo_server() {
    const ITER: usize = 4;

    let (tx, rx) = tokio::sync::oneshot::channel();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn the client thread.
    tokio::spawn(async move {
        let mut stream = TcpStream::connect(&addr).await.unwrap();

        for _ in 0..ITER {
            // Write some bytes into the socket.
            stream.write_all(TEST_BYTES).await.unwrap();
            // Expect to read the same bytes again.
            let mut buf = [0; TEST_BYTES.len()];
            stream.read_exact(&mut buf).await.unwrap();
            assert_eq!(&buf[..], TEST_BYTES);
        }

        tx.send(()).unwrap();
    });

    let (mut stream, _) = listener.accept().await.unwrap();
    let (mut rd, mut wr) = stream.split();

    let n = io::copy(&mut rd, &mut wr).await.unwrap() as usize;
    assert_eq!(n, ITER * TEST_BYTES.len());

    rx.await.unwrap();
}
