//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation -Zmiri-fixed-schedule

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_accept_nonblock();
    test_send_recv_nonblock();
}

/// Test that nonblocking TCP server sockets return [`ErrorKind::WouldBlock`] when trying
/// to accept when no incoming connection exists. This also tests that nonblocking server sockets
/// are still able to accept incoming connections should they already exist before [`TcpListener::accept`]
/// is called.
fn test_accept_nonblock() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Make server non-blocking.
    listener.set_nonblocking(true).unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    // Accepting when no incoming connecting exists should block.
    let err = listener.accept().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Start server thread.
    let handle = thread::spawn(move || {
        // Accepting when there is an existing incoming connection should
        // succeed without blocking.

        let (_stream, _peer_addr) = listener.accept().unwrap();
    });

    // The connect is blocking and thus we yield to the server thread.
    let _stream = TcpStream::connect(address).unwrap();

    handle.join().unwrap();
}

/// Test sending bytes into and receiving bytes from a connected stream without blocking.
fn test_send_recv_nonblock() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    // Start server thread.
    let handle = thread::spawn(move || {
        let (mut stream, _addr) = listener.accept().unwrap();

        // Yield back to client thread to ensure that the first read
        // is before we write anything into the socket.
        thread::yield_now();

        stream.write_all(TEST_BYTES).unwrap();
    });

    // The connect is blocking and thus we yield to the server thread.
    let mut stream = TcpStream::connect(address).unwrap();
    // Make client non-blocking.
    stream.set_nonblocking(true).unwrap();
    let mut buffer = [0; TEST_BYTES.len()];
    // Reading when no data was written should return WouldBlock.
    let err = stream.read_exact(&mut buffer).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Try to read bytes from the peer socket without blocking.
    // Since the peer socket might do partial writes, we might need to
    // sleep multiple times until we read everything.

    let mut bytes_read = 0;
    while bytes_read != TEST_BYTES.len() {
        match stream.read(&mut buffer[bytes_read..]) {
            Ok(read) => bytes_read += read,
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                // Not all data is written into the stream, yield to the server thread
                // to write more into the stream.
                thread::yield_now();
            }
            Err(err) => panic!("unexpected error whilst reading: {err}"),
        }
    }

    assert_eq!(&buffer, TEST_BYTES);

    handle.join().unwrap();
}
