//@ignore-target: windows # No libc socketpair on Windows
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::io::ErrorKind;
use std::thread;
use std::time::Duration;

use libc_utils::*;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_fcntl_nonblock_opt();
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    test_sock_nonblock_opt();
    #[cfg(not(any(target_os = "solaris", target_os = "illumos")))]
    test_ioctl_fionbio_op();

    test_send_recv_nonblock();
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    test_send_recv_dontwait();
    test_write_read_nonblock();
}

/// Test that setting the O_NONBLOCK flag changes the blocking state of a socket
/// from a socket pair.
fn test_fcntl_nonblock_opt() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Initially, both sides should have O_RDWR and nothing else.
    assert_eq!(errno_result(unsafe { libc::fcntl(fds[0], libc::F_GETFL) }).unwrap(), libc::O_RDWR);
    assert_eq!(errno_result(unsafe { libc::fcntl(fds[1], libc::F_GETFL) }).unwrap(), libc::O_RDWR);

    // Change socket to be non-blocking.
    unsafe { errno_check(libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK)) };

    let flags_sock1 = unsafe { errno_result(libc::fcntl(fds[0], libc::F_GETFL, 0)).unwrap() };
    let flags_sock2 = unsafe { errno_result(libc::fcntl(fds[1], libc::F_GETFL, 0)).unwrap() };
    // Ensure that the socket is really non-blocking.
    assert_eq!(flags_sock1 & libc::O_NONBLOCK, libc::O_NONBLOCK);
    // Ensure that the other socket is still blocking.
    assert_eq!(flags_sock2 & libc::O_NONBLOCK, 0);

    // Change socket back to be blocking.
    unsafe { errno_check(libc::fcntl(fds[0], libc::F_SETFL, 0)) };

    let flags = unsafe { errno_result(libc::fcntl(fds[0], libc::F_GETFL, 0)).unwrap() };
    // Ensure that the socket is really blocking.
    assert_eq!(flags & libc::O_NONBLOCK, 0);
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "solaris",
    target_os = "illumos"
))]
/// Test creating a pair of non-blocking sockets by using the SOCK_NONBLOCK option
/// for the `socketpair` syscall.
fn test_sock_nonblock_opt() {
    let mut fds = [-1, -1];
    errno_check(unsafe {
        libc::socketpair(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK,
            0,
            fds.as_mut_ptr(),
        )
    });

    let flags_sock1 = unsafe { errno_result(libc::fcntl(fds[0], libc::F_GETFL, 0)).unwrap() };
    let flags_sock2 = unsafe { errno_result(libc::fcntl(fds[1], libc::F_GETFL, 0)).unwrap() };
    // Ensure that both sockets are really non-blocking.
    assert_eq!(flags_sock1 & libc::O_NONBLOCK, libc::O_NONBLOCK);
    assert_eq!(flags_sock2 & libc::O_NONBLOCK, libc::O_NONBLOCK);
}

#[cfg(not(any(target_os = "solaris", target_os = "illumos")))]
/// Test changing the blocking state of a socket from a socket pair using the
/// `ioctl(fd, FIONBIO, ...)` syscall.
fn test_ioctl_fionbio_op() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    unsafe {
        // Change socket to be non-blocking.
        let mut value = 1 as libc::c_int;
        errno_check(libc::ioctl(fds[0], libc::FIONBIO, &mut value));
    }

    let flags_sock1 = unsafe { errno_result(libc::fcntl(fds[0], libc::F_GETFL, 0)).unwrap() };
    let flags_sock2 = unsafe { errno_result(libc::fcntl(fds[1], libc::F_GETFL, 0)).unwrap() };
    // Ensure that the socket is really non-blocking.
    assert_eq!(flags_sock1 & libc::O_NONBLOCK, libc::O_NONBLOCK);
    // Ensure that the other socket is still blocking.
    assert_eq!(flags_sock2 & libc::O_NONBLOCK, 0);

    unsafe {
        // Change socket back to be blocking.
        let mut value = 0 as libc::c_int;
        errno_check(libc::ioctl(fds[0], libc::FIONBIO, &mut value));
    }

    let flags = unsafe { errno_result(libc::fcntl(fds[0], libc::F_GETFL, 0)).unwrap() };
    // Ensure that the socket is really blocking.
    assert_eq!(flags & libc::O_NONBLOCK, 0);
}

/// Test sending bytes into and receiving bytes from a connected stream without blocking.
fn test_send_recv_nonblock() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    let sock1fd = fds[0]; // our client socket.
    let sock2fd = fds[1]; // our peer socket on the "server" which is connected to the client socket.

    // Change client socket to be non-blocking.
    unsafe { errno_check(libc::fcntl(sock1fd, libc::F_SETFL, libc::O_NONBLOCK)) };

    // Spawn the "server" thread. The main thread is the "client" thread.
    let server_thread = thread::spawn(move || {
        // `sock2fd` is a blocking socket now. But that's okay, the client still does non-blocking
        // reads/writes.

        // Yield back to client so that it starts receiving before we start sending.
        thread::sleep(Duration::from_millis(10));

        unsafe {
            libc_utils::write_all_generic(
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(sock2fd, buf, count, 0),
            )
            .unwrap()
        };

        // The buffer should contain `TEST_BYTES` at the beginning.
        // This will block until the client sent us this data.
        let mut buffer = [0; TEST_BYTES.len()];
        unsafe {
            libc_utils::read_exact_generic(
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                libc_utils::NoRetry,
                |buf, count| libc::recv(sock2fd, buf, count, 0),
            )
            .unwrap()
        };
        assert_eq!(&buffer, TEST_BYTES);
    });

    let mut buffer = [0; TEST_BYTES.len()];
    // Receiving from a socket when the peer is not writing is
    // not possible without blocking.
    let err = unsafe {
        errno_result(libc::recv(sock1fd, buffer.as_mut_ptr().cast(), buffer.len(), 0)).unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Try to receive bytes from the peer socket without blocking.
    // Since the peer socket might do partial writes, we might need to
    // sleep multiple times until we received everything.

    unsafe {
        libc_utils::read_exact_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::RetryAfter(Duration::from_millis(10)),
            |buf, count| libc::recv(sock1fd, buf, count, 0),
        )
        .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    // Test non-blocking writing.

    // Sending into the empty buffer should succeed without blocking.
    unsafe {
        libc_utils::write_all_generic(
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            libc_utils::NoRetry,
            |buf, count| libc::send(sock1fd, buf, count, 0),
        )
        .unwrap()
    };

    let fill_buf = [1u8; 32_000];
    // Keep sending data until the buffer is full and we would block.
    loop {
        let result = unsafe {
            libc_utils::write_all_generic(
                fill_buf.as_ptr().cast(),
                fill_buf.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(sock1fd, buf, count, 0),
            )
        };

        match result {
            Ok(_) => { /* continue to fill buffer */ }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => panic!("unexpected error whilst filling up buffer: {err}"),
        }
    }

    server_thread.join().unwrap();
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "solaris",
    target_os = "illumos"
))]
/// Test sending bytes into and receiving bytes from a par of stream sockets without blocking.
/// Instead of using non-blocking sockets, we test whether it works with blocking sockets
/// when passing the `libc::MSG_DONTWAIT` flag to the send and receive calls.
fn test_send_recv_dontwait() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    let sock1fd = fds[0]; // our client socket.
    let sock2fd = fds[1]; // our peer socket on the "server" which is connected to the client socket.

    // Spawn the "server" thread. The main thread is the "client" thread.
    let server_thread = thread::spawn(move || {
        // Similar to above we use blocking operations on the server side.

        // Yield back to client so that it starts receiving before we start sending.
        thread::sleep(Duration::from_millis(10));

        unsafe {
            libc_utils::write_all_generic(
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(sock2fd, buf, count, 0),
            )
            .unwrap()
        };

        // The buffer should contain `TEST_BYTES` at the beginning.
        // This will block until the client sent us this data.
        let mut buffer = [0; TEST_BYTES.len()];
        unsafe {
            libc_utils::read_exact_generic(
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                libc_utils::NoRetry,
                |buf, count| libc::recv(sock2fd, buf, count, 0),
            )
            .unwrap()
        };
        assert_eq!(&buffer, TEST_BYTES);
    });

    let mut buffer = [0; TEST_BYTES.len()];
    // Receiving from a socket when the peer is not writing is
    // not possible without blocking.
    let err = unsafe {
        errno_result(libc::recv(
            sock1fd,
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc::MSG_DONTWAIT,
        ))
        .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Try to receive bytes from the peer socket without blocking.
    // Since the peer socket might do partial writes, we might need to
    // sleep multiple times until we received everything.

    unsafe {
        libc_utils::read_exact_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::RetryAfter(Duration::from_millis(10)),
            |buf, count| libc::recv(sock1fd, buf, count, libc::MSG_DONTWAIT),
        )
        .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    // Test non-blocking writing.

    // Sending into the empty buffer should succeed without blocking.
    unsafe {
        libc_utils::write_all_generic(
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            libc_utils::NoRetry,
            |buf, count| libc::send(sock1fd, buf, count, libc::MSG_DONTWAIT),
        )
        .unwrap()
    };

    let fill_buf = [1u8; 32_000];
    // Keep sending data until the buffer is full and we would block.
    loop {
        let result = unsafe {
            libc_utils::write_all_generic(
                fill_buf.as_ptr().cast(),
                fill_buf.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(sock1fd, buf, count, libc::MSG_DONTWAIT),
            )
        };

        match result {
            Ok(_) => { /* continue to fill buffer */ }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => panic!("unexpected error whilst filling up buffer: {err}"),
        }
    }

    server_thread.join().unwrap();
}

/// Test writing bytes into and reading bytes from a stream socket pair without blocking.
fn test_write_read_nonblock() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    let sock1fd = fds[0]; // our client socket.
    let sock2fd = fds[1]; // our peer socket on the "server" which is connected to the client socket.

    // Change client socket to be non-blocking.
    unsafe { errno_check(libc::fcntl(sock1fd, libc::F_SETFL, libc::O_NONBLOCK)) };

    // Spawn the "server" thread. The main thread is the "client" thread.
    let server_thread = thread::spawn(move || {
        // Similar to above we use blocking operations on the server side.

        // Yield back to client so that it starts receiving before we start sending.
        thread::sleep(Duration::from_millis(10));

        libc_utils::write_all(sock2fd, TEST_BYTES).unwrap();

        // The buffer should contain `TEST_BYTES` at the beginning.
        // This will block until the client sent us this data.
        let mut buffer = [0; TEST_BYTES.len()];
        libc_utils::read_exact(sock2fd, &mut buffer).unwrap();
        assert_eq!(&buffer, TEST_BYTES);
    });

    let mut buffer = [0; TEST_BYTES.len()];
    // Reading from a socket when the peer is not writing is
    // not possible without blocking.
    let err = unsafe {
        errno_result(libc::read(sock1fd, buffer.as_mut_ptr() as *mut libc::c_void, buffer.len()))
            .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Try to read bytes from the peer socket without blocking.
    // Since the peer socket might do partial writes, we might need to
    // sleep multiple times until we read everything.

    unsafe {
        libc_utils::read_exact_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::RetryAfter(Duration::from_millis(10)),
            |buf, count| libc::read(sock1fd, buf, count),
        )
        .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    // Now we test non-blocking writing.

    // Writing into the empty buffer should succeed without blocking.
    libc_utils::write_all(sock1fd, TEST_BYTES).unwrap();

    let fill_buf = [1u8; 32_000];
    // Keep sending data until the buffer is full and we would block.
    loop {
        let result = unsafe {
            libc_utils::write_all_generic(
                fill_buf.as_ptr().cast(),
                fill_buf.len(),
                libc_utils::NoRetry,
                |buf, count| libc::write(sock1fd, buf, count),
            )
        };

        match result {
            Ok(_) => { /* continue to fill buffer */ }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => panic!("unexpected error whilst filling up buffer: {err}"),
        }
    }

    server_thread.join().unwrap();
}
