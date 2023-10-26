use super::*;
use crate::io::prelude::*;
use crate::io::{self, ErrorKind, IoSlice, IoSliceMut};
#[cfg(any(target_os = "android", target_os = "linux"))]
use crate::os::unix::io::AsRawFd;
use crate::sys_common::io::test::tmpdir;
use crate::thread;
use crate::time::Duration;

#[cfg(target_os = "android")]
use crate::os::android::net::SocketAddrExt;

#[cfg(target_os = "linux")]
use crate::os::linux::net::SocketAddrExt;

macro_rules! or_panic {
    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{e}"),
        }
    };
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn basic() {
    let dir = tmpdir();
    let socket_path = dir.path().join("sock");
    let msg1 = b"hello";
    let msg2 = b"world!";

    let listener = or_panic!(UnixListener::bind(&socket_path));
    let thread = thread::spawn(move || {
        let mut stream = or_panic!(listener.accept()).0;
        let mut buf = [0; 5];
        or_panic!(stream.read(&mut buf));
        assert_eq!(&msg1[..], &buf[..]);
        or_panic!(stream.write_all(msg2));
    });

    let mut stream = or_panic!(UnixStream::connect(&socket_path));
    assert_eq!(Some(&*socket_path), stream.peer_addr().unwrap().as_pathname());
    or_panic!(stream.write_all(msg1));
    let mut buf = vec![];
    or_panic!(stream.read_to_end(&mut buf));
    assert_eq!(&msg2[..], &buf[..]);
    drop(stream);

    thread.join().unwrap();
}

#[test]
fn vectored() {
    let (mut s1, mut s2) = or_panic!(UnixStream::pair());

    let len = or_panic!(s1.write_vectored(&[
        IoSlice::new(b"hello"),
        IoSlice::new(b" "),
        IoSlice::new(b"world!")
    ],));
    assert_eq!(len, 12);

    let mut buf1 = [0; 6];
    let mut buf2 = [0; 7];
    let len =
        or_panic!(s2.read_vectored(&mut [IoSliceMut::new(&mut buf1), IoSliceMut::new(&mut buf2)],));
    assert_eq!(len, 12);
    assert_eq!(&buf1, b"hello ");
    assert_eq!(&buf2, b"world!\0");
}

#[test]
fn pair() {
    let msg1 = b"hello";
    let msg2 = b"world!";

    let (mut s1, mut s2) = or_panic!(UnixStream::pair());
    let thread = thread::spawn(move || {
        // s1 must be moved in or the test will hang!
        let mut buf = [0; 5];
        or_panic!(s1.read(&mut buf));
        assert_eq!(&msg1[..], &buf[..]);
        or_panic!(s1.write_all(msg2));
    });

    or_panic!(s2.write_all(msg1));
    let mut buf = vec![];
    or_panic!(s2.read_to_end(&mut buf));
    assert_eq!(&msg2[..], &buf[..]);
    drop(s2);

    thread.join().unwrap();
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn try_clone() {
    let dir = tmpdir();
    let socket_path = dir.path().join("sock");
    let msg1 = b"hello";
    let msg2 = b"world";

    let listener = or_panic!(UnixListener::bind(&socket_path));
    let thread = thread::spawn(move || {
        let mut stream = or_panic!(listener.accept()).0;
        or_panic!(stream.write_all(msg1));
        or_panic!(stream.write_all(msg2));
    });

    let mut stream = or_panic!(UnixStream::connect(&socket_path));
    let mut stream2 = or_panic!(stream.try_clone());

    let mut buf = [0; 5];
    or_panic!(stream.read(&mut buf));
    assert_eq!(&msg1[..], &buf[..]);
    or_panic!(stream2.read(&mut buf));
    assert_eq!(&msg2[..], &buf[..]);

    thread.join().unwrap();
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn iter() {
    let dir = tmpdir();
    let socket_path = dir.path().join("sock");

    let listener = or_panic!(UnixListener::bind(&socket_path));
    let thread = thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let mut stream = or_panic!(stream);
            let mut buf = [0];
            or_panic!(stream.read(&mut buf));
        }
    });

    for _ in 0..2 {
        let mut stream = or_panic!(UnixStream::connect(&socket_path));
        or_panic!(stream.write_all(&[0]));
    }

    thread.join().unwrap();
}

#[test]
fn long_path() {
    let dir = tmpdir();
    let socket_path = dir.path().join(
        "asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfa\
                                sasdfasdfasdasdfasdfasdfadfasdfasdfasdfasdfasdf",
    );
    match UnixStream::connect(&socket_path) {
        Err(ref e) if e.kind() == io::ErrorKind::InvalidInput => {}
        Err(e) => panic!("unexpected error {e}"),
        Ok(_) => panic!("unexpected success"),
    }

    match UnixListener::bind(&socket_path) {
        Err(ref e) if e.kind() == io::ErrorKind::InvalidInput => {}
        Err(e) => panic!("unexpected error {e}"),
        Ok(_) => panic!("unexpected success"),
    }

    match UnixDatagram::bind(&socket_path) {
        Err(ref e) if e.kind() == io::ErrorKind::InvalidInput => {}
        Err(e) => panic!("unexpected error {e}"),
        Ok(_) => panic!("unexpected success"),
    }
}

#[test]
#[cfg(not(target_os = "nto"))]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn timeouts() {
    let dir = tmpdir();
    let socket_path = dir.path().join("sock");

    let _listener = or_panic!(UnixListener::bind(&socket_path));

    let stream = or_panic!(UnixStream::connect(&socket_path));
    let dur = Duration::new(15410, 0);

    assert_eq!(None, or_panic!(stream.read_timeout()));

    or_panic!(stream.set_read_timeout(Some(dur)));
    assert_eq!(Some(dur), or_panic!(stream.read_timeout()));

    assert_eq!(None, or_panic!(stream.write_timeout()));

    or_panic!(stream.set_write_timeout(Some(dur)));
    assert_eq!(Some(dur), or_panic!(stream.write_timeout()));

    or_panic!(stream.set_read_timeout(None));
    assert_eq!(None, or_panic!(stream.read_timeout()));

    or_panic!(stream.set_write_timeout(None));
    assert_eq!(None, or_panic!(stream.write_timeout()));
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_read_timeout() {
    let dir = tmpdir();
    let socket_path = dir.path().join("sock");

    let _listener = or_panic!(UnixListener::bind(&socket_path));

    let mut stream = or_panic!(UnixStream::connect(&socket_path));
    or_panic!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

    let mut buf = [0; 10];
    let kind = stream.read_exact(&mut buf).err().expect("expected error").kind();
    assert!(
        kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut,
        "unexpected_error: {:?}",
        kind
    );
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_read_with_timeout() {
    let dir = tmpdir();
    let socket_path = dir.path().join("sock");

    let listener = or_panic!(UnixListener::bind(&socket_path));

    let mut stream = or_panic!(UnixStream::connect(&socket_path));
    or_panic!(stream.set_read_timeout(Some(Duration::from_millis(1000))));

    let mut other_end = or_panic!(listener.accept()).0;
    or_panic!(other_end.write_all(b"hello world"));

    let mut buf = [0; 11];
    or_panic!(stream.read(&mut buf));
    assert_eq!(b"hello world", &buf[..]);

    let kind = stream.read_exact(&mut buf).err().expect("expected error").kind();
    assert!(
        kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut,
        "unexpected_error: {:?}",
        kind
    );
}

// Ensure the `set_read_timeout` and `set_write_timeout` calls return errors
// when passed zero Durations
#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unix_stream_timeout_zero_duration() {
    let dir = tmpdir();
    let socket_path = dir.path().join("sock");

    let listener = or_panic!(UnixListener::bind(&socket_path));
    let stream = or_panic!(UnixStream::connect(&socket_path));

    let result = stream.set_write_timeout(Some(Duration::new(0, 0)));
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    let result = stream.set_read_timeout(Some(Duration::new(0, 0)));
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    drop(listener);
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unix_datagram() {
    let dir = tmpdir();
    let path1 = dir.path().join("sock1");
    let path2 = dir.path().join("sock2");

    let sock1 = or_panic!(UnixDatagram::bind(&path1));
    let sock2 = or_panic!(UnixDatagram::bind(&path2));

    let msg = b"hello world";
    or_panic!(sock1.send_to(msg, &path2));
    let mut buf = [0; 11];
    or_panic!(sock2.recv_from(&mut buf));
    assert_eq!(msg, &buf[..]);
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unnamed_unix_datagram() {
    let dir = tmpdir();
    let path1 = dir.path().join("sock1");

    let sock1 = or_panic!(UnixDatagram::bind(&path1));
    let sock2 = or_panic!(UnixDatagram::unbound());

    let msg = b"hello world";
    or_panic!(sock2.send_to(msg, &path1));
    let mut buf = [0; 11];
    let (usize, addr) = or_panic!(sock1.recv_from(&mut buf));
    assert_eq!(usize, 11);
    assert!(addr.is_unnamed());
    assert_eq!(msg, &buf[..]);
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unix_datagram_connect_to_recv_addr() {
    let dir = tmpdir();
    let path1 = dir.path().join("sock1");
    let path2 = dir.path().join("sock2");

    let sock1 = or_panic!(UnixDatagram::bind(&path1));
    let sock2 = or_panic!(UnixDatagram::bind(&path2));

    let msg = b"hello world";
    let sock1_addr = or_panic!(sock1.local_addr());
    or_panic!(sock2.send_to_addr(msg, &sock1_addr));
    let mut buf = [0; 11];
    let (_, addr) = or_panic!(sock1.recv_from(&mut buf));

    let new_msg = b"hello back";
    let mut new_buf = [0; 10];
    or_panic!(sock2.connect_addr(&addr));
    or_panic!(sock2.send(new_msg)); // set by connect_addr
    let usize = or_panic!(sock2.recv(&mut new_buf));
    assert_eq!(usize, 10);
    assert_eq!(new_msg, &new_buf[..]);
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_connect_unix_datagram() {
    let dir = tmpdir();
    let path1 = dir.path().join("sock1");
    let path2 = dir.path().join("sock2");

    let bsock1 = or_panic!(UnixDatagram::bind(&path1));
    let bsock2 = or_panic!(UnixDatagram::bind(&path2));
    let sock = or_panic!(UnixDatagram::unbound());
    or_panic!(sock.connect(&path1));

    // Check send()
    let msg = b"hello there";
    or_panic!(sock.send(msg));
    let mut buf = [0; 11];
    let (usize, addr) = or_panic!(bsock1.recv_from(&mut buf));
    assert_eq!(usize, 11);
    assert!(addr.is_unnamed());
    assert_eq!(msg, &buf[..]);

    // Changing default socket works too
    or_panic!(sock.connect(&path2));
    or_panic!(sock.send(msg));
    or_panic!(bsock2.recv_from(&mut buf));
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unix_datagram_recv() {
    let dir = tmpdir();
    let path1 = dir.path().join("sock1");

    let sock1 = or_panic!(UnixDatagram::bind(&path1));
    let sock2 = or_panic!(UnixDatagram::unbound());
    or_panic!(sock2.connect(&path1));

    let msg = b"hello world";
    or_panic!(sock2.send(msg));
    let mut buf = [0; 11];
    let size = or_panic!(sock1.recv(&mut buf));
    assert_eq!(size, 11);
    assert_eq!(msg, &buf[..]);
}

#[test]
fn datagram_pair() {
    let msg1 = b"hello";
    let msg2 = b"world!";

    let (s1, s2) = or_panic!(UnixDatagram::pair());
    let thread = thread::spawn(move || {
        // s1 must be moved in or the test will hang!
        let mut buf = [0; 5];
        or_panic!(s1.recv(&mut buf));
        assert_eq!(&msg1[..], &buf[..]);
        or_panic!(s1.send(msg2));
    });

    or_panic!(s2.send(msg1));
    let mut buf = [0; 6];
    or_panic!(s2.recv(&mut buf));
    assert_eq!(&msg2[..], &buf[..]);
    drop(s2);

    thread.join().unwrap();
}

// Ensure the `set_read_timeout` and `set_write_timeout` calls return errors
// when passed zero Durations
#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unix_datagram_timeout_zero_duration() {
    let dir = tmpdir();
    let path = dir.path().join("sock");

    let datagram = or_panic!(UnixDatagram::bind(&path));

    let result = datagram.set_write_timeout(Some(Duration::new(0, 0)));
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    let result = datagram.set_read_timeout(Some(Duration::new(0, 0)));
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn abstract_namespace_not_allowed_connect() {
    assert!(UnixStream::connect("\0asdf").is_err());
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
fn test_abstract_stream_connect() {
    let msg1 = b"hello";
    let msg2 = b"world";

    let socket_addr = or_panic!(SocketAddr::from_abstract_name(b"name"));
    let listener = or_panic!(UnixListener::bind_addr(&socket_addr));

    let thread = thread::spawn(move || {
        let mut stream = or_panic!(listener.accept()).0;
        let mut buf = [0; 5];
        or_panic!(stream.read(&mut buf));
        assert_eq!(&msg1[..], &buf[..]);
        or_panic!(stream.write_all(msg2));
    });

    let mut stream = or_panic!(UnixStream::connect_addr(&socket_addr));

    let peer = or_panic!(stream.peer_addr());
    assert_eq!(peer.as_abstract_name().unwrap(), b"name");

    or_panic!(stream.write_all(msg1));
    let mut buf = vec![];
    or_panic!(stream.read_to_end(&mut buf));
    assert_eq!(&msg2[..], &buf[..]);
    drop(stream);

    thread.join().unwrap();
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
fn test_abstract_stream_iter() {
    let addr = or_panic!(SocketAddr::from_abstract_name(b"hidden"));
    let listener = or_panic!(UnixListener::bind_addr(&addr));

    let thread = thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let mut stream = or_panic!(stream);
            let mut buf = [0];
            or_panic!(stream.read(&mut buf));
        }
    });

    for _ in 0..2 {
        let mut stream = or_panic!(UnixStream::connect_addr(&addr));
        or_panic!(stream.write_all(&[0]));
    }

    thread.join().unwrap();
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
fn test_abstract_datagram_bind_send_to_addr() {
    let addr1 = or_panic!(SocketAddr::from_abstract_name(b"ns1"));
    let sock1 = or_panic!(UnixDatagram::bind_addr(&addr1));

    let local = or_panic!(sock1.local_addr());
    assert_eq!(local.as_abstract_name().unwrap(), b"ns1");

    let addr2 = or_panic!(SocketAddr::from_abstract_name(b"ns2"));
    let sock2 = or_panic!(UnixDatagram::bind_addr(&addr2));

    let msg = b"hello world";
    or_panic!(sock1.send_to_addr(msg, &addr2));
    let mut buf = [0; 11];
    let (len, addr) = or_panic!(sock2.recv_from(&mut buf));
    assert_eq!(msg, &buf[..]);
    assert_eq!(len, 11);
    assert_eq!(addr.as_abstract_name().unwrap(), b"ns1");
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
fn test_abstract_datagram_connect_addr() {
    let addr1 = or_panic!(SocketAddr::from_abstract_name(b"ns3"));
    let bsock1 = or_panic!(UnixDatagram::bind_addr(&addr1));

    let sock = or_panic!(UnixDatagram::unbound());
    or_panic!(sock.connect_addr(&addr1));

    let msg = b"hello world";
    or_panic!(sock.send(msg));
    let mut buf = [0; 11];
    let (len, addr) = or_panic!(bsock1.recv_from(&mut buf));
    assert_eq!(len, 11);
    assert_eq!(addr.is_unnamed(), true);
    assert_eq!(msg, &buf[..]);

    let addr2 = or_panic!(SocketAddr::from_abstract_name(b"ns4"));
    let bsock2 = or_panic!(UnixDatagram::bind_addr(&addr2));

    or_panic!(sock.connect_addr(&addr2));
    or_panic!(sock.send(msg));
    or_panic!(bsock2.recv_from(&mut buf));
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
fn test_abstract_name_too_long() {
    match SocketAddr::from_abstract_name(
        b"abcdefghijklmnopqrstuvwxyzabcdefghijklmn\
        opqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghi\
        jklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
    ) {
        Err(ref e) if e.kind() == io::ErrorKind::InvalidInput => {}
        Err(e) => panic!("unexpected error {e}"),
        Ok(_) => panic!("unexpected success"),
    }
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
fn test_abstract_no_pathname_and_not_unnamed() {
    let name = b"local";
    let addr = or_panic!(SocketAddr::from_abstract_name(name));
    assert_eq!(addr.as_pathname(), None);
    assert_eq!(addr.as_abstract_name(), Some(&name[..]));
    assert_eq!(addr.is_unnamed(), false);
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unix_stream_peek() {
    let (txdone, rxdone) = crate::sync::mpsc::channel();

    let dir = tmpdir();
    let path = dir.path().join("sock");

    let listener = or_panic!(UnixListener::bind(&path));
    let thread = thread::spawn(move || {
        let mut stream = or_panic!(listener.accept()).0;
        or_panic!(stream.write_all(&[1, 3, 3, 7]));
        or_panic!(rxdone.recv());
    });

    let mut stream = or_panic!(UnixStream::connect(&path));
    let mut buf = [0; 10];
    for _ in 0..2 {
        assert_eq!(or_panic!(stream.peek(&mut buf)), 4);
    }
    assert_eq!(or_panic!(stream.read(&mut buf)), 4);

    or_panic!(stream.set_nonblocking(true));
    match stream.peek(&mut buf) {
        Ok(_) => panic!("expected error"),
        Err(ref e) if e.kind() == ErrorKind::WouldBlock => {}
        Err(e) => panic!("unexpected error: {e}"),
    }

    or_panic!(txdone.send(()));
    thread.join().unwrap();
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unix_datagram_peek() {
    let dir = tmpdir();
    let path1 = dir.path().join("sock");

    let sock1 = or_panic!(UnixDatagram::bind(&path1));
    let sock2 = or_panic!(UnixDatagram::unbound());
    or_panic!(sock2.connect(&path1));

    let msg = b"hello world";
    or_panic!(sock2.send(msg));
    for _ in 0..2 {
        let mut buf = [0; 11];
        let size = or_panic!(sock1.peek(&mut buf));
        assert_eq!(size, 11);
        assert_eq!(msg, &buf[..]);
    }

    let mut buf = [0; 11];
    let size = or_panic!(sock1.recv(&mut buf));
    assert_eq!(size, 11);
    assert_eq!(msg, &buf[..]);
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_unix_datagram_peek_from() {
    let dir = tmpdir();
    let path1 = dir.path().join("sock");

    let sock1 = or_panic!(UnixDatagram::bind(&path1));
    let sock2 = or_panic!(UnixDatagram::unbound());
    or_panic!(sock2.connect(&path1));

    let msg = b"hello world";
    or_panic!(sock2.send(msg));
    for _ in 0..2 {
        let mut buf = [0; 11];
        let (size, _) = or_panic!(sock1.peek_from(&mut buf));
        assert_eq!(size, 11);
        assert_eq!(msg, &buf[..]);
    }

    let mut buf = [0; 11];
    let size = or_panic!(sock1.recv(&mut buf));
    assert_eq!(size, 11);
    assert_eq!(msg, &buf[..]);
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
fn test_send_vectored_fds_unix_stream() {
    let (s1, s2) = or_panic!(UnixStream::pair());

    let buf1 = [1; 8];
    let bufs_send = &[IoSlice::new(&buf1[..])][..];

    let mut ancillary1_buffer = [0; 128];
    let mut ancillary1 = SocketAncillary::new(&mut ancillary1_buffer[..]);
    assert!(ancillary1.add_fds(&[s1.as_raw_fd()][..]));

    let usize = or_panic!(s1.send_vectored_with_ancillary(&bufs_send, &mut ancillary1));
    assert_eq!(usize, 8);

    let mut buf2 = [0; 8];
    let mut bufs_recv = &mut [IoSliceMut::new(&mut buf2[..])][..];

    let mut ancillary2_buffer = [0; 128];
    let mut ancillary2 = SocketAncillary::new(&mut ancillary2_buffer[..]);

    let usize = or_panic!(s2.recv_vectored_with_ancillary(&mut bufs_recv, &mut ancillary2));
    assert_eq!(usize, 8);
    assert_eq!(buf1, buf2);

    let mut ancillary_data_vec = Vec::from_iter(ancillary2.messages());
    assert_eq!(ancillary_data_vec.len(), 1);
    if let AncillaryData::ScmRights(scm_rights) = ancillary_data_vec.pop().unwrap().unwrap() {
        let fd_vec = Vec::from_iter(scm_rights);
        assert_eq!(fd_vec.len(), 1);
        unsafe {
            libc::close(fd_vec[0]);
        }
    } else {
        unreachable!("must be ScmRights");
    }
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_send_vectored_with_ancillary_to_unix_datagram() {
    fn getpid() -> libc::pid_t {
        unsafe { libc::getpid() }
    }

    fn getuid() -> libc::uid_t {
        unsafe { libc::getuid() }
    }

    fn getgid() -> libc::gid_t {
        unsafe { libc::getgid() }
    }

    let dir = tmpdir();
    let path1 = dir.path().join("sock1");
    let path2 = dir.path().join("sock2");

    let bsock1 = or_panic!(UnixDatagram::bind(&path1));
    let bsock2 = or_panic!(UnixDatagram::bind(&path2));

    or_panic!(bsock2.set_passcred(true));

    let buf1 = [1; 8];
    let bufs_send = &[IoSlice::new(&buf1[..])][..];

    let mut ancillary1_buffer = [0; 128];
    let mut ancillary1 = SocketAncillary::new(&mut ancillary1_buffer[..]);
    let mut cred1 = SocketCred::new();
    cred1.set_pid(getpid());
    cred1.set_uid(getuid());
    cred1.set_gid(getgid());
    assert!(ancillary1.add_creds(&[cred1.clone()][..]));

    let usize =
        or_panic!(bsock1.send_vectored_with_ancillary_to(&bufs_send, &mut ancillary1, &path2));
    assert_eq!(usize, 8);

    let mut buf2 = [0; 8];
    let mut bufs_recv = &mut [IoSliceMut::new(&mut buf2[..])][..];

    let mut ancillary2_buffer = [0; 128];
    let mut ancillary2 = SocketAncillary::new(&mut ancillary2_buffer[..]);

    let (usize, truncated, _addr) =
        or_panic!(bsock2.recv_vectored_with_ancillary_from(&mut bufs_recv, &mut ancillary2));
    assert_eq!(ancillary2.truncated(), false);
    assert_eq!(usize, 8);
    assert_eq!(truncated, false);
    assert_eq!(buf1, buf2);

    let mut ancillary_data_vec = Vec::from_iter(ancillary2.messages());
    assert_eq!(ancillary_data_vec.len(), 1);
    if let AncillaryData::ScmCredentials(scm_credentials) =
        ancillary_data_vec.pop().unwrap().unwrap()
    {
        let cred_vec = Vec::from_iter(scm_credentials);
        assert_eq!(cred_vec.len(), 1);
        assert_eq!(cred1.get_pid(), cred_vec[0].get_pid());
        assert_eq!(cred1.get_uid(), cred_vec[0].get_uid());
        assert_eq!(cred1.get_gid(), cred_vec[0].get_gid());
    } else {
        unreachable!("must be ScmCredentials");
    }
}

#[cfg(any(target_os = "android", target_os = "linux"))]
#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating Unix sockets
fn test_send_vectored_with_ancillary_unix_datagram() {
    let dir = tmpdir();
    let path1 = dir.path().join("sock1");
    let path2 = dir.path().join("sock2");

    let bsock1 = or_panic!(UnixDatagram::bind(&path1));
    let bsock2 = or_panic!(UnixDatagram::bind(&path2));

    let buf1 = [1; 8];
    let bufs_send = &[IoSlice::new(&buf1[..])][..];

    let mut ancillary1_buffer = [0; 128];
    let mut ancillary1 = SocketAncillary::new(&mut ancillary1_buffer[..]);
    assert!(ancillary1.add_fds(&[bsock1.as_raw_fd()][..]));

    or_panic!(bsock1.connect(&path2));
    let usize = or_panic!(bsock1.send_vectored_with_ancillary(&bufs_send, &mut ancillary1));
    assert_eq!(usize, 8);

    let mut buf2 = [0; 8];
    let mut bufs_recv = &mut [IoSliceMut::new(&mut buf2[..])][..];

    let mut ancillary2_buffer = [0; 128];
    let mut ancillary2 = SocketAncillary::new(&mut ancillary2_buffer[..]);

    let (usize, truncated) =
        or_panic!(bsock2.recv_vectored_with_ancillary(&mut bufs_recv, &mut ancillary2));
    assert_eq!(usize, 8);
    assert_eq!(truncated, false);
    assert_eq!(buf1, buf2);

    let mut ancillary_data_vec = Vec::from_iter(ancillary2.messages());
    assert_eq!(ancillary_data_vec.len(), 1);
    if let AncillaryData::ScmRights(scm_rights) = ancillary_data_vec.pop().unwrap().unwrap() {
        let fd_vec = Vec::from_iter(scm_rights);
        assert_eq!(fd_vec.len(), 1);
        unsafe {
            libc::close(fd_vec[0]);
        }
    } else {
        unreachable!("must be ScmRights");
    }
}

struct ControlMessagesBuf {
    bytes: Vec<u8>,
}

impl ControlMessagesBuf {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn push(&mut self, cmsg: ControlMessage<'_>) {
        let mut tmp = [core::mem::MaybeUninit::new(0u8); 100];
        let size = cmsg.cmsg_space();
        self.bytes.extend_from_slice(cmsg.copy_to_slice(&mut tmp[..size]));
    }

    fn messages(&self) -> &ControlMessages {
        ControlMessages::from_bytes(&self.bytes)
    }
}

#[test]
fn control_messages() {
    let mut buf = ControlMessagesBuf::new();

    let cmsg_1 = ControlMessage::new(11, 22, &[3, 4, 5]);
    assert_eq!(cmsg_1.cmsg_level(), 11);
    assert_eq!(cmsg_1.cmsg_type(), 22);
    assert_eq!(cmsg_1.data(), &[3, 4, 5]);
    buf.push(cmsg_1);

    let cmsg_2 = ControlMessage::new(66, 77, &[8, 9, 10]);
    buf.push(cmsg_2);

    let mut iter = buf.messages().iter();
    assert_eq!(iter.next(), Some(cmsg_1));
    assert_eq!(iter.next(), Some(cmsg_2));
    assert_eq!(iter.next(), None);
}

#[test]
fn control_messages_truncated() {
    let mut big_buf = ControlMessagesBuf::new();
    big_buf.push(ControlMessage::new(11, 22, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
    let big_bytes = big_buf.bytes;

    let mut small_buf = ControlMessagesBuf::new();
    small_buf.push(ControlMessage::new(11, 22, &[1]));
    let small_bytes = small_buf.bytes;

    let trunc_bytes = &big_bytes[..small_bytes.len()];
    let mut iter = ControlMessages::from_bytes(trunc_bytes).iter();
    let trunc_cmsg = iter.next().unwrap();
    assert_eq!(iter.next(), None);

    assert!(trunc_cmsg.truncated());

    // Verify that the truncation state is preserved by ControlMessagesBuf.
    let mut trunc_buf = ControlMessagesBuf::new();
    trunc_buf.push(trunc_cmsg);
    assert_eq!(trunc_buf.bytes, trunc_bytes);
}

#[test]
fn control_messages_match_libc() {
    // Message data lengths test behavior with 4-byte and 8-byte padding.
    let MSG_DATA_LEN_3: &[u8] = &[31, 32, 33];
    let MSG_DATA_LEN_4: &[u8] = &[41, 42, 43, 44];
    let MSG_DATA_LEN_5: &[u8] = &[51, 52, 53, 54, 55];
    let MSG_DATA_LEN_7: &[u8] = &[71, 72, 73, 74, 75, 76, 77];
    let MSG_DATA_LEN_8: &[u8] = &[81, 82, 83, 84, 85, 86, 87, 88];
    let MSG_DATA_LEN_9: &[u8] = &[91, 92, 93, 94, 95, 96, 97, 98, 99];

    let mut buf = ControlMessagesBuf::new();
    buf.push(ControlMessage::new(300, 301, MSG_DATA_LEN_3));
    buf.push(ControlMessage::new(400, 401, MSG_DATA_LEN_4));
    buf.push(ControlMessage::new(500, 501, MSG_DATA_LEN_5));
    buf.push(ControlMessage::new(700, 701, MSG_DATA_LEN_7));
    buf.push(ControlMessage::new(800, 801, MSG_DATA_LEN_8));
    buf.push(ControlMessage::new(900, 901, MSG_DATA_LEN_9));

    const LIBC_BUF_CAPACITY: usize = 500;
    assert!(LIBC_BUF_CAPACITY >= buf.bytes.len());

    union aligned_cmsgbuf {
        _hdr: libc::cmsghdr,
        buf: [u8; LIBC_BUF_CAPACITY],
    }

    let mut msg: libc::msghdr = unsafe { core::mem::zeroed() };
    let mut cmsgbuf: aligned_cmsgbuf = unsafe { core::mem::zeroed() };

    let libc_control_messages_bytes = unsafe {
        msg.msg_control = (&mut cmsgbuf.buf).as_mut_ptr().cast();
        msg.msg_controllen = core::mem::size_of_val(&cmsgbuf.buf);

        let mut libc_buf_len: usize = 0;

        let mut libc_push_cmsg = |cmsg, cmsg_level, cmsg_type, data: &[u8]| {
            let cmsg: *mut libc::cmsghdr = cmsg;
            (*cmsg).cmsg_len = libc::CMSG_LEN(data.len() as _) as _;
            (*cmsg).cmsg_level = cmsg_level;
            (*cmsg).cmsg_type = cmsg_type;
            let cmsg_data = libc::CMSG_DATA(cmsg);
            cmsg_data.copy_from(data.as_ptr(), data.len());
            libc_buf_len += libc::CMSG_SPACE(data.len() as _) as usize;
        };

        let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
        libc_push_cmsg(cmsg, 300, 301, MSG_DATA_LEN_3);

        cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        libc_push_cmsg(cmsg, 400, 401, MSG_DATA_LEN_4);

        cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        libc_push_cmsg(cmsg, 500, 501, MSG_DATA_LEN_5);

        cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        libc_push_cmsg(cmsg, 700, 701, MSG_DATA_LEN_7);

        cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        libc_push_cmsg(cmsg, 800, 801, MSG_DATA_LEN_8);

        cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        libc_push_cmsg(cmsg, 900, 901, MSG_DATA_LEN_9);

        &cmsgbuf.buf[..libc_buf_len]
    };

    assert_eq!(buf.bytes, libc_control_messages_bytes);
}
