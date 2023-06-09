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

#[cfg(any(target_os = "android", target_os = "linux", target_os = "freebsd"))]
#[test]
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
