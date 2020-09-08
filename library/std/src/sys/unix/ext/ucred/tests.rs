use crate::os::unix::net::UnixStream;
use libc::{getegid, geteuid};

#[test]
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "ios",
    target_os = "macos",
    target_os = "openbsd"
))]
fn test_socket_pair() {
    // Create two connected sockets and get their peer credentials. They should be equal.
    let (sock_a, sock_b) = UnixStream::pair().unwrap();
    let (cred_a, cred_b) = (sock_a.peer_cred().unwrap(), sock_b.peer_cred().unwrap());
    assert_eq!(cred_a, cred_b);

    // Check that the UID and GIDs match up.
    let uid = unsafe { geteuid() };
    let gid = unsafe { getegid() };
    assert_eq!(cred_a.uid, uid);
    assert_eq!(cred_a.gid, gid);
}
