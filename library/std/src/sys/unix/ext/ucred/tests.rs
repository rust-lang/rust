use crate::os::unix::net::UnixStream;
use libc::{getegid, geteuid};

#[test]
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
