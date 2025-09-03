//! Utils that need libc.
#![allow(dead_code)]

pub unsafe fn read_all(
    fd: libc::c_int,
    buf: *mut libc::c_void,
    count: libc::size_t,
) -> libc::ssize_t {
    assert!(count > 0);
    let mut read_so_far = 0;
    while read_so_far < count {
        let res = libc::read(fd, buf.add(read_so_far), count - read_so_far);
        if res < 0 {
            return res;
        }
        if res == 0 {
            // EOF
            break;
        }
        read_so_far += res as libc::size_t;
    }
    return read_so_far as libc::ssize_t;
}

pub unsafe fn write_all(
    fd: libc::c_int,
    buf: *const libc::c_void,
    count: libc::size_t,
) -> libc::ssize_t {
    assert!(count > 0);
    let mut written_so_far = 0;
    while written_so_far < count {
        let res = libc::write(fd, buf.add(written_so_far), count - written_so_far);
        if res < 0 {
            return res;
        }
        // Apparently a return value of 0 is just a short write, nothing special (unlike reads).
        written_so_far += res as libc::size_t;
    }
    return written_so_far as libc::ssize_t;
}
