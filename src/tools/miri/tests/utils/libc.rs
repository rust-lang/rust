//! Utils that need libc.
#![allow(dead_code)]

use std::{fmt, io};

/// Handles the usual libc function that returns `-1` to indicate an error.
#[track_caller]
pub fn errno_result<T: From<i8> + Ord>(ret: T) -> io::Result<T> {
    use std::cmp::Ordering;
    match ret.cmp(&(-1i8).into()) {
        Ordering::Equal => Err(io::Error::last_os_error()),
        Ordering::Greater => Ok(ret),
        Ordering::Less => panic!("unexpected return value: less than -1"),
    }
}
/// Check that a function with errno error handling succeeded (i.e., returned 0).
#[track_caller]
pub fn errno_check<T: From<i8> + Ord + fmt::Debug>(ret: T) {
    assert_eq!(errno_result(ret).unwrap(), 0i8.into(), "wrong successful result");
}

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

/// Try to fill the given slice by reading from `fd`. Panic if that many bytes could not be read.
#[track_caller]
pub fn read_all_into_slice(fd: libc::c_int, buf: &mut [u8]) -> io::Result<()> {
    let res = errno_result(unsafe { read_all(fd, buf.as_mut_ptr().cast(), buf.len()) })?;
    assert_eq!(res as usize, buf.len());
    Ok(())
}

/// Read exactly `N` bytes from `fd`. Error if that many bytes could not be read.
#[track_caller]
pub fn read_all_into_array<const N: usize>(fd: libc::c_int) -> io::Result<[u8; N]> {
    let mut buf = [0; N];
    read_all_into_slice(fd, &mut buf)?;
    Ok(buf)
}

/// Do a single read from `fd` and return the part of the buffer that was written into,
/// and the rest.
#[track_caller]
pub fn read_into_slice(fd: libc::c_int, buf: &mut [u8]) -> io::Result<(&mut [u8], &mut [u8])> {
    let res = errno_result(unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) })?;
    Ok(buf.split_at_mut(res as usize))
}

/// Read from `fd` until we get EOF and return the part of the buffer that was written into,
/// and the rest.
#[track_caller]
pub fn read_until_eof_into_slice(
    fd: libc::c_int,
    buf: &mut [u8],
) -> io::Result<(&mut [u8], &mut [u8])> {
    let res = errno_result(unsafe { read_all(fd, buf.as_mut_ptr().cast(), buf.len()) })?;
    Ok(buf.split_at_mut(res as usize))
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

/// Write the entire `buf` to `fd`. Panic if not all bytes could be written.
#[track_caller]
pub fn write_all_from_slice(fd: libc::c_int, buf: &[u8]) -> io::Result<()> {
    let res = errno_result(unsafe { write_all(fd, buf.as_ptr().cast(), buf.len()) })?;
    assert_eq!(res as usize, buf.len());
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "illumos"))]
#[allow(unused_imports)]
pub mod epoll {
    use libc::c_int;
    pub use libc::{EPOLL_CTL_ADD, EPOLL_CTL_DEL, EPOLL_CTL_MOD};
    // Re-export some constants we need a lot for this.
    pub use libc::{EPOLLET, EPOLLHUP, EPOLLIN, EPOLLOUT, EPOLLRDHUP};

    use super::*;

    /// The libc epoll_event type doesn't fit to the EPOLLIN etc constants, so we have our
    /// own type. We also make the data field an int since we typically want to store FDs there.
    #[derive(PartialEq, Debug, Clone, Copy)]
    pub struct Ev {
        pub events: c_int,
        pub data: c_int,
    }

    #[track_caller]
    pub fn epoll_ctl(epfd: c_int, op: c_int, fd: c_int, event: Ev) -> io::Result<()> {
        let mut event = libc::epoll_event {
            events: event.events.cast_unsigned(),
            u64: event.data.try_into().unwrap(),
        };
        let ret = errno_result(unsafe { libc::epoll_ctl(epfd, op, fd, &raw mut event) })?;
        assert_eq!(ret, 0);
        Ok(())
    }

    /// Helper for the common case of adding an FD to an epoll with the FD itself being
    /// the `data`.
    #[track_caller]
    pub fn epoll_ctl_add(epfd: c_int, fd: c_int, events: c_int) -> io::Result<()> {
        epoll_ctl(epfd, EPOLL_CTL_ADD, fd, Ev { events, data: fd })
    }

    #[track_caller]
    pub fn check_epoll_wait<const N: usize>(epfd: i32, expected: &[Ev], timeout: i32) {
        let mut array: [libc::epoll_event; N] = [libc::epoll_event { events: 0, u64: 0 }; N];
        let num = errno_result(unsafe {
            libc::epoll_wait(epfd, array.as_mut_ptr(), N.try_into().unwrap(), timeout)
        })
        .expect("epoll_wait returned an error");
        let got = &mut array[..num.try_into().unwrap()];
        let got = got
            .iter()
            .map(|e| Ev { events: e.events.cast_signed(), data: e.u64.try_into().unwrap() })
            .collect::<Vec<_>>();
        assert_eq!(got, expected, "got wrong notifications");
    }

    #[track_caller]
    pub fn check_epoll_wait_noblock<const N: usize>(epfd: i32, expected: &[Ev]) {
        check_epoll_wait::<N>(epfd, expected, 0);
    }
}

pub mod net {
    /// IPv4 localhost address bytes
    pub const IPV4_LOCALHOST: [u8; 4] = [127, 0, 0, 1];
    /// IPv6 localhost address bytes
    pub const IPV6_LOCALHOST: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

    /// Create a libc representation of an IPv4 address given the address bytes and a port.
    pub fn ipv4_sock_addr(addr_bytes: [u8; 4], port: u16) -> libc::sockaddr_in {
        libc::sockaddr_in {
            sin_family: libc::AF_INET as libc::sa_family_t,
            sin_port: port.to_be(),
            // `addr_bytes` is already in big-endian and that's the format `sin_addr` expects.
            #[expect(unnecessary_transmutes)]
            sin_addr: libc::in_addr {
                s_addr: unsafe { std::mem::transmute::<[u8; 4], u32>(addr_bytes) },
            },
            ..unsafe { core::mem::zeroed() }
        }
    }

    /// Create a libc representation of an IPv6 address given the address bytes and a port.
    ///
    /// This method sets `flowinfo` and `scope_id` to 0.
    pub fn ipv6_sock_addr(addr_bytes: [u8; 16], port: u16) -> libc::sockaddr_in6 {
        ipv6_sock_addr_full(addr_bytes, port, 0, 0)
    }

    /// Create a libc representation of a full IPv6 address given the address bytes, a port
    /// as well as a flowinfo and scope id.
    pub fn ipv6_sock_addr_full(
        addr_bytes: [u8; 16],
        port: u16,
        flowinfo: u32,
        scope_id: u32,
    ) -> libc::sockaddr_in6 {
        #[allow(clippy::needless_update)]
        libc::sockaddr_in6 {
            sin6_family: libc::AF_INET6 as libc::sa_family_t,
            sin6_port: port.to_be(),
            sin6_addr: libc::in6_addr { s6_addr: addr_bytes },
            sin6_flowinfo: flowinfo,
            sin6_scope_id: scope_id,
            // This is only needed on some targets where an additional `sin6_len` field exists.
            ..unsafe { core::mem::zeroed() }
        }
    }
}
