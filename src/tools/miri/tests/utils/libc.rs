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
    use std::io;

    use super::{errno_check, errno_result};

    /// IPv4 localhost address bytes
    pub const IPV4_LOCALHOST: [u8; 4] = [127, 0, 0, 1];
    /// IPv6 localhost address bytes
    pub const IPV6_LOCALHOST: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

    /// Create a libc representation of an IPv4 address given the address bytes and a port.
    pub fn sock_addr_ipv4(addr_bytes: [u8; 4], port: u16) -> libc::sockaddr_in {
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
    pub fn sock_addr_ipv6(addr_bytes: [u8; 16], port: u16) -> libc::sockaddr_in6 {
        sock_addr_full_ipv6(addr_bytes, port, 0, 0)
    }

    /// Create a libc representation of a full IPv6 address given the address bytes, a port
    /// as well as a flowinfo and scope id.
    pub fn sock_addr_full_ipv6(
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

    /// Create an IPv4 TCP socket which listens on a random port at the localhost address.
    /// Returns the socket file descriptor and the actual socket address the socket is listening on.
    pub fn make_listener_ipv4(
        options: libc::c_int,
    ) -> io::Result<(libc::c_int, libc::sockaddr_in)> {
        let sockfd =
            unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM | options, 0))? };
        // Turn address into socket address with a random free port.
        let addr = sock_addr_ipv4(IPV4_LOCALHOST, 0);
        unsafe {
            errno_result(libc::bind(
                sockfd,
                (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
                size_of::<libc::sockaddr_in>() as libc::socklen_t,
            ))?;
        }

        unsafe {
            errno_result(libc::listen(sockfd, 16))?;
        }

        // Retrieve actual listener address because we used a randomized port.
        let (_, addr_with_port) =
            sockname_ipv4(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })?;

        Ok((sockfd, addr_with_port))
    }

    /// Create an IPv6 TCP socket which listens on a random port at the localhost address.
    /// Returns the socket file descriptor and the actual socket address the socket is listening on.
    pub fn make_listener_ipv6(
        options: libc::c_int,
    ) -> io::Result<(libc::c_int, libc::sockaddr_in6)> {
        let sockfd =
            unsafe { errno_result(libc::socket(libc::AF_INET6, libc::SOCK_STREAM | options, 0))? };
        // Turn address into socket address with a random free port.
        let addr = sock_addr_ipv6(IPV6_LOCALHOST, 0);
        unsafe {
            errno_result(libc::bind(
                sockfd,
                (&addr as *const libc::sockaddr_in6).cast::<libc::sockaddr>(),
                size_of::<libc::sockaddr_in6>() as libc::socklen_t,
            ))?;
        }

        unsafe {
            errno_result(libc::listen(sockfd, 16))?;
        }

        // Retrieve actual listener address because we used a randomized port.
        let (_, addr_with_port) =
            sockname_ipv6(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })?;

        Ok((sockfd, addr_with_port))
    }

    /// Accept an incoming IPv4 connection.
    pub fn accept_ipv4(sockfd: libc::c_int) -> io::Result<(libc::c_int, libc::sockaddr_in)> {
        sockname_ipv4(|storage, len| unsafe { libc::accept(sockfd, storage, len) })
    }

    /// Accept an incoming IPv6 connection.
    pub fn accept_ipv6(sockfd: libc::c_int) -> io::Result<(libc::c_int, libc::sockaddr_in6)> {
        sockname_ipv6(|storage, len| unsafe { libc::accept(sockfd, storage, len) })
    }

    /// Connect the socket to the specified IPv4 address.
    pub fn connect_ipv4(sockfd: libc::c_int, addr: libc::sockaddr_in) {
        unsafe {
            errno_check(libc::connect(
                sockfd,
                (&addr as *const libc::sockaddr_in).cast(),
                size_of::<libc::sockaddr_in>() as libc::socklen_t,
            ));
        }
    }

    /// Connect the socket to the specified IPv6 address.
    pub fn connect_ipv6(sockfd: libc::c_int, addr: libc::sockaddr_in6) {
        unsafe {
            errno_check(libc::connect(
                sockfd,
                (&addr as *const libc::sockaddr_in6).cast(),
                size_of::<libc::sockaddr_in6>() as libc::socklen_t,
            ));
        }
    }

    /// Set a socket option. It's the caller's responsibility to ensure that `T` is
    /// associated with the given socket option.
    ///
    /// This function is directly copied from the standard library implementation
    /// for sockets on UNIX targets.
    pub fn setsockopt<T>(
        sockfd: i32,
        level: libc::c_int,
        option_name: libc::c_int,
        option_value: T,
    ) -> io::Result<()> {
        let option_len = size_of::<T>() as libc::socklen_t;

        errno_result(unsafe {
            libc::setsockopt(
                sockfd,
                level,
                option_name,
                (&raw const option_value) as *const _,
                option_len,
            )
        })?;
        Ok(())
    }

    /// Wraps a call to a platform function that returns an IPv4 socket address.
    /// Returns a tuple containing the actual return value of the performed
    /// syscall and the written address of it.
    pub fn sockname_ipv4<F>(f: F) -> io::Result<(libc::c_int, libc::sockaddr_in)>
    where
        F: FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> libc::c_int,
    {
        let (result, addr) = sockname(f)?;
        let LibcSocketAddr::V4(addr) = addr else { panic!("expected IPv4 address") };

        Ok((result, addr))
    }

    /// Wraps a call to a platform function that returns an IPv6 socket address.
    /// Returns a tuple containing the actual return value of the performed
    /// syscall and the written address of it.
    pub fn sockname_ipv6<F>(f: F) -> io::Result<(libc::c_int, libc::sockaddr_in6)>
    where
        F: FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> libc::c_int,
    {
        let (result, addr) = sockname(f)?;
        let LibcSocketAddr::V6(addr) = addr else { panic!("expected IPv6 address") };

        Ok((result, addr))
    }

    enum LibcSocketAddr {
        V4(libc::sockaddr_in),
        V6(libc::sockaddr_in6),
    }

    /// Wraps a call to a platform function that returns a socket address.
    /// This is very much the same as the function with the same name in the
    /// standard library implementation.
    /// Returns a tuple containing the actual return value of the performed
    /// syscall and the written address of it.
    fn sockname<F>(f: F) -> io::Result<(libc::c_int, LibcSocketAddr)>
    where
        F: FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> libc::c_int,
    {
        let mut storage = std::mem::MaybeUninit::<libc::sockaddr_storage>::zeroed();
        let mut len = size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        let value = errno_result(f(storage.as_mut_ptr().cast(), &mut len))?;
        // SAFETY:
        // The caller guarantees that the storage has been successfully initialized
        // and its size written to `len` if `f` returns a success.
        let address = unsafe {
            match (*storage.as_ptr()).ss_family as libc::c_int {
                libc::AF_INET => {
                    assert!(len as usize >= size_of::<libc::sockaddr_in>());
                    LibcSocketAddr::V4(*(storage.as_ptr() as *const _ as *const libc::sockaddr_in))
                }
                libc::AF_INET6 => {
                    assert!(len as usize >= size_of::<libc::sockaddr_in6>());
                    LibcSocketAddr::V6(*(storage.as_ptr() as *const _ as *const libc::sockaddr_in6))
                }
                _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid argument")),
            }
        };

        Ok((value, address))
    }

    pub unsafe fn recv_all(
        fd: libc::c_int,
        buf: *mut libc::c_void,
        count: libc::size_t,
        flags: libc::c_int,
    ) -> libc::ssize_t {
        assert!(count > 0);
        let mut read_so_far = 0;
        while read_so_far < count {
            let res = libc::recv(fd, buf.add(read_so_far), count - read_so_far, flags);
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

    pub unsafe fn send_all(
        fd: libc::c_int,
        buf: *const libc::c_void,
        count: libc::size_t,
        flags: libc::c_int,
    ) -> libc::ssize_t {
        assert!(count > 0);
        let mut written_so_far = 0;
        while written_so_far < count {
            let res = libc::send(fd, buf.add(written_so_far), count - written_so_far, flags);
            if res < 0 {
                return res;
            }
            // Apparently a return value of 0 is just a short write, nothing special (unlike reads).
            written_so_far += res as libc::size_t;
        }
        return written_so_far as libc::ssize_t;
    }
}
