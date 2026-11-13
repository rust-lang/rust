//@ignore-target: windows # No libc fstat on non-file FDs on Windows
//@compile-flags: -Zmiri-disable-isolation

use std::mem::MaybeUninit;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::errno_check;

fn main() {
    test_fstat_socketpair();
    test_fstat_pipe();
    #[cfg(target_os = "linux")]
    test_fstat_eventfd();
    #[cfg(target_os = "linux")]
    test_fstat_epoll();
}

/// Calls fstat and returns a reference to the result.
/// We use `assume_init_ref` rather than `assume_init` because not all fields
/// of `libc::stat` may be written by fstat (e.g. `st_lspare` on macOS).
fn do_fstat(fd: i32, buf: &mut MaybeUninit<libc::stat>) -> &libc::stat {
    let res = unsafe { libc::fstat(fd, buf.as_mut_ptr()) };
    assert_eq!(res, 0, "fstat failed on fd {}", fd);
    unsafe { buf.assume_init_ref() }
}

fn assert_stat_fields_are_accessible(stat: &libc::stat) {
    let _st_nlink = stat.st_nlink;
    let _st_blksize = stat.st_blksize;
    let _st_blocks = stat.st_blocks;
    let _st_ino = stat.st_ino;
    let _st_dev = stat.st_dev;
    let _st_uid = stat.st_uid;
    let _st_gid = stat.st_gid;
    let _st_rdev = stat.st_rdev;
    let _st_atime = stat.st_atime;
    let _st_mtime = stat.st_mtime;
    let _st_ctime = stat.st_ctime;
    let _st_atime_nsec = stat.st_atime_nsec;
    let _st_mtime_nsec = stat.st_mtime_nsec;
    let _st_ctime_nsec = stat.st_ctime_nsec;
}

/// Test fstat on socketpair file descriptors.
fn test_fstat_socketpair() {
    let mut fds = [0i32; 2];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    for fd in fds.iter() {
        let mut buf = MaybeUninit::uninit();
        let stat = do_fstat(*fd, &mut buf);
        assert_eq!(
            stat.st_mode & libc::S_IFMT,
            libc::S_IFSOCK,
            "socketpair should have S_IFSOCK mode"
        );
        assert_ne!(stat.st_mode & !libc::S_IFMT, 0, "socketpair should have permissions");
        assert_eq!(stat.st_size, 0, "socketpair should have size 0");
        assert_stat_fields_are_accessible(stat);
    }

    errno_check(unsafe { libc::close(fds[0]) });
    errno_check(unsafe { libc::close(fds[1]) });
}

/// Test fstat on pipe file descriptors.
fn test_fstat_pipe() {
    let mut fds = [0i32; 2];
    errno_check(unsafe { libc::pipe(fds.as_mut_ptr()) });

    for fd in fds.iter() {
        let mut buf = MaybeUninit::uninit();
        let stat = do_fstat(*fd, &mut buf);
        assert_eq!(stat.st_mode & libc::S_IFMT, libc::S_IFIFO, "pipe should have S_IFIFO mode");
        assert_ne!(stat.st_mode & !libc::S_IFMT, 0, "pipe should have permissions");
        assert_eq!(stat.st_size, 0, "pipe should have size 0");
        assert_stat_fields_are_accessible(stat);
    }

    errno_check(unsafe { libc::close(fds[0]) });
    errno_check(unsafe { libc::close(fds[1]) });
}

/// Test fstat on eventfd file descriptors (Linux only).
#[cfg(target_os = "linux")]
fn test_fstat_eventfd() {
    let flags = libc::EFD_CLOEXEC | libc::EFD_NONBLOCK;
    let fd = libc_utils::errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    let mut buf = MaybeUninit::uninit();
    let stat = do_fstat(fd, &mut buf);
    assert_eq!(stat.st_size, 0, "eventfd should have size 0");
    assert_stat_fields_are_accessible(stat);

    errno_check(unsafe { libc::close(fd) });
}

/// Test fstat on epoll file descriptors (Linux only).
#[cfg(target_os = "linux")]
fn test_fstat_epoll() {
    let fd = libc_utils::errno_result(unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) }).unwrap();

    let mut buf = MaybeUninit::uninit();
    let stat = do_fstat(fd, &mut buf);
    assert_eq!(stat.st_size, 0, "epoll should have size 0");
    assert_stat_fields_are_accessible(stat);

    errno_check(unsafe { libc::close(fd) });
}
