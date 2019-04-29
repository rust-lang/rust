use crate::io::{self, IoSlice, IoSliceMut};
use crate::mem;
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sys::fd::FileDesc;
use crate::sys::{cvt, cvt_r};

use libc::c_int;

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

pub struct AnonPipe(FileDesc);

pub fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    syscall! { fn pipe2(fds: *mut c_int, flags: c_int) -> c_int }
    static INVALID: AtomicBool = AtomicBool::new(false);

    let mut fds = [0; 2];

    // Unfortunately the only known way right now to create atomically set the
    // CLOEXEC flag is to use the `pipe2` syscall on Linux. This was added in
    // 2.6.27, however, and because we support 2.6.18 we must detect this
    // support dynamically.
    if cfg!(any(target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "linux",
                target_os = "netbsd",
                target_os = "openbsd")) &&
       !INVALID.load(Ordering::SeqCst)
    {

        // Note that despite calling a glibc function here we may still
        // get ENOSYS. Glibc has `pipe2` since 2.9 and doesn't try to
        // emulate on older kernels, so if you happen to be running on
        // an older kernel you may see `pipe2` as a symbol but still not
        // see the syscall.
        match cvt(unsafe { pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) }) {
            Ok(_) => {
                return Ok((AnonPipe(FileDesc::new(fds[0])),
                            AnonPipe(FileDesc::new(fds[1]))));
            }
            Err(ref e) if e.raw_os_error() == Some(libc::ENOSYS) => {
                INVALID.store(true, Ordering::SeqCst);
            }
            Err(e) => return Err(e),
        }
    }
    cvt(unsafe { libc::pipe(fds.as_mut_ptr()) })?;

    let fd0 = FileDesc::new(fds[0]);
    let fd1 = FileDesc::new(fds[1]);
    fd0.set_cloexec()?;
    fd1.set_cloexec()?;
    Ok((AnonPipe(fd0), AnonPipe(fd1)))
}

impl AnonPipe {
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    pub fn fd(&self) -> &FileDesc { &self.0 }
    pub fn into_fd(self) -> FileDesc { self.0 }
}

pub fn read2(p1: AnonPipe,
             v1: &mut Vec<u8>,
             p2: AnonPipe,
             v2: &mut Vec<u8>) -> io::Result<()> {

    // Set both pipes into nonblocking mode as we're gonna be reading from both
    // in the `select` loop below, and we wouldn't want one to block the other!
    let p1 = p1.into_fd();
    let p2 = p2.into_fd();
    p1.set_nonblocking(true)?;
    p2.set_nonblocking(true)?;

    let mut fds: [libc::pollfd; 2] = unsafe { mem::zeroed() };
    fds[0].fd = p1.raw();
    fds[0].events = libc::POLLIN;
    fds[1].fd = p2.raw();
    fds[1].events = libc::POLLIN;
    loop {
        // wait for either pipe to become readable using `poll`
        cvt_r(|| unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) })?;

        if fds[0].revents != 0 && read(&p1, v1)? {
            p2.set_nonblocking(false)?;
            return p2.read_to_end(v2).map(|_| ());
        }
        if fds[1].revents != 0 && read(&p2, v2)? {
            p1.set_nonblocking(false)?;
            return p1.read_to_end(v1).map(|_| ());
        }
    }

    // Read as much as we can from each pipe, ignoring EWOULDBLOCK or
    // EAGAIN. If we hit EOF, then this will happen because the underlying
    // reader will return Ok(0), in which case we'll see `Ok` ourselves. In
    // this case we flip the other fd back into blocking mode and read
    // whatever's leftover on that file descriptor.
    fn read(fd: &FileDesc, dst: &mut Vec<u8>) -> Result<bool, io::Error> {
        match fd.read_to_end(dst) {
            Ok(_) => Ok(true),
            Err(e) => {
                if e.raw_os_error() == Some(libc::EWOULDBLOCK) ||
                   e.raw_os_error() == Some(libc::EAGAIN) {
                    Ok(false)
                } else {
                    Err(e)
                }
            }
        }
    }
}
