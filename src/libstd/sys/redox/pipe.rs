use crate::io::{self, IoSlice, IoSliceMut};
use crate::sys::{cvt, syscall};
use crate::sys::fd::FileDesc;

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

pub struct AnonPipe(FileDesc);

pub fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    let mut fds = [0; 2];
    cvt(syscall::pipe2(&mut fds, syscall::O_CLOEXEC))?;
    Ok((AnonPipe(FileDesc::new(fds[0])), AnonPipe(FileDesc::new(fds[1]))))
}

impl AnonPipe {
    pub fn from_fd(fd: FileDesc) -> io::Result<AnonPipe> {
        fd.set_cloexec()?;
        Ok(AnonPipe(fd))
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    pub fn fd(&self) -> &FileDesc { &self.0 }
    pub fn into_fd(self) -> FileDesc { self.0 }
}

pub fn read2(p1: AnonPipe,
             v1: &mut Vec<u8>,
             p2: AnonPipe,
             v2: &mut Vec<u8>) -> io::Result<()> {
    //FIXME: Use event based I/O multiplexing
    //unimplemented!()

    p1.0.read_to_end(v1)?;
    p2.0.read_to_end(v2)?;

    Ok(())

    /*
    // Set both pipes into nonblocking mode as we're gonna be reading from both
    // in the `select` loop below, and we wouldn't want one to block the other!
    let p1 = p1.into_fd();
    let p2 = p2.into_fd();
    p1.set_nonblocking(true)?;
    p2.set_nonblocking(true)?;

    loop {
        // wait for either pipe to become readable using `select`
        cvt_r(|| unsafe {
            let mut read: libc::fd_set = mem::zeroed();
            libc::FD_SET(p1.raw(), &mut read);
            libc::FD_SET(p2.raw(), &mut read);
            libc::select(max + 1, &mut read, ptr::null_mut(), ptr::null_mut(),
                         ptr::null_mut())
        })?;

        // Read as much as we can from each pipe, ignoring EWOULDBLOCK or
        // EAGAIN. If we hit EOF, then this will happen because the underlying
        // reader will return Ok(0), in which case we'll see `Ok` ourselves. In
        // this case we flip the other fd back into blocking mode and read
        // whatever's leftover on that file descriptor.
        let read = |fd: &FileDesc, dst: &mut Vec<u8>| {
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
        };
        if read(&p1, v1)? {
            p2.set_nonblocking(false)?;
            return p2.read_to_end(v2).map(|_| ());
        }
        if read(&p2, v2)? {
            p1.set_nonblocking(false)?;
            return p1.read_to_end(v1).map(|_| ());
        }
    }
    */
}
