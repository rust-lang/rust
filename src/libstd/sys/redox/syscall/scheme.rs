use core::{mem, slice};

use super::*;

pub trait Scheme {
    fn handle(&self, packet: &mut Packet) {
        packet.a = Error::mux(match packet.a {
            SYS_OPEN => self.open(unsafe { slice::from_raw_parts(packet.b as *const u8, packet.c) }, packet.d, packet.uid, packet.gid),
            SYS_CHMOD => self.chmod(unsafe { slice::from_raw_parts(packet.b as *const u8, packet.c) }, packet.d as u16, packet.uid, packet.gid),
            SYS_RMDIR => self.rmdir(unsafe { slice::from_raw_parts(packet.b as *const u8, packet.c) }, packet.uid, packet.gid),
            SYS_UNLINK => self.unlink(unsafe { slice::from_raw_parts(packet.b as *const u8, packet.c) }, packet.uid, packet.gid),

            SYS_DUP => self.dup(packet.b, unsafe { slice::from_raw_parts(packet.c as *const u8, packet.d) }),
            SYS_READ => self.read(packet.b, unsafe { slice::from_raw_parts_mut(packet.c as *mut u8, packet.d) }),
            SYS_WRITE => self.write(packet.b, unsafe { slice::from_raw_parts(packet.c as *const u8, packet.d) }),
            SYS_LSEEK => self.seek(packet.b, packet.c, packet.d),
            SYS_FCNTL => self.fcntl(packet.b, packet.c, packet.d),
            SYS_FEVENT => self.fevent(packet.b, packet.c),
            SYS_FMAP => self.fmap(packet.b, packet.c, packet.d),
            SYS_FPATH => self.fpath(packet.b, unsafe { slice::from_raw_parts_mut(packet.c as *mut u8, packet.d) }),
            SYS_FSTAT => if packet.d >= mem::size_of::<Stat>() { self.fstat(packet.b, unsafe { &mut *(packet.c as *mut Stat) }) } else { Err(Error::new(EFAULT)) },
            SYS_FSTATVFS => if packet.d >= mem::size_of::<StatVfs>() { self.fstatvfs(packet.b, unsafe { &mut *(packet.c as *mut StatVfs) }) } else { Err(Error::new(EFAULT)) },
            SYS_FSYNC => self.fsync(packet.b),
            SYS_FTRUNCATE => self.ftruncate(packet.b, packet.c),
            SYS_CLOSE => self.close(packet.b),

            _ => Err(Error::new(ENOSYS))
        });
    }

    /* Scheme operations */

    #[allow(unused_variables)]
    fn open(&self, path: &[u8], flags: usize, uid: u32, gid: u32) -> Result<usize> {
        Err(Error::new(ENOENT))
    }

    #[allow(unused_variables)]
    fn chmod(&self, path: &[u8], mode: u16, uid: u32, gid: u32) -> Result<usize> {
        Err(Error::new(ENOENT))
    }

    #[allow(unused_variables)]
    fn rmdir(&self, path: &[u8], uid: u32, gid: u32) -> Result<usize> {
        Err(Error::new(ENOENT))
    }

    #[allow(unused_variables)]
    fn unlink(&self, path: &[u8], uid: u32, gid: u32) -> Result<usize> {
        Err(Error::new(ENOENT))
    }

    /* Resource operations */
    #[allow(unused_variables)]
    fn dup(&self, old_id: usize, buf: &[u8]) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn read(&self, id: usize, buf: &mut [u8]) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn write(&self, id: usize, buf: &[u8]) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn seek(&self, id: usize, pos: usize, whence: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fcntl(&self, id: usize, cmd: usize, arg: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fevent(&self, id: usize, flags: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fmap(&self, id: usize, offset: usize, size: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fpath(&self, id: usize, buf: &mut [u8]) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fstat(&self, id: usize, stat: &mut Stat) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fstatvfs(&self, id: usize, stat: &mut StatVfs) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fsync(&self, id: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn ftruncate(&self, id: usize, len: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn close(&self, id: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }
}

pub trait SchemeMut {
    fn handle(&mut self, packet: &mut Packet) {
        packet.a = Error::mux(match packet.a {
            SYS_OPEN => self.open(unsafe { slice::from_raw_parts(packet.b as *const u8, packet.c) }, packet.d, packet.uid, packet.gid),
            SYS_CHMOD => self.chmod(unsafe { slice::from_raw_parts(packet.b as *const u8, packet.c) }, packet.d as u16, packet.uid, packet.gid),
            SYS_RMDIR => self.rmdir(unsafe { slice::from_raw_parts(packet.b as *const u8, packet.c) }, packet.uid, packet.gid),
            SYS_UNLINK => self.unlink(unsafe { slice::from_raw_parts(packet.b as *const u8, packet.c) }, packet.uid, packet.gid),

            SYS_DUP => self.dup(packet.b, unsafe { slice::from_raw_parts(packet.c as *const u8, packet.d) }),
            SYS_READ => self.read(packet.b, unsafe { slice::from_raw_parts_mut(packet.c as *mut u8, packet.d) }),
            SYS_WRITE => self.write(packet.b, unsafe { slice::from_raw_parts(packet.c as *const u8, packet.d) }),
            SYS_LSEEK => self.seek(packet.b, packet.c, packet.d),
            SYS_FCNTL => self.fcntl(packet.b, packet.c, packet.d),
            SYS_FEVENT => self.fevent(packet.b, packet.c),
            SYS_FMAP => self.fmap(packet.b, packet.c, packet.d),
            SYS_FPATH => self.fpath(packet.b, unsafe { slice::from_raw_parts_mut(packet.c as *mut u8, packet.d) }),
            SYS_FSTAT => if packet.d >= mem::size_of::<Stat>() { self.fstat(packet.b, unsafe { &mut *(packet.c as *mut Stat) }) } else { Err(Error::new(EFAULT)) },
            SYS_FSTATVFS => if packet.d >= mem::size_of::<StatVfs>() { self.fstatvfs(packet.b, unsafe { &mut *(packet.c as *mut StatVfs) }) } else { Err(Error::new(EFAULT)) },
            SYS_FSYNC => self.fsync(packet.b),
            SYS_FTRUNCATE => self.ftruncate(packet.b, packet.c),
            SYS_CLOSE => self.close(packet.b),

            _ => Err(Error::new(ENOSYS))
        });
    }

    /* Scheme operations */
    #[allow(unused_variables)]
    fn open(&mut self, path: &[u8], flags: usize, uid: u32, gid: u32) -> Result<usize> {
        Err(Error::new(ENOENT))
    }

    #[allow(unused_variables)]
    fn chmod(&self, path: &[u8], mode: u16, uid: u32, gid: u32) -> Result<usize> {
        Err(Error::new(ENOENT))
    }

    #[allow(unused_variables)]
    fn rmdir(&mut self, path: &[u8], uid: u32, gid: u32) -> Result<usize> {
        Err(Error::new(ENOENT))
    }

    #[allow(unused_variables)]
    fn unlink(&mut self, path: &[u8], uid: u32, gid: u32) -> Result<usize> {
        Err(Error::new(ENOENT))
    }

    /* Resource operations */
    #[allow(unused_variables)]
    fn dup(&mut self, old_id: usize, buf: &[u8]) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn read(&mut self, id: usize, buf: &mut [u8]) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn write(&mut self, id: usize, buf: &[u8]) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn seek(&mut self, id: usize, pos: usize, whence: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fcntl(&mut self, id: usize, cmd: usize, arg: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fevent(&mut self, id: usize, flags: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fmap(&mut self, id: usize, offset: usize, size: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fpath(&mut self, id: usize, buf: &mut [u8]) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fstat(&mut self, id: usize, stat: &mut Stat) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fstatvfs(&self, id: usize, stat: &mut StatVfs) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn fsync(&mut self, id: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn ftruncate(&mut self, id: usize, len: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }

    #[allow(unused_variables)]
    fn close(&mut self, id: usize) -> Result<usize> {
        Err(Error::new(EBADF))
    }
}
