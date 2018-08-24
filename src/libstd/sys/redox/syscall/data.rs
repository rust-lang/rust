use core::ops::{Deref, DerefMut};
use core::{mem, slice};

#[derive(Copy, Clone, Debug, Default)]
pub struct Event {
    pub id: usize,
    pub flags: usize,
    pub data: usize
}

impl Deref for Event {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self as *const Event as *const u8,
                mem::size_of::<Event>()
            ) as &[u8]
        }
    }
}

impl DerefMut for Event {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self as *mut Event as *mut u8,
                mem::size_of::<Event>()
            ) as &mut [u8]
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Packet {
    pub id: u64,
    pub pid: usize,
    pub uid: u32,
    pub gid: u32,
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize
}

impl Deref for Packet {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self as *const Packet as *const u8,
                mem::size_of::<Packet>()
            ) as &[u8]
        }
    }
}

impl DerefMut for Packet {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self as *mut Packet as *mut u8,
                mem::size_of::<Packet>()
            ) as &mut [u8]
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SigAction {
    pub sa_handler: extern "C" fn(usize),
    pub sa_mask: [u64; 2],
    pub sa_flags: usize,
}

impl Default for SigAction {
    fn default() -> Self {
        Self {
            sa_handler: unsafe { mem::transmute(0usize) },
            sa_mask: [0; 2],
            sa_flags: 0,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Stat {
    pub st_dev: u64,
    pub st_ino: u64,
    pub st_mode: u16,
    pub st_nlink: u32,
    pub st_uid: u32,
    pub st_gid: u32,
    pub st_size: u64,
    pub st_blksize: u32,
    pub st_blocks: u64,
    pub st_mtime: u64,
    pub st_mtime_nsec: u32,
    pub st_atime: u64,
    pub st_atime_nsec: u32,
    pub st_ctime: u64,
    pub st_ctime_nsec: u32,
}

impl Deref for Stat {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self as *const Stat as *const u8,
                mem::size_of::<Stat>()
            ) as &[u8]
        }
    }
}

impl DerefMut for Stat {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self as *mut Stat as *mut u8,
                mem::size_of::<Stat>()
            ) as &mut [u8]
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct StatVfs {
    pub f_bsize: u32,
    pub f_blocks: u64,
    pub f_bfree: u64,
    pub f_bavail: u64,
}

impl Deref for StatVfs {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self as *const StatVfs as *const u8,
                mem::size_of::<StatVfs>()
            ) as &[u8]
        }
    }
}

impl DerefMut for StatVfs {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self as *mut StatVfs as *mut u8,
                mem::size_of::<StatVfs>()
            ) as &mut [u8]
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct TimeSpec {
    pub tv_sec: i64,
    pub tv_nsec: i32,
}

impl Deref for TimeSpec {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self as *const TimeSpec as *const u8,
                mem::size_of::<TimeSpec>()
            ) as &[u8]
        }
    }
}

impl DerefMut for TimeSpec {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self as *mut TimeSpec as *mut u8,
                mem::size_of::<TimeSpec>()
            ) as &mut [u8]
        }
    }
}
