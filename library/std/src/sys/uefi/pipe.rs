//! An implementation of Pipes using UEFI variables

use crate::os::uefi::ffi::OsStrExt;
use crate::{
    ffi::OsStr,
    io::{self, IoSlice, IoSliceMut},
};

pub struct AnonPipe(uefi_pipe::Pipe);

impl AnonPipe {
    pub(crate) fn new<K: AsRef<OsStr>>(key: K) -> Self {
        AnonPipe(uefi_pipe::Pipe::new(key.as_ref().to_ffi_string()))
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn diverge(&self) -> ! {
        todo!()
    }
}

pub fn read2(p1: AnonPipe, v1: &mut Vec<u8>, p2: AnonPipe, v2: &mut Vec<u8>) -> io::Result<()> {
    p1.0.read_to_end(v1)?;
    p2.0.read_to_end(v2)?;
    Ok(())
}

// Impementation of Pipes using variables in UEFI. Might evolve into a Protocol or something in the
// future
pub(crate) mod uefi_pipe {
    use super::super::common;
    use crate::io;
    use crate::os::uefi;
    use crate::os::uefi::io::status_to_io_error;

    type Ucs2Key = Vec<u16>;

    const PIPE_GUID: r_efi::efi::Guid = r_efi::efi::Guid::from_fields(
        0x49e41342,
        0x5446,
        0x4914,
        0x92,
        0xc3,
        &[0xa6, 0x40, 0xee, 0x90, 0x18, 0xd9],
    );

    pub struct Pipe {
        key: Ucs2Key,
    }

    impl Pipe {
        pub fn new(key: Ucs2Key) -> Self {
            Pipe { key }
        }

        pub fn clear(&self) -> io::Result<()> {
            unsafe {
                Self::write_raw(
                    self.key.as_ptr() as *mut u16,
                    r_efi::efi::VARIABLE_BOOTSERVICE_ACCESS,
                    0,
                    crate::ptr::null_mut(),
                )
            }
        }

        pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
            let mut buf_size = buf.len();

            match unsafe {
                Self::read_raw(
                    self.key.as_ptr() as *mut u16,
                    &mut buf_size,
                    buf.as_mut_ptr().cast(),
                )
            } {
                Ok(()) => {
                    // Reaching this means whole buffer has been read
                    let _ = self.clear();
                    return Ok(buf_size);
                }
                Err(e) => match e.kind() {
                    io::ErrorKind::FileTooLarge => {}
                    // Variable Already Cleared
                    io::ErrorKind::NotFound => return Ok(0),
                    _ => return Err(e),
                },
            }

            let mut new_buf: Vec<u8> = Vec::with_capacity(buf_size);
            unsafe {
                Self::read_raw(
                    self.key.as_ptr() as *mut u16,
                    &mut buf_size,
                    new_buf.as_mut_ptr().cast(),
                )
            }?;
            unsafe { new_buf.set_len(buf_size) };

            buf.copy_from_slice(&new_buf[..(buf.len())]);
            unsafe {
                Self::write_raw(
                    self.key.as_ptr() as *mut u16,
                    r_efi::efi::VARIABLE_BOOTSERVICE_ACCESS,
                    buf_size - buf.len(),
                    (&mut new_buf[(buf.len())..]).as_mut_ptr().cast(),
                )
            }?;
            Ok(buf.len())
        }

        pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<()> {
            let mut buf_size = buf.capacity();

            match unsafe {
                Self::read_raw(
                    self.key.as_ptr() as *mut u16,
                    &mut buf_size,
                    buf.as_mut_ptr().cast(),
                )
            } {
                Ok(()) => {
                    // Reaching this means whole buffer has been read
                    let _ = self.clear();
                    unsafe { buf.set_len(buf_size) };
                    return Ok(());
                }
                Err(e) => match e.kind() {
                    io::ErrorKind::FileTooLarge => {}
                    // Variable Already Cleared
                    io::ErrorKind::NotFound => return Ok(()),
                    _ => return Err(e),
                },
            }

            buf.reserve(buf_size);
            unsafe {
                Self::read_raw(
                    self.key.as_ptr() as *mut u16,
                    &mut buf_size,
                    buf.as_mut_ptr().cast(),
                )
            }?;
            unsafe { buf.set_len(buf_size) };

            let _ = self.clear();
            Ok(())
        }

        unsafe fn read_raw(
            key: *mut u16,
            buf_size: *mut usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let runtime_services =
                uefi::env::get_runtime_services().ok_or(common::RUNTIME_SERVICES_ERROR)?;
            let mut guid = PIPE_GUID;
            let r = unsafe {
                ((*runtime_services.as_ptr()).get_variable)(
                    key,
                    &mut guid,
                    crate::ptr::null_mut(),
                    buf_size,
                    buf,
                )
            };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
            let mut buf = buf.to_vec();
            let buf_len = buf.len();
            const ATTR: u32 =
                r_efi::efi::VARIABLE_BOOTSERVICE_ACCESS | r_efi::efi::VARIABLE_APPEND_WRITE;
            match unsafe {
                Self::write_raw(
                    self.key.as_ptr() as *mut u16,
                    ATTR,
                    buf_len,
                    buf.as_mut_ptr().cast(),
                )
            } {
                Ok(_) => Ok(buf_len),
                Err(e) => Err(e),
            }
        }

        unsafe fn write_raw(
            key: *mut u16,
            attr: u32,
            buf_len: usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let runtime_services =
                uefi::env::get_runtime_services().ok_or(common::RUNTIME_SERVICES_ERROR)?;
            let mut guid = PIPE_GUID;
            let r = unsafe {
                ((*runtime_services.as_ptr()).set_variable)(key, &mut guid, attr, buf_len, buf)
            };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }
    }
}
