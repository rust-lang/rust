#![unstable(issue = "none", feature = "windows_handle")]

#[cfg(test)]
mod tests;

use crate::cmp;
use crate::io::{self, BorrowedCursor, ErrorKind, IoSlice, IoSliceMut, Read};
use crate::mem;
use crate::os::windows::io::{
    AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, IntoRawHandle, OwnedHandle, RawHandle,
};
use crate::ptr;
use crate::sys::c;
use crate::sys::cvt;
use crate::sys_common::{AsInner, FromInner, IntoInner};

/// An owned container for `HANDLE` object, closing them on Drop.
///
/// All methods are inherited through a `Deref` impl to `RawHandle`
pub struct Handle(OwnedHandle);

impl Handle {
    pub fn new_event(manual: bool, init: bool) -> io::Result<Handle> {
        unsafe {
            let event =
                c::CreateEventW(ptr::null_mut(), manual as c::BOOL, init as c::BOOL, ptr::null());
            if event.is_null() {
                Err(io::Error::last_os_error())
            } else {
                Ok(Handle::from_raw_handle(event))
            }
        }
    }
}

impl AsInner<OwnedHandle> for Handle {
    fn as_inner(&self) -> &OwnedHandle {
        &self.0
    }
}

impl IntoInner<OwnedHandle> for Handle {
    fn into_inner(self) -> OwnedHandle {
        self.0
    }
}

impl FromInner<OwnedHandle> for Handle {
    fn from_inner(file_desc: OwnedHandle) -> Self {
        Self(file_desc)
    }
}

impl AsHandle for Handle {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.0.as_handle()
    }
}

impl AsRawHandle for Handle {
    fn as_raw_handle(&self) -> RawHandle {
        self.0.as_raw_handle()
    }
}

impl IntoRawHandle for Handle {
    fn into_raw_handle(self) -> RawHandle {
        self.0.into_raw_handle()
    }
}

impl FromRawHandle for Handle {
    unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
        Self(FromRawHandle::from_raw_handle(raw_handle))
    }
}

impl Handle {
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let res = unsafe { self.synchronous_read(buf.as_mut_ptr().cast(), buf.len(), None) };

        match res {
            Ok(read) => Ok(read as usize),

            // The special treatment of BrokenPipe is to deal with Windows
            // pipe semantics, which yields this error when *reading* from
            // a pipe after the other end has closed; we interpret that as
            // EOF on the pipe.
            Err(ref e) if e.kind() == ErrorKind::BrokenPipe => Ok(0),

            Err(e) => Err(e),
        }
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let res =
            unsafe { self.synchronous_read(buf.as_mut_ptr().cast(), buf.len(), Some(offset)) };

        match res {
            Ok(read) => Ok(read as usize),
            Err(ref e) if e.raw_os_error() == Some(c::ERROR_HANDLE_EOF as i32) => Ok(0),
            Err(e) => Err(e),
        }
    }

    pub fn read_buf(&self, mut cursor: BorrowedCursor<'_>) -> io::Result<()> {
        let res =
            unsafe { self.synchronous_read(cursor.as_mut().as_mut_ptr(), cursor.capacity(), None) };

        match res {
            Ok(read) => {
                // Safety: `read` bytes were written to the initialized portion of the buffer
                unsafe {
                    cursor.advance(read as usize);
                }
                Ok(())
            }

            // The special treatment of BrokenPipe is to deal with Windows
            // pipe semantics, which yields this error when *reading* from
            // a pipe after the other end has closed; we interpret that as
            // EOF on the pipe.
            Err(ref e) if e.kind() == ErrorKind::BrokenPipe => Ok(()),

            Err(e) => Err(e),
        }
    }

    pub unsafe fn read_overlapped(
        &self,
        buf: &mut [u8],
        overlapped: *mut c::OVERLAPPED,
    ) -> io::Result<Option<usize>> {
        let len = cmp::min(buf.len(), <c::DWORD>::MAX as usize) as c::DWORD;
        let mut amt = 0;
        let res = cvt(c::ReadFile(
            self.as_handle(),
            buf.as_ptr() as c::LPVOID,
            len,
            &mut amt,
            overlapped,
        ));
        match res {
            Ok(_) => Ok(Some(amt as usize)),
            Err(e) => {
                if e.raw_os_error() == Some(c::ERROR_IO_PENDING as i32) {
                    Ok(None)
                } else if e.raw_os_error() == Some(c::ERROR_BROKEN_PIPE as i32) {
                    Ok(Some(0))
                } else {
                    Err(e)
                }
            }
        }
    }

    pub fn overlapped_result(
        &self,
        overlapped: *mut c::OVERLAPPED,
        wait: bool,
    ) -> io::Result<usize> {
        unsafe {
            let mut bytes = 0;
            let wait = if wait { c::TRUE } else { c::FALSE };
            let res =
                cvt(c::GetOverlappedResult(self.as_raw_handle(), overlapped, &mut bytes, wait));
            match res {
                Ok(_) => Ok(bytes as usize),
                Err(e) => {
                    if e.raw_os_error() == Some(c::ERROR_HANDLE_EOF as i32)
                        || e.raw_os_error() == Some(c::ERROR_BROKEN_PIPE as i32)
                    {
                        Ok(0)
                    } else {
                        Err(e)
                    }
                }
            }
        }
    }

    pub fn cancel_io(&self) -> io::Result<()> {
        unsafe { cvt(c::CancelIo(self.as_raw_handle())).map(drop) }
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.synchronous_write(&buf, None)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        self.synchronous_write(&buf, Some(offset))
    }

    pub fn try_clone(&self) -> io::Result<Self> {
        Ok(Self(self.0.try_clone()?))
    }

    pub fn duplicate(
        &self,
        access: c::DWORD,
        inherit: bool,
        options: c::DWORD,
    ) -> io::Result<Self> {
        Ok(Self(self.0.as_handle().duplicate(access, inherit, options)?))
    }

    /// Performs a synchronous read.
    ///
    /// If the handle is opened for asynchronous I/O then this abort the process.
    /// See #81357.
    ///
    /// If `offset` is `None` then the current file position is used.
    unsafe fn synchronous_read(
        &self,
        buf: *mut mem::MaybeUninit<u8>,
        len: usize,
        offset: Option<u64>,
    ) -> io::Result<usize> {
        let mut io_status = c::IO_STATUS_BLOCK::default();

        // The length is clamped at u32::MAX.
        let len = cmp::min(len, c::DWORD::MAX as usize) as c::DWORD;
        let status = c::NtReadFile(
            self.as_handle(),
            ptr::null_mut(),
            None,
            ptr::null_mut(),
            &mut io_status,
            buf,
            len,
            offset.map(|n| n as _).as_ref(),
            None,
        );

        let status = if status == c::STATUS_PENDING {
            c::WaitForSingleObject(self.as_raw_handle(), c::INFINITE);
            io_status.status()
        } else {
            status
        };
        match status {
            // If the operation has not completed then abort the process.
            // Doing otherwise means that the buffer and stack may be written to
            // after this function returns.
            c::STATUS_PENDING => rtabort!("I/O error: operation failed to complete synchronously"),

            // Return `Ok(0)` when there's nothing more to read.
            c::STATUS_END_OF_FILE => Ok(0),

            // Success!
            status if c::nt_success(status) => Ok(io_status.Information),

            status => {
                let error = c::RtlNtStatusToDosError(status);
                Err(io::Error::from_raw_os_error(error as _))
            }
        }
    }

    /// Performs a synchronous write.
    ///
    /// If the handle is opened for asynchronous I/O then this abort the process.
    /// See #81357.
    ///
    /// If `offset` is `None` then the current file position is used.
    fn synchronous_write(&self, buf: &[u8], offset: Option<u64>) -> io::Result<usize> {
        let mut io_status = c::IO_STATUS_BLOCK::default();

        // The length is clamped at u32::MAX.
        let len = cmp::min(buf.len(), c::DWORD::MAX as usize) as c::DWORD;
        let status = unsafe {
            c::NtWriteFile(
                self.as_handle(),
                ptr::null_mut(),
                None,
                ptr::null_mut(),
                &mut io_status,
                buf.as_ptr(),
                len,
                offset.map(|n| n as _).as_ref(),
                None,
            )
        };
        let status = if status == c::STATUS_PENDING {
            unsafe { c::WaitForSingleObject(self.as_raw_handle(), c::INFINITE) };
            io_status.status()
        } else {
            status
        };
        match status {
            // If the operation has not completed then abort the process.
            // Doing otherwise means that the buffer may be read and the stack
            // written to after this function returns.
            c::STATUS_PENDING => rtabort!("I/O error: operation failed to complete synchronously"),

            // Success!
            status if c::nt_success(status) => Ok(io_status.Information),

            status => {
                let error = unsafe { c::RtlNtStatusToDosError(status) };
                Err(io::Error::from_raw_os_error(error as _))
            }
        }
    }
}

impl<'a> Read for &'a Handle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        (**self).read_vectored(bufs)
    }
}
