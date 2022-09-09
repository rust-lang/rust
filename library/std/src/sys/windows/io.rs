use crate::marker::PhantomData;
use crate::mem::size_of;
use crate::os::windows::io::{AsHandle, AsRawHandle, BorrowedHandle};
use crate::slice;
use crate::sys::{c, Align8};
use core;
use libc;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct IoSlice<'a> {
    vec: c::WSABUF,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoSlice<'a> {
        assert!(buf.len() <= c::ULONG::MAX as usize);
        IoSlice {
            vec: c::WSABUF {
                len: buf.len() as c::ULONG,
                buf: buf.as_ptr() as *mut u8 as *mut c::CHAR,
            },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        if (self.vec.len as usize) < n {
            panic!("advancing IoSlice beyond its length");
        }

        unsafe {
            self.vec.len -= n as c::ULONG;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.vec.buf as *mut u8, self.vec.len as usize) }
    }
}

#[repr(transparent)]
pub struct IoSliceMut<'a> {
    vec: c::WSABUF,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        assert!(buf.len() <= c::ULONG::MAX as usize);
        IoSliceMut {
            vec: c::WSABUF { len: buf.len() as c::ULONG, buf: buf.as_mut_ptr() as *mut c::CHAR },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        if (self.vec.len as usize) < n {
            panic!("advancing IoSliceMut beyond its length");
        }

        unsafe {
            self.vec.len -= n as c::ULONG;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.vec.buf as *mut u8, self.vec.len as usize) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.vec.buf as *mut u8, self.vec.len as usize) }
    }
}

pub fn is_terminal(h: &impl AsHandle) -> bool {
    unsafe { handle_is_console(h.as_handle()) }
}

unsafe fn handle_is_console(handle: BorrowedHandle<'_>) -> bool {
    let handle = handle.as_raw_handle();

    // A null handle means the process has no console.
    if handle.is_null() {
        return false;
    }

    let mut out = 0;
    if c::GetConsoleMode(handle, &mut out) != 0 {
        // False positives aren't possible. If we got a console then we definitely have a console.
        return true;
    }

    // At this point, we *could* have a false negative. We can determine that this is a true
    // negative if we can detect the presence of a console on any of the standard I/O streams. If
    // another stream has a console, then we know we're in a Windows console and can therefore
    // trust the negative.
    for std_handle in [c::STD_INPUT_HANDLE, c::STD_OUTPUT_HANDLE, c::STD_ERROR_HANDLE] {
        let std_handle = c::GetStdHandle(std_handle);
        if !std_handle.is_null()
            && std_handle != handle
            && c::GetConsoleMode(std_handle, &mut out) != 0
        {
            return false;
        }
    }

    // Otherwise, we fall back to an msys hack to see if we can detect the presence of a pty.
    msys_tty_on(handle)
}

unsafe fn msys_tty_on(handle: c::HANDLE) -> bool {
    const SIZE: usize = size_of::<c::FILE_NAME_INFO>() + c::MAX_PATH * size_of::<c::WCHAR>();
    let mut name_info_bytes = Align8([0u8; SIZE]);
    let res = c::GetFileInformationByHandleEx(
        handle,
        c::FileNameInfo,
        name_info_bytes.0.as_mut_ptr() as *mut libc::c_void,
        SIZE as u32,
    );
    if res == 0 {
        return false;
    }
    let name_info: &c::FILE_NAME_INFO = &*(name_info_bytes.0.as_ptr() as *const c::FILE_NAME_INFO);
    let name_len = name_info.FileNameLength as usize / 2;
    // Offset to get the `FileName` field.
    let name_ptr = name_info_bytes.0.as_ptr().offset(size_of::<c::DWORD>() as isize).cast::<u16>();
    let s = core::slice::from_raw_parts(name_ptr, name_len);
    let name = String::from_utf16_lossy(s);
    // This checks whether 'pty' exists in the file name, which indicates that
    // a pseudo-terminal is attached. To mitigate against false positives
    // (e.g., an actual file name that contains 'pty'), we also require that
    // either the strings 'msys-' or 'cygwin-' are in the file name as well.)
    let is_msys = name.contains("msys-") || name.contains("cygwin-");
    let is_pty = name.contains("-pty");
    is_msys && is_pty
}
