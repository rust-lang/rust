use crate::io;
use crate::marker::PhantomData;
use crate::slice;
use crate::sys;
use crate::sys::c;
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

#[unstable(feature = "is_terminal", issue = "80937")]
impl io::IsTerminal for sys::stdio::Stdin {
    fn is_terminal() -> bool {
        let fd = c::STD_INPUT_HANDLE;
        let others = [c::STD_ERROR_HANDLE, c::STD_OUTPUT_HANDLE];

        if unsafe { console_on_any(&[fd]) } {
            // False positives aren't possible. If we got a console then
            // we definitely have a tty on stdin.
            return true;
        }

        // At this point, we *could* have a false negative. We can determine that
        // this is true negative if we can detect the presence of a console on
        // any of the other streams. If another stream has a console, then we know
        // we're in a Windows console and can therefore trust the negative.
        if unsafe { console_on_any(&others) } {
            return false;
        }

        // Otherwise, we fall back to a very strange msys hack to see if we can
        // sneakily detect the presence of a tty.
        unsafe { msys_tty_on(fd) }
    }
}

#[unstable(feature = "is_terminal", issue = "80937")]
impl io::IsTerminal for sys::stdio::Stdout {
    fn is_terminal() -> bool {
        let fd = c::STD_OUTPUT_HANDLE;
        let others = [c::STD_INPUT_HANDLE, c::STD_ERROR_HANDLE];

        if unsafe { console_on_any(&[fd]) } {
            // False positives aren't possible. If we got a console then
            // we definitely have a tty on stdout.
            return true;
        }

        // At this point, we *could* have a false negative. We can determine that
        // this is true negative if we can detect the presence of a console on
        // any of the other streams. If another stream has a console, then we know
        // we're in a Windows console and can therefore trust the negative.
        if unsafe { console_on_any(&others) } {
            return false;
        }

        // Otherwise, we fall back to a very strange msys hack to see if we can
        // sneakily detect the presence of a tty.
        unsafe { msys_tty_on(fd) }
    }
}

#[unstable(feature = "is_terminal", issue = "80937")]
impl io::IsTerminal for sys::stdio::Stderr {
    fn is_terminal() -> bool {
        let fd = c::STD_ERROR_HANDLE;
        let others = [c::STD_INPUT_HANDLE, c::STD_OUTPUT_HANDLE];

        if unsafe { console_on_any(&[fd]) } {
            // False positives aren't possible. If we got a console then
            // we definitely have a tty on stderr.
            return true;
        }

        // At this point, we *could* have a false negative. We can determine that
        // this is true negative if we can detect the presence of a console on
        // any of the other streams. If another stream has a console, then we know
        // we're in a Windows console and can therefore trust the negative.
        if unsafe { console_on_any(&others) } {
            return false;
        }

        // Otherwise, we fall back to a very strange msys hack to see if we can
        // sneakily detect the presence of a tty.
        unsafe { msys_tty_on(fd) }
    }
}

#[unstable(feature = "is_terminal", issue = "80937")]
unsafe fn console_on_any(fds: &[c::DWORD]) -> bool {
    for &fd in fds {
        let mut out = 0;
        let handle = c::GetStdHandle(fd);
        if c::GetConsoleMode(handle, &mut out) != 0 {
            return true;
        }
    }
    false
}
#[unstable(feature = "is_terminal", issue = "80937")]
unsafe fn msys_tty_on(fd: c::DWORD) -> bool {
    let size = core::mem::size_of::<c::FILE_NAME_INFO>();
    let mut name_info_bytes = vec![0u8; size + c::MAX_PATH * core::mem::size_of::<c::WCHAR>()];
    let res = c::GetFileInformationByHandleEx(
        c::GetStdHandle(fd),
        c::FileNameInfo,
        &mut *name_info_bytes as *mut _ as *mut libc::c_void,
        name_info_bytes.len() as u32,
    );
    if res == 0 {
        return false;
    }
    let name_info: &c::FILE_NAME_INFO = &*(name_info_bytes.as_ptr() as *const c::FILE_NAME_INFO);
    let s = core::slice::from_raw_parts(
        name_info.FileName.as_ptr(),
        name_info.FileNameLength as usize / 2,
    );
    let name = String::from_utf16_lossy(s);
    // This checks whether 'pty' exists in the file name, which indicates that
    // a pseudo-terminal is attached. To mitigate against false positives
    // (e.g., an actual file name that contains 'pty'), we also require that
    // either the strings 'msys-' or 'cygwin-' are in the file name as well.)
    let is_msys = name.contains("msys-") || name.contains("cygwin-");
    let is_pty = name.contains("-pty");
    is_msys && is_pty
}
