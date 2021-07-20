use crate::alloc;
use crate::ffi::CStr;
use crate::fmt;
use crate::io;
use crate::mem;
use crate::ops::Deref;
use crate::os::unix::ffi::OsStrExt;
use crate::path::Path;
use crate::ptr;
use crate::slice;
use crate::sys_common::memchr;

/// A heap-allocated C-string for doing syscalls with.
///
/// Unlike CString, this is only one pointer in size, as the size is stored
/// on the heap right before the data.
///
/// That means it's not trivially convertible to and from Vec<u8> or String,
/// but we don't need that.
///
/// Because it is so small, it can be stored inside io::Error.
pub struct OsPathBuf {
    ptr: ptr::NonNull<u8>,
}

unsafe impl Send for OsPathBuf {}
unsafe impl Sync for OsPathBuf {}

impl OsPathBuf {
    pub fn new(path: &Path) -> io::Result<Self> {
        if memchr::memchr(0, path.as_os_str().as_bytes()).is_some() {
            return Err(io::Error::new_const(
                io::ErrorKind::InvalidInput,
                &"path contains interior nul byte",
            ));
        }
        let path_len = path.as_os_str().len();
        let layout = layout_for_len(path_len);
        let ptr = ptr::NonNull::new(unsafe { alloc::alloc(layout) })
            .unwrap_or_else(|| alloc::handle_alloc_error(layout));
        unsafe {
            ptr.cast::<usize>().as_ptr().write(path_len);
            ptr.as_ptr()
                .add(mem::size_of::<usize>())
                .copy_from_nonoverlapping(path.as_os_str().as_bytes().as_ptr(), path_len);
            ptr.as_ptr().add(mem::size_of::<usize>() + path_len).write(0);
        }
        Ok(Self { ptr })
    }

    pub fn len(&self) -> usize {
        unsafe { self.ptr.cast::<usize>().as_ptr().read() }
    }

    pub fn as_ptr(&self) -> *const libc::c_char {
        unsafe { self.ptr.as_ptr().add(mem::size_of::<usize>()) as *const _ }
    }

    pub fn as_cstr(&self) -> &CStr {
        unsafe {
            CStr::from_bytes_with_nul_unchecked(slice::from_raw_parts(
                self.as_ptr() as *const u8,
                self.len() + 1,
            ))
        }
    }
}

impl Deref for OsPathBuf {
    type Target = CStr;

    fn deref(&self) -> &CStr {
        self.as_cstr()
    }
}

impl fmt::Debug for OsPathBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_cstr(), f)
    }
}

impl Drop for OsPathBuf {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), layout_for_len(self.len()));
        }
    }
}

fn layout_for_len(path_len: usize) -> alloc::Layout {
    unsafe {
        alloc::Layout::from_size_align_unchecked(
            mem::size_of::<usize>() + path_len + 1,
            mem::align_of::<usize>(),
        )
    }
}
