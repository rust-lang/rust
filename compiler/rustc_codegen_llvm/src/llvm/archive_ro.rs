//! A wrapper around LLVM's archive (.a) code

use rustc_fs_util::path_to_c_string;
use std::path::Path;
use std::slice;
use std::str;

pub struct ArchiveRO {
    pub raw: &'static mut super::Archive,
}

unsafe impl Send for ArchiveRO {}

pub struct Iter<'a> {
    raw: &'a mut super::ArchiveIterator<'a>,
}

pub struct Child<'a> {
    pub raw: &'a mut super::ArchiveChild<'a>,
}

impl ArchiveRO {
    /// Opens a static archive for read-only purposes. This is more optimized
    /// than the `open` method because it uses LLVM's internal `Archive` class
    /// rather than shelling out to `ar` for everything.
    ///
    /// If this archive is used with a mutable method, then an error will be
    /// raised.
    pub fn open(dst: &Path) -> Result<ArchiveRO, String> {
        unsafe {
            let s = path_to_c_string(dst);
            let ar = super::LLVMRustOpenArchive(s.as_ptr()).ok_or_else(|| {
                super::last_error().unwrap_or_else(|| "failed to open archive".to_owned())
            })?;
            Ok(ArchiveRO { raw: ar })
        }
    }

    pub fn iter(&self) -> Iter<'_> {
        unsafe { Iter { raw: super::LLVMRustArchiveIteratorNew(self.raw) } }
    }
}

impl Drop for ArchiveRO {
    fn drop(&mut self) {
        unsafe {
            super::LLVMRustDestroyArchive(&mut *(self.raw as *mut _));
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Result<Child<'a>, String>;

    fn next(&mut self) -> Option<Result<Child<'a>, String>> {
        unsafe {
            match super::LLVMRustArchiveIteratorNext(self.raw) {
                Some(raw) => Some(Ok(Child { raw })),
                None => super::last_error().map(Err),
            }
        }
    }
}

impl<'a> Drop for Iter<'a> {
    fn drop(&mut self) {
        unsafe {
            super::LLVMRustArchiveIteratorFree(&mut *(self.raw as *mut _));
        }
    }
}

impl<'a> Child<'a> {
    pub fn name(&self) -> Option<&'a str> {
        unsafe {
            let mut name_len = 0;
            let name_ptr = super::LLVMRustArchiveChildName(self.raw, &mut name_len);
            if name_ptr.is_null() {
                None
            } else {
                let name = slice::from_raw_parts(name_ptr as *const u8, name_len as usize);
                str::from_utf8(name).ok().map(|s| s.trim())
            }
        }
    }
}

impl<'a> Drop for Child<'a> {
    fn drop(&mut self) {
        unsafe {
            super::LLVMRustArchiveChildFree(&mut *(self.raw as *mut _));
        }
    }
}
