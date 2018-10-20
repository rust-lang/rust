// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A wrapper around LLVM's archive (.a) code

use std::ffi::CString;
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
        return unsafe {
            let s = path2cstr(dst);
            let ar = super::LLVMRustOpenArchive(s.as_ptr()).ok_or_else(|| {
                super::last_error().unwrap_or_else(|| "failed to open archive".to_owned())
            })?;
            Ok(ArchiveRO { raw: ar })
        };

        #[cfg(unix)]
        fn path2cstr(p: &Path) -> CString {
            use std::os::unix::prelude::*;
            use std::ffi::OsStr;
            let p: &OsStr = p.as_ref();
            CString::new(p.as_bytes()).unwrap()
        }
        #[cfg(windows)]
        fn path2cstr(p: &Path) -> CString {
            CString::new(p.to_str().unwrap()).unwrap()
        }
    }

    pub fn iter(&self) -> Iter {
        unsafe {
            Iter {
                raw: super::LLVMRustArchiveIteratorNew(self.raw),
            }
        }
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

    pub fn data(&self) -> &'a [u8] {
        unsafe {
            let mut data_len = 0;
            let data_ptr = super::LLVMRustArchiveChildData(self.raw, &mut data_len);
            if data_ptr.is_null() {
                panic!("failed to read data from archive child");
            }
            slice::from_raw_parts(data_ptr as *const u8, data_len as usize)
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
