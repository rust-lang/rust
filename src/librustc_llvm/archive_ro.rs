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

use ArchiveRef;

use std::ffi::CString;
use std::path::Path;
use std::slice;
use std::str;

pub struct ArchiveRO { ptr: ArchiveRef }

pub struct Iter<'a> {
    archive: &'a ArchiveRO,
    ptr: ::ArchiveIteratorRef,
}

pub struct Child<'a> {
    name: Option<&'a str>,
    data: &'a [u8],
}

impl ArchiveRO {
    /// Opens a static archive for read-only purposes. This is more optimized
    /// than the `open` method because it uses LLVM's internal `Archive` class
    /// rather than shelling out to `ar` for everything.
    ///
    /// If this archive is used with a mutable method, then an error will be
    /// raised.
    pub fn open(dst: &Path) -> Option<ArchiveRO> {
        return unsafe {
            let s = path2cstr(dst);
            let ar = ::LLVMRustOpenArchive(s.as_ptr());
            if ar.is_null() {
                None
            } else {
                Some(ArchiveRO { ptr: ar })
            }
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
            Iter { ptr: ::LLVMRustArchiveIteratorNew(self.ptr), archive: self }
        }
    }
}

impl Drop for ArchiveRO {
    fn drop(&mut self) {
        unsafe {
            ::LLVMRustDestroyArchive(self.ptr);
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Child<'a>;

    fn next(&mut self) -> Option<Child<'a>> {
        unsafe {
            let ptr = ::LLVMRustArchiveIteratorCurrent(self.ptr);
            if ptr.is_null() {
                return None
            }
            let mut name_len = 0;
            let name_ptr = ::LLVMRustArchiveChildName(ptr, &mut name_len);
            let mut data_len = 0;
            let data_ptr = ::LLVMRustArchiveChildData(ptr, &mut data_len);
            let child = Child {
                name: if name_ptr.is_null() {
                    None
                } else {
                    let name = slice::from_raw_parts(name_ptr as *const u8,
                                                     name_len as usize);
                    str::from_utf8(name).ok().map(|s| s.trim())
                },
                data: slice::from_raw_parts(data_ptr as *const u8,
                                            data_len as usize),
            };
            ::LLVMRustArchiveIteratorNext(self.ptr);
            Some(child)
        }
    }
}

#[unsafe_destructor]
impl<'a> Drop for Iter<'a> {
    fn drop(&mut self) {
        unsafe {
            ::LLVMRustArchiveIteratorFree(self.ptr);
        }
    }
}

impl<'a> Child<'a> {
    pub fn name(&self) -> Option<&'a str> { self.name }
    pub fn data(&self) -> &'a [u8] { self.data }
}
