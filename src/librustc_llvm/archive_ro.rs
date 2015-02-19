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

use libc;
use ArchiveRef;

use std::ffi::CString;
use std::mem;
use std::raw;

pub struct ArchiveRO {
    ptr: ArchiveRef,
}

impl ArchiveRO {
    /// Opens a static archive for read-only purposes. This is more optimized
    /// than the `open` method because it uses LLVM's internal `Archive` class
    /// rather than shelling out to `ar` for everything.
    ///
    /// If this archive is used with a mutable method, then an error will be
    /// raised.
    pub fn open(dst: &Path) -> Option<ArchiveRO> {
        unsafe {
            let s = CString::new(dst.as_vec()).unwrap();
            let ar = ::LLVMRustOpenArchive(s.as_ptr());
            if ar.is_null() {
                None
            } else {
                Some(ArchiveRO { ptr: ar })
            }
        }
    }

    /// Reads a file in the archive
    pub fn read<'a>(&'a self, file: &str) -> Option<&'a [u8]> {
        unsafe {
            let mut size = 0 as libc::size_t;
            let file = CString::new(file).unwrap();
            let ptr = ::LLVMRustArchiveReadSection(self.ptr, file.as_ptr(),
                                                   &mut size);
            if ptr.is_null() {
                None
            } else {
                Some(mem::transmute(raw::Slice {
                    data: ptr,
                    len: size as uint,
                }))
            }
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
