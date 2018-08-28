// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![deny(bare_trait_objects)]

pub use self::IntPredicate::*;
pub use self::RealPredicate::*;
pub use self::AtomicRmwBinOp::*;
pub use self::MetadataType::*;
pub use self::CodeGenOptSize::*;
pub use self::CallConv::*;
pub use self::Linkage::*;

use std::str::FromStr;
use std::string::FromUtf8Error;
use std::slice;
use std::ffi::CStr;
use std::cell::RefCell;
use libc::{self, c_uint, c_char, size_t};
use rustc_data_structures::small_c_str::SmallCStr;
use common;

pub mod archive_ro;
pub mod diagnostic;
mod ffi;

pub use self::ffi::*;

impl LLVMRustResult {
    pub fn into_result(self) -> Result<(), ()> {
        match self {
            LLVMRustResult::Success => Ok(()),
            LLVMRustResult::Failure => Err(()),
        }
    }
}

pub fn AddFunctionAttrStringValue(llfn: &'a Value,
                                  idx: AttributePlace,
                                  attr: &CStr,
                                  value: &CStr) {
    unsafe {
        LLVMRustAddFunctionAttrStringValue(llfn,
                                           idx.as_uint(),
                                           attr.as_ptr(),
                                           value.as_ptr())
    }
}

#[derive(Copy, Clone)]
pub enum AttributePlace {
    ReturnValue,
    Argument(u32),
    Function,
}

impl AttributePlace {
    pub fn as_uint(self) -> c_uint {
        match self {
            AttributePlace::ReturnValue => 0,
            AttributePlace::Argument(i) => 1 + i,
            AttributePlace::Function => !0,
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptSize {
    CodeGenOptSizeNone = 0,
    CodeGenOptSizeDefault = 1,
    CodeGenOptSizeAggressive = 2,
}

impl FromStr for ArchiveKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gnu" => Ok(ArchiveKind::K_GNU),
            "bsd" => Ok(ArchiveKind::K_BSD),
            "coff" => Ok(ArchiveKind::K_COFF),
            _ => Err(()),
        }
    }
}

#[repr(C)]
pub struct RustString {
    bytes: RefCell<Vec<u8>>,
}

/// Appending to a Rust string -- used by RawRustStringOstream.
#[no_mangle]
pub unsafe extern "C" fn LLVMRustStringWriteImpl(sr: &RustString,
                                                 ptr: *const c_char,
                                                 size: size_t) {
    let slice = slice::from_raw_parts(ptr as *const u8, size as usize);

    sr.bytes.borrow_mut().extend_from_slice(slice);
}

pub fn SetInstructionCallConv(instr: &'a Value, cc: CallConv) {
    unsafe {
        LLVMSetInstructionCallConv(instr, cc as c_uint);
    }
}
pub fn SetFunctionCallConv(fn_: &'a Value, cc: CallConv) {
    unsafe {
        LLVMSetFunctionCallConv(fn_, cc as c_uint);
    }
}

// Externally visible symbols that might appear in multiple codegen units need to appear in
// their own comdat section so that the duplicates can be discarded at link time. This can for
// example happen for generics when using multiple codegen units. This function simply uses the
// value's name as the comdat value to make sure that it is in a 1-to-1 relationship to the
// function.
// For more details on COMDAT sections see e.g. http://www.airs.com/blog/archives/52
pub fn SetUniqueComdat(llmod: &Module, val: &'a Value) {
    unsafe {
        LLVMRustSetComdat(llmod, val, LLVMGetValueName(val));
    }
}

pub fn UnsetComdat(val: &'a Value) {
    unsafe {
        LLVMRustUnsetComdat(val);
    }
}

pub fn SetUnnamedAddr(global: &'a Value, unnamed: bool) {
    unsafe {
        LLVMSetUnnamedAddr(global, unnamed as Bool);
    }
}

pub fn set_thread_local(global: &'a Value, is_thread_local: bool) {
    unsafe {
        LLVMSetThreadLocal(global, is_thread_local as Bool);
    }
}
pub fn set_thread_local_mode(global: &'a Value, mode: ThreadLocalMode) {
    unsafe {
        LLVMSetThreadLocalMode(global, mode);
    }
}

impl Attribute {
    pub fn apply_llfn(&self, idx: AttributePlace, llfn: &Value) {
        unsafe { LLVMRustAddFunctionAttribute(llfn, idx.as_uint(), *self) }
    }

    pub fn apply_callsite(&self, idx: AttributePlace, callsite: &Value) {
        unsafe { LLVMRustAddCallSiteAttribute(callsite, idx.as_uint(), *self) }
    }

    pub fn unapply_llfn(&self, idx: AttributePlace, llfn: &Value) {
        unsafe { LLVMRustRemoveFunctionAttributes(llfn, idx.as_uint(), *self) }
    }

    pub fn toggle_llfn(&self, idx: AttributePlace, llfn: &Value, set: bool) {
        if set {
            self.apply_llfn(idx, llfn);
        } else {
            self.unapply_llfn(idx, llfn);
        }
    }
}

// Memory-managed interface to object files.

pub struct ObjectFile {
    pub llof: &'static mut ffi::ObjectFile,
}

unsafe impl Send for ObjectFile {}

impl ObjectFile {
    // This will take ownership of llmb
    pub fn new(llmb: &'static mut MemoryBuffer) -> Option<ObjectFile> {
        unsafe {
            let llof = LLVMCreateObjectFile(llmb)?;
            Some(ObjectFile { llof: llof })
        }
    }
}

impl Drop for ObjectFile {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeObjectFile(&mut *(self.llof as *mut _));
        }
    }
}

// Memory-managed interface to section iterators.

pub struct SectionIter<'a> {
    pub llsi: &'a mut SectionIterator<'a>,
}

impl Drop for SectionIter<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeSectionIterator(&mut *(self.llsi as *mut _));
        }
    }
}

pub fn mk_section_iter(llof: &'a ffi::ObjectFile) -> SectionIter<'a> {
    unsafe { SectionIter { llsi: LLVMGetSections(llof) } }
}

/// Safe wrapper around `LLVMGetParam`, because segfaults are no fun.
pub fn get_param(llfn: &'a Value, index: c_uint) -> &'a Value {
    unsafe {
        assert!(index < LLVMCountParams(llfn),
            "out of bounds argument access: {} out of {} arguments", index, LLVMCountParams(llfn));
        LLVMGetParam(llfn, index)
    }
}

pub fn build_string(f: impl FnOnce(&RustString)) -> Result<String, FromUtf8Error> {
    let sr = RustString {
        bytes: RefCell::new(Vec::new()),
    };
    f(&sr);
    String::from_utf8(sr.bytes.into_inner())
}

pub fn twine_to_string(tr: &Twine) -> String {
    unsafe {
        build_string(|s| LLVMRustWriteTwineToString(tr, s))
            .expect("got a non-UTF8 Twine from LLVM")
    }
}

pub fn last_error() -> Option<String> {
    unsafe {
        let cstr = LLVMRustGetLastError();
        if cstr.is_null() {
            None
        } else {
            let err = CStr::from_ptr(cstr).to_bytes();
            let err = String::from_utf8_lossy(err).to_string();
            libc::free(cstr as *mut _);
            Some(err)
        }
    }
}

pub struct OperandBundleDef<'a> {
    pub raw: &'a mut ffi::OperandBundleDef<'a>,
}

impl OperandBundleDef<'a> {
    pub fn new(name: &str, vals: &[&'a Value]) -> Self {
        let name = SmallCStr::new(name);
        let def = unsafe {
            LLVMRustBuildOperandBundleDef(name.as_ptr(), vals.as_ptr(), vals.len() as c_uint)
        };
        OperandBundleDef { raw: def }
    }

    pub fn from_generic(bundle : &common::OperandBundleDef<'a, &'a Value>) -> Self {
        Self::new(bundle.name, &[bundle.val])
    }
}

impl Drop for OperandBundleDef<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMRustFreeOperandBundleDef(&mut *(self.raw as *mut _));
        }
    }
}
