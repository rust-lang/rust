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
#![allow(dead_code)]

#![crate_name = "rustc_llvm"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(associated_consts)]
#![feature(box_syntax)]
#![feature(concat_idents)]
#![feature(libc)]
#![feature(link_args)]
#![feature(staged_api)]
#![feature(rustc_private)]

extern crate libc;
#[macro_use]
#[no_link]
extern crate rustc_bitflags;

pub use self::IntPredicate::*;
pub use self::RealPredicate::*;
pub use self::TypeKind::*;
pub use self::AtomicRmwBinOp::*;
pub use self::MetadataType::*;
pub use self::CodeGenOptSize::*;
pub use self::DiagnosticKind::*;
pub use self::CallConv::*;
pub use self::DiagnosticSeverity::*;
pub use self::Linkage::*;

use std::str::FromStr;
use std::slice;
use std::ffi::{CString, CStr};
use std::cell::RefCell;
use libc::{c_uint, c_char, size_t};

pub mod archive_ro;
pub mod diagnostic;
pub mod ffi;

pub use ffi::*;

impl LLVMRustResult {
    pub fn into_result(self) -> Result<(), ()> {
        match self {
            LLVMRustResult::Success => Ok(()),
            LLVMRustResult::Failure => Err(()),
        }
    }
}

pub fn AddFunctionAttrStringValue(llfn: ValueRef,
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

#[repr(C)]
#[derive(Copy, Clone)]
pub enum AttributePlace {
    Argument(u32),
    Function,
}

impl AttributePlace {
    pub fn ReturnValue() -> Self {
        AttributePlace::Argument(0)
    }

    pub fn as_uint(self) -> c_uint {
        match self {
            AttributePlace::Function => !0,
            AttributePlace::Argument(i) => i,
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
            "mips64" => Ok(ArchiveKind::K_MIPS64),
            "bsd" => Ok(ArchiveKind::K_BSD),
            "coff" => Ok(ArchiveKind::K_COFF),
            _ => Err(()),
        }
    }
}

#[allow(missing_copy_implementations)]
pub enum RustString_opaque {}
pub type RustStringRef = *mut RustString_opaque;
type RustStringRepr = *mut RefCell<Vec<u8>>;

/// Appending to a Rust string -- used by RawRustStringOstream.
#[no_mangle]
pub unsafe extern "C" fn LLVMRustStringWriteImpl(sr: RustStringRef,
                                                 ptr: *const c_char,
                                                 size: size_t) {
    let slice = slice::from_raw_parts(ptr as *const u8, size as usize);

    let sr = sr as RustStringRepr;
    (*sr).borrow_mut().extend_from_slice(slice);
}

pub fn SetInstructionCallConv(instr: ValueRef, cc: CallConv) {
    unsafe {
        LLVMSetInstructionCallConv(instr, cc as c_uint);
    }
}
pub fn SetFunctionCallConv(fn_: ValueRef, cc: CallConv) {
    unsafe {
        LLVMSetFunctionCallConv(fn_, cc as c_uint);
    }
}

// Externally visible symbols that might appear in multiple translation units need to appear in
// their own comdat section so that the duplicates can be discarded at link time. This can for
// example happen for generics when using multiple codegen units. This function simply uses the
// value's name as the comdat value to make sure that it is in a 1-to-1 relationship to the
// function.
// For more details on COMDAT sections see e.g. http://www.airs.com/blog/archives/52
pub fn SetUniqueComdat(llmod: ModuleRef, val: ValueRef) {
    unsafe {
        LLVMRustSetComdat(llmod, val, LLVMGetValueName(val));
    }
}

pub fn UnsetComdat(val: ValueRef) {
    unsafe {
        LLVMRustUnsetComdat(val);
    }
}

pub fn SetUnnamedAddr(global: ValueRef, unnamed: bool) {
    unsafe {
        LLVMSetUnnamedAddr(global, unnamed as Bool);
    }
}

pub fn set_thread_local(global: ValueRef, is_thread_local: bool) {
    unsafe {
        LLVMSetThreadLocal(global, is_thread_local as Bool);
    }
}

impl Attribute {
    pub fn apply_llfn(&self, idx: AttributePlace, llfn: ValueRef) {
        unsafe { LLVMRustAddFunctionAttribute(llfn, idx.as_uint(), *self) }
    }

    pub fn apply_callsite(&self, idx: AttributePlace, callsite: ValueRef) {
        unsafe { LLVMRustAddCallSiteAttribute(callsite, idx.as_uint(), *self) }
    }

    pub fn unapply_llfn(&self, idx: AttributePlace, llfn: ValueRef) {
        unsafe { LLVMRustRemoveFunctionAttributes(llfn, idx.as_uint(), *self) }
    }

    pub fn toggle_llfn(&self, idx: AttributePlace, llfn: ValueRef, set: bool) {
        if set {
            self.apply_llfn(idx, llfn);
        } else {
            self.unapply_llfn(idx, llfn);
        }
    }
}

// Memory-managed interface to target data.

pub struct TargetData {
    pub lltd: TargetDataRef,
}

impl Drop for TargetData {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeTargetData(self.lltd);
        }
    }
}

pub fn mk_target_data(string_rep: &str) -> TargetData {
    let string_rep = CString::new(string_rep).unwrap();
    TargetData { lltd: unsafe { LLVMCreateTargetData(string_rep.as_ptr()) } }
}

// Memory-managed interface to object files.

pub struct ObjectFile {
    pub llof: ObjectFileRef,
}

impl ObjectFile {
    // This will take ownership of llmb
    pub fn new(llmb: MemoryBufferRef) -> Option<ObjectFile> {
        unsafe {
            let llof = LLVMCreateObjectFile(llmb);
            if llof as isize == 0 {
                // LLVMCreateObjectFile took ownership of llmb
                return None;
            }

            Some(ObjectFile { llof: llof })
        }
    }
}

impl Drop for ObjectFile {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeObjectFile(self.llof);
        }
    }
}

// Memory-managed interface to section iterators.

pub struct SectionIter {
    pub llsi: SectionIteratorRef,
}

impl Drop for SectionIter {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeSectionIterator(self.llsi);
        }
    }
}

pub fn mk_section_iter(llof: ObjectFileRef) -> SectionIter {
    unsafe { SectionIter { llsi: LLVMGetSections(llof) } }
}

/// Safe wrapper around `LLVMGetParam`, because segfaults are no fun.
pub fn get_param(llfn: ValueRef, index: c_uint) -> ValueRef {
    unsafe {
        assert!(index < LLVMCountParams(llfn),
            "out of bounds argument access: {} out of {} arguments", index, LLVMCountParams(llfn));
        LLVMGetParam(llfn, index)
    }
}

pub fn get_params(llfn: ValueRef) -> Vec<ValueRef> {
    unsafe {
        let num_params = LLVMCountParams(llfn);
        let mut params = Vec::with_capacity(num_params as usize);
        for idx in 0..num_params {
            params.push(LLVMGetParam(llfn, idx));
        }

        params
    }
}

pub fn build_string<F>(f: F) -> Option<String>
    where F: FnOnce(RustStringRef)
{
    let mut buf = RefCell::new(Vec::new());
    f(&mut buf as RustStringRepr as RustStringRef);
    String::from_utf8(buf.into_inner()).ok()
}

pub unsafe fn twine_to_string(tr: TwineRef) -> String {
    build_string(|s| LLVMRustWriteTwineToString(tr, s)).expect("got a non-UTF8 Twine from LLVM")
}

pub unsafe fn debug_loc_to_string(c: ContextRef, tr: DebugLocRef) -> String {
    build_string(|s| LLVMRustWriteDebugLocToString(c, tr, s))
        .expect("got a non-UTF8 DebugLoc from LLVM")
}

pub fn initialize_available_targets() {
    macro_rules! init_target(
        ($cfg:meta, $($method:ident),*) => { {
            #[cfg($cfg)]
            fn init() {
                extern {
                    $(fn $method();)*
                }
                unsafe {
                    $($method();)*
                }
            }
            #[cfg(not($cfg))]
            fn init() { }
            init();
        } }
    );
    init_target!(llvm_component = "x86",
                 LLVMInitializeX86TargetInfo,
                 LLVMInitializeX86Target,
                 LLVMInitializeX86TargetMC,
                 LLVMInitializeX86AsmPrinter,
                 LLVMInitializeX86AsmParser);
    init_target!(llvm_component = "arm",
                 LLVMInitializeARMTargetInfo,
                 LLVMInitializeARMTarget,
                 LLVMInitializeARMTargetMC,
                 LLVMInitializeARMAsmPrinter,
                 LLVMInitializeARMAsmParser);
    init_target!(llvm_component = "aarch64",
                 LLVMInitializeAArch64TargetInfo,
                 LLVMInitializeAArch64Target,
                 LLVMInitializeAArch64TargetMC,
                 LLVMInitializeAArch64AsmPrinter,
                 LLVMInitializeAArch64AsmParser);
    init_target!(llvm_component = "mips",
                 LLVMInitializeMipsTargetInfo,
                 LLVMInitializeMipsTarget,
                 LLVMInitializeMipsTargetMC,
                 LLVMInitializeMipsAsmPrinter,
                 LLVMInitializeMipsAsmParser);
    init_target!(llvm_component = "powerpc",
                 LLVMInitializePowerPCTargetInfo,
                 LLVMInitializePowerPCTarget,
                 LLVMInitializePowerPCTargetMC,
                 LLVMInitializePowerPCAsmPrinter,
                 LLVMInitializePowerPCAsmParser);
    init_target!(llvm_component = "pnacl",
                 LLVMInitializePNaClTargetInfo,
                 LLVMInitializePNaClTarget,
                 LLVMInitializePNaClTargetMC);
    init_target!(llvm_component = "systemz",
                 LLVMInitializeSystemZTargetInfo,
                 LLVMInitializeSystemZTarget,
                 LLVMInitializeSystemZTargetMC,
                 LLVMInitializeSystemZAsmPrinter,
                 LLVMInitializeSystemZAsmParser);
    init_target!(llvm_component = "jsbackend",
                 LLVMInitializeJSBackendTargetInfo,
                 LLVMInitializeJSBackendTarget,
                 LLVMInitializeJSBackendTargetMC);
    init_target!(llvm_component = "msp430",
                 LLVMInitializeMSP430TargetInfo,
                 LLVMInitializeMSP430Target,
                 LLVMInitializeMSP430TargetMC,
                 LLVMInitializeMSP430AsmPrinter);
    init_target!(llvm_component = "sparc",
                 LLVMInitializeSparcTargetInfo,
                 LLVMInitializeSparcTarget,
                 LLVMInitializeSparcTargetMC,
                 LLVMInitializeSparcAsmPrinter,
                 LLVMInitializeSparcAsmParser);
    init_target!(llvm_component = "nvptx",
                 LLVMInitializeNVPTXTargetInfo,
                 LLVMInitializeNVPTXTarget,
                 LLVMInitializeNVPTXTargetMC,
                 LLVMInitializeNVPTXAsmPrinter);
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

pub struct OperandBundleDef {
    inner: OperandBundleDefRef,
}

impl OperandBundleDef {
    pub fn new(name: &str, vals: &[ValueRef]) -> OperandBundleDef {
        let name = CString::new(name).unwrap();
        let def = unsafe {
            LLVMRustBuildOperandBundleDef(name.as_ptr(), vals.as_ptr(), vals.len() as c_uint)
        };
        OperandBundleDef { inner: def }
    }

    pub fn raw(&self) -> OperandBundleDefRef {
        self.inner
    }
}

impl Drop for OperandBundleDef {
    fn drop(&mut self) {
        unsafe {
            LLVMRustFreeOperandBundleDef(self.inner);
        }
    }
}

// The module containing the native LLVM dependencies, generated by the build system
// Note that this must come after the rustllvm extern declaration so that
// parts of LLVM that rustllvm depends on aren't thrown away by the linker.
// Works to the above fix for #15460 to ensure LLVM dependencies that
// are only used by rustllvm don't get stripped by the linker.
#[cfg(not(cargobuild))]
mod llvmdeps {
    include! { env!("CFG_LLVM_LINKAGE_FILE") }
}
