// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! Declare various LLVM values.
//!
//! Prefer using functions and methods from this module rather than calling LLVM
//! functions directly. These functions do some additional work to ensure we do
//! the right thing given the preconceptions of trans.
//!
//! Some useful guidelines:
//!
//! * Use declare_* family of methods if you are declaring, but are not
//!   interested in defining the ValueRef they return.
//! * Use define_* family of methods when you might be defining the ValueRef.
//! * When in doubt, define.

use llvm::{self, ValueRef};
use llvm::AttributePlace::Function;
use rustc::ty::Ty;
use rustc::session::config::Sanitizer;
use rustc_back::PanicStrategy;
use abi::{Abi, FnType};
use attributes;
use context::CodegenCx;
use common;
use type_::Type;
use value::Value;

use std::ffi::CString;


/// Declare a global value.
///
/// If there’s a value with the same name already declared, the function will
/// return its ValueRef instead.
pub fn declare_global(cx: &CodegenCx, name: &str, ty: Type) -> llvm::ValueRef {
    debug!("declare_global(name={:?})", name);
    let namebuf = CString::new(name).unwrap_or_else(|_|{
        bug!("name {:?} contains an interior null byte", name)
    });
    unsafe {
        llvm::LLVMRustGetOrInsertGlobal(cx.llmod, namebuf.as_ptr(), ty.to_ref())
    }
}


/// Declare a function.
///
/// If there’s a value with the same name already declared, the function will
/// update the declaration and return existing ValueRef instead.
fn declare_raw_fn(cx: &CodegenCx, name: &str, callconv: llvm::CallConv, ty: Type) -> ValueRef {
    debug!("declare_raw_fn(name={:?}, ty={:?})", name, ty);
    let namebuf = CString::new(name).unwrap_or_else(|_|{
        bug!("name {:?} contains an interior null byte", name)
    });
    let llfn = unsafe {
        llvm::LLVMRustGetOrInsertFunction(cx.llmod, namebuf.as_ptr(), ty.to_ref())
    };

    llvm::SetFunctionCallConv(llfn, callconv);
    // Function addresses in Rust are never significant, allowing functions to
    // be merged.
    llvm::SetUnnamedAddr(llfn, true);

    if cx.tcx.sess.opts.cg.no_redzone
        .unwrap_or(cx.tcx.sess.target.target.options.disable_redzone) {
        llvm::Attribute::NoRedZone.apply_llfn(Function, llfn);
    }

    if let Some(ref sanitizer) = cx.tcx.sess.opts.debugging_opts.sanitizer {
        match *sanitizer {
            Sanitizer::Address => {
                llvm::Attribute::SanitizeAddress.apply_llfn(Function, llfn);
            },
            Sanitizer::Memory => {
                llvm::Attribute::SanitizeMemory.apply_llfn(Function, llfn);
            },
            Sanitizer::Thread => {
                llvm::Attribute::SanitizeThread.apply_llfn(Function, llfn);
            },
            _ => {}
        }
    }

    match cx.tcx.sess.opts.cg.opt_level.as_ref().map(String::as_ref) {
        Some("s") => {
            llvm::Attribute::OptimizeForSize.apply_llfn(Function, llfn);
        },
        Some("z") => {
            llvm::Attribute::MinSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.apply_llfn(Function, llfn);
        },
        _ => {},
    }

    if cx.tcx.sess.panic_strategy() != PanicStrategy::Unwind {
        attributes::unwind(llfn, false);
    }

    llfn
}


/// Declare a C ABI function.
///
/// Only use this for foreign function ABIs and glue. For Rust functions use
/// `declare_fn` instead.
///
/// If there’s a value with the same name already declared, the function will
/// update the declaration and return existing ValueRef instead.
pub fn declare_cfn(cx: &CodegenCx, name: &str, fn_type: Type) -> ValueRef {
    declare_raw_fn(cx, name, llvm::CCallConv, fn_type)
}


/// Declare a Rust function.
///
/// If there’s a value with the same name already declared, the function will
/// update the declaration and return existing ValueRef instead.
pub fn declare_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, name: &str,
                            fn_type: Ty<'tcx>) -> ValueRef {
    debug!("declare_rust_fn(name={:?}, fn_type={:?})", name, fn_type);
    let sig = common::ty_fn_sig(cx, fn_type);
    let sig = cx.tcx.erase_late_bound_regions_and_normalize(&sig);
    debug!("declare_rust_fn (after region erasure) sig={:?}", sig);

    let fty = FnType::new(cx, sig, &[]);
    let llfn = declare_raw_fn(cx, name, fty.cconv, fty.llvm_type(cx));

    // FIXME(canndrew): This is_never should really be an is_uninhabited
    if sig.output().is_never() {
        llvm::Attribute::NoReturn.apply_llfn(Function, llfn);
    }

    if sig.abi != Abi::Rust && sig.abi != Abi::RustCall {
        attributes::unwind(llfn, false);
    }

    fty.apply_attrs_llfn(llfn);

    llfn
}


/// Declare a global with an intention to define it.
///
/// Use this function when you intend to define a global. This function will
/// return None if the name already has a definition associated with it. In that
/// case an error should be reported to the user, because it usually happens due
/// to user’s fault (e.g. misuse of #[no_mangle] or #[export_name] attributes).
pub fn define_global(cx: &CodegenCx, name: &str, ty: Type) -> Option<ValueRef> {
    if get_defined_value(cx, name).is_some() {
        None
    } else {
        Some(declare_global(cx, name, ty))
    }
}

/// Declare a Rust function with an intention to define it.
///
/// Use this function when you intend to define a function. This function will
/// return panic if the name already has a definition associated with it. This
/// can happen with #[no_mangle] or #[export_name], for example.
pub fn define_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                           name: &str,
                           fn_type: Ty<'tcx>) -> ValueRef {
    if get_defined_value(cx, name).is_some() {
        cx.sess().fatal(&format!("symbol `{}` already defined", name))
    } else {
        declare_fn(cx, name, fn_type)
    }
}

/// Declare a Rust function with an intention to define it.
///
/// Use this function when you intend to define a function. This function will
/// return panic if the name already has a definition associated with it. This
/// can happen with #[no_mangle] or #[export_name], for example.
pub fn define_internal_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                    name: &str,
                                    fn_type: Ty<'tcx>) -> ValueRef {
    let llfn = define_fn(cx, name, fn_type);
    unsafe { llvm::LLVMRustSetLinkage(llfn, llvm::Linkage::InternalLinkage) };
    llfn
}


/// Get declared value by name.
pub fn get_declared_value(cx: &CodegenCx, name: &str) -> Option<ValueRef> {
    debug!("get_declared_value(name={:?})", name);
    let namebuf = CString::new(name).unwrap_or_else(|_|{
        bug!("name {:?} contains an interior null byte", name)
    });
    let val = unsafe { llvm::LLVMRustGetNamedValue(cx.llmod, namebuf.as_ptr()) };
    if val.is_null() {
        debug!("get_declared_value: {:?} value is null", name);
        None
    } else {
        debug!("get_declared_value: {:?} => {:?}", name, Value(val));
        Some(val)
    }
}

/// Get defined or externally defined (AvailableExternally linkage) value by
/// name.
pub fn get_defined_value(cx: &CodegenCx, name: &str) -> Option<ValueRef> {
    get_declared_value(cx, name).and_then(|val|{
        let declaration = unsafe {
            llvm::LLVMIsDeclaration(val) != 0
        };
        if !declaration {
            Some(val)
        } else {
            None
        }
    })
}
