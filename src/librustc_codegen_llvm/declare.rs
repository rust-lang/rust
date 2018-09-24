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
//! the right thing given the preconceptions of codegen.
//!
//! Some useful guidelines:
//!
//! * Use declare_* family of methods if you are declaring, but are not
//!   interested in defining the Value they return.
//! * Use define_* family of methods when you might be defining the Value.
//! * When in doubt, define.

use llvm;
use llvm::AttributePlace::Function;
use rustc::ty::{self, PolyFnSig};
use rustc::ty::layout::LayoutOf;
use rustc::session::config::Sanitizer;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_target::spec::PanicStrategy;
use abi::{Abi, FnType, FnTypeExt};
use attributes;
use context::CodegenCx;
use type_::Type;
use interfaces::*;
use value::Value;

/// Declare a function.
///
/// If there’s a value with the same name already declared, the function will
/// update the declaration and return existing Value instead.
fn declare_raw_fn(
    cx: &CodegenCx<'ll, '_>,
    name: &str,
    callconv: llvm::CallConv,
    ty: &'ll Type,
) -> &'ll Value {
    debug!("declare_raw_fn(name={:?}, ty={:?})", name, ty);
    let namebuf = SmallCStr::new(name);
    let llfn = unsafe {
        llvm::LLVMRustGetOrInsertFunction(cx.llmod, namebuf.as_ptr(), ty)
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

    attributes::non_lazy_bind(cx.sess(), llfn);

    llfn
}

impl DeclareMethods<'tcx> for CodegenCx<'ll, 'tcx> {

    /// Declare a global value.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// return its Value instead.
    fn declare_global(
        &self,
        name: &str, ty: &'ll Type
    ) -> &'ll Value {
        debug!("declare_global(name={:?})", name);
        let namebuf = SmallCStr::new(name);
        unsafe {
            llvm::LLVMRustGetOrInsertGlobal(self.llmod, namebuf.as_ptr(), ty)
        }
    }

    /// Declare a C ABI function.
    ///
    /// Only use this for foreign function ABIs and glue. For Rust functions use
    /// `declare_fn` instead.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    fn declare_cfn(
        &self,
        name: &str,
        fn_type: &'ll Type
    ) -> &'ll Value {
        declare_raw_fn(self, name, llvm::CCallConv, fn_type)
    }


    /// Declare a Rust function.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    fn declare_fn(
        &self,
        name: &str,
        sig: PolyFnSig<'tcx>,
    ) -> &'ll Value {
        debug!("declare_rust_fn(name={:?}, sig={:?})", name, sig);
        let sig = self.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
        debug!("declare_rust_fn (after region erasure) sig={:?}", sig);

        let fty = FnType::new(self, sig, &[]);
        let llfn = declare_raw_fn(self, name, fty.llvm_cconv(), fty.llvm_type(self));

        if self.layout_of(sig.output()).abi.is_uninhabited() {
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
    fn define_global(
        &self,
        name: &str,
        ty: &'ll Type
    ) -> Option<&'ll Value> {
        if self.get_defined_value(name).is_some() {
            None
        } else {
            Some(self.declare_global(name, ty))
        }
    }

    /// Declare a private global
    ///
    /// Use this function when you intend to define a global without a name.
    fn define_private_global(&self, ty: &'ll Type) -> &'ll Value {
        unsafe {
            llvm::LLVMRustInsertPrivateGlobal(self.llmod, ty)
        }
    }

    /// Declare a Rust function with an intention to define it.
    ///
    /// Use this function when you intend to define a function. This function will
    /// return panic if the name already has a definition associated with it. This
    /// can happen with #[no_mangle] or #[export_name], for example.
    fn define_fn(
        &self,
        name: &str,
        fn_sig: PolyFnSig<'tcx>,
    ) -> &'ll Value {
        if self.get_defined_value(name).is_some() {
            self.sess().fatal(&format!("symbol `{}` already defined", name))
        } else {
            self.declare_fn(name, fn_sig)
        }
    }

    /// Declare a Rust function with an intention to define it.
    ///
    /// Use this function when you intend to define a function. This function will
    /// return panic if the name already has a definition associated with it. This
    /// can happen with #[no_mangle] or #[export_name], for example.
    fn define_internal_fn(
        &self,
        name: &str,
        fn_sig: PolyFnSig<'tcx>,
    ) -> &'ll Value {
        let llfn = self.define_fn(name, fn_sig);
        unsafe { llvm::LLVMRustSetLinkage(llfn, llvm::Linkage::InternalLinkage) };
        llfn
    }


    /// Get declared value by name.
    fn get_declared_value(&self, name: &str) -> Option<&'ll Value> {
        debug!("get_declared_value(name={:?})", name);
        let namebuf = SmallCStr::new(name);
        unsafe { llvm::LLVMRustGetNamedValue(self.llmod, namebuf.as_ptr()) }
    }

    /// Get defined or externally defined (AvailableExternally linkage) value by
    /// name.
    fn get_defined_value(&self, name: &str) -> Option<&'ll Value> {
        self.get_declared_value(name).and_then(|val|{
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
}
