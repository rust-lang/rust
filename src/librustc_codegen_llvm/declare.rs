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

use crate::abi::{FnAbi, FnAbiLlvmExt};
use crate::attributes;
use crate::context::CodegenCx;
use crate::llvm;
use crate::llvm::AttributePlace::Function;
use crate::type_::Type;
use crate::value::Value;
use log::debug;
use rustc::session::config::Sanitizer;
use rustc::ty::Ty;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::small_c_str::SmallCStr;

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
    let llfn = unsafe { llvm::LLVMRustGetOrInsertFunction(cx.llmod, namebuf.as_ptr(), ty) };

    llvm::SetFunctionCallConv(llfn, callconv);
    // Function addresses in Rust are never significant, allowing functions to
    // be merged.
    llvm::SetUnnamedAddr(llfn, true);

    if cx.tcx.sess.opts.cg.no_redzone.unwrap_or(cx.tcx.sess.target.target.options.disable_redzone) {
        llvm::Attribute::NoRedZone.apply_llfn(Function, llfn);
    }

    if let Some(ref sanitizer) = cx.tcx.sess.opts.debugging_opts.sanitizer {
        match *sanitizer {
            Sanitizer::Address => {
                llvm::Attribute::SanitizeAddress.apply_llfn(Function, llfn);
            }
            Sanitizer::Memory => {
                llvm::Attribute::SanitizeMemory.apply_llfn(Function, llfn);
            }
            Sanitizer::Thread => {
                llvm::Attribute::SanitizeThread.apply_llfn(Function, llfn);
            }
            _ => {}
        }
    }

    attributes::default_optimisation_attrs(cx.tcx.sess, llfn);
    attributes::non_lazy_bind(cx.sess(), llfn);
    llfn
}

impl DeclareMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn declare_global(&self, name: &str, ty: &'ll Type) -> &'ll Value {
        debug!("declare_global(name={:?})", name);
        unsafe { llvm::LLVMRustGetOrInsertGlobal(self.llmod, name.as_ptr().cast(), name.len(), ty) }
    }

    fn declare_cfn(&self, name: &str, fn_type: &'ll Type) -> &'ll Value {
        declare_raw_fn(self, name, llvm::CCallConv, fn_type)
    }

    fn declare_fn(&self, name: &str, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> &'ll Value {
        debug!("declare_rust_fn(name={:?}, fn_abi={:?})", name, fn_abi);

        let llfn = declare_raw_fn(self, name, fn_abi.llvm_cconv(), fn_abi.llvm_type(self));
        fn_abi.apply_attrs_llfn(self, llfn);
        llfn
    }

    fn define_global(&self, name: &str, ty: &'ll Type) -> Option<&'ll Value> {
        if self.get_defined_value(name).is_some() {
            None
        } else {
            Some(self.declare_global(name, ty))
        }
    }

    fn define_private_global(&self, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMRustInsertPrivateGlobal(self.llmod, ty) }
    }

    fn get_declared_value(&self, name: &str) -> Option<&'ll Value> {
        debug!("get_declared_value(name={:?})", name);
        let namebuf = SmallCStr::new(name);
        unsafe { llvm::LLVMRustGetNamedValue(self.llmod, namebuf.as_ptr()) }
    }

    fn get_defined_value(&self, name: &str) -> Option<&'ll Value> {
        self.get_declared_value(name).and_then(|val| {
            let declaration = unsafe { llvm::LLVMIsDeclaration(val) != 0 };
            if !declaration { Some(val) } else { None }
        })
    }
}
