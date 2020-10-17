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
use rustc_codegen_ssa::traits::*;
use rustc_middle::ty::Ty;
use tracing::debug;

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
    let llfn = unsafe {
        llvm::LLVMRustGetOrInsertFunction(cx.llmod, name.as_ptr().cast(), name.len(), ty)
    };

    llvm::SetFunctionCallConv(llfn, callconv);
    // Function addresses in Rust are never significant, allowing functions to
    // be merged.
    llvm::SetUnnamedAddress(llfn, llvm::UnnamedAddr::Global);

    if cx.tcx.sess.opts.cg.no_redzone.unwrap_or(cx.tcx.sess.target.options.disable_redzone) {
        llvm::Attribute::NoRedZone.apply_llfn(Function, llfn);
    }

    attributes::default_optimisation_attrs(cx.tcx.sess, llfn);
    attributes::non_lazy_bind(cx.sess(), llfn);
    llfn
}

impl CodegenCx<'ll, 'tcx> {
    /// Declare a global value.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// return its Value instead.
    pub fn declare_global(&self, name: &str, ty: &'ll Type) -> &'ll Value {
        debug!("declare_global(name={:?})", name);
        unsafe { llvm::LLVMRustGetOrInsertGlobal(self.llmod, name.as_ptr().cast(), name.len(), ty) }
    }

    /// Declare a C ABI function.
    ///
    /// Only use this for foreign function ABIs and glue. For Rust functions use
    /// `declare_fn` instead.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    pub fn declare_cfn(&self, name: &str, fn_type: &'ll Type) -> &'ll Value {
        declare_raw_fn(self, name, llvm::CCallConv, fn_type)
    }

    /// Declare a Rust function.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    pub fn declare_fn(&self, name: &str, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> &'ll Value {
        debug!("declare_rust_fn(name={:?}, fn_abi={:?})", name, fn_abi);

        let llfn = declare_raw_fn(self, name, fn_abi.llvm_cconv(), fn_abi.llvm_type(self));
        fn_abi.apply_attrs_llfn(self, llfn);
        llfn
    }

    /// Declare a global with an intention to define it.
    ///
    /// Use this function when you intend to define a global. This function will
    /// return `None` if the name already has a definition associated with it. In that
    /// case an error should be reported to the user, because it usually happens due
    /// to user’s fault (e.g., misuse of `#[no_mangle]` or `#[export_name]` attributes).
    pub fn define_global(&self, name: &str, ty: &'ll Type) -> Option<&'ll Value> {
        if self.get_defined_value(name).is_some() {
            None
        } else {
            Some(self.declare_global(name, ty))
        }
    }

    /// Declare a private global
    ///
    /// Use this function when you intend to define a global without a name.
    pub fn define_private_global(&self, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMRustInsertPrivateGlobal(self.llmod, ty) }
    }

    /// Gets declared value by name.
    pub fn get_declared_value(&self, name: &str) -> Option<&'ll Value> {
        debug!("get_declared_value(name={:?})", name);
        unsafe { llvm::LLVMRustGetNamedValue(self.llmod, name.as_ptr().cast(), name.len()) }
    }

    /// Gets defined or externally defined (AvailableExternally linkage) value by
    /// name.
    pub fn get_defined_value(&self, name: &str) -> Option<&'ll Value> {
        self.get_declared_value(name).and_then(|val| {
            let declaration = unsafe { llvm::LLVMIsDeclaration(val) != 0 };
            if !declaration { Some(val) } else { None }
        })
    }
}
