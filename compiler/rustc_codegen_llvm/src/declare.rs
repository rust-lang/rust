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

use std::borrow::Borrow;

use itertools::Itertools;
use rustc_codegen_ssa::traits::TypeMembershipCodegenMethods;
use rustc_data_structures::fx::FxIndexSet;
use rustc_middle::ty::{Instance, Ty};
use rustc_sanitizers::{cfi, kcfi};
use rustc_target::callconv::FnAbi;
use smallvec::SmallVec;
use tracing::debug;

use crate::abi::FnAbiLlvmExt;
use crate::common::AsCCharPtr;
use crate::context::{CodegenCx, GenericCx, SCx, SimpleCx};
use crate::llvm::AttributePlace::Function;
use crate::llvm::{FromGeneric, Visibility};
use crate::type_::Type;
use crate::value::Value;
use crate::{attributes, llvm};

/// Declare a function with a SimpleCx.
///
/// If there’s a value with the same name already declared, the function will
/// update the declaration and return existing Value instead.
pub(crate) fn declare_simple_fn<'ll>(
    cx: &SimpleCx<'ll>,
    name: &str,
    callconv: llvm::CallConv,
    unnamed: llvm::UnnamedAddr,
    visibility: llvm::Visibility,
    ty: &'ll Type,
) -> &'ll Value {
    debug!("declare_simple_fn(name={:?}, ty={:?})", name, ty);
    let llfn = unsafe {
        llvm::LLVMRustGetOrInsertFunction(cx.llmod, name.as_c_char_ptr(), name.len(), ty)
    };

    llvm::SetFunctionCallConv(llfn, callconv);
    llvm::set_unnamed_address(llfn, unnamed);
    llvm::set_visibility(llfn, visibility);

    llfn
}

/// Declare a function.
///
/// If there’s a value with the same name already declared, the function will
/// update the declaration and return existing Value instead.
pub(crate) fn declare_raw_fn<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    name: &str,
    callconv: llvm::CallConv,
    unnamed: llvm::UnnamedAddr,
    visibility: llvm::Visibility,
    ty: &'ll Type,
) -> &'ll Value {
    debug!("declare_raw_fn(name={:?}, ty={:?})", name, ty);
    let llfn = declare_simple_fn(cx, name, callconv, unnamed, visibility, ty);

    let mut attrs = SmallVec::<[_; 4]>::new();

    if cx.tcx.sess.opts.cg.no_redzone.unwrap_or(cx.tcx.sess.target.disable_redzone) {
        attrs.push(llvm::AttributeKind::NoRedZone.create_attr(cx.llcx));
    }

    attrs.extend(attributes::non_lazy_bind_attr(cx, cx.tcx.sess));

    attributes::apply_to_llfn(llfn, Function, &attrs);

    llfn
}

impl<'ll, CX: Borrow<SCx<'ll>>> GenericCx<'ll, CX> {
    /// Declare a global value.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// return its Value instead.
    pub(crate) fn declare_global(&self, name: &str, ty: &'ll Type) -> &'ll Value {
        debug!("declare_global(name={:?})", name);
        unsafe {
            llvm::LLVMRustGetOrInsertGlobal(
                (**self).borrow().llmod,
                name.as_c_char_ptr(),
                name.len(),
                ty,
            )
        }
    }
}

impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
    /// Declare a C ABI function.
    ///
    /// Only use this for foreign function ABIs and glue. For Rust functions use
    /// `declare_fn` instead.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    pub(crate) fn declare_cfn(
        &self,
        name: &str,
        unnamed: llvm::UnnamedAddr,
        fn_type: &'ll Type,
    ) -> &'ll Value {
        // Visibility should always be default for declarations, otherwise the linker may report an
        // error.
        declare_raw_fn(self, name, llvm::CCallConv, unnamed, Visibility::Default, fn_type)
    }

    /// Declare an entry Function
    ///
    /// The ABI of this function can change depending on the target (although for now the same as
    /// `declare_cfn`)
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    pub(crate) fn declare_entry_fn(
        &self,
        name: &str,
        callconv: llvm::CallConv,
        unnamed: llvm::UnnamedAddr,
        fn_type: &'ll Type,
    ) -> &'ll Value {
        let visibility = Visibility::from_generic(self.tcx.sess.default_visibility());
        declare_raw_fn(self, name, callconv, unnamed, visibility, fn_type)
    }

    /// Declare a Rust function.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    pub(crate) fn declare_fn(
        &self,
        name: &str,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        instance: Option<Instance<'tcx>>,
    ) -> &'ll Value {
        debug!("declare_rust_fn(name={:?}, fn_abi={:?})", name, fn_abi);

        // Function addresses in Rust are never significant, allowing functions to
        // be merged.
        let llfn = declare_raw_fn(
            self,
            name,
            fn_abi.llvm_cconv(self),
            llvm::UnnamedAddr::Global,
            llvm::Visibility::Default,
            fn_abi.llvm_type(self),
        );
        fn_abi.apply_attrs_llfn(self, llfn, instance);

        if self.tcx.sess.is_sanitizer_cfi_enabled() {
            if let Some(instance) = instance {
                let mut typeids = FxIndexSet::default();
                for options in [
                    cfi::TypeIdOptions::GENERALIZE_POINTERS,
                    cfi::TypeIdOptions::NORMALIZE_INTEGERS,
                    cfi::TypeIdOptions::USE_CONCRETE_SELF,
                ]
                .into_iter()
                .powerset()
                .map(cfi::TypeIdOptions::from_iter)
                {
                    let typeid = cfi::typeid_for_instance(self.tcx, instance, options);
                    if typeids.insert(typeid.clone()) {
                        self.add_type_metadata(llfn, typeid.as_bytes());
                    }
                }
            } else {
                for options in [
                    cfi::TypeIdOptions::GENERALIZE_POINTERS,
                    cfi::TypeIdOptions::NORMALIZE_INTEGERS,
                ]
                .into_iter()
                .powerset()
                .map(cfi::TypeIdOptions::from_iter)
                {
                    let typeid = cfi::typeid_for_fnabi(self.tcx, fn_abi, options);
                    self.add_type_metadata(llfn, typeid.as_bytes());
                }
            }
        }

        if self.tcx.sess.is_sanitizer_kcfi_enabled() {
            // LLVM KCFI does not support multiple !kcfi_type attachments
            let mut options = kcfi::TypeIdOptions::empty();
            if self.tcx.sess.is_sanitizer_cfi_generalize_pointers_enabled() {
                options.insert(kcfi::TypeIdOptions::GENERALIZE_POINTERS);
            }
            if self.tcx.sess.is_sanitizer_cfi_normalize_integers_enabled() {
                options.insert(kcfi::TypeIdOptions::NORMALIZE_INTEGERS);
            }

            if let Some(instance) = instance {
                let kcfi_typeid = kcfi::typeid_for_instance(self.tcx, instance, options);
                self.set_kcfi_type_metadata(llfn, kcfi_typeid);
            } else {
                let kcfi_typeid = kcfi::typeid_for_fnabi(self.tcx, fn_abi, options);
                self.set_kcfi_type_metadata(llfn, kcfi_typeid);
            }
        }

        llfn
    }
}

impl<'ll, CX: Borrow<SCx<'ll>>> GenericCx<'ll, CX> {
    /// Declare a global with an intention to define it.
    ///
    /// Use this function when you intend to define a global. This function will
    /// return `None` if the name already has a definition associated with it. In that
    /// case an error should be reported to the user, because it usually happens due
    /// to user’s fault (e.g., misuse of `#[no_mangle]` or `#[export_name]` attributes).
    pub(crate) fn define_global(&self, name: &str, ty: &'ll Type) -> Option<&'ll Value> {
        if self.get_defined_value(name).is_some() {
            None
        } else {
            Some(self.declare_global(name, ty))
        }
    }

    /// Declare a private global
    ///
    /// Use this function when you intend to define a global without a name.
    pub(crate) fn define_private_global(&self, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMRustInsertPrivateGlobal(self.llmod(), ty) }
    }

    /// Gets declared value by name.
    pub(crate) fn get_declared_value(&self, name: &str) -> Option<&'ll Value> {
        debug!("get_declared_value(name={:?})", name);
        unsafe { llvm::LLVMRustGetNamedValue(self.llmod(), name.as_c_char_ptr(), name.len()) }
    }

    /// Gets defined or externally defined (AvailableExternally linkage) value by
    /// name.
    pub(crate) fn get_defined_value(&self, name: &str) -> Option<&'ll Value> {
        self.get_declared_value(name).and_then(|val| {
            let declaration = llvm::is_declaration(val);
            if !declaration { Some(val) } else { None }
        })
    }
}
