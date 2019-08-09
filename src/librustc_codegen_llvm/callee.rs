//! Handles codegen of callees as well as other call-related
//! things. Callees are a superset of normal rust values and sometimes
//! have different representations. In particular, top-level fn items
//! and methods are represented as just a fn ptr and not a full
//! closure.

use crate::attributes;
use crate::llvm;
use crate::context::CodegenCx;
use crate::value::Value;
use rustc_codegen_ssa::traits::*;

use rustc::ty::{TypeFoldable, Instance};
use rustc::ty::layout::{LayoutOf, HasTyCtxt};

/// Codegens a reference to a fn/method item, monomorphizing and
/// inlining as it goes.
///
/// # Parameters
///
/// - `cx`: the crate context
/// - `instance`: the instance to be instantiated
pub fn get_fn(
    cx: &CodegenCx<'ll, 'tcx>,
    instance: Instance<'tcx>,
) -> &'ll Value {
    let tcx = cx.tcx();

    debug!("get_fn(instance={:?})", instance);

    assert!(!instance.substs.needs_infer());
    assert!(!instance.substs.has_escaping_bound_vars());
    assert!(!instance.substs.has_param_types());

    let sig = instance.fn_sig(cx.tcx());
    if let Some(&llfn) = cx.instances().borrow().get(&instance) {
        return llfn;
    }

    let sym = tcx.symbol_name(instance).as_str();
    debug!("get_fn({:?}: {:?}) => {}", instance, sig, sym);

    // Create a fn pointer with the substituted signature.
    let fn_ptr_ty = tcx.mk_fn_ptr(sig);
    let llptrty = cx.backend_type(cx.layout_of(fn_ptr_ty));

    let llfn = if let Some(llfn) = cx.get_declared_value(&sym) {
        // This is subtle and surprising, but sometimes we have to bitcast
        // the resulting fn pointer.  The reason has to do with external
        // functions.  If you have two crates that both bind the same C
        // library, they may not use precisely the same types: for
        // example, they will probably each declare their own structs,
        // which are distinct types from LLVM's point of view (nominal
        // types).
        //
        // Now, if those two crates are linked into an application, and
        // they contain inlined code, you can wind up with a situation
        // where both of those functions wind up being loaded into this
        // application simultaneously. In that case, the same function
        // (from LLVM's point of view) requires two types. But of course
        // LLVM won't allow one function to have two types.
        //
        // What we currently do, therefore, is declare the function with
        // one of the two types (whichever happens to come first) and then
        // bitcast as needed when the function is referenced to make sure
        // it has the type we expect.
        //
        // This can occur on either a crate-local or crate-external
        // reference. It also occurs when testing libcore and in some
        // other weird situations. Annoying.
        if cx.val_ty(llfn) != llptrty {
            debug!("get_fn: casting {:?} to {:?}", llfn, llptrty);
            cx.const_ptrcast(llfn, llptrty)
        } else {
            debug!("get_fn: not casting pointer!");
            llfn
        }
    } else {
        let llfn = cx.declare_fn(&sym, sig);
        assert_eq!(cx.val_ty(llfn), llptrty);
        debug!("get_fn: not casting pointer!");

        if instance.def.is_inline(tcx) {
            attributes::inline(cx, llfn, attributes::InlineAttr::Hint);
        }
        attributes::from_fn_attrs(cx, llfn, Some(instance.def.def_id()), sig);

        let instance_def_id = instance.def_id();

        // Apply an appropriate linkage/visibility value to our item that we
        // just declared.
        //
        // This is sort of subtle. Inside our codegen unit we started off
        // compilation by predefining all our own `MonoItem` instances. That
        // is, everything we're codegenning ourselves is already defined. That
        // means that anything we're actually codegenning in this codegen unit
        // will have hit the above branch in `get_declared_value`. As a result,
        // we're guaranteed here that we're declaring a symbol that won't get
        // defined, or in other words we're referencing a value from another
        // codegen unit or even another crate.
        //
        // So because this is a foreign value we blanket apply an external
        // linkage directive because it's coming from a different object file.
        // The visibility here is where it gets tricky. This symbol could be
        // referencing some foreign crate or foreign library (an `extern`
        // block) in which case we want to leave the default visibility. We may
        // also, though, have multiple codegen units. It could be a
        // monomorphization, in which case its expected visibility depends on
        // whether we are sharing generics or not. The important thing here is
        // that the visibility we apply to the declaration is the same one that
        // has been applied to the definition (wherever that definition may be).
        unsafe {
            llvm::LLVMRustSetLinkage(llfn, llvm::Linkage::ExternalLinkage);

            let is_generic = instance.substs.non_erasable_generics().next().is_some();

            if is_generic {
                // This is a monomorphization. Its expected visibility depends
                // on whether we are in share-generics mode.

                if cx.tcx.sess.opts.share_generics() {
                    // We are in share_generics mode.

                    if instance_def_id.is_local() {
                        // This is a definition from the current crate. If the
                        // definition is unreachable for downstream crates or
                        // the current crate does not re-export generics, the
                        // definition of the instance will have been declared
                        // as `hidden`.
                        if cx.tcx.is_unreachable_local_definition(instance_def_id) ||
                           !cx.tcx.local_crate_exports_generics() {
                            llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
                        }
                    } else {
                        // This is a monomorphization of a generic function
                        // defined in an upstream crate.
                        if cx.tcx.upstream_monomorphizations_for(instance_def_id)
                                 .map(|set| set.contains_key(instance.substs))
                                 .unwrap_or(false) {
                            // This is instantiated in another crate. It cannot
                            // be `hidden`.
                        } else {
                            // This is a local instantiation of an upstream definition.
                            // If the current crate does not re-export it
                            // (because it is a C library or an executable), it
                            // will have been declared `hidden`.
                            if !cx.tcx.local_crate_exports_generics() {
                                llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
                            }
                        }
                    }
                } else {
                    // When not sharing generics, all instances are in the same
                    // crate and have hidden visibility
                    llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
                }
            } else {
                // This is a non-generic function
                if cx.tcx.is_codegened_item(instance_def_id) {
                    // This is a function that is instantiated in the local crate

                    if instance_def_id.is_local() {
                        // This is function that is defined in the local crate.
                        // If it is not reachable, it is hidden.
                        if !cx.tcx.is_reachable_non_generic(instance_def_id) {
                            llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
                        }
                    } else {
                        // This is a function from an upstream crate that has
                        // been instantiated here. These are always hidden.
                        llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
                    }
                }
            }
        }

        if cx.use_dll_storage_attrs &&
            tcx.is_dllimport_foreign_item(instance_def_id)
        {
            unsafe {
                llvm::LLVMSetDLLStorageClass(llfn, llvm::DLLStorageClass::DllImport);
            }
        }

        llfn
    };

    cx.instances.borrow_mut().insert(instance, llfn);

    llfn
}
