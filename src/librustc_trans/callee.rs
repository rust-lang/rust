// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Handles translation of callees as well as other call-related
//! things.  Callees are a superset of normal rust values and sometimes
//! have different representations.  In particular, top-level fn items
//! and methods are represented as just a fn ptr and not a full
//! closure.

use attributes;
use common::{self, CodegenCx};
use consts;
use declare;
use llvm::{self, ValueRef};
use monomorphize::Instance;
use type_of::LayoutLlvmExt;

use rustc::hir::def_id::DefId;
use rustc::ty::{self, TypeFoldable};
use rustc::ty::layout::LayoutOf;
use rustc::traits;
use rustc::ty::subst::Substs;
use rustc_back::PanicStrategy;

/// Translates a reference to a fn/method item, monomorphizing and
/// inlining as it goes.
///
/// # Parameters
///
/// - `cx`: the crate context
/// - `instance`: the instance to be instantiated
pub fn get_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                        instance: Instance<'tcx>)
                        -> ValueRef
{
    let tcx = cx.tcx;

    debug!("get_fn(instance={:?})", instance);

    assert!(!instance.substs.needs_infer());
    assert!(!instance.substs.has_escaping_regions());
    assert!(!instance.substs.has_param_types());

    let fn_ty = instance.ty(cx.tcx);
    if let Some(&llfn) = cx.instances.borrow().get(&instance) {
        return llfn;
    }

    let sym = tcx.symbol_name(instance);
    debug!("get_fn({:?}: {:?}) => {}", instance, fn_ty, sym);

    // Create a fn pointer with the substituted signature.
    let fn_ptr_ty = tcx.mk_fn_ptr(common::ty_fn_sig(cx, fn_ty));
    let llptrty = cx.layout_of(fn_ptr_ty).llvm_type(cx);

    let llfn = if let Some(llfn) = declare::get_declared_value(cx, &sym) {
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
        if common::val_ty(llfn) != llptrty {
            debug!("get_fn: casting {:?} to {:?}", llfn, llptrty);
            consts::ptrcast(llfn, llptrty)
        } else {
            debug!("get_fn: not casting pointer!");
            llfn
        }
    } else {
        let llfn = declare::declare_fn(cx, &sym, fn_ty);
        assert_eq!(common::val_ty(llfn), llptrty);
        debug!("get_fn: not casting pointer!");

        if instance.def.is_inline(tcx) {
            attributes::inline(llfn, attributes::InlineAttr::Hint);
        }
        attributes::from_fn_attrs(cx, llfn, instance.def.def_id());

        let instance_def_id = instance.def_id();

        // Perhaps questionable, but we assume that anything defined
        // *in Rust code* may unwind. Foreign items like `extern "C" {
        // fn foo(); }` are assumed not to unwind **unless** they have
        // a `#[unwind]` attribute.
        if tcx.sess.panic_strategy() == PanicStrategy::Unwind {
            if !tcx.is_foreign_item(instance_def_id) {
                attributes::unwind(llfn, true);
            }
        }

        // Apply an appropriate linkage/visibility value to our item that we
        // just declared.
        //
        // This is sort of subtle. Inside our codegen unit we started off
        // compilation by predefining all our own `TransItem` instances. That
        // is, everything we're translating ourselves is already defined. That
        // means that anything we're actually translating ourselves will have
        // hit the above branch in `get_declared_value`. As a result, we're
        // guaranteed here that we're declaring a symbol that won't get defined,
        // or in other words we're referencing a foreign value.
        //
        // So because this is a foreign value we blanket apply an external
        // linkage directive because it's coming from a different object file.
        // The visibility here is where it gets tricky. This symbol could be
        // referencing some foreign crate or foreign library (an `extern`
        // block) in which case we want to leave the default visibility. We may
        // also, though, have multiple codegen units.
        //
        // In the situation of multiple codegen units this function may be
        // referencing a function from another codegen unit. If we're
        // indeed referencing a symbol in another codegen unit then we're in one
        // of two cases:
        //
        //  * This is a symbol defined in a foreign crate and we're just
        //    monomorphizing in another codegen unit. In this case this symbols
        //    is for sure not exported, so both codegen units will be using
        //    hidden visibility. Hence, we apply a hidden visibility here.
        //
        //  * This is a symbol defined in our local crate. If the symbol in the
        //    other codegen unit is also not exported then like with the foreign
        //    case we apply a hidden visibility. If the symbol is exported from
        //    the foreign object file, however, then we leave this at the
        //    default visibility as we'll just import it naturally.
        unsafe {
            llvm::LLVMRustSetLinkage(llfn, llvm::Linkage::ExternalLinkage);

            if cx.tcx.is_translated_function(instance_def_id) {
                if instance_def_id.is_local() {
                    if !cx.tcx.is_exported_symbol(instance_def_id) {
                        llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
                    }
                } else {
                    llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
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

pub fn resolve_and_get_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                    def_id: DefId,
                                    substs: &'tcx Substs<'tcx>)
                                    -> ValueRef
{
    get_fn(
        cx,
        ty::Instance::resolve(
            cx.tcx,
            ty::ParamEnv::empty(traits::Reveal::All),
            def_id,
            substs
        ).unwrap()
    )
}
