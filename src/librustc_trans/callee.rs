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
use common::{self, CrateContext};
use consts;
use declare;
use llvm::{self, ValueRef};
use monomorphize::{self, Instance};
use rustc::hir::def_id::DefId;
use rustc::ty::TypeFoldable;
use rustc::ty::subst::Substs;
use type_of;

/// Translates a reference to a fn/method item, monomorphizing and
/// inlining as it goes.
///
/// # Parameters
///
/// - `ccx`: the crate context
/// - `instance`: the instance to be instantiated
pub fn get_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                        instance: Instance<'tcx>)
                        -> ValueRef
{
    let tcx = ccx.tcx();

    debug!("get_fn(instance={:?})", instance);

    assert!(!instance.substs.needs_infer());
    assert!(!instance.substs.has_escaping_regions());
    assert!(!instance.substs.has_param_types());

    let fn_ty = common::instance_ty(ccx.shared(), &instance);
    if let Some(&llfn) = ccx.instances().borrow().get(&instance) {
        return llfn;
    }

    let sym = tcx.symbol_name(instance);
    debug!("get_fn({:?}: {:?}) => {}", instance, fn_ty, sym);

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

    // Create a fn pointer with the substituted signature.
    let fn_ptr_ty = tcx.mk_fn_ptr(common::ty_fn_sig(ccx, fn_ty));
    let llptrty = type_of::type_of(ccx, fn_ptr_ty);

    let llfn = if let Some(llfn) = declare::get_declared_value(ccx, &sym) {
        if common::val_ty(llfn) != llptrty {
            debug!("get_fn: casting {:?} to {:?}", llfn, llptrty);
            consts::ptrcast(llfn, llptrty)
        } else {
            debug!("get_fn: not casting pointer!");
            llfn
        }
    } else {
        let llfn = declare::declare_fn(ccx, &sym, fn_ty);
        assert_eq!(common::val_ty(llfn), llptrty);
        debug!("get_fn: not casting pointer!");

        if common::is_inline_instance(tcx, &instance) {
            attributes::inline(llfn, attributes::InlineAttr::Hint);
        }
        let attrs = instance.def.attrs(ccx.tcx());
        attributes::from_fn_attrs(ccx, &attrs, llfn);

        // Perhaps questionable, but we assume that anything defined
        // *in Rust code* may unwind. Foreign items like `extern "C" {
        // fn foo(); }` are assumed not to unwind **unless** they have
        // a `#[unwind]` attribute.
        if !tcx.is_foreign_item(instance.def_id()) {
            attributes::unwind(llfn, true);
            unsafe {
                llvm::LLVMRustSetLinkage(llfn, llvm::Linkage::ExternalLinkage);
            }
        }

        if ccx.use_dll_storage_attrs() &&
            ccx.sess().cstore.is_dllimport_foreign_item(instance.def_id())
        {
            unsafe {
                llvm::LLVMSetDLLStorageClass(llfn, llvm::DLLStorageClass::DllImport);
            }
        }
        llfn
    };

    ccx.instances().borrow_mut().insert(instance, llfn);

    llfn
}

pub fn resolve_and_get_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    def_id: DefId,
                                    substs: &'tcx Substs<'tcx>)
                                    -> ValueRef
{
    get_fn(ccx, monomorphize::resolve(ccx.shared(), def_id, substs))
}
