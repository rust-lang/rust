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

use llvm::{self, ValueRef, get_params};
use rustc::hir::def_id::DefId;
use rustc::ty::subst::{Substs, Subst};
use abi::{Abi, FnType};
use attributes;
use builder::Builder;
use common::{self, CrateContext};
use cleanup::CleanupScope;
use mir::lvalue::LvalueRef;
use monomorphize;
use consts;
use declare;
use value::Value;
use monomorphize::Instance;
use back::symbol_names::symbol_name;
use trans_item::TransItem;
use type_of;
use rustc::ty::{self, TypeFoldable};
use std::iter;

use mir::lvalue::Alignment;

fn trans_fn_once_adapter_shim<'a, 'tcx>(
    ccx: &'a CrateContext<'a, 'tcx>,
    def_id: DefId,
    substs: ty::ClosureSubsts<'tcx>,
    method_instance: Instance<'tcx>,
    llreffn: ValueRef)
    -> ValueRef
{
    if let Some(&llfn) = ccx.instances().borrow().get(&method_instance) {
        return llfn;
    }

    debug!("trans_fn_once_adapter_shim(def_id={:?}, substs={:?}, llreffn={:?})",
           def_id, substs, Value(llreffn));

    let tcx = ccx.tcx();

    // Find a version of the closure type. Substitute static for the
    // region since it doesn't really matter.
    let closure_ty = tcx.mk_closure_from_closure_substs(def_id, substs);
    let ref_closure_ty = tcx.mk_imm_ref(tcx.mk_region(ty::ReErased), closure_ty);

    // Make a version with the type of by-ref closure.
    let sig = tcx.closure_type(def_id).subst(tcx, substs.substs);
    let sig = tcx.erase_late_bound_regions_and_normalize(&sig);
    assert_eq!(sig.abi, Abi::RustCall);
    let llref_fn_sig = tcx.mk_fn_sig(
        iter::once(ref_closure_ty).chain(sig.inputs().iter().cloned()),
        sig.output(),
        sig.variadic,
        sig.unsafety,
        Abi::RustCall
    );
    let llref_fn_ty = tcx.mk_fn_ptr(ty::Binder(llref_fn_sig));
    debug!("trans_fn_once_adapter_shim: llref_fn_ty={:?}",
           llref_fn_ty);


    // Make a version of the closure type with the same arguments, but
    // with argument #0 being by value.
    let sig = tcx.mk_fn_sig(
        iter::once(closure_ty).chain(sig.inputs().iter().cloned()),
        sig.output(),
        sig.variadic,
        sig.unsafety,
        Abi::RustCall
    );

    let fn_ty = FnType::new(ccx, sig, &[]);
    let llonce_fn_ty = tcx.mk_fn_ptr(ty::Binder(sig));

    // Create the by-value helper.
    let function_name = symbol_name(method_instance, ccx.shared());
    let lloncefn = declare::define_internal_fn(ccx, &function_name, llonce_fn_ty);
    attributes::set_frame_pointer_elimination(ccx, lloncefn);

    let orig_fn_ty = fn_ty;
    let mut bcx = Builder::new_block(ccx, lloncefn, "entry-block");

    // the first argument (`self`) will be the (by value) closure env.

    let mut llargs = get_params(lloncefn);
    let fn_ty = FnType::new(ccx, llref_fn_sig, &[]);
    let self_idx = fn_ty.ret.is_indirect() as usize;
    let env_arg = &orig_fn_ty.args[0];
    let env = if env_arg.is_indirect() {
        LvalueRef::new_sized_ty(llargs[self_idx], closure_ty, Alignment::AbiAligned)
    } else {
        let scratch = LvalueRef::alloca(&bcx, closure_ty, "self");
        let mut llarg_idx = self_idx;
        env_arg.store_fn_arg(&bcx, &mut llarg_idx, scratch.llval);
        scratch
    };

    debug!("trans_fn_once_adapter_shim: env={:?}", env);
    // Adjust llargs such that llargs[self_idx..] has the call arguments.
    // For zero-sized closures that means sneaking in a new argument.
    if env_arg.is_ignore() {
        llargs.insert(self_idx, env.llval);
    } else {
        llargs[self_idx] = env.llval;
    }

    // Call the by-ref closure body with `self` in a cleanup scope,
    // to drop `self` when the body returns, or in case it unwinds.
    let self_scope = CleanupScope::schedule_drop_mem(&bcx, env);

    let llret;
    if let Some(landing_pad) = self_scope.landing_pad {
        let normal_bcx = bcx.build_sibling_block("normal-return");
        llret = bcx.invoke(llreffn, &llargs[..], normal_bcx.llbb(), landing_pad, None);
        bcx = normal_bcx;
    } else {
        llret = bcx.call(llreffn, &llargs[..], None);
    }
    fn_ty.apply_attrs_callsite(llret);

    if sig.output().is_never() {
        bcx.unreachable();
    } else {
        self_scope.trans(&bcx);

        if fn_ty.ret.is_indirect() || fn_ty.ret.is_ignore() {
            bcx.ret_void();
        } else {
            bcx.ret(llret);
        }
    }

    ccx.instances().borrow_mut().insert(method_instance, lloncefn);

    lloncefn
}


/// Translates a reference to a fn/method item, monomorphizing and
/// inlining as it goes.
///
/// # Parameters
///
/// - `ccx`: the crate context
/// - `def_id`: def id of the fn or method item being referenced
/// - `substs`: values for each of the fn/method's parameters
fn do_get_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
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

    let sym = ccx.symbol_map().get_or_compute(ccx.shared(),
                                              TransItem::Fn(instance));
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

        let attrs = instance.def.attrs(ccx.tcx());
        attributes::from_fn_attrs(ccx, &attrs, llfn);

        let is_local_def = ccx.shared().translation_items().borrow()
                              .contains(&TransItem::Fn(instance));
        if is_local_def {
            // FIXME(eddyb) Doubt all extern fn should allow unwinding.
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

pub fn get_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                        instance: Instance<'tcx>)
                        -> ValueRef
{
    match instance.def {
        ty::InstanceDef::Intrinsic(_) => {
            bug!("intrinsic {} getting reified", instance)
        }
        ty::InstanceDef::ClosureOnceShim { .. } => {
            let closure_ty = instance.substs.type_at(0);
            let (closure_def_id, closure_substs) = match closure_ty.sty {
                ty::TyClosure(def_id, substs) => (def_id, substs),
                _ => bug!("bad closure instance {:?}", instance)
            };

            trans_fn_once_adapter_shim(
                ccx,
                closure_def_id,
                closure_substs,
                instance,
                do_get_fn(
                    ccx,
                    Instance::new(closure_def_id, closure_substs.substs)
                )
            )
        }
        ty::InstanceDef::FnPtrShim(..) |
        ty::InstanceDef::Item(..) |
        ty::InstanceDef::Virtual(..) => {
            do_get_fn(ccx, instance)
        }
    }
}

pub fn resolve_and_get_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    def_id: DefId,
                                    substs: &'tcx Substs<'tcx>)
                                    -> ValueRef
{
    get_fn(ccx, monomorphize::resolve(ccx.shared(), def_id, substs))
}
