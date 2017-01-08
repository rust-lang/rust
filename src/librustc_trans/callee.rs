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

pub use self::CalleeData::*;

use llvm::{self, ValueRef, get_params};
use rustc::hir::def_id::DefId;
use rustc::ty::subst::Substs;
use rustc::traits;
use abi::{Abi, FnType};
use attributes;
use base;
use builder::Builder;
use common::{self, CrateContext, SharedCrateContext};
use cleanup::CleanupScope;
use mir::lvalue::LvalueRef;
use consts;
use declare;
use value::Value;
use meth;
use monomorphize::{self, Instance};
use trans_item::TransItem;
use type_of;
use Disr;
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::hir;
use std::iter;

use syntax_pos::DUMMY_SP;

#[derive(Debug)]
pub enum CalleeData {
    /// Constructor for enum variant/tuple-like-struct.
    NamedTupleConstructor(Disr),

    /// Function pointer.
    Fn(ValueRef),

    Intrinsic,

    /// Trait object found in the vtable at that index.
    Virtual(usize)
}

#[derive(Debug)]
pub struct Callee<'tcx> {
    pub data: CalleeData,
    pub ty: Ty<'tcx>
}

impl<'tcx> Callee<'tcx> {
    /// Function pointer.
    pub fn ptr(llfn: ValueRef, ty: Ty<'tcx>) -> Callee<'tcx> {
        Callee {
            data: Fn(llfn),
            ty: ty
        }
    }

    /// Function or method definition.
    pub fn def<'a>(ccx: &CrateContext<'a, 'tcx>, def_id: DefId, substs: &'tcx Substs<'tcx>)
                   -> Callee<'tcx> {
        let tcx = ccx.tcx();

        if let Some(trait_id) = tcx.trait_of_item(def_id) {
            return Callee::trait_method(ccx, trait_id, def_id, substs);
        }

        let fn_ty = def_ty(ccx.shared(), def_id, substs);
        if let ty::TyFnDef(.., f) = fn_ty.sty {
            if f.abi == Abi::RustIntrinsic || f.abi == Abi::PlatformIntrinsic {
                return Callee {
                    data: Intrinsic,
                    ty: fn_ty
                };
            }
        }

        // FIXME(eddyb) Detect ADT constructors more efficiently.
        if let Some(adt_def) = fn_ty.fn_ret().skip_binder().ty_adt_def() {
            if let Some(v) = adt_def.variants.iter().find(|v| def_id == v.did) {
                return Callee {
                    data: NamedTupleConstructor(Disr::from(v.disr_val)),
                    ty: fn_ty
                };
            }
        }

        let (llfn, ty) = get_fn(ccx, def_id, substs);
        Callee::ptr(llfn, ty)
    }

    /// Trait method, which has to be resolved to an impl method.
    pub fn trait_method<'a>(ccx: &CrateContext<'a, 'tcx>,
                            trait_id: DefId,
                            def_id: DefId,
                            substs: &'tcx Substs<'tcx>)
                            -> Callee<'tcx> {
        let tcx = ccx.tcx();

        let trait_ref = ty::TraitRef::from_method(tcx, trait_id, substs);
        let trait_ref = tcx.normalize_associated_type(&ty::Binder(trait_ref));
        match common::fulfill_obligation(ccx.shared(), DUMMY_SP, trait_ref) {
            traits::VtableImpl(vtable_impl) => {
                let name = tcx.item_name(def_id);
                let (def_id, substs) = traits::find_method(tcx, name, substs, &vtable_impl);

                // Translate the function, bypassing Callee::def.
                // That is because default methods have the same ID as the
                // trait method used to look up the impl method that ended
                // up here, so calling Callee::def would infinitely recurse.
                let (llfn, ty) = get_fn(ccx, def_id, substs);
                Callee::ptr(llfn, ty)
            }
            traits::VtableClosure(vtable_closure) => {
                // The substitutions should have no type parameters remaining
                // after passing through fulfill_obligation
                let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_id).unwrap();
                let instance = Instance::new(def_id, substs);
                let llfn = trans_closure_method(
                    ccx,
                    vtable_closure.closure_def_id,
                    vtable_closure.substs,
                    instance,
                    trait_closure_kind);

                let method_ty = def_ty(ccx.shared(), def_id, substs);
                Callee::ptr(llfn, method_ty)
            }
            traits::VtableFnPointer(vtable_fn_pointer) => {
                let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_id).unwrap();
                let instance = Instance::new(def_id, substs);
                let llfn = trans_fn_pointer_shim(ccx, instance,
                                                 trait_closure_kind,
                                                 vtable_fn_pointer.fn_ty);

                let method_ty = def_ty(ccx.shared(), def_id, substs);
                Callee::ptr(llfn, method_ty)
            }
            traits::VtableObject(ref data) => {
                Callee {
                    data: Virtual(tcx.get_vtable_index_of_object_method(data, def_id)),
                    ty: def_ty(ccx.shared(), def_id, substs)
                }
            }
            vtable => {
                bug!("resolved vtable bad vtable {:?} in trans", vtable);
            }
        }
    }

    /// Get the abi::FnType for a direct call. Mainly deals with the fact
    /// that a Virtual call doesn't take the vtable, like its shim does.
    /// The extra argument types are for variadic (extern "C") functions.
    pub fn direct_fn_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>,
                              extra_args: &[Ty<'tcx>]) -> FnType {
        let abi = self.ty.fn_abi();
        let sig = ccx.tcx().erase_late_bound_regions_and_normalize(self.ty.fn_sig());
        let mut fn_ty = FnType::unadjusted(ccx, abi, &sig, extra_args);
        if let Virtual(_) = self.data {
            // Don't pass the vtable, it's not an argument of the virtual fn.
            fn_ty.args[1].ignore();
        }
        fn_ty.adjust_for_abi(ccx, abi, &sig);
        fn_ty
    }

    /// Turn the callee into a function pointer.
    pub fn reify<'a>(self, ccx: &CrateContext<'a, 'tcx>) -> ValueRef {
        match self.data {
            Fn(llfn) => llfn,
            Virtual(_) => meth::trans_object_shim(ccx, self),
            NamedTupleConstructor(disr) => match self.ty.sty {
                ty::TyFnDef(def_id, substs, _) => {
                    let instance = Instance::new(def_id, substs);
                    if let Some(&llfn) = ccx.instances().borrow().get(&instance) {
                        return llfn;
                    }

                    let sym = ccx.symbol_map().get_or_compute(ccx.shared(),
                                                              TransItem::Fn(instance));
                    assert!(!ccx.codegen_unit().contains_item(&TransItem::Fn(instance)));
                    let lldecl = declare::define_internal_fn(ccx, &sym, self.ty);
                    base::trans_ctor_shim(ccx, def_id, substs, disr, lldecl);
                    ccx.instances().borrow_mut().insert(instance, lldecl);

                    lldecl
                }
                _ => bug!("expected fn item type, found {}", self.ty)
            },
            Intrinsic => bug!("intrinsic {} getting reified", self.ty)
        }
    }
}

/// Given a DefId and some Substs, produces the monomorphic item type.
fn def_ty<'a, 'tcx>(shared: &SharedCrateContext<'a, 'tcx>,
                    def_id: DefId,
                    substs: &'tcx Substs<'tcx>)
                    -> Ty<'tcx> {
    let ty = shared.tcx().item_type(def_id);
    monomorphize::apply_param_substs(shared, substs, &ty)
}


fn trans_closure_method<'a, 'tcx>(ccx: &'a CrateContext<'a, 'tcx>,
                                  def_id: DefId,
                                  substs: ty::ClosureSubsts<'tcx>,
                                  method_instance: Instance<'tcx>,
                                  trait_closure_kind: ty::ClosureKind)
                                  -> ValueRef
{
    // If this is a closure, redirect to it.
    let (llfn, _) = get_fn(ccx, def_id, substs.substs);

    // If the closure is a Fn closure, but a FnOnce is needed (etc),
    // then adapt the self type
    let llfn_closure_kind = ccx.tcx().closure_kind(def_id);

    debug!("trans_closure_adapter_shim(llfn_closure_kind={:?}, \
           trait_closure_kind={:?}, llfn={:?})",
           llfn_closure_kind, trait_closure_kind, Value(llfn));

    match needs_fn_once_adapter_shim(llfn_closure_kind, trait_closure_kind) {
        Ok(true) => trans_fn_once_adapter_shim(ccx,
                                               def_id,
                                               substs,
                                               method_instance,
                                               llfn),
        Ok(false) => llfn,
        Err(()) => {
            bug!("trans_closure_adapter_shim: cannot convert {:?} to {:?}",
                 llfn_closure_kind,
                 trait_closure_kind);
        }
    }
}

pub fn needs_fn_once_adapter_shim(actual_closure_kind: ty::ClosureKind,
                                  trait_closure_kind: ty::ClosureKind)
                                  -> Result<bool, ()>
{
    match (actual_closure_kind, trait_closure_kind) {
        (ty::ClosureKind::Fn, ty::ClosureKind::Fn) |
        (ty::ClosureKind::FnMut, ty::ClosureKind::FnMut) |
        (ty::ClosureKind::FnOnce, ty::ClosureKind::FnOnce) => {
            // No adapter needed.
           Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnMut) => {
            // The closure fn `llfn` is a `fn(&self, ...)`.  We want a
            // `fn(&mut self, ...)`. In fact, at trans time, these are
            // basically the same thing, so we can just return llfn.
            Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnOnce) |
        (ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
            // The closure fn `llfn` is a `fn(&self, ...)` or `fn(&mut
            // self, ...)`.  We want a `fn(self, ...)`. We can produce
            // this by doing something like:
            //
            //     fn call_once(self, ...) { call_mut(&self, ...) }
            //     fn call_once(mut self, ...) { call_mut(&mut self, ...) }
            //
            // These are both the same at trans time.
            Ok(true)
        }
        _ => Err(()),
    }
}

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
    let ty::ClosureTy { unsafety, abi, mut sig } = tcx.closure_type(def_id, substs);
    sig.0 = tcx.mk_fn_sig(
        iter::once(ref_closure_ty).chain(sig.0.inputs().iter().cloned()),
        sig.0.output(),
        sig.0.variadic
    );
    let llref_fn_ty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
        unsafety: unsafety,
        abi: abi,
        sig: sig.clone()
    }));
    debug!("trans_fn_once_adapter_shim: llref_fn_ty={:?}",
           llref_fn_ty);


    // Make a version of the closure type with the same arguments, but
    // with argument #0 being by value.
    assert_eq!(abi, Abi::RustCall);
    sig.0 = tcx.mk_fn_sig(
        iter::once(closure_ty).chain(sig.0.inputs().iter().skip(1).cloned()),
        sig.0.output(),
        sig.0.variadic
    );

    let sig = tcx.erase_late_bound_regions_and_normalize(&sig);
    let fn_ty = FnType::new(ccx, abi, &sig, &[]);

    let llonce_fn_ty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
        unsafety: unsafety,
        abi: abi,
        sig: ty::Binder(sig)
    }));

    // Create the by-value helper.
    let function_name = method_instance.symbol_name(ccx.shared());
    let lloncefn = declare::define_internal_fn(ccx, &function_name, llonce_fn_ty);
    attributes::set_frame_pointer_elimination(ccx, lloncefn);

    let orig_fn_ty = fn_ty;
    let mut bcx = Builder::new_block(ccx, lloncefn, "entry-block");

    let callee = Callee {
        data: Fn(llreffn),
        ty: llref_fn_ty
    };

    // the first argument (`self`) will be the (by value) closure env.

    let mut llargs = get_params(lloncefn);
    let fn_ret = callee.ty.fn_ret();
    let fn_ty = callee.direct_fn_type(bcx.ccx, &[]);
    let self_idx = fn_ty.ret.is_indirect() as usize;
    let env_arg = &orig_fn_ty.args[0];
    let llenv = if env_arg.is_indirect() {
        llargs[self_idx]
    } else {
        let scratch = bcx.alloca_ty(closure_ty, "self");
        let mut llarg_idx = self_idx;
        env_arg.store_fn_arg(&bcx, &mut llarg_idx, scratch);
        scratch
    };

    debug!("trans_fn_once_adapter_shim: env={:?}", Value(llenv));
    // Adjust llargs such that llargs[self_idx..] has the call arguments.
    // For zero-sized closures that means sneaking in a new argument.
    if env_arg.is_ignore() {
        llargs.insert(self_idx, llenv);
    } else {
        llargs[self_idx] = llenv;
    }

    // Call the by-ref closure body with `self` in a cleanup scope,
    // to drop `self` when the body returns, or in case it unwinds.
    let self_scope = CleanupScope::schedule_drop_mem(
        &bcx, LvalueRef::new_sized_ty(llenv, closure_ty)
    );

    let llfn = callee.reify(bcx.ccx);
    let llret;
    if let Some(landing_pad) = self_scope.landing_pad {
        let normal_bcx = bcx.build_sibling_block("normal-return");
        llret = bcx.invoke(llfn, &llargs[..], normal_bcx.llbb(), landing_pad, None);
        bcx = normal_bcx;
    } else {
        llret = bcx.call(llfn, &llargs[..], None);
    }
    fn_ty.apply_attrs_callsite(llret);

    if fn_ret.0.is_never() {
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

/// Translates an adapter that implements the `Fn` trait for a fn
/// pointer. This is basically the equivalent of something like:
///
/// ```
/// impl<'a> Fn(&'a int) -> &'a int for fn(&int) -> &int {
///     extern "rust-abi" fn call(&self, args: (&'a int,)) -> &'a int {
///         (*self)(args.0)
///     }
/// }
/// ```
///
/// but for the bare function type given.
fn trans_fn_pointer_shim<'a, 'tcx>(
    ccx: &'a CrateContext<'a, 'tcx>,
    method_instance: Instance<'tcx>,
    closure_kind: ty::ClosureKind,
    bare_fn_ty: Ty<'tcx>)
    -> ValueRef
{
    let tcx = ccx.tcx();

    // Normalize the type for better caching.
    let bare_fn_ty = tcx.normalize_associated_type(&bare_fn_ty);

    // If this is an impl of `Fn` or `FnMut` trait, the receiver is `&self`.
    let is_by_ref = match closure_kind {
        ty::ClosureKind::Fn | ty::ClosureKind::FnMut => true,
        ty::ClosureKind::FnOnce => false,
    };

    let llfnpointer = match bare_fn_ty.sty {
        ty::TyFnDef(def_id, substs, _) => {
            // Function definitions have to be turned into a pointer.
            let llfn = Callee::def(ccx, def_id, substs).reify(ccx);
            if !is_by_ref {
                // A by-value fn item is ignored, so the shim has
                // the same signature as the original function.
                return llfn;
            }
            Some(llfn)
        }
        _ => None
    };

    let bare_fn_ty_maybe_ref = if is_by_ref {
        tcx.mk_imm_ref(tcx.mk_region(ty::ReErased), bare_fn_ty)
    } else {
        bare_fn_ty
    };

    // Check if we already trans'd this shim.
    if let Some(&llval) = ccx.fn_pointer_shims().borrow().get(&bare_fn_ty_maybe_ref) {
        return llval;
    }

    debug!("trans_fn_pointer_shim(bare_fn_ty={:?})",
           bare_fn_ty);

    // Construct the "tuply" version of `bare_fn_ty`. It takes two arguments: `self`,
    // which is the fn pointer, and `args`, which is the arguments tuple.
    let sig = match bare_fn_ty.sty {
        ty::TyFnDef(..,
                    &ty::BareFnTy { unsafety: hir::Unsafety::Normal,
                                    abi: Abi::Rust,
                                    ref sig }) |
        ty::TyFnPtr(&ty::BareFnTy { unsafety: hir::Unsafety::Normal,
                                    abi: Abi::Rust,
                                    ref sig }) => sig,

        _ => {
            bug!("trans_fn_pointer_shim invoked on invalid type: {}",
                 bare_fn_ty);
        }
    };
    let sig = tcx.erase_late_bound_regions_and_normalize(sig);
    let tuple_input_ty = tcx.intern_tup(sig.inputs());
    let sig = tcx.mk_fn_sig(
        [bare_fn_ty_maybe_ref, tuple_input_ty].iter().cloned(),
        sig.output(),
        false
    );
    let fn_ty = FnType::new(ccx, Abi::RustCall, &sig, &[]);
    let tuple_fn_ty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
        unsafety: hir::Unsafety::Normal,
        abi: Abi::RustCall,
        sig: ty::Binder(sig)
    }));
    debug!("tuple_fn_ty: {:?}", tuple_fn_ty);

    //
    let function_name = method_instance.symbol_name(ccx.shared());
    let llfn = declare::define_internal_fn(ccx, &function_name, tuple_fn_ty);
    attributes::set_frame_pointer_elimination(ccx, llfn);
    //
    let bcx = Builder::new_block(ccx, llfn, "entry-block");

    let mut llargs = get_params(llfn);

    let self_arg = llargs.remove(fn_ty.ret.is_indirect() as usize);
    let llfnpointer = llfnpointer.unwrap_or_else(|| {
        // the first argument (`self`) will be ptr to the fn pointer
        if is_by_ref {
            bcx.load(self_arg)
        } else {
            self_arg
        }
    });

    let callee = Callee {
        data: Fn(llfnpointer),
        ty: bare_fn_ty
    };
    let fn_ret = callee.ty.fn_ret();
    let fn_ty = callee.direct_fn_type(ccx, &[]);
    let llret = bcx.call(llfnpointer, &llargs, None);
    fn_ty.apply_attrs_callsite(llret);

    if fn_ret.0.is_never() {
        bcx.unreachable();
    } else {
        if fn_ty.ret.is_indirect() || fn_ty.ret.is_ignore() {
            bcx.ret_void();
        } else {
            bcx.ret(llret);
        }
    }

    ccx.fn_pointer_shims().borrow_mut().insert(bare_fn_ty_maybe_ref, llfn);

    llfn
}

/// Translates a reference to a fn/method item, monomorphizing and
/// inlining as it goes.
///
/// # Parameters
///
/// - `ccx`: the crate context
/// - `def_id`: def id of the fn or method item being referenced
/// - `substs`: values for each of the fn/method's parameters
fn get_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                    def_id: DefId,
                    substs: &'tcx Substs<'tcx>)
                    -> (ValueRef, Ty<'tcx>) {
    let tcx = ccx.tcx();

    debug!("get_fn(def_id={:?}, substs={:?})", def_id, substs);

    assert!(!substs.needs_infer());
    assert!(!substs.has_escaping_regions());
    assert!(!substs.has_param_types());

    let substs = tcx.normalize_associated_type(&substs);
    let instance = Instance::new(def_id, substs);
    let item_ty = ccx.tcx().item_type(def_id);
    let fn_ty = monomorphize::apply_param_substs(ccx.shared(), substs, &item_ty);

    if let Some(&llfn) = ccx.instances().borrow().get(&instance) {
        return (llfn, fn_ty);
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
    let fn_ptr_ty = tcx.mk_fn_ptr(tcx.mk_bare_fn(common::ty_fn_ty(ccx, fn_ty).into_owned()));
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

        let attrs = ccx.tcx().get_attrs(def_id);
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
        if ccx.use_dll_storage_attrs() && ccx.sess().cstore.is_dllimport_foreign_item(def_id) {
            unsafe {
                llvm::LLVMSetDLLStorageClass(llfn, llvm::DLLStorageClass::DllImport);
            }
        }
        llfn
    };

    ccx.instances().borrow_mut().insert(instance, llfn);

    (llfn, fn_ty)
}
