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
pub use self::CallArgs::*;

use arena::TypedArena;
use back::symbol_names;
use llvm::{self, ValueRef, get_params};
use middle::cstore::LOCAL_CRATE;
use rustc::hir::def_id::DefId;
use rustc::ty::subst;
use rustc::traits;
use rustc::hir::map as hir_map;
use abi::{Abi, FnType};
use adt;
use attributes;
use base;
use base::*;
use build::*;
use cleanup;
use cleanup::CleanupMethods;
use closure;
use common::{self, Block, Result, CrateContext, FunctionContext};
use common::{C_uint, C_undef};
use consts;
use datum::*;
use debuginfo::DebugLoc;
use declare;
use expr;
use glue;
use inline;
use intrinsic;
use machine::{llalign_of_min, llsize_of_store};
use meth;
use monomorphize::{self, Instance};
use type_::Type;
use type_of;
use value::Value;
use Disr;
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::hir;

use syntax::codemap::DUMMY_SP;
use syntax::errors;
use syntax::ptr::P;

use std::cmp;

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
    pub fn ptr(datum: Datum<'tcx, Rvalue>) -> Callee<'tcx> {
        Callee {
            data: Fn(datum.val),
            ty: datum.ty
        }
    }

    /// Trait or impl method call.
    pub fn method_call<'blk>(bcx: Block<'blk, 'tcx>,
                             method_call: ty::MethodCall)
                             -> Callee<'tcx> {
        let method = bcx.tcx().tables.borrow().method_map[&method_call];
        Callee::method(bcx, method)
    }

    /// Trait or impl method.
    pub fn method<'blk>(bcx: Block<'blk, 'tcx>,
                        method: ty::MethodCallee<'tcx>) -> Callee<'tcx> {
        let substs = bcx.fcx.monomorphize(&method.substs);
        Callee::def(bcx.ccx(), method.def_id, substs)
    }

    /// Function or method definition.
    pub fn def<'a>(ccx: &CrateContext<'a, 'tcx>,
                   def_id: DefId,
                   substs: &'tcx subst::Substs<'tcx>)
                   -> Callee<'tcx> {
        let tcx = ccx.tcx();

        if substs.self_ty().is_some() {
            // Only trait methods can have a Self parameter.
            return Callee::trait_method(ccx, def_id, substs);
        }

        let maybe_node_id = inline::get_local_instance(ccx, def_id)
            .and_then(|def_id| tcx.map.as_local_node_id(def_id));
        let maybe_ast_node = maybe_node_id.and_then(|node_id| {
            tcx.map.find(node_id)
        });

        let data = match maybe_ast_node {
            Some(hir_map::NodeStructCtor(_)) => {
                NamedTupleConstructor(Disr(0))
            }
            Some(hir_map::NodeVariant(_)) => {
                let vinfo = common::inlined_variant_def(ccx, maybe_node_id.unwrap());
                NamedTupleConstructor(Disr::from(vinfo.disr_val))
            }
            Some(hir_map::NodeForeignItem(fi)) if {
                let abi = tcx.map.get_foreign_abi(fi.id);
                abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic
            } => Intrinsic,

            _ => return Callee::ptr(get_fn(ccx, def_id, substs))
        };

        Callee {
            data: data,
            ty: def_ty(tcx, def_id, substs)
        }
    }

    /// Trait method, which has to be resolved to an impl method.
    pub fn trait_method<'a>(ccx: &CrateContext<'a, 'tcx>,
                            def_id: DefId,
                            substs: &'tcx subst::Substs<'tcx>)
                            -> Callee<'tcx> {
        let tcx = ccx.tcx();

        let method_item = tcx.impl_or_trait_item(def_id);
        let trait_id = method_item.container().id();
        let trait_ref = ty::Binder(substs.to_trait_ref(tcx, trait_id));
        let trait_ref = tcx.normalize_associated_type(&trait_ref);
        match common::fulfill_obligation(ccx.shared(), DUMMY_SP, trait_ref) {
            traits::VtableImpl(vtable_impl) => {
                let impl_did = vtable_impl.impl_def_id;
                let mname = tcx.item_name(def_id);
                // create a concatenated set of substitutions which includes
                // those from the impl and those from the method:
                let impl_substs = vtable_impl.substs.with_method_from(&substs);
                let substs = tcx.mk_substs(impl_substs);
                let mth = meth::get_impl_method(tcx, impl_did, substs, mname);

                // Translate the function, bypassing Callee::def.
                // That is because default methods have the same ID as the
                // trait method used to look up the impl method that ended
                // up here, so calling Callee::def would infinitely recurse.
                Callee::ptr(get_fn(ccx, mth.method.def_id, mth.substs))
            }
            traits::VtableClosure(vtable_closure) => {
                // The substitutions should have no type parameters remaining
                // after passing through fulfill_obligation
                let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_id).unwrap();
                let llfn = closure::trans_closure_method(ccx,
                                                         vtable_closure.closure_def_id,
                                                         vtable_closure.substs,
                                                         trait_closure_kind);

                let method_ty = def_ty(tcx, def_id, substs);
                let fn_ptr_ty = match method_ty.sty {
                    ty::TyFnDef(_, _, fty) => tcx.mk_fn_ptr(fty),
                    _ => bug!("expected fn item type, found {}",
                              method_ty)
                };
                Callee::ptr(immediate_rvalue(llfn, fn_ptr_ty))
            }
            traits::VtableFnPointer(fn_ty) => {
                let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_id).unwrap();
                let llfn = trans_fn_pointer_shim(ccx, trait_closure_kind, fn_ty);

                let method_ty = def_ty(tcx, def_id, substs);
                let fn_ptr_ty = match method_ty.sty {
                    ty::TyFnDef(_, _, fty) => tcx.mk_fn_ptr(fty),
                    _ => bug!("expected fn item type, found {}",
                              method_ty)
                };
                Callee::ptr(immediate_rvalue(llfn, fn_ptr_ty))
            }
            traits::VtableObject(ref data) => {
                Callee {
                    data: Virtual(tcx.get_vtable_index_of_object_method(data, def_id)),
                    ty: def_ty(tcx, def_id, substs)
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
        let sig = ccx.tcx().erase_late_bound_regions(self.ty.fn_sig());
        let sig = ccx.tcx().normalize_associated_type(&sig);
        let mut fn_ty = FnType::unadjusted(ccx, abi, &sig, extra_args);
        if let Virtual(_) = self.data {
            // Don't pass the vtable, it's not an argument of the virtual fn.
            fn_ty.args[1].ignore();
        }
        fn_ty.adjust_for_abi(ccx, abi, &sig);
        fn_ty
    }

    /// This behemoth of a function translates function calls. Unfortunately, in
    /// order to generate more efficient LLVM output at -O0, it has quite a complex
    /// signature (refactoring this into two functions seems like a good idea).
    ///
    /// In particular, for lang items, it is invoked with a dest of None, and in
    /// that case the return value contains the result of the fn. The lang item must
    /// not return a structural type or else all heck breaks loose.
    ///
    /// For non-lang items, `dest` is always Some, and hence the result is written
    /// into memory somewhere. Nonetheless we return the actual return value of the
    /// function.
    pub fn call<'a, 'blk>(self, bcx: Block<'blk, 'tcx>,
                          debug_loc: DebugLoc,
                          args: CallArgs<'a, 'tcx>,
                          dest: Option<expr::Dest>)
                          -> Result<'blk, 'tcx> {
        trans_call_inner(bcx, debug_loc, self, args, dest)
    }

    /// Turn the callee into a function pointer.
    pub fn reify<'a>(self, ccx: &CrateContext<'a, 'tcx>)
                     -> Datum<'tcx, Rvalue> {
        let fn_ptr_ty = match self.ty.sty {
            ty::TyFnDef(_, _, f) => ccx.tcx().mk_fn_ptr(f),
            _ => self.ty
        };
        match self.data {
            Fn(llfn) => {
                immediate_rvalue(llfn, fn_ptr_ty)
            }
            Virtual(idx) => {
                let llfn = meth::trans_object_shim(ccx, self.ty, idx);
                immediate_rvalue(llfn, fn_ptr_ty)
            }
            NamedTupleConstructor(_) => match self.ty.sty {
                ty::TyFnDef(def_id, substs, _) => {
                    return get_fn(ccx, def_id, substs);
                }
                _ => bug!("expected fn item type, found {}", self.ty)
            },
            Intrinsic => bug!("intrinsic {} getting reified", self.ty)
        }
    }
}

/// Given a DefId and some Substs, produces the monomorphic item type.
fn def_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    def_id: DefId,
                    substs: &'tcx subst::Substs<'tcx>)
                    -> Ty<'tcx> {
    let ty = tcx.lookup_item_type(def_id).ty;
    monomorphize::apply_param_substs(tcx, substs, &ty)
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
pub fn trans_fn_pointer_shim<'a, 'tcx>(
    ccx: &'a CrateContext<'a, 'tcx>,
    closure_kind: ty::ClosureKind,
    bare_fn_ty: Ty<'tcx>)
    -> ValueRef
{
    let _icx = push_ctxt("trans_fn_pointer_shim");
    let tcx = ccx.tcx();

    // Normalize the type for better caching.
    let bare_fn_ty = tcx.erase_regions(&bare_fn_ty);

    // If this is an impl of `Fn` or `FnMut` trait, the receiver is `&self`.
    let is_by_ref = match closure_kind {
        ty::ClosureKind::Fn | ty::ClosureKind::FnMut => true,
        ty::ClosureKind::FnOnce => false,
    };

    let llfnpointer = match bare_fn_ty.sty {
        ty::TyFnDef(def_id, substs, _) => {
            // Function definitions have to be turned into a pointer.
            let llfn = Callee::def(ccx, def_id, substs).reify(ccx).val;
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
        tcx.mk_imm_ref(tcx.mk_region(ty::ReStatic), bare_fn_ty)
    } else {
        bare_fn_ty
    };

    // Check if we already trans'd this shim.
    match ccx.fn_pointer_shims().borrow().get(&bare_fn_ty_maybe_ref) {
        Some(&llval) => { return llval; }
        None => { }
    }

    debug!("trans_fn_pointer_shim(bare_fn_ty={:?})",
           bare_fn_ty);

    // Construct the "tuply" version of `bare_fn_ty`. It takes two arguments: `self`,
    // which is the fn pointer, and `args`, which is the arguments tuple.
    let sig = match bare_fn_ty.sty {
        ty::TyFnDef(_, _,
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
    let sig = tcx.erase_late_bound_regions(sig);
    let sig = ccx.tcx().normalize_associated_type(&sig);
    let tuple_input_ty = tcx.mk_tup(sig.inputs.to_vec());
    let sig = ty::FnSig {
        inputs: vec![bare_fn_ty_maybe_ref,
                     tuple_input_ty],
        output: sig.output,
        variadic: false
    };
    let fn_ty = FnType::new(ccx, Abi::RustCall, &sig, &[]);
    let tuple_fn_ty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
        unsafety: hir::Unsafety::Normal,
        abi: Abi::RustCall,
        sig: ty::Binder(sig)
    }));
    debug!("tuple_fn_ty: {:?}", tuple_fn_ty);

    //
    let function_name =
        symbol_names::internal_name_from_type_and_suffix(ccx,
                                                         bare_fn_ty,
                                                         "fn_pointer_shim");
    let llfn = declare::define_internal_fn(ccx, &function_name, tuple_fn_ty);

    //
    let (block_arena, fcx): (TypedArena<_>, FunctionContext);
    block_arena = TypedArena::new();
    fcx = FunctionContext::new(ccx, llfn, fn_ty, None, &block_arena);
    let mut bcx = fcx.init(false, None);

    let llargs = get_params(fcx.llfn);

    let self_idx = fcx.fn_ty.ret.is_indirect() as usize;
    let llfnpointer = llfnpointer.unwrap_or_else(|| {
        // the first argument (`self`) will be ptr to the fn pointer
        if is_by_ref {
            Load(bcx, llargs[self_idx])
        } else {
            llargs[self_idx]
        }
    });

    assert!(!fcx.needs_ret_allocas);

    let dest = fcx.llretslotptr.get().map(|_|
        expr::SaveIn(fcx.get_ret_slot(bcx, "ret_slot"))
    );

    let callee = Callee {
        data: Fn(llfnpointer),
        ty: bare_fn_ty
    };
    bcx = callee.call(bcx, DebugLoc::None, ArgVals(&llargs[(self_idx + 1)..]), dest).bcx;

    fcx.finish(bcx, DebugLoc::None);

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
                    substs: &'tcx subst::Substs<'tcx>)
                    -> Datum<'tcx, Rvalue> {
    let tcx = ccx.tcx();

    debug!("get_fn(def_id={:?}, substs={:?})", def_id, substs);

    assert!(!substs.types.needs_infer());
    assert!(!substs.types.has_escaping_regions());

    // Check whether this fn has an inlined copy and, if so, redirect
    // def_id to the local id of the inlined copy.
    let def_id = inline::maybe_instantiate_inline(ccx, def_id);

    fn is_named_tuple_constructor(tcx: TyCtxt, def_id: DefId) -> bool {
        let node_id = match tcx.map.as_local_node_id(def_id) {
            Some(n) => n,
            None => { return false; }
        };
        let map_node = errors::expect(
            &tcx.sess.diagnostic(),
            tcx.map.find(node_id),
            || "local item should be in ast map".to_string());

        match map_node {
            hir_map::NodeVariant(v) => {
                v.node.data.is_tuple()
            }
            hir_map::NodeStructCtor(_) => true,
            _ => false
        }
    }
    let must_monomorphise =
        !substs.types.is_empty() || is_named_tuple_constructor(tcx, def_id);

    debug!("get_fn({:?}) must_monomorphise: {}",
           def_id, must_monomorphise);

    // Create a monomorphic version of generic functions
    if must_monomorphise {
        // Should be either intra-crate or inlined.
        assert_eq!(def_id.krate, LOCAL_CRATE);

        let substs = tcx.mk_substs(substs.clone().erase_regions());
        let (val, fn_ty) = monomorphize::monomorphic_fn(ccx, def_id, substs);
        let fn_ptr_ty = match fn_ty.sty {
            ty::TyFnDef(_, _, fty) => {
                // Create a fn pointer with the substituted signature.
                tcx.mk_fn_ptr(fty)
            }
            _ => bug!("expected fn item type, found {}", fn_ty)
        };
        assert_eq!(type_of::type_of(ccx, fn_ptr_ty), common::val_ty(val));
        return immediate_rvalue(val, fn_ptr_ty);
    }

    // Find the actual function pointer.
    let ty = ccx.tcx().lookup_item_type(def_id).ty;
    let fn_ptr_ty = match ty.sty {
        ty::TyFnDef(_, _, ref fty) => {
            // Create a fn pointer with the normalized signature.
            tcx.mk_fn_ptr(tcx.normalize_associated_type(fty))
        }
        _ => bug!("expected fn item type, found {}", ty)
    };

    let instance = Instance::mono(ccx.tcx(), def_id);
    if let Some(&llfn) = ccx.instances().borrow().get(&instance) {
        return immediate_rvalue(llfn, fn_ptr_ty);
    }

    let local_id = ccx.tcx().map.as_local_node_id(def_id);
    let local_item = match local_id.and_then(|id| tcx.map.find(id)) {
        Some(hir_map::NodeItem(&hir::Item {
            span, node: hir::ItemFn(..), ..
        })) |
        Some(hir_map::NodeTraitItem(&hir::TraitItem {
            span, node: hir::MethodTraitItem(_, Some(_)), ..
        })) |
        Some(hir_map::NodeImplItem(&hir::ImplItem {
            span, node: hir::ImplItemKind::Method(..), ..
        })) => {
            Some(span)
        }
        _ => None
    };

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

    let sym = instance.symbol_name(ccx.shared());
    let llptrty = type_of::type_of(ccx, fn_ptr_ty);
    let llfn = if let Some(llfn) = declare::get_declared_value(ccx, &sym) {
        if let Some(span) = local_item {
            if declare::get_defined_value(ccx, &sym).is_some() {
                ccx.sess().span_fatal(span,
                    &format!("symbol `{}` is already defined", sym));
            }
        }

        if common::val_ty(llfn) != llptrty {
            if local_item.is_some() {
                bug!("symbol `{}` previously declared as {:?}, now wanted as {:?}",
                     sym, Value(llfn), llptrty);
            }
            debug!("get_fn: casting {:?} to {:?}", llfn, llptrty);
            consts::ptrcast(llfn, llptrty)
        } else {
            debug!("get_fn: not casting pointer!");
            llfn
        }
    } else {
        let llfn = declare::declare_fn(ccx, &sym, ty);
        assert_eq!(common::val_ty(llfn), llptrty);
        debug!("get_fn: not casting pointer!");

        let attrs = ccx.tcx().get_attrs(def_id);
        attributes::from_fn_attrs(ccx, &attrs, llfn);
        if local_item.is_some() {
            // FIXME(eddyb) Doubt all extern fn should allow unwinding.
            attributes::unwind(llfn, true);
        }

        llfn
    };

    ccx.instances().borrow_mut().insert(instance, llfn);

    immediate_rvalue(llfn, fn_ptr_ty)
}

// ______________________________________________________________________
// Translating calls

fn trans_call_inner<'a, 'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                    debug_loc: DebugLoc,
                                    callee: Callee<'tcx>,
                                    args: CallArgs<'a, 'tcx>,
                                    dest: Option<expr::Dest>)
                                    -> Result<'blk, 'tcx> {
    // Introduce a temporary cleanup scope that will contain cleanups
    // for the arguments while they are being evaluated. The purpose
    // this cleanup is to ensure that, should a panic occur while
    // evaluating argument N, the values for arguments 0...N-1 are all
    // cleaned up. If no panic occurs, the values are handed off to
    // the callee, and hence none of the cleanups in this temporary
    // scope will ever execute.
    let fcx = bcx.fcx;
    let ccx = fcx.ccx;

    let abi = callee.ty.fn_abi();
    let sig = callee.ty.fn_sig();
    let output = bcx.tcx().erase_late_bound_regions(&sig.output());
    let output = bcx.tcx().normalize_associated_type(&output);

    let extra_args = match args {
        ArgExprs(args) if abi != Abi::RustCall => {
            args[sig.0.inputs.len()..].iter().map(|expr| {
                common::expr_ty_adjusted(bcx, expr)
            }).collect()
        }
        _ => vec![]
    };
    let fn_ty = callee.direct_fn_type(ccx, &extra_args);

    let mut callee = match callee.data {
        Intrinsic => {
            assert!(abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic);
            assert!(dest.is_some());

            return intrinsic::trans_intrinsic_call(bcx, callee.ty, &fn_ty,
                                                   args, dest.unwrap(),
                                                   debug_loc);
        }
        NamedTupleConstructor(disr) => {
            assert!(dest.is_some());

            return base::trans_named_tuple_constructor(bcx,
                                                       callee.ty,
                                                       disr,
                                                       args,
                                                       dest.unwrap(),
                                                       debug_loc);
        }
        f => f
    };

    // Generate a location to store the result. If the user does
    // not care about the result, just make a stack slot.
    let opt_llretslot = dest.and_then(|dest| match dest {
        expr::SaveIn(dst) => Some(dst),
        expr::Ignore => {
            let needs_drop = || match output {
                ty::FnConverging(ret_ty) => bcx.fcx.type_needs_drop(ret_ty),
                ty::FnDiverging => false
            };
            if fn_ty.ret.is_indirect() || fn_ty.ret.cast.is_some() || needs_drop() {
                // Push the out-pointer if we use an out-pointer for this
                // return type, otherwise push "undef".
                if fn_ty.ret.is_ignore() {
                    Some(C_undef(fn_ty.ret.original_ty.ptr_to()))
                } else {
                    let llresult = alloca(bcx, fn_ty.ret.original_ty, "__llret");
                    call_lifetime_start(bcx, llresult);
                    Some(llresult)
                }
            } else {
                None
            }
        }
    });

    // If there no destination, return must be direct, with no cast.
    if opt_llretslot.is_none() {
        assert!(!fn_ty.ret.is_indirect() && fn_ty.ret.cast.is_none());
    }

    let mut llargs = Vec::new();

    if fn_ty.ret.is_indirect() {
        let mut llretslot = opt_llretslot.unwrap();
        if let Some(ty) = fn_ty.ret.cast {
            llretslot = PointerCast(bcx, llretslot, ty.ptr_to());
        }
        llargs.push(llretslot);
    }

    let arg_cleanup_scope = fcx.push_custom_cleanup_scope();
    bcx = trans_args(bcx, abi, &fn_ty, &mut callee, args, &mut llargs,
                     cleanup::CustomScope(arg_cleanup_scope));
    fcx.scopes.borrow_mut().last_mut().unwrap().drop_non_lifetime_clean();

    let llfn = match callee {
        Fn(f) => f,
        _ => bug!("expected fn pointer callee, found {:?}", callee)
    };

    let (llret, mut bcx) = base::invoke(bcx, llfn, &llargs, debug_loc);
    if !bcx.unreachable.get() {
        fn_ty.apply_attrs_callsite(llret);
    }

    // If the function we just called does not use an outpointer,
    // store the result into the rust outpointer. Cast the outpointer
    // type to match because some ABIs will use a different type than
    // the Rust type. e.g., a {u32,u32} struct could be returned as
    // u64.
    if !fn_ty.ret.is_ignore() && !fn_ty.ret.is_indirect() {
        if let Some(llforeign_ret_ty) = fn_ty.ret.cast {
            let llrust_ret_ty = fn_ty.ret.original_ty;
            let llretslot = opt_llretslot.unwrap();

            // The actual return type is a struct, but the ABI
            // adaptation code has cast it into some scalar type.  The
            // code that follows is the only reliable way I have
            // found to do a transform like i64 -> {i32,i32}.
            // Basically we dump the data onto the stack then memcpy it.
            //
            // Other approaches I tried:
            // - Casting rust ret pointer to the foreign type and using Store
            //   is (a) unsafe if size of foreign type > size of rust type and
            //   (b) runs afoul of strict aliasing rules, yielding invalid
            //   assembly under -O (specifically, the store gets removed).
            // - Truncating foreign type to correct integral type and then
            //   bitcasting to the struct type yields invalid cast errors.
            let llscratch = base::alloca(bcx, llforeign_ret_ty, "__cast");
            base::call_lifetime_start(bcx, llscratch);
            Store(bcx, llret, llscratch);
            let llscratch_i8 = PointerCast(bcx, llscratch, Type::i8(ccx).ptr_to());
            let llretptr_i8 = PointerCast(bcx, llretslot, Type::i8(ccx).ptr_to());
            let llrust_size = llsize_of_store(ccx, llrust_ret_ty);
            let llforeign_align = llalign_of_min(ccx, llforeign_ret_ty);
            let llrust_align = llalign_of_min(ccx, llrust_ret_ty);
            let llalign = cmp::min(llforeign_align, llrust_align);
            debug!("llrust_size={}", llrust_size);

            if !bcx.unreachable.get() {
                base::call_memcpy(&B(bcx), llretptr_i8, llscratch_i8,
                                  C_uint(ccx, llrust_size), llalign as u32);
            }
            base::call_lifetime_end(bcx, llscratch);
        } else if let Some(llretslot) = opt_llretslot {
            base::store_ty(bcx, llret, llretslot, output.unwrap());
        }
    }

    fcx.pop_and_trans_custom_cleanup_scope(bcx, arg_cleanup_scope);

    // If the caller doesn't care about the result of this fn call,
    // drop the temporary slot we made.
    match (dest, opt_llretslot, output) {
        (Some(expr::Ignore), Some(llretslot), ty::FnConverging(ret_ty)) => {
            // drop the value if it is not being saved.
            bcx = glue::drop_ty(bcx, llretslot, ret_ty, debug_loc);
            call_lifetime_end(bcx, llretslot);
        }
        _ => {}
    }

    if output == ty::FnDiverging {
        Unreachable(bcx);
    }

    Result::new(bcx, llret)
}

pub enum CallArgs<'a, 'tcx> {
    /// Supply value of arguments as a list of expressions that must be
    /// translated. This is used in the common case of `foo(bar, qux)`.
    ArgExprs(&'a [P<hir::Expr>]),

    /// Supply value of arguments as a list of LLVM value refs; frequently
    /// used with lang items and so forth, when the argument is an internal
    /// value.
    ArgVals(&'a [ValueRef]),

    /// For overloaded operators: `(lhs, Option(rhs))`.
    /// `lhs` is the left-hand-side and `rhs` is the datum
    /// of the right-hand-side argument (if any).
    ArgOverloadedOp(Datum<'tcx, Expr>, Option<Datum<'tcx, Expr>>),

    /// Supply value of arguments as a list of expressions that must be
    /// translated, for overloaded call operators.
    ArgOverloadedCall(Vec<&'a hir::Expr>),
}

fn trans_args_under_call_abi<'blk, 'tcx>(
                             mut bcx: Block<'blk, 'tcx>,
                             arg_exprs: &[P<hir::Expr>],
                             callee: &mut CalleeData,
                             fn_ty: &FnType,
                             llargs: &mut Vec<ValueRef>,
                             arg_cleanup_scope: cleanup::ScopeId)
                             -> Block<'blk, 'tcx>
{
    let mut arg_idx = 0;

    // Translate the `self` argument first.
    let arg_datum = unpack_datum!(bcx, expr::trans(bcx, &arg_exprs[0]));
    bcx = trans_arg_datum(bcx,
                          arg_datum,
                          callee, fn_ty, &mut arg_idx,
                          arg_cleanup_scope,
                          llargs);

    // Now untuple the rest of the arguments.
    let tuple_expr = &arg_exprs[1];
    let tuple_type = common::node_id_type(bcx, tuple_expr.id);

    match tuple_type.sty {
        ty::TyTuple(ref field_types) => {
            let tuple_datum = unpack_datum!(bcx,
                                            expr::trans(bcx, &tuple_expr));
            let tuple_lvalue_datum =
                unpack_datum!(bcx,
                              tuple_datum.to_lvalue_datum(bcx,
                                                          "args",
                                                          tuple_expr.id));
            let repr = adt::represent_type(bcx.ccx(), tuple_type);
            let repr_ptr = &repr;
            for (i, field_type) in field_types.iter().enumerate() {
                let arg_datum = tuple_lvalue_datum.get_element(
                    bcx,
                    field_type,
                    |srcval| {
                        adt::trans_field_ptr(bcx, repr_ptr, srcval, Disr(0), i)
                    }).to_expr_datum();
                bcx = trans_arg_datum(bcx,
                                      arg_datum,
                                      callee, fn_ty, &mut arg_idx,
                                      arg_cleanup_scope,
                                      llargs);
            }
        }
        _ => {
            span_bug!(tuple_expr.span,
                      "argument to `.call()` wasn't a tuple?!")
        }
    };

    bcx
}

pub fn trans_args<'a, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                  abi: Abi,
                                  fn_ty: &FnType,
                                  callee: &mut CalleeData,
                                  args: CallArgs<'a, 'tcx>,
                                  llargs: &mut Vec<ValueRef>,
                                  arg_cleanup_scope: cleanup::ScopeId)
                                  -> Block<'blk, 'tcx> {
    debug!("trans_args(abi={})", abi);

    let _icx = push_ctxt("trans_args");

    let mut bcx = bcx;
    let mut arg_idx = 0;

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    match args {
        ArgExprs(arg_exprs) => {
            if abi == Abi::RustCall {
                // This is only used for direct calls to the `call`,
                // `call_mut` or `call_once` functions.
                return trans_args_under_call_abi(bcx,
                                                 arg_exprs, callee, fn_ty,
                                                 llargs,
                                                 arg_cleanup_scope)
            }

            for arg_expr in arg_exprs {
                let arg_datum = unpack_datum!(bcx, expr::trans(bcx, &arg_expr));
                bcx = trans_arg_datum(bcx,
                                      arg_datum,
                                      callee, fn_ty, &mut arg_idx,
                                      arg_cleanup_scope,
                                      llargs);
            }
        }
        ArgOverloadedCall(arg_exprs) => {
            for expr in arg_exprs {
                let arg_datum =
                    unpack_datum!(bcx, expr::trans(bcx, expr));
                bcx = trans_arg_datum(bcx,
                                      arg_datum,
                                      callee, fn_ty, &mut arg_idx,
                                      arg_cleanup_scope,
                                      llargs);
            }
        }
        ArgOverloadedOp(lhs, rhs) => {
            bcx = trans_arg_datum(bcx, lhs,
                                  callee, fn_ty, &mut arg_idx,
                                  arg_cleanup_scope,
                                  llargs);

            if let Some(rhs) = rhs {
                bcx = trans_arg_datum(bcx, rhs,
                                      callee, fn_ty, &mut arg_idx,
                                      arg_cleanup_scope,
                                      llargs);
            }
        }
        ArgVals(vs) => {
            match *callee {
                Virtual(idx) => {
                    llargs.push(vs[0]);

                    let fn_ptr = meth::get_virtual_method(bcx, vs[1], idx);
                    let llty = fn_ty.llvm_type(bcx.ccx()).ptr_to();
                    *callee = Fn(PointerCast(bcx, fn_ptr, llty));
                    llargs.extend_from_slice(&vs[2..]);
                }
                _ => llargs.extend_from_slice(vs)
            }
        }
    }

    bcx
}

fn trans_arg_datum<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               arg_datum: Datum<'tcx, Expr>,
                               callee: &mut CalleeData,
                               fn_ty: &FnType,
                               next_idx: &mut usize,
                               arg_cleanup_scope: cleanup::ScopeId,
                               llargs: &mut Vec<ValueRef>)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_arg_datum");
    let mut bcx = bcx;

    debug!("trans_arg_datum({:?})", arg_datum);

    let arg = &fn_ty.args[*next_idx];
    *next_idx += 1;

    // Fill padding with undef value, where applicable.
    if let Some(ty) = arg.pad {
        llargs.push(C_undef(ty));
    }

    // Determine whether we want a by-ref datum even if not appropriate.
    let want_by_ref = arg.is_indirect() || arg.cast.is_some();

    let fat_ptr = common::type_is_fat_ptr(bcx.tcx(), arg_datum.ty);
    let (by_ref, val) = if fat_ptr && !bcx.fcx.type_needs_drop(arg_datum.ty) {
        (true, arg_datum.val)
    } else {
        // Make this an rvalue, since we are going to be
        // passing ownership.
        let arg_datum = unpack_datum!(
            bcx, arg_datum.to_rvalue_datum(bcx, "arg"));

        // Now that arg_datum is owned, get it into the appropriate
        // mode (ref vs value).
        let arg_datum = unpack_datum!(bcx, if want_by_ref {
            arg_datum.to_ref_datum(bcx)
        } else {
            arg_datum.to_appropriate_datum(bcx)
        });

        // Technically, ownership of val passes to the callee.
        // However, we must cleanup should we panic before the
        // callee is actually invoked.
        (arg_datum.kind.is_by_ref(),
         arg_datum.add_clean(bcx.fcx, arg_cleanup_scope))
    };

    if arg.is_ignore() {
        return bcx;
    }

    debug!("--- trans_arg_datum passing {:?}", Value(val));

    if fat_ptr {
        // Fat pointers should be passed without any transformations.
        assert!(!arg.is_indirect() && arg.cast.is_none());
        llargs.push(Load(bcx, expr::get_dataptr(bcx, val)));

        let info_arg = &fn_ty.args[*next_idx];
        *next_idx += 1;
        assert!(!info_arg.is_indirect() && info_arg.cast.is_none());
        let info = Load(bcx, expr::get_meta(bcx, val));

        if let Virtual(idx) = *callee {
            // We have to grab the fn pointer from the vtable when
            // handling the first argument, ensure that here.
            assert_eq!(*next_idx, 2);
            assert!(info_arg.is_ignore());
            let fn_ptr = meth::get_virtual_method(bcx, info, idx);
            let llty = fn_ty.llvm_type(bcx.ccx()).ptr_to();
            *callee = Fn(PointerCast(bcx, fn_ptr, llty));
        } else {
            assert!(!info_arg.is_ignore());
            llargs.push(info);
        }
        return bcx;
    }

    let mut val = val;
    if by_ref && !arg.is_indirect() {
        // Have to load the argument, maybe while casting it.
        if arg.original_ty == Type::i1(bcx.ccx()) {
            // We store bools as i8 so we need to truncate to i1.
            val = LoadRangeAssert(bcx, val, 0, 2, llvm::False);
            val = Trunc(bcx, val, arg.original_ty);
        } else if let Some(ty) = arg.cast {
            val = Load(bcx, PointerCast(bcx, val, ty.ptr_to()));
            if !bcx.unreachable.get() {
                let llalign = llalign_of_min(bcx.ccx(), arg.ty);
                unsafe {
                    llvm::LLVMSetAlignment(val, llalign);
                }
            }
        } else {
            val = Load(bcx, val);
        }
    }

    llargs.push(val);
    bcx
}
