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

use arena::TypedArena;
use back::symbol_names;
use llvm::{ValueRef, get_params};
use middle::cstore::LOCAL_CRATE;
use rustc::hir::def_id::DefId;
use rustc::ty::subst::Substs;
use rustc::traits;
use rustc::hir::map as hir_map;
use abi::{Abi, FnType};
use attributes;
use base;
use base::*;
use build::*;
use closure;
use common::{self, Block, Result, CrateContext, FunctionContext};
use consts;
use debuginfo::DebugLoc;
use declare;
use inline;
use meth;
use monomorphize::{self, Instance};
use trans_item::TransItem;
use type_of;
use value::Value;
use Disr;
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::hir;

use syntax_pos::DUMMY_SP;
use errors;

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
                   substs: &'tcx Substs<'tcx>)
                   -> Callee<'tcx> {
        let tcx = ccx.tcx();

        if let Some(trait_id) = tcx.trait_of_item(def_id) {
            return Callee::trait_method(ccx, trait_id, def_id, substs);
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

            _ => {
                let (llfn, ty) = get_fn(ccx, def_id, substs);
                return Callee::ptr(llfn, ty);
            }
        };

        Callee {
            data: data,
            ty: def_ty(tcx, def_id, substs)
        }
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
                let impl_did = vtable_impl.impl_def_id;
                let mname = tcx.item_name(def_id);
                // create a concatenated set of substitutions which includes
                // those from the impl and those from the method:
                let mth = meth::get_impl_method(tcx, substs, impl_did, vtable_impl.substs, mname);

                // Translate the function, bypassing Callee::def.
                // That is because default methods have the same ID as the
                // trait method used to look up the impl method that ended
                // up here, so calling Callee::def would infinitely recurse.
                let (llfn, ty) = get_fn(ccx, mth.method.def_id, mth.substs);
                Callee::ptr(llfn, ty)
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
                Callee::ptr(llfn, fn_ptr_ty)
            }
            traits::VtableFnPointer(vtable_fn_pointer) => {
                let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_id).unwrap();
                let llfn = trans_fn_pointer_shim(ccx, trait_closure_kind, vtable_fn_pointer.fn_ty);

                let method_ty = def_ty(tcx, def_id, substs);
                let fn_ptr_ty = match method_ty.sty {
                    ty::TyFnDef(_, _, fty) => tcx.mk_fn_ptr(fty),
                    _ => bug!("expected fn item type, found {}",
                              method_ty)
                };
                Callee::ptr(llfn, fn_ptr_ty)
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
                          args: &[ValueRef],
                          dest: Option<ValueRef>)
                          -> Result<'blk, 'tcx> {
        trans_call_inner(bcx, debug_loc, self, args, dest)
    }

    /// Turn the callee into a function pointer.
    pub fn reify<'a>(self, ccx: &CrateContext<'a, 'tcx>) -> ValueRef {
        match self.data {
            Fn(llfn) => llfn,
            Virtual(idx) => {
                meth::trans_object_shim(ccx, self.ty, idx)
            }
            NamedTupleConstructor(_) => match self.ty.sty {
                ty::TyFnDef(def_id, substs, _) => {
                    return get_fn(ccx, def_id, substs).0;
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
                    substs: &'tcx Substs<'tcx>)
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
    attributes::set_frame_pointer_elimination(ccx, llfn);
    //
    let (block_arena, fcx): (TypedArena<_>, FunctionContext);
    block_arena = TypedArena::new();
    fcx = FunctionContext::new(ccx, llfn, fn_ty, None, &block_arena);
    let mut bcx = fcx.init(false);

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

    let dest = fcx.llretslotptr.get();

    let callee = Callee {
        data: Fn(llfnpointer),
        ty: bare_fn_ty
    };
    bcx = callee.call(bcx, DebugLoc::None, &llargs[(self_idx + 1)..], dest).bcx;

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
                    substs: &'tcx Substs<'tcx>)
                    -> (ValueRef, Ty<'tcx>) {
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

        let substs = tcx.normalize_associated_type(&substs);
        let (val, fn_ty) = monomorphize::monomorphic_fn(ccx, def_id, substs);
        let fn_ptr_ty = match fn_ty.sty {
            ty::TyFnDef(_, _, fty) => {
                // Create a fn pointer with the substituted signature.
                tcx.mk_fn_ptr(fty)
            }
            _ => bug!("expected fn item type, found {}", fn_ty)
        };
        assert_eq!(type_of::type_of(ccx, fn_ptr_ty), common::val_ty(val));
        return (val, fn_ptr_ty);
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

    let instance = Instance::mono(ccx.shared(), def_id);
    if let Some(&llfn) = ccx.instances().borrow().get(&instance) {
        return (llfn, fn_ptr_ty);
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

    let sym = ccx.symbol_map().get_or_compute(ccx.shared(),
                                              TransItem::Fn(instance));

    let llptrty = type_of::type_of(ccx, fn_ptr_ty);
    let llfn = if let Some(llfn) = declare::get_declared_value(ccx, &sym) {
        if let Some(span) = local_item {
            if declare::get_defined_value(ccx, &sym).is_some() {
                ccx.sess().span_fatal(span,
                    &format!("symbol `{}` is already defined", &sym));
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

    (llfn, fn_ptr_ty)
}

// ______________________________________________________________________
// Translating calls

fn trans_call_inner<'a, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    debug_loc: DebugLoc,
                                    callee: Callee<'tcx>,
                                    args: &[ValueRef],
                                    opt_llretslot: Option<ValueRef>)
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

    let fn_ret = callee.ty.fn_ret();
    let fn_ty = callee.direct_fn_type(ccx, &[]);

    let mut callee = match callee.data {
        NamedTupleConstructor(_) | Intrinsic => {
            bug!("{:?} calls should not go through Callee::call", callee);
        }
        f => f
    };

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

    match callee {
        Virtual(idx) => {
            llargs.push(args[0]);

            let fn_ptr = meth::get_virtual_method(bcx, args[1], idx);
            let llty = fn_ty.llvm_type(bcx.ccx()).ptr_to();
            callee = Fn(PointerCast(bcx, fn_ptr, llty));
            llargs.extend_from_slice(&args[2..]);
        }
        _ => llargs.extend_from_slice(args)
    }

    let llfn = match callee {
        Fn(f) => f,
        _ => bug!("expected fn pointer callee, found {:?}", callee)
    };

    let (llret, bcx) = base::invoke(bcx, llfn, &llargs, debug_loc);
    if !bcx.unreachable.get() {
        fn_ty.apply_attrs_callsite(llret);

        // If the function we just called does not use an outpointer,
        // store the result into the rust outpointer. Cast the outpointer
        // type to match because some ABIs will use a different type than
        // the Rust type. e.g., a {u32,u32} struct could be returned as
        // u64.
        if !fn_ty.ret.is_indirect() {
            if let Some(llretslot) = opt_llretslot {
                fn_ty.ret.store(&bcx.build(), llret, llretslot);
            }
        }
    }

    if fn_ret.0.is_never() {
        Unreachable(bcx);
    }

    Result::new(bcx, llret)
}
