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

pub use self::AutorefArg::*;
pub use self::CalleeData::*;
pub use self::CallArgs::*;

use arena::TypedArena;
use back::link;
use session;
use llvm::{self, ValueRef, get_params};
use middle::cstore::LOCAL_CRATE;
use middle::def;
use middle::def_id::DefId;
use middle::infer;
use middle::subst;
use middle::subst::{Substs};
use rustc::front::map as hir_map;
use trans::adt;
use trans::base;
use trans::base::*;
use trans::build::*;
use trans::callee;
use trans::cleanup;
use trans::cleanup::CleanupMethods;
use trans::common::{self, Block, Result, NodeIdAndSpan, ExprId, CrateContext,
                    ExprOrMethodCall, FunctionContext, MethodCallKey};
use trans::consts;
use trans::datum::*;
use trans::debuginfo::{DebugLoc, ToDebugLoc};
use trans::declare;
use trans::expr;
use trans::glue;
use trans::inline;
use trans::foreign;
use trans::intrinsic;
use trans::meth;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of;
use middle::ty::{self, Ty, HasTypeFlags, RegionEscape};
use middle::ty::MethodCall;
use rustc_front::hir;

use syntax::abi as synabi;
use syntax::ast;
use syntax::ptr::P;

#[derive(Copy, Clone)]
pub struct MethodData {
    pub llfn: ValueRef,
    pub llself: ValueRef,
}

pub enum CalleeData<'tcx> {
    // Constructor for enum variant/tuple-like-struct
    // i.e. Some, Ok
    NamedTupleConstructor(ty::Disr),

    // Represents a (possibly monomorphized) top-level fn item or method
    // item. Note that this is just the fn-ptr and is not a Rust closure
    // value (which is a pair).
    Fn(/* llfn */ ValueRef),

    Intrinsic(ast::NodeId, subst::Substs<'tcx>),

    TraitItem(MethodData)
}

pub struct Callee<'blk, 'tcx: 'blk> {
    pub bcx: Block<'blk, 'tcx>,
    pub data: CalleeData<'tcx>,
    pub ty: Ty<'tcx>
}

fn trans<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, expr: &hir::Expr)
                     -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("trans_callee");
    debug!("callee::trans(expr={:?})", expr);

    // pick out special kinds of expressions that can be called:
    match expr.node {
        hir::ExprPath(..) => {
            return trans_def(bcx, bcx.def(expr.id), expr);
        }
        _ => {}
    }

    // any other expressions are closures:
    return datum_callee(bcx, expr);

    fn datum_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, expr: &hir::Expr)
                                -> Callee<'blk, 'tcx> {
        let DatumBlock { bcx, datum, .. } = expr::trans(bcx, expr);
        match datum.ty.sty {
            ty::TyBareFn(..) => {
                Callee {
                    bcx: bcx,
                    ty: datum.ty,
                    data: Fn(datum.to_llscalarish(bcx))
                }
            }
            _ => {
                bcx.tcx().sess.span_bug(
                    expr.span,
                    &format!("type of callee is neither bare-fn nor closure: {}",
                             datum.ty));
            }
        }
    }

    fn fn_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, datum: Datum<'tcx, Rvalue>)
                             -> Callee<'blk, 'tcx> {
        Callee {
            bcx: bcx,
            data: Fn(datum.val),
            ty: datum.ty
        }
    }

    fn trans_def<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             def: def::Def,
                             ref_expr: &hir::Expr)
                             -> Callee<'blk, 'tcx> {
        debug!("trans_def(def={:?}, ref_expr={:?})", def, ref_expr);
        let expr_ty = common::node_id_type(bcx, ref_expr.id);
        match def {
            def::DefFn(did, _) if {
                let maybe_def_id = inline::get_local_instance(bcx.ccx(), did);
                let maybe_ast_node = maybe_def_id.and_then(|def_id| {
                    let node_id = bcx.tcx().map.as_local_node_id(def_id).unwrap();
                    bcx.tcx().map.find(node_id)
                });
                match maybe_ast_node {
                    Some(hir_map::NodeStructCtor(_)) => true,
                    _ => false
                }
            } => {
                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(0),
                    ty: expr_ty
                }
            }
            def::DefFn(did, _) if match expr_ty.sty {
                ty::TyBareFn(_, ref f) => f.abi == synabi::RustIntrinsic ||
                                          f.abi == synabi::PlatformIntrinsic,
                _ => false
            } => {
                let substs = common::node_id_substs(bcx.ccx(),
                                                    ExprId(ref_expr.id),
                                                    bcx.fcx.param_substs);
                let def_id = inline::maybe_instantiate_inline(bcx.ccx(), did);
                let node_id = bcx.tcx().map.as_local_node_id(def_id).unwrap();
                Callee { bcx: bcx, data: Intrinsic(node_id, substs), ty: expr_ty }
            }
            def::DefFn(did, _) => {
                fn_callee(bcx, trans_fn_ref(bcx.ccx(), did, ExprId(ref_expr.id),
                                            bcx.fcx.param_substs))
            }
            def::DefMethod(meth_did) => {
                let method_item = bcx.tcx().impl_or_trait_item(meth_did);
                let fn_datum = match method_item.container() {
                    ty::ImplContainer(_) => {
                        trans_fn_ref(bcx.ccx(), meth_did,
                                     ExprId(ref_expr.id),
                                     bcx.fcx.param_substs)
                    }
                    ty::TraitContainer(trait_did) => {
                        meth::trans_static_method_callee(bcx.ccx(),
                                                         meth_did,
                                                         trait_did,
                                                         ref_expr.id,
                                                         bcx.fcx.param_substs)
                    }
                };
                fn_callee(bcx, fn_datum)
            }
            def::DefVariant(tid, vid, _) => {
                let vinfo = bcx.tcx().lookup_adt_def(tid).variant_with_id(vid);
                assert_eq!(vinfo.kind(), ty::VariantKind::Tuple);

                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(vinfo.disr_val),
                    ty: expr_ty
                }
            }
            def::DefStruct(_) => {
                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(0),
                    ty: expr_ty
                }
            }
            def::DefStatic(..) |
            def::DefConst(..) |
            def::DefAssociatedConst(..) |
            def::DefLocal(..) |
            def::DefUpvar(..) => {
                datum_callee(bcx, ref_expr)
            }
            def::DefMod(..) | def::DefForeignMod(..) | def::DefTrait(..) |
            def::DefTy(..) | def::DefPrimTy(..) | def::DefAssociatedTy(..) |
            def::DefLabel(..) | def::DefTyParam(..) |
            def::DefSelfTy(..) | def::DefErr => {
                bcx.tcx().sess.span_bug(
                    ref_expr.span,
                    &format!("cannot translate def {:?} \
                             to a callable thing!", def));
            }
        }
    }
}

/// Translates a reference (with id `ref_id`) to the fn/method with id `def_id` into a function
/// pointer. This may require monomorphization or inlining.
pub fn trans_fn_ref<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                              def_id: DefId,
                              node: ExprOrMethodCall,
                              param_substs: &'tcx subst::Substs<'tcx>)
                              -> Datum<'tcx, Rvalue> {
    let _icx = push_ctxt("trans_fn_ref");

    let substs = common::node_id_substs(ccx, node, param_substs);
    debug!("trans_fn_ref(def_id={:?}, node={:?}, substs={:?})",
           def_id,
           node,
           substs);
    trans_fn_ref_with_substs(ccx, def_id, node, param_substs, substs)
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
        ty::FnClosureKind | ty::FnMutClosureKind => true,
        ty::FnOnceClosureKind => false,
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
    let (opt_def_id, sig) =
        match bare_fn_ty.sty {
            ty::TyBareFn(opt_def_id,
                           &ty::BareFnTy { unsafety: hir::Unsafety::Normal,
                                           abi: synabi::Rust,
                                           ref sig }) => {
                (opt_def_id, sig)
            }

            _ => {
                tcx.sess.bug(&format!("trans_fn_pointer_shim invoked on invalid type: {}",
                                      bare_fn_ty));
            }
        };
    let sig = tcx.erase_late_bound_regions(sig);
    let sig = infer::normalize_associated_type(ccx.tcx(), &sig);
    let tuple_input_ty = tcx.mk_tup(sig.inputs.to_vec());
    let tuple_fn_ty = tcx.mk_fn(opt_def_id,
        tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: hir::Unsafety::Normal,
            abi: synabi::RustCall,
            sig: ty::Binder(ty::FnSig {
                inputs: vec![bare_fn_ty_maybe_ref,
                             tuple_input_ty],
                output: sig.output,
                variadic: false
            })}));
    debug!("tuple_fn_ty: {:?}", tuple_fn_ty);

    //
    let function_name = link::mangle_internal_name_by_type_and_seq(ccx, bare_fn_ty,
                                                                   "fn_pointer_shim");
    let llfn = declare::declare_internal_rust_fn(ccx, &function_name[..], tuple_fn_ty);

    //
    let empty_substs = tcx.mk_substs(Substs::trans_empty());
    let (block_arena, fcx): (TypedArena<_>, FunctionContext);
    block_arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx,
                      llfn,
                      ast::DUMMY_NODE_ID,
                      false,
                      sig.output,
                      empty_substs,
                      None,
                      &block_arena);
    let mut bcx = init_function(&fcx, false, sig.output);

    let llargs = get_params(fcx.llfn);

    let self_idx = fcx.arg_offset();
    // the first argument (`self`) will be ptr to the fn pointer
    let llfnpointer = if is_by_ref {
        Load(bcx, llargs[self_idx])
    } else {
        llargs[self_idx]
    };

    assert!(!fcx.needs_ret_allocas);

    let dest = fcx.llretslotptr.get().map(|_|
        expr::SaveIn(fcx.get_ret_slot(bcx, sig.output, "ret_slot"))
    );

    bcx = trans_call_inner(bcx, DebugLoc::None, |bcx, _| {
        Callee {
            bcx: bcx,
            data: Fn(llfnpointer),
            ty: bare_fn_ty
        }
    }, ArgVals(&llargs[(self_idx + 1)..]), dest).bcx;

    finish_fn(&fcx, bcx, sig.output, DebugLoc::None);

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
/// - `node`: node id of the reference to the fn/method, if applicable.
///   This parameter may be zero; but, if so, the resulting value may not
///   have the right type, so it must be cast before being used.
/// - `param_substs`: if the `node` is in a polymorphic function, these
///   are the substitutions required to monomorphize its type
/// - `substs`: values for each of the fn/method's parameters
pub fn trans_fn_ref_with_substs<'a, 'tcx>(
    ccx: &CrateContext<'a, 'tcx>,
    def_id: DefId,
    node: ExprOrMethodCall,
    param_substs: &'tcx subst::Substs<'tcx>,
    substs: subst::Substs<'tcx>)
    -> Datum<'tcx, Rvalue>
{
    let _icx = push_ctxt("trans_fn_ref_with_substs");
    let tcx = ccx.tcx();

    debug!("trans_fn_ref_with_substs(def_id={:?}, node={:?}, \
            param_substs={:?}, substs={:?})",
           def_id,
           node,
           param_substs,
           substs);

    assert!(!substs.types.needs_infer());
    assert!(!substs.types.has_escaping_regions());
    let substs = substs.erase_regions();

    // Check whether this fn has an inlined copy and, if so, redirect
    // def_id to the local id of the inlined copy.
    let def_id = inline::maybe_instantiate_inline(ccx, def_id);

    fn is_named_tuple_constructor(tcx: &ty::ctxt, def_id: DefId) -> bool {
        let node_id = match tcx.map.as_local_node_id(def_id) {
            Some(n) => n,
            None => { return false; }
        };
        let map_node = session::expect(
            &tcx.sess,
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

    debug!("trans_fn_ref_with_substs({:?}) must_monomorphise: {}",
           def_id, must_monomorphise);

    // Create a monomorphic version of generic functions
    if must_monomorphise {
        // Should be either intra-crate or inlined.
        assert_eq!(def_id.krate, LOCAL_CRATE);

        let opt_ref_id = match node {
            ExprId(id) => if id != 0 { Some(id) } else { None },
            MethodCallKey(_) => None,
        };

        let substs = tcx.mk_substs(substs);
        let (val, fn_ty, must_cast) =
            monomorphize::monomorphic_fn(ccx, def_id, substs, opt_ref_id);
        if must_cast && node != ExprId(0) {
            // Monotype of the REFERENCE to the function (type params
            // are subst'd)
            let ref_ty = match node {
                ExprId(id) => tcx.node_id_to_type(id),
                MethodCallKey(method_call) => {
                    tcx.tables.borrow().method_map[&method_call].ty
                }
            };
            let ref_ty = monomorphize::apply_param_substs(tcx,
                                                          param_substs,
                                                          &ref_ty);
            let llptrty = type_of::type_of_fn_from_ty(ccx, ref_ty).ptr_to();
            if llptrty != common::val_ty(val) {
                let val = consts::ptrcast(val, llptrty);
                return Datum::new(val, ref_ty, Rvalue::new(ByValue));
            }
        }
        return Datum::new(val, fn_ty, Rvalue::new(ByValue));
    }

    // Type scheme of the function item (may have type params)
    let fn_type_scheme = tcx.lookup_item_type(def_id);
    let fn_type = infer::normalize_associated_type(tcx, &fn_type_scheme.ty);

    // Find the actual function pointer.
    let mut val = {
        if let Some(node_id) = ccx.tcx().map.as_local_node_id(def_id) {
            // Internal reference.
            get_item_val(ccx, node_id)
        } else {
            // External reference.
            trans_external_path(ccx, def_id, fn_type)
        }
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
    let llty = type_of::type_of_fn_from_ty(ccx, fn_type);
    let llptrty = llty.ptr_to();
    if common::val_ty(val) != llptrty {
        debug!("trans_fn_ref_with_substs(): casting pointer!");
        val = consts::ptrcast(val, llptrty);
    } else {
        debug!("trans_fn_ref_with_substs(): not casting pointer!");
    }

    Datum::new(val, fn_type, Rvalue::new(ByValue))
}

// ______________________________________________________________________
// Translating calls

pub fn trans_call<'a, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                  call_expr: &hir::Expr,
                                  f: &hir::Expr,
                                  args: CallArgs<'a, 'tcx>,
                                  dest: expr::Dest)
                                  -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_call");
    trans_call_inner(bcx,
                     call_expr.debug_loc(),
                     |bcx, _| trans(bcx, f),
                     args,
                     Some(dest)).bcx
}

pub fn trans_method_call<'a, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                         call_expr: &hir::Expr,
                                         rcvr: &hir::Expr,
                                         args: CallArgs<'a, 'tcx>,
                                         dest: expr::Dest)
                                         -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_method_call");
    debug!("trans_method_call(call_expr={:?})", call_expr);
    let method_call = MethodCall::expr(call_expr.id);
    trans_call_inner(
        bcx,
        call_expr.debug_loc(),
        |cx, arg_cleanup_scope| {
            meth::trans_method_callee(cx, method_call, Some(rcvr), arg_cleanup_scope)
        },
        args,
        Some(dest)).bcx
}

pub fn trans_lang_call<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   did: DefId,
                                   args: &[ValueRef],
                                   dest: Option<expr::Dest>,
                                   debug_loc: DebugLoc)
                                   -> Result<'blk, 'tcx> {
    callee::trans_call_inner(bcx, debug_loc, |bcx, _| {
        let datum = trans_fn_ref_with_substs(bcx.ccx(),
                                             did,
                                             ExprId(0),
                                             bcx.fcx.param_substs,
                                             subst::Substs::trans_empty());
        Callee {
            bcx: bcx,
            data: Fn(datum.val),
            ty: datum.ty
        }
    }, ArgVals(args), dest)
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
pub fn trans_call_inner<'a, 'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                           debug_loc: DebugLoc,
                                           get_callee: F,
                                           args: CallArgs<'a, 'tcx>,
                                           dest: Option<expr::Dest>)
                                           -> Result<'blk, 'tcx> where
    F: FnOnce(Block<'blk, 'tcx>, cleanup::ScopeId) -> Callee<'blk, 'tcx>,
{
    // Introduce a temporary cleanup scope that will contain cleanups
    // for the arguments while they are being evaluated. The purpose
    // this cleanup is to ensure that, should a panic occur while
    // evaluating argument N, the values for arguments 0...N-1 are all
    // cleaned up. If no panic occurs, the values are handed off to
    // the callee, and hence none of the cleanups in this temporary
    // scope will ever execute.
    let fcx = bcx.fcx;
    let ccx = fcx.ccx;
    let arg_cleanup_scope = fcx.push_custom_cleanup_scope();

    let callee = get_callee(bcx, cleanup::CustomScope(arg_cleanup_scope));
    let mut bcx = callee.bcx;

    let (abi, ret_ty) = match callee.ty.sty {
        ty::TyBareFn(_, ref f) => {
            let sig = bcx.tcx().erase_late_bound_regions(&f.sig);
            let sig = infer::normalize_associated_type(bcx.tcx(), &sig);
            (f.abi, sig.output)
        }
        _ => panic!("expected bare rust fn or closure in trans_call_inner")
    };

    let (llfn, llself) = match callee.data {
        Fn(llfn) => {
            (llfn, None)
        }
        TraitItem(d) => {
            (d.llfn, Some(d.llself))
        }
        Intrinsic(node, substs) => {
            assert!(abi == synabi::RustIntrinsic || abi == synabi::PlatformIntrinsic);
            assert!(dest.is_some());

            let call_info = match debug_loc {
                DebugLoc::At(id, span) => NodeIdAndSpan { id: id, span: span },
                DebugLoc::None => {
                    bcx.sess().bug("No call info for intrinsic call?")
                }
            };

            return intrinsic::trans_intrinsic_call(bcx, node, callee.ty,
                                                   arg_cleanup_scope, args,
                                                   dest.unwrap(), substs,
                                                   call_info);
        }
        NamedTupleConstructor(disr) => {
            assert!(dest.is_some());
            fcx.pop_custom_cleanup_scope(arg_cleanup_scope);

            return base::trans_named_tuple_constructor(bcx,
                                                       callee.ty,
                                                       disr,
                                                       args,
                                                       dest.unwrap(),
                                                       debug_loc);
        }
    };

    // Intrinsics should not become actual functions.
    // We trans them in place in `trans_intrinsic_call`
    assert!(abi != synabi::RustIntrinsic && abi != synabi::PlatformIntrinsic);

    let is_rust_fn = abi == synabi::Rust || abi == synabi::RustCall;

    // Generate a location to store the result. If the user does
    // not care about the result, just make a stack slot.
    let opt_llretslot = dest.and_then(|dest| match dest {
        expr::SaveIn(dst) => Some(dst),
        expr::Ignore => {
            let ret_ty = match ret_ty {
                ty::FnConverging(ret_ty) => ret_ty,
                ty::FnDiverging => ccx.tcx().mk_nil()
            };
            if !is_rust_fn ||
              type_of::return_uses_outptr(ccx, ret_ty) ||
              bcx.fcx.type_needs_drop(ret_ty) {
                // Push the out-pointer if we use an out-pointer for this
                // return type, otherwise push "undef".
                if common::type_is_zero_size(ccx, ret_ty) {
                    let llty = type_of::type_of(ccx, ret_ty);
                    Some(common::C_undef(llty.ptr_to()))
                } else {
                    let llresult = alloc_ty(bcx, ret_ty, "__llret");
                    call_lifetime_start(bcx, llresult);
                    Some(llresult)
                }
            } else {
                None
            }
        }
    });

    let mut llresult = unsafe {
        llvm::LLVMGetUndef(Type::nil(ccx).ptr_to().to_ref())
    };

    // The code below invokes the function, using either the Rust
    // conventions (if it is a rust fn) or the native conventions
    // (otherwise).  The important part is that, when all is said
    // and done, either the return value of the function will have been
    // written in opt_llretslot (if it is Some) or `llresult` will be
    // set appropriately (otherwise).
    if is_rust_fn {
        let mut llargs = Vec::new();

        if let (ty::FnConverging(ret_ty), Some(mut llretslot)) = (ret_ty, opt_llretslot) {
            if type_of::return_uses_outptr(ccx, ret_ty) {
                let llformal_ret_ty = type_of::type_of(ccx, ret_ty).ptr_to();
                let llret_ty = common::val_ty(llretslot);
                if llformal_ret_ty != llret_ty {
                    // this could happen due to e.g. subtyping
                    debug!("casting actual return type ({}) to match formal ({})",
                        bcx.llty_str(llret_ty), bcx.llty_str(llformal_ret_ty));
                    llretslot = PointerCast(bcx, llretslot, llformal_ret_ty);
                }
                llargs.push(llretslot);
            }
        }

        // Push a trait object's self.
        if let Some(llself) = llself {
            llargs.push(llself);
        }

        // Push the arguments.
        bcx = trans_args(bcx,
                         args,
                         callee.ty,
                         &mut llargs,
                         cleanup::CustomScope(arg_cleanup_scope),
                         llself.is_some(),
                         abi);

        fcx.scopes.borrow_mut().last_mut().unwrap().drop_non_lifetime_clean();

        // Invoke the actual rust fn and update bcx/llresult.
        let (llret, b) = base::invoke(bcx,
                                      llfn,
                                      &llargs[..],
                                      callee.ty,
                                      debug_loc);
        bcx = b;
        llresult = llret;

        // If the Rust convention for this type is return via
        // the return value, copy it into llretslot.
        match (opt_llretslot, ret_ty) {
            (Some(llretslot), ty::FnConverging(ret_ty)) => {
                if !type_of::return_uses_outptr(bcx.ccx(), ret_ty) &&
                    !common::type_is_zero_size(bcx.ccx(), ret_ty)
                {
                    store_ty(bcx, llret, llretslot, ret_ty)
                }
            }
            (_, _) => {}
        }
    } else {
        // Lang items are the only case where dest is None, and
        // they are always Rust fns.
        assert!(dest.is_some());

        let mut llargs = Vec::new();
        let arg_tys = match args {
            ArgExprs(a) => a.iter().map(|x| common::expr_ty_adjusted(bcx, &**x)).collect(),
            _ => panic!("expected arg exprs.")
        };
        bcx = trans_args(bcx,
                         args,
                         callee.ty,
                         &mut llargs,
                         cleanup::CustomScope(arg_cleanup_scope),
                         false,
                         abi);
        fcx.scopes.borrow_mut().last_mut().unwrap().drop_non_lifetime_clean();

        bcx = foreign::trans_native_call(bcx,
                                         callee.ty,
                                         llfn,
                                         opt_llretslot.unwrap(),
                                         &llargs[..],
                                         arg_tys,
                                         debug_loc);
    }

    fcx.pop_and_trans_custom_cleanup_scope(bcx, arg_cleanup_scope);

    // If the caller doesn't care about the result of this fn call,
    // drop the temporary slot we made.
    match (dest, opt_llretslot, ret_ty) {
        (Some(expr::Ignore), Some(llretslot), ty::FnConverging(ret_ty)) => {
            // drop the value if it is not being saved.
            bcx = glue::drop_ty(bcx,
                                llretslot,
                                ret_ty,
                                debug_loc);
            call_lifetime_end(bcx, llretslot);
        }
        _ => {}
    }

    if ret_ty == ty::FnDiverging {
        Unreachable(bcx);
    }

    Result::new(bcx, llresult)
}

pub enum CallArgs<'a, 'tcx> {
    // Supply value of arguments as a list of expressions that must be
    // translated. This is used in the common case of `foo(bar, qux)`.
    ArgExprs(&'a [P<hir::Expr>]),

    // Supply value of arguments as a list of LLVM value refs; frequently
    // used with lang items and so forth, when the argument is an internal
    // value.
    ArgVals(&'a [ValueRef]),

    // For overloaded operators: `(lhs, Option(rhs, rhs_id), autoref)`. `lhs`
    // is the left-hand-side and `rhs/rhs_id` is the datum/expr-id of
    // the right-hand-side argument (if any). `autoref` indicates whether the `rhs`
    // arguments should be auto-referenced
    ArgOverloadedOp(Datum<'tcx, Expr>, Option<(Datum<'tcx, Expr>, ast::NodeId)>, bool),

    // Supply value of arguments as a list of expressions that must be
    // translated, for overloaded call operators.
    ArgOverloadedCall(Vec<&'a hir::Expr>),
}

fn trans_args_under_call_abi<'blk, 'tcx>(
                             mut bcx: Block<'blk, 'tcx>,
                             arg_exprs: &[P<hir::Expr>],
                             fn_ty: Ty<'tcx>,
                             llargs: &mut Vec<ValueRef>,
                             arg_cleanup_scope: cleanup::ScopeId,
                             ignore_self: bool)
                             -> Block<'blk, 'tcx>
{
    let sig = bcx.tcx().erase_late_bound_regions(&fn_ty.fn_sig());
    let sig = infer::normalize_associated_type(bcx.tcx(), &sig);
    let args = sig.inputs;

    // Translate the `self` argument first.
    if !ignore_self {
        let arg_datum = unpack_datum!(bcx, expr::trans(bcx, &*arg_exprs[0]));
        bcx = trans_arg_datum(bcx,
                              args[0],
                              arg_datum,
                              arg_cleanup_scope,
                              DontAutorefArg,
                              llargs);
    }

    // Now untuple the rest of the arguments.
    let tuple_expr = &arg_exprs[1];
    let tuple_type = common::node_id_type(bcx, tuple_expr.id);

    match tuple_type.sty {
        ty::TyTuple(ref field_types) => {
            let tuple_datum = unpack_datum!(bcx,
                                            expr::trans(bcx, &**tuple_expr));
            let tuple_lvalue_datum =
                unpack_datum!(bcx,
                              tuple_datum.to_lvalue_datum(bcx,
                                                          "args",
                                                          tuple_expr.id));
            let repr = adt::represent_type(bcx.ccx(), tuple_type);
            let repr_ptr = &*repr;
            for (i, field_type) in field_types.iter().enumerate() {
                let arg_datum = tuple_lvalue_datum.get_element(
                    bcx,
                    field_type,
                    |srcval| {
                        adt::trans_field_ptr(bcx, repr_ptr, srcval, 0, i)
                    }).to_expr_datum();
                bcx = trans_arg_datum(bcx,
                                      field_type,
                                      arg_datum,
                                      arg_cleanup_scope,
                                      DontAutorefArg,
                                      llargs);
            }
        }
        _ => {
            bcx.sess().span_bug(tuple_expr.span,
                                "argument to `.call()` wasn't a tuple?!")
        }
    };

    bcx
}

fn trans_overloaded_call_args<'blk, 'tcx>(
                              mut bcx: Block<'blk, 'tcx>,
                              arg_exprs: Vec<&hir::Expr>,
                              fn_ty: Ty<'tcx>,
                              llargs: &mut Vec<ValueRef>,
                              arg_cleanup_scope: cleanup::ScopeId,
                              ignore_self: bool)
                              -> Block<'blk, 'tcx> {
    // Translate the `self` argument first.
    let sig = bcx.tcx().erase_late_bound_regions(&fn_ty.fn_sig());
    let sig = infer::normalize_associated_type(bcx.tcx(), &sig);
    let arg_tys = sig.inputs;

    if !ignore_self {
        let arg_datum = unpack_datum!(bcx, expr::trans(bcx, arg_exprs[0]));
        bcx = trans_arg_datum(bcx,
                              arg_tys[0],
                              arg_datum,
                              arg_cleanup_scope,
                              DontAutorefArg,
                              llargs);
    }

    // Now untuple the rest of the arguments.
    let tuple_type = arg_tys[1];
    match tuple_type.sty {
        ty::TyTuple(ref field_types) => {
            for (i, &field_type) in field_types.iter().enumerate() {
                let arg_datum =
                    unpack_datum!(bcx, expr::trans(bcx, arg_exprs[i + 1]));
                bcx = trans_arg_datum(bcx,
                                      field_type,
                                      arg_datum,
                                      arg_cleanup_scope,
                                      DontAutorefArg,
                                      llargs);
            }
        }
        _ => {
            bcx.sess().span_bug(arg_exprs[0].span,
                                "argument to `.call()` wasn't a tuple?!")
        }
    };

    bcx
}

pub fn trans_args<'a, 'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                                  args: CallArgs<'a, 'tcx>,
                                  fn_ty: Ty<'tcx>,
                                  llargs: &mut Vec<ValueRef>,
                                  arg_cleanup_scope: cleanup::ScopeId,
                                  ignore_self: bool,
                                  abi: synabi::Abi)
                                  -> Block<'blk, 'tcx> {
    debug!("trans_args(abi={})", abi);

    let _icx = push_ctxt("trans_args");
    let sig = cx.tcx().erase_late_bound_regions(&fn_ty.fn_sig());
    let sig = infer::normalize_associated_type(cx.tcx(), &sig);
    let arg_tys = sig.inputs;
    let variadic = sig.variadic;

    let mut bcx = cx;

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    match args {
        ArgExprs(arg_exprs) => {
            if abi == synabi::RustCall {
                // This is only used for direct calls to the `call`,
                // `call_mut` or `call_once` functions.
                return trans_args_under_call_abi(cx,
                                                 arg_exprs,
                                                 fn_ty,
                                                 llargs,
                                                 arg_cleanup_scope,
                                                 ignore_self)
            }

            let num_formal_args = arg_tys.len();
            for (i, arg_expr) in arg_exprs.iter().enumerate() {
                if i == 0 && ignore_self {
                    continue;
                }
                let arg_ty = if i >= num_formal_args {
                    assert!(variadic);
                    common::expr_ty_adjusted(cx, &**arg_expr)
                } else {
                    arg_tys[i]
                };

                let arg_datum = unpack_datum!(bcx, expr::trans(bcx, &**arg_expr));
                bcx = trans_arg_datum(bcx, arg_ty, arg_datum,
                                      arg_cleanup_scope,
                                      DontAutorefArg,
                                      llargs);
            }
        }
        ArgOverloadedCall(arg_exprs) => {
            return trans_overloaded_call_args(cx,
                                              arg_exprs,
                                              fn_ty,
                                              llargs,
                                              arg_cleanup_scope,
                                              ignore_self)
        }
        ArgOverloadedOp(lhs, rhs, autoref) => {
            assert!(!variadic);

            bcx = trans_arg_datum(bcx, arg_tys[0], lhs,
                                  arg_cleanup_scope,
                                  DontAutorefArg,
                                  llargs);

            if let Some((rhs, rhs_id)) = rhs {
                assert_eq!(arg_tys.len(), 2);
                bcx = trans_arg_datum(bcx, arg_tys[1], rhs,
                                      arg_cleanup_scope,
                                      if autoref { DoAutorefArg(rhs_id) } else { DontAutorefArg },
                                      llargs);
            } else {
                assert_eq!(arg_tys.len(), 1);
            }
        }
        ArgVals(vs) => {
            llargs.extend_from_slice(vs);
        }
    }

    bcx
}

#[derive(Copy, Clone)]
pub enum AutorefArg {
    DontAutorefArg,
    DoAutorefArg(ast::NodeId)
}

pub fn trans_arg_datum<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   formal_arg_ty: Ty<'tcx>,
                                   arg_datum: Datum<'tcx, Expr>,
                                   arg_cleanup_scope: cleanup::ScopeId,
                                   autoref_arg: AutorefArg,
                                   llargs: &mut Vec<ValueRef>)
                                   -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_arg_datum");
    let mut bcx = bcx;
    let ccx = bcx.ccx();

    debug!("trans_arg_datum({:?})",
           formal_arg_ty);

    let arg_datum_ty = arg_datum.ty;

    debug!("   arg datum: {}", arg_datum.to_string(bcx.ccx()));

    let mut val;
    // FIXME(#3548) use the adjustments table
    match autoref_arg {
        DoAutorefArg(arg_id) => {
            // We will pass argument by reference
            // We want an lvalue, so that we can pass by reference and
            let arg_datum = unpack_datum!(
                bcx, arg_datum.to_lvalue_datum(bcx, "arg", arg_id));
            val = arg_datum.val;
        }
        DontAutorefArg if common::type_is_fat_ptr(bcx.tcx(), arg_datum_ty) &&
                !bcx.fcx.type_needs_drop(arg_datum_ty) => {
            val = arg_datum.val
        }
        DontAutorefArg => {
            // Make this an rvalue, since we are going to be
            // passing ownership.
            let arg_datum = unpack_datum!(
                bcx, arg_datum.to_rvalue_datum(bcx, "arg"));

            // Now that arg_datum is owned, get it into the appropriate
            // mode (ref vs value).
            let arg_datum = unpack_datum!(
                bcx, arg_datum.to_appropriate_datum(bcx));

            // Technically, ownership of val passes to the callee.
            // However, we must cleanup should we panic before the
            // callee is actually invoked.
            val = arg_datum.add_clean(bcx.fcx, arg_cleanup_scope);
        }
    }

    if type_of::arg_is_indirect(ccx, formal_arg_ty) && formal_arg_ty != arg_datum_ty {
        // this could happen due to e.g. subtyping
        let llformal_arg_ty = type_of::type_of_explicit_arg(ccx, formal_arg_ty);
        debug!("casting actual type ({}) to match formal ({})",
               bcx.val_to_string(val), bcx.llty_str(llformal_arg_ty));
        debug!("Rust types: {:?}; {:?}", arg_datum_ty,
                                     formal_arg_ty);
        val = PointerCast(bcx, val, llformal_arg_ty);
    }

    debug!("--- trans_arg_datum passing {}", bcx.val_to_string(val));

    if common::type_is_fat_ptr(bcx.tcx(), formal_arg_ty) {
        llargs.push(Load(bcx, expr::get_dataptr(bcx, val)));
        llargs.push(Load(bcx, expr::get_meta(bcx, val)));
    } else {
        llargs.push(val);
    }

    bcx
}
