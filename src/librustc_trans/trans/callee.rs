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
use llvm::{ValueRef};
use llvm::get_param;
use llvm;
use metadata::csearch;
use middle::def;
use middle::subst;
use middle::subst::{Subst, Substs};
use trans::adt;
use trans::base;
use trans::base::*;
use trans::build::*;
use trans::callee;
use trans::cleanup;
use trans::cleanup::CleanupMethods;
use trans::closure;
use trans::common::{self, Block, Result, NodeIdAndSpan, ExprId, CrateContext,
                    ExprOrMethodCall, FunctionContext, MethodCallKey};
use trans::consts;
use trans::datum::*;
use trans::debuginfo::{DebugLoc, ToDebugLoc};
use trans::expr;
use trans::glue;
use trans::inline;
use trans::foreign;
use trans::intrinsic;
use trans::meth;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of;
use middle::ty::{self, Ty};
use middle::ty::MethodCall;
use util::ppaux::Repr;
use util::ppaux::ty_to_string;

use syntax::abi as synabi;
use syntax::ast;
use syntax::ast_map;
use syntax::ptr::P;

#[derive(Copy)]
pub struct MethodData {
    pub llfn: ValueRef,
    pub llself: ValueRef,
}

pub enum CalleeData<'tcx> {
    // Constructor for enum variant/tuple-like-struct
    // i.e. Some, Ok
    NamedTupleConstructor(subst::Substs<'tcx>, ty::Disr),

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
}

fn trans<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, expr: &ast::Expr)
                     -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("trans_callee");
    debug!("callee::trans(expr={})", expr.repr(bcx.tcx()));

    // pick out special kinds of expressions that can be called:
    match expr.node {
        ast::ExprPath(_) | ast::ExprQPath(_) => {
            return trans_def(bcx, bcx.def(expr.id), expr);
        }
        _ => {}
    }

    // any other expressions are closures:
    return datum_callee(bcx, expr);

    fn datum_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, expr: &ast::Expr)
                                -> Callee<'blk, 'tcx> {
        let DatumBlock { bcx, datum, .. } = expr::trans(bcx, expr);
        match datum.ty.sty {
            ty::ty_bare_fn(..) => {
                let llval = datum.to_llscalarish(bcx);
                return Callee {
                    bcx: bcx,
                    data: Fn(llval),
                };
            }
            _ => {
                bcx.tcx().sess.span_bug(
                    expr.span,
                    &format!("type of callee is neither bare-fn nor closure: \
                             {}",
                            bcx.ty_to_string(datum.ty))[]);
            }
        }
    }

    fn fn_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, llfn: ValueRef)
                             -> Callee<'blk, 'tcx> {
        return Callee {
            bcx: bcx,
            data: Fn(llfn),
        };
    }

    fn trans_def<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             def: def::Def,
                             ref_expr: &ast::Expr)
                             -> Callee<'blk, 'tcx> {
        debug!("trans_def(def={}, ref_expr={})", def.repr(bcx.tcx()), ref_expr.repr(bcx.tcx()));
        let expr_ty = common::node_id_type(bcx, ref_expr.id);
        match def {
            def::DefFn(did, _) if {
                let maybe_def_id = inline::get_local_instance(bcx.ccx(), did);
                let maybe_ast_node = maybe_def_id.and_then(|def_id| bcx.tcx().map
                                                                             .find(def_id.node));
                match maybe_ast_node {
                    Some(ast_map::NodeStructCtor(_)) => true,
                    _ => false
                }
            } => {
                let substs = common::node_id_substs(bcx.ccx(),
                                                    ExprId(ref_expr.id),
                                                    bcx.fcx.param_substs);
                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(substs, 0)
                }
            }
            def::DefFn(did, _) if match expr_ty.sty {
                ty::ty_bare_fn(_, ref f) => f.abi == synabi::RustIntrinsic,
                _ => false
            } => {
                let substs = common::node_id_substs(bcx.ccx(),
                                                    ExprId(ref_expr.id),
                                                    bcx.fcx.param_substs);
                let def_id = inline::maybe_instantiate_inline(bcx.ccx(), did);
                Callee { bcx: bcx, data: Intrinsic(def_id.node, substs) }
            }
            def::DefFn(did, _) | def::DefMethod(did, _, def::FromImpl(_)) |
            def::DefStaticMethod(did, def::FromImpl(_)) => {
                fn_callee(bcx, trans_fn_ref(bcx.ccx(), did, ExprId(ref_expr.id),
                                            bcx.fcx.param_substs).val)
            }
            def::DefStaticMethod(meth_did, def::FromTrait(trait_did)) |
            def::DefMethod(meth_did, _, def::FromTrait(trait_did)) => {
                fn_callee(bcx, meth::trans_static_method_callee(bcx.ccx(),
                                                                meth_did,
                                                                trait_did,
                                                                ref_expr.id,
                                                                bcx.fcx.param_substs).val)
            }
            def::DefVariant(tid, vid, _) => {
                let vinfo = ty::enum_variant_with_id(bcx.tcx(), tid, vid);
                let substs = common::node_id_substs(bcx.ccx(),
                                                    ExprId(ref_expr.id),
                                                    bcx.fcx.param_substs);

                // Nullary variants are not callable
                assert!(vinfo.args.len() > 0);

                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(substs, vinfo.disr_val)
                }
            }
            def::DefStruct(_) => {
                let substs = common::node_id_substs(bcx.ccx(),
                                                    ExprId(ref_expr.id),
                                                    bcx.fcx.param_substs);
                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(substs, 0)
                }
            }
            def::DefStatic(..) |
            def::DefConst(..) |
            def::DefLocal(..) |
            def::DefUpvar(..) => {
                datum_callee(bcx, ref_expr)
            }
            def::DefMod(..) | def::DefForeignMod(..) | def::DefTrait(..) |
            def::DefTy(..) | def::DefPrimTy(..) | def::DefAssociatedTy(..) |
            def::DefUse(..) | def::DefTyParamBinder(..) |
            def::DefRegion(..) | def::DefLabel(..) | def::DefTyParam(..) |
            def::DefSelfTy(..) | def::DefAssociatedPath(..) => {
                bcx.tcx().sess.span_bug(
                    ref_expr.span,
                    &format!("cannot translate def {:?} \
                             to a callable thing!", def)[]);
            }
        }
    }
}

/// Translates a reference (with id `ref_id`) to the fn/method with id `def_id` into a function
/// pointer. This may require monomorphization or inlining.
pub fn trans_fn_ref<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                              def_id: ast::DefId,
                              node: ExprOrMethodCall,
                              param_substs: &'tcx subst::Substs<'tcx>)
                              -> Datum<'tcx, Rvalue> {
    let _icx = push_ctxt("trans_fn_ref");

    let substs = common::node_id_substs(ccx, node, param_substs);
    debug!("trans_fn_ref(def_id={}, node={:?}, substs={})",
           def_id.repr(ccx.tcx()),
           node,
           substs.repr(ccx.tcx()));
    trans_fn_ref_with_substs(ccx, def_id, node, param_substs, substs)
}

fn trans_fn_ref_with_substs_to_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                  def_id: ast::DefId,
                                                  ref_id: ast::NodeId,
                                                  substs: subst::Substs<'tcx>)
                                                  -> Callee<'blk, 'tcx> {
    Callee {
        bcx: bcx,
        data: Fn(trans_fn_ref_with_substs(bcx.ccx(),
                                          def_id,
                                          ExprId(ref_id),
                                          bcx.fcx.param_substs,
                                          substs).val),
    }
}

/// Translates an adapter that implements the `Fn` trait for a fn
/// pointer. This is basically the equivalent of something like:
///
/// ```rust
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
    bare_fn_ty: Ty<'tcx>)
    -> ValueRef
{
    let _icx = push_ctxt("trans_fn_pointer_shim");
    let tcx = ccx.tcx();

    let bare_fn_ty = common::erase_regions(tcx, &bare_fn_ty);
    match ccx.fn_pointer_shims().borrow().get(&bare_fn_ty) {
        Some(&llval) => { return llval; }
        None => { }
    }

    debug!("trans_fn_pointer_shim(bare_fn_ty={})",
           bare_fn_ty.repr(tcx));

    // This is an impl of `Fn` trait, so receiver is `&self`.
    let bare_fn_ty_ref = ty::mk_imm_rptr(tcx, tcx.mk_region(ty::ReStatic), bare_fn_ty);

    // Construct the "tuply" version of `bare_fn_ty`. It takes two arguments: `self`,
    // which is the fn pointer, and `args`, which is the arguments tuple.
    let (opt_def_id, sig) =
        match bare_fn_ty.sty {
            ty::ty_bare_fn(opt_def_id,
                           &ty::BareFnTy { unsafety: ast::Unsafety::Normal,
                                           abi: synabi::Rust,
                                           ref sig }) => {
                (opt_def_id, sig)
            }

            _ => {
                tcx.sess.bug(&format!("trans_fn_pointer_shim invoked on invalid type: {}",
                                           bare_fn_ty.repr(tcx))[]);
            }
        };
    let sig = ty::erase_late_bound_regions(tcx, sig);
    let tuple_input_ty = ty::mk_tup(tcx, sig.inputs.to_vec());
    let tuple_fn_ty = ty::mk_bare_fn(tcx,
                                     opt_def_id,
                                     tcx.mk_bare_fn(ty::BareFnTy {
                                         unsafety: ast::Unsafety::Normal,
                                         abi: synabi::RustCall,
                                         sig: ty::Binder(ty::FnSig {
                                             inputs: vec![bare_fn_ty_ref,
                                                          tuple_input_ty],
                                             output: sig.output,
                                             variadic: false
                                         })}));
    debug!("tuple_fn_ty: {}", tuple_fn_ty.repr(tcx));

    //
    let function_name =
        link::mangle_internal_name_by_type_and_seq(ccx, bare_fn_ty,
                                                   "fn_pointer_shim");
    let llfn =
        decl_internal_rust_fn(ccx,
                              tuple_fn_ty,
                              &function_name[..]);

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

    // the first argument (`self`) will be ptr to the the fn pointer
    let llfnpointer =
        Load(bcx, get_param(fcx.llfn, fcx.arg_pos(0) as u32));

    // the remaining arguments will be the untupled values
    let llargs: Vec<_> =
        sig.inputs.iter()
        .enumerate()
        .map(|(i, _)| get_param(fcx.llfn, fcx.arg_pos(i+1) as u32))
        .collect();
    assert!(!fcx.needs_ret_allocas);

    let dest = fcx.llretslotptr.get().map(|_|
        expr::SaveIn(fcx.get_ret_slot(bcx, sig.output, "ret_slot"))
    );

    bcx = trans_call_inner(bcx,
                           DebugLoc::None,
                           bare_fn_ty,
                           |bcx, _| Callee { bcx: bcx, data: Fn(llfnpointer) },
                           ArgVals(&llargs[..]),
                           dest).bcx;

    finish_fn(&fcx, bcx, sig.output, DebugLoc::None);

    ccx.fn_pointer_shims().borrow_mut().insert(bare_fn_ty, llfn);

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
    def_id: ast::DefId,
    node: ExprOrMethodCall,
    param_substs: &'tcx subst::Substs<'tcx>,
    substs: subst::Substs<'tcx>)
    -> Datum<'tcx, Rvalue>
{
    let _icx = push_ctxt("trans_fn_ref_with_substs");
    let tcx = ccx.tcx();

    debug!("trans_fn_ref_with_substs(def_id={}, node={:?}, \
            param_substs={}, substs={})",
           def_id.repr(tcx),
           node,
           param_substs.repr(tcx),
           substs.repr(tcx));

    assert!(substs.types.all(|t| !ty::type_needs_infer(*t)));
    assert!(substs.types.all(|t| !ty::type_has_escaping_regions(*t)));
    let substs = substs.erase_regions();

    // Load the info for the appropriate trait if necessary.
    match ty::trait_of_item(tcx, def_id) {
        None => {}
        Some(trait_id) => {
            ty::populate_implementations_for_trait_if_necessary(tcx, trait_id)
        }
    }

    // We need to do a bunch of special handling for default methods.
    // We need to modify the def_id and our substs in order to monomorphize
    // the function.
    let (is_default, def_id, substs) = match ty::provided_source(tcx, def_id) {
        None => {
            (false, def_id, tcx.mk_substs(substs))
        }
        Some(source_id) => {
            // There are two relevant substitutions when compiling
            // default methods. First, there is the substitution for
            // the type parameters of the impl we are using and the
            // method we are calling. This substitution is the substs
            // argument we already have.
            // In order to compile a default method, though, we need
            // to consider another substitution: the substitution for
            // the type parameters on trait; the impl we are using
            // implements the trait at some particular type
            // parameters, and we need to substitute for those first.
            // So, what we need to do is find this substitution and
            // compose it with the one we already have.

            let impl_id = ty::impl_or_trait_item(tcx, def_id).container()
                                                             .id();
            let impl_or_trait_item = ty::impl_or_trait_item(tcx, source_id);
            match impl_or_trait_item {
                ty::MethodTraitItem(method) => {
                    let trait_ref = ty::impl_trait_ref(tcx, impl_id).unwrap();

                    // Compute the first substitution
                    let first_subst =
                        ty::make_substs_for_receiver_types(tcx, &*trait_ref, &*method)
                        .erase_regions();

                    // And compose them
                    let new_substs = tcx.mk_substs(first_subst.subst(tcx, &substs));

                    debug!("trans_fn_with_vtables - default method: \
                            substs = {}, trait_subst = {}, \
                            first_subst = {}, new_subst = {}",
                           substs.repr(tcx), trait_ref.substs.repr(tcx),
                           first_subst.repr(tcx), new_substs.repr(tcx));

                    (true, source_id, new_substs)
                }
                ty::TypeTraitItem(_) => {
                    tcx.sess.bug("trans_fn_ref_with_vtables() tried \
                                  to translate an associated type?!")
                }
            }
        }
    };

    // If this is a closure, redirect to it.
    match closure::get_or_create_declaration_if_closure(ccx, def_id, substs) {
        None => {}
        Some(llfn) => return llfn,
    }

    // Check whether this fn has an inlined copy and, if so, redirect
    // def_id to the local id of the inlined copy.
    let def_id = inline::maybe_instantiate_inline(ccx, def_id);

    // We must monomorphise if the fn has type parameters, is a default method,
    // or is a named tuple constructor.
    let must_monomorphise = if !substs.types.is_empty() || is_default {
        true
    } else if def_id.krate == ast::LOCAL_CRATE {
        let map_node = session::expect(
            ccx.sess(),
            tcx.map.find(def_id.node),
            || "local item should be in ast map".to_string());

        match map_node {
            ast_map::NodeVariant(v) => match v.node.kind {
                ast::TupleVariantKind(ref args) => args.len() > 0,
                _ => false
            },
            ast_map::NodeStructCtor(_) => true,
            _ => false
        }
    } else {
        false
    };

    // Create a monomorphic version of generic functions
    if must_monomorphise {
        // Should be either intra-crate or inlined.
        assert_eq!(def_id.krate, ast::LOCAL_CRATE);

        let opt_ref_id = match node {
            ExprId(id) => if id != 0 { Some(id) } else { None },
            MethodCallKey(_) => None,
        };

        let (val, fn_ty, must_cast) =
            monomorphize::monomorphic_fn(ccx, def_id, substs, opt_ref_id);
        if must_cast && node != ExprId(0) {
            // Monotype of the REFERENCE to the function (type params
            // are subst'd)
            let ref_ty = match node {
                ExprId(id) => ty::node_id_to_type(tcx, id),
                MethodCallKey(method_call) => {
                    (*tcx.method_map.borrow())[method_call].ty
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
    let fn_type_scheme = ty::lookup_item_type(tcx, def_id);
    let fn_type = monomorphize::normalize_associated_type(tcx, &fn_type_scheme.ty);

    // Find the actual function pointer.
    let mut val = {
        if def_id.krate == ast::LOCAL_CRATE {
            // Internal reference.
            get_item_val(ccx, def_id.node)
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
        debug!("trans_fn_ref_with_vtables(): casting pointer!");
        val = consts::ptrcast(val, llptrty);
    } else {
        debug!("trans_fn_ref_with_vtables(): not casting pointer!");
    }

    Datum::new(val, fn_type, Rvalue::new(ByValue))
}

// ______________________________________________________________________
// Translating calls

pub fn trans_call<'a, 'blk, 'tcx>(in_cx: Block<'blk, 'tcx>,
                                  call_expr: &ast::Expr,
                                  f: &ast::Expr,
                                  args: CallArgs<'a, 'tcx>,
                                  dest: expr::Dest)
                                  -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_call");
    trans_call_inner(in_cx,
                     call_expr.debug_loc(),
                     common::expr_ty_adjusted(in_cx, f),
                     |cx, _| trans(cx, f),
                     args,
                     Some(dest)).bcx
}

pub fn trans_method_call<'a, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                         call_expr: &ast::Expr,
                                         rcvr: &ast::Expr,
                                         args: CallArgs<'a, 'tcx>,
                                         dest: expr::Dest)
                                         -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_method_call");
    debug!("trans_method_call(call_expr={})", call_expr.repr(bcx.tcx()));
    let method_call = MethodCall::expr(call_expr.id);
    let method_ty = (*bcx.tcx().method_map.borrow())[method_call].ty;
    trans_call_inner(
        bcx,
        call_expr.debug_loc(),
        common::monomorphize_type(bcx, method_ty),
        |cx, arg_cleanup_scope| {
            meth::trans_method_callee(cx, method_call, Some(rcvr), arg_cleanup_scope)
        },
        args,
        Some(dest)).bcx
}

pub fn trans_lang_call<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   did: ast::DefId,
                                   args: &[ValueRef],
                                   dest: Option<expr::Dest>,
                                   debug_loc: DebugLoc)
                                   -> Result<'blk, 'tcx> {
    let fty = if did.krate == ast::LOCAL_CRATE {
        ty::node_id_to_type(bcx.tcx(), did.node)
    } else {
        csearch::get_type(bcx.tcx(), did).ty
    };
    callee::trans_call_inner(bcx,
                             debug_loc,
                             fty,
                             |bcx, _| {
                                trans_fn_ref_with_substs_to_callee(bcx,
                                                                   did,
                                                                   0,
                                                                   subst::Substs::trans_empty())
                             },
                             ArgVals(args),
                             dest)
}

/// This behemoth of a function translates function calls. Unfortunately, in order to generate more
/// efficient LLVM output at -O0, it has quite a complex signature (refactoring this into two
/// functions seems like a good idea).
///
/// In particular, for lang items, it is invoked with a dest of None, and in that case the return
/// value contains the result of the fn. The lang item must not return a structural type or else
/// all heck breaks loose.
///
/// For non-lang items, `dest` is always Some, and hence the result is written into memory
/// somewhere. Nonetheless we return the actual return value of the function.
pub fn trans_call_inner<'a, 'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                           debug_loc: DebugLoc,
                                           callee_ty: Ty<'tcx>,
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

    let (abi, ret_ty) = match callee_ty.sty {
        ty::ty_bare_fn(_, ref f) => {
            let output = ty::erase_late_bound_regions(bcx.tcx(), &f.sig.output());
            (f.abi, output)
        }
        _ => panic!("expected bare rust fn or closure in trans_call_inner")
    };

    let (llfn, llenv, llself) = match callee.data {
        Fn(llfn) => {
            (llfn, None, None)
        }
        TraitItem(d) => {
            (d.llfn, None, Some(d.llself))
        }
        Intrinsic(node, substs) => {
            assert!(abi == synabi::RustIntrinsic);
            assert!(dest.is_some());

            let call_info = match debug_loc {
                DebugLoc::At(id, span) => NodeIdAndSpan { id: id, span: span },
                DebugLoc::None => {
                    bcx.sess().bug("No call info for intrinsic call?")
                }
            };

            return intrinsic::trans_intrinsic_call(bcx, node, callee_ty,
                                                   arg_cleanup_scope, args,
                                                   dest.unwrap(), substs,
                                                   call_info);
        }
        NamedTupleConstructor(substs, disr) => {
            assert!(dest.is_some());
            fcx.pop_custom_cleanup_scope(arg_cleanup_scope);

            let ctor_ty = callee_ty.subst(bcx.tcx(), &substs);
            return base::trans_named_tuple_constructor(bcx,
                                                       ctor_ty,
                                                       disr,
                                                       args,
                                                       dest.unwrap(),
                                                       debug_loc);
        }
    };

    // Intrinsics should not become actual functions.
    // We trans them in place in `trans_intrinsic_call`
    assert!(abi != synabi::RustIntrinsic);

    let is_rust_fn = abi == synabi::Rust || abi == synabi::RustCall;

    // Generate a location to store the result. If the user does
    // not care about the result, just make a stack slot.
    let opt_llretslot = dest.and_then(|dest| match dest {
        expr::SaveIn(dst) => Some(dst),
        expr::Ignore => {
            let ret_ty = match ret_ty {
                ty::FnConverging(ret_ty) => ret_ty,
                ty::FnDiverging => ty::mk_nil(ccx.tcx())
            };
            if !is_rust_fn ||
              type_of::return_uses_outptr(ccx, ret_ty) ||
              common::type_needs_drop(bcx.tcx(), ret_ty) {
                // Push the out-pointer if we use an out-pointer for this
                // return type, otherwise push "undef".
                if common::type_is_zero_size(ccx, ret_ty) {
                    let llty = type_of::type_of(ccx, ret_ty);
                    Some(common::C_undef(llty.ptr_to()))
                } else {
                    Some(alloc_ty(bcx, ret_ty, "__llret"))
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

        if let (ty::FnConverging(ret_ty), Some(llretslot)) = (ret_ty, opt_llretslot) {
            if type_of::return_uses_outptr(ccx, ret_ty) {
                llargs.push(llretslot);
            }
        }

        // Push the environment (or a trait object's self).
        match (llenv, llself) {
            (Some(llenv), None) => llargs.push(llenv),
            (None, Some(llself)) => llargs.push(llself),
            _ => {}
        }

        // Push the arguments.
        bcx = trans_args(bcx,
                         args,
                         callee_ty,
                         &mut llargs,
                         cleanup::CustomScope(arg_cleanup_scope),
                         llself.is_some(),
                         abi);

        fcx.scopes.borrow_mut().last_mut().unwrap().drop_non_lifetime_clean();

        // Invoke the actual rust fn and update bcx/llresult.
        let (llret, b) = base::invoke(bcx,
                                      llfn,
                                      &llargs[..],
                                      callee_ty,
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
            ArgExprs(a) => a.iter().map(|x| common::expr_ty(bcx, &**x)).collect(),
            _ => panic!("expected arg exprs.")
        };
        bcx = trans_args(bcx,
                         args,
                         callee_ty,
                         &mut llargs,
                         cleanup::CustomScope(arg_cleanup_scope),
                         false,
                         abi);
        fcx.scopes.borrow_mut().last_mut().unwrap().drop_non_lifetime_clean();

        bcx = foreign::trans_native_call(bcx,
                                         callee_ty,
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
    ArgExprs(&'a [P<ast::Expr>]),

    // Supply value of arguments as a list of LLVM value refs; frequently
    // used with lang items and so forth, when the argument is an internal
    // value.
    ArgVals(&'a [ValueRef]),

    // For overloaded operators: `(lhs, Vec(rhs, rhs_id), autoref)`. `lhs`
    // is the left-hand-side and `rhs/rhs_id` is the datum/expr-id of
    // the right-hand-side arguments (if any). `autoref` indicates whether the `rhs`
    // arguments should be auto-referenced
    ArgOverloadedOp(Datum<'tcx, Expr>, Vec<(Datum<'tcx, Expr>, ast::NodeId)>, bool),

    // Supply value of arguments as a list of expressions that must be
    // translated, for overloaded call operators.
    ArgOverloadedCall(Vec<&'a ast::Expr>),
}

fn trans_args_under_call_abi<'blk, 'tcx>(
                             mut bcx: Block<'blk, 'tcx>,
                             arg_exprs: &[P<ast::Expr>],
                             fn_ty: Ty<'tcx>,
                             llargs: &mut Vec<ValueRef>,
                             arg_cleanup_scope: cleanup::ScopeId,
                             ignore_self: bool)
                             -> Block<'blk, 'tcx>
{
    let args =
        ty::erase_late_bound_regions(
            bcx.tcx(), &ty::ty_fn_args(fn_ty));

    // Translate the `self` argument first.
    if !ignore_self {
        let arg_datum = unpack_datum!(bcx, expr::trans(bcx, &*arg_exprs[0]));
        llargs.push(unpack_result!(bcx, {
            trans_arg_datum(bcx,
                            args[0],
                            arg_datum,
                            arg_cleanup_scope,
                            DontAutorefArg)
        }))
    }

    // Now untuple the rest of the arguments.
    let tuple_expr = &arg_exprs[1];
    let tuple_type = common::node_id_type(bcx, tuple_expr.id);

    match tuple_type.sty {
        ty::ty_tup(ref field_types) => {
            let tuple_datum = unpack_datum!(bcx,
                                            expr::trans(bcx, &**tuple_expr));
            let tuple_lvalue_datum =
                unpack_datum!(bcx,
                              tuple_datum.to_lvalue_datum(bcx,
                                                          "args",
                                                          tuple_expr.id));
            let repr = adt::represent_type(bcx.ccx(), tuple_type);
            let repr_ptr = &*repr;
            for i in 0..field_types.len() {
                let arg_datum = tuple_lvalue_datum.get_element(
                    bcx,
                    field_types[i],
                    |srcval| {
                        adt::trans_field_ptr(bcx, repr_ptr, srcval, 0, i)
                    });
                let arg_datum = arg_datum.to_expr_datum();
                let arg_datum =
                    unpack_datum!(bcx, arg_datum.to_rvalue_datum(bcx, "arg"));
                let arg_datum =
                    unpack_datum!(bcx, arg_datum.to_appropriate_datum(bcx));
                llargs.push(arg_datum.add_clean(bcx.fcx, arg_cleanup_scope));
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
                              arg_exprs: Vec<&ast::Expr>,
                              fn_ty: Ty<'tcx>,
                              llargs: &mut Vec<ValueRef>,
                              arg_cleanup_scope: cleanup::ScopeId,
                              ignore_self: bool)
                              -> Block<'blk, 'tcx> {
    // Translate the `self` argument first.
    let arg_tys = ty::erase_late_bound_regions(bcx.tcx(),  &ty::ty_fn_args(fn_ty));
    if !ignore_self {
        let arg_datum = unpack_datum!(bcx, expr::trans(bcx, arg_exprs[0]));
        llargs.push(unpack_result!(bcx, {
            trans_arg_datum(bcx,
                            arg_tys[0],
                            arg_datum,
                            arg_cleanup_scope,
                            DontAutorefArg)
        }))
    }

    // Now untuple the rest of the arguments.
    let tuple_type = arg_tys[1];
    match tuple_type.sty {
        ty::ty_tup(ref field_types) => {
            for (i, &field_type) in field_types.iter().enumerate() {
                let arg_datum =
                    unpack_datum!(bcx, expr::trans(bcx, arg_exprs[i + 1]));
                llargs.push(unpack_result!(bcx, {
                    trans_arg_datum(bcx,
                                    field_type,
                                    arg_datum,
                                    arg_cleanup_scope,
                                    DontAutorefArg)
                }))
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
    let arg_tys = ty::erase_late_bound_regions(cx.tcx(), &ty::ty_fn_args(fn_ty));
    let variadic = ty::fn_is_variadic(fn_ty);

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
                llargs.push(unpack_result!(bcx, {
                    trans_arg_datum(bcx, arg_ty, arg_datum,
                                    arg_cleanup_scope,
                                    DontAutorefArg)
                }));
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

            llargs.push(unpack_result!(bcx, {
                trans_arg_datum(bcx, arg_tys[0], lhs,
                                arg_cleanup_scope,
                                DontAutorefArg)
            }));

            assert_eq!(arg_tys.len(), 1 + rhs.len());
            for (rhs, rhs_id) in rhs {
                llargs.push(unpack_result!(bcx, {
                    trans_arg_datum(bcx, arg_tys[1], rhs,
                                    arg_cleanup_scope,
                                    if autoref { DoAutorefArg(rhs_id) } else { DontAutorefArg })
                }));
            }
        }
        ArgVals(vs) => {
            llargs.push_all(vs);
        }
    }

    bcx
}

#[derive(Copy)]
pub enum AutorefArg {
    DontAutorefArg,
    DoAutorefArg(ast::NodeId)
}

pub fn trans_arg_datum<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   formal_arg_ty: Ty<'tcx>,
                                   arg_datum: Datum<'tcx, Expr>,
                                   arg_cleanup_scope: cleanup::ScopeId,
                                   autoref_arg: AutorefArg)
                                   -> Result<'blk, 'tcx> {
    let _icx = push_ctxt("trans_arg_datum");
    let mut bcx = bcx;
    let ccx = bcx.ccx();

    debug!("trans_arg_datum({})",
           formal_arg_ty.repr(bcx.tcx()));

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

    if formal_arg_ty != arg_datum_ty {
        // this could happen due to e.g. subtyping
        let llformal_arg_ty = type_of::type_of_explicit_arg(ccx, formal_arg_ty);
        debug!("casting actual type ({}) to match formal ({})",
               bcx.val_to_string(val), bcx.llty_str(llformal_arg_ty));
        debug!("Rust types: {}; {}", ty_to_string(bcx.tcx(), arg_datum_ty),
                                     ty_to_string(bcx.tcx(), formal_arg_ty));
        val = PointerCast(bcx, val, llformal_arg_ty);
    }

    debug!("--- trans_arg_datum passing {}", bcx.val_to_string(val));
    Result::new(bcx, val)
}
