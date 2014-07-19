// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Handles translation of callees as well as other call-related
 * things.  Callees are a superset of normal rust values and sometimes
 * have different representations.  In particular, top-level fn items
 * and methods are represented as just a fn ptr and not a full
 * closure.
 */

use arena::TypedArena;
use back::abi;
use back::link;
use driver::session;
use llvm::{ValueRef, get_param};
use llvm;
use metadata::csearch;
use middle::def;
use middle::subst;
use middle::subst::{Subst, VecPerParamSpace};
use middle::trans::adt;
use middle::trans::base;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::closure;
use middle::trans::common;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::datum::{Datum, KindOps};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::inline;
use middle::trans::foreign;
use middle::trans::intrinsic;
use middle::trans::meth;
use middle::trans::monomorphize;
use middle::trans::type_::Type;
use middle::trans::type_of;
use middle::ty;
use middle::typeck;
use middle::typeck::coherence::make_substs_for_receiver_types;
use middle::typeck::MethodCall;
use util::ppaux::Repr;

use std::gc::Gc;
use syntax::ast;
use syntax::ast_map;
use synabi = syntax::abi;

pub struct MethodData {
    pub llfn: ValueRef,
    pub llself: ValueRef,
}

pub enum CalleeData {
    Closure(Datum<Lvalue>),

    // Constructor for enum variant/tuple-like-struct
    // i.e. Some, Ok
    NamedTupleConstructor(subst::Substs, ty::Disr),

    // Represents a (possibly monomorphized) top-level fn item or method
    // item. Note that this is just the fn-ptr and is not a Rust closure
    // value (which is a pair).
    Fn(/* llfn */ ValueRef),

    Intrinsic(ast::NodeId, subst::Substs),

    TraitMethod(MethodData)
}

pub struct Callee<'a> {
    pub bcx: &'a Block<'a>,
    pub data: CalleeData,
}

fn trans<'a>(bcx: &'a Block<'a>, expr: &ast::Expr) -> Callee<'a> {
    let _icx = push_ctxt("trans_callee");
    debug!("callee::trans(expr={})", expr.repr(bcx.tcx()));

    // pick out special kinds of expressions that can be called:
    match expr.node {
        ast::ExprPath(_) => {
            return trans_def(bcx, bcx.def(expr.id), expr);
        }
        _ => {}
    }

    // any other expressions are closures:
    return datum_callee(bcx, expr);

    fn datum_callee<'a>(bcx: &'a Block<'a>, expr: &ast::Expr) -> Callee<'a> {
        let DatumBlock {bcx: mut bcx, datum} = expr::trans(bcx, expr);
        match ty::get(datum.ty).sty {
            ty::ty_bare_fn(..) => {
                let llval = datum.to_llscalarish(bcx);
                return Callee {
                    bcx: bcx,
                    data: Fn(llval),
                };
            }
            ty::ty_closure(..) => {
                let datum = unpack_datum!(
                    bcx, datum.to_lvalue_datum(bcx, "callee", expr.id));
                return Callee {
                    bcx: bcx,
                    data: Closure(datum),
                };
            }
            _ => {
                bcx.tcx().sess.span_bug(
                    expr.span,
                    format!("type of callee is neither bare-fn nor closure: \
                             {}",
                            bcx.ty_to_string(datum.ty)).as_slice());
            }
        }
    }

    fn fn_callee<'a>(bcx: &'a Block<'a>, llfn: ValueRef) -> Callee<'a> {
        return Callee {
            bcx: bcx,
            data: Fn(llfn),
        };
    }

    fn trans_def<'a>(bcx: &'a Block<'a>, def: def::Def, ref_expr: &ast::Expr)
                 -> Callee<'a> {
        debug!("trans_def(def={}, ref_expr={})", def.repr(bcx.tcx()), ref_expr.repr(bcx.tcx()));
        let expr_ty = node_id_type(bcx, ref_expr.id);
        match def {
            def::DefFn(did, _) if {
                let def_id = if did.krate != ast::LOCAL_CRATE {
                    inline::maybe_instantiate_inline(bcx.ccx(), did)
                } else {
                    did
                };
                match bcx.tcx().map.find(def_id.node) {
                    Some(ast_map::NodeStructCtor(_)) => true,
                    _ => false
                }
            } => {
                let substs = node_id_substs(bcx, ExprId(ref_expr.id));
                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(substs, 0)
                }
            }
            def::DefFn(did, _) if match ty::get(expr_ty).sty {
                ty::ty_bare_fn(ref f) => f.abi == synabi::RustIntrinsic,
                _ => false
            } => {
                let substs = node_id_substs(bcx, ExprId(ref_expr.id));
                let def_id = if did.krate != ast::LOCAL_CRATE {
                    inline::maybe_instantiate_inline(bcx.ccx(), did)
                } else {
                    did
                };
                Callee { bcx: bcx, data: Intrinsic(def_id.node, substs) }
            }
            def::DefFn(did, _) |
            def::DefStaticMethod(did, def::FromImpl(_), _) => {
                fn_callee(bcx, trans_fn_ref(bcx, did, ExprId(ref_expr.id)))
            }
            def::DefStaticMethod(impl_did,
                                 def::FromTrait(trait_did),
                                 _) => {
                fn_callee(bcx, meth::trans_static_method_callee(bcx, impl_did,
                                                                trait_did,
                                                                ref_expr.id))
            }
            def::DefVariant(tid, vid, _) => {
                let vinfo = ty::enum_variant_with_id(bcx.tcx(), tid, vid);
                let substs = node_id_substs(bcx, ExprId(ref_expr.id));

                // Nullary variants are not callable
                assert!(vinfo.args.len() > 0u);

                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(substs, vinfo.disr_val)
                }
            }
            def::DefStruct(_) => {
                let substs = node_id_substs(bcx, ExprId(ref_expr.id));
                Callee {
                    bcx: bcx,
                    data: NamedTupleConstructor(substs, 0)
                }
            }
            def::DefStatic(..) |
            def::DefArg(..) |
            def::DefLocal(..) |
            def::DefBinding(..) |
            def::DefUpvar(..) => {
                datum_callee(bcx, ref_expr)
            }
            def::DefMod(..) | def::DefForeignMod(..) | def::DefTrait(..) |
            def::DefTy(..) | def::DefPrimTy(..) |
            def::DefUse(..) | def::DefTyParamBinder(..) |
            def::DefRegion(..) | def::DefLabel(..) | def::DefTyParam(..) |
            def::DefSelfTy(..) | def::DefMethod(..) => {
                bcx.tcx().sess.span_bug(
                    ref_expr.span,
                    format!("cannot translate def {:?} \
                             to a callable thing!", def).as_slice());
            }
        }
    }
}

pub fn trans_fn_ref(bcx: &Block, def_id: ast::DefId, node: ExprOrMethodCall) -> ValueRef {
    /*!
     * Translates a reference (with id `ref_id`) to the fn/method
     * with id `def_id` into a function pointer.  This may require
     * monomorphization or inlining.
     */

    let _icx = push_ctxt("trans_fn_ref");

    let substs = node_id_substs(bcx, node);
    let vtable_key = match node {
        ExprId(id) => MethodCall::expr(id),
        MethodCall(method_call) => method_call
    };
    let vtables = node_vtables(bcx, vtable_key);
    debug!("trans_fn_ref(def_id={}, node={:?}, substs={}, vtables={})",
           def_id.repr(bcx.tcx()),
           node,
           substs.repr(bcx.tcx()),
           vtables.repr(bcx.tcx()));
    trans_fn_ref_with_vtables(bcx, def_id, node, substs, vtables)
}

fn trans_fn_ref_with_vtables_to_callee<'a>(bcx: &'a Block<'a>,
                                           def_id: ast::DefId,
                                           ref_id: ast::NodeId,
                                           substs: subst::Substs,
                                           vtables: typeck::vtable_res)
                                           -> Callee<'a> {
    Callee {
        bcx: bcx,
        data: Fn(trans_fn_ref_with_vtables(bcx,
                                           def_id,
                                           ExprId(ref_id),
                                           substs,
                                           vtables)),
    }
}

fn resolve_default_method_vtables(bcx: &Block,
                                  impl_id: ast::DefId,
                                  substs: &subst::Substs,
                                  impl_vtables: typeck::vtable_res)
                                  -> typeck::vtable_res
{
    // Get the vtables that the impl implements the trait at
    let impl_res = ty::lookup_impl_vtables(bcx.tcx(), impl_id);

    // Build up a param_substs that we are going to resolve the
    // trait_vtables under.
    let param_substs = param_substs {
        substs: (*substs).clone(),
        vtables: impl_vtables.clone()
    };

    let mut param_vtables = resolve_vtables_under_param_substs(
        bcx.tcx(), &param_substs, &impl_res);

    // Now we pull any vtables for parameters on the actual method.
    param_vtables.push_all(subst::FnSpace,
                           impl_vtables.get_slice(subst::FnSpace));

    param_vtables
}

/// Translates the adapter that deconstructs a `Box<Trait>` object into
/// `Trait` so that a by-value self method can be called.
pub fn trans_unboxing_shim(bcx: &Block,
                           llshimmedfn: ValueRef,
                           method: &ty::Method,
                           method_id: ast::DefId,
                           substs: subst::Substs)
                           -> ValueRef {
    let _icx = push_ctxt("trans_unboxing_shim");
    let ccx = bcx.ccx();
    let tcx = bcx.tcx();

    // Transform the self type to `Box<self_type>`.
    let self_type = *method.fty.sig.inputs.get(0);
    let boxed_self_type = ty::mk_uniq(tcx, self_type);
    let boxed_function_type = ty::FnSig {
        binder_id: method.fty.sig.binder_id,
        inputs: method.fty.sig.inputs.iter().enumerate().map(|(i, typ)| {
            if i == 0 {
                boxed_self_type
            } else {
                *typ
            }
        }).collect(),
        output: method.fty.sig.output,
        variadic: false,
    };
    let boxed_function_type = ty::BareFnTy {
        fn_style: method.fty.fn_style,
        abi: method.fty.abi,
        sig: boxed_function_type,
    };
    let boxed_function_type =
        ty::mk_bare_fn(tcx, boxed_function_type).subst(tcx, &substs);
    let function_type =
        ty::mk_bare_fn(tcx, method.fty.clone()).subst(tcx, &substs);

    let function_name = ty::with_path(tcx, method_id, |path| {
        link::mangle_internal_name_by_path_and_seq(path, "unboxing_shim")
    });
    let llfn = decl_internal_rust_fn(ccx,
                                     boxed_function_type,
                                     function_name.as_slice());

    let block_arena = TypedArena::new();
    let empty_param_substs = param_substs::empty();
    let return_type = ty::ty_fn_ret(boxed_function_type);
    let fcx = new_fn_ctxt(ccx,
                          llfn,
                          -1,
                          false,
                          return_type,
                          &empty_param_substs,
                          None,
                          &block_arena);
    let mut bcx = init_function(&fcx, false, return_type);

    // Create the substituted versions of the self type.
    let arg_scope = fcx.push_custom_cleanup_scope();
    let arg_scope_id = cleanup::CustomScope(arg_scope);
    let boxed_arg_types = ty::ty_fn_args(boxed_function_type);
    let boxed_self_type = *boxed_arg_types.get(0);
    let arg_types = ty::ty_fn_args(function_type);
    let self_type = *arg_types.get(0);
    let boxed_self_kind = arg_kind(&fcx, boxed_self_type);

    // Create a datum for self.
    let llboxedself = get_param(fcx.llfn, fcx.arg_pos(0) as u32);
    let llboxedself = Datum::new(llboxedself,
                                 boxed_self_type,
                                 boxed_self_kind);
    let boxed_self =
        unpack_datum!(bcx,
                      llboxedself.to_lvalue_datum_in_scope(bcx,
                                                           "boxedself",
                                                           arg_scope_id));

    // This `Load` is needed because lvalue data are always by-ref.
    let llboxedself = Load(bcx, boxed_self.val);

    let llself = if type_is_immediate(ccx, self_type) {
        let llboxedself = Load(bcx, llboxedself);
        immediate_rvalue(llboxedself, self_type)
    } else {
        let llself = rvalue_scratch_datum(bcx, self_type, "self");
        memcpy_ty(bcx, llself.val, llboxedself, self_type);
        llself
    };

    // Make sure we don't free the box twice!
    boxed_self.kind.post_store(bcx, boxed_self.val, boxed_self_type);

    // Schedule a cleanup to free the box.
    fcx.schedule_free_value(arg_scope_id,
                            llboxedself,
                            cleanup::HeapExchange,
                            self_type);

    // Now call the function.
    let mut llshimmedargs = vec!(llself.val);
    for i in range(1, arg_types.len()) {
        llshimmedargs.push(get_param(fcx.llfn, fcx.arg_pos(i) as u32));
    }
    bcx = trans_call_inner(bcx,
                           None,
                           function_type,
                           |bcx, _| {
                               Callee {
                                   bcx: bcx,
                                   data: Fn(llshimmedfn),
                               }
                           },
                           ArgVals(llshimmedargs.as_slice()),
                           match fcx.llretptr.get() {
                               None => None,
                               Some(llretptr) => Some(expr::SaveIn(llretptr)),
                           }).bcx;

    bcx = fcx.pop_and_trans_custom_cleanup_scope(bcx, arg_scope);
    finish_fn(&fcx, bcx, return_type);

    llfn
}

pub fn trans_fn_ref_with_vtables(
    bcx: &Block,                 //
    def_id: ast::DefId,          // def id of fn
    node: ExprOrMethodCall,      // node id of use of fn; may be zero if N/A
    substs: subst::Substs,       // values for fn's ty params
    vtables: typeck::vtable_res) // vtables for the call
    -> ValueRef
{
    /*!
     * Translates a reference to a fn/method item, monomorphizing and
     * inlining as it goes.
     *
     * # Parameters
     *
     * - `bcx`: the current block where the reference to the fn occurs
     * - `def_id`: def id of the fn or method item being referenced
     * - `node`: node id of the reference to the fn/method, if applicable.
     *   This parameter may be zero; but, if so, the resulting value may not
     *   have the right type, so it must be cast before being used.
     * - `substs`: values for each of the fn/method's parameters
     * - `vtables`: values for each bound on each of the type parameters
     */

    let _icx = push_ctxt("trans_fn_ref_with_vtables");
    let ccx = bcx.ccx();
    let tcx = bcx.tcx();

    debug!("trans_fn_ref_with_vtables(bcx={}, def_id={}, node={:?}, \
            substs={}, vtables={})",
           bcx.to_str(),
           def_id.repr(tcx),
           node,
           substs.repr(tcx),
           vtables.repr(tcx));

    assert!(substs.types.all(|t| !ty::type_needs_infer(*t)));

    // Load the info for the appropriate trait if necessary.
    match ty::trait_of_method(tcx, def_id) {
        None => {}
        Some(trait_id) => {
            ty::populate_implementations_for_trait_if_necessary(tcx, trait_id)
        }
    }

    // We need to do a bunch of special handling for default methods.
    // We need to modify the def_id and our substs in order to monomorphize
    // the function.
    let (is_default, def_id, substs, vtables) =
        match ty::provided_source(tcx, def_id) {
        None => (false, def_id, substs, vtables),
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

            let impl_id = ty::method(tcx, def_id).container_id();
            let method = ty::method(tcx, source_id);
            let trait_ref = ty::impl_trait_ref(tcx, impl_id)
                .expect("could not find trait_ref for impl with \
                         default methods");

            // Compute the first substitution
            let first_subst = make_substs_for_receiver_types(
                tcx, &*trait_ref, &*method);

            // And compose them
            let new_substs = first_subst.subst(tcx, &substs);

            debug!("trans_fn_with_vtables - default method: \
                    substs = {}, trait_subst = {}, \
                    first_subst = {}, new_subst = {}, \
                    vtables = {}",
                   substs.repr(tcx), trait_ref.substs.repr(tcx),
                   first_subst.repr(tcx), new_substs.repr(tcx),
                   vtables.repr(tcx));

            let param_vtables =
                resolve_default_method_vtables(bcx, impl_id, &substs, vtables);

            debug!("trans_fn_with_vtables - default method: \
                    param_vtables = {}",
                   param_vtables.repr(tcx));

            (true, source_id, new_substs, param_vtables)
        }
    };

    // If this is an unboxed closure, redirect to it.
    match closure::get_or_create_declaration_if_unboxed_closure(ccx, def_id) {
        None => {}
        Some(llfn) => return llfn,
    }

    // Check whether this fn has an inlined copy and, if so, redirect
    // def_id to the local id of the inlined copy.
    let def_id = {
        if def_id.krate != ast::LOCAL_CRATE {
            inline::maybe_instantiate_inline(ccx, def_id)
        } else {
            def_id
        }
    };

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
            MethodCall(_) => None,
        };

        let (val, must_cast) =
            monomorphize::monomorphic_fn(ccx, def_id, &substs,
                                         vtables, opt_ref_id);
        let mut val = val;
        if must_cast && node != ExprId(0) {
            // Monotype of the REFERENCE to the function (type params
            // are subst'd)
            let ref_ty = match node {
                ExprId(id) => node_id_type(bcx, id),
                MethodCall(method_call) => {
                    let t = bcx.tcx().method_map.borrow().get(&method_call).ty;
                    monomorphize_type(bcx, t)
                }
            };

            val = PointerCast(
                bcx, val, type_of::type_of_fn_from_ty(ccx, ref_ty).ptr_to());
        }
        return val;
    }

    // Polytype of the function item (may have type params)
    let fn_tpt = ty::lookup_item_type(tcx, def_id);

    // Find the actual function pointer.
    let mut val = {
        if def_id.krate == ast::LOCAL_CRATE {
            // Internal reference.
            get_item_val(ccx, def_id.node)
        } else {
            // External reference.
            trans_external_path(ccx, def_id, fn_tpt.ty)
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
    let llty = type_of::type_of_fn_from_ty(ccx, fn_tpt.ty);
    let llptrty = llty.ptr_to();
    if val_ty(val) != llptrty {
        debug!("trans_fn_ref_with_vtables(): casting pointer!");
        val = BitCast(bcx, val, llptrty);
    } else {
        debug!("trans_fn_ref_with_vtables(): not casting pointer!");
    }

    val
}

// ______________________________________________________________________
// Translating calls

pub fn trans_call<'a>(
                  in_cx: &'a Block<'a>,
                  call_ex: &ast::Expr,
                  f: &ast::Expr,
                  args: CallArgs,
                  dest: expr::Dest)
                  -> &'a Block<'a> {
    let _icx = push_ctxt("trans_call");
    trans_call_inner(in_cx,
                     Some(common::expr_info(call_ex)),
                     expr_ty(in_cx, f),
                     |cx, _| trans(cx, f),
                     args,
                     Some(dest)).bcx
}

pub fn trans_method_call<'a>(
                         bcx: &'a Block<'a>,
                         call_ex: &ast::Expr,
                         rcvr: &ast::Expr,
                         args: CallArgs,
                         dest: expr::Dest)
                         -> &'a Block<'a> {
    let _icx = push_ctxt("trans_method_call");
    debug!("trans_method_call(call_ex={})", call_ex.repr(bcx.tcx()));
    let method_call = MethodCall::expr(call_ex.id);
    let method_ty = bcx.tcx().method_map.borrow().get(&method_call).ty;
    trans_call_inner(
        bcx,
        Some(common::expr_info(call_ex)),
        monomorphize_type(bcx, method_ty),
        |cx, arg_cleanup_scope| {
            meth::trans_method_callee(cx, method_call, Some(rcvr), arg_cleanup_scope)
        },
        args,
        Some(dest)).bcx
}

pub fn trans_lang_call<'a>(
                       bcx: &'a Block<'a>,
                       did: ast::DefId,
                       args: &[ValueRef],
                       dest: Option<expr::Dest>)
                       -> Result<'a> {
    let fty = if did.krate == ast::LOCAL_CRATE {
        ty::node_id_to_type(bcx.tcx(), did.node)
    } else {
        csearch::get_type(bcx.tcx(), did).ty
    };
    callee::trans_call_inner(bcx,
                             None,
                             fty,
                             |bcx, _| {
                                trans_fn_ref_with_vtables_to_callee(bcx,
                                                                    did,
                                                                    0,
                                                                    subst::Substs::empty(),
                                                                    VecPerParamSpace::empty())
                             },
                             ArgVals(args),
                             dest)
}

pub fn trans_call_inner<'a>(
                        bcx: &'a Block<'a>,
                        call_info: Option<NodeInfo>,
                        callee_ty: ty::t,
                        get_callee: |bcx: &'a Block<'a>,
                                     arg_cleanup_scope: cleanup::ScopeId|
                                     -> Callee<'a>,
                        args: CallArgs,
                        dest: Option<expr::Dest>)
                        -> Result<'a> {
    /*!
     * This behemoth of a function translates function calls.
     * Unfortunately, in order to generate more efficient LLVM
     * output at -O0, it has quite a complex signature (refactoring
     * this into two functions seems like a good idea).
     *
     * In particular, for lang items, it is invoked with a dest of
     * None, and in that case the return value contains the result of
     * the fn. The lang item must not return a structural type or else
     * all heck breaks loose.
     *
     * For non-lang items, `dest` is always Some, and hence the result
     * is written into memory somewhere. Nonetheless we return the
     * actual return value of the function.
     */

    // Introduce a temporary cleanup scope that will contain cleanups
    // for the arguments while they are being evaluated. The purpose
    // this cleanup is to ensure that, should a failure occur while
    // evaluating argument N, the values for arguments 0...N-1 are all
    // cleaned up. If no failure occurs, the values are handed off to
    // the callee, and hence none of the cleanups in this temporary
    // scope will ever execute.
    let fcx = bcx.fcx;
    let ccx = fcx.ccx;
    let arg_cleanup_scope = fcx.push_custom_cleanup_scope();

    let callee = get_callee(bcx, cleanup::CustomScope(arg_cleanup_scope));
    let mut bcx = callee.bcx;

    let (abi, ret_ty) = match ty::get(callee_ty).sty {
        ty::ty_bare_fn(ref f) => (f.abi, f.sig.output),
        ty::ty_closure(ref f) => (f.abi, f.sig.output),
        _ => fail!("expected bare rust fn or closure in trans_call_inner")
    };

    let (llfn, llenv, llself) = match callee.data {
        Fn(llfn) => {
            (llfn, None, None)
        }
        TraitMethod(d) => {
            (d.llfn, None, Some(d.llself))
        }
        Closure(d) => {
            // Closures are represented as (llfn, llclosure) pair:
            // load the requisite values out.
            let pair = d.to_llref();
            let llfn = GEPi(bcx, pair, [0u, abi::fn_field_code]);
            let llfn = Load(bcx, llfn);
            let llenv = GEPi(bcx, pair, [0u, abi::fn_field_box]);
            let llenv = Load(bcx, llenv);
            (llfn, Some(llenv), None)
        }
        Intrinsic(node, substs) => {
            assert!(abi == synabi::RustIntrinsic);
            assert!(dest.is_some());

            return intrinsic::trans_intrinsic_call(bcx, node, callee_ty,
                                                   arg_cleanup_scope, args,
                                                   dest.unwrap(), substs);
        }
        NamedTupleConstructor(substs, disr) => {
            assert!(dest.is_some());
            fcx.pop_custom_cleanup_scope(arg_cleanup_scope);

            let ctor_ty = callee_ty.subst(bcx.tcx(), &substs);
            return base::trans_named_tuple_constructor(bcx, ctor_ty, disr,
                                                       args, dest.unwrap());
        }
    };

    // Intrinsics should not become actual functions.
    // We trans them in place in `trans_intrinsic_call`
    assert!(abi != synabi::RustIntrinsic);

    // Generate a location to store the result. If the user does
    // not care about the result, just make a stack slot.
    let opt_llretslot = match dest {
        None => {
            assert!(!type_of::return_uses_outptr(ccx, ret_ty));
            None
        }
        Some(expr::SaveIn(dst)) => Some(dst),
        Some(expr::Ignore) => {
            if !type_is_zero_size(ccx, ret_ty) {
                Some(alloc_ty(bcx, ret_ty, "__llret"))
            } else {
                let llty = type_of::type_of(ccx, ret_ty);
                Some(C_undef(llty.ptr_to()))
            }
        }
    };

    let mut llresult = unsafe {
        llvm::LLVMGetUndef(Type::nil(ccx).ptr_to().to_ref())
    };

    // The code below invokes the function, using either the Rust
    // conventions (if it is a rust fn) or the native conventions
    // (otherwise).  The important part is that, when all is sad
    // and done, either the return value of the function will have been
    // written in opt_llretslot (if it is Some) or `llresult` will be
    // set appropriately (otherwise).
    if abi == synabi::Rust || abi == synabi::RustCall {
        let mut llargs = Vec::new();

        // Push the out-pointer if we use an out-pointer for this
        // return type, otherwise push "undef".
        if type_of::return_uses_outptr(ccx, ret_ty) {
            llargs.push(opt_llretslot.unwrap());
        }

        // Push the environment (or a trait object's self).
        match (llenv, llself) {
            (Some(llenv), None) => {
                llargs.push(llenv)
            },
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

        fcx.pop_custom_cleanup_scope(arg_cleanup_scope);

        // Invoke the actual rust fn and update bcx/llresult.
        let (llret, b) = base::invoke(bcx,
                                      llfn,
                                      llargs,
                                      callee_ty,
                                      call_info);
        bcx = b;
        llresult = llret;

        // If the Rust convention for this type is return via
        // the return value, copy it into llretslot.
        match opt_llretslot {
            Some(llretslot) => {
                if !type_of::return_uses_outptr(bcx.ccx(), ret_ty) &&
                    !type_is_zero_size(bcx.ccx(), ret_ty)
                {
                    store_ty(bcx, llret, llretslot, ret_ty)
                }
            }
            None => {}
        }
    } else {
        // Lang items are the only case where dest is None, and
        // they are always Rust fns.
        assert!(dest.is_some());

        let mut llargs = Vec::new();
        let arg_tys = match args {
            ArgExprs(a) => a.iter().map(|x| expr_ty(bcx, &**x)).collect(),
            _ => fail!("expected arg exprs.")
        };
        bcx = trans_args(bcx,
                         args,
                         callee_ty,
                         &mut llargs,
                         cleanup::CustomScope(arg_cleanup_scope),
                         false,
                         abi);
        fcx.pop_custom_cleanup_scope(arg_cleanup_scope);
        bcx = foreign::trans_native_call(bcx, callee_ty,
                                         llfn, opt_llretslot.unwrap(),
                                         llargs.as_slice(), arg_tys);
    }

    // If the caller doesn't care about the result of this fn call,
    // drop the temporary slot we made.
    match dest {
        None => {
            assert!(!type_of::return_uses_outptr(bcx.ccx(), ret_ty));
        }
        Some(expr::Ignore) => {
            // drop the value if it is not being saved.
            bcx = glue::drop_ty(bcx, opt_llretslot.unwrap(), ret_ty);
        }
        Some(expr::SaveIn(_)) => { }
    }

    if ty::type_is_bot(ret_ty) {
        Unreachable(bcx);
    }

    Result::new(bcx, llresult)
}

pub enum CallArgs<'a> {
    // Supply value of arguments as a list of expressions that must be
    // translated. This is used in the common case of `foo(bar, qux)`.
    ArgExprs(&'a [Gc<ast::Expr>]),

    // Supply value of arguments as a list of LLVM value refs; frequently
    // used with lang items and so forth, when the argument is an internal
    // value.
    ArgVals(&'a [ValueRef]),

    // For overloaded operators: `(lhs, Option(rhs, rhs_id))`. `lhs`
    // is the left-hand-side and `rhs/rhs_id` is the datum/expr-id of
    // the right-hand-side (if any).
    ArgOverloadedOp(Datum<Expr>, Option<(Datum<Expr>, ast::NodeId)>),

    // Supply value of arguments as a list of expressions that must be
    // translated, for overloaded call operators.
    ArgOverloadedCall(&'a [Gc<ast::Expr>]),
}

fn trans_args_under_call_abi<'a>(
                             mut bcx: &'a Block<'a>,
                             arg_exprs: &[Gc<ast::Expr>],
                             fn_ty: ty::t,
                             llargs: &mut Vec<ValueRef>,
                             arg_cleanup_scope: cleanup::ScopeId,
                             ignore_self: bool)
                             -> &'a Block<'a> {
    // Translate the `self` argument first.
    let arg_tys = ty::ty_fn_args(fn_ty);
    if !ignore_self {
        let arg_datum = unpack_datum!(bcx, expr::trans(bcx, &*arg_exprs[0]));
        llargs.push(unpack_result!(bcx, {
            trans_arg_datum(bcx,
                            *arg_tys.get(0),
                            arg_datum,
                            arg_cleanup_scope,
                            DontAutorefArg)
        }))
    }

    // Now untuple the rest of the arguments.
    let tuple_expr = arg_exprs[1];
    let tuple_type = node_id_type(bcx, tuple_expr.id);

    match ty::get(tuple_type).sty {
        ty::ty_tup(ref field_types) => {
            let tuple_datum = unpack_datum!(bcx,
                                            expr::trans(bcx, &*tuple_expr));
            let tuple_lvalue_datum =
                unpack_datum!(bcx,
                              tuple_datum.to_lvalue_datum(bcx,
                                                          "args",
                                                          tuple_expr.id));
            let repr = adt::represent_type(bcx.ccx(), tuple_type);
            let repr_ptr = &*repr;
            for i in range(0, field_types.len()) {
                let arg_datum = tuple_lvalue_datum.get_element(
                    *field_types.get(i),
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
        ty::ty_nil => {}
        _ => {
            bcx.sess().span_bug(tuple_expr.span,
                                "argument to `.call()` wasn't a tuple?!")
        }
    };

    bcx
}

fn trans_overloaded_call_args<'a>(
                              mut bcx: &'a Block<'a>,
                              arg_exprs: &[Gc<ast::Expr>],
                              fn_ty: ty::t,
                              llargs: &mut Vec<ValueRef>,
                              arg_cleanup_scope: cleanup::ScopeId,
                              ignore_self: bool)
                              -> &'a Block<'a> {
    // Translate the `self` argument first.
    let arg_tys = ty::ty_fn_args(fn_ty);
    if !ignore_self {
        let arg_datum = unpack_datum!(bcx, expr::trans(bcx, &*arg_exprs[0]));
        llargs.push(unpack_result!(bcx, {
            trans_arg_datum(bcx,
                            *arg_tys.get(0),
                            arg_datum,
                            arg_cleanup_scope,
                            DontAutorefArg)
        }))
    }

    // Now untuple the rest of the arguments.
    let tuple_type = *arg_tys.get(1);
    match ty::get(tuple_type).sty {
        ty::ty_tup(ref field_types) => {
            for (i, &field_type) in field_types.iter().enumerate() {
                let arg_datum =
                    unpack_datum!(bcx, expr::trans(bcx, &*arg_exprs[i + 1]));
                llargs.push(unpack_result!(bcx, {
                    trans_arg_datum(bcx,
                                    field_type,
                                    arg_datum,
                                    arg_cleanup_scope,
                                    DontAutorefArg)
                }))
            }
        }
        ty::ty_nil => {}
        _ => {
            bcx.sess().span_bug(arg_exprs[0].span,
                                "argument to `.call()` wasn't a tuple?!")
        }
    };

    bcx
}

pub fn trans_args<'a>(
                  cx: &'a Block<'a>,
                  args: CallArgs,
                  fn_ty: ty::t,
                  llargs: &mut Vec<ValueRef> ,
                  arg_cleanup_scope: cleanup::ScopeId,
                  ignore_self: bool,
                  abi: synabi::Abi)
                  -> &'a Block<'a> {
    debug!("trans_args(abi={})", abi);

    let _icx = push_ctxt("trans_args");
    let arg_tys = ty::ty_fn_args(fn_ty);
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
                    expr_ty_adjusted(cx, &**arg_expr)
                } else {
                    *arg_tys.get(i)
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
        ArgOverloadedOp(lhs, rhs) => {
            assert!(!variadic);

            llargs.push(unpack_result!(bcx, {
                trans_arg_datum(bcx, *arg_tys.get(0), lhs,
                                arg_cleanup_scope,
                                DontAutorefArg)
            }));

            match rhs {
                Some((rhs, rhs_id)) => {
                    assert_eq!(arg_tys.len(), 2);

                    llargs.push(unpack_result!(bcx, {
                        trans_arg_datum(bcx, *arg_tys.get(1), rhs,
                                        arg_cleanup_scope,
                                        DoAutorefArg(rhs_id))
                    }));
                }
                None => assert_eq!(arg_tys.len(), 1)
            }
        }
        ArgVals(vs) => {
            llargs.push_all(vs);
        }
    }

    bcx
}

pub enum AutorefArg {
    DontAutorefArg,
    DoAutorefArg(ast::NodeId)
}

pub fn trans_arg_datum<'a>(
                      bcx: &'a Block<'a>,
                      formal_arg_ty: ty::t,
                      arg_datum: Datum<Expr>,
                      arg_cleanup_scope: cleanup::ScopeId,
                      autoref_arg: AutorefArg)
                      -> Result<'a> {
    let _icx = push_ctxt("trans_arg_datum");
    let mut bcx = bcx;
    let ccx = bcx.ccx();

    debug!("trans_arg_datum({})",
           formal_arg_ty.repr(bcx.tcx()));

    let arg_datum_ty = arg_datum.ty;

    debug!("   arg datum: {}", arg_datum.to_string(bcx.ccx()));

    let mut val;
    if ty::type_is_bot(arg_datum_ty) {
        // For values of type _|_, we generate an
        // "undef" value, as such a value should never
        // be inspected. It's important for the value
        // to have type lldestty (the callee's expected type).
        let llformal_arg_ty = type_of::type_of_explicit_arg(ccx, formal_arg_ty);
        unsafe {
            val = llvm::LLVMGetUndef(llformal_arg_ty.to_ref());
        }
    } else {
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
                // However, we must cleanup should we fail before the
                // callee is actually invoked.
                val = arg_datum.add_clean(bcx.fcx, arg_cleanup_scope);
            }
        }

        if formal_arg_ty != arg_datum_ty {
            // this could happen due to e.g. subtyping
            let llformal_arg_ty = type_of::type_of_explicit_arg(ccx, formal_arg_ty);
            debug!("casting actual type ({}) to match formal ({})",
                   bcx.val_to_string(val), bcx.llty_str(llformal_arg_ty));
            val = PointerCast(bcx, val, llformal_arg_ty);
        }
    }

    debug!("--- trans_arg_datum passing {}", bcx.val_to_string(val));
    Result::new(bcx, val)
}
