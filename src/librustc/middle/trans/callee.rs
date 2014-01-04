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

use std::vec;

use back::abi;
use driver::session;
use lib::llvm::{ValueRef, NoAliasAttribute, StructRetAttribute};
use lib::llvm::llvm;
use metadata::csearch;
use middle::trans::base;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::datum::Datum;
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::inline;
use middle::trans::meth;
use middle::trans::monomorphize;
use middle::trans::type_of;
use middle::trans::foreign;
use middle::ty;
use middle::subst::Subst;
use middle::typeck;
use middle::typeck::coherence::make_substs_for_receiver_types;
use util::ppaux::Repr;

use middle::trans::type_::Type;

use syntax::ast;
use syntax::abi::AbiSet;
use syntax::ast_map;

// Represents a (possibly monomorphized) top-level fn item or method
// item.  Note that this is just the fn-ptr and is not a Rust closure
// value (which is a pair).
pub struct FnData {
    llfn: ValueRef,
}

pub struct MethodData {
    llfn: ValueRef,
    llself: ValueRef,
    temp_cleanup: Option<ValueRef>,
    self_mode: ty::SelfMode,
}

pub enum CalleeData {
    Closure(Datum),
    Fn(FnData),
    Method(MethodData)
}

pub struct Callee {
    bcx: @Block,
    data: CalleeData
}

pub fn trans(bcx: @Block, expr: &ast::Expr) -> Callee {
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

    fn datum_callee(bcx: @Block, expr: &ast::Expr) -> Callee {
        let DatumBlock {bcx, datum} = expr::trans_to_datum(bcx, expr);
        match ty::get(datum.ty).sty {
            ty::ty_bare_fn(..) => {
                let llval = datum.to_appropriate_llval(bcx);
                return Callee {bcx: bcx, data: Fn(FnData {llfn: llval})};
            }
            ty::ty_closure(..) => {
                return Callee {bcx: bcx, data: Closure(datum)};
            }
            _ => {
                bcx.tcx().sess.span_bug(
                    expr.span,
                    format!("Type of callee is neither bare-fn nor closure: {}",
                         bcx.ty_to_str(datum.ty)));
            }
        }
    }

    fn fn_callee(bcx: @Block, fd: FnData) -> Callee {
        return Callee {bcx: bcx, data: Fn(fd)};
    }

    fn trans_def(bcx: @Block, def: ast::Def, ref_expr: &ast::Expr) -> Callee {
        match def {
            ast::DefFn(did, _) |
            ast::DefStaticMethod(did, ast::FromImpl(_), _) => {
                fn_callee(bcx, trans_fn_ref(bcx, did, ref_expr.id))
            }
            ast::DefStaticMethod(impl_did,
                                   ast::FromTrait(trait_did),
                                   _) => {
                fn_callee(bcx, meth::trans_static_method_callee(bcx, impl_did,
                                                                trait_did,
                                                                ref_expr.id))
            }
            ast::DefVariant(tid, vid, _) => {
                // nullary variants are not callable
                assert!(ty::enum_variant_with_id(bcx.tcx(),
                                                      tid,
                                                      vid).args.len() > 0u);
                fn_callee(bcx, trans_fn_ref(bcx, vid, ref_expr.id))
            }
            ast::DefStruct(def_id) => {
                fn_callee(bcx, trans_fn_ref(bcx, def_id, ref_expr.id))
            }
            ast::DefStatic(..) |
            ast::DefArg(..) |
            ast::DefLocal(..) |
            ast::DefBinding(..) |
            ast::DefUpvar(..) |
            ast::DefSelf(..) => {
                datum_callee(bcx, ref_expr)
            }
            ast::DefMod(..) | ast::DefForeignMod(..) | ast::DefTrait(..) |
            ast::DefTy(..) | ast::DefPrimTy(..) |
            ast::DefUse(..) | ast::DefTyParamBinder(..) |
            ast::DefRegion(..) | ast::DefLabel(..) | ast::DefTyParam(..) |
            ast::DefSelfTy(..) | ast::DefMethod(..) => {
                bcx.tcx().sess.span_bug(
                    ref_expr.span,
                    format!("Cannot translate def {:?} \
                          to a callable thing!", def));
            }
        }
    }
}

pub fn trans_fn_ref_to_callee(bcx: @Block,
                              def_id: ast::DefId,
                              ref_id: ast::NodeId) -> Callee {
    Callee {bcx: bcx,
            data: Fn(trans_fn_ref(bcx, def_id, ref_id))}
}

pub fn trans_fn_ref(bcx: @Block,
                    def_id: ast::DefId,
                    ref_id: ast::NodeId) -> FnData {
    /*!
     *
     * Translates a reference (with id `ref_id`) to the fn/method
     * with id `def_id` into a function pointer.  This may require
     * monomorphization or inlining. */

    let _icx = push_ctxt("trans_fn_ref");

    let type_params = node_id_type_params(bcx, ref_id);
    let vtables = node_vtables(bcx, ref_id);
    debug!("trans_fn_ref(def_id={}, ref_id={:?}, type_params={}, vtables={})",
           def_id.repr(bcx.tcx()), ref_id, type_params.repr(bcx.tcx()),
           vtables.repr(bcx.tcx()));
    trans_fn_ref_with_vtables(bcx, def_id, ref_id, type_params, vtables)
}

pub fn trans_fn_ref_with_vtables_to_callee(
        bcx: @Block,
        def_id: ast::DefId,
        ref_id: ast::NodeId,
        type_params: &[ty::t],
        vtables: Option<typeck::vtable_res>)
     -> Callee {
    Callee {bcx: bcx,
            data: Fn(trans_fn_ref_with_vtables(bcx, def_id, ref_id,
                                               type_params, vtables))}
}

fn resolve_default_method_vtables(bcx: @Block,
                                  impl_id: ast::DefId,
                                  method: &ty::Method,
                                  substs: &ty::substs,
                                  impl_vtables: Option<typeck::vtable_res>)
                          -> (typeck::vtable_res, typeck::vtable_param_res) {

    // Get the vtables that the impl implements the trait at
    let impl_res = ty::lookup_impl_vtables(bcx.tcx(), impl_id);

    // Build up a param_substs that we are going to resolve the
    // trait_vtables under.
    let param_substs = Some(@param_substs {
        tys: substs.tps.clone(),
        self_ty: substs.self_ty,
        vtables: impl_vtables,
        self_vtables: None
    });

    let trait_vtables_fixed = resolve_vtables_under_param_substs(
        bcx.tcx(), param_substs, impl_res.trait_vtables);

    // Now we pull any vtables for parameters on the actual method.
    let num_method_vtables = method.generics.type_param_defs.len();
    let method_vtables = match impl_vtables {
        Some(vtables) => {
            let num_impl_type_parameters =
                vtables.len() - num_method_vtables;
            vtables.tailn(num_impl_type_parameters).to_owned()
        },
        None => vec::from_elem(num_method_vtables, @~[])
    };

    let param_vtables = @(*trait_vtables_fixed + method_vtables);

    let self_vtables = resolve_param_vtables_under_param_substs(
        bcx.tcx(), param_substs, impl_res.self_vtables);

    (param_vtables, self_vtables)
}


pub fn trans_fn_ref_with_vtables(
        bcx: @Block,       //
        def_id: ast::DefId,   // def id of fn
        ref_id: ast::NodeId,  // node id of use of fn; may be zero if N/A
        type_params: &[ty::t], // values for fn's ty params
        vtables: Option<typeck::vtable_res>) // vtables for the call
     -> FnData {
    /*!
     * Translates a reference to a fn/method item, monomorphizing and
     * inlining as it goes.
     *
     * # Parameters
     *
     * - `bcx`: the current block where the reference to the fn occurs
     * - `def_id`: def id of the fn or method item being referenced
     * - `ref_id`: node id of the reference to the fn/method, if applicable.
     *   This parameter may be zero; but, if so, the resulting value may not
     *   have the right type, so it must be cast before being used.
     * - `type_params`: values for each of the fn/method's type parameters
     * - `vtables`: values for each bound on each of the type parameters
     */

    let _icx = push_ctxt("trans_fn_ref_with_vtables");
    let ccx = bcx.ccx();
    let tcx = ccx.tcx;

    debug!("trans_fn_ref_with_vtables(bcx={}, def_id={}, ref_id={:?}, \
            type_params={}, vtables={})",
           bcx.to_str(),
           def_id.repr(bcx.tcx()),
           ref_id,
           type_params.repr(bcx.tcx()),
           vtables.repr(bcx.tcx()));

    assert!(type_params.iter().all(|t| !ty::type_needs_infer(*t)));

    // Polytype of the function item (may have type params)
    let fn_tpt = ty::lookup_item_type(tcx, def_id);

    let substs = ty::substs { regions: ty::ErasedRegions,
                              self_ty: None,
                              tps: /*bad*/ type_params.to_owned() };

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
    let (is_default, def_id, substs, self_vtables, vtables) =
        match ty::provided_source(tcx, def_id) {
        None => (false, def_id, substs, None, vtables),
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
                tcx, impl_id, trait_ref, method);

            // And compose them
            let new_substs = first_subst.subst(tcx, &substs);


            let (param_vtables, self_vtables) =
                resolve_default_method_vtables(bcx, impl_id,
                                               method, &substs, vtables);

            debug!("trans_fn_with_vtables - default method: \
                    substs = {}, trait_subst = {}, \
                    first_subst = {}, new_subst = {}, \
                    vtables = {}, \
                    self_vtable = {}, param_vtables = {}",
                   substs.repr(tcx), trait_ref.substs.repr(tcx),
                   first_subst.repr(tcx), new_substs.repr(tcx),
                   vtables.repr(tcx),
                   self_vtables.repr(tcx), param_vtables.repr(tcx));

            (true, source_id,
             new_substs, Some(self_vtables), Some(param_vtables))
        }
    };

    // Check whether this fn has an inlined copy and, if so, redirect
    // def_id to the local id of the inlined copy.
    let def_id = {
        if def_id.crate != ast::LOCAL_CRATE {
            inline::maybe_instantiate_inline(ccx, def_id)
        } else {
            def_id
        }
    };

    // We must monomorphise if the fn has type parameters, is a rust
    // intrinsic, or is a default method.  In particular, if we see an
    // intrinsic that is inlined from a different crate, we want to reemit the
    // intrinsic instead of trying to call it in the other crate.
    let must_monomorphise;
    if type_params.len() > 0 || is_default {
        must_monomorphise = true;
    } else if def_id.crate == ast::LOCAL_CRATE {
        {
            let items = ccx.tcx.items.borrow();
            let map_node = session::expect(
                ccx.sess,
                items.get().find(&def_id.node),
                || format!("local item should be in ast map"));

            match *map_node {
                ast_map::node_foreign_item(_, abis, _, _) => {
                    must_monomorphise = abis.is_intrinsic()
                }
                _ => {
                    must_monomorphise = false;
                }
            }
        }
    } else {
        must_monomorphise = false;
    }

    // Create a monomorphic verison of generic functions
    if must_monomorphise {
        // Should be either intra-crate or inlined.
        assert_eq!(def_id.crate, ast::LOCAL_CRATE);

        let (val, must_cast) =
            monomorphize::monomorphic_fn(ccx, def_id, &substs,
                                         vtables, self_vtables,
                                         Some(ref_id));
        let mut val = val;
        if must_cast && ref_id != 0 {
            // Monotype of the REFERENCE to the function (type params
            // are subst'd)
            let ref_ty = common::node_id_type(bcx, ref_id);

            val = PointerCast(
                bcx, val, type_of::type_of_fn_from_ty(ccx, ref_ty).ptr_to());
        }
        return FnData {llfn: val};
    }

    // Find the actual function pointer.
    let mut val = {
        if def_id.crate == ast::LOCAL_CRATE {
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
        val = BitCast(bcx, val, llptrty);
    }

    return FnData {llfn: val};
}

// ______________________________________________________________________
// Translating calls

pub fn trans_call(in_cx: @Block,
                  call_ex: &ast::Expr,
                  f: &ast::Expr,
                  args: CallArgs,
                  id: ast::NodeId,
                  dest: expr::Dest)
                  -> @Block {
    let _icx = push_ctxt("trans_call");
    trans_call_inner(in_cx,
                     call_ex.info(),
                     expr_ty(in_cx, f),
                     node_id_type(in_cx, id),
                     |cx| trans(cx, f),
                     args,
                     Some(dest),
                     DontAutorefArg).bcx
}

pub fn trans_method_call(in_cx: @Block,
                         call_ex: &ast::Expr,
                         callee_id: ast::NodeId,
                         rcvr: &ast::Expr,
                         args: CallArgs,
                         dest: expr::Dest)
                         -> @Block {
    let _icx = push_ctxt("trans_method_call");
    debug!("trans_method_call(call_ex={}, rcvr={})",
           call_ex.repr(in_cx.tcx()),
           rcvr.repr(in_cx.tcx()));
    trans_call_inner(
        in_cx,
        call_ex.info(),
        node_id_type(in_cx, callee_id),
        expr_ty(in_cx, call_ex),
        |cx| {
            let origin_opt = {
                let mut method_map = cx.ccx().maps.method_map.borrow_mut();
                method_map.get().find_copy(&call_ex.id)
            };
            match origin_opt {
                Some(origin) => {
                    debug!("origin for {}: {}",
                           call_ex.repr(in_cx.tcx()),
                           origin.repr(in_cx.tcx()));

                    meth::trans_method_callee(cx,
                                              callee_id,
                                              rcvr,
                                              origin)
                }
                None => {
                    cx.tcx().sess.span_bug(call_ex.span, "method call expr wasn't in method map")
                }
            }
        },
        args,
        Some(dest),
        DontAutorefArg).bcx
}

pub fn trans_lang_call(bcx: @Block,
                       did: ast::DefId,
                       args: &[ValueRef],
                       dest: Option<expr::Dest>)
    -> Result {
    let fty = if did.crate == ast::LOCAL_CRATE {
        ty::node_id_to_type(bcx.ccx().tcx, did.node)
    } else {
        csearch::get_type(bcx.ccx().tcx, did).ty
    };
    let rty = ty::ty_fn_ret(fty);
    callee::trans_call_inner(bcx,
                             None,
                             fty,
                             rty,
                             |bcx| {
                                trans_fn_ref_with_vtables_to_callee(bcx,
                                                                    did,
                                                                    0,
                                                                    [],
                                                                    None)
                             },
                             ArgVals(args),
                             dest,
                             DontAutorefArg)
}

pub fn trans_lang_call_with_type_params(bcx: @Block,
                                        did: ast::DefId,
                                        args: &[ValueRef],
                                        type_params: &[ty::t],
                                        dest: expr::Dest)
    -> @Block {
    let fty;
    if did.crate == ast::LOCAL_CRATE {
        fty = ty::node_id_to_type(bcx.tcx(), did.node);
    } else {
        fty = csearch::get_type(bcx.tcx(), did).ty;
    }

    let rty = ty::ty_fn_ret(fty);
    return callee::trans_call_inner(
        bcx, None, fty, rty,
        |bcx| {
            let callee =
                trans_fn_ref_with_vtables_to_callee(bcx, did, 0,
                                                    type_params,
                                                    None);

            let new_llval;
            match callee.data {
                Fn(fn_data) => {
                    let substituted = ty::subst_tps(callee.bcx.tcx(),
                                                    type_params,
                                                    None,
                                                    fty);
                    let llfnty = type_of::type_of(callee.bcx.ccx(),
                                                      substituted);
                    new_llval = PointerCast(callee.bcx, fn_data.llfn, llfnty);
                }
                _ => fail!()
            }
            Callee { bcx: callee.bcx, data: Fn(FnData { llfn: new_llval }) }
        },
        ArgVals(args), Some(dest), DontAutorefArg).bcx;
}

pub fn trans_call_inner(in_cx: @Block,
                        call_info: Option<NodeInfo>,
                        callee_ty: ty::t,
                        ret_ty: ty::t,
                        get_callee: |@Block| -> Callee,
                        args: CallArgs,
                        dest: Option<expr::Dest>,
                        autoref_arg: AutorefArg)
                        -> Result {
    /*!
     * This behemoth of a function translates function calls.
     * Unfortunately, in order to generate more efficient LLVM
     * output at -O0, it has quite a complex signature (refactoring
     * this into two functions seems like a good idea).
     *
     * In particular, for lang items, it is invoked with a dest of
     * None, and
     */


    base::with_scope_result(in_cx, call_info, "call", |cx| {
        let callee = get_callee(cx);
        let mut bcx = callee.bcx;
        let ccx = cx.ccx();

        let (llfn, llenv) = unsafe {
            match callee.data {
                Fn(d) => {
                    (d.llfn, llvm::LLVMGetUndef(Type::opaque_box(ccx).ptr_to().to_ref()))
                }
                Method(d) => {
                    // Weird but true: we pass self in the *environment* slot!
                    (d.llfn, d.llself)
                }
                Closure(d) => {
                    // Closures are represented as (llfn, llclosure) pair:
                    // load the requisite values out.
                    let pair = d.to_ref_llval(bcx);
                    let llfn = GEPi(bcx, pair, [0u, abi::fn_field_code]);
                    let llfn = Load(bcx, llfn);
                    let llenv = GEPi(bcx, pair, [0u, abi::fn_field_box]);
                    let llenv = Load(bcx, llenv);
                    (llfn, llenv)
                }
            }
        };

        let abi = match ty::get(callee_ty).sty {
            ty::ty_bare_fn(ref f) => f.abis,
            _ => AbiSet::Rust()
        };
        let is_rust_fn =
            abi.is_rust() ||
            abi.is_intrinsic();

        // Generate a location to store the result. If the user does
        // not care about the result, just make a stack slot.
        let opt_llretslot = match dest {
            None => {
                assert!(!type_of::return_uses_outptr(in_cx.ccx(), ret_ty));
                None
            }
            Some(expr::SaveIn(dst)) => Some(dst),
            Some(expr::Ignore) => {
                if !ty::type_is_voidish(in_cx.tcx(), ret_ty) {
                    Some(alloc_ty(bcx, ret_ty, "__llret"))
                } else {
                    unsafe {
                        Some(llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref()))
                    }
                }
            }
        };

        let mut llresult = unsafe {
            llvm::LLVMGetUndef(Type::nil().ptr_to().to_ref())
        };

        // The code below invokes the function, using either the Rust
        // conventions (if it is a rust fn) or the native conventions
        // (otherwise).  The important part is that, when all is sad
        // and done, either the return value of the function will have been
        // written in opt_llretslot (if it is Some) or `llresult` will be
        // set appropriately (otherwise).
        if is_rust_fn {
            let mut llargs = ~[];

            // Push the out-pointer if we use an out-pointer for this
            // return type, otherwise push "undef".
            if type_of::return_uses_outptr(in_cx.ccx(), ret_ty) {
                llargs.push(opt_llretslot.unwrap());
            }

            // Push the environment.
            llargs.push(llenv);

            // Push the arguments.
            bcx = trans_args(bcx, args, callee_ty,
                             autoref_arg, &mut llargs);

            // Now that the arguments have finished evaluating, we
            // need to revoke the cleanup for the self argument
            match callee.data {
                Method(d) => {
                    for &v in d.temp_cleanup.iter() {
                        revoke_clean(bcx, v);
                    }
                }
                _ => {}
            }

            // A function pointer is called without the declaration available, so we have to apply
            // any attributes with ABI implications directly to the call instruction. Right now, the
            // only attribute we need to worry about is `sret`.
            let mut attrs = ~[];
            if type_of::return_uses_outptr(in_cx.ccx(), ret_ty) {
                attrs.push((1, StructRetAttribute));
            }

            // The `noalias` attribute on the return value is useful to a function ptr caller.
            match ty::get(ret_ty).sty {
                // `~` pointer return values never alias because ownership is transferred
                ty::ty_uniq(..) |
                ty::ty_evec(_, ty::vstore_uniq) => {
                    attrs.push((0, NoAliasAttribute));
                }
                _ => ()
            }

            // Invoke the actual rust fn and update bcx/llresult.
            let (llret, b) = base::invoke(bcx, llfn, llargs, attrs, call_info);
            bcx = b;
            llresult = llret;

            // If the Rust convention for this type is return via
            // the return value, copy it into llretslot.
            match opt_llretslot {
                Some(llretslot) => {
                    if !type_of::return_uses_outptr(bcx.ccx(), ret_ty) &&
                        !ty::type_is_voidish(bcx.tcx(), ret_ty)
                    {
                        Store(bcx, llret, llretslot);
                    }
                }
                None => {}
            }
        } else {
            // Lang items are the only case where dest is None, and
            // they are always Rust fns.
            assert!(dest.is_some());

            let mut llargs = ~[];
            bcx = trans_args(bcx, args, callee_ty,
                             autoref_arg, &mut llargs);
            let arg_tys = match args {
                ArgExprs(a) => a.iter().map(|x| expr_ty(bcx, *x)).collect(),
                ArgVals(_) => fail!("expected arg exprs.")
            };
            bcx = foreign::trans_native_call(bcx, callee_ty,
                                             llfn, opt_llretslot.unwrap(), llargs, arg_tys);
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

        rslt(bcx, llresult)
    })
}

pub enum CallArgs<'a> {
    ArgExprs(&'a [@ast::Expr]),
    ArgVals(&'a [ValueRef])
}

pub fn trans_args(cx: @Block,
                  args: CallArgs,
                  fn_ty: ty::t,
                  autoref_arg: AutorefArg,
                  llargs: &mut ~[ValueRef]) -> @Block
{
    let _icx = push_ctxt("trans_args");
    let mut temp_cleanups = ~[];
    let arg_tys = ty::ty_fn_args(fn_ty);
    let variadic = ty::fn_is_variadic(fn_ty);

    let mut bcx = cx;

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    match args {
      ArgExprs(arg_exprs) => {
        let num_formal_args = arg_tys.len();
        for (i, arg_expr) in arg_exprs.iter().enumerate() {
            let arg_ty = if i >= num_formal_args {
                assert!(variadic);
                expr_ty_adjusted(cx, *arg_expr)
            } else {
                arg_tys[i]
            };
            let arg_val = unpack_result!(bcx, {
                trans_arg_expr(bcx,
                               arg_ty,
                               ty::ByCopy,
                               *arg_expr,
                               &mut temp_cleanups,
                               autoref_arg)
            });
            llargs.push(arg_val);
        }
      }
      ArgVals(vs) => {
        llargs.push_all(vs);
      }
    }

    // now that all arguments have been successfully built, we can revoke any
    // temporary cleanups, as they are only needed if argument construction
    // should fail (for example, cleanup of copy mode args).
    for c in temp_cleanups.iter() {
        revoke_clean(bcx, *c)
    }

    bcx
}

pub enum AutorefArg {
    DontAutorefArg,
    DoAutorefArg
}

// temp_cleanups: cleanups that should run only if failure occurs before the
// call takes place:
pub fn trans_arg_expr(bcx: @Block,
                      formal_arg_ty: ty::t,
                      self_mode: ty::SelfMode,
                      arg_expr: &ast::Expr,
                      temp_cleanups: &mut ~[ValueRef],
                      autoref_arg: AutorefArg) -> Result {
    let _icx = push_ctxt("trans_arg_expr");
    let ccx = bcx.ccx();

    debug!("trans_arg_expr(formal_arg_ty=({}), self_mode={:?}, arg_expr={})",
           formal_arg_ty.repr(bcx.tcx()),
           self_mode,
           arg_expr.repr(bcx.tcx()));

    // translate the arg expr to a datum
    let arg_datumblock = expr::trans_to_datum(bcx, arg_expr);
    let arg_datum = arg_datumblock.datum;
    let bcx = arg_datumblock.bcx;

    debug!("   arg datum: {}", arg_datum.to_str(bcx.ccx()));

    let mut val;
    if ty::type_is_bot(arg_datum.ty) {
        // For values of type _|_, we generate an
        // "undef" value, as such a value should never
        // be inspected. It's important for the value
        // to have type lldestty (the callee's expected type).
        let llformal_arg_ty = type_of::type_of(ccx, formal_arg_ty);
        unsafe {
            val = llvm::LLVMGetUndef(llformal_arg_ty.to_ref());
        }
    } else {
        // FIXME(#3548) use the adjustments table
        match autoref_arg {
            DoAutorefArg => {
                val = arg_datum.to_ref_llval(bcx);
            }
            DontAutorefArg => {
                let need_scratch = ty::type_needs_drop(bcx.tcx(), arg_datum.ty) ||
                    (bcx.expr_is_lval(arg_expr) &&
                     arg_datum.appropriate_mode(bcx.ccx()).is_by_ref());

                let arg_datum = if need_scratch {
                    let scratch = scratch_datum(bcx, arg_datum.ty, "__self", false);
                    arg_datum.store_to_datum(bcx, INIT, scratch);

                    // Technically, ownership of val passes to the callee.
                    // However, we must cleanup should we fail before the
                    // callee is actually invoked.
                    scratch.add_clean(bcx);
                    temp_cleanups.push(scratch.val);

                    scratch
                } else {
                    arg_datum
                };

                val = match self_mode {
                    ty::ByRef => {
                        debug!("by ref arg with type {}", bcx.ty_to_str(arg_datum.ty));
                        arg_datum.to_ref_llval(bcx)
                    }
                    ty::ByCopy => {
                        debug!("by copy arg with type {}", bcx.ty_to_str(arg_datum.ty));
                        arg_datum.to_appropriate_llval(bcx)
                    }
                }
            }
        }

        if formal_arg_ty != arg_datum.ty {
            // this could happen due to e.g. subtyping
            let llformal_arg_ty = type_of::type_of_explicit_arg(ccx, formal_arg_ty);
            debug!("casting actual type ({}) to match formal ({})",
                   bcx.val_to_str(val), bcx.llty_str(llformal_arg_ty));
            val = PointerCast(bcx, val, llformal_arg_ty);
        }
    }

    debug!("--- trans_arg_expr passing {}", bcx.val_to_str(val));
    return rslt(bcx, val);
}
