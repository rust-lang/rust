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

use std::slice;

use back::abi;
use driver::session;
use lib::llvm::{NoAliasAttribute, StructRetAttribute};
use lib::llvm::llvm;
use middle::trans::base;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::common::*;
use middle::trans::datum::*;
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
use middle::typeck::MethodCall;
use util::ppaux::Repr;

use middle::trans::type_::Type;

use std::vec;
use syntax::ast;
use syntax::abi::AbiSet;
use syntax::ast_map;

pub enum Callee {
    Closure(Datum<Lvalue>),

    // Represents a (possibly monomorphized) top-level fn item or method
    // item. Note that this is just the fn-ptr and is not a Rust closure
    // value (which is a pair).
    Fn(Datum<PodValue>),

    // Method type and index for a trait object method call.
    TraitMethod(ty::t, uint)
}

pub fn trans<'a>(bcx: &'a Block<'a>, expr: &ast::Expr) -> (&'a Block<'a>, Callee) {
    let _icx = push_ctxt("trans_callee");
    debug!("callee::trans(expr={})", expr.repr(bcx.tcx()));

    return match expr.node {
        // pick out special kinds of expressions that can be called:
        ast::ExprPath(_) => trans_def(bcx, bcx.def(expr.id), expr),
        // any other expressions are closures:
        _ => datum_callee(bcx, expr)
    };

    fn datum_callee<'a>(bcx: &'a Block<'a>, expr: &ast::Expr) -> (&'a Block<'a>, Callee) {
        let DatumBlock {mut bcx, datum} = expr::trans(bcx, expr);
        match ty::get(datum.ty).sty {
            ty::ty_bare_fn(..) => {
                let fn_ty = datum.ty;
                (bcx, Fn(pod_value(bcx.tcx(), datum.to_llscalarish(bcx), fn_ty)))
            }
            ty::ty_closure(..) => {
                let datum = unpack_datum!(
                    bcx, datum.to_lvalue_datum(bcx, "callee", expr.id));
                (bcx, Closure(datum))
            }
            _ => {
                bcx.tcx().sess.span_bug(
                    expr.span,
                    format!("type of callee is neither bare-fn nor closure: {}",
                         bcx.ty_to_str(datum.ty)));
            }
        }
    }

    fn trans_def<'a>(bcx: &'a Block<'a>, def: ast::Def, ref_expr: &ast::Expr)
                 -> (&'a Block<'a>, Callee) {
        match def {
            ast::DefFn(def_id, _) |
            ast::DefStaticMethod(def_id, ast::FromImpl(_), _) |
            ast::DefStruct(def_id) => {
                (bcx, Fn(trans_fn_ref(bcx, def_id, ExprId(ref_expr.id))))
            }
            ast::DefStaticMethod(impl_did,
                                   ast::FromTrait(trait_did),
                                   _) => {
                (bcx, Fn(meth::trans_static_method_callee(bcx, impl_did,
                                                          trait_did,
                                                          ref_expr.id)))
            }
            ast::DefVariant(tid, vid, _) => {
                // nullary variants are not callable
                assert!(ty::enum_variant_with_id(bcx.tcx(),
                                                      tid,
                                                      vid).args.len() > 0u);
                (bcx, Fn(trans_fn_ref(bcx, vid, ExprId(ref_expr.id))))
            }
            ast::DefStatic(..) |
            ast::DefArg(..) |
            ast::DefLocal(..) |
            ast::DefBinding(..) |
            ast::DefUpvar(..) => {
                datum_callee(bcx, ref_expr)
            }
            ast::DefMod(..) | ast::DefForeignMod(..) | ast::DefTrait(..) |
            ast::DefTy(..) | ast::DefPrimTy(..) |
            ast::DefUse(..) | ast::DefTyParamBinder(..) |
            ast::DefRegion(..) | ast::DefLabel(..) | ast::DefTyParam(..) |
            ast::DefSelfTy(..) | ast::DefMethod(..) => {
                bcx.tcx().sess.span_bug(
                    ref_expr.span,
                    format!("cannot translate def {:?} \
                          to a callable thing!", def));
            }
        }
    }
}

pub fn trans_fn_ref(bcx: &Block, def_id: ast::DefId, node: ExprOrMethodCall) -> Datum<PodValue> {
    /*!
     *
     * Translates a reference (with id `ref_id`) to the fn/method
     * with id `def_id` into a function pointer.  This may require
     * monomorphization or inlining. */

    let _icx = push_ctxt("trans_fn_ref");

    let type_params = node_id_type_params(bcx, node);
    let vtable_key = match node {
        ExprId(id) => MethodCall::expr(id),
        MethodCall(method_call) => method_call
    };
    let vtables = node_vtables(bcx, vtable_key);
    debug!("trans_fn_ref(def_id={}, node={:?}, type_params={}, vtables={})",
           def_id.repr(bcx.tcx()), node, type_params.repr(bcx.tcx()),
           vtables.repr(bcx.tcx()));
    trans_fn_ref_with_vtables(bcx, def_id, node,
                              type_params.as_slice(),
                              vtables)
}

fn resolve_default_method_vtables(bcx: &Block,
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
    let num_method_vtables = method.generics.type_param_defs().len();
    let method_vtables = match impl_vtables {
        Some(vtables) => {
            let num_impl_type_parameters =
                vtables.len() - num_method_vtables;
            vtables.tailn(num_impl_type_parameters).to_owned()
        },
        None => slice::from_elem(num_method_vtables, @Vec::new())
    };

    let param_vtables = @(vec::append((*trait_vtables_fixed).clone(),
                                          method_vtables));

    let self_vtables = resolve_param_vtables_under_param_substs(
        bcx.tcx(), param_substs, impl_res.self_vtables);

    (param_vtables, self_vtables)
}


pub fn trans_fn_ref_with_vtables(
        bcx: &Block,       //
        def_id: ast::DefId,   // def id of fn
        node: ExprOrMethodCall,  // node id of use of fn; may be zero if N/A
        type_params: &[ty::t], // values for fn's ty params
        vtables: Option<typeck::vtable_res>) // vtables for the call
     -> Datum<PodValue> {
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
     * - `type_params`: values for each of the fn/method's type parameters
     * - `vtables`: values for each bound on each of the type parameters
     */

    let _icx = push_ctxt("trans_fn_ref_with_vtables");
    let ccx = bcx.ccx();
    let tcx = bcx.tcx();

    debug!("trans_fn_ref_with_vtables(bcx={}, def_id={}, node={:?}, \
            type_params={}, vtables={})",
           bcx.to_str(),
           def_id.repr(tcx),
           node,
           type_params.repr(tcx),
           vtables.repr(tcx));

    assert!(type_params.iter().all(|t| !ty::type_needs_infer(*t)));

    // Polytype of the function item (may have type params)
    let fn_tpt = ty::lookup_item_type(tcx, def_id);

    let substs = ty::substs { regions: ty::ErasedRegions,
                              self_ty: None,
                              tps: /*bad*/ Vec::from_slice(type_params) };

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
        if def_id.krate != ast::LOCAL_CRATE {
            inline::maybe_instantiate_inline(ccx, def_id)
        } else {
            def_id
        }
    };

    // We must monomorphise if the fn has type parameters, is a rust
    // intrinsic, or is a default method.  In particular, if we see an
    // intrinsic that is inlined from a different crate, we want to reemit the
    // intrinsic instead of trying to call it in the other crate.
    let must_monomorphise = if type_params.len() > 0 || is_default {
        true
    } else if def_id.krate == ast::LOCAL_CRATE {
        let map_node = session::expect(
            ccx.sess(),
            tcx.map.find(def_id.node),
            || format!("local item should be in ast map"));

        match map_node {
            ast_map::NodeForeignItem(_) => {
                tcx.map.get_foreign_abis(def_id.node).is_intrinsic()
            }
            _ => false
        }
    } else {
        false
    };

    // Create a monomorphic verison of generic functions
    if must_monomorphise {
        // Should be either intra-crate or inlined.
        assert_eq!(def_id.krate, ast::LOCAL_CRATE);

        let opt_ref_id = match node {
            ExprId(id) => if id != 0 { Some(id) } else { None },
            MethodCall(_) => None,
        };

        return monomorphize::monomorphic_fn(ccx, def_id, &substs, vtables,
                                            self_vtables, opt_ref_id);
    }

    // Find the actual function pointer.
    let datum = if def_id.krate == ast::LOCAL_CRATE {
        // Internal reference.
        match get_item_val(ccx, def_id.node) {
            LvalueDatum(_) => {
                ccx.sess().bug("found Lvalue instead of PodValue for fn");
            }
            PodValueDatum(datum) => datum
        }
    } else {
        // External reference.
        trans_external_path(ccx, def_id, fn_tpt.ty)
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
    if datum.ty != fn_tpt.ty {
        let llty = type_of::type_of_fn_from_ty(ccx, fn_tpt.ty);
        pod_value(bcx.tcx(), BitCast(bcx, datum.val, llty.ptr_to()), fn_tpt.ty)
    } else {
        datum
    }
}

// ______________________________________________________________________
// Translating calls

pub fn trans_lang_call<'a>(
                       bcx: &'a Block<'a>,
                       did: ast::DefId,
                       args: &[Datum<PodValue>],
                       dest: Option<expr::Dest>)
                       -> Result<'a> {
    trans_call(bcx, None,
               Fn(trans_fn_ref_with_vtables(bcx, did, ExprId(0), [], None)),
               args.iter(), |bcx, arg| DatumBlock(bcx, arg.to_expr_datum()),
               dest)
}

pub fn trans_call<'a, A, I: Iterator<A>>(
        bcx: &'a Block<'a>,
        call_info: Option<NodeInfo>,
        callee: Callee,
        args: I,
        trans_arg: |&'a Block<'a>, A| -> DatumBlock<'a, Expr>,
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
    let ccx = bcx.ccx();
    let mut args = args.enumerate();
    let arg_cleanup_scope = bcx.fcx.push_custom_cleanup_scope();

    let (bcx, callee_ty, llfn, llenv_or_self) = match callee {
        Fn(callee) => {
            (bcx, callee.ty, callee.val, None)
        }
        TraitMethod(method_ty, method_idx) => {
            let (_, self_arg) = args.next().expect("trans_call for TraitMethod missing self");
            let DatumBlock {bcx, datum} = trans_arg(bcx, self_arg);
            let (bcx, llfn, llself) =
                meth::trans_trait_callee(bcx,
                                         method_ty,
                                         method_idx,
                                         datum,
                                         cleanup::CustomScope(arg_cleanup_scope));
            (bcx, method_ty, llfn, Some(llself))
        }
        Closure(closure) => {
            // Closures are represented as (llfn, llclosure) pair:
            // load the requisite values out.
            let pair = closure.to_llref();
            let llfn = GEPi(bcx, pair, [0u, abi::fn_field_code]);
            let llfn = Load(bcx, llfn);
            let llenv = GEPi(bcx, pair, [0u, abi::fn_field_box]);
            let llenv = Load(bcx, llenv);
            (bcx, closure.ty, llfn, Some(llenv))
        }
    };

    let (abi, ret_ty) = match ty::get(callee_ty).sty {
        ty::ty_bare_fn(ref f) => (f.abis, f.sig.output),
        ty::ty_closure(ref f) => (AbiSet::Rust(), f.sig.output),
        _ => fail!("expected bare rust fn or closure in trans_call")
    };
    let is_rust_fn =
        abi.is_rust() ||
        abi.is_intrinsic();

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
                Some(C_undef(type_of::type_of(ccx, ret_ty).ptr_to()))
            }
        }
    };

    // The code below invokes the function, using either the Rust
    // conventions (if it is a rust fn) or the native conventions
    // (otherwise).  The important part is that, when all is sad
    // and done, either the return value of the function will have been
    // written in opt_llretslot (if it is Some) or `llresult` will be
    // set appropriately (otherwise).
    let (mut bcx, llresult) = if is_rust_fn {
        let mut llargs = Vec::new();

        // Push the out-pointer if we use an out-pointer for this
        // return type, otherwise push "undef".
        if type_of::return_uses_outptr(ccx, ret_ty) {
            llargs.push(opt_llretslot.unwrap());
        }

        // Push the environment (or a trait object's self).
        match llenv_or_self {
            Some(llenv_or_self) => llargs.push(llenv_or_self),
            _ => {}
        }

        // Push the arguments.
        let bcx = trans_args(bcx, callee_ty,
                             cleanup::CustomScope(arg_cleanup_scope),
                             args, trans_arg,
                             |arg| {
                                llargs.push(arg.datum.val);
                                arg.bcx
                             });

        bcx.fcx.pop_custom_cleanup_scope(arg_cleanup_scope);

        // A function pointer is called without the declaration
        // available, so we have to apply any attributes with ABI
        // implications directly to the call instruction. Right now,
        // the only attribute we need to worry about is `sret`.
        let mut attrs = Vec::new();
        if type_of::return_uses_outptr(ccx, ret_ty) {
            attrs.push((1, StructRetAttribute));
        }

        // The `noalias` attribute on the return value is useful to a
        // function ptr caller.
        match ty::get(ret_ty).sty {
            // `~` pointer return values never alias because ownership
            // is transferred
            ty::ty_uniq(..) | ty::ty_vec(_, ty::vstore_uniq) => {
                attrs.push((0, NoAliasAttribute));
            }
            _ => {}
        }

        // Invoke the actual rust fn and update bcx/llresult.
        let (llret, bcx) = base::invoke(bcx,
                                        llfn,
                                        llargs,
                                        attrs.as_slice(),
                                        call_info);

        // If the Rust convention for this type is return via
        // the return value, copy it into llretslot.
        match opt_llretslot {
            Some(llretslot) => {
                if !type_of::return_uses_outptr(bcx.ccx(), ret_ty) &&
                    !type_is_zero_size(bcx.ccx(), ret_ty)
                {
                    Store(bcx, llret, llretslot);
                }
            }
            None => {}
        }
        (bcx, llret)
    } else {
        // Lang items are the only case where dest is None, and
        // they are always Rust fns.
        assert!(dest.is_some());

        // Also, this can't be a trait object or closure call.
        assert!(llenv_or_self.is_none());

        let mut arg_datums = Vec::new();
        let bcx = trans_args(bcx, callee_ty,
                             cleanup::CustomScope(arg_cleanup_scope),
                             args, trans_arg,
                             |DatumBlock {bcx, datum}| {
                                arg_datums.push(datum);
                                bcx
                             });
        bcx.fcx.pop_custom_cleanup_scope(arg_cleanup_scope);
        let bcx = foreign::trans_native_call(bcx, callee_ty, llfn,
                                             opt_llretslot.unwrap(),
                                             arg_datums);
        let llresult = unsafe {
            llvm::LLVMGetUndef(Type::nil(ccx).ptr_to().to_ref())
        };
        (bcx, llresult)
    };

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
}

// This would be much nicer if it could be an iterator.
fn trans_args<'a, A, I: Iterator<(uint, A)>>(
        bcx: &'a Block<'a>,
        fn_ty: ty::t,
        arg_cleanup_scope: cleanup::ScopeId,
        mut args: I,
        trans_arg: |&'a Block<'a>, A| -> DatumBlock<'a, Expr>,
        each_arg: |DatumBlock<'a, Rvalue>| -> &'a Block<'a>)
        -> &'a Block<'a> {
    let _icx = push_ctxt("trans_args");

    // Should borrow instead of cloning the arg_tys vector.
    let arg_tys = ty::ty_fn_args(fn_ty);
    let variadic = ty::fn_is_variadic(fn_ty);

    let mut bcx = bcx;

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    for (i, arg) in args {
        let arg = trans_arg(bcx, arg);
        let arg_ty = if i >= arg_tys.len() {
            assert!(variadic);
            arg.datum.ty
        } else {
            *arg_tys.get(i)
        };

        bcx = each_arg(trans_arg_datum(arg, arg_ty, arg_cleanup_scope));
    }

    bcx
}

pub fn trans_arg_datum<'a>(arg: DatumBlock<'a, Expr>,
                           formal_arg_ty: ty::t,
                           arg_cleanup_scope: cleanup::ScopeId)
                           -> DatumBlock<'a, Rvalue> {
    let _icx = push_ctxt("trans_arg_datum");
    let ccx = arg.ccx();

    debug!("trans_arg_datum({})",
           formal_arg_ty.repr(arg.tcx()));

    let arg_datum_ty = arg.datum.ty;

    debug!("   arg datum: {}", arg.to_str());

    let (bcx, val, mode) = if ty::type_is_bot(arg_datum_ty) {
        // For values of type _|_, we generate an
        // "undef" value, as such a value should never
        // be inspected. It's important for the value
        // to have type lldestty (the callee's expected type).
        let llformal_arg_ty = type_of::type_of(ccx, formal_arg_ty).to_ref();
        unsafe {
            (arg.bcx, llvm::LLVMGetUndef(llformal_arg_ty), ByValue)
        }
    } else {
        // Make this an rvalue, since we are going to be
        // passing ownership.
        let arg = arg.to_rvalue_datumblock("arg");

        // Now that the argument datum is owned, get it into
        // the appropriate mode (ref vs value).
        let arg = arg.to_appropriate_datumblock();


        // Technically, ownership of val passes to the callee.
        // However, we must cleanup should we fail before the
        // callee is actually invoked.
        let mode = arg.datum.kind.mode;
        let bcx = arg.bcx;
        let val = arg.datum.add_clean(bcx.fcx, arg_cleanup_scope);

        if formal_arg_ty != arg_datum_ty {
            // this could happen due to e.g. subtyping
            let llformal_arg_ty = type_of::type_of_explicit_arg(ccx, formal_arg_ty);
            debug!("casting actual type ({}) to match formal ({})",
                   bcx.val_to_str(val), bcx.llty_str(llformal_arg_ty));
            (bcx, PointerCast(bcx, val, llformal_arg_ty), mode)
        } else {
            (bcx, val, mode)
        }
    };

    let datum = Datum(val, formal_arg_ty, Rvalue(mode));

    debug!("--- trans_arg_datum passing {}", datum.to_str(ccx));
    DatumBlock(bcx, datum)
}
