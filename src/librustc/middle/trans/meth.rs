// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::abi;
use llvm;
use llvm::ValueRef;
use metadata::csearch;
use middle::subst::VecPerParamSpace;
use middle::subst;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee::*;
use middle::trans::callee;
use middle::trans::cleanup;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::expr::{SaveIn, Ignore};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::monomorphize;
use middle::trans::type_::Type;
use middle::trans::type_of::*;
use middle::ty;
use middle::typeck;
use middle::typeck::MethodCall;
use util::common::indenter;
use util::ppaux::Repr;

use std::c_str::ToCStr;
use std::gc::Gc;
use syntax::abi::{Rust, RustCall};
use syntax::parse::token;
use syntax::{ast, ast_map, visit};
use syntax::ast_util::PostExpansionMethod;

/**
The main "translation" pass for methods.  Generates code
for non-monomorphized methods only.  Other methods will
be generated once they are invoked with specific type parameters,
see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
*/
pub fn trans_impl(ccx: &CrateContext,
                  name: ast::Ident,
                  methods: &[Gc<ast::Method>],
                  generics: &ast::Generics,
                  id: ast::NodeId) {
    let _icx = push_ctxt("meth::trans_impl");
    let tcx = ccx.tcx();

    debug!("trans_impl(name={}, id={:?})", name.repr(tcx), id);

    // Both here and below with generic methods, be sure to recurse and look for
    // items that we need to translate.
    if !generics.ty_params.is_empty() {
        let mut v = TransItemVisitor{ ccx: ccx };
        for method in methods.iter() {
            visit::walk_method_helper(&mut v, &**method, ());
        }
        return;
    }
    for method in methods.iter() {
        if method.pe_generics().ty_params.len() == 0u {
            let llfn = get_item_val(ccx, method.id);
            trans_fn(ccx,
                     &*method.pe_fn_decl(),
                     &*method.pe_body(),
                     llfn,
                     &param_substs::empty(),
                     method.id,
                     [],
                     TranslateItems);
        } else {
            let mut v = TransItemVisitor{ ccx: ccx };
            visit::walk_method_helper(&mut v, &**method, ());
        }
    }
}

pub fn trans_method_callee<'a>(
                           bcx: &'a Block<'a>,
                           method_call: MethodCall,
                           self_expr: Option<&ast::Expr>,
                           arg_cleanup_scope: cleanup::ScopeId)
                           -> Callee<'a> {
    let _icx = push_ctxt("meth::trans_method_callee");

    let (origin, method_ty) = match bcx.tcx().method_map
                                       .borrow().find(&method_call) {
        Some(method) => {
            debug!("trans_method_callee({:?}, method={})",
                   method_call, method.repr(bcx.tcx()));
            (method.origin, method.ty)
        }
        None => {
            bcx.sess().span_bug(bcx.tcx().map.span(method_call.expr_id),
                                "method call expr wasn't in method map")
        }
    };

    match origin {
        typeck::MethodStatic(did) |
        typeck::MethodStaticUnboxedClosure(did) => {
            Callee {
                bcx: bcx,
                data: Fn(callee::trans_fn_ref(bcx,
                                              did,
                                              MethodCall(method_call))),
            }
        }
        typeck::MethodParam(typeck::MethodParam {
            trait_id: trait_id,
            method_num: off,
            param_num: p,
            bound_num: b
        }) => {
            ty::populate_implementations_for_trait_if_necessary(
                bcx.tcx(),
                trait_id);

            let vtbl = find_vtable(bcx.tcx(), bcx.fcx.param_substs, p, b);
            trans_monomorphized_callee(bcx, method_call,
                                       trait_id, off, vtbl)
        }

        typeck::MethodObject(ref mt) => {
            let self_expr = match self_expr {
                Some(self_expr) => self_expr,
                None => {
                    bcx.sess().span_bug(bcx.tcx().map.span(method_call.expr_id),
                                        "self expr wasn't provided for trait object \
                                         callee (trying to call overloaded op?)")
                }
            };
            trans_trait_callee(bcx,
                               monomorphize_type(bcx, method_ty),
                               mt.real_index,
                               self_expr,
                               arg_cleanup_scope)
        }
    }
}

pub fn trans_static_method_callee(bcx: &Block,
                                  method_id: ast::DefId,
                                  trait_id: ast::DefId,
                                  expr_id: ast::NodeId)
                                  -> ValueRef {
    let _icx = push_ctxt("meth::trans_static_method_callee");
    let ccx = bcx.ccx();

    debug!("trans_static_method_callee(method_id={:?}, trait_id={}, \
            expr_id={:?})",
           method_id,
           ty::item_path_str(bcx.tcx(), trait_id),
           expr_id);
    let _indenter = indenter();

    ty::populate_implementations_for_trait_if_necessary(bcx.tcx(), trait_id);

    let mname = if method_id.krate == ast::LOCAL_CRATE {
        match bcx.tcx().map.get(method_id.node) {
            ast_map::NodeTraitMethod(method) => {
                let ident = match *method {
                    ast::Required(ref m) => m.ident,
                    ast::Provided(ref m) => m.pe_ident()
                };
                ident.name
            }
            _ => fail!("callee is not a trait method")
        }
    } else {
        csearch::get_item_path(bcx.tcx(), method_id).last().unwrap().name()
    };
    debug!("trans_static_method_callee: method_id={:?}, expr_id={:?}, \
            name={}", method_id, expr_id, token::get_name(mname));

    let vtable_key = MethodCall::expr(expr_id);
    let vtbls = resolve_vtables_in_fn_ctxt(
        bcx.fcx,
        ccx.tcx.vtable_map.borrow().get(&vtable_key));

    match *vtbls.get_self().unwrap().get(0) {
        typeck::vtable_static(impl_did, ref rcvr_substs, ref rcvr_origins) => {
            assert!(rcvr_substs.types.all(|t| !ty::type_needs_infer(*t)));

            let mth_id = method_with_name(ccx, impl_did, mname);
            let (callee_substs, callee_origins) =
                combine_impl_and_methods_tps(
                    bcx, ExprId(expr_id),
                    (*rcvr_substs).clone(), (*rcvr_origins).clone());

            let llfn = trans_fn_ref_with_vtables(bcx, mth_id, ExprId(expr_id),
                                                 callee_substs,
                                                 callee_origins);

            let callee_ty = node_id_type(bcx, expr_id);
            let llty = type_of_fn_from_ty(ccx, callee_ty).ptr_to();
            PointerCast(bcx, llfn, llty)
        }
        typeck::vtable_unboxed_closure(_) => {
            bcx.tcx().sess.bug("can't call a closure vtable in a static way");
        }
        _ => {
            fail!("vtable_param left in monomorphized \
                   function's vtable substs");
        }
    }
}

fn method_with_name(ccx: &CrateContext,
                    impl_id: ast::DefId,
                    name: ast::Name) -> ast::DefId {
    match ccx.impl_method_cache.borrow().find_copy(&(impl_id, name)) {
        Some(m) => return m,
        None => {}
    }

    let methods = ccx.tcx.impl_methods.borrow();
    let methods = methods.find(&impl_id)
                         .expect("could not find impl while translating");
    let meth_did = methods.iter().find(|&did| ty::method(&ccx.tcx, *did).ident.name == name)
                                 .expect("could not find method while translating");

    ccx.impl_method_cache.borrow_mut().insert((impl_id, name), *meth_did);
    *meth_did
}

fn trans_monomorphized_callee<'a>(
                              bcx: &'a Block<'a>,
                              method_call: MethodCall,
                              trait_id: ast::DefId,
                              n_method: uint,
                              vtbl: typeck::vtable_origin)
                              -> Callee<'a> {
    let _icx = push_ctxt("meth::trans_monomorphized_callee");
    match vtbl {
      typeck::vtable_static(impl_did, rcvr_substs, rcvr_origins) => {
          let ccx = bcx.ccx();
          let mname = ty::trait_method(ccx.tcx(), trait_id, n_method).ident;
          let mth_id = method_with_name(bcx.ccx(), impl_did, mname.name);

          // create a concatenated set of substitutions which includes
          // those from the impl and those from the method:
          let (callee_substs, callee_origins) =
              combine_impl_and_methods_tps(
                  bcx, MethodCall(method_call), rcvr_substs, rcvr_origins);

          // translate the function
          let llfn = trans_fn_ref_with_vtables(bcx,
                                               mth_id,
                                               MethodCall(method_call),
                                               callee_substs,
                                               callee_origins);

          Callee { bcx: bcx, data: Fn(llfn) }
      }
      typeck::vtable_unboxed_closure(closure_def_id) => {
          // The static region and type parameters are lies, but we're in
          // trans so it doesn't matter.
          //
          // FIXME(pcwalton): Is this true in the case of type parameters?
          let callee_substs = get_callee_substitutions_for_unboxed_closure(
                bcx,
                closure_def_id);

          let llfn = trans_fn_ref_with_vtables(bcx,
                                               closure_def_id,
                                               MethodCall(method_call),
                                               callee_substs,
                                               VecPerParamSpace::empty());

          Callee {
              bcx: bcx,
              data: Fn(llfn),
          }
      }
      typeck::vtable_param(..) => {
          bcx.tcx().sess.bug(
              "vtable_param left in monomorphized function's vtable substs");
      }
      typeck::vtable_error => {
          bcx.tcx().sess.bug(
              "vtable_error left in monomorphized function's vtable substs");
      }
    }
}

fn combine_impl_and_methods_tps(bcx: &Block,
                                node: ExprOrMethodCall,
                                rcvr_substs: subst::Substs,
                                rcvr_origins: typeck::vtable_res)
                                -> (subst::Substs, typeck::vtable_res)
{
    /*!
     * Creates a concatenated set of substitutions which includes
     * those from the impl and those from the method.  This are
     * some subtle complications here.  Statically, we have a list
     * of type parameters like `[T0, T1, T2, M1, M2, M3]` where
     * `Tn` are type parameters that appear on the receiver.  For
     * example, if the receiver is a method parameter `A` with a
     * bound like `trait<B,C,D>` then `Tn` would be `[B,C,D]`.
     *
     * The weird part is that the type `A` might now be bound to
     * any other type, such as `foo<X>`.  In that case, the vector
     * we want is: `[X, M1, M2, M3]`.  Therefore, what we do now is
     * to slice off the method type parameters and append them to
     * the type parameters from the type that the receiver is
     * mapped to.
     */

    let ccx = bcx.ccx();

    let vtable_key = match node {
        ExprId(id) => MethodCall::expr(id),
        MethodCall(method_call) => method_call
    };
    let node_substs = node_id_substs(bcx, node);
    let node_vtables = node_vtables(bcx, vtable_key);

    debug!("rcvr_substs={:?}", rcvr_substs.repr(ccx.tcx()));
    debug!("node_substs={:?}", node_substs.repr(ccx.tcx()));

    // Break apart the type parameters from the node and type
    // parameters from the receiver.
    let (_, _, node_method) = node_substs.types.split();
    let (rcvr_type, rcvr_self, rcvr_method) = rcvr_substs.types.clone().split();
    assert!(rcvr_method.is_empty());
    let ty_substs = subst::Substs {
        regions: subst::ErasedRegions,
        types: subst::VecPerParamSpace::new(rcvr_type, rcvr_self, node_method)
    };

    // Now do the same work for the vtables.
    let (rcvr_type, rcvr_self, rcvr_method) = rcvr_origins.split();
    let (_, _, node_method) = node_vtables.split();
    assert!(rcvr_method.is_empty());
    let vtables = subst::VecPerParamSpace::new(rcvr_type, rcvr_self, node_method);

    (ty_substs, vtables)
}

fn trans_trait_callee<'a>(bcx: &'a Block<'a>,
                          method_ty: ty::t,
                          n_method: uint,
                          self_expr: &ast::Expr,
                          arg_cleanup_scope: cleanup::ScopeId)
                          -> Callee<'a> {
    /*!
     * Create a method callee where the method is coming from a trait
     * object (e.g., Box<Trait> type).  In this case, we must pull the fn
     * pointer out of the vtable that is packaged up with the object.
     * Objects are represented as a pair, so we first evaluate the self
     * expression and then extract the self data and vtable out of the
     * pair.
     */

    let _icx = push_ctxt("meth::trans_trait_callee");
    let mut bcx = bcx;

    // Translate self_datum and take ownership of the value by
    // converting to an rvalue.
    let self_datum = unpack_datum!(
        bcx, expr::trans(bcx, self_expr));

    let llval = if ty::type_needs_drop(bcx.tcx(), self_datum.ty) {
        let self_datum = unpack_datum!(
            bcx, self_datum.to_rvalue_datum(bcx, "trait_callee"));

        // Convert to by-ref since `trans_trait_callee_from_llval` wants it
        // that way.
        let self_datum = unpack_datum!(
            bcx, self_datum.to_ref_datum(bcx));

        // Arrange cleanup in case something should go wrong before the
        // actual call occurs.
        self_datum.add_clean(bcx.fcx, arg_cleanup_scope)
    } else {
        // We don't have to do anything about cleanups for &Trait and &mut Trait.
        assert!(self_datum.kind.is_by_ref());
        self_datum.val
    };

    trans_trait_callee_from_llval(bcx, method_ty, n_method, llval)
}

pub fn trans_trait_callee_from_llval<'a>(bcx: &'a Block<'a>,
                                         callee_ty: ty::t,
                                         n_method: uint,
                                         llpair: ValueRef)
                                         -> Callee<'a> {
    /*!
     * Same as `trans_trait_callee()` above, except that it is given
     * a by-ref pointer to the object pair.
     */

    let _icx = push_ctxt("meth::trans_trait_callee");
    let ccx = bcx.ccx();

    // Load the data pointer from the object.
    debug!("(translating trait callee) loading second index from pair");
    let llboxptr = GEPi(bcx, llpair, [0u, abi::trt_field_box]);
    let llbox = Load(bcx, llboxptr);
    let llself = PointerCast(bcx, llbox, Type::i8p(ccx));

    // Load the function from the vtable and cast it to the expected type.
    debug!("(translating trait callee) loading method");
    // Replace the self type (&Self or Box<Self>) with an opaque pointer.
    let llcallee_ty = match ty::get(callee_ty).sty {
        ty::ty_bare_fn(ref f) if f.abi == Rust || f.abi == RustCall => {
            type_of_rust_fn(ccx,
                            Some(Type::i8p(ccx)),
                            f.sig.inputs.slice_from(1),
                            f.sig.output,
                            f.abi)
        }
        _ => {
            ccx.sess().bug("meth::trans_trait_callee given non-bare-rust-fn");
        }
    };
    let llvtable = Load(bcx,
                        PointerCast(bcx,
                                    GEPi(bcx, llpair,
                                         [0u, abi::trt_field_vtable]),
                                    Type::vtable(ccx).ptr_to().ptr_to()));
    let mptr = Load(bcx, GEPi(bcx, llvtable, [0u, n_method + 1]));
    let mptr = PointerCast(bcx, mptr, llcallee_ty.ptr_to());

    return Callee {
        bcx: bcx,
        data: TraitMethod(MethodData {
            llfn: mptr,
            llself: llself,
        })
    };
}

/// Creates the self type and (fake) callee substitutions for an unboxed
/// closure with the given def ID. The static region and type parameters are
/// lies, but we're in trans so it doesn't matter.
fn get_callee_substitutions_for_unboxed_closure(bcx: &Block,
                                                def_id: ast::DefId)
                                                -> subst::Substs {
    let self_ty = ty::mk_unboxed_closure(bcx.tcx(), def_id);
    subst::Substs::erased(
        VecPerParamSpace::new(Vec::new(),
                              vec![
                                  ty::mk_rptr(bcx.tcx(),
                                              ty::ReStatic,
                                              ty::mt {
                                                ty: self_ty,
                                                mutbl: ast::MutMutable,
                                              })
                              ],
                              Vec::new()))
}

/// Creates a returns a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
fn get_vtable(bcx: &Block,
              self_ty: ty::t,
              origins: typeck::vtable_param_res)
              -> ValueRef
{
    debug!("get_vtable(self_ty={}, origins={})",
           self_ty.repr(bcx.tcx()),
           origins.repr(bcx.tcx()));

    let ccx = bcx.ccx();
    let _icx = push_ctxt("meth::get_vtable");

    // Check the cache.
    let hash_id = (self_ty, monomorphize::make_vtable_id(ccx, origins.get(0)));
    match ccx.vtables.borrow().find(&hash_id) {
        Some(&val) => { return val }
        None => { }
    }

    // Not in the cache. Actually build it.
    let methods = origins.move_iter().flat_map(|origin| {
        match origin {
            typeck::vtable_static(id, substs, sub_vtables) => {
                emit_vtable_methods(bcx, id, substs, sub_vtables).move_iter()
            }
            typeck::vtable_unboxed_closure(closure_def_id) => {
                let callee_substs =
                    get_callee_substitutions_for_unboxed_closure(
                        bcx,
                        closure_def_id);

                let llfn = trans_fn_ref_with_vtables(
                    bcx,
                    closure_def_id,
                    ExprId(0),
                    callee_substs,
                    VecPerParamSpace::empty());

                (vec!(llfn)).move_iter()
            }
            _ => ccx.sess().bug("get_vtable: expected a static origin"),
        }
    });

    // Generate a destructor for the vtable.
    let drop_glue = glue::get_drop_glue(ccx, self_ty);
    let vtable = make_vtable(ccx, drop_glue, methods);

    ccx.vtables.borrow_mut().insert(hash_id, vtable);
    vtable
}

/// Helper function to declare and initialize the vtable.
pub fn make_vtable<I: Iterator<ValueRef>>(ccx: &CrateContext,
                                          drop_glue: ValueRef,
                                          ptrs: I)
                                          -> ValueRef {
    let _icx = push_ctxt("meth::make_vtable");

    let components: Vec<_> = Some(drop_glue).move_iter().chain(ptrs).collect();

    unsafe {
        let tbl = C_struct(ccx, components.as_slice(), false);
        let sym = token::gensym("vtable");
        let vt_gvar = format!("vtable{}", sym.uint()).with_c_str(|buf| {
            llvm::LLVMAddGlobal(ccx.llmod, val_ty(tbl).to_ref(), buf)
        });
        llvm::LLVMSetInitializer(vt_gvar, tbl);
        llvm::LLVMSetGlobalConstant(vt_gvar, llvm::True);
        llvm::SetLinkage(vt_gvar, llvm::InternalLinkage);
        vt_gvar
    }
}

fn emit_vtable_methods(bcx: &Block,
                       impl_id: ast::DefId,
                       substs: subst::Substs,
                       vtables: typeck::vtable_res)
                       -> Vec<ValueRef> {
    let ccx = bcx.ccx();
    let tcx = ccx.tcx();

    let trt_id = match ty::impl_trait_ref(tcx, impl_id) {
        Some(t_id) => t_id.def_id,
        None       => ccx.sess().bug("make_impl_vtable: don't know how to \
                                      make a vtable for a type impl!")
    };

    ty::populate_implementations_for_trait_if_necessary(bcx.tcx(), trt_id);

    let trait_method_def_ids = ty::trait_method_def_ids(tcx, trt_id);
    trait_method_def_ids.iter().map(|method_def_id| {
        let ident = ty::method(tcx, *method_def_id).ident;
        // The substitutions we have are on the impl, so we grab
        // the method type from the impl to substitute into.
        let m_id = method_with_name(ccx, impl_id, ident.name);
        let m = ty::method(tcx, m_id);
        debug!("(making impl vtable) emitting method {} at subst {}",
               m.repr(tcx),
               substs.repr(tcx));
        if m.generics.has_type_params(subst::FnSpace) ||
           ty::type_has_self(ty::mk_bare_fn(tcx, m.fty.clone())) {
            debug!("(making impl vtable) method has self or type params: {}",
                   token::get_ident(ident));
            C_null(Type::nil(ccx).ptr_to())
        } else {
            let mut fn_ref = trans_fn_ref_with_vtables(bcx,
                                                       m_id,
                                                       ExprId(0),
                                                       substs.clone(),
                                                       vtables.clone());
            if m.explicit_self == ty::ByValueExplicitSelfCategory {
                fn_ref = trans_unboxing_shim(bcx,
                                             fn_ref,
                                             &*m,
                                             m_id,
                                             substs.clone());
            }
            fn_ref
        }
    }).collect()
}

pub fn trans_trait_cast<'a>(bcx: &'a Block<'a>,
                            datum: Datum<Expr>,
                            id: ast::NodeId,
                            dest: expr::Dest)
                            -> &'a Block<'a> {
    /*!
     * Generates the code to convert from a pointer (`Box<T>`, `&T`, etc)
     * into an object (`Box<Trait>`, `&Trait`, etc). This means creating a
     * pair where the first word is the vtable and the second word is
     * the pointer.
     */

    let mut bcx = bcx;
    let _icx = push_ctxt("meth::trans_cast");

    let lldest = match dest {
        Ignore => {
            return datum.clean(bcx, "trait_cast", id);
        }
        SaveIn(dest) => dest
    };

    let ccx = bcx.ccx();
    let v_ty = datum.ty;
    let llbox_ty = type_of(bcx.ccx(), datum.ty);

    // Store the pointer into the first half of pair.
    let mut llboxdest = GEPi(bcx, lldest, [0u, abi::trt_field_box]);
    llboxdest = PointerCast(bcx, llboxdest, llbox_ty.ptr_to());
    bcx = datum.store_to(bcx, llboxdest);

    // Store the vtable into the second half of pair.
    let origins = {
        let vtable_map = ccx.tcx.vtable_map.borrow();
        // This trait cast might be because of implicit coercion
        let method_call = match ccx.tcx.adjustments.borrow().find(&id) {
            Some(&ty::AutoObject(..)) => MethodCall::autoobject(id),
            _ => MethodCall::expr(id)
        };
        let vres = vtable_map.get(&method_call).get_self().unwrap();
        resolve_param_vtables_under_param_substs(ccx.tcx(), bcx.fcx.param_substs, vres)
    };
    let vtable = get_vtable(bcx, v_ty, origins);
    let llvtabledest = GEPi(bcx, lldest, [0u, abi::trt_field_vtable]);
    let llvtabledest = PointerCast(bcx, llvtabledest, val_ty(vtable).ptr_to());
    Store(bcx, vtable, llvtabledest);

    bcx
}
