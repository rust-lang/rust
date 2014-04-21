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
use lib::llvm::llvm;
use lib::llvm::ValueRef;
use lib;
use metadata::csearch;
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
use syntax::abi::Rust;
use syntax::parse::token;
use syntax::{ast, ast_map, visit};

/**
The main "translation" pass for methods.  Generates code
for non-monomorphized methods only.  Other methods will
be generated once they are invoked with specific type parameters,
see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
*/
pub fn trans_impl(ccx: &CrateContext,
                  name: ast::Ident,
                  methods: &[@ast::Method],
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
            visit::walk_method_helper(&mut v, *method, ());
        }
        return;
    }
    for method in methods.iter() {
        if method.generics.ty_params.len() == 0u {
            let llfn = get_item_val(ccx, method.id);
            trans_fn(ccx, method.decl, method.body,
                     llfn, None, method.id, []);
        } else {
            let mut v = TransItemVisitor{ ccx: ccx };
            visit::walk_method_helper(&mut v, *method, ());
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
        typeck::MethodStatic(did) => {
            Callee {
                bcx: bcx,
                data: Fn(callee::trans_fn_ref(bcx, did, MethodCall(method_call)))
            }
        }
        typeck::MethodParam(typeck::MethodParam {
            trait_id: trait_id,
            method_num: off,
            param_num: p,
            bound_num: b
        }) => {
            match bcx.fcx.param_substs {
                Some(substs) => {
                    ty::populate_implementations_for_trait_if_necessary(
                        bcx.tcx(),
                        trait_id);

                    let vtbl = find_vtable(bcx.tcx(), substs, p, b);
                    trans_monomorphized_callee(bcx, method_call,
                                               trait_id, off, vtbl)
                }
                // how to get rid of this?
                None => fail!("trans_method_callee: missing param_substs")
            }
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

    // When we translate a static fn defined in a trait like:
    //
    //   trait<T1...Tn> Trait {
    //       fn foo<M1...Mn>(...) {...}
    //   }
    //
    // this winds up being translated as something like:
    //
    //   fn foo<T1...Tn,self: Trait<T1...Tn>,M1...Mn>(...) {...}
    //
    // So when we see a call to this function foo, we have to figure
    // out which impl the `Trait<T1...Tn>` bound on the type `self` was
    // bound to.
    let bound_index = ty::lookup_trait_def(bcx.tcx(), trait_id).
        generics.type_param_defs().len();

    let mname = if method_id.krate == ast::LOCAL_CRATE {
        match bcx.tcx().map.get(method_id.node) {
            ast_map::NodeTraitMethod(method) => {
                let ident = match *method {
                    ast::Required(ref m) => m.ident,
                    ast::Provided(ref m) => m.ident
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
    let vtbls = resolve_vtables_in_fn_ctxt(bcx.fcx, ccx.tcx.vtable_map.borrow()
                                                       .get(&vtable_key).as_slice());

    match vtbls.move_iter().nth(bound_index).unwrap().move_iter().nth(0).unwrap() {
        typeck::vtable_static(impl_did, rcvr_substs, rcvr_origins) => {
            assert!(rcvr_substs.iter().all(|t| !ty::type_needs_infer(*t)));

            let mth_id = method_with_name(ccx, impl_did, mname);
            let (callee_substs, callee_origins) =
                combine_impl_and_methods_tps(
                    bcx, mth_id, ExprId(expr_id),
                    rcvr_substs, rcvr_origins);

            let llfn = trans_fn_ref_with_vtables(bcx, mth_id, ExprId(expr_id),
                                                 callee_substs,
                                                 Some(callee_origins));

            let callee_ty = node_id_type(bcx, expr_id);
            let llty = type_of_fn_from_ty(ccx, callee_ty).ptr_to();
            PointerCast(bcx, llfn, llty)
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

fn trans_monomorphized_callee<'a>(bcx: &'a Block<'a>,
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
                  bcx, mth_id,  MethodCall(method_call),
                  rcvr_substs, rcvr_origins);

          // translate the function
          let llfn = trans_fn_ref_with_vtables(bcx,
                                               mth_id,
                                               MethodCall(method_call),
                                               callee_substs,
                                               Some(callee_origins));

          Callee { bcx: bcx, data: Fn(llfn) }
      }
      typeck::vtable_param(..) => {
          fail!("vtable_param left in monomorphized function's vtable substs");
      }
    }
}

fn combine_impl_and_methods_tps(bcx: &Block,
                                mth_did: ast::DefId,
                                node: ExprOrMethodCall,
                                rcvr_substs: Vec<ty::t>,
                                rcvr_origins: typeck::vtable_res)
                                -> (Vec<ty::t>, typeck::vtable_res) {
    /*!
    *
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
    * mapped to. */

    let ccx = bcx.ccx();
    let method = ty::method(ccx.tcx(), mth_did);
    let n_m_tps = method.generics.type_param_defs().len();
    let node_substs = node_id_type_params(bcx, node);
    debug!("rcvr_substs={:?}", rcvr_substs.repr(ccx.tcx()));
    debug!("node_substs={:?}", node_substs.repr(ccx.tcx()));
    let mut ty_substs = rcvr_substs;
    {
        let start = node_substs.len() - n_m_tps;
        ty_substs.extend(node_substs.move_iter().skip(start));
    }
    debug!("n_m_tps={:?}", n_m_tps);
    debug!("ty_substs={:?}", ty_substs.repr(ccx.tcx()));


    // Now, do the same work for the vtables.  The vtables might not
    // exist, in which case we need to make them.
    let vtable_key = match node {
        ExprId(id) => MethodCall::expr(id),
        MethodCall(method_call) => method_call
    };
    let mut vtables = rcvr_origins;
    match node_vtables(bcx, vtable_key) {
        Some(vt) => {
            let start = vt.len() - n_m_tps;
            vtables.extend(vt.move_iter().skip(start));
        }
        None => {
            vtables.extend(range(0, n_m_tps).map(
                |_| -> typeck::vtable_param_res {
                    Vec::new()
                }
            ));
        }
    }

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
     * object (e.g., ~Trait type).  In this case, we must pull the fn
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
    // Replace the self type (&Self or ~Self) with an opaque pointer.
    let llcallee_ty = match ty::get(callee_ty).sty {
        ty::ty_bare_fn(ref f) if f.abi == Rust => {
            type_of_rust_fn(ccx, true, f.sig.inputs.slice_from(1), f.sig.output)
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

/// Creates a returns a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
fn get_vtable(bcx: &Block,
              self_ty: ty::t,
              origins: typeck::vtable_param_res)
              -> ValueRef {
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
        let vt_gvar = format!("vtable{}", sym).with_c_str(|buf| {
            llvm::LLVMAddGlobal(ccx.llmod, val_ty(tbl).to_ref(), buf)
        });
        llvm::LLVMSetInitializer(vt_gvar, tbl);
        llvm::LLVMSetGlobalConstant(vt_gvar, lib::llvm::True);
        lib::llvm::SetLinkage(vt_gvar, lib::llvm::InternalLinkage);
        vt_gvar
    }
}

fn emit_vtable_methods(bcx: &Block,
                       impl_id: ast::DefId,
                       substs: Vec<ty::t>,
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
        if m.generics.has_type_params() ||
           ty::type_has_self(ty::mk_bare_fn(tcx, m.fty.clone())) {
            debug!("(making impl vtable) method has self or type params: {}",
                   token::get_ident(ident));
            C_null(Type::nil(ccx).ptr_to())
        } else {
            trans_fn_ref_with_vtables(bcx, m_id, ExprId(0),
                                      substs.clone(), Some(vtables.clone()))
        }
    }).collect()
}

pub fn trans_trait_cast<'a>(bcx: &'a Block<'a>,
                            datum: Datum<Expr>,
                            id: ast::NodeId,
                            dest: expr::Dest)
                            -> &'a Block<'a> {
    /*!
     * Generates the code to convert from a pointer (`~T`, `&T`, etc)
     * into an object (`~Trait`, `&Trait`, etc). This means creating a
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
        resolve_param_vtables_under_param_substs(ccx.tcx(),
            bcx.fcx.param_substs,
            vtable_map.get(&MethodCall::expr(id)).get(0).as_slice())
    };
    let vtable = get_vtable(bcx, v_ty, origins);
    let llvtabledest = GEPi(bcx, lldest, [0u, abi::trt_field_vtable]);
    let llvtabledest = PointerCast(bcx, llvtabledest, val_ty(vtable).ptr_to());
    Store(bcx, vtable, llvtabledest);

    bcx
}
