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
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::expr::{SaveIn, Ignore};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::monomorphize;
use middle::trans::type_of::*;
use middle::ty;
use middle::typeck;
use util::common::indenter;
use util::ppaux::Repr;

use middle::trans::type_::Type;

use std::c_str::ToCStr;
use std::vec;
use syntax::ast_map::{Path, PathMod, PathName, PathPrettyName};
use syntax::parse::token;
use syntax::{ast, ast_map, ast_util, visit};

/**
The main "translation" pass for methods.  Generates code
for non-monomorphized methods only.  Other methods will
be generated once they are invoked with specific type parameters,
see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
*/
pub fn trans_impl(ccx: @CrateContext,
                  path: Path,
                  name: ast::Ident,
                  methods: &[@ast::Method],
                  generics: &ast::Generics,
                  id: ast::NodeId) {
    let _icx = push_ctxt("impl::trans_impl");
    let tcx = ccx.tcx;

    debug!("trans_impl(path={}, name={}, id={:?})",
           path.repr(tcx), name.repr(tcx), id);

    // Both here and below with generic methods, be sure to recurse and look for
    // items that we need to translate.
    if !generics.ty_params.is_empty() {
        let mut v = TransItemVisitor{ ccx: ccx };
        for method in methods.iter() {
            visit::walk_method_helper(&mut v, *method, ());
        }
        return;
    }
    let sub_path = vec::append_one(path, PathName(name));
    for method in methods.iter() {
        if method.generics.ty_params.len() == 0u {
            let llfn = get_item_val(ccx, method.id);
            let path = vec::append_one(sub_path.clone(),
                                       PathName(method.ident));

            trans_method(ccx,
                         path,
                         *method,
                         None,
                         |_| llfn);
        } else {
            let mut v = TransItemVisitor{ ccx: ccx };
            visit::walk_method_helper(&mut v, *method, ());
        }
    }
}

/// Translates a (possibly monomorphized) method body.
///
/// Parameters:
/// * `path`: the path to the method
/// * `method`: the AST node for the method
/// * `param_substs`: if this is a generic method, the current values for
///   type parameters and so forth, else none
/// * `llfn`: a closure returning the LLVM ValueRef for the method
/// * `impl_id`: the node ID of the impl this method is inside
///
/// XXX(pcwalton) Can we take `path` by reference?
pub fn trans_method(ccx: @CrateContext,
                    path: Path,
                    method: &ast::Method,
                    param_substs: Option<@param_substs>,
                    llfn_with_self: |Option<ty::t>| -> ValueRef) -> ValueRef {
    // figure out how self is being passed
    let self_ty = match method.explicit_self.node {
      ast::SelfStatic => None,
      _ => {
        // determine the (monomorphized) type that `self` maps to for
        // this method
        let self_ty = ty::node_id_to_type(ccx.tcx, method.self_id);
        let self_ty = match param_substs {
            None => self_ty,
            Some(param_substs) => {
                ty::subst_tps(ccx.tcx,
                              param_substs.tys,
                              param_substs.self_ty,
                              self_ty)
            }
        };
        debug!("calling trans_fn with self_ty {}", self_ty.repr(ccx.tcx));
        Some(self_ty)
      }
    };

    let llfn = llfn_with_self(self_ty);

    // generate the actual code
    trans_fn(ccx,
             path,
             method.decl,
             method.body,
             llfn,
             self_ty,
             param_substs,
             method.id,
             Some(method),
             []);
    llfn
}

pub fn trans_method_callee<'a>(
                           bcx: &'a Block<'a>,
                           callee_id: ast::NodeId,
                           this: &ast::Expr,
                           mentry: typeck::method_map_entry)
                           -> Callee<'a> {
    let _icx = push_ctxt("impl::trans_method_callee");

    debug!("trans_method_callee(callee_id={:?}, this={}, mentry={})",
           callee_id,
           bcx.expr_to_str(this),
           mentry.repr(bcx.tcx()));

    match mentry.origin {
        typeck::method_static(did) => {
            let self_ty = monomorphize_type(bcx, mentry.self_ty);
            let mut temp_cleanups = ~[];
            let Result {bcx, val} = trans_arg_expr(bcx, self_ty, this,
                                                   &mut temp_cleanups,
                                                   DontAutorefArg);
            // HACK should not need the pointer cast, eventually trans_fn_ref
            // should return a function type with the right type for self.
            let callee_fn = callee::trans_fn_ref(bcx, did, callee_id);
            let fn_ty = node_id_type(bcx, callee_id);
            let llfn_ty = type_of_fn_from_ty(bcx.ccx(), Some(self_ty), fn_ty).ptr_to();
            let llfn_val = PointerCast(bcx, callee_fn.llfn, llfn_ty);
            Callee {
                bcx: bcx,
                data: Method(MethodData {
                    llfn: llfn_val,
                    llself: val,
                    temp_cleanup: temp_cleanups.head_opt().map(|v| *v)
                })
            }
        }
        typeck::method_param(typeck::method_param {
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
                    trans_monomorphized_callee(bcx, callee_id, this, mentry,
                                               trait_id, off, vtbl)
                }
                // how to get rid of this?
                None => fail!("trans_method_callee: missing param_substs")
            }
        }

        typeck::method_object(ref mt) => {
            trans_trait_callee(bcx,
                               callee_id,
                               mt.real_index,
                               this)
        }
    }
}

pub fn trans_static_method_callee(bcx: &Block,
                                  method_id: ast::DefId,
                                  trait_id: ast::DefId,
                                  callee_id: ast::NodeId)
                                  -> FnData {
    let _icx = push_ctxt("impl::trans_static_method_callee");
    let ccx = bcx.ccx();

    debug!("trans_static_method_callee(method_id={:?}, trait_id={}, \
            callee_id={:?})",
           method_id,
           ty::item_path_str(bcx.tcx(), trait_id),
           callee_id);
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
        generics.type_param_defs.len();

    let mname = if method_id.crate == ast::LOCAL_CRATE {
        {
            let items = bcx.tcx().items.borrow();
            match items.get().get_copy(&method_id.node) {
                ast_map::NodeTraitMethod(trait_method, _, _) => {
                    ast_util::trait_method_to_ty_method(trait_method).ident
                }
                _ => fail!("callee is not a trait method")
            }
        }
    } else {
        let path = csearch::get_item_path(bcx.tcx(), method_id);
        match path[path.len()-1] {
            PathPrettyName(s, _) | PathName(s) => { s }
            PathMod(_) => { fail!("path doesn't have a name?") }
        }
    };
    debug!("trans_static_method_callee: method_id={:?}, callee_id={:?}, \
            name={}", method_id, callee_id, ccx.sess.str_of(mname));

    let vtbls = {
        let vtable_map = ccx.maps.vtable_map.borrow();
        vtable_map.get().get_copy(&callee_id)
    };
    let vtbls = resolve_vtables_in_fn_ctxt(bcx.fcx, vtbls);

    match vtbls[bound_index][0] {
        typeck::vtable_static(impl_did, ref rcvr_substs, rcvr_origins) => {
            assert!(rcvr_substs.iter().all(|t| !ty::type_needs_infer(*t)));

            let mth_id = method_with_name(ccx, impl_did, mname.name);
            let (callee_substs, callee_origins) =
                combine_impl_and_methods_tps(
                    bcx, mth_id, callee_id,
                    *rcvr_substs, rcvr_origins);

            let FnData {llfn: lval} =
                trans_fn_ref_with_vtables(bcx,
                                          mth_id,
                                          callee_id,
                                          callee_substs,
                                          Some(callee_origins));

            let callee_ty = node_id_type(bcx, callee_id);
            let llty = type_of_fn_from_ty(ccx, None, callee_ty).ptr_to();
            FnData {llfn: PointerCast(bcx, lval, llty)}
        }
        _ => {
            fail!("vtable_param left in monomorphized \
                   function's vtable substs");
        }
    }
}

pub fn method_with_name(ccx: &CrateContext,
                        impl_id: ast::DefId,
                        name: ast::Name) -> ast::DefId {
    {
        let impl_method_cache = ccx.impl_method_cache.borrow();
        let meth_id_opt = impl_method_cache.get().find_copy(&(impl_id, name));
        match meth_id_opt {
            Some(m) => return m,
            None => {}
        }
    }

    let impls = ccx.tcx.impls.borrow();
    let imp = impls.get().find(&impl_id)
        .expect("could not find impl while translating");
    let meth = imp.methods.iter().find(|m| m.ident.name == name)
        .expect("could not find method while translating");

    let mut impl_method_cache = ccx.impl_method_cache.borrow_mut();
    impl_method_cache.get().insert((impl_id, name), meth.def_id);
    meth.def_id
}

pub fn trans_monomorphized_callee<'a>(
                                  bcx: &'a Block<'a>,
                                  callee_id: ast::NodeId,
                                  base: &ast::Expr,
                                  mentry: typeck::method_map_entry,
                                  trait_id: ast::DefId,
                                  n_method: uint,
                                  vtbl: typeck::vtable_origin)
                                  -> Callee<'a> {
    let _icx = push_ctxt("impl::trans_monomorphized_callee");
    return match vtbl {
      typeck::vtable_static(impl_did, ref rcvr_substs, rcvr_origins) => {
          let ccx = bcx.ccx();
          let mname = ty::trait_method(ccx.tcx, trait_id, n_method).ident;
          let mth_id = method_with_name(bcx.ccx(), impl_did, mname.name);

          // obtain the `self` value:
          let self_ty = monomorphize_type(bcx, mentry.self_ty);
          let mut temp_cleanups = ~[];
          let Result {bcx, val} = trans_arg_expr(bcx, self_ty, base,
                                                 &mut temp_cleanups,
                                                 DontAutorefArg);

          // create a concatenated set of substitutions which includes
          // those from the impl and those from the method:
          let (callee_substs, callee_origins) =
              combine_impl_and_methods_tps(
                  bcx, mth_id, callee_id,
                  *rcvr_substs, rcvr_origins);

          // translate the function
          let callee = trans_fn_ref_with_vtables(bcx,
                                                 mth_id,
                                                 callee_id,
                                                 callee_substs,
                                                 Some(callee_origins));

          // create a llvalue that represents the fn ptr
          // HACK should not need the pointer cast (add self in trans_fn_ref_with_vtables).
          let fn_ty = node_id_type(bcx, callee_id);
          let llfn_ty = type_of_fn_from_ty(ccx, Some(self_ty), fn_ty).ptr_to();
          let llfn_val = PointerCast(bcx, callee.llfn, llfn_ty);

          // combine the self environment with the rest
          Callee {
              bcx: bcx,
              data: Method(MethodData {
                  llfn: llfn_val,
                  llself: val,
                  temp_cleanup: temp_cleanups.head_opt().map(|v| *v)
              })
          }
      }
      typeck::vtable_param(..) => {
          fail!("vtable_param left in monomorphized function's vtable substs");
      }
    };

}

pub fn combine_impl_and_methods_tps(bcx: &Block,
                                    mth_did: ast::DefId,
                                    callee_id: ast::NodeId,
                                    rcvr_substs: &[ty::t],
                                    rcvr_origins: typeck::vtable_res)
                                    -> (~[ty::t], typeck::vtable_res) {
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
    let method = ty::method(ccx.tcx, mth_did);
    let n_m_tps = method.generics.type_param_defs.len();
    let node_substs = node_id_type_params(bcx, callee_id);
    debug!("rcvr_substs={:?}", rcvr_substs.repr(ccx.tcx));
    let ty_substs
        = vec::append(rcvr_substs.to_owned(),
                      node_substs.tailn(node_substs.len() - n_m_tps));
    debug!("n_m_tps={:?}", n_m_tps);
    debug!("node_substs={:?}", node_substs.repr(ccx.tcx));
    debug!("ty_substs={:?}", ty_substs.repr(ccx.tcx));


    // Now, do the same work for the vtables.  The vtables might not
    // exist, in which case we need to make them.
    let r_m_origins = match node_vtables(bcx, callee_id) {
        Some(vt) => vt,
        None => @vec::from_elem(node_substs.len(), @~[])
    };
    let vtables
        = @vec::append(rcvr_origins.to_owned(),
                       r_m_origins.tailn(r_m_origins.len() - n_m_tps));

    return (ty_substs, vtables);
}

pub fn trans_trait_callee<'a>(
                          bcx: &'a Block<'a>,
                          callee_id: ast::NodeId,
                          n_method: uint,
                          self_expr: &ast::Expr)
                          -> Callee<'a> {
    /*!
     * Create a method callee where the method is coming from a trait
     * object (e.g., @Trait type).  In this case, we must pull the fn
     * pointer out of the vtable that is packaged up with the object.
     * Objects are represented as a pair, so we first evaluate the self
     * expression and then extract the self data and vtable out of the
     * pair.
     */

    let _icx = push_ctxt("impl::trans_trait_callee");
    let mut bcx = bcx;

    // make a local copy for trait if needed
    let self_ty = expr_ty_adjusted(bcx, self_expr);
    let self_scratch = match ty::get(self_ty).sty {
        ty::ty_trait(_, _, ty::RegionTraitStore(..), _, _) => {
            unpack_datum!(bcx, expr::trans_to_datum(bcx, self_expr))
        }
        _ => {
            let d = scratch_datum(bcx, self_ty, "__trait_callee", false);
            bcx = expr::trans_into(bcx, self_expr, expr::SaveIn(d.val));
            // Arrange a temporary cleanup for the object in case something
            // should go wrong before the method is actually *invoked*.
            d.add_clean(bcx);
            d
        }
    };


    let callee_ty = node_id_type(bcx, callee_id);
    trans_trait_callee_from_llval(bcx,
                                  callee_ty,
                                  n_method,
                                  self_scratch.val,
                                  Some(self_scratch.val))
}

pub fn trans_trait_callee_from_llval<'a>(
                                     bcx: &'a Block<'a>,
                                     callee_ty: ty::t,
                                     n_method: uint,
                                     llpair: ValueRef,
                                     temp_cleanup: Option<ValueRef>)
                                     -> Callee<'a> {
    /*!
     * Same as `trans_trait_callee()` above, except that it is given
     * a by-ref pointer to the object pair.
     */

    let _icx = push_ctxt("impl::trans_trait_callee");
    let ccx = bcx.ccx();

    // Load the data pointer from the object.
    debug!("(translating trait callee) loading second index from pair");
    let llboxptr = GEPi(bcx, llpair, [0u, abi::trt_field_box]);
    let llbox = Load(bcx, llboxptr);
    let llself = PointerCast(bcx, llbox, Type::opaque_box(ccx).ptr_to());

    // Load the function from the vtable and cast it to the expected type.
    debug!("(translating trait callee) loading method");
    let llcallee_ty = type_of_fn_from_ty(ccx, None, callee_ty);
    let llvtable = Load(bcx,
                        PointerCast(bcx,
                                    GEPi(bcx, llpair,
                                         [0u, abi::trt_field_vtable]),
                                    Type::vtable().ptr_to().ptr_to()));
    let mptr = Load(bcx, GEPi(bcx, llvtable, [0u, n_method + 1]));
    let mptr = PointerCast(bcx, mptr, llcallee_ty.ptr_to());

    return Callee {
        bcx: bcx,
        data: Method(MethodData {
            llfn: mptr,
            llself: llself,
            temp_cleanup: temp_cleanup
        })
    };
}

pub fn vtable_id(ccx: @CrateContext,
                 origin: &typeck::vtable_origin)
              -> mono_id {
    match origin {
        &typeck::vtable_static(impl_id, ref substs, sub_vtables) => {
            let psubsts = param_substs {
                tys: (*substs).clone(),
                vtables: Some(sub_vtables),
                self_ty: None,
                self_vtables: None
            };

            monomorphize::make_mono_id(
                ccx,
                impl_id,
                &psubsts)
        }

        // can't this be checked at the callee?
        _ => fail!("vtable_id")
    }
}

/// Creates a returns a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
pub fn get_vtable(bcx: &Block,
                  self_ty: ty::t,
                  origins: typeck::vtable_param_res)
                  -> ValueRef {
    let ccx = bcx.ccx();
    let _icx = push_ctxt("impl::get_vtable");

    // Check the cache.
    let hash_id = (self_ty, vtable_id(ccx, &origins[0]));
    {
        let vtables = ccx.vtables.borrow();
        match vtables.get().find(&hash_id) {
            Some(&val) => { return val }
            None => { }
        }
    }

    // Not in the cache. Actually build it.
    let methods = origins.flat_map(|origin| {
        match *origin {
            typeck::vtable_static(id, ref substs, sub_vtables) => {
                emit_vtable_methods(bcx, id, *substs, sub_vtables)
            }
            _ => ccx.sess.bug("get_vtable: expected a static origin"),
        }
    });

    // Generate a type descriptor for the vtable.
    let tydesc = get_tydesc(ccx, self_ty);
    glue::lazily_emit_all_tydesc_glue(ccx, tydesc);

    let vtable = make_vtable(ccx, tydesc, methods);

    let mut vtables = ccx.vtables.borrow_mut();
    vtables.get().insert(hash_id, vtable);
    return vtable;
}

/// Helper function to declare and initialize the vtable.
pub fn make_vtable(ccx: &CrateContext,
                   tydesc: &tydesc_info,
                   ptrs: &[ValueRef])
                   -> ValueRef {
    unsafe {
        let _icx = push_ctxt("impl::make_vtable");

        let mut components = ~[ tydesc.tydesc ];
        for &ptr in ptrs.iter() {
            components.push(ptr)
        }

        let tbl = C_struct(components, false);
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
                       substs: &[ty::t],
                       vtables: typeck::vtable_res)
                       -> ~[ValueRef] {
    let ccx = bcx.ccx();
    let tcx = ccx.tcx;

    let trt_id = match ty::impl_trait_ref(tcx, impl_id) {
        Some(t_id) => t_id.def_id,
        None       => ccx.sess.bug("make_impl_vtable: don't know how to \
                                    make a vtable for a type impl!")
    };

    ty::populate_implementations_for_trait_if_necessary(bcx.tcx(), trt_id);

    let trait_method_def_ids = ty::trait_method_def_ids(tcx, trt_id);
    trait_method_def_ids.map(|method_def_id| {
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
                   tcx.sess.str_of(ident));
            C_null(Type::nil().ptr_to())
        } else {
            trans_fn_ref_with_vtables(bcx, m_id, 0,
                                      substs, Some(vtables)).llfn
        }
    })
}

pub fn trans_trait_cast<'a>(
                        bcx: &'a Block<'a>,
                        val: &ast::Expr,
                        id: ast::NodeId,
                        dest: expr::Dest,
                        obj: Option<Datum>)
                        -> &'a Block<'a> {
    let mut bcx = bcx;
    let _icx = push_ctxt("impl::trans_cast");

    let lldest = match dest {
        Ignore => {
            return expr::trans_into(bcx, val, Ignore);
        }
        SaveIn(dest) => dest
    };

    let ccx = bcx.ccx();
    let v_ty = expr_ty(bcx, val);

    let mut llboxdest = GEPi(bcx, lldest, [0u, abi::trt_field_box]);
    // Just store the pointer into the pair. (Region/borrowed
    // and boxed trait objects are represented as pairs, and
    // have no type descriptor field.)
    llboxdest = PointerCast(bcx,
                            llboxdest,
                            type_of(bcx.ccx(), v_ty).ptr_to());
    bcx = match obj {
        Some(datum) => {
            datum.store_to_dest(bcx, SaveIn(llboxdest))
        }
        None => expr::trans_into(bcx, val, SaveIn(llboxdest))
    };

    // Store the vtable into the pair or triple.
    // This is structured a bit funny because of dynamic borrow failures.
    let origins = {
        let res = {
            let vtable_map = ccx.maps.vtable_map.borrow();
            *vtable_map.get().get(&id)
        };
        let res = resolve_vtables_in_fn_ctxt(bcx.fcx, res);
        res[0]
    };
    let vtable = get_vtable(bcx, v_ty, origins);
    Store(bcx, vtable, PointerCast(bcx,
                                   GEPi(bcx, lldest, [0u, abi::trt_field_vtable]),
                                   val_ty(vtable).ptr_to()));

    bcx
}
