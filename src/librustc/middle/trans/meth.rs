// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

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
use middle::trans::expr::{SaveIn, Ignore};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::inline;
use middle::trans::monomorphize;
use middle::trans::type_of::*;
use middle::ty;
use middle::typeck;
use util::common::indenter;
use util::ppaux::Repr;

use core::str;
use core::vec;
use syntax::ast_map::{path, path_mod, path_name};
use syntax::ast_util;
use syntax::{ast, ast_map};

/**
The main "translation" pass for methods.  Generates code
for non-monomorphized methods only.  Other methods will
be generated once they are invoked with specific type parameters,
see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
*/
pub fn trans_impl(ccx: @CrateContext,
                  path: path,
                  name: ast::ident,
                  methods: &[@ast::method],
                  generics: &ast::Generics,
                  self_ty: Option<ty::t>,
                  id: ast::node_id) {
    let _icx = ccx.insn_ctxt("impl::trans_impl");
    let tcx = ccx.tcx;

    debug!("trans_impl(path=%s, name=%s, self_ty=%s, id=%?)",
           path.repr(tcx), name.repr(tcx), self_ty.repr(tcx), id);

    if !generics.ty_params.is_empty() { return; }
    let sub_path = vec::append_one(path, path_name(name));
    for methods.each |method| {
        if method.generics.ty_params.len() == 0u {
            let llfn = get_item_val(ccx, method.id);
            let path = vec::append_one(/*bad*/copy sub_path,
                                       path_name(method.ident));

            let param_substs_opt;
            match self_ty {
                None => param_substs_opt = None,
                Some(self_ty) => {
                    param_substs_opt = Some(@param_substs {
                        tys: ~[],
                        vtables: None,
                        type_param_defs: @~[],
                        self_ty: Some(self_ty)
                    });
                }
            }

            trans_method(ccx,
                         path,
                         *method,
                         param_substs_opt,
                         self_ty,
                         llfn,
                         ast_util::local_def(id));
        }
    }
}

/**
Translates a (possibly monomorphized) method body.

# Parameters

- `path`: the path to the method
- `method`: the AST node for the method
- `param_substs`: if this is a generic method, the current values for
  type parameters and so forth, else none
- `base_self_ty`: optionally, the explicit self type for this method. This
  will be none if this is not a default method and must always be present
  if this is a default method.
- `llfn`: the LLVM ValueRef for the method
- `impl_id`: the node ID of the impl this method is inside
*/
pub fn trans_method(ccx: @CrateContext,
                    path: path,
                    method: &ast::method,
                    param_substs: Option<@param_substs>,
                    base_self_ty: Option<ty::t>,
                    llfn: ValueRef,
                    impl_id: ast::def_id) {
    // figure out how self is being passed
    let self_arg = match method.explicit_self.node {
      ast::sty_static => {
        no_self
      }
      _ => {
        // determine the (monomorphized) type that `self` maps to for
        // this method
        let self_ty = match base_self_ty {
            None => ty::node_id_to_type(ccx.tcx, method.self_id),
            Some(provided_self_ty) => provided_self_ty,
        };
        let self_ty = match param_substs {
            None => self_ty,
            Some(@param_substs {tys: ref tys, _}) => {
                ty::subst_tps(ccx.tcx, *tys, None, self_ty)
            }
        };
        debug!("calling trans_fn with base_self_ty %s, self_ty %s",
               base_self_ty.repr(ccx.tcx),
               self_ty.repr(ccx.tcx));
        match method.explicit_self.node {
          ast::sty_value => {
            impl_owned_self(self_ty)
          }
          _ => {
            impl_self(self_ty)
          }
        }
      }
    };

    // generate the actual code
    trans_fn(ccx,
             path,
             &method.decl,
             &method.body,
             llfn,
             self_arg,
             param_substs,
             method.id,
             Some(impl_id),
             []);
}

pub fn trans_self_arg(bcx: block,
                      base: @ast::expr,
                      mentry: typeck::method_map_entry) -> Result {
    let _icx = bcx.insn_ctxt("impl::trans_self_arg");
    let mut temp_cleanups = ~[];

    // Compute the type of self.
    let self_ty = monomorphize_type(bcx, mentry.self_ty);

    let result = trans_arg_expr(bcx,
                                self_ty,
                                mentry.self_mode,
                                base,
                                &mut temp_cleanups,
                                None,
                                DontAutorefArg);

    // FIXME(#3446)---this is wrong, actually.  The temp_cleanups
    // should be revoked only after all arguments have been passed.
    for temp_cleanups.each |c| {
        revoke_clean(bcx, *c)
    }

    return result;
}

pub fn trans_method_callee(bcx: block,
                           callee_id: ast::node_id,
                           this: @ast::expr,
                           mentry: typeck::method_map_entry)
                           -> Callee {
    let _icx = bcx.insn_ctxt("impl::trans_method_callee");
    let tcx = bcx.tcx();

    debug!("trans_method_callee(callee_id=%?, this=%s, mentry=%s)",
           callee_id,
           bcx.expr_to_str(this),
           mentry.repr(bcx.tcx()));

    // Replace method_self with method_static here.
    let mut origin = mentry.origin;
    match origin {
        typeck::method_self(copy trait_id, copy method_index) => {
            // Get the ID of the impl we're inside.
            let impl_def_id = bcx.fcx.impl_id.get();

            debug!("impl_def_id is %?", impl_def_id);

            // Get the ID of the method we're calling.
            let method_name =
                ty::trait_method(tcx, trait_id, method_index).ident;
            let method_id =
                method_with_name(bcx.ccx(), impl_def_id, method_name);
            origin = typeck::method_static(method_id);
        }
        typeck::method_super(trait_id, method_index) => {
            // <self_ty> is the self type for this method call
            let self_ty = node_id_type(bcx, this.id);
            // <impl_id> is the ID of the implementation of
            // trait <trait_id> for type <self_ty>
            let impl_id = ty::get_impl_id(tcx, trait_id, self_ty);
            // Get the supertrait's methods
            let supertrait_method_def_ids = ty::trait_method_def_ids(tcx, trait_id);
            // Make sure to fail with a readable error message if
            // there's some internal error here
            if !(method_index < supertrait_method_def_ids.len()) {
                tcx.sess.bug("trans_method_callee: supertrait method \
                              index is out of bounds");
            }
            // Get the method name using the method index in the origin
            let method_name =
                ty::method(tcx, supertrait_method_def_ids[method_index]).ident;
            // Now that we know the impl ID, we can look up the method
            // ID from its name
            origin = typeck::method_static(method_with_name(bcx.ccx(),
                                                            impl_id,
                                                            method_name));
        }
        typeck::method_static(*) | typeck::method_param(*) |
        typeck::method_trait(*) => {}
    }

    debug!("origin=%?", origin);

    match origin {
        typeck::method_static(did) => {
            let callee_fn = callee::trans_fn_ref(bcx, did, callee_id);
            let Result {bcx, val} = trans_self_arg(bcx, this, mentry);
            Callee {
                bcx: bcx,
                data: Method(MethodData {
                    llfn: callee_fn.llfn,
                    llself: val,
                    self_ty: node_id_type(bcx, this.id),
                    self_mode: mentry.self_mode,
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
                    let vtbl = find_vtable(bcx.tcx(), substs, p, b);
                    trans_monomorphized_callee(bcx, callee_id, this, mentry,
                                               trait_id, off, vtbl)
                }
                // how to get rid of this?
                None => fail!("trans_method_callee: missing param_substs")
            }
        }
        typeck::method_trait(_, off, store) => {
            trans_trait_callee(bcx,
                               callee_id,
                               off,
                               this,
                               store,
                               mentry.explicit_self)
        }
        typeck::method_self(*) | typeck::method_super(*) => {
            fail!("method_self or method_super should have been handled \
                above")
        }
    }
}

pub fn trans_static_method_callee(bcx: block,
                                  method_id: ast::def_id,
                                  trait_id: ast::def_id,
                                  callee_id: ast::node_id)
                               -> FnData {
    let _icx = bcx.insn_ctxt("impl::trans_static_method_callee");
    let ccx = bcx.ccx();

    debug!("trans_static_method_callee(method_id=%?, trait_id=%s, \
            callee_id=%?)",
           method_id,
           ty::item_path_str(bcx.tcx(), trait_id),
           callee_id);
    let _indenter = indenter();

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
    // bound to.  Due to the fact that we use a flattened list of
    // impls, one per bound, this means we have to total up the bounds
    // found on the type parametesr T1...Tn to find the index of the
    // one we are interested in.
    let bound_index = {
        let trait_def = ty::lookup_trait_def(bcx.tcx(), trait_id);
        ty::count_traits_and_supertraits(
            bcx.tcx(), *trait_def.generics.type_param_defs)
    };

    let mname = if method_id.crate == ast::local_crate {
        match bcx.tcx().items.get_copy(&method_id.node) {
            ast_map::node_trait_method(trait_method, _, _) => {
                ast_util::trait_method_to_ty_method(trait_method).ident
            }
            _ => fail!("callee is not a trait method")
        }
    } else {
        let path = csearch::get_item_path(bcx.tcx(), method_id);
        match path[path.len()-1] {
            path_name(s) => { s }
            path_mod(_) => { fail!("path doesn't have a name?") }
        }
    };
    debug!("trans_static_method_callee: method_id=%?, callee_id=%?, \
            name=%s", method_id, callee_id, *ccx.sess.str_of(mname));

    let vtbls = resolve_vtables_in_fn_ctxt(
        bcx.fcx, ccx.maps.vtable_map.get_copy(&callee_id));

    match vtbls[bound_index] {
        typeck::vtable_static(impl_did, ref rcvr_substs, rcvr_origins) => {
            assert!(rcvr_substs.all(|t| !ty::type_needs_infer(*t)));

            let mth_id = method_with_name(bcx.ccx(), impl_did, mname);
            let callee_substs = combine_impl_and_methods_tps(
                bcx, mth_id, impl_did, callee_id, *rcvr_substs);
            let callee_origins = combine_impl_and_methods_origins(
                bcx, mth_id, impl_did, callee_id, rcvr_origins);

            let FnData {llfn: lval} =
                trans_fn_ref_with_vtables(bcx,
                                          mth_id,
                                          callee_id,
                                          callee_substs,
                                          Some(callee_origins));

            let callee_ty = node_id_type(bcx, callee_id);
            let llty = T_ptr(type_of_fn_from_ty(ccx, callee_ty));
            FnData {llfn: PointerCast(bcx, lval, llty)}
        }
        _ => {
            fail!("vtable_param left in monomorphized \
                   function's vtable substs");
        }
    }
}

pub fn method_from_methods(ms: &[@ast::method], name: ast::ident)
    -> Option<ast::def_id> {
    ms.find(|m| m.ident == name).map(|m| ast_util::local_def(m.id))
}

pub fn method_with_name(ccx: @CrateContext, impl_id: ast::def_id,
                        name: ast::ident) -> ast::def_id {
    if impl_id.crate == ast::local_crate {
        match ccx.tcx.items.get_copy(&impl_id.node) {
          ast_map::node_item(@ast::item {
                node: ast::item_impl(_, _, _, ref ms),
                _
            }, _) => {
            method_from_methods(*ms, name).get()
          }
          _ => fail!("method_with_name")
        }
    } else {
        csearch::get_impl_method(ccx.sess.cstore, impl_id, name)
    }
}

pub fn method_with_name_or_default(ccx: @CrateContext,
                                   impl_id: ast::def_id,
                                   name: ast::ident) -> ast::def_id {
    if impl_id.crate == ast::local_crate {
        match ccx.tcx.items.get_copy(&impl_id.node) {
          ast_map::node_item(@ast::item {
                node: ast::item_impl(_, _, _, ref ms), _
          }, _) => {
              let did = method_from_methods(*ms, name);
              if did.is_some() {
                  return did.get();
              } else {
                  // Look for a default method
                  let pmm = ccx.tcx.provided_methods;
                  match pmm.find(&impl_id) {
                      Some(pmis) => {
                          for pmis.each |pmi| {
                              if pmi.method_info.ident == name {
                                  debug!("pmi.method_info.did = %?", pmi.method_info.did);
                                  return pmi.method_info.did;
                              }
                          }
                          fail!()
                      }
                      None => fail!()
                  }
              }
          }
          _ => fail!("method_with_name")
        }
    } else {
        csearch::get_impl_method(ccx.sess.cstore, impl_id, name)
    }
}

pub fn method_ty_param_count(ccx: @CrateContext, m_id: ast::def_id,
                             i_id: ast::def_id) -> uint {
    debug!("method_ty_param_count: m_id: %?, i_id: %?", m_id, i_id);
    if m_id.crate == ast::local_crate {
        match ccx.tcx.items.find(&m_id.node) {
            Some(&ast_map::node_method(m, _, _)) => m.generics.ty_params.len(),
            None => {
                match ccx.tcx.provided_method_sources.find(&m_id) {
                    Some(source) => {
                        method_ty_param_count(
                            ccx, source.method_id, source.impl_id)
                    }
                    None => fail!()
                }
            }
            Some(&ast_map::node_trait_method(@ast::provided(@ref m),
                                            _, _)) => {
                m.generics.ty_params.len()
            }
            copy e => fail!("method_ty_param_count %?", e)
        }
    } else {
        csearch::get_type_param_count(ccx.sess.cstore, m_id) -
            csearch::get_type_param_count(ccx.sess.cstore, i_id)
    }
}

pub fn trans_monomorphized_callee(bcx: block,
                                  callee_id: ast::node_id,
                                  base: @ast::expr,
                                  mentry: typeck::method_map_entry,
                                  trait_id: ast::def_id,
                                  n_method: uint,
                                  vtbl: typeck::vtable_origin)
                                  -> Callee {
    let _icx = bcx.insn_ctxt("impl::trans_monomorphized_callee");
    return match vtbl {
      typeck::vtable_static(impl_did, ref rcvr_substs, rcvr_origins) => {
          let ccx = bcx.ccx();
          let mname = ty::trait_method(ccx.tcx, trait_id, n_method).ident;
          let mth_id = method_with_name_or_default(
              bcx.ccx(), impl_did, mname);

          // obtain the `self` value:
          let Result {bcx, val: llself_val} =
              trans_self_arg(bcx, base, mentry);

          // create a concatenated set of substitutions which includes
          // those from the impl and those from the method:
          let callee_substs = combine_impl_and_methods_tps(
              bcx, mth_id, impl_did, callee_id, *rcvr_substs);
          let callee_origins = combine_impl_and_methods_origins(
              bcx, mth_id, impl_did, callee_id, rcvr_origins);

          // translate the function
          let callee = trans_fn_ref_with_vtables(bcx,
                                                 mth_id,
                                                 callee_id,
                                                 callee_substs,
                                                 Some(callee_origins));

          // create a llvalue that represents the fn ptr
          let fn_ty = node_id_type(bcx, callee_id);
          let llfn_ty = T_ptr(type_of_fn_from_ty(ccx, fn_ty));
          let llfn_val = PointerCast(bcx, callee.llfn, llfn_ty);

          // combine the self environment with the rest
          Callee {
              bcx: bcx,
              data: Method(MethodData {
                  llfn: llfn_val,
                  llself: llself_val,
                  self_ty: node_id_type(bcx, base.id),
                  self_mode: mentry.self_mode,
              })
          }
      }
      typeck::vtable_param(*) => {
          fail!("vtable_param left in monomorphized function's vtable substs");
      }
    };

}

pub fn combine_impl_and_methods_tps(bcx: block,
                                    mth_did: ast::def_id,
                                    impl_did: ast::def_id,
                                    callee_id: ast::node_id,
                                    rcvr_substs: &[ty::t])
                                    -> ~[ty::t] {
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
    let n_m_tps = method_ty_param_count(ccx, mth_did, impl_did);
    let node_substs = node_id_type_params(bcx, callee_id);
    debug!("rcvr_substs=%?", rcvr_substs.map(|t| bcx.ty_to_str(*t)));
    let ty_substs
        = vec::append(rcvr_substs.to_vec(),
                      vec::tailn(node_substs,
                                 node_substs.len() - n_m_tps));
    debug!("n_m_tps=%?", n_m_tps);
    debug!("node_substs=%?", node_substs.map(|t| bcx.ty_to_str(*t)));
    debug!("ty_substs=%?", ty_substs.map(|t| bcx.ty_to_str(*t)));

    return ty_substs;
}

pub fn combine_impl_and_methods_origins(bcx: block,
                                        mth_did: ast::def_id,
                                        impl_did: ast::def_id,
                                        callee_id: ast::node_id,
                                        rcvr_origins: typeck::vtable_res)
                                     -> typeck::vtable_res {
    /*!
     *
     * Similar to `combine_impl_and_methods_tps`, but for vtables.
     * This is much messier because of the flattened layout we are
     * currently using (for some reason that I fail to understand).
     * The proper fix is described in #3446.
     */


    // Find the bounds for the method, which are the tail of the
    // bounds found in the item type, as the item type combines the
    // rcvr + method bounds.
    let ccx = bcx.ccx(), tcx = bcx.tcx();
    let n_m_tps = method_ty_param_count(ccx, mth_did, impl_did);
    let ty::ty_param_bounds_and_ty {
        generics: r_m_generics,
        _
    } = ty::lookup_item_type(tcx, mth_did);
    let n_r_m_tps = r_m_generics.type_param_defs.len(); // rcvr + method tps
    let m_type_param_defs =
        vec::slice(*r_m_generics.type_param_defs, n_r_m_tps - n_m_tps, n_r_m_tps);

    // Flatten out to find the number of vtables the method expects.
    let m_vtables = ty::count_traits_and_supertraits(tcx, m_type_param_defs);

    // Find the vtables we computed at type check time and monomorphize them
    let r_m_origins = match node_vtables(bcx, callee_id) {
        Some(vt) => vt,
        None => @~[]
    };

    // Extract those that belong to method:
    let m_origins = vec::tailn(*r_m_origins, r_m_origins.len() - m_vtables);

    // Combine rcvr + method to find the final result:
    @vec::append(/*bad*/copy *rcvr_origins, m_origins)
}


pub fn trans_trait_callee(bcx: block,
                          callee_id: ast::node_id,
                          n_method: uint,
                          self_expr: @ast::expr,
                          store: ty::TraitStore,
                          explicit_self: ast::explicit_self_)
                       -> Callee {
    //!
    //
    // Create a method callee where the method is coming from a trait
    // instance (e.g., @Trait type).  In this case, we must pull the
    // fn pointer out of the vtable that is packaged up with the
    // @/~/&Trait instance.  @/~/&Traits are represented as a pair, so we
    // first evaluate the self expression (expected a by-ref result) and then
    // extract the self data and vtable out of the pair.

    let _icx = bcx.insn_ctxt("impl::trans_trait_callee");
    let mut bcx = bcx;
    let self_datum = unpack_datum!(bcx,
        expr::trans_to_datum(bcx, self_expr));
    let llpair = self_datum.to_ref_llval(bcx);

    let llpair = match explicit_self {
        ast::sty_region(*) => Load(bcx, llpair),
        ast::sty_static | ast::sty_value |
        ast::sty_box(_) | ast::sty_uniq(_) => llpair
    };

    let callee_ty = node_id_type(bcx, callee_id);
    trans_trait_callee_from_llval(bcx,
                                  callee_ty,
                                  n_method,
                                  llpair,
                                  store,
                                  explicit_self)
}

pub fn trans_trait_callee_from_llval(bcx: block,
                                     callee_ty: ty::t,
                                     n_method: uint,
                                     llpair: ValueRef,
                                     store: ty::TraitStore,
                                     explicit_self: ast::explicit_self_)
                                  -> Callee {
    //!
    //
    // Same as `trans_trait_callee()` above, except that it is given
    // a by-ref pointer to the @Trait pair.

    let _icx = bcx.insn_ctxt("impl::trans_trait_callee");
    let ccx = bcx.ccx();
    let mut bcx = bcx;

    // Load the vtable from the @Trait pair
    debug!("(translating trait callee) loading vtable from pair %s",
           val_str(bcx.ccx().tn, llpair));
    let llvtable = Load(bcx,
                      PointerCast(bcx,
                                  GEPi(bcx, llpair,
                                       [0u, abi::trt_field_vtable]),
                                  T_ptr(T_ptr(T_vtable()))));

    // Load the box from the @Trait pair and GEP over the box header if
    // necessary:
    let mut llself;
    debug!("(translating trait callee) loading second index from pair");
    let llbox = Load(bcx, GEPi(bcx, llpair, [0u, abi::trt_field_box]));

    // Munge `llself` appropriately for the type of `self` in the method.
    let self_mode;
    match explicit_self {
        ast::sty_static => {
            bcx.tcx().sess.bug("shouldn't see static method here");
        }
        ast::sty_value => {
            bcx.tcx().sess.bug("methods with by-value self should not be \
                                called on objects");
        }
        ast::sty_region(*) => {
            // As before, we need to pass a pointer to a pointer to the
            // payload.
            match store {
                ty::BoxTraitStore |
                ty::UniqTraitStore => {
                    llself = GEPi(bcx, llbox, [0u, abi::box_field_body]);
                }
                ty::RegionTraitStore(_) => {
                    llself = llbox;
                }
            }

            let llscratch = alloca(bcx, val_ty(llself));
            Store(bcx, llself, llscratch);
            llself = llscratch;

            self_mode = ty::ByRef;
        }
        ast::sty_box(_) => {
            // Bump the reference count on the box.
            debug!("(translating trait callee) callee type is `%s`",
                   bcx.ty_to_str(callee_ty));
            bcx = glue::take_ty(bcx, llbox, callee_ty);

            // Pass a pointer to the box.
            match store {
                ty::BoxTraitStore => llself = llbox,
                _ => bcx.tcx().sess.bug("@self receiver with non-@Trait")
            }

            let llscratch = alloca(bcx, val_ty(llself));
            Store(bcx, llself, llscratch);
            llself = llscratch;

            self_mode = ty::ByRef;
        }
        ast::sty_uniq(_) => {
            // Pass the unique pointer.
            match store {
                ty::UniqTraitStore => llself = llbox,
                _ => bcx.tcx().sess.bug("~self receiver with non-~Trait")
            }

            let llscratch = alloca(bcx, val_ty(llself));
            Store(bcx, llself, llscratch);
            llself = llscratch;

            self_mode = ty::ByRef;
        }
    }

    // Load the function from the vtable and cast it to the expected type.
    debug!("(translating trait callee) loading method");
    let llcallee_ty = type_of_fn_from_ty(ccx, callee_ty);

    // Plus one in order to skip past the type descriptor.
    let mptr = Load(bcx, GEPi(bcx, llvtable, [0u, n_method + 1]));

    let mptr = PointerCast(bcx, mptr, T_ptr(llcallee_ty));

    return Callee {
        bcx: bcx,
        data: Method(MethodData {
            llfn: mptr,
            llself: llself,
            self_ty: ty::mk_opaque_box(bcx.tcx()),
            self_mode: self_mode,
            /* XXX: Some(llbox) */
        })
    };
}

pub fn vtable_id(ccx: @CrateContext,
                 origin: &typeck::vtable_origin)
              -> mono_id {
    match origin {
        &typeck::vtable_static(impl_id, ref substs, sub_vtables) => {
            monomorphize::make_mono_id(
                ccx,
                impl_id,
                *substs,
                if sub_vtables.is_empty() {
                    None
                } else {
                    Some(sub_vtables)
                },
                None,
                None)
        }

        // can't this be checked at the callee?
        _ => fail!("vtable_id")
    }
}

/// Creates a returns a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
pub fn get_vtable(ccx: @CrateContext,
                  self_ty: ty::t,
                  origin: typeck::vtable_origin)
                  -> ValueRef {
    let hash_id = vtable_id(ccx, &origin);
    match ccx.vtables.find(&hash_id) {
        Some(&val) => val,
        None => {
            match origin {
                typeck::vtable_static(id, substs, sub_vtables) => {
                    make_impl_vtable(ccx, id, self_ty, substs, sub_vtables)
                }
                _ => fail!("get_vtable: expected a static origin"),
            }
        }
    }
}

/// Helper function to declare and initialize the vtable.
pub fn make_vtable(ccx: @CrateContext,
                   tydesc: @mut tydesc_info,
                   ptrs: &[ValueRef])
                   -> ValueRef {
    unsafe {
        let _icx = ccx.insn_ctxt("impl::make_vtable");

        let mut components = ~[ tydesc.tydesc ];
        for ptrs.each |&ptr| {
            components.push(ptr)
        }

        let tbl = C_struct(components);
        let vtable = ccx.sess.str_of((ccx.names)("vtable"));
        let vt_gvar = do str::as_c_str(*vtable) |buf| {
            llvm::LLVMAddGlobal(ccx.llmod, val_ty(tbl), buf)
        };
        llvm::LLVMSetInitializer(vt_gvar, tbl);
        llvm::LLVMSetGlobalConstant(vt_gvar, lib::llvm::True);
        lib::llvm::SetLinkage(vt_gvar, lib::llvm::InternalLinkage);
        vt_gvar
    }
}

/// Generates a dynamic vtable for objects.
pub fn make_impl_vtable(ccx: @CrateContext,
                        impl_id: ast::def_id,
                        self_ty: ty::t,
                        substs: ~[ty::t],
                        vtables: typeck::vtable_res)
                        -> ValueRef {
    let _icx = ccx.insn_ctxt("impl::make_impl_vtable");
    let tcx = ccx.tcx;

    let trt_id = match ty::impl_trait_ref(tcx, impl_id) {
        Some(t_id) => t_id.def_id,
        None       => ccx.sess.bug("make_impl_vtable: don't know how to \
                                    make a vtable for a type impl!")
    };

    let has_tps =
        !ty::lookup_item_type(ccx.tcx, impl_id).generics.type_param_defs.is_empty();

    let trait_method_def_ids = ty::trait_method_def_ids(tcx, trt_id);
    let methods = do trait_method_def_ids.map |method_def_id| {
        let im = ty::method(tcx, *method_def_id);
        let fty = ty::subst_tps(tcx,
                                substs,
                                None,
                                ty::mk_bare_fn(tcx, copy im.fty));
        if im.generics.has_type_params() || ty::type_has_self(fty) {
            debug!("(making impl vtable) method has self or type params: %s",
                   *tcx.sess.str_of(im.ident));
            C_null(T_ptr(T_nil()))
        } else {
            debug!("(making impl vtable) adding method to vtable: %s",
                   *tcx.sess.str_of(im.ident));
            let mut m_id = method_with_name(ccx, impl_id, im.ident);
            if has_tps {
                // If the method is in another crate, need to make an inlined
                // copy first
                if m_id.crate != ast::local_crate {
                    // XXX: Set impl ID here?
                    m_id = inline::maybe_instantiate_inline(ccx, m_id, true);
                }
                let (val, _) = monomorphize::monomorphic_fn(ccx, m_id, substs,
                                Some(vtables), None, None);
                val
            } else if m_id.crate == ast::local_crate {
                get_item_val(ccx, m_id.node)
            } else {
                trans_external_path(ccx, m_id, fty)
            }
        }
    };

    // Generate a type descriptor for the vtable.
    let tydesc = get_tydesc(ccx, self_ty);
    glue::lazily_emit_all_tydesc_glue(ccx, tydesc);

    make_vtable(ccx, tydesc, methods)
}

pub fn trans_trait_cast(bcx: block,
                        val: @ast::expr,
                        id: ast::node_id,
                        dest: expr::Dest,
                        _store: ty::TraitStore)
                     -> block {
    let mut bcx = bcx;
    let _icx = bcx.insn_ctxt("impl::trans_cast");

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
                            T_ptr(type_of(bcx.ccx(), v_ty)));
    bcx = expr::trans_into(bcx, val, SaveIn(llboxdest));

    // Store the vtable into the pair or triple.
    let orig = /*bad*/copy ccx.maps.vtable_map.get(&id)[0];
    let orig = resolve_vtable_in_fn_ctxt(bcx.fcx, orig);
    let vtable = get_vtable(bcx.ccx(), v_ty, orig);
    Store(bcx, vtable, PointerCast(bcx,
                                   GEPi(bcx, lldest, [0u, abi::trt_field_vtable]),
                                   T_ptr(val_ty(vtable))));

    bcx
}
