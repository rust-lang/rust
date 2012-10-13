use libc::c_uint;
use base::*;
use common::*;
use type_of::*;
use build::*;
use driver::session::{session, expect};
use syntax::{ast, ast_map};
use ast_map::{path, path_mod, path_name, node_id_to_str};
use syntax::ast_util::local_def;
use metadata::csearch;
use back::{link, abi};
use lib::llvm::llvm;
use lib::llvm::{ValueRef, TypeRef};
use lib::llvm::llvm::LLVMGetParam;
use std::map::HashMap;
use util::ppaux::{ty_to_str, tys_to_str};
use callee::*;
use syntax::print::pprust::expr_to_str;
use expr::{SaveIn, Ignore};

fn macros() { include!("macros.rs"); } // FIXME(#3114): Macro import/export.

/**
The main "translation" pass for methods.  Generates code
for non-monomorphized methods only.  Other methods will
be generated once they are invoked with specific type parameters,
see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
*/
fn trans_impl(ccx: @crate_ctxt, path: path, name: ast::ident,
              methods: ~[@ast::method], tps: ~[ast::ty_param],
              self_ty: Option<ty::t>) {
    let _icx = ccx.insn_ctxt("impl::trans_impl");
    if tps.len() > 0u { return; }
    let sub_path = vec::append_one(path, path_name(name));
    for vec::each(methods) |method| {
        if method.tps.len() == 0u {
            let llfn = get_item_val(ccx, method.id);
            let path = vec::append_one(sub_path, path_name(method.ident));
            trans_method(ccx, path, *method, None, self_ty, llfn);
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
*/
fn trans_method(ccx: @crate_ctxt,
                path: path,
                method: &ast::method,
                param_substs: Option<param_substs>,
                base_self_ty: Option<ty::t>,
                llfn: ValueRef) {

    // figure out how self is being passed
    let self_arg = match method.self_ty.node {
      ast::sty_static => {
        no_self
      }
      _ => {
        // determine the (monomorphized) type that `self` maps to for
        // this method
        let self_ty;
        match base_self_ty {
            None => self_ty = ty::node_id_to_type(ccx.tcx, method.self_id),
            Some(provided_self_ty) => self_ty = provided_self_ty
        }
        let self_ty = match param_substs {
          None => self_ty,
          Some({tys: ref tys, _}) => ty::subst_tps(ccx.tcx, *tys, self_ty)
        };
        match method.self_ty.node {
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
             method.decl,
             method.body,
             llfn,
             self_arg,
             param_substs,
             method.id);
}

fn trans_self_arg(bcx: block, base: @ast::expr,
                  mentry: typeck::method_map_entry) -> Result {
    let _icx = bcx.insn_ctxt("impl::trans_self_arg");
    let mut temp_cleanups = ~[];
    let self_arg = {mode: mentry.self_arg.mode,
                    ty: monomorphize_type(bcx, mentry.self_arg.ty)};
    let result = trans_arg_expr(bcx, self_arg, base,
                                &mut temp_cleanups, None, DontAutorefArg);

    // FIXME(#3446)---this is wrong, actually.  The temp_cleanups
    // should be revoked only after all arguments have been passed.
    for temp_cleanups.each |c| {
        revoke_clean(bcx, *c)
    }

    return result;
}

fn trans_method_callee(bcx: block, callee_id: ast::node_id,
                       self: @ast::expr, mentry: typeck::method_map_entry)
    -> Callee
{
    let _icx = bcx.insn_ctxt("impl::trans_method_callee");
    match mentry.origin {
        typeck::method_static(did) => {
            let callee_fn = callee::trans_fn_ref(bcx, did, callee_id);
            let Result {bcx, val} = trans_self_arg(bcx, self, mentry);
            let tcx = bcx.tcx();
            Callee {
                bcx: bcx,
                data: Method(MethodData {
                    llfn: callee_fn.llfn,
                    llself: val,
                    self_ty: node_id_type(bcx, self.id),
                    self_mode: ty::resolved_mode(tcx, mentry.self_arg.mode)
                })
            }
        }
        typeck::method_param({trait_id:trait_id, method_num:off,
                              param_num:p, bound_num:b}) => {
            match bcx.fcx.param_substs {
                Some(ref substs) => {
                    let vtbl = base::find_vtable(bcx.tcx(), substs, p, b);
                    trans_monomorphized_callee(bcx, callee_id, self, mentry,
                                               trait_id, off, vtbl)
                }
                // how to get rid of this?
                None => fail ~"trans_method_callee: missing param_substs"
            }
        }
        typeck::method_trait(_, off, vstore) => {
            trans_trait_callee(bcx, callee_id, off, self, vstore)
        }
        typeck::method_self(*) => {
            bcx.tcx().sess.span_bug(self.span, ~"self method call");
        }
    }
}

fn trans_static_method_callee(bcx: block,
                              method_id: ast::def_id,
                              trait_id: ast::def_id,
                              callee_id: ast::node_id) -> FnData
{
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
    //       static fn foo<M1...Mn>(...) {...}
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
        let trait_polyty = ty::lookup_item_type(bcx.tcx(), trait_id);
        let mut index = 0;
        for trait_polyty.bounds.each |param_bounds| {
            for param_bounds.each |param_bound| {
                match *param_bound {
                    ty::bound_trait(_) => { index += 1; }
                    ty::bound_copy | ty::bound_owned |
                    ty::bound_send | ty::bound_const => {}
                }
            }
        }
        index
    };

    let mname = if method_id.crate == ast::local_crate {
        match bcx.tcx().items.get(method_id.node) {
            ast_map::node_trait_method(trait_method, _, _) => {
                ast_util::trait_method_to_ty_method(*trait_method).ident
            }
            _ => fail ~"callee is not a trait method"
        }
    } else {
        let path = csearch::get_item_path(bcx.tcx(), method_id);
        match path[path.len()-1] {
            path_name(s) => { s }
            path_mod(_) => { fail ~"path doesn't have a name?" }
        }
    };
    debug!("trans_static_method_callee: method_id=%?, callee_id=%?, \
            name=%s", method_id, callee_id, ccx.sess.str_of(mname));

    let vtbls = resolve_vtables_in_fn_ctxt(
        bcx.fcx, ccx.maps.vtable_map.get(callee_id));

    match vtbls[bound_index] {
        typeck::vtable_static(impl_did, rcvr_substs, rcvr_origins) => {

            let mth_id = method_with_name(bcx.ccx(), impl_did, mname);
            let callee_substs = combine_impl_and_methods_tps(
                bcx, mth_id, impl_did, callee_id, rcvr_substs);
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
            fail ~"vtable_param left in monomorphized \
                   function's vtable substs";
        }
    }
}

fn method_from_methods(ms: ~[@ast::method], name: ast::ident)
    -> ast::def_id {
  local_def(option::get(&vec::find(ms, |m| m.ident == name)).id)
}

fn method_with_name(ccx: @crate_ctxt, impl_id: ast::def_id,
                    name: ast::ident) -> ast::def_id {
    if impl_id.crate == ast::local_crate {
        match ccx.tcx.items.get(impl_id.node) {
          ast_map::node_item(@{node: ast::item_impl(_, _, _, ms), _}, _) => {
            method_from_methods(ms, name)
          }
          ast_map::node_item(@{node:
              ast::item_class(struct_def, _), _}, _) => {
            method_from_methods(struct_def.methods, name)
          }
          _ => fail ~"method_with_name"
        }
    } else {
        csearch::get_impl_method(ccx.sess.cstore, impl_id, name)
    }
}

fn method_ty_param_count(ccx: @crate_ctxt, m_id: ast::def_id,
                         i_id: ast::def_id) -> uint {
    if m_id.crate == ast::local_crate {
        match ccx.tcx.items.get(m_id.node) {
          ast_map::node_method(m, _, _) => vec::len(m.tps),
          _ => fail ~"method_ty_param_count"
        }
    } else {
        csearch::get_type_param_count(ccx.sess.cstore, m_id) -
            csearch::get_type_param_count(ccx.sess.cstore, i_id)
    }
}

fn trans_monomorphized_callee(bcx: block,
                              callee_id: ast::node_id,
                              base: @ast::expr,
                              mentry: typeck::method_map_entry,
                              trait_id: ast::def_id,
                              n_method: uint,
                              vtbl: typeck::vtable_origin)
    -> Callee
{
    let _icx = bcx.insn_ctxt("impl::trans_monomorphized_callee");
    return match vtbl {
      typeck::vtable_static(impl_did, rcvr_substs, rcvr_origins) => {
          let ccx = bcx.ccx();
          let mname = ty::trait_methods(ccx.tcx, trait_id)[n_method].ident;
          let mth_id = method_with_name(bcx.ccx(), impl_did, mname);

          // obtain the `self` value:
          let Result {bcx, val: llself_val} =
              trans_self_arg(bcx, base, mentry);

          // create a concatenated set of substitutions which includes
          // those from the impl and those from the method:
          let callee_substs = combine_impl_and_methods_tps(
              bcx, mth_id, impl_did, callee_id, rcvr_substs);
          let callee_origins = combine_impl_and_methods_origins(
              bcx, mth_id, impl_did, callee_id, rcvr_origins);

          // translate the function
          let callee = trans_fn_ref_with_vtables(
              bcx, mth_id, callee_id, callee_substs, Some(callee_origins));

          // create a llvalue that represents the fn ptr
          let fn_ty = node_id_type(bcx, callee_id);
          let llfn_ty = T_ptr(type_of_fn_from_ty(ccx, fn_ty));
          let llfn_val = PointerCast(bcx, callee.llfn, llfn_ty);

          // combine the self environment with the rest
          let tcx = bcx.tcx();
          Callee {
              bcx: bcx,
              data: Method(MethodData {
                  llfn: llfn_val,
                  llself: llself_val,
                  self_ty: node_id_type(bcx, base.id),
                  self_mode: ty::resolved_mode(tcx, mentry.self_arg.mode)
              })
          }
      }
      typeck::vtable_trait(_, _) => {
          trans_trait_callee(bcx, callee_id, n_method, base, ty::vstore_box)
      }
      typeck::vtable_param(*) => {
          fail ~"vtable_param left in monomorphized function's vtable substs";
      }
    };

}

fn combine_impl_and_methods_tps(bcx: block,
                                mth_did: ast::def_id,
                                impl_did: ast::def_id,
                                callee_id: ast::node_id,
                                rcvr_substs: ~[ty::t])
    -> ~[ty::t]
{
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
    let ty_substs
        = vec::append(rcvr_substs,
                      vec::tailn(node_substs,
                                 node_substs.len() - n_m_tps));
    debug!("n_m_tps=%?", n_m_tps);
    debug!("rcvr_substs=%?", rcvr_substs.map(|t| bcx.ty_to_str(*t)));
    debug!("node_substs=%?", node_substs.map(|t| bcx.ty_to_str(*t)));
    debug!("ty_substs=%?", ty_substs.map(|t| bcx.ty_to_str(*t)));

    return ty_substs;
}

fn combine_impl_and_methods_origins(bcx: block,
                                    mth_did: ast::def_id,
                                    impl_did: ast::def_id,
                                    callee_id: ast::node_id,
                                    rcvr_origins: typeck::vtable_res)
    -> typeck::vtable_res
{
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
    let {bounds: r_m_bounds, _} = ty::lookup_item_type(tcx, mth_did);
    let n_r_m_tps = r_m_bounds.len(); // rcvr + method tps
    let m_boundss = vec::view(*r_m_bounds, n_r_m_tps - n_m_tps, n_r_m_tps);

    // Flatten out to find the number of vtables the method expects.
    let m_vtables = m_boundss.foldl(0, |sum, m_bounds| {
        m_bounds.foldl(*sum, |sum, m_bound| {
            (*sum) + match (*m_bound) {
                ty::bound_copy | ty::bound_owned |
                ty::bound_send | ty::bound_const => 0,
                ty::bound_trait(_) => 1
            }
        })
    });

    // Find the vtables we computed at type check time and monomorphize them
    let r_m_origins = match node_vtables(bcx, callee_id) {
        Some(vt) => vt,
        None => @~[]
    };

    // Extract those that belong to method:
    let m_origins = vec::tailn(*r_m_origins, r_m_origins.len() - m_vtables);

    // Combine rcvr + method to find the final result:
    @vec::append(*rcvr_origins, m_origins)
}


fn trans_trait_callee(bcx: block,
                      callee_id: ast::node_id,
                      n_method: uint,
                      self_expr: @ast::expr,
                      vstore: ty::vstore)
    -> Callee
{
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
    let self_datum = unpack_datum!(bcx, expr::trans_to_datum(bcx, self_expr));
    let llpair = self_datum.to_ref_llval(bcx);
    let callee_ty = node_id_type(bcx, callee_id);
    trans_trait_callee_from_llval(bcx, callee_ty, n_method, llpair, vstore)
}

fn trans_trait_callee_from_llval(bcx: block,
                                 callee_ty: ty::t,
                                 n_method: uint,
                                 llpair: ValueRef,
                                 vstore: ty::vstore)
    -> Callee
{
    //!
    //
    // Same as `trans_trait_callee()` above, except that it is given
    // a by-ref pointer to the @Trait pair.

    let _icx = bcx.insn_ctxt("impl::trans_trait_callee");
    let ccx = bcx.ccx();
    let mut bcx = bcx;

    // Load the vtable from the @Trait pair
    let llvtable = Load(bcx,
                      PointerCast(bcx,
                                  GEPi(bcx, llpair, [0u, 0u]),
                                  T_ptr(T_ptr(T_vtable()))));

    // Load the box from the @Trait pair and GEP over the box header if
    // necessary:
    let llself;
    let llbox = Load(bcx, GEPi(bcx, llpair, [0u, 1u]));
    match vstore {
        ty::vstore_box | ty::vstore_uniq => {
            llself = GEPi(bcx, llbox, [0u, abi::box_field_body]);
        }
        ty::vstore_slice(_) => {
            llself = llbox;
        }
        ty::vstore_fixed(*) => {
            bcx.tcx().sess.bug(~"vstore_fixed trait");
        }
    }

    // Load the function from the vtable and cast it to the expected type.
    let llcallee_ty = type_of::type_of_fn_from_ty(ccx, callee_ty);
    let mptr = Load(bcx, GEPi(bcx, llvtable, [0u, n_method]));
    let mptr = PointerCast(bcx, mptr, T_ptr(llcallee_ty));

    return Callee {
        bcx: bcx,
        data: Method(MethodData {
            llfn: mptr,
            llself: llself,
            self_ty: ty::mk_opaque_box(bcx.tcx()),
            self_mode: ast::by_ref, // XXX: is this bogosity?
            /* XXX: Some(llbox) */
        })
    };
}

fn vtable_id(ccx: @crate_ctxt, origin: typeck::vtable_origin) -> mono_id {
    match origin {
        typeck::vtable_static(impl_id, substs, sub_vtables) => {
            monomorphize::make_mono_id(
                ccx, impl_id, substs,
                if (*sub_vtables).len() == 0u { None }
                else { Some(sub_vtables) }, None)
        }
        typeck::vtable_trait(trait_id, substs) => {
            @{def: trait_id,
              params: vec::map(substs, |t| mono_precise(*t, None))}
        }
        // can't this be checked at the callee?
        _ => fail ~"vtable_id"
    }
}

fn get_vtable(ccx: @crate_ctxt, origin: typeck::vtable_origin)
    -> ValueRef {
    let hash_id = vtable_id(ccx, origin);
    match ccx.vtables.find(hash_id) {
      Some(val) => val,
      None => match origin {
        typeck::vtable_static(id, substs, sub_vtables) => {
            make_impl_vtable(ccx, id, substs, sub_vtables)
        }
        _ => fail ~"get_vtable: expected a static origin"
      }
    }
}

fn make_vtable(ccx: @crate_ctxt, ptrs: ~[ValueRef]) -> ValueRef {
    let _icx = ccx.insn_ctxt("impl::make_vtable");
    let tbl = C_struct(ptrs);
    let vt_gvar = str::as_c_str(ccx.sess.str_of(ccx.names(~"vtable")), |buf| {
        llvm::LLVMAddGlobal(ccx.llmod, val_ty(tbl), buf)
    });
    llvm::LLVMSetInitializer(vt_gvar, tbl);
    llvm::LLVMSetGlobalConstant(vt_gvar, lib::llvm::True);
    lib::llvm::SetLinkage(vt_gvar, lib::llvm::InternalLinkage);
    vt_gvar
}

fn make_impl_vtable(ccx: @crate_ctxt, impl_id: ast::def_id, substs: ~[ty::t],
                    vtables: typeck::vtable_res) -> ValueRef {
    let _icx = ccx.insn_ctxt("impl::make_impl_vtable");
    let tcx = ccx.tcx;

    // XXX: This should support multiple traits.
    let trt_id = driver::session::expect(
        tcx.sess,
        ty::ty_to_def_id(ty::impl_traits(tcx, impl_id, ty::vstore_box)[0]),
        || ~"make_impl_vtable: non-trait-type implemented");

    let has_tps = (*ty::lookup_item_type(ccx.tcx, impl_id).bounds).len() > 0u;
    make_vtable(ccx, vec::map(*ty::trait_methods(tcx, trt_id), |im| {
        let fty = ty::subst_tps(tcx, substs, ty::mk_fn(tcx, im.fty));
        if (*im.tps).len() > 0u || ty::type_has_self(fty) {
            C_null(T_ptr(T_nil()))
        } else {
            let mut m_id = method_with_name(ccx, impl_id, im.ident);
            if has_tps {
                // If the method is in another crate, need to make an inlined
                // copy first
                if m_id.crate != ast::local_crate {
                    m_id = inline::maybe_instantiate_inline(ccx, m_id);
                }
                monomorphize::monomorphic_fn(ccx, m_id, substs,
                                             Some(vtables), None).val
            } else if m_id.crate == ast::local_crate {
                get_item_val(ccx, m_id.node)
            } else {
                trans_external_path(ccx, m_id, fty)
            }
        }
    }))
}

fn trans_trait_cast(bcx: block,
                    val: @ast::expr,
                    id: ast::node_id,
                    dest: expr::Dest)
    -> block
{
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

    let mut llboxdest = GEPi(bcx, lldest, [0u, 1u]);
    if bcx.tcx().legacy_boxed_traits.contains_key(id) {
        // Allocate an @ box and store the value into it
        let {bcx: new_bcx, box: llbox, body: body} = malloc_boxed(bcx, v_ty);
        bcx = new_bcx;
        add_clean_free(bcx, llbox, heap_shared);
        bcx = expr::trans_into(bcx, val, SaveIn(body));
        revoke_clean(bcx, llbox);

        // Store the @ box into the pair
        Store(bcx, llbox, PointerCast(bcx, llboxdest, T_ptr(val_ty(llbox))));
    } else {
        // Just store the @ box into the pair.
        llboxdest = PointerCast(bcx, llboxdest,
                                T_ptr(type_of::type_of(bcx.ccx(), v_ty)));
        bcx = expr::trans_into(bcx, val, SaveIn(llboxdest));
    }

    // Store the vtable into the pair
    let orig = ccx.maps.vtable_map.get(id)[0];
    let orig = resolve_vtable_in_fn_ctxt(bcx.fcx, orig);
    let vtable = get_vtable(bcx.ccx(), orig);
    Store(bcx, vtable, PointerCast(bcx,
                                   GEPi(bcx, lldest, [0u, 0u]),
                                   T_ptr(val_ty(vtable))));

    bcx
}
