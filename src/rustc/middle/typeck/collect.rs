import astconv::{type_rscope, instantiate_iface_ref, ty_of_item,
                 empty_rscope, ty_of_native_item, ast_conv};

// Item collection - a pair of bootstrap passes:
//
// (1) Collect the IDs of all type items (typedefs) and store them in a table.
//
// (2) Translate the AST fragments that describe types to determine a type for
//     each item. When we encounter a named type, we consult the table built
//     in pass 1 to find its item, and recursively translate it.
//
// We then annotate the AST with the resulting types and return the annotated
// AST, along with a table mapping item IDs to their types.
fn get_enum_variant_types(ccx: @crate_ctxt,
                          enum_ty: ty::t,
                          variants: [ast::variant],
                          ty_params: [ast::ty_param],
                          rp: ast::region_param) {
    let tcx = ccx.tcx;

    // Create a set of parameter types shared among all the variants.
    for variants.each {|variant|
        // Nullary enum constructors get turned into constants; n-ary enum
        // constructors get turned into functions.
        let result_ty = if vec::len(variant.node.args) == 0u {
            enum_ty
        } else {
            let rs = type_rscope(rp);
            let args = variant.node.args.map { |va|
                let arg_ty = ccx.to_ty(rs, va.ty);
                {mode: ast::expl(ast::by_copy), ty: arg_ty}
            };
            ty::mk_fn(tcx, {proto: ast::proto_box,
                            inputs: args,
                            output: enum_ty,
                            ret_style: ast::return_val,
                            constraints: []})
        };
        let tpt = {bounds: astconv::ty_param_bounds(ccx, ty_params),
                   rp: rp,
                   ty: result_ty};
        tcx.tcache.insert(local_def(variant.node.id), tpt);
        write_ty_to_tcx(tcx, variant.node.id, result_ty);
    }
}

fn ensure_iface_methods(ccx: @crate_ctxt, id: ast::node_id) {
    fn store_methods<T>(ccx: @crate_ctxt, id: ast::node_id,
                        stuff: [T], f: fn@(T) -> ty::method) {
        ty::store_iface_methods(ccx.tcx, id, @vec::map(stuff, f));
    }

    let tcx = ccx.tcx;
    alt check tcx.items.get(id) {
      ast_map::node_item(@{node: ast::item_iface(_, rp, ms), _}, _) {
        store_methods::<ast::ty_method>(ccx, id, ms) {|m|
            ty_of_ty_method(ccx, m, rp)
        };
      }
      ast_map::node_item(@{node: ast::item_class(_,_,its,_,_,rp), _}, _) {
        let (_,ms) = split_class_items(its);
        // All methods need to be stored, since lookup_method
        // relies on the same method cache for self-calls
        store_methods::<@ast::method>(ccx, id, ms) {|m|
            ty_of_method(ccx, m, rp)
        };
      }
    }
}

fn check_methods_against_iface(ccx: @crate_ctxt,
                               tps: [ast::ty_param],
                               rp: ast::region_param,
                               selfty: ty::t,
                               a_ifacety: @ast::iface_ref,
                               ms: [@ast::method]) {

    let tcx = ccx.tcx;
    let i_bounds = astconv::ty_param_bounds(ccx, tps);
    let my_methods = convert_methods(ccx, ms, rp, i_bounds, selfty);
    let (did, tpt) = instantiate_iface_ref(ccx, a_ifacety, rp);
    if did.crate == ast::local_crate {
        ensure_iface_methods(ccx, did.node);
    }
    for vec::each(*ty::iface_methods(tcx, did)) {|if_m|
        alt vec::find(my_methods, {|m| if_m.ident == m.mty.ident}) {
          some({mty: m, id, span}) {
            if m.purity != if_m.purity {
                ccx.tcx.sess.span_err(
                    span, #fmt["method `%s`'s purity \
                                not match the iface method's \
                                purity", m.ident]);
            }
            let mt = compare_impl_method(
                ccx.tcx, span, m, vec::len(tps),
                if_m, tpt.substs, selfty);
            let old = tcx.tcache.get(local_def(id));
            if old.ty != mt {
                tcx.tcache.insert(
                    local_def(id),
                    {bounds: old.bounds,
                     rp: old.rp,
                     ty: mt});
                write_ty_to_tcx(tcx, id, mt);
            }
          }
          none {
            tcx.sess.span_err(
                a_ifacety.path.span,
                #fmt["missing method `%s`", if_m.ident]);
          }
        } // alt
    } // |if_m|
} // fn

fn convert_class_item(ccx: @crate_ctxt,
                      rp: ast::region_param,
                      bounds: @[ty::param_bounds],
                      v: ast_util::ivar) {
    /* we want to do something here, b/c within the
    scope of the class, it's ok to refer to fields &
    methods unqualified */
    /* they have these types *within the scope* of the
    class. outside the class, it's done with expr_field */
    let tt = ccx.to_ty(type_rscope(rp), v.ty);
    write_ty_to_tcx(ccx.tcx, v.id, tt);
    /* add the field to the tcache */
    ccx.tcx.tcache.insert(local_def(v.id), {bounds: bounds, rp: rp, ty: tt});
}

fn convert_methods(ccx: @crate_ctxt,
                   ms: [@ast::method],
                   rp: ast::region_param,
                   i_bounds: @[ty::param_bounds],
                   self_ty: ty::t)
    -> [{mty: ty::method, id: ast::node_id, span: span}] {

    let tcx = ccx.tcx;
    vec::map(ms) { |m|
        write_ty_to_tcx(tcx, m.self_id, self_ty);
        let bounds = astconv::ty_param_bounds(ccx, m.tps);
        let mty = ty_of_method(ccx, m, rp);
        let fty = ty::mk_fn(tcx, mty.fty);
        tcx.tcache.insert(
            local_def(m.id),
            // n.b. This code is kind of sketchy (concat'ing i_bounds
            // with bounds), but removing *i_bounds breaks other stuff
            {bounds: @(*i_bounds + *bounds), rp: rp, ty: fty});
        write_ty_to_tcx(tcx, m.id, fty);
        {mty: mty, id: m.id, span: m.span}
    }
}

fn convert(ccx: @crate_ctxt, it: @ast::item) {
    let tcx = ccx.tcx;
    alt it.node {
      // These don't define types.
      ast::item_mod(_) {}
      ast::item_native_mod(m) {
        if syntax::attr::native_abi(it.attrs) ==
            either::right(ast::native_abi_rust_intrinsic) {
            for m.items.each { |item| check_intrinsic_type(ccx, item); }
        }
      }
      ast::item_enum(variants, ty_params, rp) {
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
        get_enum_variant_types(ccx, tpt.ty, variants,
                               ty_params, rp);
      }
      ast::item_impl(tps, rp, ifce, selfty, ms) {
        let i_bounds = astconv::ty_param_bounds(ccx, tps);
        let selfty = ccx.to_ty(type_rscope(rp), selfty);
        write_ty_to_tcx(tcx, it.id, selfty);
        tcx.tcache.insert(local_def(it.id),
                          {bounds: i_bounds,
                           rp: rp,
                           ty: selfty});
        alt ifce {
          some(t) {
            check_methods_against_iface(
                ccx, tps, rp,
                selfty, t, ms);
          }
          _ {
            // Still have to do this to write method types
            // into the table
            convert_methods(
                ccx, ms, rp,
                i_bounds, selfty);
          }
        }
      }
      ast::item_res(decl, tps, _, dtor_id, ctor_id, rp) {
        let {bounds, substs} = mk_substs(ccx, tps, rp);
        let def_id = local_def(it.id);
        let t_arg = astconv::ty_of_arg(ccx, type_rscope(rp),
                                       decl.inputs[0], none);
        let t_res = ty::mk_res(tcx, def_id, t_arg.ty, substs);

        let t_ctor = ty::mk_fn(tcx, {
            proto: ast::proto_box,
            inputs: [{mode: ast::expl(ast::by_copy), ty: t_arg.ty}],
            output: t_res,
            ret_style: ast::return_val, constraints: []
        });
        let t_dtor = ty::mk_fn(tcx, {
            proto: ast::proto_box,
            inputs: [t_arg], output: ty::mk_nil(tcx),
            ret_style: ast::return_val, constraints: []
        });
        write_ty_to_tcx(tcx, it.id, t_res);
        write_ty_to_tcx(tcx, ctor_id, t_ctor);
        tcx.tcache.insert(local_def(ctor_id),
                          {bounds: bounds,
                           rp: rp,
                           ty: t_ctor});
        tcx.tcache.insert(def_id, {bounds: bounds,
                                   rp: rp,
                                   ty: t_res});
        write_ty_to_tcx(tcx, dtor_id, t_dtor);
      }
      ast::item_iface(*) {
        let tpt = ty_of_item(ccx, it);
        #debug["item_iface(it.id=%d, tpt.ty=%s)",
               it.id, ty_to_str(tcx, tpt.ty)];
        write_ty_to_tcx(tcx, it.id, tpt.ty);
        ensure_iface_methods(ccx, it.id);
      }
      ast::item_class(tps, ifaces, members, ctor, m_dtor, rp) {
        // Write the class type
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
        tcx.tcache.insert(local_def(it.id), {bounds: tpt.bounds,
              rp: rp, ty: tpt.ty});
        // Write the ctor type
        let t_ctor =
            ty::mk_fn(
                tcx,
                astconv::ty_of_fn_decl(ccx,
                                       empty_rscope,
                                       ast::proto_any,
                                       ctor.node.dec,
                                       none));
        write_ty_to_tcx(tcx, ctor.node.id, t_ctor);
        tcx.tcache.insert(local_def(ctor.node.id),
                          {bounds: tpt.bounds,
                           rp: ast::rp_none,
                           ty: t_ctor});
        option::iter(m_dtor) {|dtor|
            // Write the dtor type
            let t_dtor = ty::mk_fn(
                tcx,
                // not sure about empty_rscope
                // FIXME
                astconv::ty_of_fn_decl(ccx,
                                       empty_rscope,
                                       ast::proto_any,
                                       ast_util::dtor_dec(),
                                       none));
            write_ty_to_tcx(tcx, dtor.node.id, t_dtor);
            tcx.tcache.insert(local_def(dtor.node.id),
                              {bounds: tpt.bounds,
                               rp: ast::rp_none,
                               ty: t_dtor});
        };
        ensure_iface_methods(ccx, it.id);
        /* FIXME: check for proper public/privateness */
        // Write the type of each of the members
        let (fields, methods) = split_class_items(members);
        for fields.each {|f|
           convert_class_item(ccx, rp, tpt.bounds, f);
        }
        // The selfty is just the class type
        let {bounds:_, substs} = mk_substs(ccx, tps, rp);
        let selfty = ty::mk_class(tcx, local_def(it.id), substs);
        // Need to convert all methods so we can check internal
        // references to private methods

        // NDM to TJC---I think we ought to be using bounds here, not @[].
        // But doing so causes errors later on.
        convert_methods(ccx, methods, rp, @[], selfty);

        /*
        Finally, check that the class really implements the ifaces
        that it claims to implement.
        */
        for ifaces.each { |ifce|
            check_methods_against_iface(ccx, tps, rp, selfty,
                                        ifce, methods);
            let t = ty::node_id_to_type(tcx, ifce.id);

            // FIXME: This assumes classes only implement
            // non-parameterized ifaces. add a test case for
            // a class implementing a parameterized iface.
            // -- tjc (#1726)
            tcx.tcache.insert(local_def(ifce.id), no_params(t));
        }
      }
      _ {
        // This call populates the type cache with the converted type
        // of the item in passing. All we have to do here is to write
        // it into the node type table.
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
      }
    }
}
fn convert_native(ccx: @crate_ctxt, i: @ast::native_item) {
    // As above, this call populates the type table with the converted
    // type of the native item. We simply write it into the node type
    // table.
    let tpt = ty_of_native_item(ccx, i);
    alt i.node {
      ast::native_item_fn(_, _) {
        write_ty_to_tcx(ccx.tcx, i.id, tpt.ty);
      }
    }
}
fn collect_item_types(ccx: @crate_ctxt, crate: @ast::crate) {
    visit::visit_crate(*crate, (), visit::mk_simple_visitor(@{
        visit_item: bind convert(ccx, _),
        visit_native_item: bind convert_native(ccx, _)
        with *visit::default_simple_visitor()
    }));
}

fn ty_of_method(ccx: @crate_ctxt,
                m: @ast::method,
                rp: ast::region_param) -> ty::method {
    {ident: m.ident,
     tps: astconv::ty_param_bounds(ccx, m.tps),
     fty: astconv::ty_of_fn_decl(ccx, type_rscope(rp), ast::proto_bare,
                                 m.decl, none),
     purity: m.decl.purity,
     vis: m.vis}
}

fn ty_of_ty_method(self: @crate_ctxt,
                   m: ast::ty_method,
                   rp: ast::region_param) -> ty::method {
    {ident: m.ident,
     tps: astconv::ty_param_bounds(self, m.tps),
     fty: astconv::ty_of_fn_decl(self, type_rscope(rp), ast::proto_bare,
                                 m.decl, none),
     // assume public, because this is only invoked on iface methods
     purity: m.decl.purity, vis: ast::public}
}
