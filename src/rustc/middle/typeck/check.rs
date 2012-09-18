/*

# check.rs

Within the check phase of type check, we check each item one at a time
(bodies of function expressions are checked as part of the containing
function).  Inference is used to supply types wherever they are
unknown.

By far the most complex case is checking the body of a function. This
can be broken down into several distinct phases:

- gather: creates type variables to represent the type of each local
  variable and pattern binding.

- main: the main pass does the lion's share of the work: it
  determines the types of all expressions, resolves
  methods, checks for most invalid conditions, and so forth.  In
  some cases, where a type is unknown, it may create a type or region
  variable and use that as the type of an expression.

  In the process of checking, various constraints will be placed on
  these type variables through the subtyping relationships requested
  through the `demand` module.  The `typeck::infer` module is in charge
  of resolving those constraints.

- regionck: after main is complete, the regionck pass goes over all
  types looking for regions and making sure that they did not escape
  into places they are not in scope.  This may also influence the
  final assignments of the various region variables if there is some
  flexibility.

- vtable: find and records the impls to use for each trait bound that
  appears on a type parameter.

- writeback: writes the final types within a function body, replacing
  type variables with their final inferred types.  These final types
  are written into the `tcx.node_types` table, which should *never* contain
  any reference to a type variable.

## Intermediate types

While type checking a function, the intermediate types for the
expressions, blocks, and so forth contained within the function are
stored in `fcx.node_types` and `fcx.node_type_substs`.  These types
may contain unresolved type variables.  After type checking is
complete, the functions in the writeback module are used to take the
types from this table, resolve them, and then write them into their
permanent home in the type context `ccx.tcx`.

This means that during inferencing you should use `fcx.write_ty()`
and `fcx.expr_ty()` / `fcx.node_ty()` to write/obtain the types of
nodes within the function.

The types of top-level items, which never contain unbound type
variables, are stored directly into the `tcx` tables.

n.b.: A type variable is not the same thing as a type parameter.  A
type variable is rather an "instance" of a type parameter: that is,
given a generic function `fn foo<T>(t: T)`: while checking the
function `foo`, the type `ty_param(0)` refers to the type `T`, which
is treated in abstract.  When `foo()` is called, however, `T` will be
substituted for a fresh type variable `N`.  This variable will
eventually be resolved to some concrete type (which might itself be
type parameter).

*/

use astconv::{ast_conv, ast_path_to_ty, ast_ty_to_ty};
use astconv::{ast_region_to_region};
use middle::ty::{TyVid, vid, FnTyBase, FnMeta, FnSig};
use regionmanip::{replace_bound_regions_in_fn_ty};
use rscope::{anon_rscope, binding_rscope, empty_rscope, in_anon_rscope};
use rscope::{in_binding_rscope, region_scope, type_rscope,
                bound_self_region};
use syntax::ast::ty_i;
use typeck::infer::{resolve_type, force_tvar};
use result::{Result, Ok, Err};
use syntax::print::pprust;
use syntax::parse::token::special_idents;

use std::map::{str_hash, uint_hash};

type self_info = {
    self_ty: ty::t,
    self_id: ast::node_id,
    def_id: ast::def_id,
    explicit_self: ast::self_ty
};

/// Fields that are part of a `fn_ctxt` which are inherited by
/// closures defined within the function.  For example:
///
///     fn foo() {
///         do bar() { ... }
///     }
///
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `fn_ctxt`, but they will
/// share the inherited fields.
struct inherited {
    infcx: infer::infer_ctxt,
    locals: HashMap<ast::node_id, TyVid>,
    node_types: HashMap<ast::node_id, ty::t>,
    node_type_substs: HashMap<ast::node_id, ty::substs>,
    adjustments: HashMap<ast::node_id, @ty::AutoAdjustment>
}

struct fn_ctxt {
    // var_bindings, locals and next_var_id are shared
    // with any nested functions that capture the environment
    // (and with any functions whose environment is being captured).
    self_impl_def_id: Option<ast::def_id>,
    ret_ty: ty::t,
    // Used by loop bodies that return from the outer function
    indirect_ret_ty: Option<ty::t>,
    purity: ast::purity,

    // Sometimes we generate region pointers where the precise region
    // to use is not known. For example, an expression like `&x.f`
    // where `x` is of type `@T`: in this case, we will be rooting
    // `x` onto the stack frame, and we could choose to root it until
    // the end of (almost) any enclosing block or expression.  We
    // want to pick the narrowest block that encompasses all uses.
    //
    // What we do in such cases is to generate a region variable with
    // `region_lb` as a lower bound.  The regionck pass then adds
    // other constriants based on how the variable is used and region
    // inference selects the ultimate value.  Finally, borrowck is
    // charged with guaranteeing that the value whose address was taken
    // can actually be made to live as long as it needs to live.
    mut region_lb: ast::node_id,

    in_scope_regions: isr_alist,

    inh: @inherited,

    ccx: @crate_ctxt,
}

fn blank_inherited(ccx: @crate_ctxt) -> @inherited {
    @inherited {
        infcx: infer::new_infer_ctxt(ccx.tcx),
        locals: int_hash(),
        node_types: map::int_hash(),
        node_type_substs: map::int_hash(),
        adjustments: map::int_hash()
    }
}

// Used by check_const and check_enum_variants
fn blank_fn_ctxt(ccx: @crate_ctxt, rty: ty::t,
                 region_bnd: ast::node_id) -> @fn_ctxt {
// It's kind of a kludge to manufacture a fake function context
// and statement context, but we might as well do write the code only once
    @fn_ctxt {
        self_impl_def_id: None,
        ret_ty: rty,
        indirect_ret_ty: None,
        purity: ast::pure_fn,
        mut region_lb: region_bnd,
        in_scope_regions: @Nil,
        inh: blank_inherited(ccx),
        ccx: ccx
    }
}

// a list of mapping from in-scope-region-names ("isr") to the
// corresponding ty::region
type isr_alist = @List<(ty::bound_region, ty::region)>;

trait get_and_find_region {
    fn get(br: ty::bound_region) -> ty::region;
    fn find(br: ty::bound_region) -> Option<ty::region>;
}

impl isr_alist: get_and_find_region {
    fn get(br: ty::bound_region) -> ty::region {
        option::get(self.find(br))
    }

    fn find(br: ty::bound_region) -> Option<ty::region> {
        for list::each(self) |isr| {
            let (isr_br, isr_r) = isr;
            if isr_br == br { return Some(isr_r); }
        }
        return None;
    }
}

fn check_item_types(ccx: @crate_ctxt, crate: @ast::crate) {
    let visit = visit::mk_simple_visitor(@{
        visit_item: |a| check_item(ccx, a),
        .. *visit::default_simple_visitor()
    });
    visit::visit_crate(*crate, (), visit);
}

fn check_bare_fn(ccx: @crate_ctxt,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 id: ast::node_id,
                 self_info: Option<self_info>) {
    let fty = ty::node_id_to_type(ccx.tcx, id);
    match ty::get(fty).sty {
        ty::ty_fn(ref fn_ty) => {
            check_fn(ccx, self_info, fn_ty, decl, body, false, None)
        }
        _ => ccx.tcx.sess.impossible_case(body.span,
                                 "check_bare_fn: function type expected")
    }
}

fn check_fn(ccx: @crate_ctxt,
            self_info: Option<self_info>,
            fn_ty: &ty::FnTy,
            decl: ast::fn_decl,
            body: ast::blk,
            indirect_ret: bool,
            old_fcx: Option<@fn_ctxt>) {

    let tcx = ccx.tcx;

    // ______________________________________________________________________
    // First, we have to replace any bound regions in the fn and self
    // types with free ones.  The free region references will be bound
    // the node_id of the body block.

    let {isr, self_info, fn_ty} = {
        let old_isr = option::map_default(old_fcx, @Nil,
                                         |fcx| fcx.in_scope_regions);
        replace_bound_regions_in_fn_ty(tcx, old_isr, self_info, fn_ty,
                                       |br| ty::re_free(body.node.id, br))
    };

    let arg_tys = fn_ty.sig.inputs.map(|a| a.ty);
    let ret_ty = fn_ty.sig.output;

    debug!("check_fn(arg_tys=%?, ret_ty=%?, self_info.self_ty=%?)",
           arg_tys.map(|a| ty_to_str(tcx, a)),
           ty_to_str(tcx, ret_ty),
           option::map(self_info, |s| ty_to_str(tcx, s.self_ty)));

    // ______________________________________________________________________
    // Create the function context.  This is either derived from scratch or,
    // in the case of function expressions, based on the outer context.
    let fcx: @fn_ctxt = {
        let (purity, inherited) = match old_fcx {
            None => {
                (fn_ty.meta.purity,
                 blank_inherited(ccx))
            }
            Some(fcx) => {
                (ty::determine_inherited_purity(fcx.purity, fn_ty.meta.purity,
                                                fn_ty.meta.proto),
                 fcx.inh)
            }
        };

        let indirect_ret_ty = if indirect_ret {
            let ofcx = option::get(old_fcx);
            match ofcx.indirect_ret_ty {
              Some(t) => Some(t),
              None => Some(ofcx.ret_ty)
            }
        } else { None };

        @fn_ctxt {
            self_impl_def_id: self_info.map(|info| info.def_id),
            ret_ty: ret_ty,
            indirect_ret_ty: indirect_ret_ty,
            purity: purity,
            mut region_lb: body.node.id,
            in_scope_regions: isr,
            inh: inherited,
            ccx: ccx
        }
    };

    // Update the self_info to contain an accurate self type (taking
    // into account explicit self).
    let self_info = do self_info.chain |info| {
        // If the self type is sty_static, we don't have a self ty.
        if info.explicit_self.node == ast::sty_static {
            None
        } else  {
            let self_region = fcx.in_scope_regions.find(ty::br_self);
            let ty = method::transform_self_type_for_method(
                fcx.tcx(), self_region,
                info.self_ty, info.explicit_self.node);
            Some({self_ty: ty,.. info})
        }
    };

    gather_locals(fcx, decl, body, arg_tys, self_info);
    check_block(fcx, body);

    // We unify the tail expr's type with the
    // function result type, if there is a tail expr.
    match body.node.expr {
      Some(tail_expr) => {
        let tail_expr_ty = fcx.expr_ty(tail_expr);
        demand::suptype(fcx, tail_expr.span, fcx.ret_ty, tail_expr_ty);
      }
      None => ()
    }

    for self_info.each |info| {
        fcx.write_ty(info.self_id, info.self_ty);
    }
    do vec::iter2(decl.inputs, arg_tys) |input, arg| {
        fcx.write_ty(input.id, arg);
    }

    // If we don't have any enclosing function scope, it is time to
    // force any remaining type vars to be resolved.
    // If we have an enclosing function scope, our type variables will be
    // resolved when the enclosing scope finishes up.
    if option::is_none(old_fcx) {
        vtable::resolve_in_block(fcx, body);
        regionck::regionck_fn(fcx, decl, body);
        writeback::resolve_type_vars_in_fn(fcx, decl, body, self_info);
    }

    fn gather_locals(fcx: @fn_ctxt,
                     decl: ast::fn_decl,
                     body: ast::blk,
                     arg_tys: ~[ty::t],
                     self_info: Option<self_info>) {
        let tcx = fcx.ccx.tcx;

        let assign = fn@(span: span, nid: ast::node_id,
                         ty_opt: Option<ty::t>) {
            let var_id = fcx.infcx().next_ty_var_id();
            fcx.inh.locals.insert(nid, var_id);
            match ty_opt {
                None => {/* nothing to do */ }
                Some(typ) => {
                    infer::mk_eqty(fcx.infcx(), false, span,
                                   ty::mk_var(tcx, var_id), typ);
                }
            }
        };

        // Add the self parameter
        for self_info.each |info| {
            assign(info.explicit_self.span,
                   info.self_id, Some(info.self_ty));
            debug!("self is assigned to %s",
                   fcx.inh.locals.get(info.self_id).to_str());
        }

        // Add formal parameters.
        do vec::iter2(arg_tys, decl.inputs) |arg_ty, input| {
            assign(input.ty.span, input.id, Some(arg_ty));
            debug!("Argument %s is assigned to %s",
                   tcx.sess.str_of(input.ident),
                   fcx.inh.locals.get(input.id).to_str());
        }

        // Add explicitly-declared locals.
        let visit_local = fn@(local: @ast::local,
                              &&e: (), v: visit::vt<()>) {
            let o_ty = match local.node.ty.node {
              ast::ty_infer => None,
              _ => Some(fcx.to_ty(local.node.ty))
            };
            assign(local.span, local.node.id, o_ty);
            debug!("Local variable %s is assigned to %s",
                   pat_to_str(local.node.pat, tcx.sess.intr()),
                   fcx.inh.locals.get(local.node.id).to_str());
            visit::visit_local(local, e, v);
        };

        // Add pattern bindings.
        let visit_pat = fn@(p: @ast::pat, &&e: (), v: visit::vt<()>) {
            match p.node {
              ast::pat_ident(_, path, _)
                  if !pat_util::pat_is_variant(fcx.ccx.tcx.def_map, p) => {
                assign(p.span, p.id, None);
                debug!("Pattern binding %s is assigned to %s",
                       tcx.sess.str_of(path.idents[0]),
                       fcx.inh.locals.get(p.id).to_str());
              }
              _ => {}
            }
            visit::visit_pat(p, e, v);
        };

        let visit_block = fn@(b: ast::blk, &&e: (), v: visit::vt<()>) {
            // non-obvious: the `blk` variable maps to region lb, so
            // we have to keep this up-to-date.  This
            // is... unfortunate.  It'd be nice to not need this.
            do fcx.with_region_lb(b.node.id) {
                visit::visit_block(b, e, v);
            }
        };

        // Don't descend into fns and items
        fn visit_fn(_fk: visit::fn_kind, _decl: ast::fn_decl,
                    _body: ast::blk, _sp: span,
                    _id: ast::node_id, &&_t: (), _v: visit::vt<()>) {
        }
        fn visit_item(_i: @ast::item, &&_e: (), _v: visit::vt<()>) { }

        let visit = visit::mk_vt(@{visit_local: visit_local,
                                   visit_pat: visit_pat,
                                   visit_fn: visit_fn,
                                   visit_item: visit_item,
                                   visit_block: visit_block,
                                   .. *visit::default_visitor()});

        visit.visit_block(body, (), visit);
    }
}

fn check_method(ccx: @crate_ctxt, method: @ast::method,
                self_ty: ty::t, self_impl_def_id: ast::def_id) {
    let self_info = {self_ty: self_ty,
                     self_id: method.self_id,
                     def_id: self_impl_def_id,
                     explicit_self: method.self_ty };
    check_bare_fn(ccx, method.decl, method.body, method.id, Some(self_info));
}

fn check_no_duplicate_fields(tcx: ty::ctxt, fields:
                             ~[(ast::ident, span)]) {
    let field_names = uint_hash();

    for fields.each |p| {
        let (id, sp) = p;
        match field_names.find(id) {
          Some(orig_sp) => {
            tcx.sess.span_err(sp, fmt!("Duplicate field \
                                   name %s in record type declaration",
                                        tcx.sess.str_of(id)));
            tcx.sess.span_note(orig_sp, ~"First declaration of \
                                          this field occurred here");
            break;
          }
          None => {
            field_names.insert(id, sp);
          }
        }
    }

}

fn check_struct(ccx: @crate_ctxt, struct_def: @ast::struct_def,
                id: ast::node_id, span: span) {
    let tcx = ccx.tcx;
    let self_ty = ty::node_id_to_type(tcx, id);

    do option::iter(struct_def.ctor) |ctor| {
        let class_t = {self_ty: self_ty,
                       self_id: ctor.node.self_id,
                       def_id: local_def(id),
                       explicit_self: {node: ast::sty_by_ref,
                                       span: ast_util::dummy_sp()}};
        // typecheck the ctor
        check_bare_fn(ccx, ctor.node.dec,
                      ctor.node.body, ctor.node.id,
                      Some(class_t));
    }

    do option::iter(struct_def.dtor) |dtor| {
        let class_t = {self_ty: self_ty,
                       self_id: dtor.node.self_id,
                       def_id: local_def(id),
                       explicit_self: {node: ast::sty_by_ref,
                                       span: ast_util::dummy_sp()}};
        // typecheck the dtor
        check_bare_fn(ccx, ast_util::dtor_dec(),
                      dtor.node.body, dtor.node.id,
                      Some(class_t));
    };

    // typecheck the methods
    for struct_def.methods.each |m| {
        check_method(ccx, m, self_ty, local_def(id));
    }
    // Check that there's at least one field
    if struct_def.fields.len() < 1u {
        ccx.tcx.sess.span_err(span, ~"a struct must have at least one field");
    }
    // Check that the class is instantiable
    check_instantiable(ccx.tcx, span, id);
}

fn check_item(ccx: @crate_ctxt, it: @ast::item) {
    debug!("check_item(it.id=%d, it.ident=%s)",
           it.id,
           ty::item_path_str(ccx.tcx, local_def(it.id)));
    let _indenter = indenter();

    match it.node {
      ast::item_const(_, e) => check_const(ccx, it.span, e, it.id),
      ast::item_enum(enum_definition, _) => {
        check_enum_variants(ccx, it.span, enum_definition.variants, it.id);
      }
      ast::item_fn(decl, _, _, body) => {
        check_bare_fn(ccx, decl, body, it.id, None);
      }
      ast::item_impl(_, _, ty, ms) => {
        let rp = ccx.tcx.region_paramd_items.find(it.id);
        debug!("item_impl %s with id %d rp %?",
               ccx.tcx.sess.str_of(it.ident), it.id, rp);
        let self_ty = ccx.to_ty(rscope::type_rscope(rp), ty);
        for ms.each |m| {
            check_method(ccx, m, self_ty, local_def(it.id));
        }
      }
      ast::item_trait(_, _, trait_methods) => {
        for trait_methods.each |trait_method| {
            match trait_method {
              required(*) => {
                // Nothing to do, since required methods don't have
                // bodies to check.
              }
              provided(m) => {
                check_method(ccx, m, ty::mk_self(ccx.tcx), local_def(it.id));
              }
            }
        }
      }
      ast::item_class(struct_def, _) => {
        check_struct(ccx, struct_def, it.id, it.span);
      }
      ast::item_ty(t, tps) => {
        let tpt_ty = ty::node_id_to_type(ccx.tcx, it.id);
        check_bounds_are_used(ccx, t.span, tps, tpt_ty);
        // If this is a record ty, check for duplicate fields
        match t.node {
            ast::ty_rec(fields) => {
              check_no_duplicate_fields(ccx.tcx, fields.map(|f|
                                              (f.node.ident, f.span)));
            }
            _ => ()
        }
      }
      ast::item_foreign_mod(m) => {
        if syntax::attr::foreign_abi(it.attrs) ==
            either::Right(ast::foreign_abi_rust_intrinsic) {
            for m.items.each |item| {
                check_intrinsic_type(ccx, item);
            }
        } else {
            for m.items.each |item| {
                let tpt = ty::lookup_item_type(ccx.tcx, local_def(item.id));
                if (*tpt.bounds).is_not_empty() {
                    ccx.tcx.sess.span_err(
                        item.span,
                        fmt!("foreign items may not have type parameters"));
                }
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

impl @fn_ctxt: ast_conv {
    fn infcx() -> infer::infer_ctxt { self.inh.infcx }
    fn tcx() -> ty::ctxt { self.ccx.tcx }
    fn ccx() -> @crate_ctxt { self.ccx }

    fn get_item_ty(id: ast::def_id) -> ty::ty_param_bounds_and_ty {
        ty::lookup_item_type(self.tcx(), id)
    }

    fn ty_infer(_span: span) -> ty::t {
        self.infcx().next_ty_var()
    }
}

impl @fn_ctxt {
    fn search_in_scope_regions(br: ty::bound_region)
        -> Result<ty::region, ~str>
    {
        match self.in_scope_regions.find(br) {
            Some(r) => result::Ok(r),
            None => {
                let blk_br = ty::br_named(special_idents::blk);
                if br == blk_br {
                    result::Ok(self.block_region())
                } else {
                    result::Err(fmt!("named region `%s` not in scope here",
                                     bound_region_to_str(self.tcx(), br)))
                }
            }
        }
    }
}

impl @fn_ctxt: region_scope {
    fn anon_region(span: span) -> Result<ty::region, ~str> {
        result::Ok(self.infcx().next_region_var_nb(span))
    }
    fn self_region(_span: span) -> Result<ty::region, ~str> {
        self.search_in_scope_regions(ty::br_self)
    }
    fn named_region(_span: span, id: ast::ident) -> Result<ty::region, ~str> {
        self.search_in_scope_regions(ty::br_named(id))
    }
}

impl @fn_ctxt {
    fn tag() -> ~str { fmt!("%x", ptr::addr_of(*self) as uint) }

    fn expr_to_str(expr: @ast::expr) -> ~str {
        fmt!("expr(%?:%s)", expr.id,
             pprust::expr_to_str(expr, self.tcx().sess.intr()))
    }

    fn block_region() -> ty::region {
        ty::re_scope(self.region_lb)
    }

    #[inline(always)]
    fn write_ty(node_id: ast::node_id, ty: ty::t) {
        debug!("write_ty(%d, %s) in fcx %s",
               node_id, ty_to_str(self.tcx(), ty), self.tag());
        self.inh.node_types.insert(node_id, ty);
    }

    fn write_substs(node_id: ast::node_id, +substs: ty::substs) {
        if !ty::substs_is_noop(&substs) {
            debug!("write_substs(%d, %s) in fcx %s",
                   node_id,
                   ty::substs_to_str(self.tcx(), &substs),
                   self.tag());
            self.inh.node_type_substs.insert(node_id, substs);
        }
    }

    fn write_ty_substs(node_id: ast::node_id, ty: ty::t,
                       +substs: ty::substs) {
        let ty = ty::subst(self.tcx(), &substs, ty);
        self.write_ty(node_id, ty);
        self.write_substs(node_id, substs);
    }

    fn write_autoderef_adjustment(node_id: ast::node_id, derefs: uint) {
        if derefs == 0 { return; }
        self.write_adjustment(node_id, @{autoderefs: derefs, autoref: None});
    }

    fn write_adjustment(node_id: ast::node_id, adj: @ty::AutoAdjustment) {
        debug!("write_adjustment(node_id=%?, adj=%?)", node_id, adj);
        self.inh.adjustments.insert(node_id, adj);
    }

    fn write_nil(node_id: ast::node_id) {
        self.write_ty(node_id, ty::mk_nil(self.tcx()));
    }
    fn write_bot(node_id: ast::node_id) {
        self.write_ty(node_id, ty::mk_bot(self.tcx()));
    }

    fn to_ty(ast_t: @ast::ty) -> ty::t {
        ast_ty_to_ty(self, self, ast_t)
    }

    fn expr_to_str(expr: @ast::expr) -> ~str {
        expr_repr(self.tcx(), expr)
    }

    fn expr_ty(ex: @ast::expr) -> ty::t {
        match self.inh.node_types.find(ex.id) {
            Some(t) => t,
            None => {
                self.tcx().sess.bug(
                    fmt!("no type for %s in fcx %s",
                         self.expr_to_str(ex), self.tag()));
            }
        }
    }
    fn node_ty(id: ast::node_id) -> ty::t {
        match self.inh.node_types.find(id) {
            Some(t) => t,
            None => {
                self.tcx().sess.bug(
                    fmt!("no type for node %d: %s in fcx %s",
                         id, ast_map::node_id_to_str(
                             self.tcx().items, id,
                             self.tcx().sess.parse_sess.interner),
                         self.tag()));
            }
        }
    }
    fn node_ty_substs(id: ast::node_id) -> ty::substs {
        match self.inh.node_type_substs.find(id) {
            Some(ts) => ts,
            None => {
                self.tcx().sess.bug(
                    fmt!("no type substs for node %d: %s in fcx %s",
                         id, ast_map::node_id_to_str(
                             self.tcx().items, id,
                             self.tcx().sess.parse_sess.interner),
                         self.tag()));
            }
        }
    }
    fn opt_node_ty_substs(id: ast::node_id) -> Option<ty::substs> {
        self.inh.node_type_substs.find(id)
    }

    fn report_mismatched_types(sp: span, e: ty::t, a: ty::t,
                               err: &ty::type_err) {
        self.ccx.tcx.sess.span_err(
            sp,
            fmt!("mismatched types: expected `%s` but found `%s` (%s)",
                 self.infcx().ty_to_str(e),
                 self.infcx().ty_to_str(a),
                 ty::type_err_to_str(self.ccx.tcx, err)));
        ty::note_and_explain_type_err(self.ccx.tcx, err);
    }

    fn mk_subty(a_is_expected: bool, span: span,
                sub: ty::t, sup: ty::t) -> Result<(), ty::type_err> {
        infer::mk_subty(self.infcx(), a_is_expected, span, sub, sup)
    }

    fn can_mk_subty(sub: ty::t, sup: ty::t) -> Result<(), ty::type_err> {
        infer::can_mk_subty(self.infcx(), sub, sup)
    }

    fn mk_assignty(expr: @ast::expr, sub: ty::t, sup: ty::t)
        -> Result<(), ty::type_err>
    {
        match infer::mk_assignty(self.infcx(), false, expr.span, sub, sup) {
            Ok(None) => result::Ok(()),
            Err(e) => result::Err(e),
            Ok(Some(adjustment)) => {
                self.write_adjustment(expr.id, adjustment);
                Ok(())
            }
        }
    }

    fn can_mk_assignty(sub: ty::t, sup: ty::t) -> Result<(), ty::type_err> {
        infer::can_mk_assignty(self.infcx(), sub, sup)
    }

    fn mk_eqty(a_is_expected: bool, span: span,
               sub: ty::t, sup: ty::t) -> Result<(), ty::type_err> {
        infer::mk_eqty(self.infcx(), a_is_expected, span, sub, sup)
    }

    fn mk_subr(a_is_expected: bool, span: span,
               sub: ty::region, sup: ty::region) -> Result<(), ty::type_err> {
        infer::mk_subr(self.infcx(), a_is_expected, span, sub, sup)
    }

    fn require_unsafe(sp: span, op: ~str) {
        match self.purity {
          ast::unsafe_fn => {/*ok*/}
          _ => {
            self.ccx.tcx.sess.span_err(
                sp,
                fmt!("%s requires unsafe function or block", op));
          }
        }
    }
    fn with_region_lb<R>(lb: ast::node_id, f: fn() -> R) -> R {
        let old_region_lb = self.region_lb;
        self.region_lb = lb;
        let v <- f();
        self.region_lb = old_region_lb;
        move v
    }

    fn region_var_if_parameterized(rp: Option<ty::region_variance>,
                                   span: span,
                                   lower_bound: ty::region)
        -> Option<ty::region>
    {
        rp.map(
            |_rp| self.infcx().next_region_var_with_lb(span, lower_bound))
    }
}

fn do_autoderef(fcx: @fn_ctxt, sp: span, t: ty::t) -> (ty::t, uint) {
    /*!
     *
     * Autoderefs the type `t` as many times as possible, returning
     * a new type and a counter for how many times the type was
     * deref'd.  If the counter is non-zero, the receiver is responsible
     * for inserting an AutoAdjustment record into `tcx.adjustments`
     * so that trans/borrowck/etc know about this autoderef. */

    let mut t1 = t;
    let mut enum_dids = ~[];
    let mut autoderefs = 0;
    loop {
        let sty = structure_of(fcx, sp, t1);

        // Some extra checks to detect weird cycles and so forth:
        match sty {
            ty::ty_box(inner) | ty::ty_uniq(inner) |
            ty::ty_rptr(_, inner) => {
                match ty::get(t1).sty {
                    ty::ty_infer(ty::TyVar(v1)) => {
                        ty::occurs_check(fcx.ccx.tcx, sp, v1,
                                         ty::mk_box(fcx.ccx.tcx, inner));
                    }
                    _ => ()
                }
            }
            ty::ty_enum(did, _) => {
                // Watch out for a type like `enum t = @t`.  Such a
                // type would otherwise infinitely auto-deref.  Only
                // autoderef loops during typeck (basically, this one
                // and the loops in typeck::check::method) need to be
                // concerned with this, as an error will be reported
                // on the enum definition as well because the enum is
                // not instantiable.
                if vec::contains(enum_dids, did) {
                    return (t1, autoderefs);
                }
                vec::push(enum_dids, did);
            }
            _ => { /*ok*/ }
        }

        // Otherwise, deref if type is derefable:
        match ty::deref_sty(fcx.ccx.tcx, &sty, false) {
            None => {
                return (t1, autoderefs);
            }
            Some(mt) => {
                autoderefs += 1;
                t1 = mt.ty
            }
        }
    };
}

// AST fragment checking
fn check_lit(fcx: @fn_ctxt, lit: @ast::lit) -> ty::t {
    let tcx = fcx.ccx.tcx;

    match lit.node {
      ast::lit_str(*) => ty::mk_estr(tcx, ty::vstore_slice(ty::re_static)),
      ast::lit_int(_, t) => ty::mk_mach_int(tcx, t),
      ast::lit_uint(_, t) => ty::mk_mach_uint(tcx, t),
      ast::lit_int_unsuffixed(_) => {
        // An unsuffixed integer literal could have any integral type,
        // so we create an integral type variable for it.
        ty::mk_int_var(tcx, fcx.infcx().next_int_var_id())
      }
      ast::lit_float(_, t) => ty::mk_mach_float(tcx, t),
      ast::lit_nil => ty::mk_nil(tcx),
      ast::lit_bool(_) => ty::mk_bool(tcx)
    }
}

fn valid_range_bounds(ccx: @crate_ctxt, from: @ast::expr, to: @ast::expr)
    -> bool {
    const_eval::compare_lit_exprs(ccx.tcx, from, to) <= 0
}

fn check_expr_with(fcx: @fn_ctxt, expr: @ast::expr, expected: ty::t) -> bool {
    check_expr(fcx, expr, Some(expected))
}

fn check_expr(fcx: @fn_ctxt, expr: @ast::expr,
              expected: Option<ty::t>) -> bool {
    return do check_expr_with_unifier(fcx, expr, expected) {
        for expected.each |t| {
            demand::suptype(fcx, expr.span, t, fcx.expr_ty(expr));
        }
    };
}

// determine the `self` type, using fresh variables for all variables
// declared on the impl declaration e.g., `impl<A,B> for ~[(A,B)]`
// would return ($0, $1) where $0 and $1 are freshly instantiated type
// variables.
fn impl_self_ty(fcx: @fn_ctxt,
                expr: @ast::expr, // (potential) receiver for this impl
                did: ast::def_id) -> ty_param_substs_and_ty {
    let tcx = fcx.ccx.tcx;

    let {n_tps, region_param, raw_ty} = if did.crate == ast::local_crate {
        let region_param = fcx.tcx().region_paramd_items.find(did.node);
        match tcx.items.find(did.node) {
          Some(ast_map::node_item(@{node: ast::item_impl(ts, _, st, _),
                                  _}, _)) => {
            {n_tps: ts.len(),
             region_param: region_param,
             raw_ty: fcx.ccx.to_ty(rscope::type_rscope(region_param), st)}
          }
          Some(ast_map::node_item(@{node: ast::item_class(_, ts),
                                    id: class_id, _},_)) => {
              /* If the impl is a class, the self ty is just the class ty
                 (doing a no-op subst for the ty params; in the next step,
                 we substitute in fresh vars for them)
               */
              {n_tps: ts.len(),
               region_param: region_param,
               raw_ty: ty::mk_class(tcx, local_def(class_id),
                      {self_r: rscope::bound_self_region(region_param),
                       self_ty: None,
                       tps: ty::ty_params_to_tys(tcx, ts)})}
          }
          _ => { tcx.sess.bug(~"impl_self_ty: unbound item or item that \
               doesn't have a self_ty"); }
        }
    } else {
        let ity = ty::lookup_item_type(tcx, did);
        {n_tps: vec::len(*ity.bounds),
         region_param: ity.region_param,
         raw_ty: ity.ty}
    };

    let self_r = if region_param.is_some() {
        Some(fcx.infcx().next_region_var(expr.span, expr.id))
    } else {
        None
    };
    let tps = fcx.infcx().next_ty_vars(n_tps);

    let substs = {self_r: self_r, self_ty: None, tps: tps};
    let substd_ty = ty::subst(tcx, &substs, raw_ty);
    {substs: substs, ty: substd_ty}
}

// Only for fields! Returns <none> for methods>
// Indifferent to privacy flags
fn lookup_field_ty(tcx: ty::ctxt,
                   class_id: ast::def_id,
                   items: &[ty::field_ty],
                   fieldname: ast::ident,
                   substs: &ty::substs) -> Option<ty::t> {

    let o_field = vec::find(items, |f| f.ident == fieldname);
    do option::map(o_field) |f| {
        ty::lookup_field_type(tcx, class_id, f.id, substs)
    }
}

fn check_expr_with_unifier(fcx: @fn_ctxt,
                           expr: @ast::expr,
                           expected: Option<ty::t>,
                           unifier: fn()) -> bool {

    debug!(">> typechecking %s", fcx.expr_to_str(expr));

    // A generic function to factor out common logic from call and
    // overloaded operations
    fn check_call_inner(
        fcx: @fn_ctxt,
        sp: span,
        call_expr_id: ast::node_id,
        in_fty: ty::t,
        callee_expr: @ast::expr,
        args: ~[@ast::expr]) -> {fty: ty::t, bot: bool} {

        let mut bot = false;

        // Replace all region parameters in the arguments and return
        // type with fresh region variables.

        debug!("check_call_inner: before universal quant., in_fty=%s",
               fcx.infcx().ty_to_str(in_fty));

        // This is subtle: we expect `fty` to be a function type, which
        // normally introduce a level of binding.  In this case, we want to
        // process the types bound by the function but not by any nested
        // functions.  Therefore, we match one level of structure.
        let fn_ty =
            match structure_of(fcx, sp, in_fty) {
              ty::ty_fn(ref fn_ty) => {
                replace_bound_regions_in_fn_ty(
                    fcx.ccx.tcx, @Nil, None, fn_ty,
                    |_br| fcx.infcx().next_region_var(sp,
                                                      call_expr_id)).fn_ty
              }
              _ => {
                // I would like to make this span_err, but it's
                // really hard due to the way that expr_bind() is
                // written.
                fcx.ccx.tcx.sess.span_fatal(sp, ~"mismatched types: \
                                            expected function or foreign \
                                            function but found "
                                            + fcx.infcx().ty_to_str(in_fty));
              }
            };

        let fty = ty::mk_fn(fcx.tcx(), fn_ty);
        debug!("check_call_inner: after universal quant., fty=%s",
               fcx.infcx().ty_to_str(fty));

        let supplied_arg_count = vec::len(args);

        // Grab the argument types, supplying fresh type variables
        // if the wrong number of arguments were supplied
        let expected_arg_count = vec::len(fn_ty.sig.inputs);
        let formal_tys = if expected_arg_count == supplied_arg_count {
            fn_ty.sig.inputs.map(|a| a.ty)
        } else {
            fcx.ccx.tcx.sess.span_err(
                sp, fmt!("this function takes %u parameter%s but %u \
                          parameter%s supplied", expected_arg_count,
                         if expected_arg_count == 1u {
                             ~""
                         } else {
                             ~"s"
                         },
                         supplied_arg_count,
                         if supplied_arg_count == 1u {
                             ~" was"
                         } else {
                             ~"s were"
                         }));
            fcx.infcx().next_ty_vars(supplied_arg_count)
        };

        // Check the arguments.
        // We do this in a pretty awful way: first we typecheck any arguments
        // that are not anonymous functions, then we typecheck the anonymous
        // functions. This is so that we have more information about the types
        // of arguments when we typecheck the functions. This isn't really the
        // right way to do this.
        for [false, true]/_.each |check_blocks| {
            debug!("check_blocks=%b", check_blocks);

            // More awful hacks: before we check the blocks, try to do
            // an "opportunistic" vtable resolution of any trait
            // bounds on the call.
            if check_blocks {
                vtable::early_resolve_expr(callee_expr, fcx, true);
            }

            for args.eachi |i, arg| {
                let is_block = match arg.node {
                    ast::expr_fn_block(*) | ast::expr_loop_body(*) |
                    ast::expr_do_body(*) => true,
                    _ => false
                };

                if is_block == check_blocks {
                    debug!("checking the argument");
                    let formal_ty = formal_tys[i];

                    bot |= check_expr_with_unifier(
                        fcx, arg, Some(formal_ty),
                        || demand::assign(fcx, arg.span, formal_ty, arg)
                    );
                }
            }
        }

        {fty: fty, bot: bot}
    }

    // A generic function for checking assignment expressions
    fn check_assignment(fcx: @fn_ctxt, _sp: span, lhs: @ast::expr,
                        rhs: @ast::expr, id: ast::node_id) -> bool {
        let mut bot = check_expr(fcx, lhs, None);
        bot |= check_expr_with(fcx, rhs, fcx.expr_ty(lhs));
        fcx.write_ty(id, ty::mk_nil(fcx.ccx.tcx));
        return bot;
    }

    // A generic function for doing all of the checking for call expressions
    fn check_call(fcx: @fn_ctxt, sp: span, call_expr_id: ast::node_id,
                  f: @ast::expr, args: ~[@ast::expr]) -> bool {

        // Index expressions need to be handled seperately, to inform
        // them that they appear in call position.
        let mut bot = match f.node {
          ast::expr_field(base, field, tys) => {
            check_field(fcx, f, true, base, field, tys)
          }
          _ => check_expr(fcx, f, None)
        };
        let fn_ty = fcx.expr_ty(f);

        // Call the generic checker.
        let fty = {
            let r = check_call_inner(fcx, sp, call_expr_id,
                                     fn_ty, f, args);
            bot |= r.bot;
            r.fty
        };

        // Pull the return type out of the type of the function.
        match structure_of(fcx, sp, fty) {
          ty::ty_fn(ref f) => {
            bot |= (f.meta.ret_style == ast::noreturn);
            fcx.write_ty(call_expr_id, f.sig.output);
            return bot;
          }
          _ => fcx.ccx.tcx.sess.span_fatal(sp, ~"calling non-function")
        }
    }

    // A generic function for checking for or for-each loops
    fn check_for(fcx: @fn_ctxt, local: @ast::local,
                 element_ty: ty::t, body: ast::blk,
                 node_id: ast::node_id) -> bool {
        let locid = lookup_local(fcx, local.span, local.node.id);
        demand::suptype(fcx, local.span,
                       ty::mk_var(fcx.ccx.tcx, locid),
                       element_ty);
        let bot = check_decl_local(fcx, local);
        check_block_no_value(fcx, body);
        fcx.write_nil(node_id);
        return bot;
    }

    // A generic function for checking the then and else in an if
    // or if-check
    fn check_then_else(fcx: @fn_ctxt, thn: ast::blk,
                       elsopt: Option<@ast::expr>, id: ast::node_id,
                       _sp: span) -> bool {
        let (if_t, if_bot) =
            match elsopt {
                Some(els) => {
                    let if_t = fcx.infcx().next_ty_var();
                    let thn_bot = check_block(fcx, thn);
                    let thn_t = fcx.node_ty(thn.node.id);
                    demand::suptype(fcx, thn.span, if_t, thn_t);
                    let els_bot = check_expr_with(fcx, els, if_t);
                    (if_t, thn_bot & els_bot)
                }
                None => {
                    check_block_no_value(fcx, thn);
                    (ty::mk_nil(fcx.ccx.tcx), false)
                }
            };
        fcx.write_ty(id, if_t);
        return if_bot;
    }

    fn lookup_op_method(fcx: @fn_ctxt, op_ex: @ast::expr,
                        self_ex: @ast::expr, self_t: ty::t,
                        opname: ast::ident, args: ~[@ast::expr])
        -> Option<(ty::t, bool)>
    {
        match method::lookup(fcx, op_ex, self_ex,
                             op_ex.callee_id, opname, self_t, ~[]) {
          Some(origin) => {
            let {fty: method_ty, bot: bot} = {
                let method_ty = fcx.node_ty(op_ex.callee_id);
                check_call_inner(fcx, op_ex.span, op_ex.id,
                                 method_ty, op_ex, args)
            };
            fcx.ccx.method_map.insert(op_ex.id, origin);
            Some((ty::ty_fn_ret(method_ty), bot))
          }
          _ => None
        }
    }

    // could be either a expr_binop or an expr_assign_binop
    fn check_binop(fcx: @fn_ctxt, expr: @ast::expr,
                   op: ast::binop,
                   lhs: @ast::expr,
                   rhs: @ast::expr) -> bool {
        let tcx = fcx.ccx.tcx;

        let lhs_bot = check_expr(fcx, lhs, None);
        let lhs_t = fcx.expr_ty(lhs);
        let lhs_t = structurally_resolved_type(fcx, lhs.span, lhs_t);

        if ty::type_is_integral(lhs_t) && ast_util::is_shift_binop(op) {
            // Shift is a special case: rhs can be any integral type
            let rhs_bot = check_expr(fcx, rhs, None);
            let rhs_t = fcx.expr_ty(rhs);
            require_integral(fcx, rhs.span, rhs_t);
            fcx.write_ty(expr.id, lhs_t);
            return lhs_bot | rhs_bot;
        }

        if ty::is_binopable(tcx, lhs_t, op) {
            let tvar = fcx.infcx().next_ty_var();
            demand::suptype(fcx, expr.span, tvar, lhs_t);
            let rhs_bot = check_expr_with(fcx, rhs, tvar);

            let result_t = match op {
                ast::eq | ast::ne | ast::lt | ast::le | ast::ge | ast::gt => {
                    ty::mk_bool(tcx)
                }
                _ => {
                    lhs_t
                }
            };

            fcx.write_ty(expr.id, result_t);
            return {
                if !ast_util::lazy_binop(op) { lhs_bot | rhs_bot }
                else { lhs_bot }
            };
        }

        let (result, rhs_bot) =
            check_user_binop(fcx, expr, lhs, lhs_t, op, rhs);
        fcx.write_ty(expr.id, result);
        return lhs_bot | rhs_bot;
    }

    fn check_user_binop(fcx: @fn_ctxt, ex: @ast::expr,
                        lhs_expr: @ast::expr, lhs_resolved_t: ty::t,
                        op: ast::binop, rhs: @ast::expr) -> (ty::t, bool)
    {
        let tcx = fcx.ccx.tcx;
        match ast_util::binop_to_method_name(op) {
          Some(name) => {
            match lookup_op_method(fcx, ex, lhs_expr, lhs_resolved_t,
                                   fcx.tcx().sess.ident_of(name),
                                   ~[rhs]) {
              Some(pair) => return pair,
              _ => ()
            }
          }
          _ => ()
        }
        check_expr(fcx, rhs, None);

        tcx.sess.span_err(
            ex.span, ~"binary operation " + ast_util::binop_to_str(op) +
            ~" cannot be applied to type `" +
            fcx.infcx().ty_to_str(lhs_resolved_t) +
            ~"`");

        // If the or operator is used it might be that the user forgot to
        // supply the do keyword.  Let's be more helpful in that situation.
        if op == ast::or {
          match ty::get(lhs_resolved_t).sty {
            ty::ty_fn(_) => {
              tcx.sess.span_note(
                  ex.span, ~"did you forget the 'do' keyword for the call?");
            }
            _ => ()
          }
        }

        (lhs_resolved_t, false)
    }

    fn check_user_unop(fcx: @fn_ctxt, op_str: ~str, mname: ~str,
                       ex: @ast::expr,
                       rhs_expr: @ast::expr, rhs_t: ty::t) -> ty::t {
        match lookup_op_method(fcx, ex, rhs_expr, rhs_t,
                               fcx.tcx().sess.ident_of(mname), ~[]) {
          Some((ret_ty, _)) => ret_ty,
          _ => {
            fcx.ccx.tcx.sess.span_err(
                ex.span, fmt!("cannot apply unary operator `%s` to type `%s`",
                              op_str, fcx.infcx().ty_to_str(rhs_t)));
            rhs_t
          }
        }
    }

    // Resolves `expected` by a single level if it is a variable and passes it
    // through the `unpack` function.  It there is no expected type or
    // resolution is not possible (e.g., no constraints yet present), just
    // returns `none`.
    fn unpack_expected<O: Copy>(fcx: @fn_ctxt, expected: Option<ty::t>,
                                unpack: fn(ty::sty) -> Option<O>)
        -> Option<O> {
        match expected {
            Some(t) => {
                match resolve_type(fcx.infcx(), t, force_tvar) {
                    Ok(t) => unpack(ty::get(t).sty),
                    _ => None
                }
            }
            _ => None
        }
    }

    fn check_expr_fn(fcx: @fn_ctxt,
                     expr: @ast::expr,
                     ast_proto_opt: Option<ast::proto>,
                     decl: ast::fn_decl,
                     body: ast::blk,
                     is_loop_body: bool,
                     expected: Option<ty::t>) {
        let tcx = fcx.ccx.tcx;

        // Find the expected input/output types (if any).  Careful to
        // avoid capture of bound regions in the expected type.  See
        // def'n of br_cap_avoid() for a more lengthy explanation of
        // what's going on here.
        // Also try to pick up inferred purity and proto, defaulting
        // to impure and block. Note that we only will use those for
        // block syntax lambdas; that is, lambdas without explicit
        // protos.
        let expected_sty = unpack_expected(fcx, expected, |x| Some(x));
        let (expected_tys, expected_purity, expected_proto) =
            match expected_sty {
              Some(ty::ty_fn(ref fn_ty)) => {
                let {fn_ty, _} =
                    replace_bound_regions_in_fn_ty(
                        tcx, @Nil, None, fn_ty,
                        |br| ty::re_bound(ty::br_cap_avoid(expr.id, @br)));
                (Some({inputs: fn_ty.sig.inputs,
                       output: fn_ty.sig.output}),
                 fn_ty.meta.purity,
                 fn_ty.meta.proto)
              }
              _ => {
                (None, ast::impure_fn, ty::proto_vstore(ty::vstore_box))
              }
            };


        // Generate AST prototypes and purity.
        // If this is a block lambda (ast_proto == none), these values
        // are bogus. We'll fill in the type with the real one later.
        // XXX: This is a hack.
        let ast_proto = ast_proto_opt.get_default(ast::proto_box);
        let ast_purity = ast::impure_fn;

        // construct the function type
        let mut fn_ty = astconv::ty_of_fn_decl(fcx, fcx,
                                               ast_proto, ast_purity, @~[],
                                               decl, expected_tys, expr.span);

        // Patch up the function declaration, if necessary.
        match ast_proto_opt {
          None => {
            fn_ty.meta.purity = expected_purity;
            fn_ty.meta.proto = expected_proto;
          }
          Some(_) => { }
        }

        let fty = ty::mk_fn(tcx, fn_ty);

        debug!("check_expr_fn_with_unifier %s fty=%s",
               expr_to_str(expr, tcx.sess.intr()),
               fcx.infcx().ty_to_str(fty));

        fcx.write_ty(expr.id, fty);

        check_fn(fcx.ccx, None, &fn_ty, decl, body,
                 is_loop_body, Some(fcx));
    }


    // Check field access expressions
    fn check_field(fcx: @fn_ctxt, expr: @ast::expr, is_callee: bool,
                   base: @ast::expr, field: ast::ident, tys: ~[@ast::ty])
        -> bool
    {
        let tcx = fcx.ccx.tcx;
        let bot = check_expr(fcx, base, None);
        let expr_t = structurally_resolved_type(fcx, expr.span,
                                                fcx.expr_ty(base));
        let (base_t, derefs) = do_autoderef(fcx, expr.span, expr_t);
        let n_tys = vec::len(tys);
        match structure_of(fcx, expr.span, base_t) {
            ty::ty_rec(fields) => {
                match ty::field_idx(field, fields) {
                    Some(ix) => {
                        if n_tys > 0u {
                            tcx.sess.span_err(
                                expr.span,
                                ~"can't provide type parameters \
                                  to a field access");
                        }
                        fcx.write_ty(expr.id, fields[ix].mt.ty);
                        fcx.write_autoderef_adjustment(base.id, derefs);
                        return bot;
                    }
                    _ => ()
                }
            }
            ty::ty_class(base_id, substs) => {
                // This is just for fields -- the same code handles
                // methods in both classes and traits

                // (1) verify that the class id actually has a field called
                // field
                debug!("class named %s", ty_to_str(tcx, base_t));
                let cls_items = ty::lookup_class_fields(tcx, base_id);
                match lookup_field_ty(tcx, base_id, cls_items,
                                      field, &substs) {
                    Some(field_ty) => {
                        // (2) look up what field's type is, and return it
                        fcx.write_ty(expr.id, field_ty);
                        fcx.write_autoderef_adjustment(base.id, derefs);
                        return bot;
                    }
                    None => ()
                }
            }
            _ => ()
        }

        let tps = vec::map(tys, |ty| fcx.to_ty(ty));

        match method::lookup(fcx, expr, base, expr.id,
                             field, expr_t, tps) {
            Some(entry) => {
                fcx.ccx.method_map.insert(expr.id, entry);

                // If we have resolved to a method but this is not in
                // a callee position, error
                if !is_callee {
                    tcx.sess.span_err(
                        expr.span,
                        ~"attempted to take value of method \
                          (try writing an anonymous function)");
                }
            }
            None => {
                let t_err =
                    fcx.infcx().resolve_type_vars_if_possible(expr_t);
                let msg =
                    fmt!(
                        "attempted access of field `%s` on type `%s`, \
                         but no field or method with that name was found",
                        tcx.sess.str_of(field),
                        fcx.infcx().ty_to_str(t_err));
                tcx.sess.span_err(expr.span, msg);
                // NB: Add bogus type to allow typechecking to continue
                fcx.write_ty(expr.id, fcx.infcx().next_ty_var());
            }
        }

        return bot;
    }


    let tcx = fcx.ccx.tcx;
    let id = expr.id;
    let mut bot = false;
    match expr.node {
      ast::expr_vstore(ev, vst) => {
        let typ = match ev.node {
          ast::expr_lit(@{node: ast::lit_str(s), span:_}) => {
            let tt = ast_expr_vstore_to_vstore(fcx, ev, str::len(*s), vst);
            ty::mk_estr(tcx, tt)
          }
          ast::expr_vec(args, mutbl) => {
            let tt = ast_expr_vstore_to_vstore(fcx, ev, vec::len(args), vst);
            let t: ty::t = fcx.infcx().next_ty_var();
            for args.each |e| { bot |= check_expr_with(fcx, e, t); }
            ty::mk_evec(tcx, {ty: t, mutbl: mutbl}, tt)
          }
          ast::expr_repeat(element, count_expr, mutbl) => {
            let count = ty::eval_repeat_count(tcx, count_expr, expr.span);
            fcx.write_ty(count_expr.id, ty::mk_uint(tcx));
            let tt = ast_expr_vstore_to_vstore(fcx, ev, count, vst);
            let t: ty::t = fcx.infcx().next_ty_var();
            bot |= check_expr_with(fcx, element, t);
            ty::mk_evec(tcx, {ty: t, mutbl: mutbl}, tt)
          }
          _ =>
            tcx.sess.span_bug(expr.span, ~"vstore modifier on non-sequence")
        };
        fcx.write_ty(ev.id, typ);
        fcx.write_ty(id, typ);
      }

      ast::expr_lit(lit) => {
        let typ = check_lit(fcx, lit);
        fcx.write_ty(id, typ);
      }
      ast::expr_binary(op, lhs, rhs) => {
        bot |= check_binop(fcx, expr, op, lhs, rhs);
      }
      ast::expr_assign_op(op, lhs, rhs) => {
        bot |= check_binop(fcx, expr, op, lhs, rhs);
        let lhs_t = fcx.expr_ty(lhs);
        let result_t = fcx.expr_ty(expr);
        demand::suptype(fcx, expr.span, result_t, lhs_t);

        // Overwrite result of check_binop...this preserves existing behavior
        // but seems quite dubious with regard to user-defined methods
        // and so forth. - Niko
        fcx.write_nil(expr.id);
      }
      ast::expr_unary(unop, oprnd) => {
        let exp_inner = do unpack_expected(fcx, expected) |sty| {
            match unop {
              ast::box(_) | ast::uniq(_) => match sty {
                ty::ty_box(mt) | ty::ty_uniq(mt) => Some(mt.ty),
                _ => None
              },
              ast::not | ast::neg => expected,
              ast::deref => None
            }
        };
        bot = check_expr(fcx, oprnd, exp_inner);
        let mut oprnd_t = fcx.expr_ty(oprnd);
        match unop {
          ast::box(mutbl) => {
            oprnd_t = ty::mk_box(tcx, {ty: oprnd_t, mutbl: mutbl});
          }
          ast::uniq(mutbl) => {
            oprnd_t = ty::mk_uniq(tcx, {ty: oprnd_t, mutbl: mutbl});
          }
          ast::deref => {
            let sty = structure_of(fcx, expr.span, oprnd_t);

            // deref'ing an unsafe pointer requires that we be in an unsafe
            // context
            match sty {
              ty::ty_ptr(*) => {
                fcx.require_unsafe(
                    expr.span,
                    ~"dereference of unsafe pointer");
              }
              _ => { /*ok*/ }
            }

            match ty::deref_sty(tcx, &sty, true) {
              Some(mt) => { oprnd_t = mt.ty }
              None => {
                match sty {
                  ty::ty_enum(*) => {
                    tcx.sess.span_err(
                        expr.span,
                        ~"can only dereference enums \
                         with a single variant which has a \
                         single argument");
                  }
                  _ => {
                    tcx.sess.span_err(
                        expr.span,
                        fmt!("type %s cannot be dereferenced",
                             fcx.infcx().ty_to_str(oprnd_t)));
                  }
                }
              }
            }
          }
          ast::not => {
            oprnd_t = structurally_resolved_type(fcx, oprnd.span, oprnd_t);
            if !(ty::type_is_integral(oprnd_t) ||
                 ty::get(oprnd_t).sty == ty::ty_bool) {
                oprnd_t = check_user_unop(fcx, ~"!", ~"not", expr,
                                         oprnd, oprnd_t);
            }
          }
          ast::neg => {
            oprnd_t = structurally_resolved_type(fcx, oprnd.span, oprnd_t);
            if !(ty::type_is_integral(oprnd_t) ||
                 ty::type_is_fp(oprnd_t)) {
                oprnd_t = check_user_unop(fcx, ~"-", ~"neg", expr,
                                         oprnd, oprnd_t);
            }
          }
        }
        fcx.write_ty(id, oprnd_t);
      }
      ast::expr_addr_of(mutbl, oprnd) => {
        bot = check_expr(fcx, oprnd, unpack_expected(fcx, expected, |ty|
            match ty { ty::ty_rptr(_, mt) => Some(mt.ty), _ => None }
        ));

        // Note: at this point, we cannot say what the best lifetime
        // is to use for resulting pointer.  We want to use the
        // shortest lifetime possible so as to avoid spurious borrowck
        // errors.  Moreover, the longest lifetime will depend on the
        // precise details of the value whose address is being taken
        // (and how long it is valid), which we don't know yet until type
        // inference is complete.
        //
        // Therefore, here we simply generate a region variable with
        // the current expression as a lower bound.  The region
        // inferencer will then select the ultimate value.  Finally,
        // borrowck is charged with guaranteeing that the value whose
        // address was taken can actually be made to live as long as
        // it needs to live.
        let region = fcx.infcx().next_region_var(expr.span, expr.id);

        let tm = { ty: fcx.expr_ty(oprnd), mutbl: mutbl };
        let oprnd_t = ty::mk_rptr(tcx, region, tm);
        fcx.write_ty(id, oprnd_t);
      }
      ast::expr_path(pth) => {
        let defn = lookup_def(fcx, pth.span, id);

        let tpt = ty_param_bounds_and_ty_for_def(fcx, expr.span, defn);
        let region_lb = ty::re_scope(expr.id);
        instantiate_path(fcx, pth, tpt, expr.span, expr.id, region_lb);
      }
      ast::expr_mac(_) => tcx.sess.bug(~"unexpanded macro"),
      ast::expr_fail(expr_opt) => {
        bot = true;
        match expr_opt {
          None => {/* do nothing */ }
          Some(e) => {
            check_expr_with(fcx, e,
                            ty::mk_estr(tcx, ty::vstore_uniq));
          }
        }
        fcx.write_bot(id);
      }
      ast::expr_break(_) => { fcx.write_bot(id); bot = true; }
      ast::expr_again(_) => { fcx.write_bot(id); bot = true; }
      ast::expr_ret(expr_opt) => {
        bot = true;
        let ret_ty = match fcx.indirect_ret_ty {
          Some(t) =>  t, None => fcx.ret_ty
        };
        match expr_opt {
          None => match fcx.mk_eqty(false, expr.span,
                                    ret_ty, ty::mk_nil(tcx)) {
            result::Ok(_) => { /* fall through */ }
            result::Err(_) => {
                tcx.sess.span_err(
                    expr.span,
                    ~"`return;` in function returning non-nil");
            }
          },
          Some(e) => { check_expr_with(fcx, e, ret_ty); }
        }
        fcx.write_bot(id);
      }
      ast::expr_log(_, lv, e) => {
        bot = check_expr_with(fcx, lv, ty::mk_mach_uint(tcx, ast::ty_u32));
        // Note: this does not always execute, so do not propagate bot:
        check_expr(fcx, e, None);
        fcx.write_nil(id);
      }
      ast::expr_assert(e) => {
        bot = check_expr_with(fcx, e, ty::mk_bool(tcx));
        fcx.write_nil(id);
      }
      ast::expr_copy(a) | ast::expr_unary_move(a) => {
        bot = check_expr(fcx, a, expected);
        fcx.write_ty(id, fcx.expr_ty(a));
      }
      ast::expr_move(lhs, rhs) => {
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_assign(lhs, rhs) => {
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_swap(lhs, rhs) => {
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_if(cond, thn, elsopt) => {
        bot = check_expr_with(fcx, cond, ty::mk_bool(tcx)) |
            check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_while(cond, body) => {
        bot = check_expr_with(fcx, cond, ty::mk_bool(tcx));
        check_block_no_value(fcx, body);
        fcx.write_ty(id, ty::mk_nil(tcx));
      }
      ast::expr_loop(body, _) => {
        check_block_no_value(fcx, body);
        fcx.write_ty(id, ty::mk_nil(tcx));
        bot = !may_break(body);
      }
      ast::expr_match(discrim, arms) => {
        bot = alt::check_alt(fcx, expr, discrim, arms);
      }
      ast::expr_fn(proto, decl, body, cap_clause) => {
        check_expr_fn(fcx, expr, Some(proto),
                      decl, body, false,
                      expected);
        capture::check_capture_clause(tcx, expr.id, cap_clause);
      }
      ast::expr_fn_block(decl, body, cap_clause) => {
        check_expr_fn(fcx, expr, None,
                      decl, body, false,
                      expected);
        capture::check_capture_clause(tcx, expr.id, cap_clause);
      }
      ast::expr_loop_body(b) => {
        // a loop body is the special argument to a `for` loop.  We know that
        // there will be an expected type in this context because it can only
        // appear in the context of a call, so we get the expected type of the
        // parameter. The catch here is that we need to validate two things:
        // 1. a closure that returns a bool is expected
        // 2. the cloure that was given returns unit
        let expected_sty = unpack_expected(fcx, expected, |x| Some(x));
        let inner_ty = match expected_sty {
          Some(ty::ty_fn(fty)) => {
            match fcx.mk_subty(false, expr.span,
                               fty.sig.output, ty::mk_bool(tcx)) {
              result::Ok(_) => (),
              result::Err(_) => {
                tcx.sess.span_fatal(
                    expr.span, fmt!("a `loop` function's last argument \
                                     should return `bool`, not `%s`",
                                    fcx.infcx().ty_to_str(fty.sig.output)));
              }
            }
            ty::mk_fn(tcx, FnTyBase {
                meta: fty.meta,
                sig: FnSig {output: ty::mk_nil(tcx),
                            ..fty.sig}
            })
          }
          _ => {
            tcx.sess.span_fatal(expr.span, ~"a `loop` function's last \
                                            argument should be of function \
                                            type");
          }
        };
        match b.node {
          ast::expr_fn_block(decl, body, cap_clause) => {
            check_expr_fn(fcx, b, None,
                          decl, body, true,
                          Some(inner_ty));
            demand::suptype(fcx, b.span, inner_ty, fcx.expr_ty(b));
            capture::check_capture_clause(tcx, b.id, cap_clause);
          }
          // argh
          _ => fail ~"expr_fn_block"
        }
        let block_ty = structurally_resolved_type(
            fcx, expr.span, fcx.node_ty(b.id));
        match ty::get(block_ty).sty {
          ty::ty_fn(fty) => {
            fcx.write_ty(expr.id, ty::mk_fn(tcx, FnTyBase {
                meta: fty.meta,
                sig: FnSig {output: ty::mk_bool(tcx),
                            ..fty.sig}
            }))
          }
          _ => fail ~"expected fn type"
        }
      }
      ast::expr_do_body(b) => {
        let expected_sty = unpack_expected(fcx, expected, |x| Some(x));
        let inner_ty = match expected_sty {
          Some(ty::ty_fn(fty)) => {
            ty::mk_fn(tcx, fty)
          }
          _ => {
            tcx.sess.span_fatal(expr.span, ~"Non-function passed to a `do` \
              function as its last argument, or wrong number of arguments \
              passed to a `do` function");
          }
        };
        match b.node {
          ast::expr_fn_block(decl, body, cap_clause) => {
            check_expr_fn(fcx, b, None,
                          decl, body, true,
                          Some(inner_ty));
            demand::suptype(fcx, b.span, inner_ty, fcx.expr_ty(b));
            capture::check_capture_clause(tcx, b.id, cap_clause);
          }
          // argh
          _ => fail ~"expected fn ty"
        }
        let block_ty = structurally_resolved_type(
            fcx, expr.span, fcx.node_ty(b.id));
        match ty::get(block_ty).sty {
          ty::ty_fn(fty) => {
            fcx.write_ty(expr.id, ty::mk_fn(tcx, fty));
          }
          _ => fail ~"expected fn ty"
        }
      }
      ast::expr_block(b) => {
        // If this is an unchecked block, turn off purity-checking
        bot = check_block(fcx, b);
        let typ =
            match b.node.expr {
              Some(expr) => fcx.expr_ty(expr),
              None => ty::mk_nil(tcx)
            };
        fcx.write_ty(id, typ);
      }
      ast::expr_call(f, args, _) => {
        bot = check_call(fcx, expr.span, expr.id, f, args);
      }
      ast::expr_cast(e, t) => {
        bot = check_expr(fcx, e, None);
        let t_1 = fcx.to_ty(t);
        let t_e = fcx.expr_ty(e);

        debug!("t_1=%s", fcx.infcx().ty_to_str(t_1));
        debug!("t_e=%s", fcx.infcx().ty_to_str(t_e));

        match ty::get(t_1).sty {
          // This will be looked up later on
          ty::ty_trait(*) => (),

          _ => {
            if ty::type_is_nil(t_e) {
                tcx.sess.span_err(expr.span, ~"cast from nil: " +
                                  fcx.infcx().ty_to_str(t_e) + ~" as " +
                                  fcx.infcx().ty_to_str(t_1));
            } else if ty::type_is_nil(t_1) {
                tcx.sess.span_err(expr.span, ~"cast to nil: " +
                                  fcx.infcx().ty_to_str(t_e) + ~" as " +
                                  fcx.infcx().ty_to_str(t_1));
            }

            let t_1_is_scalar = type_is_scalar(fcx, expr.span, t_1);
            if type_is_c_like_enum(fcx,expr.span,t_e) && t_1_is_scalar {
                /* this case is allowed */
            } else if !(type_is_scalar(fcx,expr.span,t_e) && t_1_is_scalar) {
                /*
                If more type combinations should be supported than are
                supported here, then file an enhancement issue and record the
                issue number in this comment.
                */
                tcx.sess.span_err(expr.span,
                                  ~"non-scalar cast: " +
                                  fcx.infcx().ty_to_str(t_e) + ~" as " +
                                  fcx.infcx().ty_to_str(t_1));
            }
          }
        }
        fcx.write_ty(id, t_1);
      }
      ast::expr_vec(args, mutbl) => {
        let t: ty::t = fcx.infcx().next_ty_var();
        for args.each |e| { bot |= check_expr_with(fcx, e, t); }
        let typ = ty::mk_evec(tcx, {ty: t, mutbl: mutbl},
                              ty::vstore_fixed(args.len()));
        fcx.write_ty(id, typ);
      }
      ast::expr_repeat(element, count_expr, mutbl) => {
        let count = ty::eval_repeat_count(tcx, count_expr, expr.span);
        fcx.write_ty(count_expr.id, ty::mk_uint(tcx));
        let t: ty::t = fcx.infcx().next_ty_var();
        bot |= check_expr_with(fcx, element, t);
        let t = ty::mk_evec(tcx, {ty: t, mutbl: mutbl},
                            ty::vstore_fixed(count));
        fcx.write_ty(id, t);
      }
      ast::expr_tup(elts) => {
        let mut elt_ts = ~[];
        vec::reserve(elt_ts, vec::len(elts));
        let flds = unpack_expected(fcx, expected, |sty| {
            match sty { ty::ty_tup(flds) => Some(flds), _ => None }
        });
        for elts.eachi |i, e| {
            check_expr(fcx, e, flds.map(|fs| fs[i]));
            let ety = fcx.expr_ty(e);
            vec::push(elt_ts, ety);
        }
        let typ = ty::mk_tup(tcx, elt_ts);
        fcx.write_ty(id, typ);
      }
      ast::expr_rec(fields, base) => {
        option::iter(base, |b| { check_expr(fcx, b, expected); });
        let expected = if expected.is_none() && base.is_some() {
            Some(fcx.expr_ty(base.get()))
        } else { expected };
        let flds = unpack_expected(fcx, expected, |sty|
            match sty { ty::ty_rec(flds) => Some(flds), _ => None }
        );
        let fields_t = vec::map(fields, |f| {
            bot |= check_expr(fcx, f.node.expr, flds.chain(|flds|
                vec::find(flds, |tf| tf.ident == f.node.ident)
            ).map(|tf| tf.mt.ty));
            let expr_t = fcx.expr_ty(f.node.expr);
            let expr_mt = {ty: expr_t, mutbl: f.node.mutbl};
            // for the most precise error message,
            // should be f.node.expr.span, not f.span
            respan(f.node.expr.span, {ident: f.node.ident, mt: expr_mt})
        });
        match base {
          None => {
            fn get_node(f: spanned<field>) -> field { f.node }
            let typ = ty::mk_rec(tcx, vec::map(fields_t, get_node));
            fcx.write_ty(id, typ);
            /* Check for duplicate fields */
            /* Only do this check if there's no base expr -- the reason is
               that we're extending a record we know has no dup fields, and
               it would be ill-typed anyway if we duplicated one of its
               fields */
            check_no_duplicate_fields(tcx, fields.map(|f|
                                                    (f.node.ident, f.span)));
          }
          Some(bexpr) => {
            let bexpr_t = fcx.expr_ty(bexpr);
            let base_fields =  match structure_of(fcx, expr.span, bexpr_t) {
              ty::ty_rec(flds) => flds,
              _ => {
                tcx.sess.span_fatal(expr.span,
                                    ~"record update has non-record base");
              }
            };
            fcx.write_ty(id, bexpr_t);
            for fields_t.each |f| {
                let mut found = false;
                for base_fields.each |bf| {
                    if f.node.ident == bf.ident {
                        demand::suptype(fcx, f.span, bf.mt.ty, f.node.mt.ty);
                        found = true;
                    }
                }
                if !found {
                    tcx.sess.span_fatal(f.span,
                                        ~"unknown field in record update: " +
                                        tcx.sess.str_of(f.node.ident));
                }
            }
          }
        }
      }
      ast::expr_struct(path, fields, base_expr) => {
        // Resolve the path.
        let class_id;
        match tcx.def_map.find(id) {
            Some(ast::def_class(type_def_id, _)) => {
                class_id = type_def_id;
            }
            _ => {
                tcx.sess.span_bug(path.span, ~"structure constructor does \
                                               not name a structure type");
            }
        }

        // Look up the number of type parameters and the raw type, and
        // determine whether the class is region-parameterized.
        let type_parameter_count, region_parameterized, raw_type;
        if class_id.crate == ast::local_crate {
            region_parameterized =
                tcx.region_paramd_items.find(class_id.node);
            match tcx.items.find(class_id.node) {
                Some(ast_map::node_item(@{
                        node: ast::item_class(_, type_parameters),
                        _
                    }, _)) => {

                    type_parameter_count = type_parameters.len();

                    let self_region =
                        bound_self_region(region_parameterized);

                    raw_type = ty::mk_class(tcx, class_id, {
                        self_r: self_region,
                        self_ty: None,
                        tps: ty::ty_params_to_tys(tcx, type_parameters)
                    });
                }
                _ => {
                    tcx.sess.span_bug(expr.span,
                                      ~"resolve didn't map this to a class");
                }
            }
        } else {
            let item_type = ty::lookup_item_type(tcx, class_id);
            type_parameter_count = (*item_type.bounds).len();
            region_parameterized = item_type.region_param;
            raw_type = item_type.ty;
        }

        // Generate the struct type.
        let self_region =
            fcx.region_var_if_parameterized(region_parameterized,
                                            expr.span,
                                            ty::re_scope(expr.id));
        let type_parameters = fcx.infcx().next_ty_vars(type_parameter_count);
        let substitutions = {
            self_r: self_region,
            self_ty: None,
            tps: type_parameters
        };

        let struct_type = ty::subst(tcx, &substitutions, raw_type);

        // Look up the class fields and build up a map.
        let class_fields = ty::lookup_class_fields(tcx, class_id);
        let class_field_map = uint_hash();
        let mut fields_found = 0;
        for class_fields.each |field| {
            // XXX: Check visibility here.
            class_field_map.insert(field.ident, (field.id, false));
        }

        // Typecheck each field.
        for fields.each |field| {
            match class_field_map.find(field.node.ident) {
                None => {
                    tcx.sess.span_err(
                        field.span,
                        fmt!("structure has no field named field named `%s`",
                             tcx.sess.str_of(field.node.ident)));
                }
                Some((_, true)) => {
                    tcx.sess.span_err(
                        field.span,
                        fmt!("field `%s` specified more than once",
                             tcx.sess.str_of(field.node.ident)));
                }
                Some((field_id, false)) => {
                    let expected_field_type =
                        ty::lookup_field_type(tcx, class_id, field_id,
                                              &substitutions);
                    bot |= check_expr(fcx,
                                      field.node.expr,
                                      Some(expected_field_type));
                    fields_found += 1;
                }
            }
        }

        match base_expr {
            None => {
                // Make sure the programmer specified all the fields.
                assert fields_found <= class_fields.len();
                if fields_found < class_fields.len() {
                    let mut missing_fields = ~[];
                    for class_fields.each |class_field| {
                        let name = class_field.ident;
                        let (_, seen) = class_field_map.get(name);
                        if !seen {
                            vec::push(missing_fields,
                                      ~"`" + tcx.sess.str_of(name) + ~"`");
                        }
                    }

                    tcx.sess.span_err(expr.span,
                                      fmt!("missing field%s: %s",
                                           if missing_fields.len() == 1 {
                                               ~""
                                           } else {
                                               ~"s"
                                           },
                                           str::connect(missing_fields,
                                                        ~", ")));
                }
            }
            Some(base_expr) => {
                // Just check the base expression.
                check_expr(fcx, base_expr, Some(struct_type));
            }
        }

        // Write in the resulting type.
        fcx.write_ty(id, struct_type);
      }
      ast::expr_field(base, field, tys) => {
        bot = check_field(fcx, expr, false, base, field, tys);
      }
      ast::expr_index(base, idx) => {
          bot |= check_expr(fcx, base, None);
          let raw_base_t = fcx.expr_ty(base);
          let (base_t, derefs) = do_autoderef(fcx, expr.span, raw_base_t);
          bot |= check_expr(fcx, idx, None);
          let idx_t = fcx.expr_ty(idx);
          let base_sty = structure_of(fcx, expr.span, base_t);
          match ty::index_sty(tcx, &base_sty) {
              Some(mt) => {
                  require_integral(fcx, idx.span, idx_t);
                  fcx.write_ty(id, mt.ty);
                  fcx.write_autoderef_adjustment(base.id, derefs);
              }
              None => {
                  let resolved = structurally_resolved_type(fcx, expr.span,
                                                            raw_base_t);
                  match lookup_op_method(fcx, expr, base, resolved,
                                         tcx.sess.ident_of(~"index"),
                                         ~[idx]) {
                      Some((ret_ty, _)) => fcx.write_ty(id, ret_ty),
                      _ => {
                          tcx.sess.span_fatal(
                              expr.span, ~"cannot index a value of type `" +
                              fcx.infcx().ty_to_str(base_t) + ~"`");
                      }
                  }
              }
          }
      }
    }
    if bot { fcx.write_bot(expr.id); }

    debug!("type of expr %s is %s, expected is %s",
           syntax::print::pprust::expr_to_str(expr, tcx.sess.intr()),
           ty_to_str(tcx, fcx.expr_ty(expr)),
           match expected {
               Some(t) => ty_to_str(tcx, t),
               _ => ~"empty"
           });

    unifier();

    debug!("<< bot=%b", bot);
    return bot;
}

fn require_integral(fcx: @fn_ctxt, sp: span, t: ty::t) {
    if !type_is_integral(fcx, sp, t) {
        fcx.ccx.tcx.sess.span_err(sp, ~"mismatched types: expected \
                                       integral type but found `"
                                  + fcx.infcx().ty_to_str(t) + ~"`");
    }
}

fn check_decl_initializer(fcx: @fn_ctxt, nid: ast::node_id,
                          init: ast::initializer) -> bool {
    let lty = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, init.expr.span, nid));
    return check_expr_with(fcx, init.expr, lty);
}

fn check_decl_local(fcx: @fn_ctxt, local: @ast::local) -> bool {
    let mut bot = false;
    let tcx = fcx.ccx.tcx;

    let t = ty::mk_var(tcx, fcx.inh.locals.get(local.node.id));
    fcx.write_ty(local.node.id, t);

    let is_lvalue;
    match local.node.init {
        Some(init) => {
            bot = check_decl_initializer(fcx, local.node.id, init);
            is_lvalue = ty::expr_is_lval(tcx, fcx.ccx.method_map, init.expr);
        }
        _ => {
            is_lvalue = true;
        }
    }

    let region =
        ty::re_scope(tcx.region_map.get(local.node.id));
    let pcx = {
        fcx: fcx,
        map: pat_id_map(tcx.def_map, local.node.pat),
        alt_region: region,
        block_region: region,
    };
    alt::check_pat(pcx, local.node.pat, t);
    let has_guard = false;
    alt::check_legality_of_move_bindings(fcx,
                                         is_lvalue,
                                         has_guard,
                                         [local.node.pat]);
    return bot;
}

fn check_stmt(fcx: @fn_ctxt, stmt: @ast::stmt) -> bool {
    let mut node_id;
    let mut bot = false;
    match stmt.node {
      ast::stmt_decl(decl, id) => {
        node_id = id;
        match decl.node {
          ast::decl_local(ls) => for ls.each |l| {
            bot |= check_decl_local(fcx, l);
          },
          ast::decl_item(_) => {/* ignore for now */ }
        }
      }
      ast::stmt_expr(expr, id) => {
        node_id = id;
        bot = check_expr_with(fcx, expr, ty::mk_nil(fcx.ccx.tcx));
      }
      ast::stmt_semi(expr, id) => {
        node_id = id;
        bot = check_expr(fcx, expr, None);
      }
    }
    fcx.write_nil(node_id);
    return bot;
}

fn check_block_no_value(fcx: @fn_ctxt, blk: ast::blk) -> bool {
    let bot = check_block(fcx, blk);
    if !bot {
        let blkty = fcx.node_ty(blk.node.id);
        let nilty = ty::mk_nil(fcx.ccx.tcx);
        demand::suptype(fcx, blk.span, nilty, blkty);
    }
    return bot;
}

fn check_block(fcx0: @fn_ctxt, blk: ast::blk) -> bool {
    let fcx = match blk.node.rules {
      ast::unsafe_blk => @fn_ctxt {purity: ast::unsafe_fn,.. *fcx0},
      ast::default_blk => fcx0
    };
    do fcx.with_region_lb(blk.node.id) {
        let mut bot = false;
        let mut warned = false;
        for blk.node.stmts.each |s| {
            if bot && !warned &&
                match s.node {
                  ast::stmt_decl(@{node: ast::decl_local(_), _}, _) |
                  ast::stmt_expr(_, _) | ast::stmt_semi(_, _) => {
                    true
                  }
                  _ => false
                } {
                fcx.ccx.tcx.sess.span_warn(s.span, ~"unreachable statement");
                warned = true;
            }
            bot |= check_stmt(fcx, s);
        }
        match blk.node.expr {
          None => fcx.write_nil(blk.node.id),
          Some(e) => {
            if bot && !warned {
                fcx.ccx.tcx.sess.span_warn(e.span, ~"unreachable expression");
            }
            bot |= check_expr(fcx, e, None);
            let ety = fcx.expr_ty(e);
            fcx.write_ty(blk.node.id, ety);
          }
        }
        if bot {
            fcx.write_bot(blk.node.id);
        }
        bot
    }
}

fn check_const(ccx: @crate_ctxt, _sp: span, e: @ast::expr, id: ast::node_id) {
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx = blank_fn_ctxt(ccx, rty, e.id);
    check_expr(fcx, e, None);
    let cty = fcx.expr_ty(e);
    let declty = fcx.ccx.tcx.tcache.get(local_def(id)).ty;
    demand::suptype(fcx, e.span, declty, cty);
    regionck::regionck_expr(fcx, e);
    writeback::resolve_type_vars_in_expr(fcx, e);
}

/// Checks whether a type can be created without an instance of itself.
/// This is similar but different from the question of whether a type
/// can be represented.  For example, the following type:
///
///     enum foo { None, Some(foo) }
///
/// is instantiable but is not representable.  Similarly, the type
///
///     enum foo { Some(@foo) }
///
/// is representable, but not instantiable.
fn check_instantiable(tcx: ty::ctxt,
                      sp: span,
                      item_id: ast::node_id) {
    let item_ty = ty::node_id_to_type(tcx, item_id);
    if !ty::is_instantiable(tcx, item_ty) {
        tcx.sess.span_err(sp, fmt!("this type cannot be instantiated \
                                    without an instance of itself; \
                                    consider using `option<%s>`",
                                   ty_to_str(tcx, item_ty)));
    }
}

fn check_enum_variants(ccx: @crate_ctxt,
                       sp: span,
                       vs: ~[ast::variant],
                       id: ast::node_id) {
    fn do_check(ccx: @crate_ctxt, sp: span, vs: ~[ast::variant],
                id: ast::node_id, disr_vals: &mut ~[int], disr_val: &mut int,
                variants: &mut ~[ty::variant_info]) {
        let rty = ty::node_id_to_type(ccx.tcx, id);
        for vs.each |v| {
            match v.node.disr_expr {
              Some(e) => {
                let fcx = blank_fn_ctxt(ccx, rty, e.id);
                check_expr(fcx, e, None);
                let cty = fcx.expr_ty(e);
                let declty = ty::mk_int(ccx.tcx);
                demand::suptype(fcx, e.span, declty, cty);
                // FIXME: issue #1417
                // Also, check_expr (from check_const pass) doesn't guarantee
                // that the expression is in an form that eval_const_expr can
                // handle, so we may still get an internal compiler error
                match const_eval::eval_const_expr(ccx.tcx, e) {
                  const_eval::const_int(val) => {
                    *disr_val = val as int;
                  }
                  _ => {
                    ccx.tcx.sess.span_err(e.span, ~"expected signed integer \
                                                    constant");
                  }
                }
              }
              _ => ()
            }
            if vec::contains(*disr_vals, *disr_val) {
                ccx.tcx.sess.span_err(v.span,
                                      ~"discriminator value already exists");
            }
            vec::push(*disr_vals, *disr_val);
            let ctor_ty = ty::node_id_to_type(ccx.tcx, v.node.id);
            let arg_tys;

            let this_disr_val = *disr_val;
            *disr_val += 1;

            match v.node.kind {
                ast::tuple_variant_kind(args) if args.len() > 0u => {
                    arg_tys = Some(ty::ty_fn_args(ctor_ty).map(|a| a.ty));
                }
                ast::tuple_variant_kind(_) | ast::struct_variant_kind(_) => {
                    arg_tys = Some(~[]);
                }
                ast::enum_variant_kind(_) => {
                    arg_tys = None;
                    do_check(ccx, sp, vs, id, disr_vals, disr_val, variants);
                }
            }

            match arg_tys {
                None => {}
                Some(arg_tys) => {
                    vec::push(*variants, @{args: arg_tys, ctor_ty: ctor_ty,
                          name: v.node.name, id: local_def(v.node.id),
                          disr_val: this_disr_val});
                }
            }
        }
    }

    let rty = ty::node_id_to_type(ccx.tcx, id);
    let mut disr_vals: ~[int] = ~[];
    let mut disr_val = 0;
    let mut variants = ~[];

    do_check(ccx, sp, vs, id, &mut disr_vals, &mut disr_val, &mut variants);

    // cache so that ty::enum_variants won't repeat this work
    ccx.tcx.enum_var_cache.insert(local_def(id), @variants);

    // Check that it is possible to represent this enum:
    let mut outer = true, did = local_def(id);
    if ty::type_structurally_contains(ccx.tcx, rty, |sty| {
        match *sty {
          ty::ty_enum(id, _) if id == did => {
            if outer { outer = false; false }
            else { true }
          }
          _ => false
        }
    }) {
        ccx.tcx.sess.span_err(sp, ~"illegal recursive enum type; \
                                   wrap the inner value in a box to \
                                   make it representable");
    }

    // Check that it is possible to instantiate this enum:
    //
    // This *sounds* like the same that as representable, but it's
    // not.  See def'n of `check_instantiable()` for details.
    check_instantiable(ccx.tcx, sp, id);
}

fn lookup_local(fcx: @fn_ctxt, sp: span, id: ast::node_id) -> TyVid {
    match fcx.inh.locals.find(id) {
        Some(x) => x,
        _ => {
            fcx.ccx.tcx.sess.span_fatal(
                sp,
                ~"internal error looking up a local var")
        }
    }
}

fn lookup_def(fcx: @fn_ctxt, sp: span, id: ast::node_id) -> ast::def {
    lookup_def_ccx(fcx.ccx, sp, id)
}

// Returns the type parameter count and the type for the given definition.
fn ty_param_bounds_and_ty_for_def(fcx: @fn_ctxt, sp: span, defn: ast::def) ->
    ty_param_bounds_and_ty {

    match defn {
      ast::def_arg(nid, _) | ast::def_local(nid, _) |
      ast::def_self(nid) | ast::def_binding(nid, _) => {
        assert (fcx.inh.locals.contains_key(nid));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, nid));
        return no_params(typ);
      }
      ast::def_fn(_, ast::extern_fn) => {
        // extern functions are just u8 pointers
        return {
            bounds: @~[],
            region_param: None,
            ty: ty::mk_ptr(
                fcx.ccx.tcx,
                {
                    ty: ty::mk_mach_uint(fcx.ccx.tcx, ast::ty_u8),
                    mutbl: ast::m_imm
                })
        };
      }

      ast::def_fn(id, ast::unsafe_fn) |
      ast::def_static_method(id, ast::unsafe_fn) => {
        // Unsafe functions can only be touched in an unsafe context
        fcx.require_unsafe(sp, ~"access to unsafe function");
        return ty::lookup_item_type(fcx.ccx.tcx, id);
      }

      ast::def_fn(id, _) | ast::def_static_method(id, _) |
      ast::def_const(id) | ast::def_variant(_, id) |
      ast::def_class(id, _) => {
        return ty::lookup_item_type(fcx.ccx.tcx, id);
      }
      ast::def_upvar(_, inner, _, _) => {
        return ty_param_bounds_and_ty_for_def(fcx, sp, *inner);
      }
      ast::def_ty(_) | ast::def_prim_ty(_) | ast::def_ty_param(*)=> {
        fcx.ccx.tcx.sess.span_bug(sp, ~"expected value but found type");
      }
      ast::def_mod(*) | ast::def_foreign_mod(*) => {
        fcx.ccx.tcx.sess.span_bug(sp, ~"expected value but found module");
      }
      ast::def_use(*) => {
        fcx.ccx.tcx.sess.span_bug(sp, ~"expected value but found use");
      }
      ast::def_region(*) => {
        fcx.ccx.tcx.sess.span_bug(sp, ~"expected value but found region");
      }
      ast::def_typaram_binder(*) => {
        fcx.ccx.tcx.sess.span_bug(sp, ~"expected value but found type \
                                          parameter");
      }
      ast::def_label(*) => {
        fcx.ccx.tcx.sess.span_bug(sp, ~"expected value but found label");
      }
    }
}

// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
fn instantiate_path(fcx: @fn_ctxt,
                    pth: @ast::path,
                    tpt: ty_param_bounds_and_ty,
                    span: span,
                    node_id: ast::node_id,
                    region_lb: ty::region) {
    let ty_param_count = vec::len(*tpt.bounds);
    let ty_substs_len = vec::len(pth.types);

    // determine the region bound, using the value given by the user
    // (if any) and otherwise using a fresh region variable
    let self_r = match pth.rp {
      Some(r) => {
        match tpt.region_param {
          None => {
            fcx.ccx.tcx.sess.span_err
                (span, ~"this item is not region-parameterized");
            None
          }
          Some(_) => {
            Some(ast_region_to_region(fcx, fcx, span, r))
          }
        }
      }
      None => {
        fcx.region_var_if_parameterized(
            tpt.region_param, span, region_lb)
      }
    };

    // determine values for type parameters, using the values given by
    // the user (if any) and otherwise using fresh type variables
    let tps = if ty_substs_len == 0u {
        fcx.infcx().next_ty_vars(ty_param_count)
    } else if ty_param_count == 0u {
        fcx.ccx.tcx.sess.span_err
            (span, ~"this item does not take type parameters");
        fcx.infcx().next_ty_vars(ty_param_count)
    } else if ty_substs_len > ty_param_count {
        fcx.ccx.tcx.sess.span_err
            (span, ~"too many type parameters provided for this item");
        fcx.infcx().next_ty_vars(ty_param_count)
    } else if ty_substs_len < ty_param_count {
        fcx.ccx.tcx.sess.span_err
            (span, ~"not enough type parameters provided for this item");
        fcx.infcx().next_ty_vars(ty_param_count)
    } else {
        pth.types.map(|aty| fcx.to_ty(aty))
    };

    let substs = {self_r: self_r, self_ty: None, tps: tps};
    fcx.write_ty_substs(node_id, tpt.ty, substs);
}

// Resolves `typ` by a single level if `typ` is a type variable.  If no
// resolution is possible, then an error is reported.
fn structurally_resolved_type(fcx: @fn_ctxt, sp: span, tp: ty::t) -> ty::t {
    match infer::resolve_type(fcx.infcx(), tp, force_tvar) {
        Ok(t_s) if !ty::type_is_ty_var(t_s) => return t_s,
        _ => {
            fcx.ccx.tcx.sess.span_fatal
                (sp, ~"the type of this value must be known in this context");
        }
    }
}

// Returns the one-level-deep structure of the given type.
fn structure_of(fcx: @fn_ctxt, sp: span, typ: ty::t) -> ty::sty {
    ty::get(structurally_resolved_type(fcx, sp, typ)).sty
}

fn type_is_integral(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_integral(typ_s);
}

fn type_is_scalar(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_scalar(typ_s);
}

fn type_is_c_like_enum(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_c_like_enum(fcx.ccx.tcx, typ_s);
}

fn ast_expr_vstore_to_vstore(fcx: @fn_ctxt, e: @ast::expr, n: uint,
                             v: ast::expr_vstore) -> ty::vstore {
    match v {
        ast::expr_vstore_fixed(None) => ty::vstore_fixed(n),
        ast::expr_vstore_fixed(Some(u)) => {
            if n != u {
                let s = fmt!("fixed-size sequence mismatch: %u vs. %u",u, n);
                fcx.ccx.tcx.sess.span_err(e.span,s);
            }
            ty::vstore_fixed(u)
        }
        ast::expr_vstore_uniq => ty::vstore_uniq,
        ast::expr_vstore_box => ty::vstore_box,
        ast::expr_vstore_slice => {
            let r = fcx.infcx().next_region_var(e.span, e.id);
            ty::vstore_slice(r)
        }
    }
}

fn check_bounds_are_used(ccx: @crate_ctxt,
                         span: span,
                         tps: ~[ast::ty_param],
                         ty: ty::t) {
    debug!("check_bounds_are_used(n_tps=%u, ty=%s)",
           tps.len(), ty_to_str(ccx.tcx, ty));

    // make a vector of booleans initially false, set to true when used
    if tps.len() == 0u { return; }
    let tps_used = vec::to_mut(vec::from_elem(tps.len(), false));

    ty::walk_regions_and_ty(
        ccx.tcx, ty,
        |_r| {},
        |t| {
            match ty::get(t).sty {
              ty::ty_param({idx, _}) => {
                  debug!("Found use of ty param #%u", idx);
                  tps_used[idx] = true;
              }
              _ => ()
            }
            true
        });

    for tps_used.eachi |i, b| {
        if !b {
            ccx.tcx.sess.span_err(
                span, fmt!("type parameter `%s` is unused",
                           ccx.tcx.sess.str_of(tps[i].ident)));
        }
    }
}

fn check_intrinsic_type(ccx: @crate_ctxt, it: @ast::foreign_item) {
    fn param(ccx: @crate_ctxt, n: uint) -> ty::t {
        ty::mk_param(ccx.tcx, n, local_def(0))
    }
    fn arg(m: ast::rmode, ty: ty::t) -> ty::arg {
        {mode: ast::expl(m), ty: ty}
    }
    let tcx = ccx.tcx;
    let (n_tps, inputs, output) = match ccx.tcx.sess.str_of(it.ident) {
      ~"size_of" |
      ~"pref_align_of" | ~"min_align_of" => (1u, ~[], ty::mk_uint(ccx.tcx)),
      ~"init" => (1u, ~[], param(ccx, 0u)),
      ~"forget" => (1u, ~[arg(ast::by_move, param(ccx, 0u))],
                    ty::mk_nil(tcx)),
      ~"reinterpret_cast" => (2u, ~[arg(ast::by_ref, param(ccx, 0u))],
                              param(ccx, 1u)),
      ~"addr_of" => (1u, ~[arg(ast::by_ref, param(ccx, 0u))],
                      ty::mk_imm_ptr(tcx, param(ccx, 0u))),
      ~"move_val" | ~"move_val_init" => {
        (1u, ~[arg(ast::by_mutbl_ref, param(ccx, 0u)),
               arg(ast::by_move, param(ccx, 0u))],
         ty::mk_nil(tcx))
      }
      ~"needs_drop" => (1u, ~[], ty::mk_bool(tcx)),

      ~"atomic_xchg"     | ~"atomic_xadd"     | ~"atomic_xsub" |
      ~"atomic_xchg_acq" | ~"atomic_xadd_acq" | ~"atomic_xsub_acq" |
      ~"atomic_xchg_rel" | ~"atomic_xadd_rel" | ~"atomic_xsub_rel" => {
        (0u, ~[arg(ast::by_val,
                   ty::mk_mut_rptr(tcx, ty::re_bound(ty::br_anon(0)),
                                   ty::mk_int(tcx))),
               arg(ast::by_val, ty::mk_int(tcx))],
         ty::mk_int(tcx))
      }

      ~"get_tydesc" => {
        // FIXME (#2712): return *intrinsic::tydesc, not *()
        (1u, ~[], ty::mk_nil_ptr(tcx))
      }
      ~"visit_tydesc" => {
          let tydesc_name = syntax::parse::token::special_idents::tydesc;
          let ty_visitor_name = tcx.sess.ident_of(~"TyVisitor");
          assert tcx.intrinsic_defs.contains_key(tydesc_name);
          assert ccx.tcx.intrinsic_defs.contains_key(ty_visitor_name);
          let (_, tydesc_ty) = tcx.intrinsic_defs.get(tydesc_name);
          let (_, visitor_trait) = tcx.intrinsic_defs.get(ty_visitor_name);
          let td_ptr = ty::mk_ptr(ccx.tcx, {ty: tydesc_ty,
                                            mutbl: ast::m_imm});
          (0u, ~[arg(ast::by_val, td_ptr),
                 arg(ast::by_ref, visitor_trait)], ty::mk_nil(tcx))
      }
      ~"frame_address" => {
        let fty = ty::mk_fn(ccx.tcx, FnTyBase {
            meta: FnMeta {purity: ast::impure_fn,
                          proto: ty::proto_vstore(ty::vstore_slice(
                              ty::re_bound(ty::br_anon(0)))),
                          bounds: @~[],
                          ret_style: ast::return_val},
            sig: FnSig {inputs: ~[{mode: ast::expl(ast::by_val),
                                   ty: ty::mk_imm_ptr(
                                       ccx.tcx,
                                       ty::mk_mach_uint(ccx.tcx, ast::ty_u8))
                                  }],
                        output: ty::mk_nil(ccx.tcx)}
        });
        (0u, ~[arg(ast::by_ref, fty)], ty::mk_nil(tcx))
      }
      ~"morestack_addr" => {
        (0u, ~[], ty::mk_nil_ptr(tcx))
      }
      other => {
        tcx.sess.span_err(it.span, ~"unrecognized intrinsic function: `" +
                          other + ~"`");
        return;
      }
    };
    let fty = ty::mk_fn(tcx, FnTyBase {
        meta: FnMeta {purity: ast::impure_fn,
                      proto: ty::proto_bare,
                      bounds: @~[],
                      ret_style: ast::return_val},
        sig: FnSig {inputs: inputs,
                    output: output}
    });
    let i_ty = ty::lookup_item_type(ccx.tcx, local_def(it.id));
    let i_n_tps = (*i_ty.bounds).len();
    if i_n_tps != n_tps {
        tcx.sess.span_err(it.span, fmt!("intrinsic has wrong number \
                                         of type parameters: found %u, \
                                         expected %u", i_n_tps, n_tps));
    } else {
        require_same_types(
            tcx, None, false, it.span, i_ty.ty, fty,
            || fmt!("intrinsic has wrong type: \
                      expected `%s`",
                     ty_to_str(ccx.tcx, fty)));
    }
}
