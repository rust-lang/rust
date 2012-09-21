use codemap::span;
use ast::*;

export ast_fold_precursor;
export ast_fold;
export default_ast_fold;
export make_fold;
export noop_fold_crate;
export noop_fold_item;
export noop_fold_expr;
export noop_fold_pat;
export noop_fold_mod;
export noop_fold_ty;
export noop_fold_block;
export wrap;
export fold_ty_param;
export fold_ty_params;
export fold_fn_decl;
export extensions;

trait ast_fold {
    fn fold_crate(crate) -> crate;
    fn fold_crate_directive(&&@crate_directive) -> @crate_directive;
    fn fold_view_item(&&@view_item) -> @view_item;
    fn fold_foreign_item(&&@foreign_item) -> @foreign_item;
    fn fold_item(&&@item) -> Option<@item>;
    fn fold_struct_field(&&@struct_field) -> @struct_field;
    fn fold_item_underscore(item_) -> item_;
    fn fold_method(&&@method) -> @method;
    fn fold_block(blk) -> blk;
    fn fold_stmt(&&@stmt) -> @stmt;
    fn fold_arm(arm) -> arm;
    fn fold_pat(&&@pat) -> @pat;
    fn fold_decl(&&@decl) -> @decl;
    fn fold_expr(&&@expr) -> @expr;
    fn fold_ty(&&@ty) -> @ty;
    fn fold_mod(_mod) -> _mod;
    fn fold_foreign_mod(foreign_mod) -> foreign_mod;
    fn fold_variant(variant) -> variant;
    fn fold_ident(&&ident) -> ident;
    fn fold_path(&&@path) -> @path;
    fn fold_local(&&@local) -> @local;
    fn map_exprs(fn@(&&@expr) -> @expr, ~[@expr]) -> ~[@expr];
    fn new_id(node_id) -> node_id;
    fn new_span(span) -> span;
}

// We may eventually want to be able to fold over type parameters, too

type ast_fold_precursor = @{
    //unlike the others, item_ is non-trivial
    fold_crate: fn@(crate_, span, ast_fold) -> (crate_, span),
    fold_crate_directive: fn@(crate_directive_, span,
                              ast_fold) -> (crate_directive_, span),
    fold_view_item: fn@(view_item_, ast_fold) -> view_item_,
    fold_foreign_item: fn@(&&@foreign_item, ast_fold) -> @foreign_item,
    fold_item: fn@(&&@item, ast_fold) -> Option<@item>,
    fold_struct_field: fn@(&&@struct_field, ast_fold) -> @struct_field,
    fold_item_underscore: fn@(item_, ast_fold) -> item_,
    fold_method: fn@(&&@method, ast_fold) -> @method,
    fold_block: fn@(blk_, span, ast_fold) -> (blk_, span),
    fold_stmt: fn@(stmt_, span, ast_fold) -> (stmt_, span),
    fold_arm: fn@(arm, ast_fold) -> arm,
    fold_pat: fn@(pat_, span, ast_fold) -> (pat_, span),
    fold_decl: fn@(decl_, span, ast_fold) -> (decl_, span),
    fold_expr: fn@(expr_, span, ast_fold) -> (expr_, span),
    fold_ty: fn@(ty_, span, ast_fold) -> (ty_, span),
    fold_mod: fn@(_mod, ast_fold) -> _mod,
    fold_foreign_mod: fn@(foreign_mod, ast_fold) -> foreign_mod,
    fold_variant: fn@(variant_, span, ast_fold) -> (variant_, span),
    fold_ident: fn@(&&ident, ast_fold) -> ident,
    fold_path: fn@(path, ast_fold) -> path,
    fold_local: fn@(local_, span, ast_fold) -> (local_, span),
    map_exprs: fn@(fn@(&&@expr) -> @expr, ~[@expr]) -> ~[@expr],
    new_id: fn@(node_id) -> node_id,
    new_span: fn@(span) -> span};

/* some little folds that probably aren't useful to have in ast_fold itself*/

//used in noop_fold_item and noop_fold_crate and noop_fold_crate_directive
fn fold_meta_item_(&&mi: @meta_item, fld: ast_fold) -> @meta_item {
    return @{node:
              match mi.node {
                meta_word(id) => meta_word(id),
                meta_list(id, mis) => {
                  let fold_meta_item = |x|fold_meta_item_(x, fld);
                  meta_list(/* FIXME: (#2543) */ copy id,
                            vec::map(mis, fold_meta_item))
                }
                meta_name_value(id, s) => {
                  meta_name_value(id, /* FIXME (#2543) */ copy s)
                }
              },
          span: fld.new_span(mi.span)};
}
//used in noop_fold_item and noop_fold_crate
fn fold_attribute_(at: attribute, fld: ast_fold) ->
   attribute {
    return {node: {style: at.node.style,
                value: *fold_meta_item_(@at.node.value, fld),
                is_sugared_doc: at.node.is_sugared_doc },
         span: fld.new_span(at.span)};
}
//used in noop_fold_foreign_item and noop_fold_fn_decl
fn fold_arg_(a: arg, fld: ast_fold) -> arg {
    return {mode: a.mode,
         ty: fld.fold_ty(a.ty),
         ident: fld.fold_ident(a.ident),
         id: fld.new_id(a.id)};
}
//used in noop_fold_expr, and possibly elsewhere in the future
fn fold_mac_(m: mac, fld: ast_fold) -> mac {
    return {node:
             match m.node {
               mac_invoc(pth, arg, body) => {
                 mac_invoc(fld.fold_path(pth),
                           option::map(arg, |x| fld.fold_expr(x)), body)
               }
               mac_invoc_tt(*) => m.node,
               mac_ellipsis => mac_ellipsis,
               mac_aq(_,_) => /* FIXME (#2543) */ copy m.node,
               mac_var(_) => /* FIXME (#2543) */ copy m.node,
             },
         span: fld.new_span(m.span)};
}

fn fold_fn_decl(decl: ast::fn_decl, fld: ast_fold) -> ast::fn_decl {
    return {inputs: vec::map(decl.inputs, |x| fold_arg_(x, fld) ),
         output: fld.fold_ty(decl.output),
         cf: decl.cf}
}

fn fold_ty_param_bound(tpb: ty_param_bound, fld: ast_fold) -> ty_param_bound {
    match tpb {
      bound_copy | bound_send | bound_const | bound_owned => tpb,
      bound_trait(ty) => bound_trait(fld.fold_ty(ty))
    }
}

fn fold_ty_param(tp: ty_param, fld: ast_fold) -> ty_param {
    {ident: /* FIXME (#2543) */ copy tp.ident,
     id: fld.new_id(tp.id),
     bounds: @vec::map(*tp.bounds, |x| fold_ty_param_bound(x, fld) )}
}

fn fold_ty_params(tps: ~[ty_param], fld: ast_fold) -> ~[ty_param] {
    vec::map(tps, |x| fold_ty_param(x, fld) )
}

fn noop_fold_crate(c: crate_, fld: ast_fold) -> crate_ {
    let fold_meta_item = |x| fold_meta_item_(x, fld);
    let fold_attribute = |x| fold_attribute_(x, fld);

    return {
        directives: vec::map(c.directives, |x| fld.fold_crate_directive(x)),
        module: fld.fold_mod(c.module),
        attrs: vec::map(c.attrs, fold_attribute),
        config: vec::map(c.config, fold_meta_item)
    };
}

fn noop_fold_crate_directive(cd: crate_directive_, fld: ast_fold) ->
   crate_directive_ {
    return match cd {
          cdir_src_mod(id, attrs) => {
            cdir_src_mod(fld.fold_ident(id), /* FIXME (#2543) */ copy attrs)
          }
          cdir_dir_mod(id, cds, attrs) => {
            cdir_dir_mod(fld.fold_ident(id),
                         vec::map(cds, |x| fld.fold_crate_directive(x)),
                         /* FIXME (#2543) */ copy attrs)
          }
          cdir_view_item(vi) => cdir_view_item(fld.fold_view_item(vi)),
          cdir_syntax(_) => copy cd
        }
}

fn noop_fold_view_item(vi: view_item_, _fld: ast_fold) -> view_item_ {
    return /* FIXME (#2543) */ copy vi;
}


fn noop_fold_foreign_item(&&ni: @foreign_item, fld: ast_fold)
    -> @foreign_item {
    let fold_arg = |x| fold_arg_(x, fld);
    let fold_attribute = |x| fold_attribute_(x, fld);

    return @{ident: fld.fold_ident(ni.ident),
          attrs: vec::map(ni.attrs, fold_attribute),
          node:
              match ni.node {
                foreign_item_fn(fdec, purity, typms) => {
                  foreign_item_fn({inputs: vec::map(fdec.inputs, fold_arg),
                                   output: fld.fold_ty(fdec.output),
                                   cf: fdec.cf},
                                  purity,
                                  fold_ty_params(typms, fld))
                }
                foreign_item_const(t) => {
                  foreign_item_const(fld.fold_ty(t))
                }
              },
          id: fld.new_id(ni.id),
          span: fld.new_span(ni.span)};
}

fn noop_fold_item(&&i: @item, fld: ast_fold) -> Option<@item> {
    let fold_attribute = |x| fold_attribute_(x, fld);

    return Some(@{ident: fld.fold_ident(i.ident),
               attrs: vec::map(i.attrs, fold_attribute),
               id: fld.new_id(i.id),
               node: fld.fold_item_underscore(i.node),
               vis: i.vis,
               span: fld.new_span(i.span)});
}

fn noop_fold_struct_field(&&sf: @struct_field, fld: ast_fold)
                       -> @struct_field {
    @{node: {kind: copy sf.node.kind,
             id: sf.node.id,
             ty: fld.fold_ty(sf.node.ty)},
      span: sf.span}
}

fn noop_fold_item_underscore(i: item_, fld: ast_fold) -> item_ {
    return match i {
          item_const(t, e) => item_const(fld.fold_ty(t), fld.fold_expr(e)),
          item_fn(decl, purity, typms, body) => {
              item_fn(fold_fn_decl(decl, fld),
                      purity,
                      fold_ty_params(typms, fld),
                      fld.fold_block(body))
          }
          item_mod(m) => item_mod(fld.fold_mod(m)),
          item_foreign_mod(nm) => item_foreign_mod(fld.fold_foreign_mod(nm)),
          item_ty(t, typms) => item_ty(fld.fold_ty(t),
                                       fold_ty_params(typms, fld)),
          item_enum(enum_definition, typms) => {
            item_enum(ast::enum_def({
                variants: vec::map(enum_definition.variants,
                                   |x| fld.fold_variant(x)),
                common: option::map(enum_definition.common,
                                    |x| fold_struct_def(x, fld))
            }), fold_ty_params(typms, fld))
          }
          item_class(struct_def, typms) => {
            let struct_def = fold_struct_def(struct_def, fld);
              item_class(struct_def, /* FIXME (#2543) */ copy typms)
          }
          item_impl(tps, ifce, ty, methods) => {
              item_impl(fold_ty_params(tps, fld),
                        ifce.map(|p| fold_trait_ref(p, fld)),
                        fld.fold_ty(ty),
                        vec::map(methods, |x| fld.fold_method(x)))
          }
          item_trait(tps, traits, methods) => {
            item_trait(fold_ty_params(tps, fld),
                       vec::map(traits, |p| fold_trait_ref(p, fld)),
                       /* FIXME (#2543) */ copy methods)
          }
      item_mac(m) => {
        // FIXME #2888: we might actually want to do something here.
        item_mac(m)
      }
        };
}

fn fold_struct_def(struct_def: @ast::struct_def, fld: ast_fold)
                -> @ast::struct_def {
    let resulting_optional_constructor;
    match struct_def.ctor {
        None => {
            resulting_optional_constructor = None;
        }
        Some(constructor) => {
            resulting_optional_constructor = Some({
                node: {
                    body: fld.fold_block(constructor.node.body),
                    dec: fold_fn_decl(constructor.node.dec, fld),
                    id: fld.new_id(constructor.node.id),
                    .. constructor.node
                },
                .. constructor
            });
        }
    }
    let dtor = do option::map(struct_def.dtor) |dtor| {
        let dtor_body = fld.fold_block(dtor.node.body);
        let dtor_id   = fld.new_id(dtor.node.id);
        {node: {body: dtor_body,
                id: dtor_id,.. dtor.node},
            .. dtor}};
    return @{
        traits: vec::map(struct_def.traits, |p| fold_trait_ref(p, fld)),
        fields: vec::map(struct_def.fields, |f| fold_struct_field(f, fld)),
        methods: vec::map(struct_def.methods, |m| fld.fold_method(m)),
        ctor: resulting_optional_constructor,
        dtor: dtor
    };
}

fn fold_trait_ref(&&p: @trait_ref, fld: ast_fold) -> @trait_ref {
    @{path: fld.fold_path(p.path), ref_id: fld.new_id(p.ref_id),
     impl_id: fld.new_id(p.impl_id)}
}

fn fold_struct_field(&&f: @struct_field, fld: ast_fold) -> @struct_field {
    @{node: {kind: copy f.node.kind,
             id: fld.new_id(f.node.id),
             ty: fld.fold_ty(f.node.ty)},
      span: fld.new_span(f.span)}
}

fn noop_fold_method(&&m: @method, fld: ast_fold) -> @method {
    return @{ident: fld.fold_ident(m.ident),
          attrs: /* FIXME (#2543) */ copy m.attrs,
          tps: fold_ty_params(m.tps, fld),
          self_ty: m.self_ty,
          purity: m.purity,
          decl: fold_fn_decl(m.decl, fld),
          body: fld.fold_block(m.body),
          id: fld.new_id(m.id),
          span: fld.new_span(m.span),
          self_id: fld.new_id(m.self_id),
          vis: m.vis};
}


fn noop_fold_block(b: blk_, fld: ast_fold) -> blk_ {
    return {view_items: vec::map(b.view_items, |x| fld.fold_view_item(x)),
         stmts: vec::map(b.stmts, |x| fld.fold_stmt(x)),
         expr: option::map(b.expr, |x| fld.fold_expr(x)),
         id: fld.new_id(b.id),
         rules: b.rules};
}

fn noop_fold_stmt(s: stmt_, fld: ast_fold) -> stmt_ {
    return match s {
      stmt_decl(d, nid) => stmt_decl(fld.fold_decl(d), fld.new_id(nid)),
      stmt_expr(e, nid) => stmt_expr(fld.fold_expr(e), fld.new_id(nid)),
      stmt_semi(e, nid) => stmt_semi(fld.fold_expr(e), fld.new_id(nid))
    };
}

fn noop_fold_arm(a: arm, fld: ast_fold) -> arm {
    return {pats: vec::map(a.pats, |x| fld.fold_pat(x)),
         guard: option::map(a.guard, |x| fld.fold_expr(x)),
         body: fld.fold_block(a.body)};
}

fn noop_fold_pat(p: pat_, fld: ast_fold) -> pat_ {
    return match p {
          pat_wild => pat_wild,
          pat_ident(binding_mode, pth, sub) => {
            pat_ident(binding_mode,
                      fld.fold_path(pth),
                      option::map(sub, |x| fld.fold_pat(x)))
          }
          pat_lit(e) => pat_lit(fld.fold_expr(e)),
          pat_enum(pth, pats) => {
              pat_enum(fld.fold_path(pth), option::map(pats,
                       |pats| vec::map(pats, |x| fld.fold_pat(x))))
          }
          pat_rec(fields, etc) => {
            let mut fs = ~[];
            for fields.each |f| {
                vec::push(fs,
                          {ident: /* FIXME (#2543) */ copy f.ident,
                           pat: fld.fold_pat(f.pat)});
            }
            pat_rec(fs, etc)
          }
          pat_struct(pth, fields, etc) => {
            let pth_ = fld.fold_path(pth);
            let mut fs = ~[];
            for fields.each |f| {
                vec::push(fs,
                          {ident: /* FIXME (#2543) */ copy f.ident,
                           pat: fld.fold_pat(f.pat)});
            }
            pat_struct(pth_, fs, etc)
          }
          pat_tup(elts) => pat_tup(vec::map(elts, |x| fld.fold_pat(x))),
          pat_box(inner) => pat_box(fld.fold_pat(inner)),
          pat_uniq(inner) => pat_uniq(fld.fold_pat(inner)),
          pat_region(inner) => pat_region(fld.fold_pat(inner)),
          pat_range(e1, e2) => {
            pat_range(fld.fold_expr(e1), fld.fold_expr(e2))
          }
        };
}

fn noop_fold_decl(d: decl_, fld: ast_fold) -> decl_ {
    match d {
      decl_local(ls) => decl_local(vec::map(ls, |x| fld.fold_local(x))),
      decl_item(it) => match fld.fold_item(it) {
        Some(it_folded) => decl_item(it_folded),
        None => decl_local(~[])
      }
    }
}

fn wrap<T>(f: fn@(T, ast_fold) -> T)
    -> fn@(T, span, ast_fold) -> (T, span)
{
    return fn@(x: T, s: span, fld: ast_fold) -> (T, span) {
        (f(x, fld), s)
    }
}

fn noop_fold_expr(e: expr_, fld: ast_fold) -> expr_ {
    fn fold_field_(field: field, fld: ast_fold) -> field {
        return {node:
                 {mutbl: field.node.mutbl,
                  ident: fld.fold_ident(field.node.ident),
                  expr: fld.fold_expr(field.node.expr)},
             span: fld.new_span(field.span)};
    }
    let fold_field = |x| fold_field_(x, fld);

    let fold_mac = |x| fold_mac_(x, fld);

    return match e {
          expr_vstore(e, v) => {
            expr_vstore(fld.fold_expr(e), v)
          }
          expr_vec(exprs, mutt) => {
            expr_vec(fld.map_exprs(|x| fld.fold_expr(x), exprs), mutt)
          }
          expr_repeat(expr, count, mutt) =>
            expr_repeat(fld.fold_expr(expr), fld.fold_expr(count), mutt),
          expr_rec(fields, maybe_expr) => {
            expr_rec(vec::map(fields, fold_field),
                     option::map(maybe_expr, |x| fld.fold_expr(x)))
          }
          expr_tup(elts) => expr_tup(vec::map(elts, |x| fld.fold_expr(x))),
          expr_call(f, args, blk) => {
            expr_call(fld.fold_expr(f),
                      fld.map_exprs(|x| fld.fold_expr(x), args),
                      blk)
          }
          expr_binary(binop, lhs, rhs) => {
            expr_binary(binop, fld.fold_expr(lhs), fld.fold_expr(rhs))
          }
          expr_unary(binop, ohs) => expr_unary(binop, fld.fold_expr(ohs)),
          expr_loop_body(f) => expr_loop_body(fld.fold_expr(f)),
          expr_do_body(f) => expr_do_body(fld.fold_expr(f)),
          expr_lit(_) => copy e,
          expr_cast(expr, ty) => expr_cast(fld.fold_expr(expr), ty),
          expr_addr_of(m, ohs) => expr_addr_of(m, fld.fold_expr(ohs)),
          expr_if(cond, tr, fl) => {
            expr_if(fld.fold_expr(cond), fld.fold_block(tr),
                    option::map(fl, |x| fld.fold_expr(x)))
          }
          expr_while(cond, body) => {
            expr_while(fld.fold_expr(cond), fld.fold_block(body))
          }
          expr_loop(body, opt_ident) => {
              expr_loop(fld.fold_block(body),
                        option::map(opt_ident, |x| fld.fold_ident(x)))
          }
          expr_match(expr, arms) => {
            expr_match(fld.fold_expr(expr),
                     vec::map(arms, |x| fld.fold_arm(x)))
          }
          expr_fn(proto, decl, body, captures) => {
            expr_fn(proto, fold_fn_decl(decl, fld),
                    fld.fold_block(body),
                    @((*captures).map(|cap_item| {
                        @({id: fld.new_id((*cap_item).id),
                           .. *cap_item})})))
          }
          expr_fn_block(decl, body, captures) => {
            expr_fn_block(fold_fn_decl(decl, fld), fld.fold_block(body),
                          @((*captures).map(|cap_item| {
                              @({id: fld.new_id((*cap_item).id),
                                 .. *cap_item})})))
          }
          expr_block(blk) => expr_block(fld.fold_block(blk)),
          expr_move(el, er) => {
            expr_move(fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_copy(e) => expr_copy(fld.fold_expr(e)),
          expr_unary_move(e) => expr_unary_move(fld.fold_expr(e)),
          expr_assign(el, er) => {
            expr_assign(fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_swap(el, er) => {
            expr_swap(fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_assign_op(op, el, er) => {
            expr_assign_op(op, fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_field(el, id, tys) => {
            expr_field(fld.fold_expr(el), fld.fold_ident(id),
                       vec::map(tys, |x| fld.fold_ty(x)))
          }
          expr_index(el, er) => {
            expr_index(fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_path(pth) => expr_path(fld.fold_path(pth)),
          expr_fail(e) => expr_fail(option::map(e, |x| fld.fold_expr(x))),
          expr_break(opt_ident) =>
            expr_break(option::map(opt_ident, |x| fld.fold_ident(x))),
          expr_again(opt_ident) =>
            expr_again(option::map(opt_ident, |x| fld.fold_ident(x))),
          expr_ret(e) => expr_ret(option::map(e, |x| fld.fold_expr(x))),
          expr_log(i, lv, e) => expr_log(i, fld.fold_expr(lv),
                                         fld.fold_expr(e)),
          expr_assert(e) => expr_assert(fld.fold_expr(e)),
          expr_mac(mac) => expr_mac(fold_mac(mac)),
          expr_struct(path, fields, maybe_expr) => {
            expr_struct(fld.fold_path(path),
                        vec::map(fields, fold_field),
                        option::map(maybe_expr, |x| fld.fold_expr(x)))
          }
        }
}

fn noop_fold_ty(t: ty_, fld: ast_fold) -> ty_ {
    let fold_mac = |x| fold_mac_(x, fld);
    fn fold_mt(mt: mt, fld: ast_fold) -> mt {
        {ty: fld.fold_ty(mt.ty), mutbl: mt.mutbl}
    }
    fn fold_field(f: ty_field, fld: ast_fold) -> ty_field {
        {node: {ident: fld.fold_ident(f.node.ident),
                mt: fold_mt(f.node.mt, fld)},
         span: fld.new_span(f.span)}
    }
    match t {
      ty_nil | ty_bot | ty_infer => copy t,
      ty_box(mt) => ty_box(fold_mt(mt, fld)),
      ty_uniq(mt) => ty_uniq(fold_mt(mt, fld)),
      ty_vec(mt) => ty_vec(fold_mt(mt, fld)),
      ty_ptr(mt) => ty_ptr(fold_mt(mt, fld)),
      ty_rptr(region, mt) => ty_rptr(region, fold_mt(mt, fld)),
      ty_rec(fields) => ty_rec(vec::map(fields, |f| fold_field(f, fld))),
      ty_fn(proto, purity, bounds, decl) =>
        ty_fn(proto, purity,
              @vec::map(*bounds,
                        |x| fold_ty_param_bound(x, fld)),
              fold_fn_decl(decl, fld)),
      ty_tup(tys) => ty_tup(vec::map(tys, |ty| fld.fold_ty(ty))),
      ty_path(path, id) => ty_path(fld.fold_path(path), fld.new_id(id)),
      ty_fixed_length(t, vs) => ty_fixed_length(fld.fold_ty(t), vs),
      ty_mac(mac) => ty_mac(fold_mac(mac))
    }
}

// ...nor do modules
fn noop_fold_mod(m: _mod, fld: ast_fold) -> _mod {
    return {view_items: vec::map(m.view_items, |x| fld.fold_view_item(x)),
         items: vec::filter_map(m.items, |x| fld.fold_item(x))};
}

fn noop_fold_foreign_mod(nm: foreign_mod, fld: ast_fold) -> foreign_mod {
    return {sort: nm.sort,
         view_items: vec::map(nm.view_items, |x| fld.fold_view_item(x)),
         items: vec::map(nm.items, |x| fld.fold_foreign_item(x))}
}

fn noop_fold_variant(v: variant_, fld: ast_fold) -> variant_ {
    fn fold_variant_arg_(va: variant_arg, fld: ast_fold) -> variant_arg {
        return {ty: fld.fold_ty(va.ty), id: fld.new_id(va.id)};
    }
    let fold_variant_arg = |x| fold_variant_arg_(x, fld);

    let kind;
    match v.kind {
        tuple_variant_kind(variant_args) =>
            kind = tuple_variant_kind(vec::map(variant_args,
                                               fold_variant_arg)),
        struct_variant_kind(struct_def) => {
            let dtor = do option::map(struct_def.dtor) |dtor| {
                let dtor_body = fld.fold_block(dtor.node.body);
                let dtor_id   = fld.new_id(dtor.node.id);
                {node: {body: dtor_body,
                        id: dtor_id,.. dtor.node},
                    .. dtor}};
            kind = struct_variant_kind(@{
                traits: ~[],
                fields: vec::map(struct_def.fields,
                                 |f| fld.fold_struct_field(f)),
                methods: vec::map(struct_def.methods, |m| fld.fold_method(m)),
                ctor: None,
                dtor: dtor
            })
        }

        enum_variant_kind(enum_definition) => {
            let variants = vec::map(enum_definition.variants,
                                    |x| fld.fold_variant(x));
            let common = option::map(enum_definition.common,
                                     |x| fold_struct_def(x, fld));
            kind = enum_variant_kind(ast::enum_def({ variants: variants,
                                                     common: common }));
        }
    }

    let fold_attribute = |x| fold_attribute_(x, fld);
    let attrs = vec::map(v.attrs, fold_attribute);

    let de = match v.disr_expr {
      Some(e) => Some(fld.fold_expr(e)),
      None => None
    };
    return {name: /* FIXME (#2543) */ copy v.name,
         attrs: attrs,
         kind: kind,
         id: fld.new_id(v.id),
         disr_expr: de,
         vis: v.vis};
}

fn noop_fold_ident(&&i: ident, _fld: ast_fold) -> ident {
    return /* FIXME (#2543) */ copy i;
}

fn noop_fold_path(&&p: path, fld: ast_fold) -> path {
    return {span: fld.new_span(p.span), global: p.global,
         idents: vec::map(p.idents, |x| fld.fold_ident(x)),
         rp: p.rp,
         types: vec::map(p.types, |x| fld.fold_ty(x))};
}

fn noop_fold_local(l: local_, fld: ast_fold) -> local_ {
    return {is_mutbl: l.is_mutbl,
         ty: fld.fold_ty(l.ty),
         pat: fld.fold_pat(l.pat),
         init:
             match l.init {
               option::None::<initializer> => l.init,
               option::Some::<initializer>(init) => {
                 option::Some::<initializer>({op: init.op,
                                              expr: fld.fold_expr(init.expr)})
               }
             },
         id: fld.new_id(l.id)};
}

/* temporarily eta-expand because of a compiler bug with using `fn<T>` as a
   value */
fn noop_map_exprs(f: fn@(&&@expr) -> @expr, es: ~[@expr]) -> ~[@expr] {
    return vec::map(es, f);
}

fn noop_id(i: node_id) -> node_id { return i; }

fn noop_span(sp: span) -> span { return sp; }

fn default_ast_fold() -> ast_fold_precursor {
    return @{fold_crate: wrap(noop_fold_crate),
          fold_crate_directive: wrap(noop_fold_crate_directive),
          fold_view_item: noop_fold_view_item,
          fold_foreign_item: noop_fold_foreign_item,
          fold_item: noop_fold_item,
          fold_struct_field: noop_fold_struct_field,
          fold_item_underscore: noop_fold_item_underscore,
          fold_method: noop_fold_method,
          fold_block: wrap(noop_fold_block),
          fold_stmt: wrap(noop_fold_stmt),
          fold_arm: noop_fold_arm,
          fold_pat: wrap(noop_fold_pat),
          fold_decl: wrap(noop_fold_decl),
          fold_expr: wrap(noop_fold_expr),
          fold_ty: wrap(noop_fold_ty),
          fold_mod: noop_fold_mod,
          fold_foreign_mod: noop_fold_foreign_mod,
          fold_variant: wrap(noop_fold_variant),
          fold_ident: noop_fold_ident,
          fold_path: noop_fold_path,
          fold_local: wrap(noop_fold_local),
          map_exprs: noop_map_exprs,
          new_id: noop_id,
          new_span: noop_span};
}

impl ast_fold_precursor: ast_fold {
    /* naturally, a macro to write these would be nice */
    fn fold_crate(c: crate) -> crate {
        let (n, s) = self.fold_crate(c.node, c.span, self as ast_fold);
        return {node: n, span: self.new_span(s)};
    }
    fn fold_crate_directive(&&c: @crate_directive) -> @crate_directive {
        let (n, s) = self.fold_crate_directive(c.node, c.span,
                                               self as ast_fold);
        return @{node: n,
              span: self.new_span(s)};
    }
    fn fold_view_item(&&x: @view_item) ->
       @view_item {
        return @{node: self.fold_view_item(x.node, self as ast_fold),
              attrs: vec::map(x.attrs, |a|
                  fold_attribute_(a, self as ast_fold)),
              vis: x.vis,
              span: self.new_span(x.span)};
    }
    fn fold_foreign_item(&&x: @foreign_item)
        -> @foreign_item {
        return self.fold_foreign_item(x, self as ast_fold);
    }
    fn fold_item(&&i: @item) -> Option<@item> {
        return self.fold_item(i, self as ast_fold);
    }
    fn fold_struct_field(&&sf: @struct_field) -> @struct_field {
        @{node: {kind: copy sf.node.kind,
                 id: sf.node.id,
                 ty: (self as ast_fold).fold_ty(sf.node.ty)},
          span: self.new_span(sf.span)}
    }
    fn fold_item_underscore(i: item_) ->
       item_ {
        return self.fold_item_underscore(i, self as ast_fold);
    }
    fn fold_method(&&x: @method)
        -> @method {
        return self.fold_method(x, self as ast_fold);
    }
    fn fold_block(x: blk) -> blk {
        let (n, s) = self.fold_block(x.node, x.span, self as ast_fold);
        return {node: n, span: self.new_span(s)};
    }
    fn fold_stmt(&&x: @stmt) -> @stmt {
        let (n, s) = self.fold_stmt(x.node, x.span, self as ast_fold);
        return @{node: n, span: self.new_span(s)};
    }
    fn fold_arm(x: arm) -> arm {
        return self.fold_arm(x, self as ast_fold);
    }
    fn fold_pat(&&x: @pat) -> @pat {
        let (n, s) =  self.fold_pat(x.node, x.span, self as ast_fold);
        return @{id: self.new_id(x.id),
              node: n,
              span: self.new_span(s)};
    }
    fn fold_decl(&&x: @decl) -> @decl {
        let (n, s) = self.fold_decl(x.node, x.span, self as ast_fold);
        return @{node: n, span: self.new_span(s)};
    }
    fn fold_expr(&&x: @expr) -> @expr {
        let (n, s) = self.fold_expr(x.node, x.span, self as ast_fold);
        return @{id: self.new_id(x.id),
              callee_id: self.new_id(x.callee_id),
              node: n,
              span: self.new_span(s)};
    }
    fn fold_ty(&&x: @ty) -> @ty {
        let (n, s) = self.fold_ty(x.node, x.span, self as ast_fold);
        return @{id: self.new_id(x.id), node: n, span: self.new_span(s)};
    }
    fn fold_mod(x: _mod) -> _mod {
        return self.fold_mod(x, self as ast_fold);
    }
    fn fold_foreign_mod(x: foreign_mod) ->
       foreign_mod {
        return self.fold_foreign_mod(x, self as ast_fold);
    }
    fn fold_variant(x: variant) ->
       variant {
        let (n, s) = self.fold_variant(x.node, x.span, self as ast_fold);
        return {node: n, span: self.new_span(s)};
    }
    fn fold_ident(&&x: ident) -> ident {
        return self.fold_ident(x, self as ast_fold);
    }
    fn fold_path(&&x: @path) -> @path {
        @self.fold_path(*x, self as ast_fold)
    }
    fn fold_local(&&x: @local) -> @local {
        let (n, s) = self.fold_local(x.node, x.span, self as ast_fold);
        return @{node: n, span: self.new_span(s)};
    }
    fn map_exprs(f: fn@(&&@expr) -> @expr, e: ~[@expr]) -> ~[@expr] {
        self.map_exprs(f, e)
    }
    fn new_id(node_id: ast::node_id) -> node_id {
        self.new_id(node_id)
    }
    fn new_span(span: span) -> span {
        self.new_span(span)
    }
}

impl ast_fold {
    fn fold_attributes(attrs: ~[attribute]) -> ~[attribute] {
        attrs.map(|x| fold_attribute_(x, self))
    }
}

fn make_fold(afp: ast_fold_precursor) -> ast_fold {
    afp as ast_fold
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
