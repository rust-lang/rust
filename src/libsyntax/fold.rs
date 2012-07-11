import codemap::span;
import ast::*;

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

iface ast_fold {
    fn fold_crate(crate) -> crate;
    fn fold_crate_directive(&&@crate_directive) -> @crate_directive;
    fn fold_view_item(&&@view_item) -> @view_item;
    fn fold_foreign_item(&&@foreign_item) -> @foreign_item;
    fn fold_item(&&@item) -> option<@item>;
    fn fold_class_item(&&@class_member) -> @class_member;
    fn fold_item_underscore(item_) -> item_;
    fn fold_method(&&@method) -> @method;
    fn fold_block(blk) -> blk;
    fn fold_stmt(&&@stmt) -> @stmt;
    fn fold_arm(arm) -> arm;
    fn fold_pat(&&@pat) -> @pat;
    fn fold_decl(&&@decl) -> @decl;
    fn fold_expr(&&@expr) -> @expr;
    fn fold_ty(&&@ty) -> @ty;
    fn fold_constr(&&@constr) -> @constr;
    fn fold_ty_constr(&&@ty_constr) -> @ty_constr;
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
    fold_item: fn@(&&@item, ast_fold) -> option<@item>,
    fold_class_item: fn@(&&@class_member, ast_fold) -> @class_member,
    fold_item_underscore: fn@(item_, ast_fold) -> item_,
    fold_method: fn@(&&@method, ast_fold) -> @method,
    fold_block: fn@(blk_, span, ast_fold) -> (blk_, span),
    fold_stmt: fn@(stmt_, span, ast_fold) -> (stmt_, span),
    fold_arm: fn@(arm, ast_fold) -> arm,
    fold_pat: fn@(pat_, span, ast_fold) -> (pat_, span),
    fold_decl: fn@(decl_, span, ast_fold) -> (decl_, span),
    fold_expr: fn@(expr_, span, ast_fold) -> (expr_, span),
    fold_ty: fn@(ty_, span, ast_fold) -> (ty_, span),
    fold_constr: fn@(ast::constr_, span, ast_fold) -> (constr_, span),
    fold_ty_constr: fn@(ast::ty_constr_, span, ast_fold)
        -> (ty_constr_, span),
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
    ret @{node:
              alt mi.node {
                meta_word(id) { meta_word(fld.fold_ident(id)) }
                meta_list(id, mis) {
                  let fold_meta_item = |x|fold_meta_item_(x, fld);
                  meta_list(/* FIXME: (#2543) */ copy id,
                            vec::map(mis, fold_meta_item))
                }
                meta_name_value(id, s) {
                  meta_name_value(fld.fold_ident(id),
                                  /* FIXME (#2543) */ copy s)
                }
              },
          span: fld.new_span(mi.span)};
}
//used in noop_fold_item and noop_fold_crate
fn fold_attribute_(at: attribute, fld: ast_fold) ->
   attribute {
    ret {node: {style: at.node.style,
                value: *fold_meta_item_(@at.node.value, fld),
                is_sugared_doc: at.node.is_sugared_doc },
         span: fld.new_span(at.span)};
}
//used in noop_fold_foreign_item and noop_fold_fn_decl
fn fold_arg_(a: arg, fld: ast_fold) -> arg {
    ret {mode: a.mode,
         ty: fld.fold_ty(a.ty),
         ident: fld.fold_ident(a.ident),
         id: fld.new_id(a.id)};
}
//used in noop_fold_expr, and possibly elsewhere in the future
fn fold_mac_(m: mac, fld: ast_fold) -> mac {
    ret {node:
             alt m.node {
               mac_invoc(pth, arg, body) {
                 mac_invoc(fld.fold_path(pth),
                           option::map(arg, |x| fld.fold_expr(x)), body)
               }
               mac_invoc_tt(pth, tt) { m.node }
               mac_embed_type(ty) { mac_embed_type(fld.fold_ty(ty)) }
               mac_embed_block(blk) { mac_embed_block(fld.fold_block(blk)) }
               mac_ellipsis { mac_ellipsis }
               mac_aq(_,_) { /* FIXME (#2543) */ copy m.node }
               mac_var(_) { /* FIXME (#2543) */ copy m.node }
             },
         span: fld.new_span(m.span)};
}

fn fold_fn_decl(decl: ast::fn_decl, fld: ast_fold) -> ast::fn_decl {
    ret {inputs: vec::map(decl.inputs, |x| fold_arg_(x, fld) ),
         output: fld.fold_ty(decl.output),
         purity: decl.purity,
         cf: decl.cf,
         constraints: vec::map(decl.constraints, |x| fld.fold_constr(x))}
}

fn fold_ty_param_bound(tpb: ty_param_bound, fld: ast_fold) -> ty_param_bound {
    alt tpb {
      bound_copy | bound_send | bound_const { tpb }
      bound_trait(ty) { bound_trait(fld.fold_ty(ty)) }
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

    ret {directives: vec::map(c.directives, |x| fld.fold_crate_directive(x)),
         module: fld.fold_mod(c.module),
         attrs: vec::map(c.attrs, fold_attribute),
         config: vec::map(c.config, fold_meta_item)};
}

fn noop_fold_crate_directive(cd: crate_directive_, fld: ast_fold) ->
   crate_directive_ {
    ret alt cd {
          cdir_src_mod(id, attrs) {
            cdir_src_mod(fld.fold_ident(id), /* FIXME (#2543) */ copy attrs)
          }
          cdir_dir_mod(id, cds, attrs) {
            cdir_dir_mod(fld.fold_ident(id),
                         vec::map(cds, |x| fld.fold_crate_directive(x)),
                         /* FIXME (#2543) */ copy attrs)
          }
          cdir_view_item(vi) { cdir_view_item(fld.fold_view_item(vi)) }
          cdir_syntax(_) { copy cd }
        }
}

fn noop_fold_view_item(vi: view_item_, _fld: ast_fold) -> view_item_ {
    ret /* FIXME (#2543) */ copy vi;
}


fn noop_fold_foreign_item(&&ni: @foreign_item, fld: ast_fold)
    -> @foreign_item {
    let fold_arg = |x| fold_arg_(x, fld);
    let fold_attribute = |x| fold_attribute_(x, fld);

    ret @{ident: fld.fold_ident(ni.ident),
          attrs: vec::map(ni.attrs, fold_attribute),
          node:
              alt ni.node {
                foreign_item_fn(fdec, typms) {
                  foreign_item_fn({inputs: vec::map(fdec.inputs, fold_arg),
                                  output: fld.fold_ty(fdec.output),
                                  purity: fdec.purity,
                                  cf: fdec.cf,
                                  constraints:
                                      vec::map(fdec.constraints,
                                               |x| fld.fold_constr(x))},
                                 fold_ty_params(typms, fld))
                }
              },
          id: fld.new_id(ni.id),
          span: fld.new_span(ni.span)};
}

fn noop_fold_item(&&i: @item, fld: ast_fold) -> option<@item> {
    let fold_attribute = |x| fold_attribute_(x, fld);

    ret some(@{ident: fld.fold_ident(i.ident),
               attrs: vec::map(i.attrs, fold_attribute),
               id: fld.new_id(i.id),
               node: fld.fold_item_underscore(i.node),
               vis: i.vis,
               span: fld.new_span(i.span)});
}

fn noop_fold_class_item(&&ci: @class_member, fld: ast_fold)
    -> @class_member {
    @{node: alt ci.node {
        instance_var(ident, t, cm, id, p) {
           instance_var(/* FIXME (#2543) */ copy ident,
                        fld.fold_ty(t), cm, id, p)
        }
        class_method(m) { class_method(fld.fold_method(m)) }
      },
      span: ci.span}
}

fn noop_fold_item_underscore(i: item_, fld: ast_fold) -> item_ {
    ret alt i {
          item_const(t, e) { item_const(fld.fold_ty(t), fld.fold_expr(e)) }
          item_fn(decl, typms, body) {
              item_fn(fold_fn_decl(decl, fld),
                      fold_ty_params(typms, fld),
                      fld.fold_block(body))
          }
          item_mod(m) { item_mod(fld.fold_mod(m)) }
          item_foreign_mod(nm) { item_foreign_mod(fld.fold_foreign_mod(nm)) }
          item_ty(t, typms) { item_ty(fld.fold_ty(t),
                                      fold_ty_params(typms, fld)) }
          item_enum(variants, typms) {
            item_enum(vec::map(variants, |x| fld.fold_variant(x)),
                      fold_ty_params(typms, fld))
          }
          item_class(typms, traits, items, ctor, m_dtor) {
              let ctor_body = fld.fold_block(ctor.node.body);
              let ctor_decl = fold_fn_decl(ctor.node.dec, fld);
              let ctor_id   = fld.new_id(ctor.node.id);
            let dtor = do option::map(m_dtor) |dtor| {
                let dtor_body = fld.fold_block(dtor.node.body);
                let dtor_id   = fld.new_id(dtor.node.id);
                {node: {body: dtor_body,
                        id: dtor_id with dtor.node}
                    with dtor}};
              item_class(
                  /* FIXME (#2543) */ copy typms,
                  vec::map(traits, |p| fold_trait_ref(p, fld)),
                  vec::map(items, |x| fld.fold_class_item(x)),
                  {node: {body: ctor_body,
                          dec: ctor_decl,
                          id: ctor_id with ctor.node}
                      with ctor}, dtor)
          }
          item_impl(tps, ifce, ty, methods) {
              item_impl(fold_ty_params(tps, fld),
                        ifce.map(|p| fold_trait_ref(p, fld)),
                        fld.fold_ty(ty),
                        vec::map(methods, |x| fld.fold_method(x)))
          }
          item_trait(tps, methods) {
            item_trait(fold_ty_params(tps, fld),
                       /* FIXME (#2543) */ copy methods)
          }
      item_mac(m) {
        // TODO: we might actually want to do something here.
        item_mac(m)
      }
        };
}

fn fold_trait_ref(&&p: @trait_ref, fld: ast_fold) -> @trait_ref {
    @{path: fld.fold_path(p.path), id: fld.new_id(p.id)}
}

fn noop_fold_method(&&m: @method, fld: ast_fold) -> @method {
    ret @{ident: fld.fold_ident(m.ident),
          attrs: /* FIXME (#2543) */ copy m.attrs,
          tps: fold_ty_params(m.tps, fld),
          decl: fold_fn_decl(m.decl, fld),
          body: fld.fold_block(m.body),
          id: fld.new_id(m.id),
          span: fld.new_span(m.span),
          self_id: fld.new_id(m.self_id),
          vis: m.vis};
}


fn noop_fold_block(b: blk_, fld: ast_fold) -> blk_ {
    ret {view_items: vec::map(b.view_items, |x| fld.fold_view_item(x)),
         stmts: vec::map(b.stmts, |x| fld.fold_stmt(x)),
         expr: option::map(b.expr, |x| fld.fold_expr(x)),
         id: fld.new_id(b.id),
         rules: b.rules};
}

fn noop_fold_stmt(s: stmt_, fld: ast_fold) -> stmt_ {
    ret alt s {
      stmt_decl(d, nid) { stmt_decl(fld.fold_decl(d), fld.new_id(nid)) }
      stmt_expr(e, nid) { stmt_expr(fld.fold_expr(e), fld.new_id(nid)) }
      stmt_semi(e, nid) { stmt_semi(fld.fold_expr(e), fld.new_id(nid)) }
    };
}

fn noop_fold_arm(a: arm, fld: ast_fold) -> arm {
    ret {pats: vec::map(a.pats, |x| fld.fold_pat(x)),
         guard: option::map(a.guard, |x| fld.fold_expr(x)),
         body: fld.fold_block(a.body)};
}

fn noop_fold_pat(p: pat_, fld: ast_fold) -> pat_ {
    ret alt p {
          pat_wild { pat_wild }
          pat_ident(pth, sub) {
            pat_ident(fld.fold_path(pth),
                      option::map(sub, |x| fld.fold_pat(x)))
          }
          pat_lit(e) { pat_lit(fld.fold_expr(e)) }
          pat_enum(pth, pats) {
              pat_enum(fld.fold_path(pth), option::map(pats,
                       |pats| vec::map(pats, |x| fld.fold_pat(x))))
          }
          pat_rec(fields, etc) {
            let mut fs = ~[];
            for fields.each |f| {
                vec::push(fs,
                          {ident: /* FIXME (#2543) */ copy f.ident,
                           pat: fld.fold_pat(f.pat)});
            }
            pat_rec(fs, etc)
          }
          pat_tup(elts) { pat_tup(vec::map(elts, |x| fld.fold_pat(x))) }
          pat_box(inner) { pat_box(fld.fold_pat(inner)) }
          pat_uniq(inner) { pat_uniq(fld.fold_pat(inner)) }
          pat_range(e1, e2) {
            pat_range(fld.fold_expr(e1), fld.fold_expr(e2))
          }
        };
}

fn noop_fold_decl(d: decl_, fld: ast_fold) -> decl_ {
    alt d {
      decl_local(ls) { decl_local(vec::map(ls, |x| fld.fold_local(x))) }
      decl_item(it) {
        alt fld.fold_item(it) {
          some(it_folded) { decl_item(it_folded) }
          none { decl_local(~[]) }
        }
      }
    }
}

fn wrap<T>(f: fn@(T, ast_fold) -> T)
    -> fn@(T, span, ast_fold) -> (T, span)
{
    ret fn@(x: T, s: span, fld: ast_fold) -> (T, span) {
        (f(x, fld), s)
    }
}

fn noop_fold_expr(e: expr_, fld: ast_fold) -> expr_ {
    fn fold_field_(field: field, fld: ast_fold) -> field {
        ret {node:
                 {mutbl: field.node.mutbl,
                  ident: fld.fold_ident(field.node.ident),
                  expr: fld.fold_expr(field.node.expr)},
             span: fld.new_span(field.span)};
    }
    let fold_field = |x| fold_field_(x, fld);

    let fold_mac = |x| fold_mac_(x, fld);

    ret alt e {
          expr_new(p, i, v) {
            expr_new(fld.fold_expr(p),
                     fld.new_id(i),
                     fld.fold_expr(v))
          }
          expr_vstore(e, v) {
            expr_vstore(fld.fold_expr(e), v)
          }
          expr_vec(exprs, mutt) {
            expr_vec(fld.map_exprs(|x| fld.fold_expr(x), exprs), mutt)
          }
          expr_rec(fields, maybe_expr) {
            expr_rec(vec::map(fields, fold_field),
                     option::map(maybe_expr, |x| fld.fold_expr(x)))
          }
          expr_tup(elts) { expr_tup(vec::map(elts, |x| fld.fold_expr(x))) }
          expr_call(f, args, blk) {
            expr_call(fld.fold_expr(f),
                      fld.map_exprs(|x| fld.fold_expr(x), args),
                      blk)
          }
          expr_binary(binop, lhs, rhs) {
            expr_binary(binop, fld.fold_expr(lhs), fld.fold_expr(rhs))
          }
          expr_unary(binop, ohs) { expr_unary(binop, fld.fold_expr(ohs)) }
          expr_loop_body(f) { expr_loop_body(fld.fold_expr(f)) }
          expr_do_body(f) { expr_do_body(fld.fold_expr(f)) }
          expr_lit(_) { copy e }
          expr_cast(expr, ty) { expr_cast(fld.fold_expr(expr), ty) }
          expr_addr_of(m, ohs) { expr_addr_of(m, fld.fold_expr(ohs)) }
          expr_if(cond, tr, fl) {
            expr_if(fld.fold_expr(cond), fld.fold_block(tr),
                    option::map(fl, |x| fld.fold_expr(x)))
          }
          expr_while(cond, body) {
            expr_while(fld.fold_expr(cond), fld.fold_block(body))
          }
          expr_loop(body) {
              expr_loop(fld.fold_block(body))
          }
          expr_alt(expr, arms, mode) {
            expr_alt(fld.fold_expr(expr),
                     vec::map(arms, |x| fld.fold_arm(x)), mode)
          }
          expr_fn(proto, decl, body, captures) {
            expr_fn(proto, fold_fn_decl(decl, fld),
                    fld.fold_block(body),
                    @((*captures).map(|cap_item| {
                        @({id: fld.new_id((*cap_item).id)
                           with *cap_item})})))
          }
          expr_fn_block(decl, body, captures) {
            expr_fn_block(fold_fn_decl(decl, fld), fld.fold_block(body),
                          @((*captures).map(|cap_item| {
                              @({id: fld.new_id((*cap_item).id)
                                 with *cap_item})})))
          }
          expr_block(blk) { expr_block(fld.fold_block(blk)) }
          expr_move(el, er) {
            expr_move(fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_copy(e) { expr_copy(fld.fold_expr(e)) }
          expr_assign(el, er) {
            expr_assign(fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_swap(el, er) {
            expr_swap(fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_assign_op(op, el, er) {
            expr_assign_op(op, fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_field(el, id, tys) {
            expr_field(fld.fold_expr(el), fld.fold_ident(id),
                       vec::map(tys, |x| fld.fold_ty(x)))
          }
          expr_index(el, er) {
            expr_index(fld.fold_expr(el), fld.fold_expr(er))
          }
          expr_path(pth) { expr_path(fld.fold_path(pth)) }
          expr_fail(e) { expr_fail(option::map(e, |x| fld.fold_expr(x))) }
          expr_break | expr_again { copy e }
          expr_ret(e) { expr_ret(option::map(e, |x| fld.fold_expr(x))) }
          expr_log(i, lv, e) { expr_log(i, fld.fold_expr(lv),
                                        fld.fold_expr(e)) }
          expr_assert(e) { expr_assert(fld.fold_expr(e)) }
          expr_check(m, e) { expr_check(m, fld.fold_expr(e)) }
          expr_if_check(cond, tr, fl) {
            expr_if_check(fld.fold_expr(cond), fld.fold_block(tr),
                          option::map(fl, |x| fld.fold_expr(x)))
          }
          expr_mac(mac) { expr_mac(fold_mac(mac)) }
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
    alt t {
      ty_nil | ty_bot | ty_infer {copy t}
      ty_box(mt) {ty_box(fold_mt(mt, fld))}
      ty_uniq(mt) {ty_uniq(fold_mt(mt, fld))}
      ty_vec(mt) {ty_vec(fold_mt(mt, fld))}
      ty_ptr(mt) {ty_ptr(fold_mt(mt, fld))}
      ty_rptr(region, mt) {ty_rptr(region, fold_mt(mt, fld))}
      ty_rec(fields) {ty_rec(vec::map(fields, |f| fold_field(f, fld)))}
      ty_fn(proto, decl) {ty_fn(proto, fold_fn_decl(decl, fld))}
      ty_tup(tys) {ty_tup(vec::map(tys, |ty| fld.fold_ty(ty)))}
      ty_path(path, id) {ty_path(fld.fold_path(path), fld.new_id(id))}
      ty_constr(ty, constrs) {ty_constr(fld.fold_ty(ty),
                                vec::map(constrs, |x| fld.fold_ty_constr(x)))}
      ty_vstore(t, vs) {ty_vstore(fld.fold_ty(t), vs)}
      ty_mac(mac) {ty_mac(fold_mac(mac))}
    }
}

fn noop_fold_constr(c: constr_, fld: ast_fold) -> constr_ {
    {path: fld.fold_path(c.path), args: /* FIXME (#2543) */ copy c.args,
     id: fld.new_id(c.id)}
}

fn noop_fold_ty_constr(c: ty_constr_, fld: ast_fold) -> ty_constr_ {
    let rslt: ty_constr_ =
        {path: fld.fold_path(c.path), args: /* FIXME (#2543) */ copy c.args,
         id: fld.new_id(c.id)};
    rslt
}
// ...nor do modules
fn noop_fold_mod(m: _mod, fld: ast_fold) -> _mod {
    ret {view_items: vec::map(m.view_items, |x| fld.fold_view_item(x)),
         items: vec::filter_map(m.items, |x| fld.fold_item(x))};
}

fn noop_fold_foreign_mod(nm: foreign_mod, fld: ast_fold) -> foreign_mod {
    ret {view_items: vec::map(nm.view_items, |x| fld.fold_view_item(x)),
         items: vec::map(nm.items, |x| fld.fold_foreign_item(x))}
}

fn noop_fold_variant(v: variant_, fld: ast_fold) -> variant_ {
    fn fold_variant_arg_(va: variant_arg, fld: ast_fold) -> variant_arg {
        ret {ty: fld.fold_ty(va.ty), id: fld.new_id(va.id)};
    }
    let fold_variant_arg = |x| fold_variant_arg_(x, fld);
    let args = vec::map(v.args, fold_variant_arg);

    let fold_attribute = |x| fold_attribute_(x, fld);
    let attrs = vec::map(v.attrs, fold_attribute);

    let de = alt v.disr_expr {
      some(e) {some(fld.fold_expr(e))}
      none {none}
    };
    ret {name: /* FIXME (#2543) */ copy v.name,
         attrs: attrs,
         args: args, id: fld.new_id(v.id),
         disr_expr: de,
         vis: v.vis};
}

fn noop_fold_ident(&&i: ident, _fld: ast_fold) -> ident {
    ret /* FIXME (#2543) */ copy i;
}

fn noop_fold_path(&&p: path, fld: ast_fold) -> path {
    ret {span: fld.new_span(p.span), global: p.global,
         idents: vec::map(p.idents, |x| fld.fold_ident(x)),
         rp: p.rp,
         types: vec::map(p.types, |x| fld.fold_ty(x))};
}

fn noop_fold_local(l: local_, fld: ast_fold) -> local_ {
    ret {is_mutbl: l.is_mutbl,
         ty: fld.fold_ty(l.ty),
         pat: fld.fold_pat(l.pat),
         init:
             alt l.init {
               option::none::<initializer> { l.init }
               option::some::<initializer>(init) {
                 option::some::<initializer>({op: init.op,
                                              expr: fld.fold_expr(init.expr)})
               }
             },
         id: fld.new_id(l.id)};
}

/* temporarily eta-expand because of a compiler bug with using `fn<T>` as a
   value */
fn noop_map_exprs(f: fn@(&&@expr) -> @expr, es: ~[@expr]) -> ~[@expr] {
    ret vec::map(es, f);
}

fn noop_id(i: node_id) -> node_id { ret i; }

fn noop_span(sp: span) -> span { ret sp; }

fn default_ast_fold() -> ast_fold_precursor {
    ret @{fold_crate: wrap(noop_fold_crate),
          fold_crate_directive: wrap(noop_fold_crate_directive),
          fold_view_item: noop_fold_view_item,
          fold_foreign_item: noop_fold_foreign_item,
          fold_item: noop_fold_item,
          fold_class_item: noop_fold_class_item,
          fold_item_underscore: noop_fold_item_underscore,
          fold_method: noop_fold_method,
          fold_block: wrap(noop_fold_block),
          fold_stmt: wrap(noop_fold_stmt),
          fold_arm: noop_fold_arm,
          fold_pat: wrap(noop_fold_pat),
          fold_decl: wrap(noop_fold_decl),
          fold_expr: wrap(noop_fold_expr),
          fold_ty: wrap(noop_fold_ty),
          fold_constr: wrap(noop_fold_constr),
          fold_ty_constr: wrap(noop_fold_ty_constr),
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

impl of ast_fold for ast_fold_precursor {
    /* naturally, a macro to write these would be nice */
    fn fold_crate(c: crate) -> crate {
        let (n, s) = self.fold_crate(c.node, c.span, self as ast_fold);
        ret {node: n, span: self.new_span(s)};
    }
    fn fold_crate_directive(&&c: @crate_directive) -> @crate_directive {
        let (n, s) = self.fold_crate_directive(c.node, c.span,
                                               self as ast_fold);
        ret @{node: n,
              span: self.new_span(s)};
    }
    fn fold_view_item(&&x: @view_item) ->
       @view_item {
        ret @{node: self.fold_view_item(x.node, self as ast_fold),
              attrs: vec::map(x.attrs, |a|
                  fold_attribute_(a, self as ast_fold)),
              vis: x.vis,
              span: self.new_span(x.span)};
    }
    fn fold_foreign_item(&&x: @foreign_item)
        -> @foreign_item {
        ret self.fold_foreign_item(x, self as ast_fold);
    }
    fn fold_item(&&i: @item) -> option<@item> {
        ret self.fold_item(i, self as ast_fold);
    }
    fn fold_class_item(&&ci: @class_member) -> @class_member {
        @{node: alt ci.node {
           instance_var(nm, t, mt, id, p) {
               instance_var(/* FIXME (#2543) */ copy nm,
                            (self as ast_fold).fold_ty(t), mt, id, p)
           }
           class_method(m) {
               class_method(self.fold_method(m, self as ast_fold))
           }
          }, span: self.new_span(ci.span)}
    }
    fn fold_item_underscore(i: item_) ->
       item_ {
        ret self.fold_item_underscore(i, self as ast_fold);
    }
    fn fold_method(&&x: @method)
        -> @method {
        ret self.fold_method(x, self as ast_fold);
    }
    fn fold_block(x: blk) -> blk {
        let (n, s) = self.fold_block(x.node, x.span, self as ast_fold);
        ret {node: n, span: self.new_span(s)};
    }
    fn fold_stmt(&&x: @stmt) -> @stmt {
        let (n, s) = self.fold_stmt(x.node, x.span, self as ast_fold);
        ret @{node: n, span: self.new_span(s)};
    }
    fn fold_arm(x: arm) -> arm {
        ret self.fold_arm(x, self as ast_fold);
    }
    fn fold_pat(&&x: @pat) -> @pat {
        let (n, s) =  self.fold_pat(x.node, x.span, self as ast_fold);
        ret @{id: self.new_id(x.id),
              node: n,
              span: self.new_span(s)};
    }
    fn fold_decl(&&x: @decl) -> @decl {
        let (n, s) = self.fold_decl(x.node, x.span, self as ast_fold);
        ret @{node: n, span: self.new_span(s)};
    }
    fn fold_expr(&&x: @expr) -> @expr {
        let (n, s) = self.fold_expr(x.node, x.span, self as ast_fold);
        ret @{id: self.new_id(x.id),
              node: n,
              span: self.new_span(s)};
    }
    fn fold_ty(&&x: @ty) -> @ty {
        let (n, s) = self.fold_ty(x.node, x.span, self as ast_fold);
        ret @{id: self.new_id(x.id), node: n, span: self.new_span(s)};
    }
    fn fold_constr(&&x: @ast::constr) ->
       @ast::constr {
        let (n, s) = self.fold_constr(x.node, x.span, self as ast_fold);
        ret @{node: n, span: self.new_span(s)};
    }
    fn fold_ty_constr(&&x: @ast::ty_constr) ->
       @ast::ty_constr {
        let (n, s) : (ty_constr_, span) =
            self.fold_ty_constr(x.node, x.span, self as ast_fold);
        ret @{node: n, span: self.new_span(s)};
    }
    fn fold_mod(x: _mod) -> _mod {
        ret self.fold_mod(x, self as ast_fold);
    }
    fn fold_foreign_mod(x: foreign_mod) ->
       foreign_mod {
        ret self.fold_foreign_mod(x, self as ast_fold);
    }
    fn fold_variant(x: variant) ->
       variant {
        let (n, s) = self.fold_variant(x.node, x.span, self as ast_fold);
        ret {node: n, span: self.new_span(s)};
    }
    fn fold_ident(&&x: ident) -> ident {
        ret self.fold_ident(x, self as ast_fold);
    }
    fn fold_path(&&x: @path) -> @path {
        @self.fold_path(*x, self as ast_fold)
    }
    fn fold_local(&&x: @local) -> @local {
        let (n, s) = self.fold_local(x.node, x.span, self as ast_fold);
        ret @{node: n, span: self.new_span(s)};
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
