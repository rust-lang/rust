// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::*;
use ast;
use codemap::{span, spanned};
use parse::token;
use opt_vec::OptVec;

pub trait ast_fold {
    fn fold_crate(@self, &crate) -> crate;
    fn fold_view_item(@self, &view_item) -> view_item;
    fn fold_foreign_item(@self, @foreign_item) -> @foreign_item;
    fn fold_item(@self, @item) -> Option<@item>;
    fn fold_struct_field(@self, @struct_field) -> @struct_field;
    fn fold_item_underscore(@self, &item_) -> item_;
    fn fold_method(@self, @method) -> @method;
    fn fold_block(@self, &blk) -> blk;
    fn fold_stmt(@self, &stmt) -> Option<@stmt>;
    fn fold_arm(@self, &arm) -> arm;
    fn fold_pat(@self, @pat) -> @pat;
    fn fold_decl(@self, @decl) -> Option<@decl>;
    fn fold_expr(@self, @expr) -> @expr;
    fn fold_ty(@self, &Ty) -> Ty;
    fn fold_mod(@self, &_mod) -> _mod;
    fn fold_foreign_mod(@self, &foreign_mod) -> foreign_mod;
    fn fold_variant(@self, &variant) -> variant;
    fn fold_ident(@self, ident) -> ident;
    fn fold_path(@self, &Path) -> Path;
    fn fold_local(@self, @local) -> @local;
    fn map_exprs(@self, @fn(@expr) -> @expr, &[@expr]) -> ~[@expr];
    fn new_id(@self, node_id) -> node_id;
    fn new_span(@self, span) -> span;
}

// We may eventually want to be able to fold over type parameters, too

pub struct AstFoldFns {
    //unlike the others, item_ is non-trivial
    fold_crate: @fn(&crate_, span, @ast_fold) -> (crate_, span),
    fold_view_item: @fn(&view_item_, @ast_fold) -> view_item_,
    fold_foreign_item: @fn(@foreign_item, @ast_fold) -> @foreign_item,
    fold_item: @fn(@item, @ast_fold) -> Option<@item>,
    fold_struct_field: @fn(@struct_field, @ast_fold) -> @struct_field,
    fold_item_underscore: @fn(&item_, @ast_fold) -> item_,
    fold_method: @fn(@method, @ast_fold) -> @method,
    fold_block: @fn(&blk, @ast_fold) -> blk,
    fold_stmt: @fn(&stmt_, span, @ast_fold) -> (Option<stmt_>, span),
    fold_arm: @fn(&arm, @ast_fold) -> arm,
    fold_pat: @fn(&pat_, span, @ast_fold) -> (pat_, span),
    fold_decl: @fn(&decl_, span, @ast_fold) -> (Option<decl_>, span),
    fold_expr: @fn(&expr_, span, @ast_fold) -> (expr_, span),
    fold_ty: @fn(&ty_, span, @ast_fold) -> (ty_, span),
    fold_mod: @fn(&_mod, @ast_fold) -> _mod,
    fold_foreign_mod: @fn(&foreign_mod, @ast_fold) -> foreign_mod,
    fold_variant: @fn(&variant_, span, @ast_fold) -> (variant_, span),
    fold_ident: @fn(ident, @ast_fold) -> ident,
    fold_path: @fn(&Path, @ast_fold) -> Path,
    fold_local: @fn(&local_, span, @ast_fold) -> (local_, span),
    map_exprs: @fn(@fn(@expr) -> @expr, &[@expr]) -> ~[@expr],
    new_id: @fn(node_id) -> node_id,
    new_span: @fn(span) -> span
}

pub type ast_fold_fns = @AstFoldFns;

/* some little folds that probably aren't useful to have in ast_fold itself*/

//used in noop_fold_item and noop_fold_crate and noop_fold_crate_directive
fn fold_meta_item_(mi: @meta_item, fld: @ast_fold) -> @meta_item {
    @spanned {
        node:
            match mi.node {
                meta_word(id) => meta_word(id),
                meta_list(id, ref mis) => {
                    let fold_meta_item = |x| fold_meta_item_(x, fld);
                    meta_list(
                        id,
                        mis.map(|e| fold_meta_item(*e))
                    )
                }
                meta_name_value(id, s) => {
                    meta_name_value(id, s)
                }
            },
        span: fld.new_span(mi.span) }
}
//used in noop_fold_item and noop_fold_crate
fn fold_attribute_(at: attribute, fld: @ast_fold) -> attribute {
    spanned {
        node: ast::attribute_ {
            style: at.node.style,
            value: fold_meta_item_(at.node.value, fld),
            is_sugared_doc: at.node.is_sugared_doc,
        },
        span: fld.new_span(at.span),
    }
}
//used in noop_fold_foreign_item and noop_fold_fn_decl
fn fold_arg_(a: arg, fld: @ast_fold) -> arg {
    ast::arg {
        is_mutbl: a.is_mutbl,
        ty: fld.fold_ty(&a.ty),
        pat: fld.fold_pat(a.pat),
        id: fld.new_id(a.id),
    }
}

//used in noop_fold_expr, and possibly elsewhere in the future
fn fold_mac_(m: &mac, fld: @ast_fold) -> mac {
    spanned {
        node: match m.node {
            mac_invoc_tt(ref p,ref tts) =>
            mac_invoc_tt(fld.fold_path(p),
                         fold_tts(*tts,fld))
        },
        span: fld.new_span(m.span)
    }
}

fn fold_tts(tts : &[token_tree], fld: @ast_fold) -> ~[token_tree] {
    do tts.map |tt| {
        match *tt {
            tt_tok(span, ref tok) =>
            tt_tok(span,maybe_fold_ident(tok,fld)),
            tt_delim(ref tts) =>
            tt_delim(fold_tts(*tts,fld)),
            tt_seq(span, ref pattern, ref sep, is_optional) =>
            tt_seq(span,
                   fold_tts(*pattern,fld),
                   sep.map(|tok|maybe_fold_ident(tok,fld)),
                   is_optional),
            tt_nonterminal(sp,ref ident) =>
            tt_nonterminal(sp,fld.fold_ident(*ident))
        }
    }
}

// apply ident folder if it's an ident, otherwise leave it alone
fn maybe_fold_ident(t : &token::Token, fld: @ast_fold) -> token::Token {
    match *t {
        token::IDENT(id,followed_by_colons) =>
        token::IDENT(fld.fold_ident(id),followed_by_colons),
        _ => copy *t
    }
}

pub fn fold_fn_decl(decl: &ast::fn_decl, fld: @ast_fold) -> ast::fn_decl {
    ast::fn_decl {
        inputs: decl.inputs.map(|x| fold_arg_(/*bad*/ copy *x, fld)),
        output: fld.fold_ty(&decl.output),
        cf: decl.cf,
    }
}

fn fold_ty_param_bound(tpb: &TyParamBound, fld: @ast_fold) -> TyParamBound {
    match *tpb {
        TraitTyParamBound(ref ty) => TraitTyParamBound(fold_trait_ref(ty, fld)),
        RegionTyParamBound => RegionTyParamBound
    }
}

pub fn fold_ty_param(tp: TyParam,
                     fld: @ast_fold) -> TyParam {
    TyParam {ident: tp.ident,
             id: fld.new_id(tp.id),
             bounds: tp.bounds.map(|x| fold_ty_param_bound(x, fld))}
}

pub fn fold_ty_params(tps: &OptVec<TyParam>,
                      fld: @ast_fold) -> OptVec<TyParam> {
    let tps = /*bad*/ copy *tps;
    tps.map_consume(|tp| fold_ty_param(tp, fld))
}

pub fn fold_lifetime(l: &Lifetime,
                     fld: @ast_fold) -> Lifetime {
    Lifetime {id: fld.new_id(l.id),
              span: fld.new_span(l.span),
              ident: l.ident}
}

pub fn fold_lifetimes(lts: &OptVec<Lifetime>,
                      fld: @ast_fold) -> OptVec<Lifetime> {
    lts.map(|l| fold_lifetime(l, fld))
}

pub fn fold_generics(generics: &Generics, fld: @ast_fold) -> Generics {
    Generics {ty_params: fold_ty_params(&generics.ty_params, fld),
              lifetimes: fold_lifetimes(&generics.lifetimes, fld)}
}

pub fn noop_fold_crate(c: &crate_, fld: @ast_fold) -> crate_ {
    let fold_meta_item = |x| fold_meta_item_(x, fld);
    let fold_attribute = |x| fold_attribute_(x, fld);

    crate_ {
        module: fld.fold_mod(&c.module),
        attrs: c.attrs.map(|x| fold_attribute(*x)),
        config: c.config.map(|x| fold_meta_item(*x)),
    }
}

fn noop_fold_view_item(vi: &view_item_, _fld: @ast_fold) -> view_item_ {
    return /* FIXME (#2543) */ copy *vi;
}


fn noop_fold_foreign_item(ni: @foreign_item, fld: @ast_fold)
    -> @foreign_item {
    let fold_arg = |x| fold_arg_(x, fld);
    let fold_attribute = |x| fold_attribute_(x, fld);

    @ast::foreign_item {
        ident: fld.fold_ident(ni.ident),
        attrs: ni.attrs.map(|x| fold_attribute(*x)),
        node:
            match ni.node {
                foreign_item_fn(ref fdec, purity, ref generics) => {
                    foreign_item_fn(
                        ast::fn_decl {
                            inputs: fdec.inputs.map(|a| fold_arg(/*bad*/copy *a)),
                            output: fld.fold_ty(&fdec.output),
                            cf: fdec.cf,
                        },
                        purity,
                        fold_generics(generics, fld))
                }
                foreign_item_static(ref t, m) => {
                    foreign_item_static(fld.fold_ty(t), m)
                }
            },
        id: fld.new_id(ni.id),
        span: fld.new_span(ni.span),
        vis: ni.vis,
    }
}

pub fn noop_fold_item(i: @item, fld: @ast_fold) -> Option<@item> {
    let fold_attribute = |x| fold_attribute_(x, fld);

    Some(@ast::item { ident: fld.fold_ident(i.ident),
                      attrs: i.attrs.map(|e| fold_attribute(*e)),
                      id: fld.new_id(i.id),
                      node: fld.fold_item_underscore(&i.node),
                      vis: i.vis,
                      span: fld.new_span(i.span) })
}

fn noop_fold_struct_field(sf: @struct_field, fld: @ast_fold)
                       -> @struct_field {
    let fold_attribute = |x| fold_attribute_(x, fld);

    @spanned {
        node: ast::struct_field_ {
            kind: sf.node.kind,
            id: sf.node.id,
            ty: fld.fold_ty(&sf.node.ty),
            attrs: sf.node.attrs.map(|e| fold_attribute(*e))
        },
        span: sf.span
    }
}

pub fn noop_fold_item_underscore(i: &item_, fld: @ast_fold) -> item_ {
    match *i {
        item_static(ref t, m, e) => item_static(fld.fold_ty(t), m, fld.fold_expr(e)),
        item_fn(ref decl, purity, abi, ref generics, ref body) => {
            item_fn(
                fold_fn_decl(decl, fld),
                purity,
                abi,
                fold_generics(generics, fld),
                fld.fold_block(body)
            )
        }
        item_mod(ref m) => item_mod(fld.fold_mod(m)),
        item_foreign_mod(ref nm) => {
            item_foreign_mod(fld.fold_foreign_mod(nm))
        }
        item_ty(ref t, ref generics) => {
            item_ty(fld.fold_ty(t), fold_generics(generics, fld))
        }
        item_enum(ref enum_definition, ref generics) => {
            item_enum(
                ast::enum_def {
                    variants: do enum_definition.variants.map |x| {
                        fld.fold_variant(x)
                    },
                },
                fold_generics(generics, fld))
        }
        item_struct(ref struct_def, ref generics) => {
            let struct_def = fold_struct_def(*struct_def, fld);
            item_struct(struct_def, /* FIXME (#2543) */ copy *generics)
        }
        item_impl(ref generics, ref ifce, ref ty, ref methods) => {
            item_impl(
                fold_generics(generics, fld),
                ifce.map(|p| fold_trait_ref(p, fld)),
                fld.fold_ty(ty),
                methods.map(|x| fld.fold_method(*x))
            )
        }
        item_trait(ref generics, ref traits, ref methods) => {
            let methods = do methods.map |method| {
                match *method {
                    required(*) => copy *method,
                    provided(method) => provided(fld.fold_method(method))
                }
            };
            item_trait(
                fold_generics(generics, fld),
                traits.map(|p| fold_trait_ref(p, fld)),
                methods
            )
        }
        item_mac(ref m) => {
            // FIXME #2888: we might actually want to do something here.
            // ... okay, we're doing something. It would probably be nicer
            // to add something to the ast_fold trait, but I'll defer
            // that work.
            item_mac(fold_mac_(m,fld))
        }
    }
}

fn fold_struct_def(struct_def: @ast::struct_def, fld: @ast_fold)
                -> @ast::struct_def {
    @ast::struct_def {
        fields: struct_def.fields.map(|f| fold_struct_field(*f, fld)),
        ctor_id: struct_def.ctor_id.map(|cid| fld.new_id(*cid)),
    }
}

fn fold_trait_ref(p: &trait_ref, fld: @ast_fold) -> trait_ref {
    ast::trait_ref {
        path: fld.fold_path(&p.path),
        ref_id: fld.new_id(p.ref_id),
    }
}

fn fold_struct_field(f: @struct_field, fld: @ast_fold) -> @struct_field {
    @spanned {
        node: ast::struct_field_ {
            kind: f.node.kind,
            id: fld.new_id(f.node.id),
            ty: fld.fold_ty(&f.node.ty),
            attrs: /* FIXME (#2543) */ copy f.node.attrs,
        },
        span: fld.new_span(f.span),
    }
}

fn noop_fold_method(m: @method, fld: @ast_fold) -> @method {
    @ast::method {
        ident: fld.fold_ident(m.ident),
        attrs: /* FIXME (#2543) */ copy m.attrs,
        generics: fold_generics(&m.generics, fld),
        explicit_self: m.explicit_self,
        purity: m.purity,
        decl: fold_fn_decl(&m.decl, fld),
        body: fld.fold_block(&m.body),
        id: fld.new_id(m.id),
        span: fld.new_span(m.span),
        self_id: fld.new_id(m.self_id),
        vis: m.vis,
    }
}


pub fn noop_fold_block(b: &blk, fld: @ast_fold) -> blk {
    let view_items = b.view_items.map(|x| fld.fold_view_item(x));
    let mut stmts = ~[];
    for b.stmts.iter().advance |stmt| {
        match fld.fold_stmt(*stmt) {
            None => {}
            Some(stmt) => stmts.push(stmt)
        }
    }
    ast::blk {
        view_items: view_items,
        stmts: stmts,
        expr: b.expr.map(|x| fld.fold_expr(*x)),
        id: fld.new_id(b.id),
        rules: b.rules,
        span: b.span,
    }
}

fn noop_fold_stmt(s: &stmt_, fld: @ast_fold) -> Option<stmt_> {
    let fold_mac = |x| fold_mac_(x, fld);
    match *s {
        stmt_decl(d, nid) => {
            match fld.fold_decl(d) {
                Some(d) => Some(stmt_decl(d, fld.new_id(nid))),
                None => None,
            }
        }
        stmt_expr(e, nid) => {
            Some(stmt_expr(fld.fold_expr(e), fld.new_id(nid)))
        }
        stmt_semi(e, nid) => {
            Some(stmt_semi(fld.fold_expr(e), fld.new_id(nid)))
        }
        stmt_mac(ref mac, semi) => Some(stmt_mac(fold_mac(mac), semi))
    }
}

fn noop_fold_arm(a: &arm, fld: @ast_fold) -> arm {
    arm {
        pats: a.pats.map(|x| fld.fold_pat(*x)),
        guard: a.guard.map(|x| fld.fold_expr(*x)),
        body: fld.fold_block(&a.body),
    }
}

pub fn noop_fold_pat(p: &pat_, fld: @ast_fold) -> pat_ {
    match *p {
        pat_wild => pat_wild,
        pat_ident(binding_mode, ref pth, ref sub) => {
            pat_ident(
                binding_mode,
                fld.fold_path(pth),
                sub.map(|x| fld.fold_pat(*x))
            )
        }
        pat_lit(e) => pat_lit(fld.fold_expr(e)),
        pat_enum(ref pth, ref pats) => {
            pat_enum(
                fld.fold_path(pth),
                pats.map(|pats| pats.map(|x| fld.fold_pat(*x)))
            )
        }
        pat_struct(ref pth, ref fields, etc) => {
            let pth_ = fld.fold_path(pth);
            let fs = do fields.map |f| {
                ast::field_pat {
                    ident: f.ident,
                    pat: fld.fold_pat(f.pat)
                }
            };
            pat_struct(pth_, fs, etc)
        }
        pat_tup(ref elts) => pat_tup(elts.map(|x| fld.fold_pat(*x))),
        pat_box(inner) => pat_box(fld.fold_pat(inner)),
        pat_uniq(inner) => pat_uniq(fld.fold_pat(inner)),
        pat_region(inner) => pat_region(fld.fold_pat(inner)),
        pat_range(e1, e2) => {
            pat_range(fld.fold_expr(e1), fld.fold_expr(e2))
        },
        pat_vec(ref before, ref slice, ref after) => {
            pat_vec(
                before.map(|x| fld.fold_pat(*x)),
                slice.map(|x| fld.fold_pat(*x)),
                after.map(|x| fld.fold_pat(*x))
            )
        }
    }
}

fn noop_fold_decl(d: &decl_, fld: @ast_fold) -> Option<decl_> {
    match *d {
        decl_local(ref l) => Some(decl_local(fld.fold_local(*l))),
        decl_item(it) => {
            match fld.fold_item(it) {
                Some(it_folded) => Some(decl_item(it_folded)),
                None => None,
            }
        }
    }
}

pub fn wrap<T>(f: @fn(&T, @ast_fold) -> T)
            -> @fn(&T, span, @ast_fold) -> (T, span) {
    let result: @fn(&T, span, @ast_fold) -> (T, span) = |x, s, fld| {
        (f(x, fld), s)
    };
    result
}

pub fn noop_fold_expr(e: &expr_, fld: @ast_fold) -> expr_ {
    fn fold_field_(field: field, fld: @ast_fold) -> field {
        spanned {
            node: ast::field_ {
                ident: fld.fold_ident(field.node.ident),
                expr: fld.fold_expr(field.node.expr),
            },
            span: fld.new_span(field.span),
        }
    }
    let fold_field = |x| fold_field_(x, fld);

    let fold_mac = |x| fold_mac_(x, fld);

    match *e {
        expr_vstore(e, v) => {
            expr_vstore(fld.fold_expr(e), v)
        }
        expr_vec(ref exprs, mutt) => {
            expr_vec(fld.map_exprs(|x| fld.fold_expr(x), *exprs), mutt)
        }
        expr_repeat(expr, count, mutt) => {
            expr_repeat(fld.fold_expr(expr), fld.fold_expr(count), mutt)
        }
        expr_tup(ref elts) => expr_tup(elts.map(|x| fld.fold_expr(*x))),
        expr_call(f, ref args, blk) => {
            expr_call(
                fld.fold_expr(f),
                fld.map_exprs(|x| fld.fold_expr(x), *args),
                blk
            )
        }
        expr_method_call(callee_id, f, i, ref tps, ref args, blk) => {
            expr_method_call(
                fld.new_id(callee_id),
                fld.fold_expr(f),
                fld.fold_ident(i),
                tps.map(|x| fld.fold_ty(x)),
                fld.map_exprs(|x| fld.fold_expr(x), *args),
                blk
            )
        }
        expr_binary(callee_id, binop, lhs, rhs) => {
            expr_binary(
                fld.new_id(callee_id),
                binop,
                fld.fold_expr(lhs),
                fld.fold_expr(rhs)
            )
        }
        expr_unary(callee_id, binop, ohs) => {
            expr_unary(
                fld.new_id(callee_id),
                binop,
                fld.fold_expr(ohs)
            )
        }
        expr_loop_body(f) => expr_loop_body(fld.fold_expr(f)),
        expr_do_body(f) => expr_do_body(fld.fold_expr(f)),
        expr_lit(_) => copy *e,
        expr_cast(expr, ref ty) => expr_cast(fld.fold_expr(expr), copy *ty),
        expr_addr_of(m, ohs) => expr_addr_of(m, fld.fold_expr(ohs)),
        expr_if(cond, ref tr, fl) => {
            expr_if(
                fld.fold_expr(cond),
                fld.fold_block(tr),
                fl.map(|x| fld.fold_expr(*x))
            )
        }
        expr_while(cond, ref body) => {
            expr_while(fld.fold_expr(cond), fld.fold_block(body))
        }
        expr_loop(ref body, opt_ident) => {
            expr_loop(
                fld.fold_block(body),
                opt_ident.map(|x| fld.fold_ident(*x))
            )
        }
        expr_match(expr, ref arms) => {
            expr_match(
                fld.fold_expr(expr),
                arms.map(|x| fld.fold_arm(x))
            )
        }
        expr_fn_block(ref decl, ref body) => {
            expr_fn_block(
                fold_fn_decl(decl, fld),
                fld.fold_block(body)
            )
        }
        expr_block(ref blk) => expr_block(fld.fold_block(blk)),
        expr_copy(e) => expr_copy(fld.fold_expr(e)),
        expr_assign(el, er) => {
            expr_assign(fld.fold_expr(el), fld.fold_expr(er))
        }
        expr_assign_op(callee_id, op, el, er) => {
            expr_assign_op(
                fld.new_id(callee_id),
                op,
                fld.fold_expr(el),
                fld.fold_expr(er)
            )
        }
        expr_field(el, id, ref tys) => {
            expr_field(
                fld.fold_expr(el), fld.fold_ident(id),
                tys.map(|x| fld.fold_ty(x))
            )
        }
        expr_index(callee_id, el, er) => {
            expr_index(
                fld.new_id(callee_id),
                fld.fold_expr(el),
                fld.fold_expr(er)
            )
        }
        expr_path(ref pth) => expr_path(fld.fold_path(pth)),
        expr_self => expr_self,
        expr_break(ref opt_ident) => {
            expr_break(opt_ident.map(|x| fld.fold_ident(*x)))
        }
        expr_again(ref opt_ident) => {
            expr_again(opt_ident.map(|x| fld.fold_ident(*x)))
        }
        expr_ret(ref e) => {
            expr_ret(e.map(|x| fld.fold_expr(*x)))
        }
        expr_log(lv, e) => {
            expr_log(
                fld.fold_expr(lv),
                fld.fold_expr(e)
            )
        }
        expr_inline_asm(ref a) => {
            expr_inline_asm(inline_asm {
                inputs: a.inputs.map(|&(c, in)| (c, fld.fold_expr(in))),
                outputs: a.outputs.map(|&(c, out)| (c, fld.fold_expr(out))),
                .. copy *a
            })
        }
        expr_mac(ref mac) => expr_mac(fold_mac(mac)),
        expr_struct(ref path, ref fields, maybe_expr) => {
            expr_struct(
                fld.fold_path(path),
                fields.map(|x| fold_field(*x)),
                maybe_expr.map(|x| fld.fold_expr(*x))
            )
        },
        expr_paren(ex) => expr_paren(fld.fold_expr(ex))
    }
}

pub fn noop_fold_ty(t: &ty_, fld: @ast_fold) -> ty_ {
    let fold_mac = |x| fold_mac_(x, fld);
    fn fold_mt(mt: &mt, fld: @ast_fold) -> mt {
        mt {
            ty: ~fld.fold_ty(mt.ty),
            mutbl: mt.mutbl,
        }
    }
    fn fold_field(f: ty_field, fld: @ast_fold) -> ty_field {
        spanned {
            node: ast::ty_field_ {
                ident: fld.fold_ident(f.node.ident),
                mt: fold_mt(&f.node.mt, fld),
            },
            span: fld.new_span(f.span),
        }
    }
    fn fold_opt_bounds(b: &Option<OptVec<TyParamBound>>, fld: @ast_fold)
                        -> Option<OptVec<TyParamBound>> {
        do b.map |bounds| {
            do bounds.map |bound| { fold_ty_param_bound(bound, fld) }
        }
    }
    match *t {
        ty_nil | ty_bot | ty_infer => copy *t,
        ty_box(ref mt) => ty_box(fold_mt(mt, fld)),
        ty_uniq(ref mt) => ty_uniq(fold_mt(mt, fld)),
        ty_vec(ref mt) => ty_vec(fold_mt(mt, fld)),
        ty_ptr(ref mt) => ty_ptr(fold_mt(mt, fld)),
        ty_rptr(region, ref mt) => ty_rptr(region, fold_mt(mt, fld)),
        ty_closure(ref f) => {
            ty_closure(@TyClosure {
                sigil: f.sigil,
                purity: f.purity,
                region: f.region,
                onceness: f.onceness,
                bounds: fold_opt_bounds(&f.bounds, fld),
                decl: fold_fn_decl(&f.decl, fld),
                lifetimes: copy f.lifetimes,
            })
        }
        ty_bare_fn(ref f) => {
            ty_bare_fn(@TyBareFn {
                lifetimes: copy f.lifetimes,
                purity: f.purity,
                abis: f.abis,
                decl: fold_fn_decl(&f.decl, fld)
            })
        }
        ty_tup(ref tys) => ty_tup(tys.map(|ty| fld.fold_ty(ty))),
        ty_path(ref path, ref bounds, id) =>
            ty_path(fld.fold_path(path), fold_opt_bounds(bounds, fld), fld.new_id(id)),
        ty_fixed_length_vec(ref mt, e) => {
            ty_fixed_length_vec(
                fold_mt(mt, fld),
                fld.fold_expr(e)
            )
        }
        ty_mac(ref mac) => ty_mac(fold_mac(mac))
    }
}

// ...nor do modules
pub fn noop_fold_mod(m: &_mod, fld: @ast_fold) -> _mod {
    ast::_mod {
        view_items: m.view_items.iter().transform(|x| fld.fold_view_item(x)).collect(),
        items: m.items.iter().filter_map(|x| fld.fold_item(*x)).collect(),
    }
}

fn noop_fold_foreign_mod(nm: &foreign_mod, fld: @ast_fold) -> foreign_mod {
    ast::foreign_mod {
        sort: nm.sort,
        abis: nm.abis,
        view_items: nm.view_items.iter().transform(|x| fld.fold_view_item(x)).collect(),
        items: nm.items.iter().transform(|x| fld.fold_foreign_item(*x)).collect(),
    }
}

fn noop_fold_variant(v: &variant_, fld: @ast_fold) -> variant_ {
    fn fold_variant_arg_(va: variant_arg, fld: @ast_fold) -> variant_arg {
        ast::variant_arg { ty: fld.fold_ty(&va.ty), id: fld.new_id(va.id) }
    }
    let fold_variant_arg = |x| fold_variant_arg_(x, fld);

    let kind;
    match v.kind {
        tuple_variant_kind(ref variant_args) => {
            kind = tuple_variant_kind(do variant_args.map |x| {
                fold_variant_arg(/*bad*/ copy *x)
            })
        }
        struct_variant_kind(struct_def) => {
            kind = struct_variant_kind(@ast::struct_def {
                fields: struct_def.fields.iter()
                    .transform(|f| fld.fold_struct_field(*f)).collect(),
                ctor_id: struct_def.ctor_id.map(|c| fld.new_id(*c))
            })
        }
    }

    let fold_attribute = |x| fold_attribute_(x, fld);
    let attrs = v.attrs.map(|x| fold_attribute(*x));

    let de = match v.disr_expr {
      Some(e) => Some(fld.fold_expr(e)),
      None => None
    };
    ast::variant_ {
        name: v.name,
        attrs: attrs,
        kind: kind,
        id: fld.new_id(v.id),
        disr_expr: de,
        vis: v.vis,
    }
}

fn noop_fold_ident(i: ident, _fld: @ast_fold) -> ident {
    i
}

fn noop_fold_path(p: &Path, fld: @ast_fold) -> Path {
    ast::Path {
        span: fld.new_span(p.span),
        global: p.global,
        idents: p.idents.map(|x| fld.fold_ident(*x)),
        rp: p.rp,
        types: p.types.map(|x| fld.fold_ty(x)),
    }
}

fn noop_fold_local(l: &local_, fld: @ast_fold) -> local_ {
    local_ {
        is_mutbl: l.is_mutbl,
        ty: fld.fold_ty(&l.ty),
        pat: fld.fold_pat(l.pat),
        init: l.init.map(|e| fld.fold_expr(*e)),
        id: fld.new_id(l.id),
    }
}

/* temporarily eta-expand because of a compiler bug with using `fn<T>` as a
   value */
fn noop_map_exprs(f: @fn(@expr) -> @expr, es: &[@expr]) -> ~[@expr] {
    es.map(|x| f(*x))
}

fn noop_id(i: node_id) -> node_id { return i; }

fn noop_span(sp: span) -> span { return sp; }

pub fn default_ast_fold() -> ast_fold_fns {
    @AstFoldFns {
        fold_crate: wrap(noop_fold_crate),
        fold_view_item: noop_fold_view_item,
        fold_foreign_item: noop_fold_foreign_item,
        fold_item: noop_fold_item,
        fold_struct_field: noop_fold_struct_field,
        fold_item_underscore: noop_fold_item_underscore,
        fold_method: noop_fold_method,
        fold_block: noop_fold_block,
        fold_stmt: |x, s, fld| (noop_fold_stmt(x, fld), s),
        fold_arm: noop_fold_arm,
        fold_pat: wrap(noop_fold_pat),
        fold_decl: |x, s, fld| (noop_fold_decl(x, fld), s),
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
        new_span: noop_span,
    }
}

impl ast_fold for AstFoldFns {
    /* naturally, a macro to write these would be nice */
    fn fold_crate(@self, c: &crate) -> crate {
        let (n, s) = (self.fold_crate)(&c.node, c.span, self as @ast_fold);
        spanned { node: n, span: (self.new_span)(s) }
    }
    fn fold_view_item(@self, x: &view_item) -> view_item {
        ast::view_item {
            node: (self.fold_view_item)(&x.node, self as @ast_fold),
            attrs: x.attrs.iter().transform(|a| fold_attribute_(*a, self as @ast_fold)).collect(),
            vis: x.vis,
            span: (self.new_span)(x.span),
        }
    }
    fn fold_foreign_item(@self, x: @foreign_item) -> @foreign_item {
        (self.fold_foreign_item)(x, self as @ast_fold)
    }
    fn fold_item(@self, i: @item) -> Option<@item> {
        (self.fold_item)(i, self as @ast_fold)
    }
    fn fold_struct_field(@self, sf: @struct_field) -> @struct_field {
        @spanned {
            node: ast::struct_field_ {
                kind: sf.node.kind,
                id: sf.node.id,
                ty: self.fold_ty(&sf.node.ty),
                attrs: copy sf.node.attrs,
            },
            span: (self.new_span)(sf.span),
        }
    }
    fn fold_item_underscore(@self, i: &item_) -> item_ {
        (self.fold_item_underscore)(i, self as @ast_fold)
    }
    fn fold_method(@self, x: @method) -> @method {
        (self.fold_method)(x, self as @ast_fold)
    }
    fn fold_block(@self, x: &blk) -> blk {
        (self.fold_block)(x, self as @ast_fold)
    }
    fn fold_stmt(@self, x: &stmt) -> Option<@stmt> {
        let (n_opt, s) = (self.fold_stmt)(&x.node, x.span, self as @ast_fold);
        match n_opt {
            Some(n) => Some(@spanned { node: n, span: (self.new_span)(s) }),
            None => None,
        }
    }
    fn fold_arm(@self, x: &arm) -> arm {
        (self.fold_arm)(x, self as @ast_fold)
    }
    fn fold_pat(@self, x: @pat) -> @pat {
        let (n, s) =  (self.fold_pat)(&x.node, x.span, self as @ast_fold);
        @pat {
            id: (self.new_id)(x.id),
            node: n,
            span: (self.new_span)(s),
        }
    }
    fn fold_decl(@self, x: @decl) -> Option<@decl> {
        let (n_opt, s) = (self.fold_decl)(&x.node, x.span, self as @ast_fold);
        match n_opt {
            Some(n) => Some(@spanned { node: n, span: (self.new_span)(s) }),
            None => None,
        }
    }
    fn fold_expr(@self, x: @expr) -> @expr {
        let (n, s) = (self.fold_expr)(&x.node, x.span, self as @ast_fold);
        @expr {
            id: (self.new_id)(x.id),
            node: n,
            span: (self.new_span)(s),
        }
    }
    fn fold_ty(@self, x: &Ty) -> Ty {
        let (n, s) = (self.fold_ty)(&x.node, x.span, self as @ast_fold);
        Ty {
            id: (self.new_id)(x.id),
            node: n,
            span: (self.new_span)(s),
        }
    }
    fn fold_mod(@self, x: &_mod) -> _mod {
        (self.fold_mod)(x, self as @ast_fold)
    }
    fn fold_foreign_mod(@self, x: &foreign_mod) -> foreign_mod {
        (self.fold_foreign_mod)(x, self as @ast_fold)
    }
    fn fold_variant(@self, x: &variant) -> variant {
        let (n, s) = (self.fold_variant)(&x.node, x.span, self as @ast_fold);
        spanned { node: n, span: (self.new_span)(s) }
    }
    fn fold_ident(@self, x: ident) -> ident {
        (self.fold_ident)(x, self as @ast_fold)
    }
    fn fold_path(@self, x: &Path) -> Path {
        (self.fold_path)(x, self as @ast_fold)
    }
    fn fold_local(@self, x: @local) -> @local {
        let (n, s) = (self.fold_local)(&x.node, x.span, self as @ast_fold);
        @spanned { node: n, span: (self.new_span)(s) }
    }
    fn map_exprs(@self,
                 f: @fn(@expr) -> @expr,
                 e: &[@expr])
              -> ~[@expr] {
        (self.map_exprs)(f, e)
    }
    fn new_id(@self, node_id: ast::node_id) -> node_id {
        (self.new_id)(node_id)
    }
    fn new_span(@self, span: span) -> span {
        (self.new_span)(span)
    }
}

pub trait AstFoldExtensions {
    fn fold_attributes(&self, attrs: ~[attribute]) -> ~[attribute];
}

impl AstFoldExtensions for @ast_fold {
    fn fold_attributes(&self, attrs: ~[attribute]) -> ~[attribute] {
        attrs.map(|x| fold_attribute_(*x, *self))
    }
}

pub fn make_fold(afp: ast_fold_fns) -> @ast_fold {
    afp as @ast_fold
}

#[cfg(test)]
mod test {
    use ast;
    use util::parser_testing::{string_to_crate, matches_codepattern};
    use parse::token;
    use print::pprust;
    use super::*;

    // taken from expand
    // given a function from idents to idents, produce
    // an ast_fold that applies that function:
    pub fn fun_to_ident_folder(f: @fn(ast::ident)->ast::ident) -> @ast_fold{
        let afp = default_ast_fold();
        let f_pre = @AstFoldFns{
            fold_ident : |id, _| f(id),
            .. *afp
        };
        make_fold(f_pre)
    }

    // this version doesn't care about getting comments or docstrings in.
    fn fake_print_crate(s: @pprust::ps, crate: &ast::crate) {
        pprust::print_mod(s, &crate.node.module, crate.node.attrs);
    }

    // change every identifier to "zz"
    pub fn to_zz() -> @fn(ast::ident)->ast::ident {
        let zz_id = token::str_to_ident("zz");
        |_id| {zz_id}
    }

    // maybe add to expand.rs...
    macro_rules! assert_pred (
        ($pred:expr, $predname:expr, $a:expr , $b:expr) => (
            {
                let pred_val = $pred;
                let a_val = $a;
                let b_val = $b;
                if !(pred_val(a_val,b_val)) {
                    fail!("expected args satisfying %s, got %? and %?",
                          $predname, a_val, b_val);
                }
            }
        )
    )

    // make sure idents get transformed everywhere
    #[test] fn ident_transformation () {
        let zz_fold = fun_to_ident_folder(to_zz());
        let ast = string_to_crate(@"#[a] mod b {fn c (d : e, f : g) {h!(i,j,k);l;m}}");
        assert_pred!(matches_codepattern,
                     "matches_codepattern",
                     pprust::to_str(&zz_fold.fold_crate(ast),fake_print_crate,
                                    token::get_ident_interner()),
                     ~"#[a]mod zz{fn zz(zz:zz,zz:zz){zz!(zz,zz,zz);zz;zz}}");
    }

    // even inside macro defs....
    #[test] fn ident_transformation_in_defs () {
        let zz_fold = fun_to_ident_folder(to_zz());
        let ast = string_to_crate(@"macro_rules! a {(b $c:expr $(d $e:token)f+
=> (g $(d $d $e)+))} ");
        assert_pred!(matches_codepattern,
                     "matches_codepattern",
                     pprust::to_str(&zz_fold.fold_crate(ast),fake_print_crate,
                                    token::get_ident_interner()),
                     ~"zz!zz((zz$zz:zz$(zz $zz:zz)zz+=>(zz$(zz$zz$zz)+)))");
    }
}
