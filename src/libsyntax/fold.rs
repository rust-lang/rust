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

// We may eventually want to be able to fold over type parameters, too.
pub trait ast_fold {
    fn fold_crate(&self, c: &Crate) -> Crate {
        noop_fold_crate(c, self)
    }

    fn fold_view_item(&self, vi: &view_item) -> view_item {
        return /* FIXME (#2543) */ (*vi).clone();
    }

    fn fold_foreign_item(&self, ni: @foreign_item) -> @foreign_item {
        let fold_arg = |x| fold_arg_(x, self);
        let fold_attribute = |x| fold_attribute_(x, self);

        @ast::foreign_item {
            ident: self.fold_ident(ni.ident),
            attrs: ni.attrs.map(|x| fold_attribute(*x)),
            node:
                match ni.node {
                    foreign_item_fn(ref fdec, ref generics) => {
                        foreign_item_fn(
                            ast::fn_decl {
                                inputs: fdec.inputs.map(|a|
                                    fold_arg(/*bad*/(*a).clone())),
                                output: self.fold_ty(&fdec.output),
                                cf: fdec.cf,
                            },
                            fold_generics(generics, self))
                    }
                    foreign_item_static(ref t, m) => {
                        foreign_item_static(self.fold_ty(t), m)
                    }
                },
            id: self.new_id(ni.id),
            span: self.new_span(ni.span),
            vis: ni.vis,
        }
    }

    fn fold_item(&self, i: @item) -> Option<@item> {
        noop_fold_item(i, self)
    }

    fn fold_struct_field(&self, sf: @struct_field) -> @struct_field {
        let fold_attribute = |x| fold_attribute_(x, self);

        @spanned {
            node: ast::struct_field_ {
                kind: sf.node.kind,
                id: sf.node.id,
                ty: self.fold_ty(&sf.node.ty),
                attrs: sf.node.attrs.map(|e| fold_attribute(*e))
            },
            span: self.new_span(sf.span)
        }
    }

    fn fold_item_underscore(&self, i: &item_) -> item_ {
        noop_fold_item_underscore(i, self)
    }

    fn fold_method(&self, m: @method) -> @method {
        @ast::method {
            ident: self.fold_ident(m.ident),
            attrs: /* FIXME (#2543) */ m.attrs.clone(),
            generics: fold_generics(&m.generics, self),
            explicit_self: m.explicit_self,
            purity: m.purity,
            decl: fold_fn_decl(&m.decl, self),
            body: self.fold_block(&m.body),
            id: self.new_id(m.id),
            span: self.new_span(m.span),
            self_id: self.new_id(m.self_id),
            vis: m.vis,
        }
    }

    fn fold_block(&self, b: &Block) -> Block {
        noop_fold_block(b, self)
    }

    fn fold_stmt(&self, s: &stmt) -> Option<@stmt> {
        noop_fold_stmt(s, self)
    }

    fn fold_arm(&self, a: &arm) -> arm {
        arm {
            pats: a.pats.map(|x| self.fold_pat(*x)),
            guard: a.guard.map_move(|x| self.fold_expr(x)),
            body: self.fold_block(&a.body),
        }
    }

    fn fold_pat(&self, p: @pat) -> @pat {
        let node = match p.node {
            pat_wild => pat_wild,
            pat_ident(binding_mode, ref pth, ref sub) => {
                pat_ident(
                    binding_mode,
                    self.fold_path(pth),
                    sub.map_move(|x| self.fold_pat(x))
                )
            }
            pat_lit(e) => pat_lit(self.fold_expr(e)),
            pat_enum(ref pth, ref pats) => {
                pat_enum(
                    self.fold_path(pth),
                    pats.map(|pats| pats.map(|x| self.fold_pat(*x)))
                )
            }
            pat_struct(ref pth, ref fields, etc) => {
                let pth_ = self.fold_path(pth);
                let fs = do fields.map |f| {
                    ast::field_pat {
                        ident: f.ident,
                        pat: self.fold_pat(f.pat)
                    }
                };
                pat_struct(pth_, fs, etc)
            }
            pat_tup(ref elts) => pat_tup(elts.map(|x| self.fold_pat(*x))),
            pat_box(inner) => pat_box(self.fold_pat(inner)),
            pat_uniq(inner) => pat_uniq(self.fold_pat(inner)),
            pat_region(inner) => pat_region(self.fold_pat(inner)),
            pat_range(e1, e2) => {
                pat_range(self.fold_expr(e1), self.fold_expr(e2))
            },
            pat_vec(ref before, ref slice, ref after) => {
                pat_vec(
                    before.map(|x| self.fold_pat(*x)),
                    slice.map_move(|x| self.fold_pat(x)),
                    after.map(|x| self.fold_pat(*x))
                )
            }
        };

        @pat {
            id: self.new_id(p.id),
            span: self.new_span(p.span),
            node: node,
        }
    }

    fn fold_decl(&self, d: @decl) -> Option<@decl> {
        let node = match d.node {
            decl_local(ref l) => Some(decl_local(self.fold_local(*l))),
            decl_item(it) => {
                match self.fold_item(it) {
                    Some(it_folded) => Some(decl_item(it_folded)),
                    None => None,
                }
            }
        };

        node.map_move(|node| {
            @spanned {
                node: node,
                span: d.span,
            }
        })
    }

    fn fold_expr(&self, e: @expr) -> @expr {
        noop_fold_expr(e, self)
    }

    fn fold_ty(&self, t: &Ty) -> Ty {
        let fold_mac = |x| fold_mac_(x, self);
        let node = match t.node {
            ty_nil | ty_bot | ty_infer => t.node.clone(),
            ty_box(ref mt) => ty_box(fold_mt(mt, self)),
            ty_uniq(ref mt) => ty_uniq(fold_mt(mt, self)),
            ty_vec(ref mt) => ty_vec(fold_mt(mt, self)),
            ty_ptr(ref mt) => ty_ptr(fold_mt(mt, self)),
            ty_rptr(region, ref mt) => ty_rptr(region, fold_mt(mt, self)),
            ty_closure(ref f) => {
                ty_closure(@TyClosure {
                    sigil: f.sigil,
                    purity: f.purity,
                    region: f.region,
                    onceness: f.onceness,
                    bounds: fold_opt_bounds(&f.bounds, self),
                    decl: fold_fn_decl(&f.decl, self),
                    lifetimes: f.lifetimes.clone(),
                })
            }
            ty_bare_fn(ref f) => {
                ty_bare_fn(@TyBareFn {
                    lifetimes: f.lifetimes.clone(),
                    purity: f.purity,
                    abis: f.abis,
                    decl: fold_fn_decl(&f.decl, self)
                })
            }
            ty_tup(ref tys) => ty_tup(tys.map(|ty| self.fold_ty(ty))),
            ty_path(ref path, ref bounds, id) => {
                ty_path(self.fold_path(path),
                        fold_opt_bounds(bounds, self),
                        self.new_id(id))
            }
            ty_fixed_length_vec(ref mt, e) => {
                ty_fixed_length_vec(fold_mt(mt, self), self.fold_expr(e))
            }
            ty_mac(ref mac) => ty_mac(fold_mac(mac))
        };
        Ty {
            id: self.new_id(t.id),
            span: self.new_span(t.span),
            node: node,
        }
    }

    fn fold_mod(&self, m: &_mod) -> _mod {
        noop_fold_mod(m, self)
    }

    fn fold_foreign_mod(&self, nm: &foreign_mod) -> foreign_mod {
        ast::foreign_mod {
            sort: nm.sort,
            abis: nm.abis,
            view_items: nm.view_items
                          .iter()
                          .map(|x| self.fold_view_item(x))
                          .collect(),
            items: nm.items
                     .iter()
                     .map(|x| self.fold_foreign_item(*x))
                     .collect(),
        }
    }

    fn fold_variant(&self, v: &variant) -> variant {
        let fold_variant_arg = |x| fold_variant_arg_(x, self);

        let kind;
        match v.node.kind {
            tuple_variant_kind(ref variant_args) => {
                kind = tuple_variant_kind(do variant_args.map |x| {
                    fold_variant_arg(/*bad*/ (*x).clone())
                })
            }
            struct_variant_kind(ref struct_def) => {
                kind = struct_variant_kind(@ast::struct_def {
                    fields: struct_def.fields.iter()
                        .map(|f| self.fold_struct_field(*f)).collect(),
                    ctor_id: struct_def.ctor_id.map(|c| self.new_id(*c))
                })
            }
        }

        let fold_attribute = |x| fold_attribute_(x, self);
        let attrs = v.node.attrs.map(|x| fold_attribute(*x));

        let de = match v.node.disr_expr {
          Some(e) => Some(self.fold_expr(e)),
          None => None
        };
        let node = ast::variant_ {
            name: v.node.name,
            attrs: attrs,
            kind: kind,
            id: self.new_id(v.node.id),
            disr_expr: de,
            vis: v.node.vis,
        };
        spanned {
            node: node,
            span: self.new_span(v.span),
        }
    }

    fn fold_ident(&self, i: ident) -> ident {
        i
    }

    fn fold_path(&self, p: &Path) -> Path {
        ast::Path {
            span: self.new_span(p.span),
            global: p.global,
            segments: p.segments.map(|segment| ast::PathSegment {
                identifier: self.fold_ident(segment.identifier),
                lifetime: segment.lifetime,
                types: segment.types.map(|typ| self.fold_ty(typ)),
            })
        }
    }

    fn fold_local(&self, l: @Local) -> @Local {
        @Local {
            is_mutbl: l.is_mutbl,
            ty: self.fold_ty(&l.ty),
            pat: self.fold_pat(l.pat),
            init: l.init.map_move(|e| self.fold_expr(e)),
            id: self.new_id(l.id),
            span: self.new_span(l.span),
        }
    }

    fn map_exprs(&self, f: &fn(@expr) -> @expr, es: &[@expr]) -> ~[@expr] {
        es.map(|x| f(*x))
    }

    fn new_id(&self, i: NodeId) -> NodeId {
        i
    }

    fn new_span(&self, sp: span) -> span {
        sp
    }
}

/* some little folds that probably aren't useful to have in ast_fold itself*/

//used in noop_fold_item and noop_fold_crate and noop_fold_crate_directive
fn fold_meta_item_<T:ast_fold>(mi: @MetaItem, fld: &T) -> @MetaItem {
    @spanned {
        node:
            match mi.node {
                MetaWord(id) => MetaWord(id),
                MetaList(id, ref mis) => {
                    let fold_meta_item = |x| fold_meta_item_(x, fld);
                    MetaList(
                        id,
                        mis.map(|e| fold_meta_item(*e))
                    )
                }
                MetaNameValue(id, s) => MetaNameValue(id, s)
            },
        span: fld.new_span(mi.span) }
}

//used in noop_fold_item and noop_fold_crate
fn fold_attribute_<T:ast_fold>(at: Attribute, fld: &T) -> Attribute {
    spanned {
        span: fld.new_span(at.span),
        node: ast::Attribute_ {
            style: at.node.style,
            value: fold_meta_item_(at.node.value, fld),
            is_sugared_doc: at.node.is_sugared_doc
        }
    }
}

//used in noop_fold_foreign_item and noop_fold_fn_decl
fn fold_arg_<T:ast_fold>(a: arg, fld: &T) -> arg {
    ast::arg {
        is_mutbl: a.is_mutbl,
        ty: fld.fold_ty(&a.ty),
        pat: fld.fold_pat(a.pat),
        id: fld.new_id(a.id),
    }
}

//used in noop_fold_expr, and possibly elsewhere in the future
fn fold_mac_<T:ast_fold>(m: &mac, fld: &T) -> mac {
    spanned {
        node: match m.node {
            mac_invoc_tt(ref p,ref tts) =>
            mac_invoc_tt(fld.fold_path(p),
                         fold_tts(*tts,fld))
        },
        span: fld.new_span(m.span)
    }
}

fn fold_tts<T:ast_fold>(tts: &[token_tree], fld: &T) -> ~[token_tree] {
    do tts.map |tt| {
        match *tt {
            tt_tok(span, ref tok) =>
            tt_tok(span,maybe_fold_ident(tok,fld)),
            tt_delim(ref tts) =>
            tt_delim(@mut fold_tts(**tts, fld)),
            tt_seq(span, ref pattern, ref sep, is_optional) =>
            tt_seq(span,
                   @mut fold_tts(**pattern, fld),
                   sep.map(|tok|maybe_fold_ident(tok,fld)),
                   is_optional),
            tt_nonterminal(sp,ref ident) =>
            tt_nonterminal(sp,fld.fold_ident(*ident))
        }
    }
}

// apply ident folder if it's an ident, otherwise leave it alone
fn maybe_fold_ident<T:ast_fold>(t: &token::Token, fld: &T) -> token::Token {
    match *t {
        token::IDENT(id,followed_by_colons) =>
        token::IDENT(fld.fold_ident(id),followed_by_colons),
        _ => (*t).clone()
    }
}

pub fn fold_fn_decl<T:ast_fold>(decl: &ast::fn_decl, fld: &T)
                                -> ast::fn_decl {
    ast::fn_decl {
        inputs: decl.inputs.map(|x| fold_arg_((*x).clone(), fld)), // bad copy
        output: fld.fold_ty(&decl.output),
        cf: decl.cf,
    }
}

fn fold_ty_param_bound<T:ast_fold>(tpb: &TyParamBound, fld: &T)
                                   -> TyParamBound {
    match *tpb {
        TraitTyParamBound(ref ty) => TraitTyParamBound(fold_trait_ref(ty, fld)),
        RegionTyParamBound => RegionTyParamBound
    }
}

pub fn fold_ty_param<T:ast_fold>(tp: TyParam, fld: &T) -> TyParam {
    TyParam {
        ident: tp.ident,
        id: fld.new_id(tp.id),
        bounds: tp.bounds.map(|x| fold_ty_param_bound(x, fld)),
    }
}

pub fn fold_ty_params<T:ast_fold>(tps: &OptVec<TyParam>, fld: &T)
                                  -> OptVec<TyParam> {
    let tps = (*tps).clone(); // bad
    tps.map_move(|tp| fold_ty_param(tp, fld))
}

pub fn fold_lifetime<T:ast_fold>(l: &Lifetime, fld: &T) -> Lifetime {
    Lifetime {
        id: fld.new_id(l.id),
        span: fld.new_span(l.span),
        ident: l.ident
    }
}

pub fn fold_lifetimes<T:ast_fold>(lts: &OptVec<Lifetime>, fld: &T)
                                  -> OptVec<Lifetime> {
    lts.map(|l| fold_lifetime(l, fld))
}

pub fn fold_generics<T:ast_fold>(generics: &Generics, fld: &T) -> Generics {
    Generics {ty_params: fold_ty_params(&generics.ty_params, fld),
              lifetimes: fold_lifetimes(&generics.lifetimes, fld)}
}

fn fold_struct_def<T:ast_fold>(struct_def: @ast::struct_def, fld: &T)
                               -> @ast::struct_def {
    @ast::struct_def {
        fields: struct_def.fields.map(|f| fold_struct_field(*f, fld)),
        ctor_id: struct_def.ctor_id.map(|cid| fld.new_id(*cid)),
    }
}

fn fold_trait_ref<T:ast_fold>(p: &trait_ref, fld: &T) -> trait_ref {
    ast::trait_ref {
        path: fld.fold_path(&p.path),
        ref_id: fld.new_id(p.ref_id),
    }
}

fn fold_struct_field<T:ast_fold>(f: @struct_field, fld: &T) -> @struct_field {
    @spanned {
        node: ast::struct_field_ {
            kind: f.node.kind,
            id: fld.new_id(f.node.id),
            ty: fld.fold_ty(&f.node.ty),
            attrs: f.node.attrs.clone(),    // FIXME (#2543)
        },
        span: fld.new_span(f.span),
    }
}

fn fold_field_<T:ast_fold>(field: Field, folder: &T) -> Field {
    ast::Field {
        ident: folder.fold_ident(field.ident),
        expr: folder.fold_expr(field.expr),
        span: folder.new_span(field.span),
    }
}

fn fold_mt<T:ast_fold>(mt: &mt, folder: &T) -> mt {
    mt {
        ty: ~folder.fold_ty(mt.ty),
        mutbl: mt.mutbl,
    }
}

fn fold_field<T:ast_fold>(f: TypeField, folder: &T) -> TypeField {
    ast::TypeField {
        ident: folder.fold_ident(f.ident),
        mt: fold_mt(&f.mt, folder),
        span: folder.new_span(f.span),
    }
}

fn fold_opt_bounds<T:ast_fold>(b: &Option<OptVec<TyParamBound>>, folder: &T)
                               -> Option<OptVec<TyParamBound>> {
    do b.map |bounds| {
        do bounds.map |bound| {
            fold_ty_param_bound(bound, folder)
        }
    }
}

fn fold_variant_arg_<T:ast_fold>(va: variant_arg, folder: &T) -> variant_arg {
    ast::variant_arg {
        ty: folder.fold_ty(&va.ty),
        id: folder.new_id(va.id)
    }
}

pub fn noop_fold_block<T:ast_fold>(b: &Block, folder: &T) -> Block {
    let view_items = b.view_items.map(|x| folder.fold_view_item(x));
    let mut stmts = ~[];
    for stmt in b.stmts.iter() {
        match folder.fold_stmt(*stmt) {
            None => {}
            Some(stmt) => stmts.push(stmt)
        }
    }
    ast::Block {
        view_items: view_items,
        stmts: stmts,
        expr: b.expr.map(|x| folder.fold_expr(*x)),
        id: folder.new_id(b.id),
        rules: b.rules,
        span: folder.new_span(b.span),
    }
}

pub fn noop_fold_item_underscore<T:ast_fold>(i: &item_, folder: &T) -> item_ {
    match *i {
        item_static(ref t, m, e) => {
            item_static(folder.fold_ty(t), m, folder.fold_expr(e))
        }
        item_fn(ref decl, purity, abi, ref generics, ref body) => {
            item_fn(
                fold_fn_decl(decl, folder),
                purity,
                abi,
                fold_generics(generics, folder),
                folder.fold_block(body)
            )
        }
        item_mod(ref m) => item_mod(folder.fold_mod(m)),
        item_foreign_mod(ref nm) => {
            item_foreign_mod(folder.fold_foreign_mod(nm))
        }
        item_ty(ref t, ref generics) => {
            item_ty(folder.fold_ty(t),
                    fold_generics(generics, folder))
        }
        item_enum(ref enum_definition, ref generics) => {
            item_enum(
                ast::enum_def {
                    variants: do enum_definition.variants.map |x| {
                        folder.fold_variant(x)
                    },
                },
                fold_generics(generics, folder))
        }
        item_struct(ref struct_def, ref generics) => {
            let struct_def = fold_struct_def(*struct_def, folder);
            item_struct(struct_def,
                        /* FIXME (#2543) */ (*generics).clone())
        }
        item_impl(ref generics, ref ifce, ref ty, ref methods) => {
            item_impl(fold_generics(generics, folder),
                      ifce.map(|p| fold_trait_ref(p, folder)),
                      folder.fold_ty(ty),
                      methods.map(|x| folder.fold_method(*x))
            )
        }
        item_trait(ref generics, ref traits, ref methods) => {
            let methods = do methods.map |method| {
                match *method {
                    required(*) => (*method).clone(),
                    provided(method) => provided(folder.fold_method(method))
                }
            };
            item_trait(fold_generics(generics, folder),
                       traits.map(|p| fold_trait_ref(p, folder)),
                       methods)
        }
        item_mac(ref m) => {
            // FIXME #2888: we might actually want to do something here.
            // ... okay, we're doing something. It would probably be nicer
            // to add something to the ast_fold trait, but I'll defer
            // that work.
            item_mac(fold_mac_(m, folder))
        }
    }
}

pub fn noop_fold_mod<T:ast_fold>(m: &_mod, folder: &T) -> _mod {
    ast::_mod {
        view_items: m.view_items
                     .iter()
                     .map(|x| folder.fold_view_item(x)).collect(),
        items: m.items.iter().filter_map(|x| folder.fold_item(*x)).collect(),
    }
}

pub fn noop_fold_crate<T:ast_fold>(c: &Crate, folder: &T) -> Crate {
    let fold_meta_item = |x| fold_meta_item_(x, folder);
    let fold_attribute = |x| fold_attribute_(x, folder);

    Crate {
        module: folder.fold_mod(&c.module),
        attrs: c.attrs.map(|x| fold_attribute(*x)),
        config: c.config.map(|x| fold_meta_item(*x)),
        span: folder.new_span(c.span),
    }
}

pub fn noop_fold_item<T:ast_fold>(i: @ast::item, folder: &T)
                                  -> Option<@ast::item> {
    let fold_attribute = |x| fold_attribute_(x, folder);

    Some(@ast::item {
        ident: folder.fold_ident(i.ident),
        attrs: i.attrs.map(|e| fold_attribute(*e)),
        id: folder.new_id(i.id),
        node: folder.fold_item_underscore(&i.node),
        vis: i.vis,
        span: folder.new_span(i.span)
    })
}

pub fn noop_fold_expr<T:ast_fold>(e: @ast::expr, folder: &T) -> @ast::expr {
    let fold_field = |x| fold_field_(x, folder);

    let fold_mac = |x| fold_mac_(x, folder);

    let node = match e.node {
        expr_vstore(e, v) => {
            expr_vstore(folder.fold_expr(e), v)
        }
        expr_vec(ref exprs, mutt) => {
            expr_vec(folder.map_exprs(|x| folder.fold_expr(x), *exprs), mutt)
        }
        expr_repeat(expr, count, mutt) => {
            expr_repeat(folder.fold_expr(expr), folder.fold_expr(count), mutt)
        }
        expr_tup(ref elts) => expr_tup(elts.map(|x| folder.fold_expr(*x))),
        expr_call(f, ref args, blk) => {
            expr_call(
                folder.fold_expr(f),
                folder.map_exprs(|x| folder.fold_expr(x), *args),
                blk
            )
        }
        expr_method_call(callee_id, f, i, ref tps, ref args, blk) => {
            expr_method_call(
                folder.new_id(callee_id),
                folder.fold_expr(f),
                folder.fold_ident(i),
                tps.map(|x| folder.fold_ty(x)),
                folder.map_exprs(|x| folder.fold_expr(x), *args),
                blk
            )
        }
        expr_binary(callee_id, binop, lhs, rhs) => {
            expr_binary(
                folder.new_id(callee_id),
                binop,
                folder.fold_expr(lhs),
                folder.fold_expr(rhs)
            )
        }
        expr_unary(callee_id, binop, ohs) => {
            expr_unary(folder.new_id(callee_id), binop, folder.fold_expr(ohs))
        }
        expr_do_body(f) => expr_do_body(folder.fold_expr(f)),
        expr_lit(_) => e.node.clone(),
        expr_cast(expr, ref ty) => {
            expr_cast(folder.fold_expr(expr), (*ty).clone())
        }
        expr_addr_of(m, ohs) => expr_addr_of(m, folder.fold_expr(ohs)),
        expr_if(cond, ref tr, fl) => {
            expr_if(
                folder.fold_expr(cond),
                folder.fold_block(tr),
                fl.map_move(|x| folder.fold_expr(x))
            )
        }
        expr_while(cond, ref body) => {
            expr_while(folder.fold_expr(cond), folder.fold_block(body))
        }
        expr_for_loop(pat, iter, ref body) => {
            expr_for_loop(folder.fold_pat(pat),
                          folder.fold_expr(iter),
                          folder.fold_block(body))
        }
        expr_loop(ref body, opt_ident) => {
            expr_loop(
                folder.fold_block(body),
                opt_ident.map_move(|x| folder.fold_ident(x))
            )
        }
        expr_match(expr, ref arms) => {
            expr_match(
                folder.fold_expr(expr),
                arms.map(|x| folder.fold_arm(x))
            )
        }
        expr_fn_block(ref decl, ref body) => {
            expr_fn_block(
                fold_fn_decl(decl, folder),
                folder.fold_block(body)
            )
        }
        expr_block(ref blk) => expr_block(folder.fold_block(blk)),
        expr_assign(el, er) => {
            expr_assign(folder.fold_expr(el), folder.fold_expr(er))
        }
        expr_assign_op(callee_id, op, el, er) => {
            expr_assign_op(
                folder.new_id(callee_id),
                op,
                folder.fold_expr(el),
                folder.fold_expr(er)
            )
        }
        expr_field(el, id, ref tys) => {
            expr_field(
                folder.fold_expr(el), folder.fold_ident(id),
                tys.map(|x| folder.fold_ty(x))
            )
        }
        expr_index(callee_id, el, er) => {
            expr_index(
                folder.new_id(callee_id),
                folder.fold_expr(el),
                folder.fold_expr(er)
            )
        }
        expr_path(ref pth) => expr_path(folder.fold_path(pth)),
        expr_self => expr_self,
        expr_break(ref opt_ident) => {
            expr_break(opt_ident.map_move(|x| folder.fold_ident(x)))
        }
        expr_again(ref opt_ident) => {
            expr_again(opt_ident.map_move(|x| folder.fold_ident(x)))
        }
        expr_ret(ref e) => {
            expr_ret(e.map_move(|x| folder.fold_expr(x)))
        }
        expr_log(lv, e) => {
            expr_log(
                folder.fold_expr(lv),
                folder.fold_expr(e)
            )
        }
        expr_inline_asm(ref a) => {
            expr_inline_asm(inline_asm {
                inputs: a.inputs.map(|&(c, input)| (c, folder.fold_expr(input))),
                outputs: a.outputs.map(|&(c, out)| (c, folder.fold_expr(out))),
                .. (*a).clone()
            })
        }
        expr_mac(ref mac) => expr_mac(fold_mac(mac)),
        expr_struct(ref path, ref fields, maybe_expr) => {
            expr_struct(
                folder.fold_path(path),
                fields.map(|x| fold_field(*x)),
                maybe_expr.map_move(|x| folder.fold_expr(x))
            )
        },
        expr_paren(ex) => expr_paren(folder.fold_expr(ex))
    };

    @expr {
        id: folder.new_id(e.id),
        node: node,
        span: folder.new_span(e.span),
    }
}

pub fn noop_fold_stmt<T:ast_fold>(s: &stmt, folder: &T) -> Option<@stmt> {
    let fold_mac = |x| fold_mac_(x, folder);
    let node = match s.node {
        stmt_decl(d, nid) => {
            match folder.fold_decl(d) {
                Some(d) => Some(stmt_decl(d, folder.new_id(nid))),
                None => None,
            }
        }
        stmt_expr(e, nid) => {
            Some(stmt_expr(folder.fold_expr(e), folder.new_id(nid)))
        }
        stmt_semi(e, nid) => {
            Some(stmt_semi(folder.fold_expr(e), folder.new_id(nid)))
        }
        stmt_mac(ref mac, semi) => Some(stmt_mac(fold_mac(mac), semi))
    };

    node.map_move(|node| @spanned {
        node: node,
        span: folder.new_span(s.span),
    })
}

#[cfg(test)]
mod test {
    use ast;
    use util::parser_testing::{string_to_crate, matches_codepattern};
    use parse::token;
    use print::pprust;
    use super::*;

    struct IdentFolder {
        f: @fn(ast::ident)->ast::ident,
    }

    impl ast_fold for IdentFolder {
        fn fold_ident(@self, i: ident) -> ident {
            (self.f)(i)
        }
    }

    // taken from expand
    // given a function from idents to idents, produce
    // an ast_fold that applies that function:
    pub fn fun_to_ident_folder(f: @fn(ast::ident)->ast::ident) -> @ast_fold {
        @IdentFolder {
            f: f,
        } as @ast_fold
    }

    // this version doesn't care about getting comments or docstrings in.
    fn fake_print_crate(s: @pprust::ps, crate: &ast::Crate) {
        pprust::print_mod(s, &crate.module, crate.attrs);
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

