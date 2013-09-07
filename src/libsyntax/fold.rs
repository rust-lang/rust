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
use codemap::{Span, Spanned};
use parse::token;
use opt_vec::OptVec;

// this file defines an ast_fold trait for objects that can perform
// a "fold" on Rust ASTs. It also contains a structure that implements
// that trait, and a "default_fold" whose fields contain closures
// that perform "default traversals", visiting all of the sub-elements
// and re-assembling the result. The "fun_to_ident_folder" in the
// test module provides a simple example of creating a very simple
// fold that only looks at identifiers.

pub trait ast_fold {
    fn fold_crate(@self, &Crate) -> Crate;
    fn fold_view_item(@self, &view_item) -> view_item;
    fn fold_foreign_item(@self, @foreign_item) -> @foreign_item;
    fn fold_item(@self, @item) -> Option<@item>;
    fn fold_struct_field(@self, @struct_field) -> @struct_field;
    fn fold_item_underscore(@self, &item_) -> item_;
    fn fold_type_method(@self, m: &TypeMethod) -> TypeMethod;
    fn fold_method(@self, @method) -> @method;
    fn fold_block(@self, &Block) -> Block;
    fn fold_stmt(@self, &Stmt) -> Option<@Stmt>;
    fn fold_arm(@self, &Arm) -> Arm;
    fn fold_pat(@self, @Pat) -> @Pat;
    fn fold_decl(@self, @Decl) -> Option<@Decl>;
    fn fold_expr(@self, @Expr) -> @Expr;
    fn fold_ty(@self, &Ty) -> Ty;
    fn fold_mod(@self, &_mod) -> _mod;
    fn fold_foreign_mod(@self, &foreign_mod) -> foreign_mod;
    fn fold_variant(@self, &variant) -> variant;
    fn fold_ident(@self, Ident) -> Ident;
    fn fold_path(@self, &Path) -> Path;
    fn fold_local(@self, @Local) -> @Local;
    fn fold_mac(@self, &mac) -> mac;
    fn map_exprs(@self, @fn(@Expr) -> @Expr, &[@Expr]) -> ~[@Expr];
    fn new_id(@self, NodeId) -> NodeId;
    fn new_span(@self, Span) -> Span;

    // New style, using default methods:

    fn fold_variant_arg(@self, va: &variant_arg) -> variant_arg {
        variant_arg {
            ty: self.fold_ty(&va.ty),
            id: self.new_id(va.id)
        }
    }

    fn fold_spanned<T>(@self, s: &Spanned<T>, f: &fn(&T) -> T) -> Spanned<T> {
        Spanned {
            node: f(&s.node),
            span: self.new_span(s.span)
        }
    }

    fn fold_view_path(@self, vp: &view_path) -> view_path {
        self.fold_spanned(vp, |v| self.fold_view_path_(v))
    }

    fn fold_view_paths(@self, vps: &[@view_path]) -> ~[@view_path] {
        vps.map(|vp| @self.fold_view_path(*vp))
    }

    fn fold_view_path_(@self, vp: &view_path_) -> view_path_ {
        match *vp {
            view_path_simple(ident, ref path, node_id) => {
                view_path_simple(self.fold_ident(ident),
                                 self.fold_path(path),
                                 self.new_id(node_id))
            }
            view_path_glob(ref path, node_id) => {
                view_path_glob(self.fold_path(path),
                               self.new_id(node_id))
            }
            view_path_list(ref path, ref idents, node_id) => {
                view_path_list(self.fold_path(path),
                               self.fold_path_list_idents(*idents),
                               self.new_id(node_id))
            }
        }
    }

    fn fold_path_list_idents(@self, idents: &[path_list_ident]) -> ~[path_list_ident] {
        idents.map(|i| self.fold_path_list_ident(i))
    }

    fn fold_path_list_ident(@self, ident: &path_list_ident) -> path_list_ident {
        self.fold_spanned(ident, |i| self.fold_path_list_ident_(i))
    }

    fn fold_path_list_ident_(@self, ident: &path_list_ident_) -> path_list_ident_ {
        path_list_ident_ {
            name: self.fold_ident(ident.name),
            id: self.new_id(ident.id)
        }
    }

    fn fold_arg(@self, a: &arg) -> arg {
        arg {
            is_mutbl: a.is_mutbl,
            ty: self.fold_ty(&a.ty),
            pat: self.fold_pat(a.pat),
            id: self.new_id(a.id),
        }
    }

    fn fold_trait_ref(@self, p: &trait_ref) -> trait_ref {
        trait_ref {
            path: self.fold_path(&p.path),
            ref_id: self.new_id(p.ref_id),
        }
    }

    fn fold_ty_param_bound(@self, tpb: &TyParamBound) -> TyParamBound {
        match *tpb {
            TraitTyParamBound(ref ty) => {
                TraitTyParamBound(self.fold_trait_ref(ty))
            }
            RegionTyParamBound => {
                RegionTyParamBound
            }
        }
    }

    fn fold_ty_param(@self, tp: &TyParam) -> TyParam {
        TyParam {
            ident: self.fold_ident(tp.ident),
            id: self.new_id(tp.id),
            bounds: tp.bounds.map(|x| self.fold_ty_param_bound(x))
        }
    }

    fn fold_ty_params(@self, tps: &OptVec<TyParam>) -> OptVec<TyParam> {
        tps.map(|tp| self.fold_ty_param(tp))
    }

    fn fold_lifetime(@self, l: &Lifetime) -> Lifetime {
        Lifetime {
            id: self.new_id(l.id),
            span: self.new_span(l.span),
            ident: l.ident, // Folding this ident causes hygiene errors - ndm
        }
    }

    fn fold_lifetimes(@self, lts: &OptVec<Lifetime>) -> OptVec<Lifetime> {
        lts.map(|l| self.fold_lifetime(l))
    }


    fn fold_meta_item(@self, mi: &MetaItem) -> @MetaItem {
        @self.fold_spanned(mi, |n| match *n {
                MetaWord(id) => {
                    MetaWord(id)
                }
                MetaList(id, ref mis) => {
                    MetaList(id, self.fold_meta_items(*mis))
                }
                MetaNameValue(id, s) => {
                    MetaNameValue(id, s)
                }
            })
    }

    fn fold_meta_items(@self, mis: &[@MetaItem]) -> ~[@MetaItem] {
        mis.map(|&mi| self.fold_meta_item(mi))
    }

    fn fold_attribute(@self, at: &Attribute) -> Attribute {
        Spanned {
            span: self.new_span(at.span),
            node: Attribute_ {
                style: at.node.style,
                value: self.fold_meta_item(at.node.value),
                is_sugared_doc: at.node.is_sugared_doc
            }
        }
    }

    fn fold_attributes(@self, attrs: &[Attribute]) -> ~[Attribute] {
        attrs.map(|x| self.fold_attribute(x))
    }
}

// We may eventually want to be able to fold over type parameters, too

pub struct AstFoldFns {
    //unlike the others, item_ is non-trivial
    fold_crate: @fn(&Crate, @ast_fold) -> Crate,
    fold_view_item: @fn(&view_item_, @ast_fold) -> view_item_,
    fold_foreign_item: @fn(@foreign_item, @ast_fold) -> @foreign_item,
    fold_item: @fn(@item, @ast_fold) -> Option<@item>,
    fold_struct_field: @fn(@struct_field, @ast_fold) -> @struct_field,
    fold_item_underscore: @fn(&item_, @ast_fold) -> item_,
    fold_type_method: @fn(&TypeMethod, @ast_fold) -> TypeMethod,
    fold_method: @fn(@method, @ast_fold) -> @method,
    fold_block: @fn(&Block, @ast_fold) -> Block,
    fold_stmt: @fn(&Stmt_, Span, @ast_fold) -> (Option<Stmt_>, Span),
    fold_arm: @fn(&Arm, @ast_fold) -> Arm,
    fold_pat: @fn(&Pat_, Span, @ast_fold) -> (Pat_, Span),
    fold_decl: @fn(&Decl_, Span, @ast_fold) -> (Option<Decl_>, Span),
    fold_expr: @fn(&Expr_, Span, @ast_fold) -> (Expr_, Span),
    fold_ty: @fn(&ty_, Span, @ast_fold) -> (ty_, Span),
    fold_mod: @fn(&_mod, @ast_fold) -> _mod,
    fold_foreign_mod: @fn(&foreign_mod, @ast_fold) -> foreign_mod,
    fold_variant: @fn(&variant_, Span, @ast_fold) -> (variant_, Span),
    fold_ident: @fn(Ident, @ast_fold) -> Ident,
    fold_path: @fn(&Path, @ast_fold) -> Path,
    fold_local: @fn(@Local, @ast_fold) -> @Local,
    fold_mac: @fn(&mac_, Span, @ast_fold) -> (mac_, Span),
    map_exprs: @fn(@fn(@Expr) -> @Expr, &[@Expr]) -> ~[@Expr],
    new_id: @fn(NodeId) -> NodeId,
    new_span: @fn(Span) -> Span
}

pub type ast_fold_fns = @AstFoldFns;

/* some little folds that probably aren't useful to have in ast_fold itself*/

pub fn fold_tts(tts : &[token_tree], fld: @ast_fold) -> ~[token_tree] {
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
fn maybe_fold_ident(t : &token::Token, f: @ast_fold) -> token::Token {
    match *t {
        token::IDENT(id,followed_by_colons) =>
        token::IDENT(f.fold_ident(id),followed_by_colons),
        _ => (*t).clone()
    }
}

pub fn fold_fn_decl(decl: &ast::fn_decl, fld: @ast_fold) -> ast::fn_decl {
    ast::fn_decl {
        inputs: decl.inputs.map(|x| fld.fold_arg(x)),
        output: fld.fold_ty(&decl.output),
        cf: decl.cf,
    }
}

pub fn fold_generics(generics: &Generics, fld: @ast_fold) -> Generics {
    Generics {ty_params: fld.fold_ty_params(&generics.ty_params),
              lifetimes: fld.fold_lifetimes(&generics.lifetimes)}
}

pub fn noop_fold_crate(c: &Crate, fld: @ast_fold) -> Crate {
    Crate {
        module: fld.fold_mod(&c.module),
        attrs: fld.fold_attributes(c.attrs),
        config: fld.fold_meta_items(c.config),
        span: fld.new_span(c.span),
    }
}

fn noop_fold_view_item(vi: &view_item_, fld: @ast_fold) -> view_item_ {
    match *vi {
        view_item_extern_mod(ident, name, ref meta_items, node_id) => {
            view_item_extern_mod(ident,
                                 name,
                                 fld.fold_meta_items(*meta_items),
                                 fld.new_id(node_id))
        }
        view_item_use(ref view_paths) => {
            view_item_use(fld.fold_view_paths(*view_paths))
        }
    }
}

fn noop_fold_foreign_item(ni: @foreign_item, fld: @ast_fold)
    -> @foreign_item {
    @ast::foreign_item {
        ident: fld.fold_ident(ni.ident),
        attrs: fld.fold_attributes(ni.attrs),
        node:
            match ni.node {
                foreign_item_fn(ref fdec, ref generics) => {
                    foreign_item_fn(
                        ast::fn_decl {
                            inputs: fdec.inputs.map(|a| fld.fold_arg(a)),
                            output: fld.fold_ty(&fdec.output),
                            cf: fdec.cf,
                        },
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
    Some(@ast::item { ident: fld.fold_ident(i.ident),
                      attrs: fld.fold_attributes(i.attrs),
                      id: fld.new_id(i.id),
                      node: fld.fold_item_underscore(&i.node),
                      vis: i.vis,
                      span: fld.new_span(i.span) })
}

fn noop_fold_struct_field(sf: @struct_field, fld: @ast_fold)
                       -> @struct_field {
    @Spanned {
        node: ast::struct_field_ {
            kind: sf.node.kind,
            id: fld.new_id(sf.node.id),
            ty: fld.fold_ty(&sf.node.ty),
            attrs: fld.fold_attributes(sf.node.attrs),
        },
        span: sf.span
    }
}

pub fn noop_fold_type_method(m: &TypeMethod, fld: @ast_fold) -> TypeMethod {
    TypeMethod {
        ident: fld.fold_ident(m.ident),
        attrs: fld.fold_attributes(m.attrs),
        purity: m.purity,
        decl: fold_fn_decl(&m.decl, fld),
        generics: fold_generics(&m.generics, fld),
        explicit_self: m.explicit_self,
        id: fld.new_id(m.id),
        span: fld.new_span(m.span),
    }
}

pub fn noop_fold_item_underscore(i: &item_, fld: @ast_fold) -> item_ {
    match *i {
        item_static(ref t, m, e) => {
            item_static(fld.fold_ty(t), m, fld.fold_expr(e))
        }
        item_fn(ref decl, purity, abi, ref generics, ref body) => {
            item_fn(
                fold_fn_decl(decl, fld),
                purity,
                abi,
                fold_generics(generics, fld),
                fld.fold_block(body)
            )
        }
        item_mod(ref m) => {
            item_mod(fld.fold_mod(m))
        }
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
            item_struct(struct_def, fold_generics(generics, fld))
        }
        item_impl(ref generics, ref ifce, ref ty, ref methods) => {
            item_impl(
                fold_generics(generics, fld),
                ifce.map(|p| fld.fold_trait_ref(p)),
                fld.fold_ty(ty),
                methods.map(|x| fld.fold_method(*x))
            )
        }
        item_trait(ref generics, ref traits, ref methods) => {
            let methods = do methods.map |method| {
                match *method {
                    required(ref m) => required(fld.fold_type_method(m)),
                    provided(method) => provided(fld.fold_method(method))
                }
            };
            item_trait(
                fold_generics(generics, fld),
                traits.map(|p| fld.fold_trait_ref(p)),
                methods
            )
        }
        item_mac(ref m) => {
            item_mac(fld.fold_mac(m))
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

fn fold_struct_field(f: @struct_field, fld: @ast_fold) -> @struct_field {
    @Spanned {
        node: ast::struct_field_ {
            kind: f.node.kind,
            id: fld.new_id(f.node.id),
            ty: fld.fold_ty(&f.node.ty),
            attrs: fld.fold_attributes(f.node.attrs),
        },
        span: fld.new_span(f.span),
    }
}

fn noop_fold_method(m: @method, fld: @ast_fold) -> @method {
    @ast::method {
        ident: fld.fold_ident(m.ident),
        attrs: fld.fold_attributes(m.attrs),
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


pub fn noop_fold_block(b: &Block, fld: @ast_fold) -> Block {
    let view_items = b.view_items.map(|x| fld.fold_view_item(x));
    let mut stmts = ~[];
    for stmt in b.stmts.iter() {
        match fld.fold_stmt(*stmt) {
            None => {}
            Some(stmt) => stmts.push(stmt)
        }
    }
    ast::Block {
        view_items: view_items,
        stmts: stmts,
        expr: b.expr.map(|x| fld.fold_expr(*x)),
        id: fld.new_id(b.id),
        rules: b.rules,
        span: b.span,
    }
}

fn noop_fold_stmt(s: &Stmt_, fld: @ast_fold) -> Option<Stmt_> {
    match *s {
        StmtDecl(d, nid) => {
            match fld.fold_decl(d) {
                Some(d) => Some(StmtDecl(d, fld.new_id(nid))),
                None => None,
            }
        }
        StmtExpr(e, nid) => {
            Some(StmtExpr(fld.fold_expr(e), fld.new_id(nid)))
        }
        StmtSemi(e, nid) => {
            Some(StmtSemi(fld.fold_expr(e), fld.new_id(nid)))
        }
        StmtMac(ref mac, semi) => Some(StmtMac(fld.fold_mac(mac), semi))
    }
}

fn noop_fold_arm(a: &Arm, fld: @ast_fold) -> Arm {
    Arm {
        pats: a.pats.map(|x| fld.fold_pat(*x)),
        guard: a.guard.map_move(|x| fld.fold_expr(x)),
        body: fld.fold_block(&a.body),
    }
}

pub fn noop_fold_pat(p: &Pat_, fld: @ast_fold) -> Pat_ {
    match *p {
        PatWild => PatWild,
        PatIdent(binding_mode, ref pth, ref sub) => {
            PatIdent(
                binding_mode,
                fld.fold_path(pth),
                sub.map_move(|x| fld.fold_pat(x))
            )
        }
        PatLit(e) => PatLit(fld.fold_expr(e)),
        PatEnum(ref pth, ref pats) => {
            PatEnum(
                fld.fold_path(pth),
                pats.map(|pats| pats.map(|x| fld.fold_pat(*x)))
            )
        }
        PatStruct(ref pth, ref fields, etc) => {
            let pth_ = fld.fold_path(pth);
            let fs = do fields.map |f| {
                ast::FieldPat {
                    ident: f.ident,
                    pat: fld.fold_pat(f.pat)
                }
            };
            PatStruct(pth_, fs, etc)
        }
        PatTup(ref elts) => PatTup(elts.map(|x| fld.fold_pat(*x))),
        PatBox(inner) => PatBox(fld.fold_pat(inner)),
        PatUniq(inner) => PatUniq(fld.fold_pat(inner)),
        PatRegion(inner) => PatRegion(fld.fold_pat(inner)),
        PatRange(e1, e2) => {
            PatRange(fld.fold_expr(e1), fld.fold_expr(e2))
        },
        PatVec(ref before, ref slice, ref after) => {
            PatVec(
                before.map(|x| fld.fold_pat(*x)),
                slice.map_move(|x| fld.fold_pat(x)),
                after.map(|x| fld.fold_pat(*x))
            )
        }
    }
}

fn noop_fold_decl(d: &Decl_, fld: @ast_fold) -> Option<Decl_> {
    match *d {
        DeclLocal(ref l) => Some(DeclLocal(fld.fold_local(*l))),
        DeclItem(it) => {
            match fld.fold_item(it) {
                Some(it_folded) => Some(DeclItem(it_folded)),
                None => None,
            }
        }
    }
}

// lift a function in ast-thingy X fold -> ast-thingy to a function
// in (ast-thingy X span X fold) -> (ast-thingy X span). Basically,
// carries the span around.
// It seems strange to me that the call to new_fold doesn't happen
// here but instead in the impl down below.... probably just an
// accident?
pub fn wrap<T>(f: @fn(&T, @ast_fold) -> T)
            -> @fn(&T, Span, @ast_fold) -> (T, Span) {
    let result: @fn(&T, Span, @ast_fold) -> (T, Span) = |x, s, fld| {
        (f(x, fld), s)
    };
    result
}

pub fn noop_fold_expr(e: &Expr_, fld: @ast_fold) -> Expr_ {
    fn fold_field_(field: Field, fld: @ast_fold) -> Field {
        ast::Field {
            ident: fld.fold_ident(field.ident),
            expr: fld.fold_expr(field.expr),
            span: fld.new_span(field.span),
        }
    }
    let fold_field = |x| fold_field_(x, fld);

    match *e {
        ExprVstore(e, v) => {
            ExprVstore(fld.fold_expr(e), v)
        }
        ExprVec(ref exprs, mutt) => {
            ExprVec(fld.map_exprs(|x| fld.fold_expr(x), *exprs), mutt)
        }
        ExprRepeat(expr, count, mutt) => {
            ExprRepeat(fld.fold_expr(expr), fld.fold_expr(count), mutt)
        }
        ExprTup(ref elts) => ExprTup(elts.map(|x| fld.fold_expr(*x))),
        ExprCall(f, ref args, blk) => {
            ExprCall(
                fld.fold_expr(f),
                fld.map_exprs(|x| fld.fold_expr(x), *args),
                blk
            )
        }
        ExprMethodCall(callee_id, f, i, ref tps, ref args, blk) => {
            ExprMethodCall(
                fld.new_id(callee_id),
                fld.fold_expr(f),
                fld.fold_ident(i),
                tps.map(|x| fld.fold_ty(x)),
                fld.map_exprs(|x| fld.fold_expr(x), *args),
                blk
            )
        }
        ExprBinary(callee_id, binop, lhs, rhs) => {
            ExprBinary(
                fld.new_id(callee_id),
                binop,
                fld.fold_expr(lhs),
                fld.fold_expr(rhs)
            )
        }
        ExprUnary(callee_id, binop, ohs) => {
            ExprUnary(
                fld.new_id(callee_id),
                binop,
                fld.fold_expr(ohs)
            )
        }
        ExprDoBody(f) => ExprDoBody(fld.fold_expr(f)),
        ExprLit(_) => (*e).clone(),
        ExprCast(expr, ref ty) => {
            ExprCast(fld.fold_expr(expr), fld.fold_ty(ty))
        }
        ExprAddrOf(m, ohs) => ExprAddrOf(m, fld.fold_expr(ohs)),
        ExprIf(cond, ref tr, fl) => {
            ExprIf(
                fld.fold_expr(cond),
                fld.fold_block(tr),
                fl.map_move(|x| fld.fold_expr(x))
            )
        }
        ExprWhile(cond, ref body) => {
            ExprWhile(fld.fold_expr(cond), fld.fold_block(body))
        }
        ExprForLoop(pat, iter, ref body, opt_ident) => {
            ExprForLoop(fld.fold_pat(pat),
                        fld.fold_expr(iter),
                        fld.fold_block(body),
                        opt_ident.map_move(|x| fld.fold_ident(x)))
        }
        ExprLoop(ref body, opt_ident) => {
            ExprLoop(
                fld.fold_block(body),
                opt_ident.map_move(|x| fld.fold_ident(x))
            )
        }
        ExprMatch(expr, ref arms) => {
            ExprMatch(
                fld.fold_expr(expr),
                arms.map(|x| fld.fold_arm(x))
            )
        }
        ExprFnBlock(ref decl, ref body) => {
            ExprFnBlock(
                fold_fn_decl(decl, fld),
                fld.fold_block(body)
            )
        }
        ExprBlock(ref blk) => ExprBlock(fld.fold_block(blk)),
        ExprAssign(el, er) => {
            ExprAssign(fld.fold_expr(el), fld.fold_expr(er))
        }
        ExprAssignOp(callee_id, op, el, er) => {
            ExprAssignOp(
                fld.new_id(callee_id),
                op,
                fld.fold_expr(el),
                fld.fold_expr(er)
            )
        }
        ExprField(el, id, ref tys) => {
            ExprField(
                fld.fold_expr(el), fld.fold_ident(id),
                tys.map(|x| fld.fold_ty(x))
            )
        }
        ExprIndex(callee_id, el, er) => {
            ExprIndex(
                fld.new_id(callee_id),
                fld.fold_expr(el),
                fld.fold_expr(er)
            )
        }
        ExprPath(ref pth) => ExprPath(fld.fold_path(pth)),
        ExprSelf => ExprSelf,
        ExprBreak(ref opt_ident) => {
            ExprBreak(opt_ident.map_move(|x| fld.fold_ident(x)))
        }
        ExprAgain(ref opt_ident) => {
            ExprAgain(opt_ident.map_move(|x| fld.fold_ident(x)))
        }
        ExprRet(ref e) => {
            ExprRet(e.map_move(|x| fld.fold_expr(x)))
        }
        ExprLogLevel => ExprLogLevel,
        ExprInlineAsm(ref a) => {
            ExprInlineAsm(inline_asm {
                inputs: a.inputs.map(|&(c, input)| (c, fld.fold_expr(input))),
                outputs: a.outputs.map(|&(c, out)| (c, fld.fold_expr(out))),
                .. (*a).clone()
            })
        }
        ExprMac(ref mac) => ExprMac(fld.fold_mac(mac)),
        ExprStruct(ref path, ref fields, maybe_expr) => {
            ExprStruct(
                fld.fold_path(path),
                fields.map(|x| fold_field(*x)),
                maybe_expr.map_move(|x| fld.fold_expr(x))
            )
        },
        ExprParen(ex) => ExprParen(fld.fold_expr(ex))
    }
}

pub fn noop_fold_ty(t: &ty_, fld: @ast_fold) -> ty_ {
    fn fold_mt(mt: &mt, fld: @ast_fold) -> mt {
        mt {
            ty: ~fld.fold_ty(mt.ty),
            mutbl: mt.mutbl,
        }
    }
    fn fold_field(f: TypeField, fld: @ast_fold) -> TypeField {
        ast::TypeField {
            ident: fld.fold_ident(f.ident),
            mt: fold_mt(&f.mt, fld),
            span: fld.new_span(f.span),
        }
    }
    fn fold_opt_bounds(b: &Option<OptVec<TyParamBound>>, fld: @ast_fold)
                        -> Option<OptVec<TyParamBound>> {
        do b.map |bounds| {
            do bounds.map |bound| { fld.fold_ty_param_bound(bound) }
        }
    }
    match *t {
        ty_nil | ty_bot | ty_infer => (*t).clone(),
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
                lifetimes: fld.fold_lifetimes(&f.lifetimes)
            })
        }
        ty_bare_fn(ref f) => {
            ty_bare_fn(@TyBareFn {
                lifetimes: fld.fold_lifetimes(&f.lifetimes),
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
        ty_typeof(e) => ty_typeof(fld.fold_expr(e)),
        ty_mac(ref mac) => ty_mac(fld.fold_mac(mac))
    }
}

// ...nor do modules
pub fn noop_fold_mod(m: &_mod, fld: @ast_fold) -> _mod {
    ast::_mod {
        view_items: m.view_items.iter().map(|x| fld.fold_view_item(x)).collect(),
        items: m.items.iter().filter_map(|x| fld.fold_item(*x)).collect(),
    }
}

fn noop_fold_foreign_mod(nm: &foreign_mod, fld: @ast_fold) -> foreign_mod {
    ast::foreign_mod {
        sort: nm.sort,
        abis: nm.abis,
        view_items: nm.view_items.iter().map(|x| fld.fold_view_item(x)).collect(),
        items: nm.items.iter().map(|x| fld.fold_foreign_item(*x)).collect(),
    }
}

fn noop_fold_variant(v: &variant_, fld: @ast_fold) -> variant_ {
    let kind = match v.kind {
        tuple_variant_kind(ref variant_args) => {
            tuple_variant_kind(variant_args.map(|x| fld.fold_variant_arg(x)))
        }
        struct_variant_kind(ref struct_def) => {
            struct_variant_kind(@ast::struct_def {
                fields: struct_def.fields.iter()
                    .map(|f| fld.fold_struct_field(*f)).collect(),
                ctor_id: struct_def.ctor_id.map(|c| fld.new_id(*c))
            })
        }
    };

    let attrs = fld.fold_attributes(v.attrs);

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

fn noop_fold_ident(i: Ident, _fld: @ast_fold) -> Ident {
    i
}

fn noop_fold_path(p: &Path, fld: @ast_fold) -> Path {
    ast::Path {
        span: fld.new_span(p.span),
        global: p.global,
        segments: p.segments.map(|segment| ast::PathSegment {
            identifier: fld.fold_ident(segment.identifier),
            lifetime: segment.lifetime,
            types: segment.types.map(|typ| fld.fold_ty(typ)),
        })
    }
}

fn noop_fold_local(l: @Local, fld: @ast_fold) -> @Local {
    @Local {
        is_mutbl: l.is_mutbl,
        ty: fld.fold_ty(&l.ty),
        pat: fld.fold_pat(l.pat),
        init: l.init.map_move(|e| fld.fold_expr(e)),
        id: fld.new_id(l.id),
        span: fld.new_span(l.span),
    }
}

// the default macro traversal. visit the path
// using fold_path, and the tts using fold_tts,
// and the span using new_span
fn noop_fold_mac(m: &mac_, fld: @ast_fold) -> mac_ {
    match *m {
        mac_invoc_tt(ref p,ref tts,ctxt) =>
        mac_invoc_tt(fld.fold_path(p),
                     fold_tts(*tts,fld),
                     ctxt)
    }
}


/* temporarily eta-expand because of a compiler bug with using `fn<T>` as a
   value */
fn noop_map_exprs(f: @fn(@Expr) -> @Expr, es: &[@Expr]) -> ~[@Expr] {
    es.map(|x| f(*x))
}

fn noop_id(i: NodeId) -> NodeId { return i; }

fn noop_span(sp: Span) -> Span { return sp; }

pub fn default_ast_fold() -> ast_fold_fns {
    @AstFoldFns {
        fold_crate: noop_fold_crate,
        fold_view_item: noop_fold_view_item,
        fold_foreign_item: noop_fold_foreign_item,
        fold_item: noop_fold_item,
        fold_struct_field: noop_fold_struct_field,
        fold_item_underscore: noop_fold_item_underscore,
        fold_type_method: noop_fold_type_method,
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
        fold_local: noop_fold_local,
        fold_mac: wrap(noop_fold_mac),
        map_exprs: noop_map_exprs,
        new_id: noop_id,
        new_span: noop_span,
    }
}

impl ast_fold for AstFoldFns {
    /* naturally, a macro to write these would be nice */
    fn fold_crate(@self, c: &Crate) -> Crate {
        (self.fold_crate)(c, self as @ast_fold)
    }
    fn fold_view_item(@self, x: &view_item) -> view_item {
        ast::view_item {
            node: (self.fold_view_item)(&x.node, self as @ast_fold),
            attrs: self.fold_attributes(x.attrs),
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
        @Spanned {
            node: ast::struct_field_ {
                kind: sf.node.kind,
                id: (self.new_id)(sf.node.id),
                ty: self.fold_ty(&sf.node.ty),
                attrs: self.fold_attributes(sf.node.attrs),
            },
            span: (self.new_span)(sf.span),
        }
    }
    fn fold_item_underscore(@self, i: &item_) -> item_ {
        (self.fold_item_underscore)(i, self as @ast_fold)
    }
    fn fold_type_method(@self, m: &TypeMethod) -> TypeMethod {
        (self.fold_type_method)(m, self as @ast_fold)
    }
    fn fold_method(@self, x: @method) -> @method {
        (self.fold_method)(x, self as @ast_fold)
    }
    fn fold_block(@self, x: &Block) -> Block {
        (self.fold_block)(x, self as @ast_fold)
    }
    fn fold_stmt(@self, x: &Stmt) -> Option<@Stmt> {
        let (n_opt, s) = (self.fold_stmt)(&x.node, x.span, self as @ast_fold);
        match n_opt {
            Some(n) => Some(@Spanned { node: n, span: (self.new_span)(s) }),
            None => None,
        }
    }
    fn fold_arm(@self, x: &Arm) -> Arm {
        (self.fold_arm)(x, self as @ast_fold)
    }
    fn fold_pat(@self, x: @Pat) -> @Pat {
        let (n, s) =  (self.fold_pat)(&x.node, x.span, self as @ast_fold);
        @Pat {
            id: (self.new_id)(x.id),
            node: n,
            span: (self.new_span)(s),
        }
    }
    fn fold_decl(@self, x: @Decl) -> Option<@Decl> {
        let (n_opt, s) = (self.fold_decl)(&x.node, x.span, self as @ast_fold);
        match n_opt {
            Some(n) => Some(@Spanned { node: n, span: (self.new_span)(s) }),
            None => None,
        }
    }
    fn fold_expr(@self, x: @Expr) -> @Expr {
        let (n, s) = (self.fold_expr)(&x.node, x.span, self as @ast_fold);
        @Expr {
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
        Spanned { node: n, span: (self.new_span)(s) }
    }
    fn fold_ident(@self, x: Ident) -> Ident {
        (self.fold_ident)(x, self as @ast_fold)
    }
    fn fold_path(@self, x: &Path) -> Path {
        (self.fold_path)(x, self as @ast_fold)
    }
    fn fold_local(@self, x: @Local) -> @Local {
        (self.fold_local)(x, self as @ast_fold)
    }
    fn fold_mac(@self, x: &mac) -> mac {
        let (n, s) = (self.fold_mac)(&x.node, x.span, self as @ast_fold);
        Spanned { node: n, span: (self.new_span)(s) }
    }
    fn map_exprs(@self,
                 f: @fn(@Expr) -> @Expr,
                 e: &[@Expr])
              -> ~[@Expr] {
        (self.map_exprs)(f, e)
    }
    fn new_id(@self, node_id: ast::NodeId) -> NodeId {
        (self.new_id)(node_id)
    }
    fn new_span(@self, span: Span) -> Span {
        (self.new_span)(span)
    }
}

// brson agrees with me that this function's existence is probably
// not a good or useful thing.
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
    pub fn fun_to_ident_folder(f: @fn(ast::Ident)->ast::Ident) -> @ast_fold{
        let afp = default_ast_fold();
        let f_pre = @AstFoldFns{
            fold_ident : |id, _| f(id),
            .. *afp
        };
        make_fold(f_pre)
    }

    // this version doesn't care about getting comments or docstrings in.
    fn fake_print_crate(s: @pprust::ps, crate: &ast::Crate) {
        pprust::print_mod(s, &crate.module, crate.attrs);
    }

    // change every identifier to "zz"
    pub fn to_zz() -> @fn(ast::Ident)->ast::Ident {
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

    // and in cast expressions... this appears to be an existing bug.
    #[test] fn ident_transformation_in_types () {
        let zz_fold = fun_to_ident_folder(to_zz());
        let ast = string_to_crate(@"fn a() {let z = 13 as int;}");
        assert_pred!(matches_codepattern,
                     "matches_codepattern",
                     pprust::to_str(&zz_fold.fold_crate(ast),fake_print_crate,
                                    token::get_ident_interner()),
                     ~"fn zz(){let zz=13 as zz;}");
    }
}
