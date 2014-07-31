// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A Folder represents an AST->AST fold; it accepts an AST piece,
//! and returns a piece of the same type. So, for instance, macro
//! expansion is a Folder that walks over an AST and produces another
//! AST.
//!
//! Note: using a Folder (other than the MacroExpander Folder) on
//! an AST before macro expansion is probably a bad idea. For instance,
//! a folder renaming item names in a module will miss all of those
//! that are created by the expansion of a macro.

use ast::*;
use ast;
use ast_util;
use codemap::{respan, Span, Spanned};
use parse::token;
use owned_slice::OwnedSlice;
use util::small_vector::SmallVector;

use std::rc::Rc;
use std::gc::{Gc, GC};

pub trait Folder {
    // Any additions to this trait should happen in form
    // of a call to a public `noop_*` function that only calls
    // out to the folder again, not other `noop_*` functions.
    //
    // This is a necessary API workaround to the problem of not
    // being able to call out to the super default method
    // in an overridden default method.

    fn fold_crate(&mut self, c: Crate) -> Crate {
        noop_fold_crate(c, self)
    }

    fn fold_meta_items(&mut self, meta_items: &[Gc<MetaItem>]) -> Vec<Gc<MetaItem>> {
        noop_fold_meta_items(meta_items, self)
    }

    fn fold_meta_item(&mut self, meta_item: &MetaItem) -> MetaItem {
        noop_fold_meta_item(meta_item, self)
    }

    fn fold_view_path(&mut self, view_path: Gc<ViewPath>) -> Gc<ViewPath> {
        noop_fold_view_path(view_path, self)
    }

    fn fold_view_item(&mut self, vi: &ViewItem) -> ViewItem {
        noop_fold_view_item(vi, self)
    }

    fn fold_foreign_item(&mut self, ni: Gc<ForeignItem>) -> Gc<ForeignItem> {
        noop_fold_foreign_item(&*ni, self)
    }

    fn fold_item(&mut self, i: Gc<Item>) -> SmallVector<Gc<Item>> {
        noop_fold_item(&*i, self)
    }

    fn fold_item_simple(&mut self, i: &Item) -> Item {
        noop_fold_item_simple(i, self)
    }

    fn fold_struct_field(&mut self, sf: &StructField) -> StructField {
        noop_fold_struct_field(sf, self)
    }

    fn fold_item_underscore(&mut self, i: &Item_) -> Item_ {
        noop_fold_item_underscore(i, self)
    }

    fn fold_fn_decl(&mut self, d: &FnDecl) -> P<FnDecl> {
        noop_fold_fn_decl(d, self)
    }

    fn fold_type_method(&mut self, m: &TypeMethod) -> TypeMethod {
        noop_fold_type_method(m, self)
    }

    fn fold_method(&mut self, m: Gc<Method>) -> SmallVector<Gc<Method>>  {
        noop_fold_method(&*m, self)
    }

    fn fold_block(&mut self, b: P<Block>) -> P<Block> {
        noop_fold_block(b, self)
    }

    fn fold_stmt(&mut self, s: &Stmt) -> SmallVector<Gc<Stmt>> {
        noop_fold_stmt(s, self)
    }

    fn fold_arm(&mut self, a: &Arm) -> Arm {
        noop_fold_arm(a, self)
    }

    fn fold_pat(&mut self, p: Gc<Pat>) -> Gc<Pat> {
        noop_fold_pat(p, self)
    }

    fn fold_decl(&mut self, d: Gc<Decl>) -> SmallVector<Gc<Decl>> {
        noop_fold_decl(d, self)
    }

    fn fold_expr(&mut self, e: Gc<Expr>) -> Gc<Expr> {
        noop_fold_expr(e, self)
    }

    fn fold_ty(&mut self, t: P<Ty>) -> P<Ty> {
        noop_fold_ty(t, self)
    }

    fn fold_mod(&mut self, m: &Mod) -> Mod {
        noop_fold_mod(m, self)
    }

    fn fold_foreign_mod(&mut self, nm: &ForeignMod) -> ForeignMod {
        noop_fold_foreign_mod(nm, self)
    }

    fn fold_variant(&mut self, v: &Variant) -> P<Variant> {
        noop_fold_variant(v, self)
    }

    fn fold_ident(&mut self, i: Ident) -> Ident {
        noop_fold_ident(i, self)
    }

    fn fold_path(&mut self, p: &Path) -> Path {
        noop_fold_path(p, self)
    }

    fn fold_local(&mut self, l: Gc<Local>) -> Gc<Local> {
        noop_fold_local(l, self)
    }

    fn fold_mac(&mut self, _macro: &Mac) -> Mac {
        fail!("fold_mac disabled by default");
        // NB: see note about macros above.
        // if you really want a folder that
        // works on macros, use this
        // definition in your trait impl:
        // fold::noop_fold_mac(_macro, self)
    }

    fn fold_explicit_self(&mut self, es: &ExplicitSelf) -> ExplicitSelf {
        noop_fold_explicit_self(es, self)
    }

    fn fold_explicit_self_underscore(&mut self, es: &ExplicitSelf_) -> ExplicitSelf_ {
        noop_fold_explicit_self_underscore(es, self)
    }

    fn fold_lifetime(&mut self, l: &Lifetime) -> Lifetime {
        noop_fold_lifetime(l, self)
    }

    fn fold_attribute(&mut self, at: Attribute) -> Attribute {
        noop_fold_attribute(at, self)
    }

    fn fold_arg(&mut self, a: &Arg) -> Arg {
        noop_fold_arg(a, self)
    }

    fn fold_generics(&mut self, generics: &Generics) -> Generics {
        noop_fold_generics(generics, self)
    }

    fn fold_trait_ref(&mut self, p: &TraitRef) -> TraitRef {
        noop_fold_trait_ref(p, self)
    }

    fn fold_struct_def(&mut self, struct_def: Gc<StructDef>) -> Gc<StructDef> {
        noop_fold_struct_def(struct_def, self)
    }

    fn fold_lifetimes(&mut self, lts: &[Lifetime]) -> Vec<Lifetime> {
        noop_fold_lifetimes(lts, self)
    }

    fn fold_ty_param(&mut self, tp: &TyParam) -> TyParam {
        noop_fold_ty_param(tp, self)
    }

    fn fold_ty_params(&mut self, tps: &[TyParam]) -> OwnedSlice<TyParam> {
        noop_fold_ty_params(tps, self)
    }

    fn fold_tt(&mut self, tt: &TokenTree) -> TokenTree {
        noop_fold_tt(tt, self)
    }

    fn fold_tts(&mut self, tts: &[TokenTree]) -> Vec<TokenTree> {
        noop_fold_tts(tts, self)
    }

    fn fold_token(&mut self, t: &token::Token) -> token::Token {
        noop_fold_token(t, self)
    }

    fn fold_interpolated(&mut self, nt : &token::Nonterminal) -> token::Nonterminal {
        noop_fold_interpolated(nt, self)
    }

    fn fold_opt_lifetime(&mut self, o_lt: &Option<Lifetime>) -> Option<Lifetime> {
        noop_fold_opt_lifetime(o_lt, self)
    }

    fn fold_variant_arg(&mut self, va: &VariantArg) -> VariantArg {
        noop_fold_variant_arg(va, self)
    }

    fn fold_ty_param_bound(&mut self, tpb: &TyParamBound) -> TyParamBound {
        noop_fold_ty_param_bound(tpb, self)
    }

    fn fold_opt_bounds(&mut self, b: &Option<OwnedSlice<TyParamBound>>)
                       -> Option<OwnedSlice<TyParamBound>> {
        noop_fold_opt_bounds(b, self)
    }

    fn fold_mt(&mut self, mt: &MutTy) -> MutTy {
        noop_fold_mt(mt, self)
    }

    fn fold_field(&mut self, field: Field) -> Field {
        noop_fold_field(field, self)
    }

// Helper methods:

    fn map_exprs(&self, f: |Gc<Expr>| -> Gc<Expr>,
                 es: &[Gc<Expr>]) -> Vec<Gc<Expr>> {
        es.iter().map(|x| f(*x)).collect()
    }

    fn new_id(&mut self, i: NodeId) -> NodeId {
        i
    }

    fn new_span(&mut self, sp: Span) -> Span {
        sp
    }
}

pub fn noop_fold_meta_items<T: Folder>(meta_items: &[Gc<MetaItem>], fld: &mut T)
                                       -> Vec<Gc<MetaItem>> {
    meta_items.iter().map(|x| box (GC) fld.fold_meta_item(&**x)).collect()
}

pub fn noop_fold_view_path<T: Folder>(view_path: Gc<ViewPath>, fld: &mut T) -> Gc<ViewPath> {
    let inner_view_path = match view_path.node {
        ViewPathSimple(ref ident, ref path, node_id) => {
            let id = fld.new_id(node_id);
            ViewPathSimple(ident.clone(),
                        fld.fold_path(path),
                        id)
        }
        ViewPathGlob(ref path, node_id) => {
            let id = fld.new_id(node_id);
            ViewPathGlob(fld.fold_path(path), id)
        }
        ViewPathList(ref path, ref path_list_idents, node_id) => {
            let id = fld.new_id(node_id);
            ViewPathList(fld.fold_path(path),
                        path_list_idents.iter().map(|path_list_ident| {
                            Spanned {
                                node: match path_list_ident.node {
                                    PathListIdent { id, name } =>
                                        PathListIdent {
                                            id: fld.new_id(id),
                                            name: name.clone()
                                        },
                                    PathListMod { id } =>
                                        PathListMod { id: fld.new_id(id) }
                                },
                                span: fld.new_span(path_list_ident.span)
                            }
                        }).collect(),
                        id)
        }
    };
    box(GC) Spanned {
        node: inner_view_path,
        span: fld.new_span(view_path.span),
    }
}

pub fn noop_fold_arm<T: Folder>(a: &Arm, fld: &mut T) -> Arm {
    Arm {
        attrs: a.attrs.iter().map(|x| fld.fold_attribute(*x)).collect(),
        pats: a.pats.iter().map(|x| fld.fold_pat(*x)).collect(),
        guard: a.guard.map(|x| fld.fold_expr(x)),
        body: fld.fold_expr(a.body),
    }
}

pub fn noop_fold_decl<T: Folder>(d: Gc<Decl>, fld: &mut T) -> SmallVector<Gc<Decl>> {
    let node = match d.node {
        DeclLocal(ref l) => SmallVector::one(DeclLocal(fld.fold_local(*l))),
        DeclItem(it) => {
            fld.fold_item(it).move_iter().map(|i| DeclItem(i)).collect()
        }
    };

    node.move_iter().map(|node| {
        box(GC) Spanned {
            node: node,
            span: fld.new_span(d.span),
        }
    }).collect()
}

pub fn noop_fold_ty<T: Folder>(t: P<Ty>, fld: &mut T) -> P<Ty> {
    let id = fld.new_id(t.id);
    let node = match t.node {
        TyNil | TyBot | TyInfer => t.node.clone(),
        TyBox(ty) => TyBox(fld.fold_ty(ty)),
        TyUniq(ty) => TyUniq(fld.fold_ty(ty)),
        TyVec(ty) => TyVec(fld.fold_ty(ty)),
        TyPtr(ref mt) => TyPtr(fld.fold_mt(mt)),
        TyRptr(ref region, ref mt) => {
            TyRptr(fld.fold_opt_lifetime(region), fld.fold_mt(mt))
        }
        TyClosure(ref f, ref region) => {
            TyClosure(box(GC) ClosureTy {
                fn_style: f.fn_style,
                onceness: f.onceness,
                bounds: fld.fold_opt_bounds(&f.bounds),
                decl: fld.fold_fn_decl(&*f.decl),
                lifetimes: f.lifetimes.iter().map(|l| fld.fold_lifetime(l)).collect(),
            }, fld.fold_opt_lifetime(region))
        }
        TyProc(ref f) => {
            TyProc(box(GC) ClosureTy {
                fn_style: f.fn_style,
                onceness: f.onceness,
                bounds: fld.fold_opt_bounds(&f.bounds),
                decl: fld.fold_fn_decl(&*f.decl),
                lifetimes: f.lifetimes.iter().map(|l| fld.fold_lifetime(l)).collect(),
            })
        }
        TyBareFn(ref f) => {
            TyBareFn(box(GC) BareFnTy {
                lifetimes: f.lifetimes.iter().map(|l| fld.fold_lifetime(l)).collect(),
                fn_style: f.fn_style,
                abi: f.abi,
                decl: fld.fold_fn_decl(&*f.decl)
            })
        }
        TyUnboxedFn(ref f) => {
            TyUnboxedFn(box(GC) UnboxedFnTy {
                decl: fld.fold_fn_decl(&*f.decl),
            })
        }
        TyTup(ref tys) => TyTup(tys.iter().map(|&ty| fld.fold_ty(ty)).collect()),
        TyParen(ref ty) => TyParen(fld.fold_ty(*ty)),
        TyPath(ref path, ref bounds, id) => {
            let id = fld.new_id(id);
            TyPath(fld.fold_path(path),
                    fld.fold_opt_bounds(bounds),
                    id)
        }
        TyFixedLengthVec(ty, e) => {
            TyFixedLengthVec(fld.fold_ty(ty), fld.fold_expr(e))
        }
        TyTypeof(expr) => TyTypeof(fld.fold_expr(expr)),
    };
    P(Ty {
        id: id,
        span: fld.new_span(t.span),
        node: node,
    })
}

pub fn noop_fold_foreign_mod<T: Folder>(nm: &ForeignMod, fld: &mut T) -> ForeignMod {
    ast::ForeignMod {
        abi: nm.abi,
        view_items: nm.view_items
                        .iter()
                        .map(|x| fld.fold_view_item(x))
                        .collect(),
        items: nm.items
                    .iter()
                    .map(|x| fld.fold_foreign_item(*x))
                    .collect(),
    }
}

pub fn noop_fold_variant<T: Folder>(v: &Variant, fld: &mut T) -> P<Variant> {
    let id = fld.new_id(v.node.id);
    let kind;
    match v.node.kind {
        TupleVariantKind(ref variant_args) => {
            kind = TupleVariantKind(variant_args.iter().map(|x|
                fld.fold_variant_arg(x)).collect())
        }
        StructVariantKind(ref struct_def) => {
            kind = StructVariantKind(box(GC) ast::StructDef {
                fields: struct_def.fields.iter()
                    .map(|f| fld.fold_struct_field(f)).collect(),
                ctor_id: struct_def.ctor_id.map(|c| fld.new_id(c)),
                super_struct: match struct_def.super_struct {
                    Some(t) => Some(fld.fold_ty(t)),
                    None => None
                },
                is_virtual: struct_def.is_virtual,
            })
        }
    }

    let attrs = v.node.attrs.iter().map(|x| fld.fold_attribute(*x)).collect();

    let de = match v.node.disr_expr {
        Some(e) => Some(fld.fold_expr(e)),
        None => None
    };
    let node = ast::Variant_ {
        name: v.node.name,
        attrs: attrs,
        kind: kind,
        id: id,
        disr_expr: de,
        vis: v.node.vis,
    };
    P(Spanned {
        node: node,
        span: fld.new_span(v.span),
    })
}

pub fn noop_fold_ident<T: Folder>(i: Ident, _: &mut T) -> Ident {
    i
}

pub fn noop_fold_path<T: Folder>(p: &Path, fld: &mut T) -> Path {
    ast::Path {
        span: fld.new_span(p.span),
        global: p.global,
        segments: p.segments.iter().map(|segment| ast::PathSegment {
            identifier: fld.fold_ident(segment.identifier),
            lifetimes: segment.lifetimes.iter().map(|l| fld.fold_lifetime(l)).collect(),
            types: segment.types.iter().map(|&typ| fld.fold_ty(typ)).collect(),
        }).collect()
    }
}

pub fn noop_fold_local<T: Folder>(l: Gc<Local>, fld: &mut T) -> Gc<Local> {
    let id = fld.new_id(l.id); // Needs to be first, for ast_map.
    box(GC) Local {
        id: id,
        ty: fld.fold_ty(l.ty),
        pat: fld.fold_pat(l.pat),
        init: l.init.map(|e| fld.fold_expr(e)),
        span: fld.new_span(l.span),
        source: l.source,
    }
}

pub fn noop_fold_attribute<T: Folder>(at: Attribute, fld: &mut T) -> Attribute {
    Spanned {
        span: fld.new_span(at.span),
        node: ast::Attribute_ {
            id: at.node.id,
            style: at.node.style,
            value: box (GC) fld.fold_meta_item(&*at.node.value),
            is_sugared_doc: at.node.is_sugared_doc
        }
    }
}

pub fn noop_fold_explicit_self_underscore<T: Folder>(es: &ExplicitSelf_, fld: &mut T)
                                                     -> ExplicitSelf_ {
    match *es {
        SelfStatic | SelfValue(_) => *es,
        SelfRegion(ref lifetime, m, id) => {
            SelfRegion(fld.fold_opt_lifetime(lifetime), m, id)
        }
        SelfExplicit(ref typ, id) => SelfExplicit(fld.fold_ty(*typ), id),
    }
}

pub fn noop_fold_explicit_self<T: Folder>(es: &ExplicitSelf, fld: &mut T) -> ExplicitSelf {
    Spanned {
        span: fld.new_span(es.span),
        node: fld.fold_explicit_self_underscore(&es.node)
    }
}


pub fn noop_fold_mac<T: Folder>(macro: &Mac, fld: &mut T) -> Mac {
    Spanned {
        node: match macro.node {
            MacInvocTT(ref p, ref tts, ctxt) => {
                MacInvocTT(fld.fold_path(p),
                           fld.fold_tts(tts.as_slice()),
                           ctxt)
            }
        },
        span: fld.new_span(macro.span)
    }
}

pub fn noop_fold_meta_item<T: Folder>(mi: &MetaItem, fld: &mut T) -> MetaItem {
    Spanned {
        node:
            match mi.node {
                MetaWord(ref id) => MetaWord((*id).clone()),
                MetaList(ref id, ref mis) => {
                    MetaList((*id).clone(),
                             mis.iter()
                                .map(|e| box (GC) fld.fold_meta_item(&**e)).collect())
                }
                MetaNameValue(ref id, ref s) => {
                    MetaNameValue((*id).clone(), (*s).clone())
                }
            },
        span: fld.new_span(mi.span) }
}

pub fn noop_fold_arg<T: Folder>(a: &Arg, fld: &mut T) -> Arg {
    let id = fld.new_id(a.id); // Needs to be first, for ast_map.
    Arg {
        id: id,
        ty: fld.fold_ty(a.ty),
        pat: fld.fold_pat(a.pat),
    }
}

pub fn noop_fold_tt<T: Folder>(tt: &TokenTree, fld: &mut T) -> TokenTree {
    match *tt {
        TTTok(span, ref tok) =>
            TTTok(span, fld.fold_token(tok)),
        TTDelim(ref tts) => TTDelim(Rc::new(fld.fold_tts(tts.as_slice()))),
        TTSeq(span, ref pattern, ref sep, is_optional) =>
            TTSeq(span,
                  Rc::new(fld.fold_tts(pattern.as_slice())),
                  sep.as_ref().map(|tok| fld.fold_token(tok)),
                  is_optional),
        TTNonterminal(sp,ref ident) =>
            TTNonterminal(sp,fld.fold_ident(*ident))
    }
}

pub fn noop_fold_tts<T: Folder>(tts: &[TokenTree], fld: &mut T) -> Vec<TokenTree> {
    tts.iter().map(|tt| fld.fold_tt(tt)).collect()
}

// apply ident folder if it's an ident, apply other folds to interpolated nodes
pub fn noop_fold_token<T: Folder>(t: &token::Token, fld: &mut T) -> token::Token {
    match *t {
        token::IDENT(id, followed_by_colons) => {
            token::IDENT(fld.fold_ident(id), followed_by_colons)
        }
        token::LIFETIME(id) => token::LIFETIME(fld.fold_ident(id)),
        token::INTERPOLATED(ref nt) => token::INTERPOLATED(fld.fold_interpolated(nt)),
        _ => (*t).clone()
    }
}

/// apply folder to elements of interpolated nodes
//
// NB: this can occur only when applying a fold to partially expanded code, where
// parsed pieces have gotten implanted ito *other* macro invocations. This is relevant
// for macro hygiene, but possibly not elsewhere.
//
// One problem here occurs because the types for fold_item, fold_stmt, etc. allow the
// folder to return *multiple* items; this is a problem for the nodes here, because
// they insist on having exactly one piece. One solution would be to mangle the fold
// trait to include one-to-many and one-to-one versions of these entry points, but that
// would probably confuse a lot of people and help very few. Instead, I'm just going
// to put in dynamic checks. I think the performance impact of this will be pretty much
// nonexistent. The danger is that someone will apply a fold to a partially expanded
// node, and will be confused by the fact that their "fold_item" or "fold_stmt" isn't
// getting called on NtItem or NtStmt nodes. Hopefully they'll wind up reading this
// comment, and doing something appropriate.
//
// BTW, design choice: I considered just changing the type of, e.g., NtItem to contain
// multiple items, but decided against it when I looked at parse_item_or_view_item and
// tried to figure out what I would do with multiple items there....
pub fn noop_fold_interpolated<T: Folder>(nt : &token::Nonterminal, fld: &mut T)
                                         -> token::Nonterminal {
    match *nt {
        token::NtItem(item) =>
            token::NtItem(fld.fold_item(item)
                          // this is probably okay, because the only folds likely
                          // to peek inside interpolated nodes will be renamings/markings,
                          // which map single items to single items
                          .expect_one("expected fold to produce exactly one item")),
        token::NtBlock(block) => token::NtBlock(fld.fold_block(block)),
        token::NtStmt(stmt) =>
            token::NtStmt(fld.fold_stmt(&*stmt)
                          // this is probably okay, because the only folds likely
                          // to peek inside interpolated nodes will be renamings/markings,
                          // which map single items to single items
                          .expect_one("expected fold to produce exactly one statement")),
        token::NtPat(pat) => token::NtPat(fld.fold_pat(pat)),
        token::NtExpr(expr) => token::NtExpr(fld.fold_expr(expr)),
        token::NtTy(ty) => token::NtTy(fld.fold_ty(ty)),
        token::NtIdent(ref id, is_mod_name) =>
            token::NtIdent(box fld.fold_ident(**id),is_mod_name),
        token::NtMeta(meta_item) => token::NtMeta(box (GC) fld.fold_meta_item(&*meta_item)),
        token::NtPath(ref path) => token::NtPath(box fld.fold_path(&**path)),
        token::NtTT(tt) => token::NtTT(box (GC) fld.fold_tt(&*tt)),
        // it looks to me like we can leave out the matchers: token::NtMatchers(matchers)
        _ => (*nt).clone()
    }
}

pub fn noop_fold_fn_decl<T: Folder>(decl: &FnDecl, fld: &mut T) -> P<FnDecl> {
    P(FnDecl {
        inputs: decl.inputs.iter().map(|x| fld.fold_arg(x)).collect(), // bad copy
        output: fld.fold_ty(decl.output),
        cf: decl.cf,
        variadic: decl.variadic
    })
}

pub fn noop_fold_ty_param_bound<T: Folder>(tpb: &TyParamBound, fld: &mut T)
                                           -> TyParamBound {
    match *tpb {
        TraitTyParamBound(ref ty) => TraitTyParamBound(fld.fold_trait_ref(ty)),
        StaticRegionTyParamBound => StaticRegionTyParamBound,
        UnboxedFnTyParamBound(ref unboxed_function_type) => {
            UnboxedFnTyParamBound(UnboxedFnTy {
                decl: fld.fold_fn_decl(&*unboxed_function_type.decl),
            })
        }
        OtherRegionTyParamBound(s) => OtherRegionTyParamBound(s)
    }
}

pub fn noop_fold_ty_param<T: Folder>(tp: &TyParam, fld: &mut T) -> TyParam {
    let id = fld.new_id(tp.id);
    TyParam {
        ident: tp.ident,
        id: id,
        bounds: tp.bounds.map(|x| fld.fold_ty_param_bound(x)),
        unbound: tp.unbound.as_ref().map(|x| fld.fold_ty_param_bound(x)),
        default: tp.default.map(|x| fld.fold_ty(x)),
        span: tp.span
    }
}

pub fn noop_fold_ty_params<T: Folder>(tps: &[TyParam], fld: &mut T)
                                      -> OwnedSlice<TyParam> {
    tps.iter().map(|tp| fld.fold_ty_param(tp)).collect()
}

pub fn noop_fold_lifetime<T: Folder>(l: &Lifetime, fld: &mut T) -> Lifetime {
    let id = fld.new_id(l.id);
    Lifetime {
        id: id,
        span: fld.new_span(l.span),
        name: l.name
    }
}

pub fn noop_fold_lifetimes<T: Folder>(lts: &[Lifetime], fld: &mut T) -> Vec<Lifetime> {
    lts.iter().map(|l| fld.fold_lifetime(l)).collect()
}

pub fn noop_fold_opt_lifetime<T: Folder>(o_lt: &Option<Lifetime>, fld: &mut T)
                                      -> Option<Lifetime> {
    o_lt.as_ref().map(|lt| fld.fold_lifetime(lt))
}

pub fn noop_fold_generics<T: Folder>(generics: &Generics, fld: &mut T) -> Generics {
    Generics {ty_params: fld.fold_ty_params(generics.ty_params.as_slice()),
              lifetimes: fld.fold_lifetimes(generics.lifetimes.as_slice())}
}

pub fn noop_fold_struct_def<T: Folder>(struct_def: Gc<StructDef>,
                              fld: &mut T) -> Gc<StructDef> {
    box(GC) ast::StructDef {
        fields: struct_def.fields.iter().map(|f| fld.fold_struct_field(f)).collect(),
        ctor_id: struct_def.ctor_id.map(|cid| fld.new_id(cid)),
        super_struct: match struct_def.super_struct {
            Some(t) => Some(fld.fold_ty(t)),
            None => None
        },
        is_virtual: struct_def.is_virtual,
    }
}

pub fn noop_fold_trait_ref<T: Folder>(p: &TraitRef, fld: &mut T) -> TraitRef {
    let id = fld.new_id(p.ref_id);
    ast::TraitRef {
        path: fld.fold_path(&p.path),
        ref_id: id,
    }
}

pub fn noop_fold_struct_field<T: Folder>(f: &StructField, fld: &mut T) -> StructField {
    let id = fld.new_id(f.node.id);
    Spanned {
        node: ast::StructField_ {
            kind: f.node.kind,
            id: id,
            ty: fld.fold_ty(f.node.ty),
            attrs: f.node.attrs.iter().map(|a| fld.fold_attribute(*a)).collect(),
        },
        span: fld.new_span(f.span),
    }
}

pub fn noop_fold_field<T: Folder>(field: Field, folder: &mut T) -> Field {
    ast::Field {
        ident: respan(field.ident.span, folder.fold_ident(field.ident.node)),
        expr: folder.fold_expr(field.expr),
        span: folder.new_span(field.span),
    }
}

pub fn noop_fold_mt<T: Folder>(mt: &MutTy, folder: &mut T) -> MutTy {
    MutTy {
        ty: folder.fold_ty(mt.ty),
        mutbl: mt.mutbl,
    }
}

pub fn noop_fold_opt_bounds<T: Folder>(b: &Option<OwnedSlice<TyParamBound>>, folder: &mut T)
                              -> Option<OwnedSlice<TyParamBound>> {
    b.as_ref().map(|bounds| {
        bounds.map(|bound| {
            folder.fold_ty_param_bound(bound)
        })
    })
}

pub fn noop_fold_variant_arg<T: Folder>(va: &VariantArg, folder: &mut T) -> VariantArg {
    let id = folder.new_id(va.id);
    ast::VariantArg {
        ty: folder.fold_ty(va.ty),
        id: id,
    }
}

pub fn noop_fold_view_item<T: Folder>(vi: &ViewItem, folder: &mut T)
                                       -> ViewItem{
    let inner_view_item = match vi.node {
        ViewItemExternCrate(ref ident, ref string, node_id) => {
            ViewItemExternCrate(ident.clone(),
                              (*string).clone(),
                              folder.new_id(node_id))
        }
        ViewItemUse(ref view_path) => {
            ViewItemUse(folder.fold_view_path(*view_path))
        }
    };
    ViewItem {
        node: inner_view_item,
        attrs: vi.attrs.iter().map(|a| folder.fold_attribute(*a)).collect(),
        vis: vi.vis,
        span: folder.new_span(vi.span),
    }
}

pub fn noop_fold_block<T: Folder>(b: P<Block>, folder: &mut T) -> P<Block> {
    let id = folder.new_id(b.id); // Needs to be first, for ast_map.
    let view_items = b.view_items.iter().map(|x| folder.fold_view_item(x)).collect();
    let stmts = b.stmts.iter().flat_map(|s| folder.fold_stmt(&**s).move_iter()).collect();
    P(Block {
        id: id,
        view_items: view_items,
        stmts: stmts,
        expr: b.expr.map(|x| folder.fold_expr(x)),
        rules: b.rules,
        span: folder.new_span(b.span),
    })
}

pub fn noop_fold_item_underscore<T: Folder>(i: &Item_, folder: &mut T) -> Item_ {
    match *i {
        ItemStatic(t, m, e) => {
            ItemStatic(folder.fold_ty(t), m, folder.fold_expr(e))
        }
        ItemFn(decl, fn_style, abi, ref generics, body) => {
            ItemFn(
                folder.fold_fn_decl(&*decl),
                fn_style,
                abi,
                folder.fold_generics(generics),
                folder.fold_block(body)
            )
        }
        ItemMod(ref m) => ItemMod(folder.fold_mod(m)),
        ItemForeignMod(ref nm) => ItemForeignMod(folder.fold_foreign_mod(nm)),
        ItemTy(t, ref generics) => {
            ItemTy(folder.fold_ty(t), folder.fold_generics(generics))
        }
        ItemEnum(ref enum_definition, ref generics) => {
            ItemEnum(
                ast::EnumDef {
                    variants: enum_definition.variants.iter().map(|&x| {
                        folder.fold_variant(&*x)
                    }).collect(),
                },
                folder.fold_generics(generics))
        }
        ItemStruct(ref struct_def, ref generics) => {
            let struct_def = folder.fold_struct_def(*struct_def);
            ItemStruct(struct_def, folder.fold_generics(generics))
        }
        ItemImpl(ref generics, ref ifce, ty, ref methods) => {
            ItemImpl(folder.fold_generics(generics),
                     ifce.as_ref().map(|p| folder.fold_trait_ref(p)),
                     folder.fold_ty(ty),
                     methods.iter().flat_map(|x| folder.fold_method(*x).move_iter()).collect()
            )
        }
        ItemTrait(ref generics, ref unbound, ref traits, ref methods) => {
            let methods = methods.iter().flat_map(|method| {
                let r = match *method {
                    Required(ref m) =>
                            SmallVector::one(Required(folder.fold_type_method(m))).move_iter(),
                    Provided(method) => {
                            // the awkward collect/iter idiom here is because
                            // even though an iter and a map satisfy the same trait bound,
                            // they're not actually the same type, so the method arms
                            // don't unify.
                            let methods : SmallVector<ast::TraitMethod> =
                                folder.fold_method(method).move_iter()
                                .map(|m| Provided(m)).collect();
                            methods.move_iter()
                        }
                };
                r
            }).collect();
            ItemTrait(folder.fold_generics(generics),
                      unbound.clone(),
                      traits.iter().map(|p| folder.fold_trait_ref(p)).collect(),
                      methods)
        }
        ItemMac(ref m) => ItemMac(folder.fold_mac(m)),
    }
}

pub fn noop_fold_type_method<T: Folder>(m: &TypeMethod, fld: &mut T) -> TypeMethod {
    let id = fld.new_id(m.id); // Needs to be first, for ast_map.
    TypeMethod {
        id: id,
        ident: fld.fold_ident(m.ident),
        attrs: m.attrs.iter().map(|a| fld.fold_attribute(*a)).collect(),
        fn_style: m.fn_style,
        abi: m.abi,
        decl: fld.fold_fn_decl(&*m.decl),
        generics: fld.fold_generics(&m.generics),
        explicit_self: fld.fold_explicit_self(&m.explicit_self),
        span: fld.new_span(m.span),
        vis: m.vis,
    }
}

pub fn noop_fold_mod<T: Folder>(m: &Mod, folder: &mut T) -> Mod {
    ast::Mod {
        inner: folder.new_span(m.inner),
        view_items: m.view_items
                     .iter()
                     .map(|x| folder.fold_view_item(x)).collect(),
        items: m.items.iter().flat_map(|x| folder.fold_item(*x).move_iter()).collect(),
    }
}

pub fn noop_fold_crate<T: Folder>(c: Crate, folder: &mut T) -> Crate {
    Crate {
        module: folder.fold_mod(&c.module),
        attrs: c.attrs.iter().map(|x| folder.fold_attribute(*x)).collect(),
        config: c.config.iter().map(|x| box (GC) folder.fold_meta_item(&**x)).collect(),
        span: folder.new_span(c.span),
        exported_macros: c.exported_macros
    }
}

// fold one item into possibly many items
pub fn noop_fold_item<T: Folder>(i: &Item,
                                 folder: &mut T) -> SmallVector<Gc<Item>> {
    SmallVector::one(box(GC) folder.fold_item_simple(i))
}


// fold one item into exactly one item
pub fn noop_fold_item_simple<T: Folder>(i: &Item, folder: &mut T) -> Item {
    let id = folder.new_id(i.id); // Needs to be first, for ast_map.
    let node = folder.fold_item_underscore(&i.node);
    let ident = match node {
        // The node may have changed, recompute the "pretty" impl name.
        ItemImpl(_, ref maybe_trait, ty, _) => {
            ast_util::impl_pretty_name(maybe_trait, &*ty)
        }
        _ => i.ident
    };

    Item {
        id: id,
        ident: folder.fold_ident(ident),
        attrs: i.attrs.iter().map(|e| folder.fold_attribute(*e)).collect(),
        node: node,
        vis: i.vis,
        span: folder.new_span(i.span)
    }
}

pub fn noop_fold_foreign_item<T: Folder>(ni: &ForeignItem,
                                         folder: &mut T) -> Gc<ForeignItem> {
    let id = folder.new_id(ni.id); // Needs to be first, for ast_map.
    box(GC) ForeignItem {
        id: id,
        ident: folder.fold_ident(ni.ident),
        attrs: ni.attrs.iter().map(|x| folder.fold_attribute(*x)).collect(),
        node: match ni.node {
            ForeignItemFn(ref fdec, ref generics) => {
                ForeignItemFn(P(FnDecl {
                    inputs: fdec.inputs.iter().map(|a| folder.fold_arg(a)).collect(),
                    output: folder.fold_ty(fdec.output),
                    cf: fdec.cf,
                    variadic: fdec.variadic
                }), folder.fold_generics(generics))
            }
            ForeignItemStatic(t, m) => {
                ForeignItemStatic(folder.fold_ty(t), m)
            }
        },
        span: folder.new_span(ni.span),
        vis: ni.vis,
    }
}

// Default fold over a method.
// Invariant: produces exactly one method.
pub fn noop_fold_method<T: Folder>(m: &Method, folder: &mut T) -> SmallVector<Gc<Method>> {
    let id = folder.new_id(m.id); // Needs to be first, for ast_map.
    SmallVector::one(box(GC) Method {
        attrs: m.attrs.iter().map(|a| folder.fold_attribute(*a)).collect(),
        id: id,
        span: folder.new_span(m.span),
        node: match m.node {
            MethDecl(ident,
                     ref generics,
                     abi,
                     ref explicit_self,
                     fn_style,
                     decl,
                     body,
                     vis) => {
                MethDecl(folder.fold_ident(ident),
                         folder.fold_generics(generics),
                         abi,
                         folder.fold_explicit_self(explicit_self),
                         fn_style,
                         folder.fold_fn_decl(&*decl),
                         folder.fold_block(body),
                         vis)
            },
            MethMac(ref mac) => MethMac(folder.fold_mac(mac)),
        }
    })
}

pub fn noop_fold_pat<T: Folder>(p: Gc<Pat>, folder: &mut T) -> Gc<Pat> {
    let id = folder.new_id(p.id);
    let node = match p.node {
        PatWild => PatWild,
        PatWildMulti => PatWildMulti,
        PatIdent(binding_mode, ref pth1, ref sub) => {
            PatIdent(binding_mode,
                     Spanned{span: folder.new_span(pth1.span),
                             node: folder.fold_ident(pth1.node)},
                     sub.map(|x| folder.fold_pat(x)))
        }
        PatLit(e) => PatLit(folder.fold_expr(e)),
        PatEnum(ref pth, ref pats) => {
            PatEnum(folder.fold_path(pth),
                    pats.as_ref().map(|pats| pats.iter().map(|x| folder.fold_pat(*x)).collect()))
        }
        PatStruct(ref pth, ref fields, etc) => {
            let pth_ = folder.fold_path(pth);
            let fs = fields.iter().map(|f| {
                ast::FieldPat {
                    ident: f.ident,
                    pat: folder.fold_pat(f.pat)
                }
            }).collect();
            PatStruct(pth_, fs, etc)
        }
        PatTup(ref elts) => PatTup(elts.iter().map(|x| folder.fold_pat(*x)).collect()),
        PatBox(inner) => PatBox(folder.fold_pat(inner)),
        PatRegion(inner) => PatRegion(folder.fold_pat(inner)),
        PatRange(e1, e2) => {
            PatRange(folder.fold_expr(e1), folder.fold_expr(e2))
        },
        PatVec(ref before, ref slice, ref after) => {
            PatVec(before.iter().map(|x| folder.fold_pat(*x)).collect(),
                    slice.map(|x| folder.fold_pat(x)),
                    after.iter().map(|x| folder.fold_pat(*x)).collect())
        }
        PatMac(ref mac) => PatMac(folder.fold_mac(mac)),
    };

    box(GC) Pat {
        id: id,
        span: folder.new_span(p.span),
        node: node,
    }
}

pub fn noop_fold_expr<T: Folder>(e: Gc<Expr>, folder: &mut T) -> Gc<Expr> {
    let id = folder.new_id(e.id);
    let node = match e.node {
        ExprVstore(e, v) => {
            ExprVstore(folder.fold_expr(e), v)
        }
        ExprBox(p, e) => {
            ExprBox(folder.fold_expr(p), folder.fold_expr(e))
        }
        ExprVec(ref exprs) => {
            ExprVec(exprs.iter().map(|&x| folder.fold_expr(x)).collect())
        }
        ExprRepeat(expr, count) => {
            ExprRepeat(folder.fold_expr(expr), folder.fold_expr(count))
        }
        ExprTup(ref elts) => ExprTup(elts.iter().map(|x| folder.fold_expr(*x)).collect()),
        ExprCall(f, ref args) => {
            ExprCall(folder.fold_expr(f),
                     args.iter().map(|&x| folder.fold_expr(x)).collect())
        }
        ExprMethodCall(i, ref tps, ref args) => {
            ExprMethodCall(
                respan(i.span, folder.fold_ident(i.node)),
                tps.iter().map(|&x| folder.fold_ty(x)).collect(),
                args.iter().map(|&x| folder.fold_expr(x)).collect())
        }
        ExprBinary(binop, lhs, rhs) => {
            ExprBinary(binop,
                       folder.fold_expr(lhs),
                       folder.fold_expr(rhs))
        }
        ExprUnary(binop, ohs) => {
            ExprUnary(binop, folder.fold_expr(ohs))
        }
        ExprLit(_) => e.node.clone(),
        ExprCast(expr, ty) => {
            ExprCast(folder.fold_expr(expr), folder.fold_ty(ty))
        }
        ExprAddrOf(m, ohs) => ExprAddrOf(m, folder.fold_expr(ohs)),
        ExprIf(cond, tr, fl) => {
            ExprIf(folder.fold_expr(cond),
                   folder.fold_block(tr),
                   fl.map(|x| folder.fold_expr(x)))
        }
        ExprWhile(cond, body) => {
            ExprWhile(folder.fold_expr(cond), folder.fold_block(body))
        }
        ExprForLoop(pat, iter, body, ref maybe_ident) => {
            ExprForLoop(folder.fold_pat(pat),
                        folder.fold_expr(iter),
                        folder.fold_block(body),
                        maybe_ident.map(|i| folder.fold_ident(i)))
        }
        ExprLoop(body, opt_ident) => {
            ExprLoop(folder.fold_block(body),
                     opt_ident.map(|x| folder.fold_ident(x)))
        }
        ExprMatch(expr, ref arms) => {
            ExprMatch(folder.fold_expr(expr),
                      arms.iter().map(|x| folder.fold_arm(x)).collect())
        }
        ExprFnBlock(ref decl, ref body) => {
            ExprFnBlock(folder.fold_fn_decl(&**decl),
                        folder.fold_block(body.clone()))
        }
        ExprProc(ref decl, ref body) => {
            ExprProc(folder.fold_fn_decl(&**decl),
                     folder.fold_block(body.clone()))
        }
        ExprUnboxedFn(ref decl, ref body) => {
            ExprUnboxedFn(folder.fold_fn_decl(&**decl),
                          folder.fold_block(*body))
        }
        ExprBlock(ref blk) => ExprBlock(folder.fold_block(*blk)),
        ExprAssign(el, er) => {
            ExprAssign(folder.fold_expr(el), folder.fold_expr(er))
        }
        ExprAssignOp(op, el, er) => {
            ExprAssignOp(op,
                         folder.fold_expr(el),
                         folder.fold_expr(er))
        }
        ExprField(el, id, ref tys) => {
            ExprField(folder.fold_expr(el),
                      respan(id.span, folder.fold_ident(id.node)),
                      tys.iter().map(|&x| folder.fold_ty(x)).collect())
        }
        ExprIndex(el, er) => {
            ExprIndex(folder.fold_expr(el), folder.fold_expr(er))
        }
        ExprPath(ref pth) => ExprPath(folder.fold_path(pth)),
        ExprBreak(opt_ident) => ExprBreak(opt_ident.map(|x| folder.fold_ident(x))),
        ExprAgain(opt_ident) => ExprAgain(opt_ident.map(|x| folder.fold_ident(x))),
        ExprRet(ref e) => {
            ExprRet(e.map(|x| folder.fold_expr(x)))
        }
        ExprInlineAsm(ref a) => {
            ExprInlineAsm(InlineAsm {
                inputs: a.inputs.iter().map(|&(ref c, input)| {
                    ((*c).clone(), folder.fold_expr(input))
                }).collect(),
                outputs: a.outputs.iter().map(|&(ref c, out)| {
                    ((*c).clone(), folder.fold_expr(out))
                }).collect(),
                .. (*a).clone()
            })
        }
        ExprMac(ref mac) => ExprMac(folder.fold_mac(mac)),
        ExprStruct(ref path, ref fields, maybe_expr) => {
            ExprStruct(folder.fold_path(path),
                       fields.iter().map(|x| folder.fold_field(*x)).collect(),
                       maybe_expr.map(|x| folder.fold_expr(x)))
        },
        ExprParen(ex) => ExprParen(folder.fold_expr(ex))
    };

    box(GC) Expr {
        id: id,
        node: node,
        span: folder.new_span(e.span),
    }
}

pub fn noop_fold_stmt<T: Folder>(s: &Stmt,
                                 folder: &mut T) -> SmallVector<Gc<Stmt>> {
    let nodes = match s.node {
        StmtDecl(d, id) => {
            let id = folder.new_id(id);
            folder.fold_decl(d).move_iter()
                    .map(|d| StmtDecl(d, id))
                    .collect()
        }
        StmtExpr(e, id) => {
            let id = folder.new_id(id);
            SmallVector::one(StmtExpr(folder.fold_expr(e), id))
        }
        StmtSemi(e, id) => {
            let id = folder.new_id(id);
            SmallVector::one(StmtSemi(folder.fold_expr(e), id))
        }
        StmtMac(ref mac, semi) => SmallVector::one(StmtMac(folder.fold_mac(mac), semi))
    };

    nodes.move_iter().map(|node| box(GC) Spanned {
        node: node,
        span: folder.new_span(s.span),
    }).collect()
}

#[cfg(test)]
mod test {
    use std::io;
    use ast;
    use util::parser_testing::{string_to_crate, matches_codepattern};
    use parse::token;
    use print::pprust;
    use fold;
    use super::*;

    // this version doesn't care about getting comments or docstrings in.
    fn fake_print_crate(s: &mut pprust::State,
                        krate: &ast::Crate) -> io::IoResult<()> {
        s.print_mod(&krate.module, krate.attrs.as_slice())
    }

    // change every identifier to "zz"
    struct ToZzIdentFolder;

    impl Folder for ToZzIdentFolder {
        fn fold_ident(&mut self, _: ast::Ident) -> ast::Ident {
            token::str_to_ident("zz")
        }
        fn fold_mac(&mut self, macro: &ast::Mac) -> ast::Mac {
            fold::noop_fold_mac(macro, self)
        }
    }

    // maybe add to expand.rs...
    macro_rules! assert_pred (
        ($pred:expr, $predname:expr, $a:expr , $b:expr) => (
            {
                let pred_val = $pred;
                let a_val = $a;
                let b_val = $b;
                if !(pred_val(a_val.as_slice(),b_val.as_slice())) {
                    fail!("expected args satisfying {}, got {:?} and {:?}",
                          $predname, a_val, b_val);
                }
            }
        )
    )

    // make sure idents get transformed everywhere
    #[test] fn ident_transformation () {
        let mut zz_fold = ToZzIdentFolder;
        let ast = string_to_crate(
            "#[a] mod b {fn c (d : e, f : g) {h!(i,j,k);l;m}}".to_string());
        let folded_crate = zz_fold.fold_crate(ast);
        assert_pred!(
            matches_codepattern,
            "matches_codepattern",
            pprust::to_string(|s| fake_print_crate(s, &folded_crate)),
            "#[a]mod zz{fn zz(zz:zz,zz:zz){zz!(zz,zz,zz);zz;zz}}".to_string());
    }

    // even inside macro defs....
    #[test] fn ident_transformation_in_defs () {
        let mut zz_fold = ToZzIdentFolder;
        let ast = string_to_crate(
            "macro_rules! a {(b $c:expr $(d $e:token)f+ => \
             (g $(d $d $e)+))} ".to_string());
        let folded_crate = zz_fold.fold_crate(ast);
        assert_pred!(
            matches_codepattern,
            "matches_codepattern",
            pprust::to_string(|s| fake_print_crate(s, &folded_crate)),
            "zz!zz((zz$zz:zz$(zz $zz:zz)zz+=>(zz$(zz$zz$zz)+)))".to_string());
    }
}
