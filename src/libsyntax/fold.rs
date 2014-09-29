// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
use ptr::P;
use owned_slice::OwnedSlice;
use util::small_vector::SmallVector;

use std::rc::Rc;

// This could have a better place to live.
pub trait MoveMap<T> {
    fn move_map(self, f: |T| -> T) -> Self;
}

impl<T> MoveMap<T> for Vec<T> {
    fn move_map(mut self, f: |T| -> T) -> Vec<T> {
        use std::{mem, ptr};
        for p in self.iter_mut() {
            unsafe {
                // FIXME(#5016) this shouldn't need to zero to be safe.
                mem::move_val_init(p, f(ptr::read_and_zero(p)));
            }
        }
        self
    }
}

impl<T> MoveMap<T> for OwnedSlice<T> {
    fn move_map(self, f: |T| -> T) -> OwnedSlice<T> {
        OwnedSlice::from_vec(self.into_vec().move_map(f))
    }
}

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

    fn fold_meta_items(&mut self, meta_items: Vec<P<MetaItem>>) -> Vec<P<MetaItem>> {
        noop_fold_meta_items(meta_items, self)
    }

    fn fold_meta_item(&mut self, meta_item: P<MetaItem>) -> P<MetaItem> {
        noop_fold_meta_item(meta_item, self)
    }

    fn fold_view_path(&mut self, view_path: P<ViewPath>) -> P<ViewPath> {
        noop_fold_view_path(view_path, self)
    }

    fn fold_view_item(&mut self, vi: ViewItem) -> ViewItem {
        noop_fold_view_item(vi, self)
    }

    fn fold_foreign_item(&mut self, ni: P<ForeignItem>) -> P<ForeignItem> {
        noop_fold_foreign_item(ni, self)
    }

    fn fold_item(&mut self, i: P<Item>) -> SmallVector<P<Item>> {
        noop_fold_item(i, self)
    }

    fn fold_item_simple(&mut self, i: Item) -> Item {
        noop_fold_item_simple(i, self)
    }

    fn fold_struct_field(&mut self, sf: StructField) -> StructField {
        noop_fold_struct_field(sf, self)
    }

    fn fold_item_underscore(&mut self, i: Item_) -> Item_ {
        noop_fold_item_underscore(i, self)
    }

    fn fold_fn_decl(&mut self, d: P<FnDecl>) -> P<FnDecl> {
        noop_fold_fn_decl(d, self)
    }

    fn fold_type_method(&mut self, m: TypeMethod) -> TypeMethod {
        noop_fold_type_method(m, self)
    }

    fn fold_method(&mut self, m: P<Method>) -> SmallVector<P<Method>> {
        noop_fold_method(m, self)
    }

    fn fold_block(&mut self, b: P<Block>) -> P<Block> {
        noop_fold_block(b, self)
    }

    fn fold_stmt(&mut self, s: P<Stmt>) -> SmallVector<P<Stmt>> {
        s.and_then(|s| noop_fold_stmt(s, self))
    }

    fn fold_arm(&mut self, a: Arm) -> Arm {
        noop_fold_arm(a, self)
    }

    fn fold_pat(&mut self, p: P<Pat>) -> P<Pat> {
        noop_fold_pat(p, self)
    }

    fn fold_decl(&mut self, d: P<Decl>) -> SmallVector<P<Decl>> {
        noop_fold_decl(d, self)
    }

    fn fold_expr(&mut self, e: P<Expr>) -> P<Expr> {
        e.map(|e| noop_fold_expr(e, self))
    }

    fn fold_ty(&mut self, t: P<Ty>) -> P<Ty> {
        noop_fold_ty(t, self)
    }

    fn fold_mod(&mut self, m: Mod) -> Mod {
        noop_fold_mod(m, self)
    }

    fn fold_foreign_mod(&mut self, nm: ForeignMod) -> ForeignMod {
        noop_fold_foreign_mod(nm, self)
    }

    fn fold_variant(&mut self, v: P<Variant>) -> P<Variant> {
        noop_fold_variant(v, self)
    }

    fn fold_ident(&mut self, i: Ident) -> Ident {
        noop_fold_ident(i, self)
    }

    fn fold_uint(&mut self, i: uint) -> uint {
        noop_fold_uint(i, self)
    }

    fn fold_path(&mut self, p: Path) -> Path {
        noop_fold_path(p, self)
    }

    fn fold_local(&mut self, l: P<Local>) -> P<Local> {
        noop_fold_local(l, self)
    }

    fn fold_mac(&mut self, _macro: Mac) -> Mac {
        fail!("fold_mac disabled by default");
        // NB: see note about macros above.
        // if you really want a folder that
        // works on macros, use this
        // definition in your trait impl:
        // fold::noop_fold_mac(_macro, self)
    }

    fn fold_explicit_self(&mut self, es: ExplicitSelf) -> ExplicitSelf {
        noop_fold_explicit_self(es, self)
    }

    fn fold_explicit_self_underscore(&mut self, es: ExplicitSelf_) -> ExplicitSelf_ {
        noop_fold_explicit_self_underscore(es, self)
    }

    fn fold_lifetime(&mut self, l: Lifetime) -> Lifetime {
        noop_fold_lifetime(l, self)
    }

    fn fold_lifetime_def(&mut self, l: LifetimeDef) -> LifetimeDef {
        noop_fold_lifetime_def(l, self)
    }

    fn fold_attribute(&mut self, at: Attribute) -> Attribute {
        noop_fold_attribute(at, self)
    }

    fn fold_arg(&mut self, a: Arg) -> Arg {
        noop_fold_arg(a, self)
    }

    fn fold_generics(&mut self, generics: Generics) -> Generics {
        noop_fold_generics(generics, self)
    }

    fn fold_trait_ref(&mut self, p: TraitRef) -> TraitRef {
        noop_fold_trait_ref(p, self)
    }

    fn fold_struct_def(&mut self, struct_def: P<StructDef>) -> P<StructDef> {
        noop_fold_struct_def(struct_def, self)
    }

    fn fold_lifetimes(&mut self, lts: Vec<Lifetime>) -> Vec<Lifetime> {
        noop_fold_lifetimes(lts, self)
    }

    fn fold_lifetime_defs(&mut self, lts: Vec<LifetimeDef>) -> Vec<LifetimeDef> {
        noop_fold_lifetime_defs(lts, self)
    }

    fn fold_ty_param(&mut self, tp: TyParam) -> TyParam {
        noop_fold_ty_param(tp, self)
    }

    fn fold_ty_params(&mut self, tps: OwnedSlice<TyParam>) -> OwnedSlice<TyParam> {
        noop_fold_ty_params(tps, self)
    }

    fn fold_tt(&mut self, tt: &TokenTree) -> TokenTree {
        noop_fold_tt(tt, self)
    }

    fn fold_tts(&mut self, tts: &[TokenTree]) -> Vec<TokenTree> {
        noop_fold_tts(tts, self)
    }

    fn fold_token(&mut self, t: token::Token) -> token::Token {
        noop_fold_token(t, self)
    }

    fn fold_interpolated(&mut self, nt: token::Nonterminal) -> token::Nonterminal {
        noop_fold_interpolated(nt, self)
    }

    fn fold_opt_lifetime(&mut self, o_lt: Option<Lifetime>) -> Option<Lifetime> {
        noop_fold_opt_lifetime(o_lt, self)
    }

    fn fold_variant_arg(&mut self, va: VariantArg) -> VariantArg {
        noop_fold_variant_arg(va, self)
    }

    fn fold_opt_bounds(&mut self, b: Option<OwnedSlice<TyParamBound>>)
                       -> Option<OwnedSlice<TyParamBound>> {
        noop_fold_opt_bounds(b, self)
    }

    fn fold_bounds(&mut self, b: OwnedSlice<TyParamBound>)
                       -> OwnedSlice<TyParamBound> {
        noop_fold_bounds(b, self)
    }

    fn fold_ty_param_bound(&mut self, tpb: TyParamBound) -> TyParamBound {
        noop_fold_ty_param_bound(tpb, self)
    }

    fn fold_mt(&mut self, mt: MutTy) -> MutTy {
        noop_fold_mt(mt, self)
    }

    fn fold_field(&mut self, field: Field) -> Field {
        noop_fold_field(field, self)
    }

    fn fold_where_clause(&mut self, where_clause: WhereClause)
                         -> WhereClause {
        noop_fold_where_clause(where_clause, self)
    }

    fn fold_where_predicate(&mut self, where_predicate: WherePredicate)
                            -> WherePredicate {
        noop_fold_where_predicate(where_predicate, self)
    }

    fn fold_typedef(&mut self, typedef: Typedef) -> Typedef {
        noop_fold_typedef(typedef, self)
    }

    fn fold_associated_type(&mut self, associated_type: AssociatedType)
                            -> AssociatedType {
        noop_fold_associated_type(associated_type, self)
    }

    fn new_id(&mut self, i: NodeId) -> NodeId {
        i
    }

    fn new_span(&mut self, sp: Span) -> Span {
        sp
    }
}

pub fn noop_fold_meta_items<T: Folder>(meta_items: Vec<P<MetaItem>>, fld: &mut T)
                                       -> Vec<P<MetaItem>> {
    meta_items.move_map(|x| fld.fold_meta_item(x))
}

pub fn noop_fold_view_path<T: Folder>(view_path: P<ViewPath>, fld: &mut T) -> P<ViewPath> {
    view_path.map(|Spanned {node, span}| Spanned {
        node: match node {
            ViewPathSimple(ident, path, node_id) => {
                let id = fld.new_id(node_id);
                ViewPathSimple(ident, fld.fold_path(path), id)
            }
            ViewPathGlob(path, node_id) => {
                let id = fld.new_id(node_id);
                ViewPathGlob(fld.fold_path(path), id)
            }
            ViewPathList(path, path_list_idents, node_id) => {
                let id = fld.new_id(node_id);
                ViewPathList(fld.fold_path(path),
                             path_list_idents.move_map(|path_list_ident| {
                                Spanned {
                                    node: match path_list_ident.node {
                                        PathListIdent { id, name } =>
                                            PathListIdent {
                                                id: fld.new_id(id),
                                                name: name
                                            },
                                        PathListMod { id } =>
                                            PathListMod { id: fld.new_id(id) }
                                    },
                                    span: fld.new_span(path_list_ident.span)
                                }
                             }),
                             id)
            }
        },
        span: fld.new_span(span)
    })
}

pub fn noop_fold_arm<T: Folder>(Arm {attrs, pats, guard, body}: Arm, fld: &mut T) -> Arm {
    Arm {
        attrs: attrs.move_map(|x| fld.fold_attribute(x)),
        pats: pats.move_map(|x| fld.fold_pat(x)),
        guard: guard.map(|x| fld.fold_expr(x)),
        body: fld.fold_expr(body),
    }
}

pub fn noop_fold_decl<T: Folder>(d: P<Decl>, fld: &mut T) -> SmallVector<P<Decl>> {
    d.and_then(|Spanned {node, span}| match node {
        DeclLocal(l) => SmallVector::one(P(Spanned {
            node: DeclLocal(fld.fold_local(l)),
            span: fld.new_span(span)
        })),
        DeclItem(it) => fld.fold_item(it).into_iter().map(|i| P(Spanned {
            node: DeclItem(i),
            span: fld.new_span(span)
        })).collect()
    })
}

pub fn noop_fold_ty<T: Folder>(t: P<Ty>, fld: &mut T) -> P<Ty> {
    t.map(|Ty {id, node, span}| Ty {
        id: fld.new_id(id),
        node: match node {
            TyNil | TyBot | TyInfer => node,
            TyBox(ty) => TyBox(fld.fold_ty(ty)),
            TyUniq(ty) => TyUniq(fld.fold_ty(ty)),
            TyVec(ty) => TyVec(fld.fold_ty(ty)),
            TyPtr(mt) => TyPtr(fld.fold_mt(mt)),
            TyRptr(region, mt) => {
                TyRptr(fld.fold_opt_lifetime(region), fld.fold_mt(mt))
            }
            TyClosure(f) => {
                TyClosure(f.map(|ClosureTy {fn_style, onceness, bounds, decl, lifetimes}| {
                    ClosureTy {
                        fn_style: fn_style,
                        onceness: onceness,
                        bounds: fld.fold_bounds(bounds),
                        decl: fld.fold_fn_decl(decl),
                        lifetimes: fld.fold_lifetime_defs(lifetimes)
                    }
                }))
            }
            TyProc(f) => {
                TyProc(f.map(|ClosureTy {fn_style, onceness, bounds, decl, lifetimes}| {
                    ClosureTy {
                        fn_style: fn_style,
                        onceness: onceness,
                        bounds: fld.fold_bounds(bounds),
                        decl: fld.fold_fn_decl(decl),
                        lifetimes: fld.fold_lifetime_defs(lifetimes)
                    }
                }))
            }
            TyBareFn(f) => {
                TyBareFn(f.map(|BareFnTy {lifetimes, fn_style, abi, decl}| BareFnTy {
                    lifetimes: fld.fold_lifetime_defs(lifetimes),
                    fn_style: fn_style,
                    abi: abi,
                    decl: fld.fold_fn_decl(decl)
                }))
            }
            TyUnboxedFn(f) => {
                TyUnboxedFn(f.map(|UnboxedFnTy {decl, kind}| UnboxedFnTy {
                    decl: fld.fold_fn_decl(decl),
                    kind: kind,
                }))
            }
            TyTup(tys) => TyTup(tys.move_map(|ty| fld.fold_ty(ty))),
            TyParen(ty) => TyParen(fld.fold_ty(ty)),
            TyPath(path, bounds, id) => {
                let id = fld.new_id(id);
                TyPath(fld.fold_path(path),
                        fld.fold_opt_bounds(bounds),
                        id)
            }
            TyQPath(ref qpath) => {
                TyQPath(P(QPath {
                    for_type: fld.fold_ty(qpath.for_type.clone()),
                    trait_name: fld.fold_path(qpath.trait_name.clone()),
                    item_name: fld.fold_ident(qpath.item_name.clone()),
                }))
            }
            TyFixedLengthVec(ty, e) => {
                TyFixedLengthVec(fld.fold_ty(ty), fld.fold_expr(e))
            }
            TyTypeof(expr) => TyTypeof(fld.fold_expr(expr))
        },
        span: fld.new_span(span)
    })
}

pub fn noop_fold_foreign_mod<T: Folder>(ForeignMod {abi, view_items, items}: ForeignMod,
                                        fld: &mut T) -> ForeignMod {
    ForeignMod {
        abi: abi,
        view_items: view_items.move_map(|x| fld.fold_view_item(x)),
        items: items.move_map(|x| fld.fold_foreign_item(x)),
    }
}

pub fn noop_fold_variant<T: Folder>(v: P<Variant>, fld: &mut T) -> P<Variant> {
    v.map(|Spanned {node: Variant_ {id, name, attrs, kind, disr_expr, vis}, span}| Spanned {
        node: Variant_ {
            id: fld.new_id(id),
            name: name,
            attrs: attrs.move_map(|x| fld.fold_attribute(x)),
            kind: match kind {
                TupleVariantKind(variant_args) => {
                    TupleVariantKind(variant_args.move_map(|x|
                        fld.fold_variant_arg(x)))
                }
                StructVariantKind(struct_def) => {
                    StructVariantKind(fld.fold_struct_def(struct_def))
                }
            },
            disr_expr: disr_expr.map(|e| fld.fold_expr(e)),
            vis: vis,
        },
        span: fld.new_span(span),
    })
}

pub fn noop_fold_ident<T: Folder>(i: Ident, _: &mut T) -> Ident {
    i
}

pub fn noop_fold_uint<T: Folder>(i: uint, _: &mut T) -> uint {
    i
}

pub fn noop_fold_path<T: Folder>(Path {global, segments, span}: Path, fld: &mut T) -> Path {
    Path {
        global: global,
        segments: segments.move_map(|PathSegment {identifier, lifetimes, types}| PathSegment {
            identifier: fld.fold_ident(identifier),
            lifetimes: fld.fold_lifetimes(lifetimes),
            types: types.move_map(|typ| fld.fold_ty(typ)),
        }),
        span: fld.new_span(span)
    }
}

pub fn noop_fold_local<T: Folder>(l: P<Local>, fld: &mut T) -> P<Local> {
    l.map(|Local {id, pat, ty, init, source, span}| Local {
        id: fld.new_id(id),
        ty: fld.fold_ty(ty),
        pat: fld.fold_pat(pat),
        init: init.map(|e| fld.fold_expr(e)),
        source: source,
        span: fld.new_span(span)
    })
}

pub fn noop_fold_attribute<T: Folder>(at: Attribute, fld: &mut T) -> Attribute {
    let Spanned {node: Attribute_ {id, style, value, is_sugared_doc}, span} = at;
    Spanned {
        node: Attribute_ {
            id: id,
            style: style,
            value: fld.fold_meta_item(value),
            is_sugared_doc: is_sugared_doc
        },
        span: fld.new_span(span)
    }
}

pub fn noop_fold_explicit_self_underscore<T: Folder>(es: ExplicitSelf_, fld: &mut T)
                                                     -> ExplicitSelf_ {
    match es {
        SelfStatic | SelfValue(_) => es,
        SelfRegion(lifetime, m, ident) => {
            SelfRegion(fld.fold_opt_lifetime(lifetime), m, ident)
        }
        SelfExplicit(typ, ident) => {
            SelfExplicit(fld.fold_ty(typ), ident)
        }
    }
}

pub fn noop_fold_explicit_self<T: Folder>(Spanned {span, node}: ExplicitSelf, fld: &mut T)
                                          -> ExplicitSelf {
    Spanned {
        node: fld.fold_explicit_self_underscore(node),
        span: fld.new_span(span)
    }
}


pub fn noop_fold_mac<T: Folder>(Spanned {node, span}: Mac, fld: &mut T) -> Mac {
    Spanned {
        node: match node {
            MacInvocTT(p, tts, ctxt) => {
                MacInvocTT(fld.fold_path(p), fld.fold_tts(tts.as_slice()), ctxt)
            }
        },
        span: fld.new_span(span)
    }
}

pub fn noop_fold_meta_item<T: Folder>(mi: P<MetaItem>, fld: &mut T) -> P<MetaItem> {
    mi.map(|Spanned {node, span}| Spanned {
        node: match node {
            MetaWord(id) => MetaWord(id),
            MetaList(id, mis) => {
                MetaList(id, mis.move_map(|e| fld.fold_meta_item(e)))
            }
            MetaNameValue(id, s) => MetaNameValue(id, s)
        },
        span: fld.new_span(span)
    })
}

pub fn noop_fold_arg<T: Folder>(Arg {id, pat, ty}: Arg, fld: &mut T) -> Arg {
    Arg {
        id: fld.new_id(id),
        pat: fld.fold_pat(pat),
        ty: fld.fold_ty(ty)
    }
}

pub fn noop_fold_tt<T: Folder>(tt: &TokenTree, fld: &mut T) -> TokenTree {
    match *tt {
        TTTok(span, ref tok) =>
            TTTok(span, fld.fold_token(tok.clone())),
        TTDelim(ref tts) => TTDelim(Rc::new(fld.fold_tts(tts.as_slice()))),
        TTSeq(span, ref pattern, ref sep, is_optional) =>
            TTSeq(span,
                  Rc::new(fld.fold_tts(pattern.as_slice())),
                  sep.clone().map(|tok| fld.fold_token(tok)),
                  is_optional),
        TTNonterminal(sp,ref ident) =>
            TTNonterminal(sp,fld.fold_ident(*ident))
    }
}

pub fn noop_fold_tts<T: Folder>(tts: &[TokenTree], fld: &mut T) -> Vec<TokenTree> {
    tts.iter().map(|tt| fld.fold_tt(tt)).collect()
}

// apply ident folder if it's an ident, apply other folds to interpolated nodes
pub fn noop_fold_token<T: Folder>(t: token::Token, fld: &mut T) -> token::Token {
    match t {
        token::IDENT(id, followed_by_colons) => {
            token::IDENT(fld.fold_ident(id), followed_by_colons)
        }
        token::LIFETIME(id) => token::LIFETIME(fld.fold_ident(id)),
        token::INTERPOLATED(nt) => token::INTERPOLATED(fld.fold_interpolated(nt)),
        _ => t
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
pub fn noop_fold_interpolated<T: Folder>(nt: token::Nonterminal, fld: &mut T)
                                         -> token::Nonterminal {
    match nt {
        token::NtItem(item) =>
            token::NtItem(fld.fold_item(item)
                          // this is probably okay, because the only folds likely
                          // to peek inside interpolated nodes will be renamings/markings,
                          // which map single items to single items
                          .expect_one("expected fold to produce exactly one item")),
        token::NtBlock(block) => token::NtBlock(fld.fold_block(block)),
        token::NtStmt(stmt) =>
            token::NtStmt(fld.fold_stmt(stmt)
                          // this is probably okay, because the only folds likely
                          // to peek inside interpolated nodes will be renamings/markings,
                          // which map single items to single items
                          .expect_one("expected fold to produce exactly one statement")),
        token::NtPat(pat) => token::NtPat(fld.fold_pat(pat)),
        token::NtExpr(expr) => token::NtExpr(fld.fold_expr(expr)),
        token::NtTy(ty) => token::NtTy(fld.fold_ty(ty)),
        token::NtIdent(box id, is_mod_name) =>
            token::NtIdent(box fld.fold_ident(id), is_mod_name),
        token::NtMeta(meta_item) => token::NtMeta(fld.fold_meta_item(meta_item)),
        token::NtPath(box path) => token::NtPath(box fld.fold_path(path)),
        token::NtTT(tt) => token::NtTT(P(fld.fold_tt(&*tt))),
        // it looks to me like we can leave out the matchers: token::NtMatchers(matchers)
        _ => nt
    }
}

pub fn noop_fold_fn_decl<T: Folder>(decl: P<FnDecl>, fld: &mut T) -> P<FnDecl> {
    decl.map(|FnDecl {inputs, output, cf, variadic}| FnDecl {
        inputs: inputs.move_map(|x| fld.fold_arg(x)),
        output: fld.fold_ty(output),
        cf: cf,
        variadic: variadic
    })
}

pub fn noop_fold_ty_param_bound<T>(tpb: TyParamBound, fld: &mut T)
                                   -> TyParamBound
                                   where T: Folder {
    match tpb {
        TraitTyParamBound(ty) => TraitTyParamBound(fld.fold_trait_ref(ty)),
        RegionTyParamBound(lifetime) => RegionTyParamBound(fld.fold_lifetime(lifetime)),
        UnboxedFnTyParamBound(bound) => {
            match *bound {
                UnboxedFnBound {
                    ref path,
                    ref decl,
                    ref lifetimes,
                    ref_id
                } => {
                    UnboxedFnTyParamBound(P(UnboxedFnBound {
                        path: fld.fold_path(path.clone()),
                        decl: fld.fold_fn_decl(decl.clone()),
                        lifetimes: fld.fold_lifetime_defs(lifetimes.clone()),
                        ref_id: fld.new_id(ref_id),
                    }))
                }
            }
        }
    }
}

pub fn noop_fold_ty_param<T: Folder>(tp: TyParam, fld: &mut T) -> TyParam {
    let TyParam {id, ident, bounds, unbound, default, span} = tp;
    TyParam {
        id: fld.new_id(id),
        ident: ident,
        bounds: fld.fold_bounds(bounds),
        unbound: unbound.map(|x| fld.fold_ty_param_bound(x)),
        default: default.map(|x| fld.fold_ty(x)),
        span: span
    }
}

pub fn noop_fold_ty_params<T: Folder>(tps: OwnedSlice<TyParam>, fld: &mut T)
                                      -> OwnedSlice<TyParam> {
    tps.move_map(|tp| fld.fold_ty_param(tp))
}

pub fn noop_fold_lifetime<T: Folder>(l: Lifetime, fld: &mut T) -> Lifetime {
    Lifetime {
        id: fld.new_id(l.id),
        name: l.name,
        span: fld.new_span(l.span)
    }
}

pub fn noop_fold_lifetime_def<T: Folder>(l: LifetimeDef, fld: &mut T)
                                         -> LifetimeDef {
    LifetimeDef {
        lifetime: fld.fold_lifetime(l.lifetime),
        bounds: fld.fold_lifetimes(l.bounds),
    }
}

pub fn noop_fold_lifetimes<T: Folder>(lts: Vec<Lifetime>, fld: &mut T) -> Vec<Lifetime> {
    lts.move_map(|l| fld.fold_lifetime(l))
}

pub fn noop_fold_lifetime_defs<T: Folder>(lts: Vec<LifetimeDef>, fld: &mut T)
                                          -> Vec<LifetimeDef> {
    lts.move_map(|l| fld.fold_lifetime_def(l))
}

pub fn noop_fold_opt_lifetime<T: Folder>(o_lt: Option<Lifetime>, fld: &mut T)
                                         -> Option<Lifetime> {
    o_lt.map(|lt| fld.fold_lifetime(lt))
}

pub fn noop_fold_generics<T: Folder>(Generics {ty_params, lifetimes, where_clause}: Generics,
                                     fld: &mut T) -> Generics {
    Generics {
        ty_params: fld.fold_ty_params(ty_params),
        lifetimes: fld.fold_lifetime_defs(lifetimes),
        where_clause: fld.fold_where_clause(where_clause),
    }
}

pub fn noop_fold_where_clause<T: Folder>(
                              WhereClause {id, predicates}: WhereClause,
                              fld: &mut T)
                              -> WhereClause {
    WhereClause {
        id: fld.new_id(id),
        predicates: predicates.move_map(|predicate| {
            fld.fold_where_predicate(predicate)
        })
    }
}

pub fn noop_fold_where_predicate<T: Folder>(
                                 WherePredicate {id, ident, bounds, span}: WherePredicate,
                                 fld: &mut T)
                                 -> WherePredicate {
    WherePredicate {
        id: fld.new_id(id),
        ident: fld.fold_ident(ident),
        bounds: bounds.move_map(|x| fld.fold_ty_param_bound(x)),
        span: fld.new_span(span)
    }
}

pub fn noop_fold_typedef<T>(t: Typedef, folder: &mut T)
                            -> Typedef
                            where T: Folder {
    let new_id = folder.new_id(t.id);
    let new_span = folder.new_span(t.span);
    let new_attrs = t.attrs.iter().map(|attr| {
        folder.fold_attribute((*attr).clone())
    }).collect();
    let new_ident = folder.fold_ident(t.ident);
    let new_type = folder.fold_ty(t.typ);
    ast::Typedef {
        ident: new_ident,
        typ: new_type,
        id: new_id,
        span: new_span,
        vis: t.vis,
        attrs: new_attrs,
    }
}

pub fn noop_fold_associated_type<T>(at: AssociatedType, folder: &mut T)
                                    -> AssociatedType
                                    where T: Folder {
    let new_id = folder.new_id(at.id);
    let new_span = folder.new_span(at.span);
    let new_ident = folder.fold_ident(at.ident);
    let new_attrs = at.attrs
                      .iter()
                      .map(|attr| folder.fold_attribute((*attr).clone()))
                      .collect();
    ast::AssociatedType {
        ident: new_ident,
        attrs: new_attrs,
        id: new_id,
        span: new_span,
    }
}

pub fn noop_fold_struct_def<T: Folder>(struct_def: P<StructDef>, fld: &mut T) -> P<StructDef> {
    struct_def.map(|StructDef {fields, ctor_id, super_struct, is_virtual}| StructDef {
        fields: fields.move_map(|f| fld.fold_struct_field(f)),
        ctor_id: ctor_id.map(|cid| fld.new_id(cid)),
        super_struct: super_struct.map(|t| fld.fold_ty(t)),
        is_virtual: is_virtual
    })
}

pub fn noop_fold_trait_ref<T: Folder>(p: TraitRef, fld: &mut T) -> TraitRef {
    let id = fld.new_id(p.ref_id);
    let TraitRef {
        path,
        lifetimes,
        ..
    } = p;
    ast::TraitRef {
        path: fld.fold_path(path),
        ref_id: id,
        lifetimes: fld.fold_lifetime_defs(lifetimes),
    }
}

pub fn noop_fold_struct_field<T: Folder>(f: StructField, fld: &mut T) -> StructField {
    let StructField {node: StructField_ {id, kind, ty, attrs}, span} = f;
    Spanned {
        node: StructField_ {
            id: fld.new_id(id),
            kind: kind,
            ty: fld.fold_ty(ty),
            attrs: attrs.move_map(|a| fld.fold_attribute(a))
        },
        span: fld.new_span(span)
    }
}

pub fn noop_fold_field<T: Folder>(Field {ident, expr, span}: Field, folder: &mut T) -> Field {
    Field {
        ident: respan(ident.span, folder.fold_ident(ident.node)),
        expr: folder.fold_expr(expr),
        span: folder.new_span(span)
    }
}

pub fn noop_fold_mt<T: Folder>(MutTy {ty, mutbl}: MutTy, folder: &mut T) -> MutTy {
    MutTy {
        ty: folder.fold_ty(ty),
        mutbl: mutbl,
    }
}

pub fn noop_fold_opt_bounds<T: Folder>(b: Option<OwnedSlice<TyParamBound>>, folder: &mut T)
                                       -> Option<OwnedSlice<TyParamBound>> {
    b.map(|bounds| folder.fold_bounds(bounds))
}

fn noop_fold_bounds<T: Folder>(bounds: TyParamBounds, folder: &mut T)
                          -> TyParamBounds {
    bounds.move_map(|bound| folder.fold_ty_param_bound(bound))
}

fn noop_fold_variant_arg<T: Folder>(VariantArg {id, ty}: VariantArg, folder: &mut T)
                                    -> VariantArg {
    VariantArg {
        id: folder.new_id(id),
        ty: folder.fold_ty(ty)
    }
}

pub fn noop_fold_view_item<T: Folder>(ViewItem {node, attrs, vis, span}: ViewItem,
                                      folder: &mut T) -> ViewItem {
    ViewItem {
        node: match node {
            ViewItemExternCrate(ident, string, node_id) => {
                ViewItemExternCrate(ident, string,
                                    folder.new_id(node_id))
            }
            ViewItemUse(view_path) => {
                ViewItemUse(folder.fold_view_path(view_path))
            }
        },
        attrs: attrs.move_map(|a| folder.fold_attribute(a)),
        vis: vis,
        span: folder.new_span(span)
    }
}

pub fn noop_fold_block<T: Folder>(b: P<Block>, folder: &mut T) -> P<Block> {
    b.map(|Block {id, view_items, stmts, expr, rules, span}| Block {
        id: folder.new_id(id),
        view_items: view_items.move_map(|x| folder.fold_view_item(x)),
        stmts: stmts.into_iter().flat_map(|s| folder.fold_stmt(s).into_iter()).collect(),
        expr: expr.map(|x| folder.fold_expr(x)),
        rules: rules,
        span: folder.new_span(span),
    })
}

pub fn noop_fold_item_underscore<T: Folder>(i: Item_, folder: &mut T) -> Item_ {
    match i {
        ItemStatic(t, m, e) => {
            ItemStatic(folder.fold_ty(t), m, folder.fold_expr(e))
        }
        ItemFn(decl, fn_style, abi, generics, body) => {
            ItemFn(
                folder.fold_fn_decl(decl),
                fn_style,
                abi,
                folder.fold_generics(generics),
                folder.fold_block(body)
            )
        }
        ItemMod(m) => ItemMod(folder.fold_mod(m)),
        ItemForeignMod(nm) => ItemForeignMod(folder.fold_foreign_mod(nm)),
        ItemTy(t, generics) => {
            ItemTy(folder.fold_ty(t), folder.fold_generics(generics))
        }
        ItemEnum(enum_definition, generics) => {
            ItemEnum(
                ast::EnumDef {
                    variants: enum_definition.variants.move_map(|x| folder.fold_variant(x)),
                },
                folder.fold_generics(generics))
        }
        ItemStruct(struct_def, generics) => {
            let struct_def = folder.fold_struct_def(struct_def);
            ItemStruct(struct_def, folder.fold_generics(generics))
        }
        ItemImpl(generics, ifce, ty, impl_items) => {
            let mut new_impl_items = Vec::new();
            for impl_item in impl_items.iter() {
                match *impl_item {
                    MethodImplItem(ref x) => {
                        for method in folder.fold_method((*x).clone())
                                            .move_iter() {
                            new_impl_items.push(MethodImplItem(method))
                        }
                    }
                    TypeImplItem(ref t) => {
                        new_impl_items.push(TypeImplItem(
                                P(folder.fold_typedef((**t).clone()))));
                    }
                }
            }
            let ifce = match ifce {
                None => None,
                Some(ref trait_ref) => {
                    Some(folder.fold_trait_ref((*trait_ref).clone()))
                }
            };
            ItemImpl(folder.fold_generics(generics),
                     ifce,
                     folder.fold_ty(ty),
                     new_impl_items)
        }
        ItemTrait(generics, unbound, bounds, methods) => {
            let bounds = folder.fold_bounds(bounds);
            let methods = methods.into_iter().flat_map(|method| {
                let r = match method {
                    RequiredMethod(m) => {
                            SmallVector::one(RequiredMethod(
                                    folder.fold_type_method(m)))
                                .move_iter()
                    }
                    ProvidedMethod(method) => {
                        // the awkward collect/iter idiom here is because
                        // even though an iter and a map satisfy the same
                        // trait bound, they're not actually the same type, so
                        // the method arms don't unify.
                        let methods: SmallVector<ast::TraitItem> =
                            folder.fold_method(method).move_iter()
                            .map(|m| ProvidedMethod(m)).collect();
                        methods.move_iter()
                    }
                    TypeTraitItem(at) => {
                        SmallVector::one(TypeTraitItem(P(
                                    folder.fold_associated_type(
                                        (*at).clone()))))
                            .move_iter()
                    }
                };
                r
            }).collect();
            ItemTrait(folder.fold_generics(generics),
                      unbound,
                      bounds,
                      methods)
        }
        ItemMac(m) => ItemMac(folder.fold_mac(m)),
    }
}

pub fn noop_fold_type_method<T: Folder>(m: TypeMethod, fld: &mut T) -> TypeMethod {
    let TypeMethod {
        id,
        ident,
        attrs,
        fn_style,
        abi,
        decl,
        generics,
        explicit_self,
        vis,
        span
    } = m;
    TypeMethod {
        id: fld.new_id(id),
        ident: fld.fold_ident(ident),
        attrs: attrs.move_map(|a| fld.fold_attribute(a)),
        fn_style: fn_style,
        abi: abi,
        decl: fld.fold_fn_decl(decl),
        generics: fld.fold_generics(generics),
        explicit_self: fld.fold_explicit_self(explicit_self),
        vis: vis,
        span: fld.new_span(span)
    }
}

pub fn noop_fold_mod<T: Folder>(Mod {inner, view_items, items}: Mod, folder: &mut T) -> Mod {
    Mod {
        inner: folder.new_span(inner),
        view_items: view_items.move_map(|x| folder.fold_view_item(x)),
        items: items.into_iter().flat_map(|x| folder.fold_item(x).into_iter()).collect(),
    }
}

pub fn noop_fold_crate<T: Folder>(Crate {module, attrs, config, exported_macros, span}: Crate,
                                  folder: &mut T) -> Crate {
    Crate {
        module: folder.fold_mod(module),
        attrs: attrs.move_map(|x| folder.fold_attribute(x)),
        config: folder.fold_meta_items(config),
        exported_macros: exported_macros,
        span: folder.new_span(span)
    }
}

// fold one item into possibly many items
pub fn noop_fold_item<T: Folder>(i: P<Item>, folder: &mut T) -> SmallVector<P<Item>> {
    SmallVector::one(i.map(|i| folder.fold_item_simple(i)))
}

// fold one item into exactly one item
pub fn noop_fold_item_simple<T: Folder>(Item {id, ident, attrs, node, vis, span}: Item,
                                        folder: &mut T) -> Item {
    let id = folder.new_id(id);
    let node = folder.fold_item_underscore(node);
    let ident = match node {
        // The node may have changed, recompute the "pretty" impl name.
        ItemImpl(_, ref maybe_trait, ref ty, _) => {
            ast_util::impl_pretty_name(maybe_trait, &**ty)
        }
        _ => ident
    };

    Item {
        id: id,
        ident: folder.fold_ident(ident),
        attrs: attrs.move_map(|e| folder.fold_attribute(e)),
        node: node,
        vis: vis,
        span: folder.new_span(span)
    }
}

pub fn noop_fold_foreign_item<T: Folder>(ni: P<ForeignItem>, folder: &mut T) -> P<ForeignItem> {
    ni.map(|ForeignItem {id, ident, attrs, node, span, vis}| ForeignItem {
        id: folder.new_id(id),
        ident: folder.fold_ident(ident),
        attrs: attrs.move_map(|x| folder.fold_attribute(x)),
        node: match node {
            ForeignItemFn(fdec, generics) => {
                ForeignItemFn(fdec.map(|FnDecl {inputs, output, cf, variadic}| FnDecl {
                    inputs: inputs.move_map(|a| folder.fold_arg(a)),
                    output: folder.fold_ty(output),
                    cf: cf,
                    variadic: variadic
                }), folder.fold_generics(generics))
            }
            ForeignItemStatic(t, m) => {
                ForeignItemStatic(folder.fold_ty(t), m)
            }
        },
        vis: vis,
        span: folder.new_span(span)
    })
}

// Default fold over a method.
// Invariant: produces exactly one method.
pub fn noop_fold_method<T: Folder>(m: P<Method>, folder: &mut T) -> SmallVector<P<Method>> {
    SmallVector::one(m.map(|Method {id, attrs, node, span}| Method {
        id: folder.new_id(id),
        attrs: attrs.move_map(|a| folder.fold_attribute(a)),
        node: match node {
            MethDecl(ident,
                     generics,
                     abi,
                     explicit_self,
                     fn_style,
                     decl,
                     body,
                     vis) => {
                MethDecl(folder.fold_ident(ident),
                         folder.fold_generics(generics),
                         abi,
                         folder.fold_explicit_self(explicit_self),
                         fn_style,
                         folder.fold_fn_decl(decl),
                         folder.fold_block(body),
                         vis)
            },
            MethMac(mac) => MethMac(folder.fold_mac(mac)),
        },
        span: folder.new_span(span)
    }))
}

pub fn noop_fold_pat<T: Folder>(p: P<Pat>, folder: &mut T) -> P<Pat> {
    p.map(|Pat {id, node, span}| Pat {
        id: folder.new_id(id),
        node: match node {
            PatWild(k) => PatWild(k),
            PatIdent(binding_mode, pth1, sub) => {
                PatIdent(binding_mode,
                        Spanned{span: folder.new_span(pth1.span),
                                node: folder.fold_ident(pth1.node)},
                        sub.map(|x| folder.fold_pat(x)))
            }
            PatLit(e) => PatLit(folder.fold_expr(e)),
            PatEnum(pth, pats) => {
                PatEnum(folder.fold_path(pth),
                        pats.map(|pats| pats.move_map(|x| folder.fold_pat(x))))
            }
            PatStruct(pth, fields, etc) => {
                let pth = folder.fold_path(pth);
                let fs = fields.move_map(|f| {
                    ast::FieldPat {
                        ident: f.ident,
                        pat: folder.fold_pat(f.pat)
                    }
                });
                PatStruct(pth, fs, etc)
            }
            PatTup(elts) => PatTup(elts.move_map(|x| folder.fold_pat(x))),
            PatBox(inner) => PatBox(folder.fold_pat(inner)),
            PatRegion(inner) => PatRegion(folder.fold_pat(inner)),
            PatRange(e1, e2) => {
                PatRange(folder.fold_expr(e1), folder.fold_expr(e2))
            },
            PatVec(before, slice, after) => {
                PatVec(before.move_map(|x| folder.fold_pat(x)),
                       slice.map(|x| folder.fold_pat(x)),
                       after.move_map(|x| folder.fold_pat(x)))
            }
            PatMac(mac) => PatMac(folder.fold_mac(mac))
        },
        span: folder.new_span(span)
    })
}

pub fn noop_fold_expr<T: Folder>(Expr {id, node, span}: Expr, folder: &mut T) -> Expr {
    Expr {
        id: folder.new_id(id),
        node: match node {
            ExprBox(p, e) => {
                ExprBox(folder.fold_expr(p), folder.fold_expr(e))
            }
            ExprVec(exprs) => {
                ExprVec(exprs.move_map(|x| folder.fold_expr(x)))
            }
            ExprRepeat(expr, count) => {
                ExprRepeat(folder.fold_expr(expr), folder.fold_expr(count))
            }
            ExprTup(elts) => ExprTup(elts.move_map(|x| folder.fold_expr(x))),
            ExprCall(f, args) => {
                ExprCall(folder.fold_expr(f),
                         args.move_map(|x| folder.fold_expr(x)))
            }
            ExprMethodCall(i, tps, args) => {
                ExprMethodCall(
                    respan(i.span, folder.fold_ident(i.node)),
                    tps.move_map(|x| folder.fold_ty(x)),
                    args.move_map(|x| folder.fold_expr(x)))
            }
            ExprBinary(binop, lhs, rhs) => {
                ExprBinary(binop,
                        folder.fold_expr(lhs),
                        folder.fold_expr(rhs))
            }
            ExprUnary(binop, ohs) => {
                ExprUnary(binop, folder.fold_expr(ohs))
            }
            ExprLit(l) => ExprLit(l),
            ExprCast(expr, ty) => {
                ExprCast(folder.fold_expr(expr), folder.fold_ty(ty))
            }
            ExprAddrOf(m, ohs) => ExprAddrOf(m, folder.fold_expr(ohs)),
            ExprIf(cond, tr, fl) => {
                ExprIf(folder.fold_expr(cond),
                       folder.fold_block(tr),
                       fl.map(|x| folder.fold_expr(x)))
            }
            ExprWhile(cond, body, opt_ident) => {
                ExprWhile(folder.fold_expr(cond),
                          folder.fold_block(body),
                          opt_ident.map(|i| folder.fold_ident(i)))
            }
            ExprForLoop(pat, iter, body, opt_ident) => {
                ExprForLoop(folder.fold_pat(pat),
                            folder.fold_expr(iter),
                            folder.fold_block(body),
                            opt_ident.map(|i| folder.fold_ident(i)))
            }
            ExprLoop(body, opt_ident) => {
                ExprLoop(folder.fold_block(body),
                        opt_ident.map(|i| folder.fold_ident(i)))
            }
            ExprMatch(expr, arms) => {
                ExprMatch(folder.fold_expr(expr),
                        arms.move_map(|x| folder.fold_arm(x)))
            }
            ExprFnBlock(capture_clause, decl, body) => {
                ExprFnBlock(capture_clause,
                            folder.fold_fn_decl(decl),
                            folder.fold_block(body))
            }
            ExprProc(decl, body) => {
                ExprProc(folder.fold_fn_decl(decl),
                         folder.fold_block(body))
            }
            ExprUnboxedFn(capture_clause, kind, decl, body) => {
                ExprUnboxedFn(capture_clause,
                            kind,
                            folder.fold_fn_decl(decl),
                            folder.fold_block(body))
            }
            ExprBlock(blk) => ExprBlock(folder.fold_block(blk)),
            ExprAssign(el, er) => {
                ExprAssign(folder.fold_expr(el), folder.fold_expr(er))
            }
            ExprAssignOp(op, el, er) => {
                ExprAssignOp(op,
                            folder.fold_expr(el),
                            folder.fold_expr(er))
            }
            ExprField(el, ident, tys) => {
                ExprField(folder.fold_expr(el),
                          respan(ident.span, folder.fold_ident(ident.node)),
                          tys.move_map(|x| folder.fold_ty(x)))
            }
            ExprTupField(el, ident, tys) => {
                ExprTupField(folder.fold_expr(el),
                             respan(ident.span, folder.fold_uint(ident.node)),
                             tys.move_map(|x| folder.fold_ty(x)))
            }
            ExprIndex(el, er) => {
                ExprIndex(folder.fold_expr(el), folder.fold_expr(er))
            }
            ExprSlice(e, e1, e2, m) => {
                ExprSlice(folder.fold_expr(e),
                          e1.map(|x| folder.fold_expr(x)),
                          e2.map(|x| folder.fold_expr(x)),
                          m)
            }
            ExprPath(pth) => ExprPath(folder.fold_path(pth)),
            ExprBreak(opt_ident) => ExprBreak(opt_ident.map(|x| folder.fold_ident(x))),
            ExprAgain(opt_ident) => ExprAgain(opt_ident.map(|x| folder.fold_ident(x))),
            ExprRet(e) => ExprRet(e.map(|x| folder.fold_expr(x))),
            ExprInlineAsm(InlineAsm {
                inputs,
                outputs,
                asm,
                asm_str_style,
                clobbers,
                volatile,
                alignstack,
                dialect,
                expn_id,
            }) => ExprInlineAsm(InlineAsm {
                inputs: inputs.move_map(|(c, input)| {
                    (c, folder.fold_expr(input))
                }),
                outputs: outputs.move_map(|(c, out, is_rw)| {
                    (c, folder.fold_expr(out), is_rw)
                }),
                asm: asm,
                asm_str_style: asm_str_style,
                clobbers: clobbers,
                volatile: volatile,
                alignstack: alignstack,
                dialect: dialect,
                expn_id: expn_id,
            }),
            ExprMac(mac) => ExprMac(folder.fold_mac(mac)),
            ExprStruct(path, fields, maybe_expr) => {
                ExprStruct(folder.fold_path(path),
                        fields.move_map(|x| folder.fold_field(x)),
                        maybe_expr.map(|x| folder.fold_expr(x)))
            },
            ExprParen(ex) => ExprParen(folder.fold_expr(ex))
        },
        span: folder.new_span(span)
    }
}

pub fn noop_fold_stmt<T: Folder>(Spanned {node, span}: Stmt, folder: &mut T)
                                 -> SmallVector<P<Stmt>> {
    let span = folder.new_span(span);
    match node {
        StmtDecl(d, id) => {
            let id = folder.new_id(id);
            folder.fold_decl(d).into_iter().map(|d| P(Spanned {
                node: StmtDecl(d, id),
                span: span
            })).collect()
        }
        StmtExpr(e, id) => {
            let id = folder.new_id(id);
            SmallVector::one(P(Spanned {
                node: StmtExpr(folder.fold_expr(e), id),
                span: span
            }))
        }
        StmtSemi(e, id) => {
            let id = folder.new_id(id);
            SmallVector::one(P(Spanned {
                node: StmtSemi(folder.fold_expr(e), id),
                span: span
            }))
        }
        StmtMac(mac, semi) => SmallVector::one(P(Spanned {
            node: StmtMac(folder.fold_mac(mac), semi),
            span: span
        }))
    }
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
        fn fold_mac(&mut self, macro: ast::Mac) -> ast::Mac {
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
