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
use syntax_pos::Span;
use source_map::{Spanned, respan};
use parse::token::{self, Token};
use ptr::P;
use OneVector;
use symbol::keywords;
use ThinVec;
use tokenstream::*;
use util::move_map::MoveMap;

use rustc_data_structures::sync::Lrc;
use rustc_data_structures::small_vec::ExpectOne;

pub trait Folder : Sized {
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

    fn fold_meta_items(&mut self, meta_items: Vec<MetaItem>) -> Vec<MetaItem> {
        noop_fold_meta_items(meta_items, self)
    }

    fn fold_meta_list_item(&mut self, list_item: NestedMetaItem) -> NestedMetaItem {
        noop_fold_meta_list_item(list_item, self)
    }

    fn fold_meta_item(&mut self, meta_item: MetaItem) -> MetaItem {
        noop_fold_meta_item(meta_item, self)
    }

    fn fold_use_tree(&mut self, use_tree: UseTree) -> UseTree {
        noop_fold_use_tree(use_tree, self)
    }

    fn fold_foreign_item(&mut self, ni: ForeignItem) -> OneVector<ForeignItem> {
        noop_fold_foreign_item(ni, self)
    }

    fn fold_foreign_item_simple(&mut self, ni: ForeignItem) -> ForeignItem {
        noop_fold_foreign_item_simple(ni, self)
    }

    fn fold_item(&mut self, i: P<Item>) -> OneVector<P<Item>> {
        noop_fold_item(i, self)
    }

    fn fold_item_simple(&mut self, i: Item) -> Item {
        noop_fold_item_simple(i, self)
    }

    fn fold_fn_header(&mut self, header: FnHeader) -> FnHeader {
        noop_fold_fn_header(header, self)
    }

    fn fold_struct_field(&mut self, sf: StructField) -> StructField {
        noop_fold_struct_field(sf, self)
    }

    fn fold_item_kind(&mut self, i: ItemKind) -> ItemKind {
        noop_fold_item_kind(i, self)
    }

    fn fold_trait_item(&mut self, i: TraitItem) -> OneVector<TraitItem> {
        noop_fold_trait_item(i, self)
    }

    fn fold_impl_item(&mut self, i: ImplItem) -> OneVector<ImplItem> {
        noop_fold_impl_item(i, self)
    }

    fn fold_fn_decl(&mut self, d: P<FnDecl>) -> P<FnDecl> {
        noop_fold_fn_decl(d, self)
    }

    fn fold_asyncness(&mut self, a: IsAsync) -> IsAsync {
        noop_fold_asyncness(a, self)
    }

    fn fold_block(&mut self, b: P<Block>) -> P<Block> {
        noop_fold_block(b, self)
    }

    fn fold_stmt(&mut self, s: Stmt) -> OneVector<Stmt> {
        noop_fold_stmt(s, self)
    }

    fn fold_arm(&mut self, a: Arm) -> Arm {
        noop_fold_arm(a, self)
    }

    fn fold_pat(&mut self, p: P<Pat>) -> P<Pat> {
        noop_fold_pat(p, self)
    }

    fn fold_anon_const(&mut self, c: AnonConst) -> AnonConst {
        noop_fold_anon_const(c, self)
    }

    fn fold_expr(&mut self, e: P<Expr>) -> P<Expr> {
        e.map(|e| noop_fold_expr(e, self))
    }

    fn fold_range_end(&mut self, re: RangeEnd) -> RangeEnd {
        noop_fold_range_end(re, self)
    }

    fn fold_opt_expr(&mut self, e: P<Expr>) -> Option<P<Expr>> {
        noop_fold_opt_expr(e, self)
    }

    fn fold_exprs(&mut self, es: Vec<P<Expr>>) -> Vec<P<Expr>> {
        noop_fold_exprs(es, self)
    }

    fn fold_generic_arg(&mut self, arg: GenericArg) -> GenericArg {
        match arg {
            GenericArg::Lifetime(lt) => GenericArg::Lifetime(self.fold_lifetime(lt)),
            GenericArg::Type(ty) => GenericArg::Type(self.fold_ty(ty)),
        }
    }

    fn fold_ty(&mut self, t: P<Ty>) -> P<Ty> {
        noop_fold_ty(t, self)
    }

    fn fold_lifetime(&mut self, l: Lifetime) -> Lifetime {
        noop_fold_lifetime(l, self)
    }

    fn fold_ty_binding(&mut self, t: TypeBinding) -> TypeBinding {
        noop_fold_ty_binding(t, self)
    }

    fn fold_mod(&mut self, m: Mod) -> Mod {
        noop_fold_mod(m, self)
    }

    fn fold_foreign_mod(&mut self, nm: ForeignMod) -> ForeignMod {
        noop_fold_foreign_mod(nm, self)
    }

    fn fold_global_asm(&mut self, ga: P<GlobalAsm>) -> P<GlobalAsm> {
        noop_fold_global_asm(ga, self)
    }

    fn fold_variant(&mut self, v: Variant) -> Variant {
        noop_fold_variant(v, self)
    }

    fn fold_ident(&mut self, i: Ident) -> Ident {
        noop_fold_ident(i, self)
    }

    fn fold_usize(&mut self, i: usize) -> usize {
        noop_fold_usize(i, self)
    }

    fn fold_path(&mut self, p: Path) -> Path {
        noop_fold_path(p, self)
    }

    fn fold_qpath(&mut self, qs: Option<QSelf>, p: Path) -> (Option<QSelf>, Path) {
        noop_fold_qpath(qs, p, self)
    }

    fn fold_generic_args(&mut self, p: GenericArgs) -> GenericArgs {
        noop_fold_generic_args(p, self)
    }

    fn fold_angle_bracketed_parameter_data(&mut self, p: AngleBracketedArgs)
                                           -> AngleBracketedArgs
    {
        noop_fold_angle_bracketed_parameter_data(p, self)
    }

    fn fold_parenthesized_parameter_data(&mut self, p: ParenthesisedArgs)
                                         -> ParenthesisedArgs
    {
        noop_fold_parenthesized_parameter_data(p, self)
    }

    fn fold_local(&mut self, l: P<Local>) -> P<Local> {
        noop_fold_local(l, self)
    }

    fn fold_mac(&mut self, _mac: Mac) -> Mac {
        panic!("fold_mac disabled by default");
        // NB: see note about macros above.
        // if you really want a folder that
        // works on macros, use this
        // definition in your trait impl:
        // fold::noop_fold_mac(_mac, self)
    }

    fn fold_macro_def(&mut self, def: MacroDef) -> MacroDef {
        noop_fold_macro_def(def, self)
    }

    fn fold_label(&mut self, label: Label) -> Label {
        noop_fold_label(label, self)
    }

    fn fold_attribute(&mut self, at: Attribute) -> Option<Attribute> {
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

    fn fold_poly_trait_ref(&mut self, p: PolyTraitRef) -> PolyTraitRef {
        noop_fold_poly_trait_ref(p, self)
    }

    fn fold_variant_data(&mut self, vdata: VariantData) -> VariantData {
        noop_fold_variant_data(vdata, self)
    }

    fn fold_generic_param(&mut self, param: GenericParam) -> GenericParam {
        noop_fold_generic_param(param, self)
    }

    fn fold_generic_params(&mut self, params: Vec<GenericParam>) -> Vec<GenericParam> {
        noop_fold_generic_params(params, self)
    }

    fn fold_tt(&mut self, tt: TokenTree) -> TokenTree {
        noop_fold_tt(tt, self)
    }

    fn fold_tts(&mut self, tts: TokenStream) -> TokenStream {
        noop_fold_tts(tts, self)
    }

    fn fold_token(&mut self, t: token::Token) -> token::Token {
        noop_fold_token(t, self)
    }

    fn fold_interpolated(&mut self, nt: token::Nonterminal) -> token::Nonterminal {
        noop_fold_interpolated(nt, self)
    }

    fn fold_opt_bounds(&mut self, b: Option<GenericBounds>) -> Option<GenericBounds> {
        noop_fold_opt_bounds(b, self)
    }

    fn fold_bounds(&mut self, b: GenericBounds) -> GenericBounds {
        noop_fold_bounds(b, self)
    }

    fn fold_param_bound(&mut self, tpb: GenericBound) -> GenericBound {
        noop_fold_param_bound(tpb, self)
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

    fn fold_vis(&mut self, vis: Visibility) -> Visibility {
        noop_fold_vis(vis, self)
    }

    fn new_id(&mut self, i: NodeId) -> NodeId {
        i
    }

    fn new_span(&mut self, sp: Span) -> Span {
        sp
    }
}

pub fn noop_fold_meta_items<T: Folder>(meta_items: Vec<MetaItem>, fld: &mut T) -> Vec<MetaItem> {
    meta_items.move_map(|x| fld.fold_meta_item(x))
}

pub fn noop_fold_use_tree<T: Folder>(use_tree: UseTree, fld: &mut T) -> UseTree {
    UseTree {
        span: fld.new_span(use_tree.span),
        prefix: fld.fold_path(use_tree.prefix),
        kind: match use_tree.kind {
            UseTreeKind::Simple(rename, id1, id2) =>
                UseTreeKind::Simple(rename.map(|ident| fld.fold_ident(ident)),
                                    fld.new_id(id1), fld.new_id(id2)),
            UseTreeKind::Glob => UseTreeKind::Glob,
            UseTreeKind::Nested(items) => UseTreeKind::Nested(items.move_map(|(tree, id)| {
                (fld.fold_use_tree(tree), fld.new_id(id))
            })),
        },
    }
}

pub fn fold_attrs<T: Folder>(attrs: Vec<Attribute>, fld: &mut T) -> Vec<Attribute> {
    attrs.move_flat_map(|x| fld.fold_attribute(x))
}

pub fn fold_thin_attrs<T: Folder>(attrs: ThinVec<Attribute>, fld: &mut T) -> ThinVec<Attribute> {
    fold_attrs(attrs.into(), fld).into()
}

pub fn noop_fold_arm<T: Folder>(Arm {attrs, pats, guard, body}: Arm,
    fld: &mut T) -> Arm {
    Arm {
        attrs: fold_attrs(attrs, fld),
        pats: pats.move_map(|x| fld.fold_pat(x)),
        guard: guard.map(|x| fld.fold_expr(x)),
        body: fld.fold_expr(body),
    }
}

pub fn noop_fold_ty_binding<T: Folder>(b: TypeBinding, fld: &mut T) -> TypeBinding {
    TypeBinding {
        id: fld.new_id(b.id),
        ident: fld.fold_ident(b.ident),
        ty: fld.fold_ty(b.ty),
        span: fld.new_span(b.span),
    }
}

pub fn noop_fold_ty<T: Folder>(t: P<Ty>, fld: &mut T) -> P<Ty> {
    t.map(|Ty {id, node, span}| Ty {
        id: fld.new_id(id),
        node: match node {
            TyKind::Infer | TyKind::ImplicitSelf | TyKind::Err => node,
            TyKind::Slice(ty) => TyKind::Slice(fld.fold_ty(ty)),
            TyKind::Ptr(mt) => TyKind::Ptr(fld.fold_mt(mt)),
            TyKind::Rptr(region, mt) => {
                TyKind::Rptr(region.map(|lt| noop_fold_lifetime(lt, fld)), fld.fold_mt(mt))
            }
            TyKind::BareFn(f) => {
                TyKind::BareFn(f.map(|BareFnTy {generic_params, unsafety, abi, decl}| BareFnTy {
                    generic_params: fld.fold_generic_params(generic_params),
                    unsafety,
                    abi,
                    decl: fld.fold_fn_decl(decl)
                }))
            }
            TyKind::Never => node,
            TyKind::Tup(tys) => TyKind::Tup(tys.move_map(|ty| fld.fold_ty(ty))),
            TyKind::Paren(ty) => TyKind::Paren(fld.fold_ty(ty)),
            TyKind::Path(qself, path) => {
                let (qself, path) = fld.fold_qpath(qself, path);
                TyKind::Path(qself, path)
            }
            TyKind::Array(ty, length) => {
                TyKind::Array(fld.fold_ty(ty), fld.fold_anon_const(length))
            }
            TyKind::Typeof(expr) => {
                TyKind::Typeof(fld.fold_anon_const(expr))
            }
            TyKind::TraitObject(bounds, syntax) => {
                TyKind::TraitObject(bounds.move_map(|b| fld.fold_param_bound(b)), syntax)
            }
            TyKind::ImplTrait(id, bounds) => {
                TyKind::ImplTrait(fld.new_id(id), bounds.move_map(|b| fld.fold_param_bound(b)))
            }
            TyKind::Mac(mac) => {
                TyKind::Mac(fld.fold_mac(mac))
            }
        },
        span: fld.new_span(span)
    })
}

pub fn noop_fold_foreign_mod<T: Folder>(ForeignMod {abi, items}: ForeignMod,
                                        fld: &mut T) -> ForeignMod {
    ForeignMod {
        abi,
        items: items.move_flat_map(|x| fld.fold_foreign_item(x)),
    }
}

pub fn noop_fold_global_asm<T: Folder>(ga: P<GlobalAsm>,
                                       _: &mut T) -> P<GlobalAsm> {
    ga
}

pub fn noop_fold_variant<T: Folder>(v: Variant, fld: &mut T) -> Variant {
    Spanned {
        node: Variant_ {
            ident: fld.fold_ident(v.node.ident),
            attrs: fold_attrs(v.node.attrs, fld),
            data: fld.fold_variant_data(v.node.data),
            disr_expr: v.node.disr_expr.map(|e| fld.fold_anon_const(e)),
        },
        span: fld.new_span(v.span),
    }
}

pub fn noop_fold_ident<T: Folder>(ident: Ident, fld: &mut T) -> Ident {
    Ident::new(ident.name, fld.new_span(ident.span))
}

pub fn noop_fold_usize<T: Folder>(i: usize, _: &mut T) -> usize {
    i
}

pub fn noop_fold_path<T: Folder>(Path { segments, span }: Path, fld: &mut T) -> Path {
    Path {
        segments: segments.move_map(|PathSegment { ident, args }| PathSegment {
            ident: fld.fold_ident(ident),
            args: args.map(|args| args.map(|args| fld.fold_generic_args(args))),
        }),
        span: fld.new_span(span)
    }
}

pub fn noop_fold_qpath<T: Folder>(qself: Option<QSelf>,
                                  path: Path,
                                  fld: &mut T) -> (Option<QSelf>, Path) {
    let qself = qself.map(|QSelf { ty, path_span, position }| {
        QSelf {
            ty: fld.fold_ty(ty),
            path_span: fld.new_span(path_span),
            position,
        }
    });
    (qself, fld.fold_path(path))
}

pub fn noop_fold_generic_args<T: Folder>(generic_args: GenericArgs, fld: &mut T) -> GenericArgs
{
    match generic_args {
        GenericArgs::AngleBracketed(data) => {
            GenericArgs::AngleBracketed(fld.fold_angle_bracketed_parameter_data(data))
        }
        GenericArgs::Parenthesized(data) => {
            GenericArgs::Parenthesized(fld.fold_parenthesized_parameter_data(data))
        }
    }
}

pub fn noop_fold_angle_bracketed_parameter_data<T: Folder>(data: AngleBracketedArgs,
                                                           fld: &mut T)
                                                           -> AngleBracketedArgs
{
    let AngleBracketedArgs { args, bindings, span } = data;
    AngleBracketedArgs {
        args: args.move_map(|arg| fld.fold_generic_arg(arg)),
        bindings: bindings.move_map(|b| fld.fold_ty_binding(b)),
        span: fld.new_span(span)
    }
}

pub fn noop_fold_parenthesized_parameter_data<T: Folder>(data: ParenthesisedArgs,
                                                         fld: &mut T)
                                                         -> ParenthesisedArgs
{
    let ParenthesisedArgs { inputs, output, span } = data;
    ParenthesisedArgs {
        inputs: inputs.move_map(|ty| fld.fold_ty(ty)),
        output: output.map(|ty| fld.fold_ty(ty)),
        span: fld.new_span(span)
    }
}

pub fn noop_fold_local<T: Folder>(l: P<Local>, fld: &mut T) -> P<Local> {
    l.map(|Local {id, pat, ty, init, span, attrs}| Local {
        id: fld.new_id(id),
        pat: fld.fold_pat(pat),
        ty: ty.map(|t| fld.fold_ty(t)),
        init: init.map(|e| fld.fold_expr(e)),
        span: fld.new_span(span),
        attrs: fold_attrs(attrs.into(), fld).into(),
    })
}

pub fn noop_fold_attribute<T: Folder>(attr: Attribute, fld: &mut T) -> Option<Attribute> {
    Some(Attribute {
        id: attr.id,
        style: attr.style,
        path: fld.fold_path(attr.path),
        tokens: fld.fold_tts(attr.tokens),
        is_sugared_doc: attr.is_sugared_doc,
        span: fld.new_span(attr.span),
    })
}

pub fn noop_fold_mac<T: Folder>(Spanned {node, span}: Mac, fld: &mut T) -> Mac {
    Spanned {
        node: Mac_ {
            tts: fld.fold_tts(node.stream()).into(),
            path: fld.fold_path(node.path),
            delim: node.delim,
        },
        span: fld.new_span(span)
    }
}

pub fn noop_fold_macro_def<T: Folder>(def: MacroDef, fld: &mut T) -> MacroDef {
    MacroDef {
        tokens: fld.fold_tts(def.tokens.into()).into(),
        legacy: def.legacy,
    }
}

pub fn noop_fold_meta_list_item<T: Folder>(li: NestedMetaItem, fld: &mut T)
    -> NestedMetaItem {
    Spanned {
        node: match li.node {
            NestedMetaItemKind::MetaItem(mi) =>  {
                NestedMetaItemKind::MetaItem(fld.fold_meta_item(mi))
            },
            NestedMetaItemKind::Literal(lit) => NestedMetaItemKind::Literal(lit)
        },
        span: fld.new_span(li.span)
    }
}

pub fn noop_fold_meta_item<T: Folder>(mi: MetaItem, fld: &mut T) -> MetaItem {
    MetaItem {
        ident: mi.ident,
        node: match mi.node {
            MetaItemKind::Word => MetaItemKind::Word,
            MetaItemKind::List(mis) => {
                MetaItemKind::List(mis.move_map(|e| fld.fold_meta_list_item(e)))
            },
            MetaItemKind::NameValue(s) => MetaItemKind::NameValue(s),
        },
        span: fld.new_span(mi.span)
    }
}

pub fn noop_fold_arg<T: Folder>(Arg {id, pat, ty}: Arg, fld: &mut T) -> Arg {
    Arg {
        id: fld.new_id(id),
        pat: fld.fold_pat(pat),
        ty: fld.fold_ty(ty)
    }
}

pub fn noop_fold_tt<T: Folder>(tt: TokenTree, fld: &mut T) -> TokenTree {
    match tt {
        TokenTree::Token(span, tok) =>
            TokenTree::Token(fld.new_span(span), fld.fold_token(tok)),
        TokenTree::Delimited(span, delimed) => TokenTree::Delimited(fld.new_span(span), Delimited {
            tts: fld.fold_tts(delimed.stream()).into(),
            delim: delimed.delim,
        }),
    }
}

pub fn noop_fold_tts<T: Folder>(tts: TokenStream, fld: &mut T) -> TokenStream {
    tts.map(|tt| fld.fold_tt(tt))
}

// apply ident folder if it's an ident, apply other folds to interpolated nodes
pub fn noop_fold_token<T: Folder>(t: token::Token, fld: &mut T) -> token::Token {
    match t {
        token::Ident(id, is_raw) => token::Ident(fld.fold_ident(id), is_raw),
        token::Lifetime(id) => token::Lifetime(fld.fold_ident(id)),
        token::Interpolated(nt) => {
            let nt = match Lrc::try_unwrap(nt) {
                Ok(nt) => nt,
                Err(nt) => (*nt).clone(),
            };
            Token::interpolated(fld.fold_interpolated(nt.0))
        }
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
        token::NtIdent(ident, is_raw) => token::NtIdent(fld.fold_ident(ident), is_raw),
        token::NtLifetime(ident) => token::NtLifetime(fld.fold_ident(ident)),
        token::NtLiteral(expr) => token::NtLiteral(fld.fold_expr(expr)),
        token::NtMeta(meta) => token::NtMeta(fld.fold_meta_item(meta)),
        token::NtPath(path) => token::NtPath(fld.fold_path(path)),
        token::NtTT(tt) => token::NtTT(fld.fold_tt(tt)),
        token::NtArm(arm) => token::NtArm(fld.fold_arm(arm)),
        token::NtImplItem(item) =>
            token::NtImplItem(fld.fold_impl_item(item)
                              .expect_one("expected fold to produce exactly one item")),
        token::NtTraitItem(item) =>
            token::NtTraitItem(fld.fold_trait_item(item)
                               .expect_one("expected fold to produce exactly one item")),
        token::NtGenerics(generics) => token::NtGenerics(fld.fold_generics(generics)),
        token::NtWhereClause(where_clause) =>
            token::NtWhereClause(fld.fold_where_clause(where_clause)),
        token::NtArg(arg) => token::NtArg(fld.fold_arg(arg)),
        token::NtVis(vis) => token::NtVis(fld.fold_vis(vis)),
        token::NtForeignItem(ni) =>
            token::NtForeignItem(fld.fold_foreign_item(ni)
                                 // see reasoning above
                                 .expect_one("expected fold to produce exactly one item")),
    }
}

pub fn noop_fold_asyncness<T: Folder>(asyncness: IsAsync, fld: &mut T) -> IsAsync {
    match asyncness {
        IsAsync::Async { closure_id, return_impl_trait_id } => IsAsync::Async {
            closure_id: fld.new_id(closure_id),
            return_impl_trait_id: fld.new_id(return_impl_trait_id),
        },
        IsAsync::NotAsync => IsAsync::NotAsync,
    }
}

pub fn noop_fold_fn_decl<T: Folder>(decl: P<FnDecl>, fld: &mut T) -> P<FnDecl> {
    decl.map(|FnDecl {inputs, output, variadic}| FnDecl {
        inputs: inputs.move_map(|x| fld.fold_arg(x)),
        output: match output {
            FunctionRetTy::Ty(ty) => FunctionRetTy::Ty(fld.fold_ty(ty)),
            FunctionRetTy::Default(span) => FunctionRetTy::Default(fld.new_span(span)),
        },
        variadic,
    })
}

pub fn noop_fold_param_bound<T>(pb: GenericBound, fld: &mut T) -> GenericBound where T: Folder {
    match pb {
        GenericBound::Trait(ty, modifier) => {
            GenericBound::Trait(fld.fold_poly_trait_ref(ty), modifier)
        }
        GenericBound::Outlives(lifetime) => {
            GenericBound::Outlives(noop_fold_lifetime(lifetime, fld))
        }
    }
}

pub fn noop_fold_generic_param<T: Folder>(param: GenericParam, fld: &mut T) -> GenericParam {
    let attrs: Vec<_> = param.attrs.into();
    GenericParam {
        ident: fld.fold_ident(param.ident),
        id: fld.new_id(param.id),
        attrs: attrs.into_iter()
                    .flat_map(|x| fld.fold_attribute(x).into_iter())
                    .collect::<Vec<_>>()
                    .into(),
        bounds: param.bounds.move_map(|l| noop_fold_param_bound(l, fld)),
        kind: match param.kind {
            GenericParamKind::Lifetime => GenericParamKind::Lifetime,
            GenericParamKind::Type { default } => GenericParamKind::Type {
                default: default.map(|ty| fld.fold_ty(ty))
            }
        }
    }
}

pub fn noop_fold_generic_params<T: Folder>(
    params: Vec<GenericParam>,
    fld: &mut T
) -> Vec<GenericParam> {
    params.move_map(|p| fld.fold_generic_param(p))
}

pub fn noop_fold_label<T: Folder>(label: Label, fld: &mut T) -> Label {
    Label {
        ident: fld.fold_ident(label.ident),
    }
}

fn noop_fold_lifetime<T: Folder>(l: Lifetime, fld: &mut T) -> Lifetime {
    Lifetime {
        id: fld.new_id(l.id),
        ident: fld.fold_ident(l.ident),
    }
}

pub fn noop_fold_generics<T: Folder>(Generics { params, where_clause, span }: Generics,
                                     fld: &mut T) -> Generics {
    Generics {
        params: fld.fold_generic_params(params),
        where_clause: fld.fold_where_clause(where_clause),
        span: fld.new_span(span),
    }
}

pub fn noop_fold_where_clause<T: Folder>(
                              WhereClause {id, predicates, span}: WhereClause,
                              fld: &mut T)
                              -> WhereClause {
    WhereClause {
        id: fld.new_id(id),
        predicates: predicates.move_map(|predicate| {
            fld.fold_where_predicate(predicate)
        }),
        span,
    }
}

pub fn noop_fold_where_predicate<T: Folder>(
                                 pred: WherePredicate,
                                 fld: &mut T)
                                 -> WherePredicate {
    match pred {
        ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate{bound_generic_params,
                                                                     bounded_ty,
                                                                     bounds,
                                                                     span}) => {
            ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                bound_generic_params: fld.fold_generic_params(bound_generic_params),
                bounded_ty: fld.fold_ty(bounded_ty),
                bounds: bounds.move_map(|x| fld.fold_param_bound(x)),
                span: fld.new_span(span)
            })
        }
        ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{lifetime,
                                                                       bounds,
                                                                       span}) => {
            ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate {
                span: fld.new_span(span),
                lifetime: noop_fold_lifetime(lifetime, fld),
                bounds: bounds.move_map(|bound| noop_fold_param_bound(bound, fld))
            })
        }
        ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{id,
                                                               lhs_ty,
                                                               rhs_ty,
                                                               span}) => {
            ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{
                id: fld.new_id(id),
                lhs_ty: fld.fold_ty(lhs_ty),
                rhs_ty: fld.fold_ty(rhs_ty),
                span: fld.new_span(span)
            })
        }
    }
}

pub fn noop_fold_variant_data<T: Folder>(vdata: VariantData, fld: &mut T) -> VariantData {
    match vdata {
        ast::VariantData::Struct(fields, id) => {
            ast::VariantData::Struct(fields.move_map(|f| fld.fold_struct_field(f)),
                                     fld.new_id(id))
        }
        ast::VariantData::Tuple(fields, id) => {
            ast::VariantData::Tuple(fields.move_map(|f| fld.fold_struct_field(f)),
                                    fld.new_id(id))
        }
        ast::VariantData::Unit(id) => ast::VariantData::Unit(fld.new_id(id))
    }
}

pub fn noop_fold_trait_ref<T: Folder>(p: TraitRef, fld: &mut T) -> TraitRef {
    let id = fld.new_id(p.ref_id);
    let TraitRef {
        path,
        ref_id: _,
    } = p;
    ast::TraitRef {
        path: fld.fold_path(path),
        ref_id: id,
    }
}

pub fn noop_fold_poly_trait_ref<T: Folder>(p: PolyTraitRef, fld: &mut T) -> PolyTraitRef {
    ast::PolyTraitRef {
        bound_generic_params: fld.fold_generic_params(p.bound_generic_params),
        trait_ref: fld.fold_trait_ref(p.trait_ref),
        span: fld.new_span(p.span),
    }
}

pub fn noop_fold_struct_field<T: Folder>(f: StructField, fld: &mut T) -> StructField {
    StructField {
        span: fld.new_span(f.span),
        id: fld.new_id(f.id),
        ident: f.ident.map(|ident| fld.fold_ident(ident)),
        vis: fld.fold_vis(f.vis),
        ty: fld.fold_ty(f.ty),
        attrs: fold_attrs(f.attrs, fld),
    }
}

pub fn noop_fold_field<T: Folder>(f: Field, folder: &mut T) -> Field {
    Field {
        ident: folder.fold_ident(f.ident),
        expr: folder.fold_expr(f.expr),
        span: folder.new_span(f.span),
        is_shorthand: f.is_shorthand,
        attrs: fold_thin_attrs(f.attrs, folder),
    }
}

pub fn noop_fold_mt<T: Folder>(MutTy {ty, mutbl}: MutTy, folder: &mut T) -> MutTy {
    MutTy {
        ty: folder.fold_ty(ty),
        mutbl,
    }
}

pub fn noop_fold_opt_bounds<T: Folder>(b: Option<GenericBounds>, folder: &mut T)
                                       -> Option<GenericBounds> {
    b.map(|bounds| folder.fold_bounds(bounds))
}

fn noop_fold_bounds<T: Folder>(bounds: GenericBounds, folder: &mut T)
                          -> GenericBounds {
    bounds.move_map(|bound| folder.fold_param_bound(bound))
}

pub fn noop_fold_block<T: Folder>(b: P<Block>, folder: &mut T) -> P<Block> {
    b.map(|Block {id, stmts, rules, span, recovered}| Block {
        id: folder.new_id(id),
        stmts: stmts.move_flat_map(|s| folder.fold_stmt(s).into_iter()),
        rules,
        span: folder.new_span(span),
        recovered,
    })
}

pub fn noop_fold_item_kind<T: Folder>(i: ItemKind, folder: &mut T) -> ItemKind {
    match i {
        ItemKind::ExternCrate(orig_name) => ItemKind::ExternCrate(orig_name),
        ItemKind::Use(use_tree) => {
            ItemKind::Use(use_tree.map(|tree| folder.fold_use_tree(tree)))
        }
        ItemKind::Static(t, m, e) => {
            ItemKind::Static(folder.fold_ty(t), m, folder.fold_expr(e))
        }
        ItemKind::Const(t, e) => {
            ItemKind::Const(folder.fold_ty(t), folder.fold_expr(e))
        }
        ItemKind::Fn(decl, header, generics, body) => {
            let generics = folder.fold_generics(generics);
            let header = folder.fold_fn_header(header);
            let decl = folder.fold_fn_decl(decl);
            let body = folder.fold_block(body);
            ItemKind::Fn(decl, header, generics, body)
        }
        ItemKind::Mod(m) => ItemKind::Mod(folder.fold_mod(m)),
        ItemKind::ForeignMod(nm) => ItemKind::ForeignMod(folder.fold_foreign_mod(nm)),
        ItemKind::GlobalAsm(ga) => ItemKind::GlobalAsm(folder.fold_global_asm(ga)),
        ItemKind::Ty(t, generics) => {
            ItemKind::Ty(folder.fold_ty(t), folder.fold_generics(generics))
        }
        ItemKind::Existential(bounds, generics) => ItemKind::Existential(
            folder.fold_bounds(bounds),
            folder.fold_generics(generics),
        ),
        ItemKind::Enum(enum_definition, generics) => {
            let generics = folder.fold_generics(generics);
            let variants = enum_definition.variants.move_map(|x| folder.fold_variant(x));
            ItemKind::Enum(ast::EnumDef { variants: variants }, generics)
        }
        ItemKind::Struct(struct_def, generics) => {
            let generics = folder.fold_generics(generics);
            ItemKind::Struct(folder.fold_variant_data(struct_def), generics)
        }
        ItemKind::Union(struct_def, generics) => {
            let generics = folder.fold_generics(generics);
            ItemKind::Union(folder.fold_variant_data(struct_def), generics)
        }
        ItemKind::Impl(unsafety,
                       polarity,
                       defaultness,
                       generics,
                       ifce,
                       ty,
                       impl_items) => ItemKind::Impl(
            unsafety,
            polarity,
            defaultness,
            folder.fold_generics(generics),
            ifce.map(|trait_ref| folder.fold_trait_ref(trait_ref.clone())),
            folder.fold_ty(ty),
            impl_items.move_flat_map(|item| folder.fold_impl_item(item)),
        ),
        ItemKind::Trait(is_auto, unsafety, generics, bounds, items) => ItemKind::Trait(
            is_auto,
            unsafety,
            folder.fold_generics(generics),
            folder.fold_bounds(bounds),
            items.move_flat_map(|item| folder.fold_trait_item(item)),
        ),
        ItemKind::TraitAlias(generics, bounds) => ItemKind::TraitAlias(
            folder.fold_generics(generics),
            folder.fold_bounds(bounds)),
        ItemKind::Mac(m) => ItemKind::Mac(folder.fold_mac(m)),
        ItemKind::MacroDef(def) => ItemKind::MacroDef(folder.fold_macro_def(def)),
    }
}

pub fn noop_fold_trait_item<T: Folder>(i: TraitItem, folder: &mut T)
                                       -> OneVector<TraitItem> {
    smallvec![TraitItem {
        id: folder.new_id(i.id),
        ident: folder.fold_ident(i.ident),
        attrs: fold_attrs(i.attrs, folder),
        generics: folder.fold_generics(i.generics),
        node: match i.node {
            TraitItemKind::Const(ty, default) => {
                TraitItemKind::Const(folder.fold_ty(ty),
                               default.map(|x| folder.fold_expr(x)))
            }
            TraitItemKind::Method(sig, body) => {
                TraitItemKind::Method(noop_fold_method_sig(sig, folder),
                                body.map(|x| folder.fold_block(x)))
            }
            TraitItemKind::Type(bounds, default) => {
                TraitItemKind::Type(folder.fold_bounds(bounds),
                              default.map(|x| folder.fold_ty(x)))
            }
            ast::TraitItemKind::Macro(mac) => {
                TraitItemKind::Macro(folder.fold_mac(mac))
            }
        },
        span: folder.new_span(i.span),
        tokens: i.tokens,
    }]
}

pub fn noop_fold_impl_item<T: Folder>(i: ImplItem, folder: &mut T)
                                      -> OneVector<ImplItem> {
    smallvec![ImplItem {
        id: folder.new_id(i.id),
        vis: folder.fold_vis(i.vis),
        ident: folder.fold_ident(i.ident),
        attrs: fold_attrs(i.attrs, folder),
        generics: folder.fold_generics(i.generics),
        defaultness: i.defaultness,
        node: match i.node  {
            ast::ImplItemKind::Const(ty, expr) => {
                ast::ImplItemKind::Const(folder.fold_ty(ty), folder.fold_expr(expr))
            }
            ast::ImplItemKind::Method(sig, body) => {
                ast::ImplItemKind::Method(noop_fold_method_sig(sig, folder),
                               folder.fold_block(body))
            }
            ast::ImplItemKind::Type(ty) => ast::ImplItemKind::Type(folder.fold_ty(ty)),
            ast::ImplItemKind::Existential(bounds) => {
                ast::ImplItemKind::Existential(folder.fold_bounds(bounds))
            },
            ast::ImplItemKind::Macro(mac) => ast::ImplItemKind::Macro(folder.fold_mac(mac))
        },
        span: folder.new_span(i.span),
        tokens: i.tokens,
    }]
}

pub fn noop_fold_fn_header<T: Folder>(mut header: FnHeader, folder: &mut T) -> FnHeader {
    header.asyncness = folder.fold_asyncness(header.asyncness);
    header
}

pub fn noop_fold_mod<T: Folder>(Mod {inner, items}: Mod, folder: &mut T) -> Mod {
    Mod {
        inner: folder.new_span(inner),
        items: items.move_flat_map(|x| folder.fold_item(x)),
    }
}

pub fn noop_fold_crate<T: Folder>(Crate {module, attrs, span}: Crate,
                                  folder: &mut T) -> Crate {
    let mut items = folder.fold_item(P(ast::Item {
        ident: keywords::Invalid.ident(),
        attrs,
        id: ast::DUMMY_NODE_ID,
        vis: respan(span.shrink_to_lo(), ast::VisibilityKind::Public),
        span,
        node: ast::ItemKind::Mod(module),
        tokens: None,
    })).into_iter();

    let (module, attrs, span) = match items.next() {
        Some(item) => {
            assert!(items.next().is_none(),
                    "a crate cannot expand to more than one item");
            item.and_then(|ast::Item { attrs, span, node, .. }| {
                match node {
                    ast::ItemKind::Mod(m) => (m, attrs, span),
                    _ => panic!("fold converted a module to not a module"),
                }
            })
        }
        None => (ast::Mod {
            inner: span,
            items: vec![],
        }, vec![], span)
    };

    Crate {
        module,
        attrs,
        span,
    }
}

// fold one item into possibly many items
pub fn noop_fold_item<T: Folder>(i: P<Item>, folder: &mut T) -> OneVector<P<Item>> {
    smallvec![i.map(|i| folder.fold_item_simple(i))]
}

// fold one item into exactly one item
pub fn noop_fold_item_simple<T: Folder>(Item {id, ident, attrs, node, vis, span, tokens}: Item,
                                        folder: &mut T) -> Item {
    Item {
        id: folder.new_id(id),
        vis: folder.fold_vis(vis),
        ident: folder.fold_ident(ident),
        attrs: fold_attrs(attrs, folder),
        node: folder.fold_item_kind(node),
        span: folder.new_span(span),

        // FIXME: if this is replaced with a call to `folder.fold_tts` it causes
        //        an ICE during resolve... odd!
        tokens,
    }
}

pub fn noop_fold_foreign_item<T: Folder>(ni: ForeignItem, folder: &mut T)
-> OneVector<ForeignItem> {
    smallvec![folder.fold_foreign_item_simple(ni)]
}

pub fn noop_fold_foreign_item_simple<T: Folder>(ni: ForeignItem, folder: &mut T) -> ForeignItem {
    ForeignItem {
        id: folder.new_id(ni.id),
        vis: folder.fold_vis(ni.vis),
        ident: folder.fold_ident(ni.ident),
        attrs: fold_attrs(ni.attrs, folder),
        node: match ni.node {
            ForeignItemKind::Fn(fdec, generics) => {
                ForeignItemKind::Fn(folder.fold_fn_decl(fdec), folder.fold_generics(generics))
            }
            ForeignItemKind::Static(t, m) => {
                ForeignItemKind::Static(folder.fold_ty(t), m)
            }
            ForeignItemKind::Ty => ForeignItemKind::Ty,
            ForeignItemKind::Macro(mac) => ForeignItemKind::Macro(folder.fold_mac(mac)),
        },
        span: folder.new_span(ni.span)
    }
}

pub fn noop_fold_method_sig<T: Folder>(sig: MethodSig, folder: &mut T) -> MethodSig {
    MethodSig {
        header: folder.fold_fn_header(sig.header),
        decl: folder.fold_fn_decl(sig.decl)
    }
}

pub fn noop_fold_pat<T: Folder>(p: P<Pat>, folder: &mut T) -> P<Pat> {
    p.map(|Pat {id, node, span}| Pat {
        id: folder.new_id(id),
        node: match node {
            PatKind::Wild => PatKind::Wild,
            PatKind::Ident(binding_mode, ident, sub) => {
                PatKind::Ident(binding_mode,
                               folder.fold_ident(ident),
                               sub.map(|x| folder.fold_pat(x)))
            }
            PatKind::Lit(e) => PatKind::Lit(folder.fold_expr(e)),
            PatKind::TupleStruct(pth, pats, ddpos) => {
                PatKind::TupleStruct(folder.fold_path(pth),
                        pats.move_map(|x| folder.fold_pat(x)), ddpos)
            }
            PatKind::Path(qself, pth) => {
                let (qself, pth) = folder.fold_qpath(qself, pth);
                PatKind::Path(qself, pth)
            }
            PatKind::Struct(pth, fields, etc) => {
                let pth = folder.fold_path(pth);
                let fs = fields.move_map(|f| {
                    Spanned { span: folder.new_span(f.span),
                              node: ast::FieldPat {
                                  ident: folder.fold_ident(f.node.ident),
                                  pat: folder.fold_pat(f.node.pat),
                                  is_shorthand: f.node.is_shorthand,
                                  attrs: fold_attrs(f.node.attrs.into(), folder).into()
                              }}
                });
                PatKind::Struct(pth, fs, etc)
            }
            PatKind::Tuple(elts, ddpos) => {
                PatKind::Tuple(elts.move_map(|x| folder.fold_pat(x)), ddpos)
            }
            PatKind::Box(inner) => PatKind::Box(folder.fold_pat(inner)),
            PatKind::Ref(inner, mutbl) => PatKind::Ref(folder.fold_pat(inner), mutbl),
            PatKind::Range(e1, e2, Spanned { span, node: end }) => {
                PatKind::Range(folder.fold_expr(e1),
                               folder.fold_expr(e2),
                               Spanned { span, node: folder.fold_range_end(end) })
            },
            PatKind::Slice(before, slice, after) => {
                PatKind::Slice(before.move_map(|x| folder.fold_pat(x)),
                       slice.map(|x| folder.fold_pat(x)),
                       after.move_map(|x| folder.fold_pat(x)))
            }
            PatKind::Paren(inner) => PatKind::Paren(folder.fold_pat(inner)),
            PatKind::Mac(mac) => PatKind::Mac(folder.fold_mac(mac))
        },
        span: folder.new_span(span)
    })
}

pub fn noop_fold_range_end<T: Folder>(end: RangeEnd, _folder: &mut T) -> RangeEnd {
    end
}

pub fn noop_fold_anon_const<T: Folder>(constant: AnonConst, folder: &mut T) -> AnonConst {
    let AnonConst {id, value} = constant;
    AnonConst {
        id: folder.new_id(id),
        value: folder.fold_expr(value),
    }
}

pub fn noop_fold_expr<T: Folder>(Expr {id, node, span, attrs}: Expr, folder: &mut T) -> Expr {
    Expr {
        node: match node {
            ExprKind::Box(e) => {
                ExprKind::Box(folder.fold_expr(e))
            }
            ExprKind::ObsoleteInPlace(a, b) => {
                ExprKind::ObsoleteInPlace(folder.fold_expr(a), folder.fold_expr(b))
            }
            ExprKind::Array(exprs) => {
                ExprKind::Array(folder.fold_exprs(exprs))
            }
            ExprKind::Repeat(expr, count) => {
                ExprKind::Repeat(folder.fold_expr(expr), folder.fold_anon_const(count))
            }
            ExprKind::Tup(exprs) => ExprKind::Tup(folder.fold_exprs(exprs)),
            ExprKind::Call(f, args) => {
                ExprKind::Call(folder.fold_expr(f),
                         folder.fold_exprs(args))
            }
            ExprKind::MethodCall(seg, args) => {
                ExprKind::MethodCall(
                    PathSegment {
                        ident: folder.fold_ident(seg.ident),
                        args: seg.args.map(|args| {
                            args.map(|args| folder.fold_generic_args(args))
                        }),
                    },
                    folder.fold_exprs(args))
            }
            ExprKind::Binary(binop, lhs, rhs) => {
                ExprKind::Binary(binop,
                        folder.fold_expr(lhs),
                        folder.fold_expr(rhs))
            }
            ExprKind::Unary(binop, ohs) => {
                ExprKind::Unary(binop, folder.fold_expr(ohs))
            }
            ExprKind::Lit(l) => ExprKind::Lit(l),
            ExprKind::Cast(expr, ty) => {
                ExprKind::Cast(folder.fold_expr(expr), folder.fold_ty(ty))
            }
            ExprKind::Type(expr, ty) => {
                ExprKind::Type(folder.fold_expr(expr), folder.fold_ty(ty))
            }
            ExprKind::AddrOf(m, ohs) => ExprKind::AddrOf(m, folder.fold_expr(ohs)),
            ExprKind::If(cond, tr, fl) => {
                ExprKind::If(folder.fold_expr(cond),
                       folder.fold_block(tr),
                       fl.map(|x| folder.fold_expr(x)))
            }
            ExprKind::IfLet(pats, expr, tr, fl) => {
                ExprKind::IfLet(pats.move_map(|pat| folder.fold_pat(pat)),
                          folder.fold_expr(expr),
                          folder.fold_block(tr),
                          fl.map(|x| folder.fold_expr(x)))
            }
            ExprKind::While(cond, body, opt_label) => {
                ExprKind::While(folder.fold_expr(cond),
                          folder.fold_block(body),
                          opt_label.map(|label| folder.fold_label(label)))
            }
            ExprKind::WhileLet(pats, expr, body, opt_label) => {
                ExprKind::WhileLet(pats.move_map(|pat| folder.fold_pat(pat)),
                             folder.fold_expr(expr),
                             folder.fold_block(body),
                             opt_label.map(|label| folder.fold_label(label)))
            }
            ExprKind::ForLoop(pat, iter, body, opt_label) => {
                ExprKind::ForLoop(folder.fold_pat(pat),
                            folder.fold_expr(iter),
                            folder.fold_block(body),
                            opt_label.map(|label| folder.fold_label(label)))
            }
            ExprKind::Loop(body, opt_label) => {
                ExprKind::Loop(folder.fold_block(body),
                               opt_label.map(|label| folder.fold_label(label)))
            }
            ExprKind::Match(expr, arms) => {
                ExprKind::Match(folder.fold_expr(expr),
                          arms.move_map(|x| folder.fold_arm(x)))
            }
            ExprKind::Closure(capture_clause, asyncness, movability, decl, body, span) => {
                ExprKind::Closure(capture_clause,
                                  folder.fold_asyncness(asyncness),
                                  movability,
                                  folder.fold_fn_decl(decl),
                                  folder.fold_expr(body),
                                  folder.new_span(span))
            }
            ExprKind::Block(blk, opt_label) => {
                ExprKind::Block(folder.fold_block(blk),
                                opt_label.map(|label| folder.fold_label(label)))
            }
            ExprKind::Async(capture_clause, node_id, body) => {
                ExprKind::Async(
                    capture_clause,
                    folder.new_id(node_id),
                    folder.fold_block(body),
                )
            }
            ExprKind::Assign(el, er) => {
                ExprKind::Assign(folder.fold_expr(el), folder.fold_expr(er))
            }
            ExprKind::AssignOp(op, el, er) => {
                ExprKind::AssignOp(op,
                            folder.fold_expr(el),
                            folder.fold_expr(er))
            }
            ExprKind::Field(el, ident) => {
                ExprKind::Field(folder.fold_expr(el), folder.fold_ident(ident))
            }
            ExprKind::Index(el, er) => {
                ExprKind::Index(folder.fold_expr(el), folder.fold_expr(er))
            }
            ExprKind::Range(e1, e2, lim) => {
                ExprKind::Range(e1.map(|x| folder.fold_expr(x)),
                                e2.map(|x| folder.fold_expr(x)),
                                lim)
            }
            ExprKind::Path(qself, path) => {
                let (qself, path) = folder.fold_qpath(qself, path);
                ExprKind::Path(qself, path)
            }
            ExprKind::Break(opt_label, opt_expr) => {
                ExprKind::Break(opt_label.map(|label| folder.fold_label(label)),
                                opt_expr.map(|e| folder.fold_expr(e)))
            }
            ExprKind::Continue(opt_label) => {
                ExprKind::Continue(opt_label.map(|label| folder.fold_label(label)))
            }
            ExprKind::Ret(e) => ExprKind::Ret(e.map(|x| folder.fold_expr(x))),
            ExprKind::InlineAsm(asm) => ExprKind::InlineAsm(asm.map(|asm| {
                InlineAsm {
                    inputs: asm.inputs.move_map(|(c, input)| {
                        (c, folder.fold_expr(input))
                    }),
                    outputs: asm.outputs.move_map(|out| {
                        InlineAsmOutput {
                            constraint: out.constraint,
                            expr: folder.fold_expr(out.expr),
                            is_rw: out.is_rw,
                            is_indirect: out.is_indirect,
                        }
                    }),
                    ..asm
                }
            })),
            ExprKind::Mac(mac) => ExprKind::Mac(folder.fold_mac(mac)),
            ExprKind::Struct(path, fields, maybe_expr) => {
                ExprKind::Struct(folder.fold_path(path),
                        fields.move_map(|x| folder.fold_field(x)),
                        maybe_expr.map(|x| folder.fold_expr(x)))
            },
            ExprKind::Paren(ex) => {
                let sub_expr = folder.fold_expr(ex);
                return Expr {
                    // Nodes that are equal modulo `Paren` sugar no-ops should have the same ids.
                    id: sub_expr.id,
                    node: ExprKind::Paren(sub_expr),
                    span: folder.new_span(span),
                    attrs: fold_attrs(attrs.into(), folder).into(),
                };
            }
            ExprKind::Yield(ex) => ExprKind::Yield(ex.map(|x| folder.fold_expr(x))),
            ExprKind::Try(ex) => ExprKind::Try(folder.fold_expr(ex)),
            ExprKind::Catch(body) => ExprKind::Catch(folder.fold_block(body)),
        },
        id: folder.new_id(id),
        span: folder.new_span(span),
        attrs: fold_attrs(attrs.into(), folder).into(),
    }
}

pub fn noop_fold_opt_expr<T: Folder>(e: P<Expr>, folder: &mut T) -> Option<P<Expr>> {
    Some(folder.fold_expr(e))
}

pub fn noop_fold_exprs<T: Folder>(es: Vec<P<Expr>>, folder: &mut T) -> Vec<P<Expr>> {
    es.move_flat_map(|e| folder.fold_opt_expr(e))
}

pub fn noop_fold_stmt<T: Folder>(Stmt {node, span, id}: Stmt, folder: &mut T) -> OneVector<Stmt> {
    let id = folder.new_id(id);
    let span = folder.new_span(span);
    noop_fold_stmt_kind(node, folder).into_iter().map(|node| {
        Stmt { id: id, node: node, span: span }
    }).collect()
}

pub fn noop_fold_stmt_kind<T: Folder>(node: StmtKind, folder: &mut T) -> OneVector<StmtKind> {
    match node {
        StmtKind::Local(local) => smallvec![StmtKind::Local(folder.fold_local(local))],
        StmtKind::Item(item) => folder.fold_item(item).into_iter().map(StmtKind::Item).collect(),
        StmtKind::Expr(expr) => {
            folder.fold_opt_expr(expr).into_iter().map(StmtKind::Expr).collect()
        }
        StmtKind::Semi(expr) => {
            folder.fold_opt_expr(expr).into_iter().map(StmtKind::Semi).collect()
        }
        StmtKind::Mac(mac) => smallvec![StmtKind::Mac(mac.map(|(mac, semi, attrs)| {
            (folder.fold_mac(mac), semi, fold_attrs(attrs.into(), folder).into())
        }))],
    }
}

pub fn noop_fold_vis<T: Folder>(vis: Visibility, folder: &mut T) -> Visibility {
    match vis.node {
        VisibilityKind::Restricted { path, id } => {
            respan(vis.span, VisibilityKind::Restricted {
                path: path.map(|path| folder.fold_path(path)),
                id: folder.new_id(id),
            })
        }
        _ => vis,
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use ast::{self, Ident};
    use util::parser_testing::{string_to_crate, matches_codepattern};
    use print::pprust;
    use fold;
    use with_globals;
    use super::*;

    // this version doesn't care about getting comments or docstrings in.
    fn fake_print_crate(s: &mut pprust::State,
                        krate: &ast::Crate) -> io::Result<()> {
        s.print_mod(&krate.module, &krate.attrs)
    }

    // change every identifier to "zz"
    struct ToZzIdentFolder;

    impl Folder for ToZzIdentFolder {
        fn fold_ident(&mut self, _: ast::Ident) -> ast::Ident {
            Ident::from_str("zz")
        }
        fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
            fold::noop_fold_mac(mac, self)
        }
    }

    // maybe add to expand.rs...
    macro_rules! assert_pred {
        ($pred:expr, $predname:expr, $a:expr , $b:expr) => (
            {
                let pred_val = $pred;
                let a_val = $a;
                let b_val = $b;
                if !(pred_val(&a_val, &b_val)) {
                    panic!("expected args satisfying {}, got {} and {}",
                          $predname, a_val, b_val);
                }
            }
        )
    }

    // make sure idents get transformed everywhere
    #[test] fn ident_transformation () {
        with_globals(|| {
            let mut zz_fold = ToZzIdentFolder;
            let ast = string_to_crate(
                "#[a] mod b {fn c (d : e, f : g) {h!(i,j,k);l;m}}".to_string());
            let folded_crate = zz_fold.fold_crate(ast);
            assert_pred!(
                matches_codepattern,
                "matches_codepattern",
                pprust::to_string(|s| fake_print_crate(s, &folded_crate)),
                "#[zz]mod zz{fn zz(zz:zz,zz:zz){zz!(zz,zz,zz);zz;zz}}".to_string());
        })
    }

    // even inside macro defs....
    #[test] fn ident_transformation_in_defs () {
        with_globals(|| {
            let mut zz_fold = ToZzIdentFolder;
            let ast = string_to_crate(
                "macro_rules! a {(b $c:expr $(d $e:token)f+ => \
                (g $(d $d $e)+))} ".to_string());
            let folded_crate = zz_fold.fold_crate(ast);
            assert_pred!(
                matches_codepattern,
                "matches_codepattern",
                pprust::to_string(|s| fake_print_crate(s, &folded_crate)),
                "macro_rules! zz((zz$zz:zz$(zz $zz:zz)zz+=>(zz$(zz$zz$zz)+)));".to_string());
        })
    }
}
