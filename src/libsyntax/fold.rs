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
use ast_util;
use codemap::{respan, Span, Spanned};
use parse::token;
use owned_slice::OwnedSlice;
use util::small_vector::SmallVector;

use std::rc::Rc;
use std::gc::{Gc, GC};

// We may eventually want to be able to fold over type parameters, too.
pub trait Folder {
    fn fold_crate(&mut self, c: Crate) -> Crate {
        noop_fold_crate(c, self)
    }

    fn fold_meta_items(&mut self, meta_items: &[Gc<MetaItem>]) -> Vec<Gc<MetaItem>> {
        meta_items.iter().map(|x| fold_meta_item_(*x, self)).collect()
    }

    fn fold_view_path(&mut self, view_path: Gc<ViewPath>) -> Gc<ViewPath> {
        let inner_view_path = match view_path.node {
            ViewPathSimple(ref ident, ref path, node_id) => {
                let id = self.new_id(node_id);
                ViewPathSimple(ident.clone(),
                               self.fold_path(path),
                               id)
            }
            ViewPathGlob(ref path, node_id) => {
                let id = self.new_id(node_id);
                ViewPathGlob(self.fold_path(path), id)
            }
            ViewPathList(ref path, ref path_list_idents, node_id) => {
                let id = self.new_id(node_id);
                ViewPathList(self.fold_path(path),
                             path_list_idents.iter().map(|path_list_ident| {
                                let id = self.new_id(path_list_ident.node
                                                                    .id);
                                Spanned {
                                    node: PathListIdent_ {
                                        name: path_list_ident.node
                                                             .name
                                                             .clone(),
                                        id: id,
                                    },
                                    span: self.new_span(
                                        path_list_ident.span)
                                }
                             }).collect(),
                             id)
            }
        };
        box(GC) Spanned {
            node: inner_view_path,
            span: self.new_span(view_path.span),
        }
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

    fn fold_struct_field(&mut self, sf: &StructField) -> StructField {
        let id = self.new_id(sf.node.id);
        Spanned {
            node: ast::StructField_ {
                kind: sf.node.kind,
                id: id,
                ty: self.fold_ty(sf.node.ty),
                attrs: sf.node.attrs.iter().map(|e| self.fold_attribute(*e)).collect()
            },
            span: self.new_span(sf.span)
        }
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

    fn fold_method(&mut self, m: Gc<Method>) -> Gc<Method>  {
        noop_fold_method(&*m, self)
    }

    fn fold_block(&mut self, b: P<Block>) -> P<Block> {
        noop_fold_block(b, self)
    }

    fn fold_stmt(&mut self, s: &Stmt) -> SmallVector<Gc<Stmt>> {
        noop_fold_stmt(s, self)
    }

    fn fold_arm(&mut self, a: &Arm) -> Arm {
        Arm {
            attrs: a.attrs.iter().map(|x| self.fold_attribute(*x)).collect(),
            pats: a.pats.iter().map(|x| self.fold_pat(*x)).collect(),
            guard: a.guard.map(|x| self.fold_expr(x)),
            body: self.fold_expr(a.body),
        }
    }

    fn fold_pat(&mut self, p: Gc<Pat>) -> Gc<Pat> {
        noop_fold_pat(p, self)
    }

    fn fold_decl(&mut self, d: Gc<Decl>) -> SmallVector<Gc<Decl>> {
        let node = match d.node {
            DeclLocal(ref l) => SmallVector::one(DeclLocal(self.fold_local(*l))),
            DeclItem(it) => {
                self.fold_item(it).move_iter().map(|i| DeclItem(i)).collect()
            }
        };

        node.move_iter().map(|node| {
            box(GC) Spanned {
                node: node,
                span: self.new_span(d.span),
            }
        }).collect()
    }

    fn fold_expr(&mut self, e: Gc<Expr>) -> Gc<Expr> {
        noop_fold_expr(e, self)
    }

    fn fold_ty(&mut self, t: P<Ty>) -> P<Ty> {
        let id = self.new_id(t.id);
        let node = match t.node {
            TyNil | TyBot | TyInfer => t.node.clone(),
            TyBox(ty) => TyBox(self.fold_ty(ty)),
            TyUniq(ty) => TyUniq(self.fold_ty(ty)),
            TyVec(ty) => TyVec(self.fold_ty(ty)),
            TyPtr(ref mt) => TyPtr(fold_mt(mt, self)),
            TyRptr(ref region, ref mt) => {
                TyRptr(fold_opt_lifetime(region, self), fold_mt(mt, self))
            }
            TyClosure(ref f, ref region) => {
                TyClosure(box(GC) ClosureTy {
                    fn_style: f.fn_style,
                    onceness: f.onceness,
                    bounds: fold_opt_bounds(&f.bounds, self),
                    decl: self.fold_fn_decl(&*f.decl),
                    lifetimes: f.lifetimes.iter().map(|l| self.fold_lifetime(l)).collect(),
                }, fold_opt_lifetime(region, self))
            }
            TyProc(ref f) => {
                TyProc(box(GC) ClosureTy {
                    fn_style: f.fn_style,
                    onceness: f.onceness,
                    bounds: fold_opt_bounds(&f.bounds, self),
                    decl: self.fold_fn_decl(&*f.decl),
                    lifetimes: f.lifetimes.iter().map(|l| self.fold_lifetime(l)).collect(),
                })
            }
            TyBareFn(ref f) => {
                TyBareFn(box(GC) BareFnTy {
                    lifetimes: f.lifetimes.iter().map(|l| self.fold_lifetime(l)).collect(),
                    fn_style: f.fn_style,
                    abi: f.abi,
                    decl: self.fold_fn_decl(&*f.decl)
                })
            }
            TyUnboxedFn(ref f) => {
                TyUnboxedFn(box(GC) UnboxedFnTy {
                    decl: self.fold_fn_decl(&*f.decl),
                })
            }
            TyTup(ref tys) => TyTup(tys.iter().map(|&ty| self.fold_ty(ty)).collect()),
            TyParen(ref ty) => TyParen(self.fold_ty(*ty)),
            TyPath(ref path, ref bounds, id) => {
                let id = self.new_id(id);
                TyPath(self.fold_path(path),
                       fold_opt_bounds(bounds, self),
                       id)
            }
            TyFixedLengthVec(ty, e) => {
                TyFixedLengthVec(self.fold_ty(ty), self.fold_expr(e))
            }
            TyTypeof(expr) => TyTypeof(self.fold_expr(expr)),
        };
        P(Ty {
            id: id,
            span: self.new_span(t.span),
            node: node,
        })
    }

    fn fold_mod(&mut self, m: &Mod) -> Mod {
        noop_fold_mod(m, self)
    }

    fn fold_foreign_mod(&mut self, nm: &ForeignMod) -> ForeignMod {
        ast::ForeignMod {
            abi: nm.abi,
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

    fn fold_variant(&mut self, v: &Variant) -> P<Variant> {
        let id = self.new_id(v.node.id);
        let kind;
        match v.node.kind {
            TupleVariantKind(ref variant_args) => {
                kind = TupleVariantKind(variant_args.iter().map(|x|
                    fold_variant_arg_(x, self)).collect())
            }
            StructVariantKind(ref struct_def) => {
                kind = StructVariantKind(box(GC) ast::StructDef {
                    fields: struct_def.fields.iter()
                        .map(|f| self.fold_struct_field(f)).collect(),
                    ctor_id: struct_def.ctor_id.map(|c| self.new_id(c)),
                    super_struct: match struct_def.super_struct {
                        Some(t) => Some(self.fold_ty(t)),
                        None => None
                    },
                    is_virtual: struct_def.is_virtual,
                })
            }
        }

        let attrs = v.node.attrs.iter().map(|x| self.fold_attribute(*x)).collect();

        let de = match v.node.disr_expr {
          Some(e) => Some(self.fold_expr(e)),
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
            span: self.new_span(v.span),
        })
    }

    fn fold_ident(&mut self, i: Ident) -> Ident {
        i
    }

    fn fold_path(&mut self, p: &Path) -> Path {
        ast::Path {
            span: self.new_span(p.span),
            global: p.global,
            segments: p.segments.iter().map(|segment| ast::PathSegment {
                identifier: self.fold_ident(segment.identifier),
                lifetimes: segment.lifetimes.iter().map(|l| self.fold_lifetime(l)).collect(),
                types: segment.types.iter().map(|&typ| self.fold_ty(typ)).collect(),
            }).collect()
        }
    }

    fn fold_local(&mut self, l: Gc<Local>) -> Gc<Local> {
        let id = self.new_id(l.id); // Needs to be first, for ast_map.
        box(GC) Local {
            id: id,
            ty: self.fold_ty(l.ty),
            pat: self.fold_pat(l.pat),
            init: l.init.map(|e| self.fold_expr(e)),
            span: self.new_span(l.span),
            source: l.source,
        }
    }

    fn fold_mac(&mut self, macro: &Mac) -> Mac {
        Spanned {
            node: match macro.node {
                MacInvocTT(ref p, ref tts, ctxt) => {
                    MacInvocTT(self.fold_path(p),
                               fold_tts(tts.as_slice(), self),
                               ctxt)
                }
            },
            span: self.new_span(macro.span)
        }
    }

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

    fn fold_explicit_self(&mut self, es: &ExplicitSelf) -> ExplicitSelf {
        Spanned {
            span: self.new_span(es.span),
            node: self.fold_explicit_self_(&es.node)
        }
    }

    fn fold_explicit_self_(&mut self, es: &ExplicitSelf_) -> ExplicitSelf_ {
        match *es {
            SelfStatic | SelfValue | SelfUniq => *es,
            SelfRegion(ref lifetime, m) => {
                SelfRegion(fold_opt_lifetime(lifetime, self), m)
            }
        }
    }

    fn fold_lifetime(&mut self, l: &Lifetime) -> Lifetime {
        noop_fold_lifetime(l, self)
    }

    //used in noop_fold_item and noop_fold_crate
    fn fold_attribute(&mut self, at: Attribute) -> Attribute {
        Spanned {
            span: self.new_span(at.span),
            node: ast::Attribute_ {
                id: at.node.id,
                style: at.node.style,
                value: fold_meta_item_(at.node.value, self),
                is_sugared_doc: at.node.is_sugared_doc
            }
        }
    }


}

/* some little folds that probably aren't useful to have in Folder itself*/

//used in noop_fold_item and noop_fold_crate and noop_fold_crate_directive
fn fold_meta_item_<T: Folder>(mi: Gc<MetaItem>, fld: &mut T) -> Gc<MetaItem> {
    box(GC) Spanned {
        node:
            match mi.node {
                MetaWord(ref id) => MetaWord((*id).clone()),
                MetaList(ref id, ref mis) => {
                    MetaList((*id).clone(), mis.iter().map(|e| fold_meta_item_(*e, fld)).collect())
                }
                MetaNameValue(ref id, ref s) => {
                    MetaNameValue((*id).clone(), (*s).clone())
                }
            },
        span: fld.new_span(mi.span) }
}

//used in noop_fold_foreign_item and noop_fold_fn_decl
fn fold_arg_<T: Folder>(a: &Arg, fld: &mut T) -> Arg {
    let id = fld.new_id(a.id); // Needs to be first, for ast_map.
    Arg {
        id: id,
        ty: fld.fold_ty(a.ty),
        pat: fld.fold_pat(a.pat),
    }
}

pub fn fold_tt<T: Folder>(tt: &TokenTree, fld: &mut T) -> TokenTree {
    match *tt {
        TTTok(span, ref tok) =>
            TTTok(span, fold_token(tok,fld)),
        TTDelim(ref tts) => TTDelim(Rc::new(fold_tts(tts.as_slice(), fld))),
        TTSeq(span, ref pattern, ref sep, is_optional) =>
            TTSeq(span,
                  Rc::new(fold_tts(pattern.as_slice(), fld)),
                  sep.as_ref().map(|tok| fold_token(tok,fld)),
                  is_optional),
        TTNonterminal(sp,ref ident) =>
            TTNonterminal(sp,fld.fold_ident(*ident))
    }
}

pub fn fold_tts<T: Folder>(tts: &[TokenTree], fld: &mut T) -> Vec<TokenTree> {
    tts.iter().map(|tt| fold_tt(tt,fld)).collect()
}


// apply ident folder if it's an ident, apply other folds to interpolated nodes
fn fold_token<T: Folder>(t: &token::Token, fld: &mut T) -> token::Token {
    match *t {
        token::IDENT(id, followed_by_colons) => {
            token::IDENT(fld.fold_ident(id), followed_by_colons)
        }
        token::LIFETIME(id) => token::LIFETIME(fld.fold_ident(id)),
        token::INTERPOLATED(ref nt) => token::INTERPOLATED(fold_interpolated(nt,fld)),
        _ => (*t).clone()
    }
}

// apply folder to elements of interpolated nodes
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
fn fold_interpolated<T: Folder>(nt : &token::Nonterminal, fld: &mut T) -> token::Nonterminal {
    match *nt {
        token::NtItem(item) =>
            token::NtItem(fld.fold_item(item)
                          .expect_one("expected fold to produce exactly one item")),
        token::NtBlock(block) => token::NtBlock(fld.fold_block(block)),
        token::NtStmt(stmt) =>
            token::NtStmt(fld.fold_stmt(stmt)
                          .expect_one("expected fold to produce exactly one statement")),
        token::NtPat(pat) => token::NtPat(fld.fold_pat(pat)),
        token::NtExpr(expr) => token::NtExpr(fld.fold_expr(expr)),
        token::NtTy(ty) => token::NtTy(fld.fold_ty(ty)),
        token::NtIdent(ref id, is_mod_name) =>
            token::NtIdent(box fld.fold_ident(**id),is_mod_name),
        token::NtMeta(meta_item) => token::NtMeta(fold_meta_item_(meta_item,fld)),
        token::NtPath(ref path) => token::NtPath(box fld.fold_path(*path)),
        token::NtTT(tt) => token::NtTT(box (GC) fold_tt(tt,fld)),
        // it looks to me like we can leave out the matchers: token::NtMatchers(matchers)
        _ => (*nt).clone()
    }
}

pub fn noop_fold_fn_decl<T: Folder>(decl: &FnDecl, fld: &mut T) -> P<FnDecl> {
    P(FnDecl {
        inputs: decl.inputs.iter().map(|x| fold_arg_(x, fld)).collect(), // bad copy
        output: fld.fold_ty(decl.output),
        cf: decl.cf,
        variadic: decl.variadic
    })
}

fn fold_ty_param_bound<T: Folder>(tpb: &TyParamBound, fld: &mut T)
                                    -> TyParamBound {
    match *tpb {
        TraitTyParamBound(ref ty) => TraitTyParamBound(fold_trait_ref(ty, fld)),
        StaticRegionTyParamBound => StaticRegionTyParamBound,
        UnboxedFnTyParamBound(ref unboxed_function_type) => {
            UnboxedFnTyParamBound(UnboxedFnTy {
                decl: fld.fold_fn_decl(&*unboxed_function_type.decl),
            })
        }
        OtherRegionTyParamBound(s) => OtherRegionTyParamBound(s)
    }
}

pub fn fold_ty_param<T: Folder>(tp: &TyParam, fld: &mut T) -> TyParam {
    let id = fld.new_id(tp.id);
    TyParam {
        ident: tp.ident,
        id: id,
        bounds: tp.bounds.map(|x| fold_ty_param_bound(x, fld)),
        unbound: tp.unbound.as_ref().map(|x| fold_ty_param_bound(x, fld)),
        default: tp.default.map(|x| fld.fold_ty(x)),
        span: tp.span
    }
}

pub fn fold_ty_params<T: Folder>(tps: &OwnedSlice<TyParam>, fld: &mut T)
                                   -> OwnedSlice<TyParam> {
    tps.map(|tp| fold_ty_param(tp, fld))
}

pub fn noop_fold_lifetime<T: Folder>(l: &Lifetime, fld: &mut T) -> Lifetime {
    let id = fld.new_id(l.id);
    Lifetime {
        id: id,
        span: fld.new_span(l.span),
        name: l.name
    }
}

pub fn fold_lifetimes<T: Folder>(lts: &Vec<Lifetime>, fld: &mut T)
                                   -> Vec<Lifetime> {
    lts.iter().map(|l| fld.fold_lifetime(l)).collect()
}

pub fn fold_opt_lifetime<T: Folder>(o_lt: &Option<Lifetime>, fld: &mut T)
                                      -> Option<Lifetime> {
    o_lt.as_ref().map(|lt| fld.fold_lifetime(lt))
}

pub fn fold_generics<T: Folder>(generics: &Generics, fld: &mut T) -> Generics {
    Generics {ty_params: fold_ty_params(&generics.ty_params, fld),
              lifetimes: fold_lifetimes(&generics.lifetimes, fld)}
}

fn fold_struct_def<T: Folder>(struct_def: Gc<StructDef>,
                              fld: &mut T) -> Gc<StructDef> {
    box(GC) ast::StructDef {
        fields: struct_def.fields.iter().map(|f| fold_struct_field(f, fld)).collect(),
        ctor_id: struct_def.ctor_id.map(|cid| fld.new_id(cid)),
        super_struct: match struct_def.super_struct {
            Some(t) => Some(fld.fold_ty(t)),
            None => None
        },
        is_virtual: struct_def.is_virtual,
    }
}

fn fold_trait_ref<T: Folder>(p: &TraitRef, fld: &mut T) -> TraitRef {
    let id = fld.new_id(p.ref_id);
    ast::TraitRef {
        path: fld.fold_path(&p.path),
        ref_id: id,
    }
}

fn fold_struct_field<T: Folder>(f: &StructField, fld: &mut T) -> StructField {
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

fn fold_field_<T: Folder>(field: Field, folder: &mut T) -> Field {
    ast::Field {
        ident: respan(field.ident.span, folder.fold_ident(field.ident.node)),
        expr: folder.fold_expr(field.expr),
        span: folder.new_span(field.span),
    }
}

fn fold_mt<T: Folder>(mt: &MutTy, folder: &mut T) -> MutTy {
    MutTy {
        ty: folder.fold_ty(mt.ty),
        mutbl: mt.mutbl,
    }
}

fn fold_opt_bounds<T: Folder>(b: &Option<OwnedSlice<TyParamBound>>, folder: &mut T)
                              -> Option<OwnedSlice<TyParamBound>> {
    b.as_ref().map(|bounds| {
        bounds.map(|bound| {
            fold_ty_param_bound(bound, folder)
        })
    })
}

fn fold_variant_arg_<T: Folder>(va: &VariantArg, folder: &mut T) -> VariantArg {
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
                fold_generics(generics, folder),
                folder.fold_block(body)
            )
        }
        ItemMod(ref m) => ItemMod(folder.fold_mod(m)),
        ItemForeignMod(ref nm) => ItemForeignMod(folder.fold_foreign_mod(nm)),
        ItemTy(t, ref generics) => {
            ItemTy(folder.fold_ty(t), fold_generics(generics, folder))
        }
        ItemEnum(ref enum_definition, ref generics) => {
            ItemEnum(
                ast::EnumDef {
                    variants: enum_definition.variants.iter().map(|&x| {
                        folder.fold_variant(&*x)
                    }).collect(),
                },
                fold_generics(generics, folder))
        }
        ItemStruct(ref struct_def, ref generics) => {
            let struct_def = fold_struct_def(*struct_def, folder);
            ItemStruct(struct_def, fold_generics(generics, folder))
        }
        ItemImpl(ref generics, ref ifce, ty, ref methods) => {
            ItemImpl(fold_generics(generics, folder),
                     ifce.as_ref().map(|p| fold_trait_ref(p, folder)),
                     folder.fold_ty(ty),
                     methods.iter().map(|x| folder.fold_method(*x)).collect()
            )
        }
        ItemTrait(ref generics, ref unbound, ref traits, ref methods) => {
            let methods = methods.iter().map(|method| {
                match *method {
                    Required(ref m) => Required(folder.fold_type_method(m)),
                    Provided(method) => Provided(folder.fold_method(method))
                }
            }).collect();
            ItemTrait(fold_generics(generics, folder),
                      unbound.clone(),
                      traits.iter().map(|p| fold_trait_ref(p, folder)).collect(),
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
        decl: fld.fold_fn_decl(&*m.decl),
        generics: fold_generics(&m.generics, fld),
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
        config: c.config.iter().map(|x| fold_meta_item_(*x, folder)).collect(),
        span: folder.new_span(c.span),
    }
}

// fold one item into possibly many items
pub fn noop_fold_item<T: Folder>(i: &Item,
                                 folder: &mut T) -> SmallVector<Gc<Item>> {
    SmallVector::one(box(GC) noop_fold_item_(i,folder))
}


// fold one item into exactly one item
pub fn noop_fold_item_<T: Folder>(i: &Item, folder: &mut T) -> Item {
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
                    inputs: fdec.inputs.iter().map(|a| fold_arg_(a, folder)).collect(),
                    output: folder.fold_ty(fdec.output),
                    cf: fdec.cf,
                    variadic: fdec.variadic
                }), fold_generics(generics, folder))
            }
            ForeignItemStatic(t, m) => {
                ForeignItemStatic(folder.fold_ty(t), m)
            }
        },
        span: folder.new_span(ni.span),
        vis: ni.vis,
    }
}

pub fn noop_fold_method<T: Folder>(m: &Method, folder: &mut T) -> Gc<Method> {
    let id = folder.new_id(m.id); // Needs to be first, for ast_map.
    box(GC) Method {
        id: id,
        ident: folder.fold_ident(m.ident),
        attrs: m.attrs.iter().map(|a| folder.fold_attribute(*a)).collect(),
        generics: fold_generics(&m.generics, folder),
        explicit_self: folder.fold_explicit_self(&m.explicit_self),
        fn_style: m.fn_style,
        decl: folder.fold_fn_decl(&*m.decl),
        body: folder.fold_block(m.body),
        span: folder.new_span(m.span),
        vis: m.vis
    }
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
        ExprBlock(ref blk) => ExprBlock(folder.fold_block(blk.clone())),
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
                       fields.iter().map(|x| fold_field_(*x, folder)).collect(),
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
            pprust::to_str(|s| fake_print_crate(s, &folded_crate)),
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
            pprust::to_str(|s| fake_print_crate(s, &folded_crate)),
            "zz!zz((zz$zz:zz$(zz $zz:zz)zz+=>(zz$(zz$zz$zz)+)))".to_string());
    }
}
