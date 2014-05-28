// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//use ast::*;
use ast;
use ast_util;
use codemap::{respan, Span, Spanned};
use parse::token;
use owned_slice::OwnedSlice;
use util::small_vector::SmallVector;

use std::rc::Rc;

// We may eventually want to be able to fold over type parameters, too.
pub trait Folder {
    fn fold_crate(&mut self, c: ast::Crate) -> ast::Crate {
        noop_fold_crate(c, self)
    }

    fn fold_meta_items(&mut self, meta_items: &[@ast::MetaItem]) -> Vec<@ast::MetaItem> {
        meta_items.iter().map(|x| fold_meta_item_(*x, self)).collect()
    }

    fn fold_view_path(&mut self, view_path: @ast::ViewPath) -> @ast::ViewPath {
        let inner_view_path = match view_path.node {
            ast::ViewPathSimple(ref ident, ref path, node_id) => {
                let id = self.new_id(node_id);
                ast::ViewPathSimple(ident.clone(),
                                    self.fold_path(path),
                                    id)
            }
            ast::ViewPathGlob(ref path, node_id) => {
                let id = self.new_id(node_id);
                ast::ViewPathGlob(self.fold_path(path), id)
            }
            ast::ViewPathList(ref path, ref path_list_idents, node_id) => {
                let id = self.new_id(node_id);
                ast::ViewPathList(self.fold_path(path),
                                  path_list_idents.iter().map(|path_list_ident| {
                                     let id = self.new_id(path_list_ident.node
                                                                         .id);
                                     Spanned {
                                         node: ast::PathListIdent_ {
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
        @Spanned {
            node: inner_view_path,
            span: self.new_span(view_path.span),
        }
    }

    fn fold_view_item(&mut self, vi: &ast::ViewItem) -> ast::ViewItem {
        noop_fold_view_item(vi, self)
    }

    fn fold_foreign_item(&mut self, ni: @ast::ForeignItem) -> @ast::ForeignItem {
        noop_fold_foreign_item(ni, self)
    }

    fn fold_item(&mut self, i: @ast::Item) -> SmallVector<@ast::Item> {
        noop_fold_item(i, self)
    }

    fn fold_struct_field(&mut self, sf: &ast::StructField) -> ast::StructField {
        let id = self.new_id(sf.node.id);
        Spanned {
            node: ast::StructField_ {
                kind: sf.node.kind,
                id: id,
                ty: self.fold_ty(sf.node.ty),
                attrs: sf.node.attrs.iter().map(|e| fold_attribute_(*e, self)).collect()
            },
            span: self.new_span(sf.span)
        }
    }

    fn fold_item_underscore(&mut self, i: &ast::Item_) -> ast::Item_ {
        noop_fold_item_underscore(i, self)
    }

    fn fold_fn_decl(&mut self, d: &ast::FnDecl) -> ast::P<ast::FnDecl> {
        noop_fold_fn_decl(d, self)
    }

    fn fold_type_method(&mut self, m: &ast::TypeMethod) -> ast::TypeMethod {
        noop_fold_type_method(m, self)
    }

    fn fold_method(&mut self, m: @ast::Method) -> @ast::Method {
        noop_fold_method(m, self)
    }

    fn fold_block(&mut self, b: ast::P<ast::Block>) -> ast::P<ast::Block> {
        noop_fold_block(b, self)
    }

    fn fold_stmt(&mut self, s: &ast::Stmt) -> SmallVector<@ast::Stmt> {
        noop_fold_stmt(s, self)
    }

    fn fold_arm(&mut self, a: &ast::Arm) -> ast::Arm {
        ast::Arm {
            attrs: a.attrs.iter().map(|x| fold_attribute_(*x, self)).collect(),
            pats: a.pats.iter().map(|x| self.fold_pat(*x)).collect(),
            guard: a.guard.map(|x| self.fold_expr(x)),
            body: self.fold_expr(a.body),
        }
    }

    fn fold_pat(&mut self, p: @ast::Pat) -> @ast::Pat {
        noop_fold_pat(p, self)
    }

    fn fold_decl(&mut self, d: @ast::Decl) -> SmallVector<@ast::Decl> {
        let node = match d.node {
            ast::DeclLocal(ref l) => SmallVector::one(ast::DeclLocal(self.fold_local(*l))),
            ast::DeclItem(it) => {
                self.fold_item(it).move_iter().map(|i| ast::DeclItem(i)).collect()
            }
        };

        node.move_iter().map(|node| {
            @Spanned {
                node: node,
                span: self.new_span(d.span),
            }
        }).collect()
    }

    fn fold_expr(&mut self, e: @ast::Expr) -> @ast::Expr {
        noop_fold_expr(e, self)
    }

    fn fold_ty(&mut self, t: ast::P<ast::Ty>) -> ast::P<ast::Ty> {
        let id = self.new_id(t.id);
        let node = match t.node {
            ast::TyNil | ast::TyBot | ast::TyInfer => t.node.clone(),
            ast::TyBox(ty) => ast::TyBox(self.fold_ty(ty)),
            ast::TyUniq(ty) => ast::TyUniq(self.fold_ty(ty)),
            ast::TyVec(ty) => ast::TyVec(self.fold_ty(ty)),
            ast::TyPtr(ref mt) => ast::TyPtr(fold_mt(mt, self)),
            ast::TyRptr(ref region, ref mt) => {
                ast::TyRptr(fold_opt_lifetime(region, self), fold_mt(mt, self))
            }
            ast::TyClosure(ref f, ref region) => {
                ast::TyClosure(@ast::ClosureTy {
                    fn_style: f.fn_style,
                    onceness: f.onceness,
                    bounds: fold_opt_bounds(&f.bounds, self),
                    decl: self.fold_fn_decl(f.decl),
                    lifetimes: f.lifetimes.iter().map(|l| self.fold_lifetime(l)).collect(),
                }, fold_opt_lifetime(region, self))
            }
            ast::TyProc(ref f) => {
                ast::TyProc(@ast::ClosureTy {
                    fn_style: f.fn_style,
                    onceness: f.onceness,
                    bounds: fold_opt_bounds(&f.bounds, self),
                    decl: self.fold_fn_decl(f.decl),
                    lifetimes: f.lifetimes.iter().map(|l| self.fold_lifetime(l)).collect(),
                })
            }
            ast::TyBareFn(ref f) => {
                ast::TyBareFn(@ast::BareFnTy {
                    lifetimes: f.lifetimes.iter().map(|l| self.fold_lifetime(l)).collect(),
                    fn_style: f.fn_style,
                    abi: f.abi,
                    decl: self.fold_fn_decl(f.decl)
                })
            }
            ast::TyTup(ref tys) => ast::TyTup(tys.iter().map(|&ty| self.fold_ty(ty)).collect()),
            ast::TyPath(ref path, ref bounds, id) => {
                let id = self.new_id(id);
                ast::TyPath(self.fold_path(path),
                            fold_opt_bounds(bounds, self),
                            id)
            }
            ast::TyFixedLengthVec(ty, e) => {
                ast::TyFixedLengthVec(self.fold_ty(ty), self.fold_expr(e))
            }
            ast::TyTypeof(expr) => ast::TyTypeof(self.fold_expr(expr)),
        };
        ast::P(ast::Ty {
            id: id,
            span: self.new_span(t.span),
            node: node,
        })
    }

    fn fold_mod(&mut self, m: &ast::Mod) -> ast::Mod {
        noop_fold_mod(m, self)
    }

    fn fold_foreign_mod(&mut self, nm: &ast::ForeignMod) -> ast::ForeignMod {
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

    fn fold_variant(&mut self, v: &ast::Variant) -> ast::P<ast::Variant> {
        let id = self.new_id(v.node.id);
        let kind;
        match v.node.kind {
            ast::TupleVariantKind(ref variant_args) => {
                kind = ast::TupleVariantKind(variant_args.iter().map(|x|
                    fold_variant_arg_(x, self)).collect())
            }
            ast::StructVariantKind(ref struct_def) => {
                kind = ast::StructVariantKind(@ast::StructDef {
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

        let attrs = v.node.attrs.iter().map(|x| fold_attribute_(*x, self)).collect();

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
        ast::P(Spanned {
            node: node,
            span: self.new_span(v.span),
        })
    }

    fn fold_ident(&mut self, i: ast::Ident) -> ast::Ident {
        i
    }

    fn fold_path(&mut self, p: &ast::Path) -> ast::Path {
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

    fn fold_local(&mut self, l: @ast::Local) -> @ast::Local {
        let id = self.new_id(l.id); // Needs to be first, for ast_map.
        @ast::Local {
            id: id,
            ty: self.fold_ty(l.ty),
            pat: self.fold_pat(l.pat),
            init: l.init.map(|e| self.fold_expr(e)),
            span: self.new_span(l.span),
            source: l.source,
        }
    }

    fn fold_mac(&mut self, macro: &ast::Mac) -> ast::Mac {
        Spanned {
            node: match macro.node {
                ast::MacInvocTT(ref p, ref tts, ctxt) => {
                    ast::MacInvocTT(self.fold_path(p),
                                    fold_tts(tts.as_slice(), self),
                                    ctxt)
                }
            },
            span: self.new_span(macro.span)
        }
    }

    fn map_exprs(&self, f: |@ast::Expr| -> @ast::Expr, es: &[@ast::Expr]) -> Vec<@ast::Expr> {
        es.iter().map(|x| f(*x)).collect()
    }

    fn new_id(&mut self, i: ast::NodeId) -> ast::NodeId {
        i
    }

    fn new_span(&mut self, sp: Span) -> Span {
        sp
    }

    fn fold_explicit_self(&mut self, es: &ast::ExplicitSelf) -> ast::ExplicitSelf {
        Spanned {
            span: self.new_span(es.span),
            node: self.fold_explicit_self_(&es.node)
        }
    }

    fn fold_explicit_self_(&mut self, es: &ast::ExplicitSelf_) -> ast::ExplicitSelf_ {
        match *es {
            ast::SelfStatic | ast::SelfValue | ast::SelfUniq => *es,
            ast::SelfRegion(ref lifetime, m) => {
                ast::SelfRegion(fold_opt_lifetime(lifetime, self), m)
            }
        }
    }

    fn fold_lifetime(&mut self, l: &ast::Lifetime) -> ast::Lifetime {
        noop_fold_lifetime(l, self)
    }
}

/* some little folds that probably aren't useful to have in Folder itself*/

//used in noop_fold_item and noop_fold_crate and noop_fold_crate_directive
fn fold_meta_item_<T: Folder>(mi: @ast::MetaItem, fld: &mut T) -> @ast::MetaItem {
    @Spanned {
        node:
            match mi.node {
                ast::MetaWord(ref id) => ast::MetaWord((*id).clone()),
                ast::MetaList(ref id, ref mis) => {
                    ast::MetaList((*id).clone(), mis.iter()
                        .map(|e| fold_meta_item_(*e, fld)).collect())
                }
                ast::MetaNameValue(ref id, ref s) => {
                    ast::MetaNameValue((*id).clone(), (*s).clone())
                }
            },
        span: fld.new_span(mi.span) }
}

//used in noop_fold_item and noop_fold_crate
fn fold_attribute_<T: Folder>(at: ast::Attribute, fld: &mut T) -> ast::Attribute {
    Spanned {
        span: fld.new_span(at.span),
        node: ast::Attribute_ {
            id: at.node.id,
            style: at.node.style,
            value: fold_meta_item_(at.node.value, fld),
            is_sugared_doc: at.node.is_sugared_doc
        }
    }
}

//used in noop_fold_foreign_item and noop_fold_fn_decl
fn fold_arg_<T: Folder>(a: &ast::Arg, fld: &mut T) -> ast::Arg {
    let id = fld.new_id(a.id); // Needs to be first, for ast_map.
    ast::Arg {
        id: id,
        ty: fld.fold_ty(a.ty),
        pat: fld.fold_pat(a.pat),
    }
}

// build a new vector of tts by appling the Folder's fold_ident to
// all of the identifiers in the token trees.
//
// This is part of hygiene magic. As far as hygiene is concerned, there
// are three types of let pattern bindings or loop labels:
//      - those defined and used in non-macro part of the program
//      - those used as part of macro invocation arguments
//      - those defined and used inside macro definitions
// Lexically, type 1 and 2 are in one group and type 3 the other. If they
// clash, in order for let and loop label to work hygienically, one group
// or the other needs to be renamed. The problem is that type 2 and 3 are
// parsed together (inside the macro expand function). After being parsed and
// AST being constructed, they can no longer be distinguished from each other.
//
// For that reason, type 2 let bindings and loop labels are actually renamed
// in the form of tokens instead of AST nodes, here. There are wasted effort
// since many token::IDENT are not necessary part of let bindings and most
// token::LIFETIME are certainly not loop labels. But we can't tell in their
// token form. So this is less ideal and hacky but it works.
pub fn fold_tts<T: Folder>(tts: &[ast::TokenTree], fld: &mut T) -> Vec<ast::TokenTree> {
    tts.iter().map(|tt| {
        match *tt {
            ast::TTTok(span, ref tok) =>
            ast::TTTok(span,maybe_fold_ident(tok,fld)),
            ast::TTDelim(ref tts) => ast::TTDelim(Rc::new(fold_tts(tts.as_slice(), fld))),
            ast::TTSeq(span, ref pattern, ref sep, is_optional) =>
            ast::TTSeq(span,
                  Rc::new(fold_tts(pattern.as_slice(), fld)),
                  sep.as_ref().map(|tok|maybe_fold_ident(tok,fld)),
                  is_optional),
            ast::TTNonterminal(sp,ref ident) =>
            ast::TTNonterminal(sp,fld.fold_ident(*ident))
        }
    }).collect()
}

// apply ident folder if it's an ident, otherwise leave it alone
fn maybe_fold_ident<T: Folder>(t: &token::Token, fld: &mut T) -> token::Token {
    match *t {
        token::IDENT(id, followed_by_colons) => {
            token::IDENT(fld.fold_ident(id), followed_by_colons)
        }
        token::LIFETIME(id) => token::LIFETIME(fld.fold_ident(id)),
        _ => (*t).clone()
    }
}

pub fn noop_fold_fn_decl<T: Folder>(decl: &ast::FnDecl, fld: &mut T) -> ast::P<ast::FnDecl> {
    ast::P(ast::FnDecl {
        inputs: decl.inputs.iter().map(|x| fold_arg_(x, fld)).collect(), // bad copy
        output: fld.fold_ty(decl.output),
        cf: decl.cf,
        variadic: decl.variadic
    })
}

fn fold_ty_param_bound<T: Folder>(tpb: &ast::TyParamBound, fld: &mut T)
                                    -> ast::TyParamBound {
    match *tpb {
        ast::TraitTyParamBound(ref ty) => ast::TraitTyParamBound(fold_trait_ref(ty, fld)),
        ast::StaticRegionTyParamBound => ast::StaticRegionTyParamBound,
        ast::OtherRegionTyParamBound(s) => ast::OtherRegionTyParamBound(s)
    }
}

pub fn fold_ty_param<T: Folder>(tp: &ast::TyParam, fld: &mut T) -> ast::TyParam {
    let id = fld.new_id(tp.id);
    ast::TyParam {
        ident: tp.ident,
        id: id,
        sized: tp.sized,
        bounds: tp.bounds.map(|x| fold_ty_param_bound(x, fld)),
        default: tp.default.map(|x| fld.fold_ty(x)),
        span: tp.span
    }
}

pub fn fold_ty_params<T: Folder>(tps: &OwnedSlice<ast::TyParam>, fld: &mut T)
                                   -> OwnedSlice<ast::TyParam> {
    tps.map(|tp| fold_ty_param(tp, fld))
}

pub fn noop_fold_lifetime<T: Folder>(l: &ast::Lifetime, fld: &mut T) -> ast::Lifetime {
    let id = fld.new_id(l.id);
    ast::Lifetime {
        id: id,
        span: fld.new_span(l.span),
        name: l.name
    }
}

pub fn fold_lifetimes<T: Folder>(lts: &Vec<ast::Lifetime>, fld: &mut T)
                                   -> Vec<ast::Lifetime> {
    lts.iter().map(|l| fld.fold_lifetime(l)).collect()
}

pub fn fold_opt_lifetime<T: Folder>(o_lt: &Option<ast::Lifetime>, fld: &mut T)
                                      -> Option<ast::Lifetime> {
    o_lt.as_ref().map(|lt| fld.fold_lifetime(lt))
}

pub fn fold_generics<T: Folder>(generics: &ast::Generics, fld: &mut T) -> ast::Generics {
    ast::Generics {ty_params: fold_ty_params(&generics.ty_params, fld),
                   lifetimes: fold_lifetimes(&generics.lifetimes, fld)}
}

fn fold_struct_def<T: Folder>(struct_def: @ast::StructDef, fld: &mut T) -> @ast::StructDef {
    @ast::StructDef {
        fields: struct_def.fields.iter().map(|f| fold_struct_field(f, fld)).collect(),
        ctor_id: struct_def.ctor_id.map(|cid| fld.new_id(cid)),
        super_struct: match struct_def.super_struct {
            Some(t) => Some(fld.fold_ty(t)),
            None => None
        },
        is_virtual: struct_def.is_virtual,
    }
}

fn fold_trait_ref<T: Folder>(p: &ast::TraitRef, fld: &mut T) -> ast::TraitRef {
    let id = fld.new_id(p.ref_id);
    ast::TraitRef {
        path: fld.fold_path(&p.path),
        ref_id: id,
    }
}

fn fold_struct_field<T: Folder>(f: &ast::StructField, fld: &mut T) -> ast::StructField {
    let id = fld.new_id(f.node.id);
    Spanned {
        node: ast::StructField_ {
            kind: f.node.kind,
            id: id,
            ty: fld.fold_ty(f.node.ty),
            attrs: f.node.attrs.iter().map(|a| fold_attribute_(*a, fld)).collect(),
        },
        span: fld.new_span(f.span),
    }
}

fn fold_field_<T: Folder>(field: ast::Field, folder: &mut T) -> ast::Field {
    ast::Field {
        ident: respan(field.ident.span, folder.fold_ident(field.ident.node)),
        expr: folder.fold_expr(field.expr),
        span: folder.new_span(field.span),
    }
}

fn fold_mt<T: Folder>(mt: &ast::MutTy, folder: &mut T) -> ast::MutTy {
    ast::MutTy {
        ty: folder.fold_ty(mt.ty),
        mutbl: mt.mutbl,
    }
}

fn fold_opt_bounds<T: Folder>(b: &Option<OwnedSlice<ast::TyParamBound>>, folder: &mut T)
                              -> Option<OwnedSlice<ast::TyParamBound>> {
    b.as_ref().map(|bounds| {
        bounds.map(|bound| {
            fold_ty_param_bound(bound, folder)
        })
    })
}

fn fold_variant_arg_<T: Folder>(va: &ast::VariantArg, folder: &mut T) -> ast::VariantArg {
    let id = folder.new_id(va.id);
    ast::VariantArg {
        ty: folder.fold_ty(va.ty),
        id: id,
    }
}

pub fn noop_fold_view_item<T: Folder>(vi: &ast::ViewItem, folder: &mut T)
                                       -> ast::ViewItem{
    let inner_view_item = match vi.node {
        ast::ViewItemExternCrate(ref ident, ref string, node_id) => {
            ast::ViewItemExternCrate(ident.clone(),
                                     (*string).clone(),
                                     folder.new_id(node_id))
        }
        ast::ViewItemUse(ref view_path) => {
            ast::ViewItemUse(folder.fold_view_path(*view_path))
        }
    };
    ast::ViewItem {
        node: inner_view_item,
        attrs: vi.attrs.iter().map(|a| fold_attribute_(*a, folder)).collect(),
        vis: vi.vis,
        span: folder.new_span(vi.span),
    }
}

pub fn noop_fold_block<T: Folder>(b: ast::P<ast::Block>, folder: &mut T) -> ast::P<ast::Block> {
    let id = folder.new_id(b.id); // Needs to be first, for ast_map.
    let view_items = b.view_items.iter().map(|x| folder.fold_view_item(x)).collect();
    let stmts = b.stmts.iter().flat_map(|s| folder.fold_stmt(*s).move_iter()).collect();
    ast::P(ast::Block {
        id: id,
        view_items: view_items,
        stmts: stmts,
        expr: b.expr.map(|x| folder.fold_expr(x)),
        rules: b.rules,
        span: folder.new_span(b.span),
    })
}

pub fn noop_fold_item_underscore<T: Folder>(i: &ast::Item_, folder: &mut T) -> ast::Item_ {
    match *i {
        ast::ItemStatic(t, m, e) => {
            ast::ItemStatic(folder.fold_ty(t), m, folder.fold_expr(e))
        }
        ast::ItemFn(decl, fn_style, abi, ref generics, body) => {
            ast::ItemFn(
                folder.fold_fn_decl(decl),
                fn_style,
                abi,
                fold_generics(generics, folder),
                folder.fold_block(body)
            )
        }
        ast::ItemMod(ref m) => ast::ItemMod(folder.fold_mod(m)),
        ast::ItemForeignMod(ref nm) => ast::ItemForeignMod(folder.fold_foreign_mod(nm)),
        ast::ItemTy(t, ref generics) => {
            ast::ItemTy(folder.fold_ty(t), fold_generics(generics, folder))
        }
        ast::ItemEnum(ref enum_definition, ref generics) => {
            ast::ItemEnum(
                ast::EnumDef {
                    variants: enum_definition.variants.iter().map(|&x| {
                        folder.fold_variant(x)
                    }).collect(),
                },
                fold_generics(generics, folder))
        }
        ast::ItemStruct(ref struct_def, ref generics) => {
            let struct_def = fold_struct_def(*struct_def, folder);
            ast::ItemStruct(struct_def, fold_generics(generics, folder))
        }
        ast::ItemImpl(ref generics, ref ifce, ty, ref methods) => {
            ast::ItemImpl(fold_generics(generics, folder),
                          ifce.as_ref().map(|p| fold_trait_ref(p, folder)),
                          folder.fold_ty(ty),
                          methods.iter().map(|x| folder.fold_method(*x)).collect()
            )
        }
        ast::ItemTrait(ref generics, ref sized, ref traits, ref methods) => {
            let methods = methods.iter().map(|method| {
                match *method {
                    ast::Required(ref m) => ast::Required(folder.fold_type_method(m)),
                    ast::Provided(method) => ast::Provided(folder.fold_method(method))
                }
            }).collect();
            ast::ItemTrait(fold_generics(generics, folder),
                      *sized,
                      traits.iter().map(|p| fold_trait_ref(p, folder)).collect(),
                      methods)
        }
        ast::ItemMac(ref m) => ast::ItemMac(folder.fold_mac(m)),
    }
}

pub fn noop_fold_type_method<T: Folder>(m: &ast::TypeMethod, fld: &mut T) -> ast::TypeMethod {
    let id = fld.new_id(m.id); // Needs to be first, for ast_map.
    ast::TypeMethod {
        id: id,
        ident: fld.fold_ident(m.ident),
        attrs: m.attrs.iter().map(|a| fold_attribute_(*a, fld)).collect(),
        fn_style: m.fn_style,
        decl: fld.fold_fn_decl(m.decl),
        generics: fold_generics(&m.generics, fld),
        explicit_self: fld.fold_explicit_self(&m.explicit_self),
        span: fld.new_span(m.span),
        vis: m.vis,
    }
}

pub fn noop_fold_mod<T: Folder>(m: &ast::Mod, folder: &mut T) -> ast::Mod {
    ast::Mod {
        inner: folder.new_span(m.inner),
        view_items: m.view_items
                     .iter()
                     .map(|x| folder.fold_view_item(x)).collect(),
        items: m.items.iter().flat_map(|x| folder.fold_item(*x).move_iter()).collect(),
    }
}

pub fn noop_fold_crate<T: Folder>(c: ast::Crate, folder: &mut T) -> ast::Crate {
    ast::Crate {
        module: folder.fold_mod(&c.module),
        attrs: c.attrs.iter().map(|x| fold_attribute_(*x, folder)).collect(),
        config: c.config.iter().map(|x| fold_meta_item_(*x, folder)).collect(),
        span: folder.new_span(c.span),
    }
}

pub fn noop_fold_item<T: Folder>(i: &ast::Item, folder: &mut T) -> SmallVector<@ast::Item> {
    let id = folder.new_id(i.id); // Needs to be first, for ast_map.
    let node = folder.fold_item_underscore(&i.node);
    let ident = match node {
        // The node may have changed, recompute the "pretty" impl name.
        ast::ItemImpl(_, ref maybe_trait, ty, _) => {
            ast_util::impl_pretty_name(maybe_trait, ty)
        }
        _ => i.ident
    };

    SmallVector::one(@ast::Item {
        id: id,
        ident: folder.fold_ident(ident),
        attrs: i.attrs.iter().map(|e| fold_attribute_(*e, folder)).collect(),
        node: node,
        vis: i.vis,
        span: folder.new_span(i.span)
    })
}

pub fn noop_fold_foreign_item<T: Folder>(ni: &ast::ForeignItem, folder: &mut T)
        -> @ast::ForeignItem {
    let id = folder.new_id(ni.id); // Needs to be first, for ast_map.
    @ast::ForeignItem {
        id: id,
        ident: folder.fold_ident(ni.ident),
        attrs: ni.attrs.iter().map(|x| fold_attribute_(*x, folder)).collect(),
        node: match ni.node {
            ast::ForeignItemFn(ref fdec, ref generics) => {
                ast::ForeignItemFn(ast::P(ast::FnDecl {
                    inputs: fdec.inputs.iter().map(|a| fold_arg_(a, folder)).collect(),
                    output: folder.fold_ty(fdec.output),
                    cf: fdec.cf,
                    variadic: fdec.variadic
                }), fold_generics(generics, folder))
            }
            ast::ForeignItemStatic(t, m) => {
                ast::ForeignItemStatic(folder.fold_ty(t), m)
            }
        },
        span: folder.new_span(ni.span),
        vis: ni.vis,
    }
}

pub fn noop_fold_method<T: Folder>(m: &ast::Method, folder: &mut T) -> @ast::Method {
    let id = folder.new_id(m.id); // Needs to be first, for ast_map.
    @ast::Method {
        id: id,
        ident: folder.fold_ident(m.ident),
        attrs: m.attrs.iter().map(|a| fold_attribute_(*a, folder)).collect(),
        generics: fold_generics(&m.generics, folder),
        explicit_self: folder.fold_explicit_self(&m.explicit_self),
        fn_style: m.fn_style,
        decl: folder.fold_fn_decl(m.decl),
        body: folder.fold_block(m.body),
        span: folder.new_span(m.span),
        vis: m.vis
    }
}

pub fn noop_fold_pat<T: Folder>(p: @ast::Pat, folder: &mut T) -> @ast::Pat {
    let id = folder.new_id(p.id);
    let node = match p.node {
        ast::PatWild => ast::PatWild,
        ast::PatWildMulti => ast::PatWildMulti,
        ast::PatIdent(binding_mode, ref pth, ref sub) => {
            ast::PatIdent(binding_mode,
                          folder.fold_path(pth),
                          sub.map(|x| folder.fold_pat(x)))
        }
        ast::PatLit(e) => ast::PatLit(folder.fold_expr(e)),
        ast::PatEnum(ref pth, ref pats) => {
            ast::PatEnum(folder.fold_path(pth),
                         pats.as_ref().map(|pats| pats.iter()
                             .map(|x| folder.fold_pat(*x)).collect()))
        }
        ast::PatStruct(ref pth, ref fields, etc) => {
            let pth_ = folder.fold_path(pth);
            let fs = fields.iter().map(|f| {
                ast::FieldPat {
                    ident: f.ident,
                    pat: folder.fold_pat(f.pat)
                }
            }).collect();
            ast::PatStruct(pth_, fs, etc)
        }
        ast::PatTup(ref elts) => ast::PatTup(elts.iter().map(|x| folder.fold_pat(*x)).collect()),
        ast::PatBox(inner) => ast::PatBox(folder.fold_pat(inner)),
        ast::PatRegion(inner) => ast::PatRegion(folder.fold_pat(inner)),
        ast::PatRange(e1, e2) => {
            ast::PatRange(folder.fold_expr(e1), folder.fold_expr(e2))
        },
        ast::PatVec(ref before, ref slice, ref after) => {
            ast::PatVec(before.iter().map(|x| folder.fold_pat(*x)).collect(),
                        slice.map(|x| folder.fold_pat(x)),
                        after.iter().map(|x| folder.fold_pat(*x)).collect())
        }
        ast::PatMac(ref mac) => ast::PatMac(folder.fold_mac(mac)),
    };

    @ast::Pat {
        id: id,
        span: folder.new_span(p.span),
        node: node,
    }
}

pub fn noop_fold_expr<T: Folder>(e: @ast::Expr, folder: &mut T) -> @ast::Expr {
    let id = folder.new_id(e.id);
    let node = match e.node {
        ast::ExprVstore(e, v) => {
            ast::ExprVstore(folder.fold_expr(e), v)
        }
        ast::ExprBox(p, e) => {
            ast::ExprBox(folder.fold_expr(p), folder.fold_expr(e))
        }
        ast::ExprVec(ref exprs) => {
            ast::ExprVec(exprs.iter().map(|&x| folder.fold_expr(x)).collect())
        }
        ast::ExprRepeat(expr, count) => {
            ast::ExprRepeat(folder.fold_expr(expr), folder.fold_expr(count))
        }
        ast::ExprTup(ref elts) => ast::ExprTup(elts.iter().map(|x| folder.fold_expr(*x)).collect()),
        ast::ExprCall(f, ref args) => {
            ast::ExprCall(folder.fold_expr(f),
                          args.iter().map(|&x| folder.fold_expr(x)).collect())
        }
        ast::ExprMethodCall(i, ref tps, ref args) => {
            ast::ExprMethodCall(
                respan(i.span, folder.fold_ident(i.node)),
                tps.iter().map(|&x| folder.fold_ty(x)).collect(),
                args.iter().map(|&x| folder.fold_expr(x)).collect())
        }
        ast::ExprBinary(binop, lhs, rhs) => {
            ast::ExprBinary(binop, folder.fold_expr(lhs), folder.fold_expr(rhs))
        }
        ast::ExprUnary(binop, ohs) => {
            ast::ExprUnary(binop, folder.fold_expr(ohs))
        }
        ast::ExprLit(_) => e.node.clone(),
        ast::ExprCast(expr, ty) => {
            ast::ExprCast(folder.fold_expr(expr), folder.fold_ty(ty))
        }
        ast::ExprAddrOf(m, ohs) => ast::ExprAddrOf(m, folder.fold_expr(ohs)),
        ast::ExprIf(cond, tr, fl) => {
            ast::ExprIf(folder.fold_expr(cond),
                   folder.fold_block(tr),
                   fl.map(|x| folder.fold_expr(x)))
        }
        ast::ExprWhile(cond, body) => {
            ast::ExprWhile(folder.fold_expr(cond), folder.fold_block(body))
        }
        ast::ExprForLoop(pat, iter, body, ref maybe_ident) => {
            ast::ExprForLoop(folder.fold_pat(pat),
                        folder.fold_expr(iter),
                        folder.fold_block(body),
                        maybe_ident.map(|i| folder.fold_ident(i)))
        }
        ast::ExprLoop(body, opt_ident) => {
            ast::ExprLoop(folder.fold_block(body),
                     opt_ident.map(|x| folder.fold_ident(x)))
        }
        ast::ExprMatch(expr, ref arms) => {
            ast::ExprMatch(folder.fold_expr(expr),
                      arms.iter().map(|x| folder.fold_arm(x)).collect())
        }
        ast::ExprFnBlock(decl, body) => {
            ast::ExprFnBlock(folder.fold_fn_decl(decl), folder.fold_block(body))
        }
        ast::ExprProc(decl, body) => {
            ast::ExprProc(folder.fold_fn_decl(decl), folder.fold_block(body))
        }
        ast::ExprBlock(blk) => ast::ExprBlock(folder.fold_block(blk)),
        ast::ExprAssign(el, er) => {
            ast::ExprAssign(folder.fold_expr(el), folder.fold_expr(er))
        }
        ast::ExprAssignOp(op, el, er) => {
            ast::ExprAssignOp(op,
                         folder.fold_expr(el),
                         folder.fold_expr(er))
        }
        ast::ExprField(el, id, ref tys) => {
            ast::ExprField(folder.fold_expr(el),
                           folder.fold_ident(id),
                           tys.iter().map(|&x| folder.fold_ty(x)).collect())
        }
        ast::ExprIndex(el, er) => {
            ast::ExprIndex(folder.fold_expr(el), folder.fold_expr(er))
        }
        ast::ExprPath(ref pth) => ast::ExprPath(folder.fold_path(pth)),
        ast::ExprBreak(opt_ident) => ast::ExprBreak(opt_ident.map(|x| folder.fold_ident(x))),
        ast::ExprAgain(opt_ident) => ast::ExprAgain(opt_ident.map(|x| folder.fold_ident(x))),
        ast::ExprRet(ref e) => {
            ast::ExprRet(e.map(|x| folder.fold_expr(x)))
        }
        ast::ExprInlineAsm(ref a) => {
            ast::ExprInlineAsm(ast::InlineAsm {
                inputs: a.inputs.iter().map(|&(ref c, input)| {
                    ((*c).clone(), folder.fold_expr(input))
                }).collect(),
                outputs: a.outputs.iter().map(|&(ref c, out)| {
                    ((*c).clone(), folder.fold_expr(out))
                }).collect(),
                .. (*a).clone()
            })
        }
        ast::ExprMac(ref mac) => ast::ExprMac(folder.fold_mac(mac)),
        ast::ExprStruct(ref path, ref fields, maybe_expr) => {
            ast::ExprStruct(folder.fold_path(path),
                       fields.iter().map(|x| fold_field_(*x, folder)).collect(),
                       maybe_expr.map(|x| folder.fold_expr(x)))
        },
        ast::ExprParen(ex) => ast::ExprParen(folder.fold_expr(ex))
    };

    @ast::Expr {
        id: id,
        node: node,
        span: folder.new_span(e.span),
    }
}

pub fn noop_fold_stmt<T: Folder>(s: &ast::Stmt, folder: &mut T) -> SmallVector<@ast::Stmt> {
    let nodes = match s.node {
        ast::StmtDecl(d, id) => {
            let id = folder.new_id(id);
            folder.fold_decl(d).move_iter()
                    .map(|d| ast::StmtDecl(d, id))
                    .collect()
        }
        ast::StmtExpr(e, id) => {
            let id = folder.new_id(id);
            SmallVector::one(ast::StmtExpr(folder.fold_expr(e), id))
        }
        ast::StmtSemi(e, id) => {
            let id = folder.new_id(id);
            SmallVector::one(ast::StmtSemi(folder.fold_expr(e), id))
        }
        ast::StmtMac(ref mac, semi) => SmallVector::one(ast::StmtMac(folder.fold_mac(mac), semi))
    };

    nodes.move_iter().map(|node| @Spanned {
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
