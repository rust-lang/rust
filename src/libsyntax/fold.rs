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
use codemap::{respan, Span, Spanned};
use parse::token;
use opt_vec::OptVec;
use util::small_vector::SmallVector;

// We may eventually want to be able to fold over type parameters, too.
pub trait ast_fold {
    fn fold_crate(&self, c: Crate) -> Crate {
        noop_fold_crate(c, self)
    }

    fn fold_meta_items(&self, meta_items: &[@MetaItem]) -> ~[@MetaItem] {
        meta_items.map(|x| fold_meta_item_(*x, self))
    }

    fn fold_view_paths(&self, view_paths: &[@view_path]) -> ~[@view_path] {
        view_paths.map(|view_path| {
            let inner_view_path = match view_path.node {
                view_path_simple(ref ident, ref path, node_id) => {
                    view_path_simple(ident.clone(),
                                     self.fold_path(path),
                                     self.new_id(node_id))
                }
                view_path_glob(ref path, node_id) => {
                    view_path_glob(self.fold_path(path), self.new_id(node_id))
                }
                view_path_list(ref path, ref path_list_idents, node_id) => {
                    view_path_list(self.fold_path(path),
                                   path_list_idents.map(|path_list_ident| {
                                    let id = self.new_id(path_list_ident.node
                                                                        .id);
                                    Spanned {
                                        node: path_list_ident_ {
                                            name: path_list_ident.node
                                                                 .name
                                                                 .clone(),
                                            id: id,
                                        },
                                        span: self.new_span(
                                            path_list_ident.span)
                                   }
                                  }),
                                  self.new_id(node_id))
                }
            };
            @Spanned {
                node: inner_view_path,
                span: self.new_span(view_path.span),
            }
        })
    }

    fn fold_view_item(&self, vi: &view_item) -> view_item {
        let inner_view_item = match vi.node {
            view_item_extern_mod(ref ident,
                                 string,
                                 ref meta_items,
                                 node_id) => {
                view_item_extern_mod(ident.clone(),
                                     string,
                                     self.fold_meta_items(*meta_items),
                                     self.new_id(node_id))
            }
            view_item_use(ref view_paths) => {
                view_item_use(self.fold_view_paths(*view_paths))
            }
        };
        view_item {
            node: inner_view_item,
            attrs: vi.attrs.map(|a| fold_attribute_(*a, self)),
            vis: vi.vis,
            span: self.new_span(vi.span),
        }
    }

    fn fold_foreign_item(&self, ni: @foreign_item) -> @foreign_item {
        let fold_attribute = |x| fold_attribute_(x, self);

        @ast::foreign_item {
            ident: self.fold_ident(ni.ident),
            attrs: ni.attrs.map(|x| fold_attribute(*x)),
            node:
                match ni.node {
                    foreign_item_fn(ref fdec, ref generics) => {
                        foreign_item_fn(
                            P(fn_decl {
                                inputs: fdec.inputs.map(|a| fold_arg_(a,
                                                                      self)),
                                output: self.fold_ty(fdec.output),
                                cf: fdec.cf,
                                variadic: fdec.variadic
                            }),
                            fold_generics(generics, self))
                    }
                    foreign_item_static(t, m) => {
                        foreign_item_static(self.fold_ty(t), m)
                    }
                },
            id: self.new_id(ni.id),
            span: self.new_span(ni.span),
            vis: ni.vis,
        }
    }

    fn fold_item(&self, i: @item) -> SmallVector<@item> {
        noop_fold_item(i, self)
    }

    fn fold_struct_field(&self, sf: @struct_field) -> @struct_field {
        let fold_attribute = |x| fold_attribute_(x, self);

        @Spanned {
            node: ast::struct_field_ {
                kind: sf.node.kind,
                id: self.new_id(sf.node.id),
                ty: self.fold_ty(sf.node.ty),
                attrs: sf.node.attrs.map(|e| fold_attribute(*e))
            },
            span: self.new_span(sf.span)
        }
    }

    fn fold_item_underscore(&self, i: &item_) -> item_ {
        noop_fold_item_underscore(i, self)
    }

    fn fold_type_method(&self, m: &TypeMethod) -> TypeMethod {
        noop_fold_type_method(m, self)
    }

    fn fold_method(&self, m: @method) -> @method {
        @ast::method {
            ident: self.fold_ident(m.ident),
            attrs: m.attrs.map(|a| fold_attribute_(*a, self)),
            generics: fold_generics(&m.generics, self),
            explicit_self: self.fold_explicit_self(&m.explicit_self),
            purity: m.purity,
            decl: fold_fn_decl(m.decl, self),
            body: self.fold_block(m.body),
            id: self.new_id(m.id),
            span: self.new_span(m.span),
            self_id: self.new_id(m.self_id),
            vis: m.vis,
        }
    }

    fn fold_block(&self, b: P<Block>) -> P<Block> {
        noop_fold_block(b, self)
    }

    fn fold_stmt(&self, s: &Stmt) -> SmallVector<@Stmt> {
        noop_fold_stmt(s, self)
    }

    fn fold_arm(&self, a: &Arm) -> Arm {
        Arm {
            pats: a.pats.map(|x| self.fold_pat(*x)),
            guard: a.guard.map(|x| self.fold_expr(x)),
            body: self.fold_block(a.body),
        }
    }

    fn fold_pat(&self, p: @Pat) -> @Pat {
        let node = match p.node {
            PatWild => PatWild,
            PatWildMulti => PatWildMulti,
            PatIdent(binding_mode, ref pth, ref sub) => {
                PatIdent(binding_mode,
                         self.fold_path(pth),
                         sub.map(|x| self.fold_pat(x)))
            }
            PatLit(e) => PatLit(self.fold_expr(e)),
            PatEnum(ref pth, ref pats) => {
                PatEnum(self.fold_path(pth),
                        pats.as_ref().map(|pats| pats.map(|x| self.fold_pat(*x))))
            }
            PatStruct(ref pth, ref fields, etc) => {
                let pth_ = self.fold_path(pth);
                let fs = fields.map(|f| {
                    ast::FieldPat {
                        ident: f.ident,
                        pat: self.fold_pat(f.pat)
                    }
                });
                PatStruct(pth_, fs, etc)
            }
            PatTup(ref elts) => PatTup(elts.map(|x| self.fold_pat(*x))),
            PatBox(inner) => PatBox(self.fold_pat(inner)),
            PatUniq(inner) => PatUniq(self.fold_pat(inner)),
            PatRegion(inner) => PatRegion(self.fold_pat(inner)),
            PatRange(e1, e2) => {
                PatRange(self.fold_expr(e1), self.fold_expr(e2))
            },
            PatVec(ref before, ref slice, ref after) => {
                PatVec(before.map(|x| self.fold_pat(*x)),
                       slice.map(|x| self.fold_pat(x)),
                       after.map(|x| self.fold_pat(*x)))
            }
        };

        @Pat {
            id: self.new_id(p.id),
            span: self.new_span(p.span),
            node: node,
        }
    }

    fn fold_decl(&self, d: @Decl) -> SmallVector<@Decl> {
        let node = match d.node {
            DeclLocal(ref l) => SmallVector::one(DeclLocal(self.fold_local(*l))),
            DeclItem(it) => {
                self.fold_item(it).move_iter().map(|i| DeclItem(i)).collect()
            }
        };

        node.move_iter().map(|node| {
            @Spanned {
                node: node,
                span: d.span,
            }
        }).collect()
    }

    fn fold_expr(&self, e: @Expr) -> @Expr {
        noop_fold_expr(e, self)
    }

    fn fold_ty(&self, t: P<Ty>) -> P<Ty> {
        let node = match t.node {
            ty_nil | ty_bot | ty_infer => t.node.clone(),
            ty_box(ref mt) => ty_box(fold_mt(mt, self)),
            ty_uniq(ref mt) => ty_uniq(fold_mt(mt, self)),
            ty_vec(ref mt) => ty_vec(fold_mt(mt, self)),
            ty_ptr(ref mt) => ty_ptr(fold_mt(mt, self)),
            ty_rptr(ref region, ref mt) => {
                ty_rptr(fold_opt_lifetime(region, self), fold_mt(mt, self))
            }
            ty_closure(ref f) => {
                ty_closure(@TyClosure {
                    sigil: f.sigil,
                    purity: f.purity,
                    region: fold_opt_lifetime(&f.region, self),
                    onceness: f.onceness,
                    bounds: fold_opt_bounds(&f.bounds, self),
                    decl: fold_fn_decl(f.decl, self),
                    lifetimes: f.lifetimes.map(|l| fold_lifetime(l, self)),
                })
            }
            ty_bare_fn(ref f) => {
                ty_bare_fn(@TyBareFn {
                    lifetimes: f.lifetimes.map(|l| fold_lifetime(l, self)),
                    purity: f.purity,
                    abis: f.abis,
                    decl: fold_fn_decl(f.decl, self)
                })
            }
            ty_tup(ref tys) => ty_tup(tys.map(|&ty| self.fold_ty(ty))),
            ty_path(ref path, ref bounds, id) => {
                ty_path(self.fold_path(path),
                        fold_opt_bounds(bounds, self),
                        self.new_id(id))
            }
            ty_fixed_length_vec(ref mt, e) => {
                ty_fixed_length_vec(fold_mt(mt, self), self.fold_expr(e))
            }
            ty_typeof(expr) => ty_typeof(self.fold_expr(expr)),
        };
        P(Ty {
            id: self.new_id(t.id),
            span: self.new_span(t.span),
            node: node,
        })
    }

    fn fold_mod(&self, m: &_mod) -> _mod {
        noop_fold_mod(m, self)
    }

    fn fold_foreign_mod(&self, nm: &foreign_mod) -> foreign_mod {
        ast::foreign_mod {
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

    fn fold_variant(&self, v: &variant) -> P<variant> {
        let kind;
        match v.node.kind {
            tuple_variant_kind(ref variant_args) => {
                kind = tuple_variant_kind(variant_args.map(|x|
                    fold_variant_arg_(x, self)))
            }
            struct_variant_kind(ref struct_def) => {
                kind = struct_variant_kind(@ast::struct_def {
                    fields: struct_def.fields.iter()
                        .map(|f| self.fold_struct_field(*f)).collect(),
                    ctor_id: struct_def.ctor_id.map(|c| self.new_id(c))
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
        P(Spanned {
            node: node,
            span: self.new_span(v.span),
        })
    }

    fn fold_ident(&self, i: Ident) -> Ident {
        i
    }

    fn fold_path(&self, p: &Path) -> Path {
        ast::Path {
            span: self.new_span(p.span),
            global: p.global,
            segments: p.segments.map(|segment| ast::PathSegment {
                identifier: self.fold_ident(segment.identifier),
                lifetimes: segment.lifetimes.map(|l| fold_lifetime(l, self)),
                types: segment.types.map(|&typ| self.fold_ty(typ)),
            })
        }
    }

    fn fold_local(&self, l: @Local) -> @Local {
        @Local {
            ty: self.fold_ty(l.ty),
            pat: self.fold_pat(l.pat),
            init: l.init.map(|e| self.fold_expr(e)),
            id: self.new_id(l.id),
            span: self.new_span(l.span),
        }
    }

    fn fold_mac(&self, macro: &mac) -> mac {
        Spanned {
            node: match macro.node {
                mac_invoc_tt(ref p, ref tts, ctxt) => {
                    mac_invoc_tt(self.fold_path(p),
                                 fold_tts(*tts, self),
                                 ctxt)
                }
            },
            span: self.new_span(macro.span)
        }
    }

    fn map_exprs(&self, f: |@Expr| -> @Expr, es: &[@Expr]) -> ~[@Expr] {
        es.map(|x| f(*x))
    }

    fn new_id(&self, i: NodeId) -> NodeId {
        i
    }

    fn new_span(&self, sp: Span) -> Span {
        sp
    }

    fn fold_explicit_self(&self, es: &explicit_self) -> explicit_self {
        Spanned {
            span: self.new_span(es.span),
            node: self.fold_explicit_self_(&es.node)
        }
    }

    fn fold_explicit_self_(&self, es: &explicit_self_) -> explicit_self_ {
        match *es {
            sty_static | sty_value(_) | sty_uniq(_) | sty_box(_) => {
                *es
            }
            sty_region(ref lifetime, m) => {
                sty_region(fold_opt_lifetime(lifetime, self), m)
            }
        }
    }
}

/* some little folds that probably aren't useful to have in ast_fold itself*/

//used in noop_fold_item and noop_fold_crate and noop_fold_crate_directive
fn fold_meta_item_<T:ast_fold>(mi: @MetaItem, fld: &T) -> @MetaItem {
    @Spanned {
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
    Spanned {
        span: fld.new_span(at.span),
        node: ast::Attribute_ {
            style: at.node.style,
            value: fold_meta_item_(at.node.value, fld),
            is_sugared_doc: at.node.is_sugared_doc
        }
    }
}

//used in noop_fold_foreign_item and noop_fold_fn_decl
fn fold_arg_<T:ast_fold>(a: &arg, fld: &T) -> arg {
    ast::arg {
        ty: fld.fold_ty(a.ty),
        pat: fld.fold_pat(a.pat),
        id: fld.new_id(a.id),
    }
}

// build a new vector of tts by appling the ast_fold's fold_ident to
// all of the identifiers in the token trees.
pub fn fold_tts<T:ast_fold>(tts: &[token_tree], fld: &T) -> ~[token_tree] {
    tts.map(|tt| {
        match *tt {
            tt_tok(span, ref tok) =>
            tt_tok(span,maybe_fold_ident(tok,fld)),
            tt_delim(tts) => tt_delim(@fold_tts(*tts, fld)),
            tt_seq(span, pattern, ref sep, is_optional) =>
            tt_seq(span,
                   @fold_tts(*pattern, fld),
                   sep.as_ref().map(|tok|maybe_fold_ident(tok,fld)),
                   is_optional),
            tt_nonterminal(sp,ref ident) =>
            tt_nonterminal(sp,fld.fold_ident(*ident))
        }
    })
}

// apply ident folder if it's an ident, otherwise leave it alone
fn maybe_fold_ident<T:ast_fold>(t: &token::Token, fld: &T) -> token::Token {
    match *t {
        token::IDENT(id, followed_by_colons) => {
            token::IDENT(fld.fold_ident(id), followed_by_colons)
        }
        _ => (*t).clone()
    }
}

pub fn fold_fn_decl<T:ast_fold>(decl: &ast::fn_decl, fld: &T)
                                -> P<fn_decl> {
    P(fn_decl {
        inputs: decl.inputs.map(|x| fold_arg_(x, fld)), // bad copy
        output: fld.fold_ty(decl.output),
        cf: decl.cf,
        variadic: decl.variadic
    })
}

fn fold_ty_param_bound<T:ast_fold>(tpb: &TyParamBound, fld: &T)
                                   -> TyParamBound {
    match *tpb {
        TraitTyParamBound(ref ty) => TraitTyParamBound(fold_trait_ref(ty, fld)),
        RegionTyParamBound => RegionTyParamBound
    }
}

pub fn fold_ty_param<T:ast_fold>(tp: &TyParam, fld: &T) -> TyParam {
    TyParam {
        ident: tp.ident,
        id: fld.new_id(tp.id),
        bounds: tp.bounds.map(|x| fold_ty_param_bound(x, fld)),
    }
}

pub fn fold_ty_params<T:ast_fold>(tps: &OptVec<TyParam>, fld: &T)
                                  -> OptVec<TyParam> {
    tps.map(|tp| fold_ty_param(tp, fld))
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

pub fn fold_opt_lifetime<T:ast_fold>(o_lt: &Option<Lifetime>, fld: &T)
                                     -> Option<Lifetime> {
    o_lt.as_ref().map(|lt| fold_lifetime(lt, fld))
}

pub fn fold_generics<T:ast_fold>(generics: &Generics, fld: &T) -> Generics {
    Generics {ty_params: fold_ty_params(&generics.ty_params, fld),
              lifetimes: fold_lifetimes(&generics.lifetimes, fld)}
}

fn fold_struct_def<T:ast_fold>(struct_def: @ast::struct_def, fld: &T)
                               -> @ast::struct_def {
    @ast::struct_def {
        fields: struct_def.fields.map(|f| fold_struct_field(*f, fld)),
        ctor_id: struct_def.ctor_id.map(|cid| fld.new_id(cid)),
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

fn fold_trait_ref<T:ast_fold>(p: &trait_ref, fld: &T) -> trait_ref {
    ast::trait_ref {
        path: fld.fold_path(&p.path),
        ref_id: fld.new_id(p.ref_id),
    }
}

fn fold_struct_field<T:ast_fold>(f: @struct_field, fld: &T) -> @struct_field {
    @Spanned {
        node: ast::struct_field_ {
            kind: f.node.kind,
            id: fld.new_id(f.node.id),
            ty: fld.fold_ty(f.node.ty),
            attrs: f.node.attrs.map(|a| fold_attribute_(*a, fld)),
        },
        span: fld.new_span(f.span),
    }
}

fn fold_field_<T:ast_fold>(field: Field, folder: &T) -> Field {
    ast::Field {
        ident: respan(field.ident.span, folder.fold_ident(field.ident.node)),
        expr: folder.fold_expr(field.expr),
        span: folder.new_span(field.span),
    }
}

fn fold_mt<T:ast_fold>(mt: &mt, folder: &T) -> mt {
    mt {
        ty: folder.fold_ty(mt.ty),
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
    b.as_ref().map(|bounds| {
        bounds.map(|bound| {
            fold_ty_param_bound(bound, folder)
        })
    })
}

fn fold_variant_arg_<T:ast_fold>(va: &variant_arg, folder: &T)
                                 -> variant_arg {
    ast::variant_arg {
        ty: folder.fold_ty(va.ty),
        id: folder.new_id(va.id)
    }
}

pub fn noop_fold_block<T:ast_fold>(b: P<Block>, folder: &T) -> P<Block> {
    let view_items = b.view_items.map(|x| folder.fold_view_item(x));
    let stmts = b.stmts.iter().flat_map(|s| folder.fold_stmt(*s).move_iter()).collect();
    P(Block {
        view_items: view_items,
        stmts: stmts,
        expr: b.expr.map(|x| folder.fold_expr(x)),
        id: folder.new_id(b.id),
        rules: b.rules,
        span: folder.new_span(b.span),
    })
}

pub fn noop_fold_item_underscore<T:ast_fold>(i: &item_, folder: &T) -> item_ {
    match *i {
        item_static(t, m, e) => {
            item_static(folder.fold_ty(t), m, folder.fold_expr(e))
        }
        item_fn(decl, purity, abi, ref generics, body) => {
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
        item_ty(t, ref generics) => {
            item_ty(folder.fold_ty(t),
                    fold_generics(generics, folder))
        }
        item_enum(ref enum_definition, ref generics) => {
            item_enum(
                ast::enum_def {
                    variants: enum_definition.variants.map(|&x| {
                        folder.fold_variant(x)
                    }),
                },
                fold_generics(generics, folder))
        }
        item_struct(ref struct_def, ref generics) => {
            let struct_def = fold_struct_def(*struct_def, folder);
            item_struct(struct_def, fold_generics(generics, folder))
        }
        item_impl(ref generics, ref ifce, ty, ref methods) => {
            item_impl(fold_generics(generics, folder),
                      ifce.as_ref().map(|p| fold_trait_ref(p, folder)),
                      folder.fold_ty(ty),
                      methods.map(|x| folder.fold_method(*x))
            )
        }
        item_trait(ref generics, ref traits, ref methods) => {
            let methods = methods.map(|method| {
                match *method {
                    required(ref m) => required(folder.fold_type_method(m)),
                    provided(method) => provided(folder.fold_method(method))
                }
            });
            item_trait(fold_generics(generics, folder),
                       traits.map(|p| fold_trait_ref(p, folder)),
                       methods)
        }
        item_mac(ref m) => item_mac(folder.fold_mac(m)),
    }
}

pub fn noop_fold_type_method<T:ast_fold>(m: &TypeMethod, fld: &T)
                                         -> TypeMethod {
    TypeMethod {
        ident: fld.fold_ident(m.ident),
        attrs: m.attrs.map(|a| fold_attribute_(*a, fld)),
        purity: m.purity,
        decl: fold_fn_decl(m.decl, fld),
        generics: fold_generics(&m.generics, fld),
        explicit_self: fld.fold_explicit_self(&m.explicit_self),
        id: fld.new_id(m.id),
        span: fld.new_span(m.span),
    }
}

pub fn noop_fold_mod<T:ast_fold>(m: &_mod, folder: &T) -> _mod {
    ast::_mod {
        view_items: m.view_items
                     .iter()
                     .map(|x| folder.fold_view_item(x)).collect(),
        items: m.items.iter().flat_map(|x| folder.fold_item(*x).move_iter()).collect(),
    }
}

pub fn noop_fold_crate<T:ast_fold>(c: Crate, folder: &T) -> Crate {
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
                                  -> SmallVector<@ast::item> {
    let fold_attribute = |x| fold_attribute_(x, folder);

    SmallVector::one(@ast::item {
        ident: folder.fold_ident(i.ident),
        attrs: i.attrs.map(|e| fold_attribute(*e)),
        id: folder.new_id(i.id),
        node: folder.fold_item_underscore(&i.node),
        vis: i.vis,
        span: folder.new_span(i.span)
    })
}

pub fn noop_fold_expr<T:ast_fold>(e: @ast::Expr, folder: &T) -> @ast::Expr {
    let fold_field = |x| fold_field_(x, folder);

    let node = match e.node {
        ExprVstore(e, v) => {
            ExprVstore(folder.fold_expr(e), v)
        }
        ExprVec(ref exprs, mutt) => {
            ExprVec(folder.map_exprs(|x| folder.fold_expr(x), *exprs), mutt)
        }
        ExprRepeat(expr, count, mutt) => {
            ExprRepeat(folder.fold_expr(expr), folder.fold_expr(count), mutt)
        }
        ExprTup(ref elts) => ExprTup(elts.map(|x| folder.fold_expr(*x))),
        ExprCall(f, ref args, blk) => {
            ExprCall(folder.fold_expr(f),
                     folder.map_exprs(|x| folder.fold_expr(x), *args),
                     blk)
        }
        ExprMethodCall(callee_id, f, i, ref tps, ref args, blk) => {
            ExprMethodCall(
                folder.new_id(callee_id),
                folder.fold_expr(f),
                folder.fold_ident(i),
                tps.map(|&x| folder.fold_ty(x)),
                folder.map_exprs(|x| folder.fold_expr(x), *args),
                blk
            )
        }
        ExprBinary(callee_id, binop, lhs, rhs) => {
            ExprBinary(folder.new_id(callee_id),
                       binop,
                       folder.fold_expr(lhs),
                       folder.fold_expr(rhs))
        }
        ExprUnary(callee_id, binop, ohs) => {
            ExprUnary(folder.new_id(callee_id), binop, folder.fold_expr(ohs))
        }
        ExprDoBody(f) => ExprDoBody(folder.fold_expr(f)),
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
                      arms.map(|x| folder.fold_arm(x)))
        }
        ExprFnBlock(decl, body) => {
            ExprFnBlock(
                fold_fn_decl(decl, folder),
                folder.fold_block(body)
            )
        }
        ExprProc(decl, body) => {
            ExprProc(fold_fn_decl(decl, folder), folder.fold_block(body))
        }
        ExprBlock(blk) => ExprBlock(folder.fold_block(blk)),
        ExprAssign(el, er) => {
            ExprAssign(folder.fold_expr(el), folder.fold_expr(er))
        }
        ExprAssignOp(callee_id, op, el, er) => {
            ExprAssignOp(folder.new_id(callee_id),
                         op,
                         folder.fold_expr(el),
                         folder.fold_expr(er))
        }
        ExprField(el, id, ref tys) => {
            ExprField(folder.fold_expr(el),
                      folder.fold_ident(id),
                      tys.map(|&x| folder.fold_ty(x)))
        }
        ExprIndex(callee_id, el, er) => {
            ExprIndex(folder.new_id(callee_id),
                      folder.fold_expr(el),
                      folder.fold_expr(er))
        }
        ExprPath(ref pth) => ExprPath(folder.fold_path(pth)),
        ExprSelf => ExprSelf,
        ExprLogLevel => ExprLogLevel,
        ExprBreak(opt_ident) => ExprBreak(opt_ident),
        ExprAgain(opt_ident) => ExprAgain(opt_ident),
        ExprRet(ref e) => {
            ExprRet(e.map(|x| folder.fold_expr(x)))
        }
        ExprInlineAsm(ref a) => {
            ExprInlineAsm(inline_asm {
                inputs: a.inputs.map(|&(c, input)| (c, folder.fold_expr(input))),
                outputs: a.outputs.map(|&(c, out)| (c, folder.fold_expr(out))),
                .. (*a).clone()
            })
        }
        ExprMac(ref mac) => ExprMac(folder.fold_mac(mac)),
        ExprStruct(ref path, ref fields, maybe_expr) => {
            ExprStruct(folder.fold_path(path),
                       fields.map(|x| fold_field(*x)),
                       maybe_expr.map(|x| folder.fold_expr(x)))
        },
        ExprParen(ex) => ExprParen(folder.fold_expr(ex))
    };

    @Expr {
        id: folder.new_id(e.id),
        node: node,
        span: folder.new_span(e.span),
    }
}

pub fn noop_fold_stmt<T:ast_fold>(s: &Stmt, folder: &T) -> SmallVector<@Stmt> {
    let nodes = match s.node {
        StmtDecl(d, nid) => {
            folder.fold_decl(d).move_iter()
                    .map(|d| StmtDecl(d, folder.new_id(nid)))
                    .collect()
        }
        StmtExpr(e, nid) => {
            SmallVector::one(StmtExpr(folder.fold_expr(e), folder.new_id(nid)))
        }
        StmtSemi(e, nid) => {
            SmallVector::one(StmtSemi(folder.fold_expr(e), folder.new_id(nid)))
        }
        StmtMac(ref mac, semi) => SmallVector::one(StmtMac(folder.fold_mac(mac), semi))
    };

    nodes.move_iter().map(|node| @Spanned {
        node: node,
        span: folder.new_span(s.span),
    }).collect()
}

#[cfg(test)]
mod test {
    use ast;
    use util::parser_testing::{string_to_crate, matches_codepattern};
    use parse::token;
    use print::pprust;
    use super::*;

    // this version doesn't care about getting comments or docstrings in.
    fn fake_print_crate(s: @pprust::ps, crate: &ast::Crate) {
        pprust::print_mod(s, &crate.module, crate.attrs);
    }

    // change every identifier to "zz"
    struct ToZzIdentFolder;

    impl ast_fold for ToZzIdentFolder {
        fn fold_ident(&self, _: ast::Ident) -> ast::Ident {
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
                if !(pred_val(a_val,b_val)) {
                    fail!("expected args satisfying {}, got {:?} and {:?}",
                          $predname, a_val, b_val);
                }
            }
        )
    )

    // make sure idents get transformed everywhere
    #[test] fn ident_transformation () {
        let zz_fold = ToZzIdentFolder;
        let ast = string_to_crate(@"#[a] mod b {fn c (d : e, f : g) {h!(i,j,k);l;m}}");
        assert_pred!(matches_codepattern,
                     "matches_codepattern",
                     pprust::to_str(&zz_fold.fold_crate(ast),fake_print_crate,
                                    token::get_ident_interner()),
                     ~"#[a]mod zz{fn zz(zz:zz,zz:zz){zz!(zz,zz,zz);zz;zz}}");
    }

    // even inside macro defs....
    #[test] fn ident_transformation_in_defs () {
        let zz_fold = ToZzIdentFolder;
        let ast = string_to_crate(@"macro_rules! a {(b $c:expr $(d $e:token)f+
=> (g $(d $d $e)+))} ");
        assert_pred!(matches_codepattern,
                     "matches_codepattern",
                     pprust::to_str(&zz_fold.fold_crate(ast),fake_print_crate,
                                    token::get_ident_interner()),
                     ~"zz!zz((zz$zz:zz$(zz $zz:zz)zz+=>(zz$(zz$zz$zz)+)))");
    }
}

