// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A Folder represents an HIR->HIR fold; it accepts a HIR piece,
//! and returns a piece of the same type.

use hir::*;
use syntax::ast::{Ident, Name, NodeId, DUMMY_NODE_ID, Attribute, Attribute_, MetaItem};
use syntax::ast::{MetaWord, MetaList, MetaNameValue};
use hir;
use syntax::codemap::{respan, Span, Spanned};
use syntax::owned_slice::OwnedSlice;
use syntax::ptr::P;
use syntax::parse::token;
use std::ptr;

// This could have a better place to live.
pub trait MoveMap<T> {
    fn move_map<F>(self, f: F) -> Self where F: FnMut(T) -> T;
}

impl<T> MoveMap<T> for Vec<T> {
    fn move_map<F>(mut self, mut f: F) -> Vec<T>
        where F: FnMut(T) -> T
    {
        for p in &mut self {
            unsafe {
                // FIXME(#5016) this shouldn't need to zero to be safe.
                ptr::write(p, f(ptr::read_and_drop(p)));
            }
        }
        self
    }
}

impl<T> MoveMap<T> for OwnedSlice<T> {
    fn move_map<F>(self, f: F) -> OwnedSlice<T>
        where F: FnMut(T) -> T
    {
        OwnedSlice::from_vec(self.into_vec().move_map(f))
    }
}

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

    fn fold_meta_items(&mut self, meta_items: Vec<P<MetaItem>>) -> Vec<P<MetaItem>> {
        noop_fold_meta_items(meta_items, self)
    }

    fn fold_meta_item(&mut self, meta_item: P<MetaItem>) -> P<MetaItem> {
        noop_fold_meta_item(meta_item, self)
    }

    fn fold_view_path(&mut self, view_path: P<ViewPath>) -> P<ViewPath> {
        noop_fold_view_path(view_path, self)
    }

    fn fold_foreign_item(&mut self, ni: P<ForeignItem>) -> P<ForeignItem> {
        noop_fold_foreign_item(ni, self)
    }

    fn fold_item(&mut self, i: P<Item>) -> P<Item> {
        noop_fold_item(i, self)
    }

    fn fold_struct_field(&mut self, sf: StructField) -> StructField {
        noop_fold_struct_field(sf, self)
    }

    fn fold_item_underscore(&mut self, i: Item_) -> Item_ {
        noop_fold_item_underscore(i, self)
    }

    fn fold_trait_item(&mut self, i: P<TraitItem>) -> P<TraitItem> {
        noop_fold_trait_item(i, self)
    }

    fn fold_impl_item(&mut self, i: P<ImplItem>) -> P<ImplItem> {
        noop_fold_impl_item(i, self)
    }

    fn fold_fn_decl(&mut self, d: P<FnDecl>) -> P<FnDecl> {
        noop_fold_fn_decl(d, self)
    }

    fn fold_block(&mut self, b: P<Block>) -> P<Block> {
        noop_fold_block(b, self)
    }

    fn fold_stmt(&mut self, s: P<Stmt>) -> P<Stmt> {
        noop_fold_stmt(s, self)
    }

    fn fold_arm(&mut self, a: Arm) -> Arm {
        noop_fold_arm(a, self)
    }

    fn fold_pat(&mut self, p: P<Pat>) -> P<Pat> {
        noop_fold_pat(p, self)
    }

    fn fold_decl(&mut self, d: P<Decl>) -> P<Decl> {
        noop_fold_decl(d, self)
    }

    fn fold_expr(&mut self, e: P<Expr>) -> P<Expr> {
        e.map(|e| noop_fold_expr(e, self))
    }

    fn fold_ty(&mut self, t: P<Ty>) -> P<Ty> {
        noop_fold_ty(t, self)
    }

    fn fold_ty_binding(&mut self, t: P<TypeBinding>) -> P<TypeBinding> {
        noop_fold_ty_binding(t, self)
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

    fn fold_name(&mut self, n: Name) -> Name {
        noop_fold_name(n, self)
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

    fn fold_path_parameters(&mut self, p: PathParameters) -> PathParameters {
        noop_fold_path_parameters(p, self)
    }

    fn fold_angle_bracketed_parameter_data(&mut self,
                                           p: AngleBracketedParameterData)
                                           -> AngleBracketedParameterData {
        noop_fold_angle_bracketed_parameter_data(p, self)
    }

    fn fold_parenthesized_parameter_data(&mut self,
                                         p: ParenthesizedParameterData)
                                         -> ParenthesizedParameterData {
        noop_fold_parenthesized_parameter_data(p, self)
    }

    fn fold_local(&mut self, l: P<Local>) -> P<Local> {
        noop_fold_local(l, self)
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

    fn fold_opt_lifetime(&mut self, o_lt: Option<Lifetime>) -> Option<Lifetime> {
        noop_fold_opt_lifetime(o_lt, self)
    }

    fn fold_opt_bounds(&mut self,
                       b: Option<OwnedSlice<TyParamBound>>)
                       -> Option<OwnedSlice<TyParamBound>> {
        noop_fold_opt_bounds(b, self)
    }

    fn fold_bounds(&mut self, b: OwnedSlice<TyParamBound>) -> OwnedSlice<TyParamBound> {
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

    fn fold_where_clause(&mut self, where_clause: WhereClause) -> WhereClause {
        noop_fold_where_clause(where_clause, self)
    }

    fn fold_where_predicate(&mut self, where_predicate: WherePredicate) -> WherePredicate {
        noop_fold_where_predicate(where_predicate, self)
    }

    fn new_id(&mut self, i: NodeId) -> NodeId {
        i
    }

    fn new_span(&mut self, sp: Span) -> Span {
        sp
    }
}

pub fn noop_fold_meta_items<T: Folder>(meta_items: Vec<P<MetaItem>>,
                                       fld: &mut T)
                                       -> Vec<P<MetaItem>> {
    meta_items.move_map(|x| fld.fold_meta_item(x))
}

pub fn noop_fold_view_path<T: Folder>(view_path: P<ViewPath>, fld: &mut T) -> P<ViewPath> {
    view_path.map(|Spanned { node, span }| {
        Spanned {
            node: match node {
                ViewPathSimple(name, path) => {
                    ViewPathSimple(name, fld.fold_path(path))
                }
                ViewPathGlob(path) => {
                    ViewPathGlob(fld.fold_path(path))
                }
                ViewPathList(path, path_list_idents) => {
                    ViewPathList(fld.fold_path(path),
                                 path_list_idents.move_map(|path_list_ident| {
                                     Spanned {
                                         node: match path_list_ident.node {
                                             PathListIdent { id, name, rename } => PathListIdent {
                                                 id: fld.new_id(id),
                                                 name: name,
                                                 rename: rename,
                                             },
                                             PathListMod { id, rename } => PathListMod {
                                                 id: fld.new_id(id),
                                                 rename: rename,
                                             },
                                         },
                                         span: fld.new_span(path_list_ident.span),
                                     }
                                 }))
                }
            },
            span: fld.new_span(span),
        }
    })
}

pub fn fold_attrs<T: Folder>(attrs: Vec<Attribute>, fld: &mut T) -> Vec<Attribute> {
    attrs.into_iter().flat_map(|x| fld.fold_attribute(x)).collect()
}

pub fn noop_fold_arm<T: Folder>(Arm { attrs, pats, guard, body }: Arm, fld: &mut T) -> Arm {
    Arm {
        attrs: fold_attrs(attrs, fld),
        pats: pats.move_map(|x| fld.fold_pat(x)),
        guard: guard.map(|x| fld.fold_expr(x)),
        body: fld.fold_expr(body),
    }
}

pub fn noop_fold_decl<T: Folder>(d: P<Decl>, fld: &mut T) -> P<Decl> {
    d.map(|Spanned { node, span }| {
        match node {
            DeclLocal(l) => Spanned {
                node: DeclLocal(fld.fold_local(l)),
                span: fld.new_span(span),
            },
            DeclItem(it) => Spanned {
                node: DeclItem(fld.fold_item(it)),
                span: fld.new_span(span),
            },
        }
    })
}

pub fn noop_fold_ty_binding<T: Folder>(b: P<TypeBinding>, fld: &mut T) -> P<TypeBinding> {
    b.map(|TypeBinding { id, name, ty, span }| {
        TypeBinding {
            id: fld.new_id(id),
            name: name,
            ty: fld.fold_ty(ty),
            span: fld.new_span(span),
        }
    })
}

pub fn noop_fold_ty<T: Folder>(t: P<Ty>, fld: &mut T) -> P<Ty> {
    t.map(|Ty { id, node, span }| {
        Ty {
            id: fld.new_id(id),
            node: match node {
                TyInfer => node,
                TyVec(ty) => TyVec(fld.fold_ty(ty)),
                TyPtr(mt) => TyPtr(fld.fold_mt(mt)),
                TyRptr(region, mt) => {
                    TyRptr(fld.fold_opt_lifetime(region), fld.fold_mt(mt))
                }
                TyBareFn(f) => {
                    TyBareFn(f.map(|BareFnTy { lifetimes, unsafety, abi, decl }| {
                        BareFnTy {
                            lifetimes: fld.fold_lifetime_defs(lifetimes),
                            unsafety: unsafety,
                            abi: abi,
                            decl: fld.fold_fn_decl(decl),
                        }
                    }))
                }
                TyTup(tys) => TyTup(tys.move_map(|ty| fld.fold_ty(ty))),
                TyPath(qself, path) => {
                    let qself = qself.map(|QSelf { ty, position }| {
                        QSelf {
                            ty: fld.fold_ty(ty),
                            position: position,
                        }
                    });
                    TyPath(qself, fld.fold_path(path))
                }
                TyObjectSum(ty, bounds) => {
                    TyObjectSum(fld.fold_ty(ty), fld.fold_bounds(bounds))
                }
                TyFixedLengthVec(ty, e) => {
                    TyFixedLengthVec(fld.fold_ty(ty), fld.fold_expr(e))
                }
                TyTypeof(expr) => {
                    TyTypeof(fld.fold_expr(expr))
                }
                TyPolyTraitRef(bounds) => {
                    TyPolyTraitRef(bounds.move_map(|b| fld.fold_ty_param_bound(b)))
                }
            },
            span: fld.new_span(span),
        }
    })
}

pub fn noop_fold_foreign_mod<T: Folder>(ForeignMod { abi, items }: ForeignMod,
                                        fld: &mut T)
                                        -> ForeignMod {
    ForeignMod {
        abi: abi,
        items: items.move_map(|x| fld.fold_foreign_item(x)),
    }
}

pub fn noop_fold_variant<T: Folder>(v: P<Variant>, fld: &mut T) -> P<Variant> {
    v.map(|Spanned { node: Variant_ { name, attrs, data, disr_expr }, span }| {
        Spanned {
            node: Variant_ {
                name: name,
                attrs: fold_attrs(attrs, fld),
                data: fld.fold_variant_data(data),
                disr_expr: disr_expr.map(|e| fld.fold_expr(e)),
            },
            span: fld.new_span(span),
        }
    })
}

pub fn noop_fold_name<T: Folder>(n: Name, _: &mut T) -> Name {
    n
}

pub fn noop_fold_ident<T: Folder>(i: Ident, _: &mut T) -> Ident {
    i
}

pub fn noop_fold_usize<T: Folder>(i: usize, _: &mut T) -> usize {
    i
}

pub fn noop_fold_path<T: Folder>(Path { global, segments, span }: Path, fld: &mut T) -> Path {
    Path {
        global: global,
        segments: segments.move_map(|PathSegment { identifier, parameters }| {
            PathSegment {
                identifier: fld.fold_ident(identifier),
                parameters: fld.fold_path_parameters(parameters),
            }
        }),
        span: fld.new_span(span),
    }
}

pub fn noop_fold_path_parameters<T: Folder>(path_parameters: PathParameters,
                                            fld: &mut T)
                                            -> PathParameters {
    match path_parameters {
        AngleBracketedParameters(data) =>
            AngleBracketedParameters(fld.fold_angle_bracketed_parameter_data(data)),
        ParenthesizedParameters(data) =>
            ParenthesizedParameters(fld.fold_parenthesized_parameter_data(data)),
    }
}

pub fn noop_fold_angle_bracketed_parameter_data<T: Folder>(data: AngleBracketedParameterData,
                                                           fld: &mut T)
                                                           -> AngleBracketedParameterData {
    let AngleBracketedParameterData { lifetimes, types, bindings } = data;
    AngleBracketedParameterData {
        lifetimes: fld.fold_lifetimes(lifetimes),
        types: types.move_map(|ty| fld.fold_ty(ty)),
        bindings: bindings.move_map(|b| fld.fold_ty_binding(b)),
    }
}

pub fn noop_fold_parenthesized_parameter_data<T: Folder>(data: ParenthesizedParameterData,
                                                         fld: &mut T)
                                                         -> ParenthesizedParameterData {
    let ParenthesizedParameterData { inputs, output, span } = data;
    ParenthesizedParameterData {
        inputs: inputs.move_map(|ty| fld.fold_ty(ty)),
        output: output.map(|ty| fld.fold_ty(ty)),
        span: fld.new_span(span),
    }
}

pub fn noop_fold_local<T: Folder>(l: P<Local>, fld: &mut T) -> P<Local> {
    l.map(|Local { id, pat, ty, init, span }| {
        Local {
            id: fld.new_id(id),
            ty: ty.map(|t| fld.fold_ty(t)),
            pat: fld.fold_pat(pat),
            init: init.map(|e| fld.fold_expr(e)),
            span: fld.new_span(span),
        }
    })
}

pub fn noop_fold_attribute<T: Folder>(at: Attribute, fld: &mut T) -> Option<Attribute> {
    let Spanned {node: Attribute_ {id, style, value, is_sugared_doc}, span} = at;
    Some(Spanned {
        node: Attribute_ {
            id: id,
            style: style,
            value: fld.fold_meta_item(value),
            is_sugared_doc: is_sugared_doc,
        },
        span: fld.new_span(span),
    })
}

pub fn noop_fold_explicit_self_underscore<T: Folder>(es: ExplicitSelf_,
                                                     fld: &mut T)
                                                     -> ExplicitSelf_ {
    match es {
        SelfStatic | SelfValue(_) => es,
        SelfRegion(lifetime, m, name) => {
            SelfRegion(fld.fold_opt_lifetime(lifetime), m, name)
        }
        SelfExplicit(typ, name) => {
            SelfExplicit(fld.fold_ty(typ), name)
        }
    }
}

pub fn noop_fold_explicit_self<T: Folder>(Spanned { span, node }: ExplicitSelf,
                                          fld: &mut T)
                                          -> ExplicitSelf {
    Spanned {
        node: fld.fold_explicit_self_underscore(node),
        span: fld.new_span(span),
    }
}

pub fn noop_fold_meta_item<T: Folder>(mi: P<MetaItem>, fld: &mut T) -> P<MetaItem> {
    mi.map(|Spanned { node, span }| {
        Spanned {
            node: match node {
                MetaWord(id) => MetaWord(id),
                MetaList(id, mis) => {
                    MetaList(id, mis.move_map(|e| fld.fold_meta_item(e)))
                }
                MetaNameValue(id, s) => MetaNameValue(id, s),
            },
            span: fld.new_span(span),
        }
    })
}

pub fn noop_fold_arg<T: Folder>(Arg { id, pat, ty }: Arg, fld: &mut T) -> Arg {
    Arg {
        id: fld.new_id(id),
        pat: fld.fold_pat(pat),
        ty: fld.fold_ty(ty),
    }
}

pub fn noop_fold_fn_decl<T: Folder>(decl: P<FnDecl>, fld: &mut T) -> P<FnDecl> {
    decl.map(|FnDecl { inputs, output, variadic }| {
        FnDecl {
            inputs: inputs.move_map(|x| fld.fold_arg(x)),
            output: match output {
                Return(ty) => Return(fld.fold_ty(ty)),
                DefaultReturn(span) => DefaultReturn(span),
                NoReturn(span) => NoReturn(span),
            },
            variadic: variadic,
        }
    })
}

pub fn noop_fold_ty_param_bound<T>(tpb: TyParamBound, fld: &mut T) -> TyParamBound
    where T: Folder
{
    match tpb {
        TraitTyParamBound(ty, modifier) => TraitTyParamBound(fld.fold_poly_trait_ref(ty), modifier),
        RegionTyParamBound(lifetime) => RegionTyParamBound(fld.fold_lifetime(lifetime)),
    }
}

pub fn noop_fold_ty_param<T: Folder>(tp: TyParam, fld: &mut T) -> TyParam {
    let TyParam {id, name, bounds, default, span} = tp;
    TyParam {
        id: fld.new_id(id),
        name: name,
        bounds: fld.fold_bounds(bounds),
        default: default.map(|x| fld.fold_ty(x)),
        span: span,
    }
}

pub fn noop_fold_ty_params<T: Folder>(tps: OwnedSlice<TyParam>,
                                      fld: &mut T)
                                      -> OwnedSlice<TyParam> {
    tps.move_map(|tp| fld.fold_ty_param(tp))
}

pub fn noop_fold_lifetime<T: Folder>(l: Lifetime, fld: &mut T) -> Lifetime {
    Lifetime {
        id: fld.new_id(l.id),
        name: l.name,
        span: fld.new_span(l.span),
    }
}

pub fn noop_fold_lifetime_def<T: Folder>(l: LifetimeDef, fld: &mut T) -> LifetimeDef {
    LifetimeDef {
        lifetime: fld.fold_lifetime(l.lifetime),
        bounds: fld.fold_lifetimes(l.bounds),
    }
}

pub fn noop_fold_lifetimes<T: Folder>(lts: Vec<Lifetime>, fld: &mut T) -> Vec<Lifetime> {
    lts.move_map(|l| fld.fold_lifetime(l))
}

pub fn noop_fold_lifetime_defs<T: Folder>(lts: Vec<LifetimeDef>, fld: &mut T) -> Vec<LifetimeDef> {
    lts.move_map(|l| fld.fold_lifetime_def(l))
}

pub fn noop_fold_opt_lifetime<T: Folder>(o_lt: Option<Lifetime>, fld: &mut T) -> Option<Lifetime> {
    o_lt.map(|lt| fld.fold_lifetime(lt))
}

pub fn noop_fold_generics<T: Folder>(Generics { ty_params, lifetimes, where_clause }: Generics,
                                     fld: &mut T)
                                     -> Generics {
    Generics {
        ty_params: fld.fold_ty_params(ty_params),
        lifetimes: fld.fold_lifetime_defs(lifetimes),
        where_clause: fld.fold_where_clause(where_clause),
    }
}

pub fn noop_fold_where_clause<T: Folder>(WhereClause { id, predicates }: WhereClause,
                                         fld: &mut T)
                                         -> WhereClause {
    WhereClause {
        id: fld.new_id(id),
        predicates: predicates.move_map(|predicate| fld.fold_where_predicate(predicate)),
    }
}

pub fn noop_fold_where_predicate<T: Folder>(pred: WherePredicate, fld: &mut T) -> WherePredicate {
    match pred {
        hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate{bound_lifetimes,
                                                                     bounded_ty,
                                                                     bounds,
                                                                     span}) => {
            hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                bound_lifetimes: fld.fold_lifetime_defs(bound_lifetimes),
                bounded_ty: fld.fold_ty(bounded_ty),
                bounds: bounds.move_map(|x| fld.fold_ty_param_bound(x)),
                span: fld.new_span(span),
            })
        }
        hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate{lifetime,
                                                                       bounds,
                                                                       span}) => {
            hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                span: fld.new_span(span),
                lifetime: fld.fold_lifetime(lifetime),
                bounds: bounds.move_map(|bound| fld.fold_lifetime(bound)),
            })
        }
        hir::WherePredicate::EqPredicate(hir::WhereEqPredicate{id,
                                                               path,
                                                               ty,
                                                               span}) => {
            hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                id: fld.new_id(id),
                path: fld.fold_path(path),
                ty: fld.fold_ty(ty),
                span: fld.new_span(span),
            })
        }
    }
}

pub fn noop_fold_variant_data<T: Folder>(vdata: VariantData, fld: &mut T) -> VariantData {
    match vdata {
        VariantData::Struct(fields, id) => {
            VariantData::Struct(fields.move_map(|f| fld.fold_struct_field(f)),
                                fld.new_id(id))
        }
        VariantData::Tuple(fields, id) => {
            VariantData::Tuple(fields.move_map(|f| fld.fold_struct_field(f)),
                               fld.new_id(id))
        }
        VariantData::Unit(id) => VariantData::Unit(fld.new_id(id)),
    }
}

pub fn noop_fold_trait_ref<T: Folder>(p: TraitRef, fld: &mut T) -> TraitRef {
    let id = fld.new_id(p.ref_id);
    let TraitRef {
        path,
        ref_id: _,
    } = p;
    hir::TraitRef {
        path: fld.fold_path(path),
        ref_id: id,
    }
}

pub fn noop_fold_poly_trait_ref<T: Folder>(p: PolyTraitRef, fld: &mut T) -> PolyTraitRef {
    hir::PolyTraitRef {
        bound_lifetimes: fld.fold_lifetime_defs(p.bound_lifetimes),
        trait_ref: fld.fold_trait_ref(p.trait_ref),
        span: fld.new_span(p.span),
    }
}

pub fn noop_fold_struct_field<T: Folder>(f: StructField, fld: &mut T) -> StructField {
    let StructField {node: StructField_ {id, kind, ty, attrs}, span} = f;
    Spanned {
        node: StructField_ {
            id: fld.new_id(id),
            kind: kind,
            ty: fld.fold_ty(ty),
            attrs: fold_attrs(attrs, fld),
        },
        span: fld.new_span(span),
    }
}

pub fn noop_fold_field<T: Folder>(Field { name, expr, span }: Field, folder: &mut T) -> Field {
    Field {
        name: respan(folder.new_span(name.span), folder.fold_name(name.node)),
        expr: folder.fold_expr(expr),
        span: folder.new_span(span),
    }
}

pub fn noop_fold_mt<T: Folder>(MutTy { ty, mutbl }: MutTy, folder: &mut T) -> MutTy {
    MutTy {
        ty: folder.fold_ty(ty),
        mutbl: mutbl,
    }
}

pub fn noop_fold_opt_bounds<T: Folder>(b: Option<OwnedSlice<TyParamBound>>,
                                       folder: &mut T)
                                       -> Option<OwnedSlice<TyParamBound>> {
    b.map(|bounds| folder.fold_bounds(bounds))
}

fn noop_fold_bounds<T: Folder>(bounds: TyParamBounds, folder: &mut T) -> TyParamBounds {
    bounds.move_map(|bound| folder.fold_ty_param_bound(bound))
}

pub fn noop_fold_block<T: Folder>(b: P<Block>, folder: &mut T) -> P<Block> {
    b.map(|Block { id, stmts, expr, rules, span }| {
        Block {
            id: folder.new_id(id),
            stmts: stmts.into_iter().map(|s| folder.fold_stmt(s)).collect(),
            expr: expr.map(|x| folder.fold_expr(x)),
            rules: rules,
            span: folder.new_span(span),
        }
    })
}

pub fn noop_fold_item_underscore<T: Folder>(i: Item_, folder: &mut T) -> Item_ {
    match i {
        ItemExternCrate(string) => ItemExternCrate(string),
        ItemUse(view_path) => {
            ItemUse(folder.fold_view_path(view_path))
        }
        ItemStatic(t, m, e) => {
            ItemStatic(folder.fold_ty(t), m, folder.fold_expr(e))
        }
        ItemConst(t, e) => {
            ItemConst(folder.fold_ty(t), folder.fold_expr(e))
        }
        ItemFn(decl, unsafety, constness, abi, generics, body) => {
            ItemFn(folder.fold_fn_decl(decl),
                   unsafety,
                   constness,
                   abi,
                   folder.fold_generics(generics),
                   folder.fold_block(body))
        }
        ItemMod(m) => ItemMod(folder.fold_mod(m)),
        ItemForeignMod(nm) => ItemForeignMod(folder.fold_foreign_mod(nm)),
        ItemTy(t, generics) => {
            ItemTy(folder.fold_ty(t), folder.fold_generics(generics))
        }
        ItemEnum(enum_definition, generics) => {
            ItemEnum(hir::EnumDef {
                         variants: enum_definition.variants.move_map(|x| folder.fold_variant(x)),
                     },
                     folder.fold_generics(generics))
        }
        ItemStruct(struct_def, generics) => {
            let struct_def = folder.fold_variant_data(struct_def);
            ItemStruct(struct_def, folder.fold_generics(generics))
        }
        ItemDefaultImpl(unsafety, ref trait_ref) => {
            ItemDefaultImpl(unsafety, folder.fold_trait_ref((*trait_ref).clone()))
        }
        ItemImpl(unsafety, polarity, generics, ifce, ty, impl_items) => {
            let new_impl_items = impl_items.into_iter()
                                           .map(|item| folder.fold_impl_item(item))
                                           .collect();
            let ifce = match ifce {
                None => None,
                Some(ref trait_ref) => {
                    Some(folder.fold_trait_ref((*trait_ref).clone()))
                }
            };
            ItemImpl(unsafety,
                     polarity,
                     folder.fold_generics(generics),
                     ifce,
                     folder.fold_ty(ty),
                     new_impl_items)
        }
        ItemTrait(unsafety, generics, bounds, items) => {
            let bounds = folder.fold_bounds(bounds);
            let items = items.into_iter()
                             .map(|item| folder.fold_trait_item(item))
                             .collect();
            ItemTrait(unsafety, folder.fold_generics(generics), bounds, items)
        }
    }
}

pub fn noop_fold_trait_item<T: Folder>(i: P<TraitItem>,
                                       folder: &mut T)
                                       -> P<TraitItem> {
    i.map(|TraitItem { id, name, attrs, node, span }| {
        TraitItem {
            id: folder.new_id(id),
            name: folder.fold_name(name),
            attrs: fold_attrs(attrs, folder),
            node: match node {
                ConstTraitItem(ty, default) => {
                    ConstTraitItem(folder.fold_ty(ty), default.map(|x| folder.fold_expr(x)))
                }
                MethodTraitItem(sig, body) => {
                    MethodTraitItem(noop_fold_method_sig(sig, folder),
                                    body.map(|x| folder.fold_block(x)))
                }
                TypeTraitItem(bounds, default) => {
                    TypeTraitItem(folder.fold_bounds(bounds),
                                  default.map(|x| folder.fold_ty(x)))
                }
            },
            span: folder.new_span(span),
        }
    })
}

pub fn noop_fold_impl_item<T: Folder>(i: P<ImplItem>, folder: &mut T) -> P<ImplItem> {
    i.map(|ImplItem { id, name, attrs, node, vis, span }| {
        ImplItem {
            id: folder.new_id(id),
            name: folder.fold_name(name),
            attrs: fold_attrs(attrs, folder),
            vis: vis,
            node: match node {
                ConstImplItem(ty, expr) => {
                    ConstImplItem(folder.fold_ty(ty), folder.fold_expr(expr))
                }
                MethodImplItem(sig, body) => {
                    MethodImplItem(noop_fold_method_sig(sig, folder), folder.fold_block(body))
                }
                TypeImplItem(ty) => TypeImplItem(folder.fold_ty(ty)),
            },
            span: folder.new_span(span),
        }
    })
}

pub fn noop_fold_mod<T: Folder>(Mod { inner, items }: Mod, folder: &mut T) -> Mod {
    Mod {
        inner: folder.new_span(inner),
        items: items.into_iter().map(|x| folder.fold_item(x)).collect(),
    }
}

pub fn noop_fold_crate<T: Folder>(Crate { module, attrs, config, span, exported_macros }: Crate,
                                  folder: &mut T)
                                  -> Crate {
    let config = folder.fold_meta_items(config);

    let crate_mod = folder.fold_item(P(hir::Item {
        name: token::special_idents::invalid.name,
        attrs: attrs,
        id: DUMMY_NODE_ID,
        vis: hir::Public,
        span: span,
        node: hir::ItemMod(module),
    }));

    let (module, attrs, span) =
        crate_mod.and_then(|hir::Item { attrs, span, node, .. }| {
            match node {
                hir::ItemMod(m) => (m, attrs, span),
                _ => panic!("fold converted a module to not a module"),
            }
        });

    Crate {
        module: module,
        attrs: attrs,
        config: config,
        span: span,
        exported_macros: exported_macros,
    }
}

pub fn noop_fold_item<T: Folder>(item: P<Item>, folder: &mut T) -> P<Item> {
    item.map(|Item { id, name, attrs, node, vis, span }| {
        let id = folder.new_id(id);
        let node = folder.fold_item_underscore(node);
        // FIXME: we should update the impl_pretty_name, but it uses pretty printing.
        // let ident = match node {
        //     // The node may have changed, recompute the "pretty" impl name.
        //     ItemImpl(_, _, _, ref maybe_trait, ref ty, _) => {
        //         impl_pretty_name(maybe_trait, Some(&**ty))
        //     }
        //     _ => ident
        // };

        Item {
            id: id,
            name: folder.fold_name(name),
            attrs: fold_attrs(attrs, folder),
            node: node,
            vis: vis,
            span: folder.new_span(span),
        }
    })
}

pub fn noop_fold_foreign_item<T: Folder>(ni: P<ForeignItem>, folder: &mut T) -> P<ForeignItem> {
    ni.map(|ForeignItem { id, name, attrs, node, span, vis }| {
        ForeignItem {
            id: folder.new_id(id),
            name: folder.fold_name(name),
            attrs: fold_attrs(attrs, folder),
            node: match node {
                ForeignItemFn(fdec, generics) => {
                    ForeignItemFn(folder.fold_fn_decl(fdec), folder.fold_generics(generics))
                }
                ForeignItemStatic(t, m) => {
                    ForeignItemStatic(folder.fold_ty(t), m)
                }
            },
            vis: vis,
            span: folder.new_span(span),
        }
    })
}

pub fn noop_fold_method_sig<T: Folder>(sig: MethodSig, folder: &mut T) -> MethodSig {
    MethodSig {
        generics: folder.fold_generics(sig.generics),
        abi: sig.abi,
        explicit_self: folder.fold_explicit_self(sig.explicit_self),
        unsafety: sig.unsafety,
        constness: sig.constness,
        decl: folder.fold_fn_decl(sig.decl),
    }
}

pub fn noop_fold_pat<T: Folder>(p: P<Pat>, folder: &mut T) -> P<Pat> {
    p.map(|Pat { id, node, span }| {
        Pat {
            id: folder.new_id(id),
            node: match node {
                PatWild => PatWild,
                PatIdent(binding_mode, pth1, sub) => {
                    PatIdent(binding_mode,
                             Spanned {
                                 span: folder.new_span(pth1.span),
                                 node: folder.fold_ident(pth1.node),
                             },
                             sub.map(|x| folder.fold_pat(x)))
                }
                PatLit(e) => PatLit(folder.fold_expr(e)),
                PatEnum(pth, pats) => {
                    PatEnum(folder.fold_path(pth),
                            pats.map(|pats| pats.move_map(|x| folder.fold_pat(x))))
                }
                PatQPath(qself, pth) => {
                    let qself = QSelf { ty: folder.fold_ty(qself.ty), ..qself };
                    PatQPath(qself, folder.fold_path(pth))
                }
                PatStruct(pth, fields, etc) => {
                    let pth = folder.fold_path(pth);
                    let fs = fields.move_map(|f| {
                        Spanned {
                            span: folder.new_span(f.span),
                            node: hir::FieldPat {
                                name: f.node.name,
                                pat: folder.fold_pat(f.node.pat),
                                is_shorthand: f.node.is_shorthand,
                            },
                        }
                    });
                    PatStruct(pth, fs, etc)
                }
                PatTup(elts) => PatTup(elts.move_map(|x| folder.fold_pat(x))),
                PatBox(inner) => PatBox(folder.fold_pat(inner)),
                PatRegion(inner, mutbl) => PatRegion(folder.fold_pat(inner), mutbl),
                PatRange(e1, e2) => {
                    PatRange(folder.fold_expr(e1), folder.fold_expr(e2))
                }
                PatVec(before, slice, after) => {
                    PatVec(before.move_map(|x| folder.fold_pat(x)),
                           slice.map(|x| folder.fold_pat(x)),
                           after.move_map(|x| folder.fold_pat(x)))
                }
            },
            span: folder.new_span(span),
        }
    })
}

pub fn noop_fold_expr<T: Folder>(Expr { id, node, span }: Expr, folder: &mut T) -> Expr {
    Expr {
        id: folder.new_id(id),
        node: match node {
            ExprBox(e) => {
                ExprBox(folder.fold_expr(e))
            }
            ExprVec(exprs) => {
                ExprVec(exprs.move_map(|x| folder.fold_expr(x)))
            }
            ExprRepeat(expr, count) => {
                ExprRepeat(folder.fold_expr(expr), folder.fold_expr(count))
            }
            ExprTup(elts) => ExprTup(elts.move_map(|x| folder.fold_expr(x))),
            ExprCall(f, args) => {
                ExprCall(folder.fold_expr(f), args.move_map(|x| folder.fold_expr(x)))
            }
            ExprMethodCall(name, tps, args) => {
                ExprMethodCall(respan(folder.new_span(name.span), folder.fold_name(name.node)),
                               tps.move_map(|x| folder.fold_ty(x)),
                               args.move_map(|x| folder.fold_expr(x)))
            }
            ExprBinary(binop, lhs, rhs) => {
                ExprBinary(binop, folder.fold_expr(lhs), folder.fold_expr(rhs))
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
            ExprLoop(body, opt_ident) => {
                ExprLoop(folder.fold_block(body),
                         opt_ident.map(|i| folder.fold_ident(i)))
            }
            ExprMatch(expr, arms, source) => {
                ExprMatch(folder.fold_expr(expr),
                          arms.move_map(|x| folder.fold_arm(x)),
                          source)
            }
            ExprClosure(capture_clause, decl, body) => {
                ExprClosure(capture_clause,
                            folder.fold_fn_decl(decl),
                            folder.fold_block(body))
            }
            ExprBlock(blk) => ExprBlock(folder.fold_block(blk)),
            ExprAssign(el, er) => {
                ExprAssign(folder.fold_expr(el), folder.fold_expr(er))
            }
            ExprAssignOp(op, el, er) => {
                ExprAssignOp(op, folder.fold_expr(el), folder.fold_expr(er))
            }
            ExprField(el, name) => {
                ExprField(folder.fold_expr(el),
                          respan(folder.new_span(name.span), folder.fold_name(name.node)))
            }
            ExprTupField(el, index) => {
                ExprTupField(folder.fold_expr(el),
                             respan(folder.new_span(index.span), folder.fold_usize(index.node)))
            }
            ExprIndex(el, er) => {
                ExprIndex(folder.fold_expr(el), folder.fold_expr(er))
            }
            ExprRange(e1, e2) => {
                ExprRange(e1.map(|x| folder.fold_expr(x)),
                          e2.map(|x| folder.fold_expr(x)))
            }
            ExprPath(qself, path) => {
                let qself = qself.map(|QSelf { ty, position }| {
                    QSelf {
                        ty: folder.fold_ty(ty),
                        position: position,
                    }
                });
                ExprPath(qself, folder.fold_path(path))
            }
            ExprBreak(opt_ident) => ExprBreak(opt_ident.map(|label| {
                respan(folder.new_span(label.span), folder.fold_ident(label.node))
            })),
            ExprAgain(opt_ident) => ExprAgain(opt_ident.map(|label| {
                respan(folder.new_span(label.span), folder.fold_ident(label.node))
            })),
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
                inputs: inputs.move_map(|(c, input)| (c, folder.fold_expr(input))),
                outputs: outputs.move_map(|(c, out, is_rw)| (c, folder.fold_expr(out), is_rw)),
                asm: asm,
                asm_str_style: asm_str_style,
                clobbers: clobbers,
                volatile: volatile,
                alignstack: alignstack,
                dialect: dialect,
                expn_id: expn_id,
            }),
            ExprStruct(path, fields, maybe_expr) => {
                ExprStruct(folder.fold_path(path),
                           fields.move_map(|x| folder.fold_field(x)),
                           maybe_expr.map(|x| folder.fold_expr(x)))
            }
        },
        span: folder.new_span(span),
    }
}

pub fn noop_fold_stmt<T: Folder>(stmt: P<Stmt>, folder: &mut T)
                                 -> P<Stmt> {
    stmt.map(|Spanned { node, span }| {
        let span = folder.new_span(span);
        match node {
            StmtDecl(d, id) => {
                let id = folder.new_id(id);
                Spanned {
                    node: StmtDecl(folder.fold_decl(d), id),
                    span: span
                }
            }
            StmtExpr(e, id) => {
                let id = folder.new_id(id);
                Spanned {
                    node: StmtExpr(folder.fold_expr(e), id),
                    span: span,
                }
            }
            StmtSemi(e, id) => {
                let id = folder.new_id(id);
                Spanned {
                    node: StmtSemi(folder.fold_expr(e), id),
                    span: span,
                }
            }
        }
    })
}
