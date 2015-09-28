// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Lowers the AST to the HIR

use hir;

use syntax::ast::*;
use syntax::ptr::P;
use syntax::codemap::{respan, Spanned, Span};
use syntax::owned_slice::OwnedSlice;
use syntax::parse::token::{self, str_to_ident};
use syntax::std_inject;

pub struct LoweringContext<'a, 'hir> {
    // TODO
    foo: &'hir i32,
    id_assigner: &'a NodeIdAssigner,
    crate_root: Option<&'static str>,
}

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    pub fn new(foo: &'hir i32, id_assigner: &'a NodeIdAssigner, c: &Crate) -> LoweringContext<'a, 'hir> {
        let crate_root = if std_inject::no_core(c) {
            None
        } else if std_inject::no_std(c) {
            Some("core")
        } else {
            Some("std")
        };

        LoweringContext {
            foo: foo,
            id_assigner: id_assigner,
            crate_root: crate_root,
        }
    }

    fn next_id(&self) -> NodeId {
        self.id_assigner.next_node_id()
    }
}

pub fn lower_view_path(_lctx: &LoweringContext, view_path: &ViewPath) -> P<hir::ViewPath> {
    P(Spanned {
        node: match view_path.node {
            ViewPathSimple(ident, ref path) => {
                hir::ViewPathSimple(ident.name, lower_path(_lctx, path))
            }
            ViewPathGlob(ref path) => {
                hir::ViewPathGlob(lower_path(_lctx, path))
            }
            ViewPathList(ref path, ref path_list_idents) => {
                hir::ViewPathList(lower_path(_lctx, path),
                             path_list_idents.iter().map(|path_list_ident| {
                                Spanned {
                                    node: match path_list_ident.node {
                                        PathListIdent { id, name, rename } =>
                                            hir::PathListIdent {
                                                id: id,
                                                name: name.name,
                                                rename: rename.map(|x| x.name),
                                            },
                                        PathListMod { id, rename } =>
                                            hir::PathListMod {
                                                id: id,
                                                rename: rename.map(|x| x.name)
                                            }
                                    },
                                    span: path_list_ident.span
                                }
                             }).collect())
            }
        },
        span: view_path.span,
    })
}

pub fn lower_arm(_lctx: &LoweringContext, arm: &Arm) -> hir::Arm {
    hir::Arm {
        attrs: arm.attrs.clone(),
        pats: arm.pats.iter().map(|x| lower_pat(_lctx, x)).collect(),
        guard: arm.guard.as_ref().map(|ref x| lower_expr(_lctx, x)),
        body: lower_expr(_lctx, &arm.body),
    }
}

pub fn lower_decl(_lctx: &LoweringContext, d: &Decl) -> P<hir::Decl> {
    match d.node {
        DeclLocal(ref l) => P(Spanned {
            node: hir::DeclLocal(lower_local(_lctx, l)),
            span: d.span
        }),
        DeclItem(ref it) => P(Spanned {
            node: hir::DeclItem(lower_item(_lctx, it)),
            span: d.span
        }),
    }
}

pub fn lower_ty_binding(_lctx: &LoweringContext, b: &TypeBinding) -> P<hir::TypeBinding> {
    P(hir::TypeBinding { id: b.id, name: b.ident.name, ty: lower_ty(_lctx, &b.ty), span: b.span })
}

pub fn lower_ty(_lctx: &LoweringContext, t: &Ty) -> P<hir::Ty> {
    P(hir::Ty {
        id: t.id,
        node: match t.node {
            TyInfer => hir::TyInfer,
            TyVec(ref ty) => hir::TyVec(lower_ty(_lctx, ty)),
            TyPtr(ref mt) => hir::TyPtr(lower_mt(_lctx, mt)),
            TyRptr(ref region, ref mt) => {
                hir::TyRptr(lower_opt_lifetime(_lctx, region), lower_mt(_lctx, mt))
            }
            TyBareFn(ref f) => {
                hir::TyBareFn(P(hir::BareFnTy {
                    lifetimes: lower_lifetime_defs(_lctx, &f.lifetimes),
                    unsafety: lower_unsafety(_lctx, f.unsafety),
                    abi: f.abi,
                    decl: lower_fn_decl(_lctx, &f.decl),
                }))
            }
            TyTup(ref tys) => hir::TyTup(tys.iter().map(|ty| lower_ty(_lctx, ty)).collect()),
            TyParen(ref ty) => hir::TyParen(lower_ty(_lctx, ty)),
            TyPath(ref qself, ref path) => {
                let qself = qself.as_ref().map(|&QSelf { ref ty, position }| {
                    hir::QSelf {
                        ty: lower_ty(_lctx, ty),
                        position: position,
                    }
                });
                hir::TyPath(qself, lower_path(_lctx, path))
            }
            TyObjectSum(ref ty, ref bounds) => {
                hir::TyObjectSum(lower_ty(_lctx, ty),
                            lower_bounds(_lctx, bounds))
            }
            TyFixedLengthVec(ref ty, ref e) => {
                hir::TyFixedLengthVec(lower_ty(_lctx, ty), lower_expr(_lctx, e))
            }
            TyTypeof(ref expr) => {
                hir::TyTypeof(lower_expr(_lctx, expr))
            }
            TyPolyTraitRef(ref bounds) => {
                hir::TyPolyTraitRef(bounds.iter().map(|b| lower_ty_param_bound(_lctx, b)).collect())
            }
            TyMac(_) => panic!("TyMac should have been expanded by now."),
        },
        span: t.span,
    })
}

pub fn lower_foreign_mod(_lctx: &LoweringContext, fm: &ForeignMod) -> hir::ForeignMod {
    hir::ForeignMod {
        abi: fm.abi,
        items: fm.items.iter().map(|x| lower_foreign_item(_lctx, x)).collect(),
    }
}

pub fn lower_variant(_lctx: &LoweringContext, v: &Variant) -> P<hir::Variant> {
    P(Spanned {
        node: hir::Variant_ {
            id: v.node.id,
            name: v.node.name.name,
            attrs: v.node.attrs.clone(),
            kind: match v.node.kind {
                TupleVariantKind(ref variant_args) => {
                    hir::TupleVariantKind(variant_args.iter().map(|ref x|
                        lower_variant_arg(_lctx, x)).collect())
                }
                StructVariantKind(ref struct_def) => {
                    hir::StructVariantKind(lower_struct_def(_lctx, struct_def))
                }
            },
            disr_expr: v.node.disr_expr.as_ref().map(|e| lower_expr(_lctx, e)),
        },
        span: v.span,
    })
}

pub fn lower_path(_lctx: &LoweringContext, p: &Path) -> hir::Path {
    hir::Path {
        global: p.global,
        segments: p.segments.iter().map(|&PathSegment {identifier, ref parameters}|
            hir::PathSegment {
                identifier: identifier,
                parameters: lower_path_parameters(_lctx, parameters),
            }).collect(),
        span: p.span,
    }
}

pub fn lower_path_parameters(_lctx: &LoweringContext,
                             path_parameters: &PathParameters)
                             -> hir::PathParameters {
    match *path_parameters {
        AngleBracketedParameters(ref data) =>
            hir::AngleBracketedParameters(lower_angle_bracketed_parameter_data(_lctx, data)),
        ParenthesizedParameters(ref data) =>
            hir::ParenthesizedParameters(lower_parenthesized_parameter_data(_lctx, data)),
    }
}

pub fn lower_angle_bracketed_parameter_data(_lctx: &LoweringContext,
                                            data: &AngleBracketedParameterData)
                                            -> hir::AngleBracketedParameterData {
    let &AngleBracketedParameterData { ref lifetimes, ref types, ref bindings } = data;
    hir::AngleBracketedParameterData {
        lifetimes: lower_lifetimes(_lctx, lifetimes),
        types: types.iter().map(|ty| lower_ty(_lctx, ty)).collect(),
        bindings: bindings.iter().map(|b| lower_ty_binding(_lctx, b)).collect(),
    }
}

pub fn lower_parenthesized_parameter_data(_lctx: &LoweringContext,
                                          data: &ParenthesizedParameterData)
                                          -> hir::ParenthesizedParameterData {
    let &ParenthesizedParameterData { ref inputs, ref output, span } = data;
    hir::ParenthesizedParameterData {
        inputs: inputs.iter().map(|ty| lower_ty(_lctx, ty)).collect(),
        output: output.as_ref().map(|ty| lower_ty(_lctx, ty)),
        span: span,
    }
}

pub fn lower_local(_lctx: &LoweringContext, l: &Local) -> P<hir::Local> {
    P(hir::Local {
            id: l.id,
            ty: l.ty.as_ref().map(|t| lower_ty(_lctx, t)),
            pat: lower_pat(_lctx, &l.pat),
            init: l.init.as_ref().map(|e| lower_expr(_lctx, e)),
            span: l.span,
        })
}

pub fn lower_explicit_self_underscore(_lctx: &LoweringContext,
                                      es: &ExplicitSelf_)
                                      -> hir::ExplicitSelf_ {
    match *es {
        SelfStatic => hir::SelfStatic,
        SelfValue(v) => hir::SelfValue(v.name),
        SelfRegion(ref lifetime, m, ident) => {
            hir::SelfRegion(lower_opt_lifetime(_lctx, lifetime),
                            lower_mutability(_lctx, m),
                            ident.name)
        }
        SelfExplicit(ref typ, ident) => {
            hir::SelfExplicit(lower_ty(_lctx, typ), ident.name)
        }
    }
}

pub fn lower_mutability(_lctx: &LoweringContext, m: Mutability) -> hir::Mutability {
    match m {
        MutMutable => hir::MutMutable,
        MutImmutable => hir::MutImmutable,
    }
}

pub fn lower_explicit_self(_lctx: &LoweringContext, s: &ExplicitSelf) -> hir::ExplicitSelf {
    Spanned { node: lower_explicit_self_underscore(_lctx, &s.node), span: s.span }
}

pub fn lower_arg(_lctx: &LoweringContext, arg: &Arg) -> hir::Arg {
    hir::Arg { id: arg.id, pat: lower_pat(_lctx, &arg.pat), ty: lower_ty(_lctx, &arg.ty) }
}

pub fn lower_fn_decl(_lctx: &LoweringContext, decl: &FnDecl) -> P<hir::FnDecl> {
    P(hir::FnDecl {
        inputs: decl.inputs.iter().map(|x| lower_arg(_lctx, x)).collect(),
        output: match decl.output {
            Return(ref ty) => hir::Return(lower_ty(_lctx, ty)),
            DefaultReturn(span) => hir::DefaultReturn(span),
            NoReturn(span) => hir::NoReturn(span),
        },
        variadic: decl.variadic,
    })
}

pub fn lower_ty_param_bound(_lctx: &LoweringContext, tpb: &TyParamBound) -> hir::TyParamBound {
    match *tpb {
        TraitTyParamBound(ref ty, modifier) => {
            hir::TraitTyParamBound(lower_poly_trait_ref(_lctx, ty),
                                   lower_trait_bound_modifier(_lctx, modifier))
        }
        RegionTyParamBound(ref lifetime) => {
            hir::RegionTyParamBound(lower_lifetime(_lctx, lifetime))
        }
    }
}

pub fn lower_ty_param(_lctx: &LoweringContext, tp: &TyParam) -> hir::TyParam {
    hir::TyParam {
        id: tp.id,
        name: tp.ident.name,
        bounds: lower_bounds(_lctx, &tp.bounds),
        default: tp.default.as_ref().map(|x| lower_ty(_lctx, x)),
        span: tp.span,
    }
}

pub fn lower_ty_params(_lctx: &LoweringContext,
                       tps: &OwnedSlice<TyParam>)
                       -> OwnedSlice<hir::TyParam> {
    tps.iter().map(|tp| lower_ty_param(_lctx, tp)).collect()
}

pub fn lower_lifetime(_lctx: &LoweringContext, l: &Lifetime) -> hir::Lifetime {
    hir::Lifetime { id: l.id, name: l.name, span: l.span }
}

pub fn lower_lifetime_def(_lctx: &LoweringContext, l: &LifetimeDef) -> hir::LifetimeDef {
    hir::LifetimeDef {
        lifetime: lower_lifetime(_lctx, &l.lifetime),
        bounds: lower_lifetimes(_lctx, &l.bounds)
    }
}

pub fn lower_lifetimes(_lctx: &LoweringContext, lts: &Vec<Lifetime>) -> Vec<hir::Lifetime> {
    lts.iter().map(|l| lower_lifetime(_lctx, l)).collect()
}

pub fn lower_lifetime_defs(_lctx: &LoweringContext,
                           lts: &Vec<LifetimeDef>)
                           -> Vec<hir::LifetimeDef> {
    lts.iter().map(|l| lower_lifetime_def(_lctx, l)).collect()
}

pub fn lower_opt_lifetime(_lctx: &LoweringContext,
                          o_lt: &Option<Lifetime>)
                          -> Option<hir::Lifetime> {
    o_lt.as_ref().map(|lt| lower_lifetime(_lctx, lt))
}

pub fn lower_generics(_lctx: &LoweringContext, g: &Generics) -> hir::Generics {
    hir::Generics {
        ty_params: lower_ty_params(_lctx, &g.ty_params),
        lifetimes: lower_lifetime_defs(_lctx, &g.lifetimes),
        where_clause: lower_where_clause(_lctx, &g.where_clause),
    }
}

pub fn lower_where_clause(_lctx: &LoweringContext, wc: &WhereClause) -> hir::WhereClause {
    hir::WhereClause {
        id: wc.id,
        predicates: wc.predicates.iter().map(|predicate|
            lower_where_predicate(_lctx, predicate)).collect(),
    }
}

pub fn lower_where_predicate(_lctx: &LoweringContext,
                             pred: &WherePredicate)
                             -> hir::WherePredicate {
    match *pred {
        WherePredicate::BoundPredicate(WhereBoundPredicate{ ref bound_lifetimes,
                                                            ref bounded_ty,
                                                            ref bounds,
                                                            span}) => {
            hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                bound_lifetimes: lower_lifetime_defs(_lctx, bound_lifetimes),
                bounded_ty: lower_ty(_lctx, bounded_ty),
                bounds: bounds.iter().map(|x| lower_ty_param_bound(_lctx, x)).collect(),
                span: span
            })
        }
        WherePredicate::RegionPredicate(WhereRegionPredicate{ ref lifetime,
                                                              ref bounds,
                                                              span}) => {
            hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                span: span,
                lifetime: lower_lifetime(_lctx, lifetime),
                bounds: bounds.iter().map(|bound| lower_lifetime(_lctx, bound)).collect()
            })
        }
        WherePredicate::EqPredicate(WhereEqPredicate{ id,
                                                      ref path,
                                                      ref ty,
                                                      span}) => {
            hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                id: id,
                path: lower_path(_lctx, path),
                ty:lower_ty(_lctx, ty),
                span: span
            })
        }
    }
}

pub fn lower_struct_def(_lctx: &LoweringContext, sd: &StructDef) -> P<hir::StructDef> {
    P(hir::StructDef {
        fields: sd.fields.iter().map(|f| lower_struct_field(_lctx, f)).collect(),
        ctor_id: sd.ctor_id,
    })
}

pub fn lower_trait_ref(_lctx: &LoweringContext, p: &TraitRef) -> hir::TraitRef {
    hir::TraitRef { path: lower_path(_lctx, &p.path), ref_id: p.ref_id }
}

pub fn lower_poly_trait_ref(_lctx: &LoweringContext, p: &PolyTraitRef) -> hir::PolyTraitRef {
    hir::PolyTraitRef {
        bound_lifetimes: lower_lifetime_defs(_lctx, &p.bound_lifetimes),
        trait_ref: lower_trait_ref(_lctx, &p.trait_ref),
        span: p.span,
    }
}

pub fn lower_struct_field(_lctx: &LoweringContext, f: &StructField) -> hir::StructField {
    Spanned {
        node: hir::StructField_ {
            id: f.node.id,
            kind: lower_struct_field_kind(_lctx, &f.node.kind),
            ty: lower_ty(_lctx, &f.node.ty),
            attrs: f.node.attrs.clone(),
        },
        span: f.span,
    }
}

pub fn lower_field(_lctx: &LoweringContext, f: &Field) -> hir::Field {
    hir::Field {
        name: respan(f.ident.span, f.ident.node.name),
        expr: lower_expr(_lctx, &f.expr), span: f.span
    }
}

pub fn lower_mt(_lctx: &LoweringContext, mt: &MutTy) -> hir::MutTy {
    hir::MutTy { ty: lower_ty(_lctx, &mt.ty), mutbl: lower_mutability(_lctx, mt.mutbl) }
}

pub fn lower_opt_bounds(_lctx: &LoweringContext, b: &Option<OwnedSlice<TyParamBound>>)
                        -> Option<OwnedSlice<hir::TyParamBound>> {
    b.as_ref().map(|ref bounds| lower_bounds(_lctx, bounds))
}

fn lower_bounds(_lctx: &LoweringContext, bounds: &TyParamBounds) -> hir::TyParamBounds {
    bounds.iter().map(|bound| lower_ty_param_bound(_lctx, bound)).collect()
}

fn lower_variant_arg(_lctx: &LoweringContext, va: &VariantArg) -> hir::VariantArg {
    hir::VariantArg { id: va.id, ty: lower_ty(_lctx, &va.ty) }
}

pub fn lower_block(_lctx: &LoweringContext, b: &Block) -> P<hir::Block> {
    P(hir::Block {
        id: b.id,
        stmts: b.stmts.iter().map(|s| lower_stmt(_lctx, s)).collect(),
        expr: b.expr.as_ref().map(|ref x| lower_expr(_lctx, x)),
        rules: lower_block_check_mode(_lctx, &b.rules),
        span: b.span,
    })
}

pub fn lower_item_underscore(_lctx: &LoweringContext, i: &Item_) -> hir::Item_ {
    match *i {
        ItemExternCrate(string) => hir::ItemExternCrate(string),
        ItemUse(ref view_path) => {
            hir::ItemUse(lower_view_path(_lctx, view_path))
        }
        ItemStatic(ref t, m, ref e) => {
            hir::ItemStatic(lower_ty(_lctx, t), lower_mutability(_lctx, m), lower_expr(_lctx, e))
        }
        ItemConst(ref t, ref e) => {
            hir::ItemConst(lower_ty(_lctx, t), lower_expr(_lctx, e))
        }
        ItemFn(ref decl, unsafety, constness, abi, ref generics, ref body) => {
            hir::ItemFn(
                lower_fn_decl(_lctx, decl),
                lower_unsafety(_lctx, unsafety),
                lower_constness(_lctx, constness),
                abi,
                lower_generics(_lctx, generics),
                lower_block(_lctx, body)
            )
        }
        ItemMod(ref m) => hir::ItemMod(lower_mod(_lctx, m)),
        ItemForeignMod(ref nm) => hir::ItemForeignMod(lower_foreign_mod(_lctx, nm)),
        ItemTy(ref t, ref generics) => {
            hir::ItemTy(lower_ty(_lctx, t), lower_generics(_lctx, generics))
        }
        ItemEnum(ref enum_definition, ref generics) => {
            hir::ItemEnum(
                hir::EnumDef {
                    variants: enum_definition.variants.iter().map(|x| {
                        lower_variant(_lctx, x)
                    }).collect(),
                },
                lower_generics(_lctx, generics))
        }
        ItemStruct(ref struct_def, ref generics) => {
            let struct_def = lower_struct_def(_lctx, struct_def);
            hir::ItemStruct(struct_def, lower_generics(_lctx, generics))
        }
        ItemDefaultImpl(unsafety, ref trait_ref) => {
            hir::ItemDefaultImpl(lower_unsafety(_lctx, unsafety), lower_trait_ref(_lctx, trait_ref))
        }
        ItemImpl(unsafety, polarity, ref generics, ref ifce, ref ty, ref impl_items) => {
            let new_impl_items =
                impl_items.iter().map(|item| lower_impl_item(_lctx, item)).collect();
            let ifce = ifce.as_ref().map(|trait_ref| lower_trait_ref(_lctx, trait_ref));
            hir::ItemImpl(lower_unsafety(_lctx, unsafety),
                          lower_impl_polarity(_lctx, polarity),
                          lower_generics(_lctx, generics),
                          ifce,
                          lower_ty(_lctx, ty),
                          new_impl_items)
        }
        ItemTrait(unsafety, ref generics, ref bounds, ref items) => {
            let bounds = lower_bounds(_lctx, bounds);
            let items = items.iter().map(|item| lower_trait_item(_lctx, item)).collect();
            hir::ItemTrait(lower_unsafety(_lctx, unsafety),
                           lower_generics(_lctx, generics),
                           bounds,
                           items)
        }
        ItemMac(_) => panic!("Shouldn't still be around"),
    }
}

pub fn lower_trait_item(_lctx: &LoweringContext, i: &TraitItem) -> P<hir::TraitItem> {
    P(hir::TraitItem {
        id: i.id,
        name: i.ident.name,
        attrs: i.attrs.clone(),
        node: match i.node {
            ConstTraitItem(ref ty, ref default) => {
                hir::ConstTraitItem(lower_ty(_lctx, ty),
                                    default.as_ref().map(|x| lower_expr(_lctx, x)))
            }
            MethodTraitItem(ref sig, ref body) => {
                hir::MethodTraitItem(lower_method_sig(_lctx, sig),
                                     body.as_ref().map(|x| lower_block(_lctx, x)))
            }
            TypeTraitItem(ref bounds, ref default) => {
                hir::TypeTraitItem(lower_bounds(_lctx, bounds),
                                   default.as_ref().map(|x| lower_ty(_lctx, x)))
            }
        },
        span: i.span,
    })
}

pub fn lower_impl_item(_lctx: &LoweringContext, i: &ImplItem) -> P<hir::ImplItem> {
    P(hir::ImplItem {
            id: i.id,
            name: i.ident.name,
            attrs: i.attrs.clone(),
            vis: lower_visibility(_lctx, i.vis),
            node: match i.node  {
            ConstImplItem(ref ty, ref expr) => {
                hir::ConstImplItem(lower_ty(_lctx, ty), lower_expr(_lctx, expr))
            }
            MethodImplItem(ref sig, ref body) => {
                hir::MethodImplItem(lower_method_sig(_lctx, sig),
                                    lower_block(_lctx, body))
            }
            TypeImplItem(ref ty) => hir::TypeImplItem(lower_ty(_lctx, ty)),
            MacImplItem(..) => panic!("Shouldn't exist any more"),
        },
        span: i.span,
    })
}

pub fn lower_mod(_lctx: &LoweringContext, m: &Mod) -> hir::Mod {
    hir::Mod { inner: m.inner, items: m.items.iter().map(|x| lower_item(_lctx, x)).collect() }
}

pub fn lower_crate(_lctx: &LoweringContext, c: &Crate) -> hir::Crate {
    hir::Crate {
        module: lower_mod(_lctx, &c.module),
        attrs: c.attrs.clone(),
        config: c.config.clone(),
        span: c.span,
        exported_macros: c.exported_macros.iter().map(|m| lower_macro_def(_lctx, m)).collect(),
    }
}

pub fn lower_macro_def(_lctx: &LoweringContext, m: &MacroDef) -> hir::MacroDef {
    hir::MacroDef {
        name: m.ident.name,
        attrs: m.attrs.clone(),
        id: m.id,
        span: m.span,
        imported_from: m.imported_from.map(|x| x.name),
        export: m.export,
        use_locally: m.use_locally,
        allow_internal_unstable: m.allow_internal_unstable,
        body: m.body.clone(),
    }
}

// fold one item into possibly many items
pub fn lower_item(_lctx: &LoweringContext, i: &Item) -> P<hir::Item> {
    P(lower_item_simple(_lctx, i))
}

// fold one item into exactly one item
pub fn lower_item_simple(_lctx: &LoweringContext, i: &Item) -> hir::Item {
    let node = lower_item_underscore(_lctx, &i.node);

    hir::Item {
        id: i.id,
        name: i.ident.name,
        attrs: i.attrs.clone(),
        node: node,
        vis: lower_visibility(_lctx, i.vis),
        span: i.span,
    }
}

pub fn lower_foreign_item(_lctx: &LoweringContext, i: &ForeignItem) -> P<hir::ForeignItem> {
    P(hir::ForeignItem {
        id: i.id,
        name: i.ident.name,
        attrs: i.attrs.clone(),
        node: match i.node {
            ForeignItemFn(ref fdec, ref generics) => {
                hir::ForeignItemFn(lower_fn_decl(_lctx, fdec), lower_generics(_lctx, generics))
            }
            ForeignItemStatic(ref t, m) => {
                hir::ForeignItemStatic(lower_ty(_lctx, t), m)
            }
        },
            vis: lower_visibility(_lctx, i.vis),
            span: i.span,
        })
}

pub fn lower_method_sig(_lctx: &LoweringContext, sig: &MethodSig) -> hir::MethodSig {
    hir::MethodSig {
        generics: lower_generics(_lctx, &sig.generics),
        abi: sig.abi,
        explicit_self: lower_explicit_self(_lctx, &sig.explicit_self),
        unsafety: lower_unsafety(_lctx, sig.unsafety),
        constness: lower_constness(_lctx, sig.constness),
        decl: lower_fn_decl(_lctx, &sig.decl),
    }
}

pub fn lower_unsafety(_lctx: &LoweringContext, u: Unsafety) -> hir::Unsafety {
    match u {
        Unsafety::Unsafe => hir::Unsafety::Unsafe,
        Unsafety::Normal => hir::Unsafety::Normal,
    }
}

pub fn lower_constness(_lctx: &LoweringContext, c: Constness) -> hir::Constness {
    match c {
        Constness::Const => hir::Constness::Const,
        Constness::NotConst => hir::Constness::NotConst,
    }
}

pub fn lower_unop(_lctx: &LoweringContext, u: UnOp) -> hir::UnOp {
    match u {
        UnDeref => hir::UnDeref,
        UnNot => hir::UnNot,
        UnNeg => hir::UnNeg,
    }
}

pub fn lower_binop(_lctx: &LoweringContext, b: BinOp) -> hir::BinOp {
    Spanned {
        node: match b.node {
            BiAdd => hir::BiAdd,
            BiSub => hir::BiSub,
            BiMul => hir::BiMul,
            BiDiv => hir::BiDiv,
            BiRem => hir::BiRem,
            BiAnd => hir::BiAnd,
            BiOr => hir::BiOr,
            BiBitXor => hir::BiBitXor,
            BiBitAnd => hir::BiBitAnd,
            BiBitOr => hir::BiBitOr,
            BiShl => hir::BiShl,
            BiShr => hir::BiShr,
            BiEq => hir::BiEq,
            BiLt => hir::BiLt,
            BiLe => hir::BiLe,
            BiNe => hir::BiNe,
            BiGe => hir::BiGe,
            BiGt => hir::BiGt,
        },
        span: b.span,
    }
}

pub fn lower_pat(_lctx: &LoweringContext, p: &Pat) -> P<hir::Pat> {
    P(hir::Pat {
            id: p.id,
            node: match p.node {
            PatWild(k) => hir::PatWild(lower_pat_wild_kind(_lctx, k)),
            PatIdent(ref binding_mode, pth1, ref sub) => {
                hir::PatIdent(lower_binding_mode(_lctx, binding_mode),
                        pth1,
                        sub.as_ref().map(|x| lower_pat(_lctx, x)))
            }
            PatLit(ref e) => hir::PatLit(lower_expr(_lctx, e)),
            PatEnum(ref pth, ref pats) => {
                hir::PatEnum(lower_path(_lctx, pth),
                             pats.as_ref()
                                 .map(|pats| pats.iter().map(|x| lower_pat(_lctx, x)).collect()))
            }
            PatQPath(ref qself, ref pth) => {
                let qself = hir::QSelf {
                    ty: lower_ty(_lctx, &qself.ty),
                    position: qself.position,
                };
                hir::PatQPath(qself, lower_path(_lctx, pth))
            }
            PatStruct(ref pth, ref fields, etc) => {
                let pth = lower_path(_lctx, pth);
                let fs = fields.iter().map(|f| {
                    Spanned { span: f.span,
                              node: hir::FieldPat {
                                  name: f.node.ident.name,
                                  pat: lower_pat(_lctx, &f.node.pat),
                                  is_shorthand: f.node.is_shorthand,
                              }}
                }).collect();
                hir::PatStruct(pth, fs, etc)
            }
            PatTup(ref elts) => hir::PatTup(elts.iter().map(|x| lower_pat(_lctx, x)).collect()),
            PatBox(ref inner) => hir::PatBox(lower_pat(_lctx, inner)),
            PatRegion(ref inner, mutbl) => hir::PatRegion(lower_pat(_lctx, inner),
                                                          lower_mutability(_lctx, mutbl)),
            PatRange(ref e1, ref e2) => {
                hir::PatRange(lower_expr(_lctx, e1), lower_expr(_lctx, e2))
            },
            PatVec(ref before, ref slice, ref after) => {
                hir::PatVec(before.iter().map(|x| lower_pat(_lctx, x)).collect(),
                       slice.as_ref().map(|x| lower_pat(_lctx, x)),
                       after.iter().map(|x| lower_pat(_lctx, x)).collect())
            }
            PatMac(_) => panic!("Shouldn't exist here"),
        },
        span: p.span,
    })
}

pub fn lower_expr(lctx: &LoweringContext, e: &Expr) -> P<hir::Expr> {
    P(hir::Expr {
            id: e.id,
            node: match e.node {
                ExprBox(ref e) => {
                    hir::ExprBox(lower_expr(lctx, e))
                }
                ExprVec(ref exprs) => {
                    hir::ExprVec(exprs.iter().map(|x| lower_expr(lctx, x)).collect())
                }
                ExprRepeat(ref expr, ref count) => {
                    hir::ExprRepeat(lower_expr(lctx, expr), lower_expr(lctx, count))
                }
                ExprTup(ref elts) => {
                    hir::ExprTup(elts.iter().map(|x| lower_expr(lctx, x)).collect())
                }
                ExprCall(ref f, ref args) => {
                    hir::ExprCall(lower_expr(lctx, f),
                             args.iter().map(|x| lower_expr(lctx, x)).collect())
                }
                ExprMethodCall(i, ref tps, ref args) => {
                    hir::ExprMethodCall(
                        respan(i.span, i.node.name),
                        tps.iter().map(|x| lower_ty(lctx, x)).collect(),
                        args.iter().map(|x| lower_expr(lctx, x)).collect())
                }
                ExprBinary(binop, ref lhs, ref rhs) => {
                    hir::ExprBinary(lower_binop(lctx, binop),
                            lower_expr(lctx, lhs),
                            lower_expr(lctx, rhs))
                }
                ExprUnary(op, ref ohs) => {
                    hir::ExprUnary(lower_unop(lctx, op), lower_expr(lctx, ohs))
                }
                ExprLit(ref l) => hir::ExprLit(P((**l).clone())),
                ExprCast(ref expr, ref ty) => {
                    hir::ExprCast(lower_expr(lctx, expr), lower_ty(lctx, ty))
                }
                ExprAddrOf(m, ref ohs) => {
                    hir::ExprAddrOf(lower_mutability(lctx, m), lower_expr(lctx, ohs))
                }
                // More complicated than you might expect because the else branch
                // might be `if let`.
                ExprIf(ref cond, ref blk, ref else_opt) => {
                    let else_opt = else_opt.as_ref().map(|els| match els.node {
                        ExprIfLet(..) => {
                            // wrap the if-let expr in a block
                            let span = els.span;
                            let blk = P(hir::Block {
                                stmts: vec![],
                                expr: Some(lower_expr(lctx, els)),
                                id: lctx.next_id(),
                                rules: hir::DefaultBlock,
                                span: span
                            });
                            expr_block(lctx, blk)
                        }
                        _ => lower_expr(lctx, els)
                    });

                    hir::ExprIf(lower_expr(lctx, cond),
                                lower_block(lctx, blk),
                                else_opt)
                }
                ExprWhile(ref cond, ref body, opt_ident) => {
                    hir::ExprWhile(lower_expr(lctx, cond),
                              lower_block(lctx, body),
                              opt_ident)
                }
                ExprLoop(ref body, opt_ident) => {
                    hir::ExprLoop(lower_block(lctx, body),
                            opt_ident)
                }
                ExprMatch(ref expr, ref arms, ref source) => {
                    hir::ExprMatch(lower_expr(lctx, expr),
                            arms.iter().map(|x| lower_arm(lctx, x)).collect(),
                            lower_match_source(lctx, source))
                }
                ExprClosure(capture_clause, ref decl, ref body) => {
                    hir::ExprClosure(lower_capture_clause(lctx, capture_clause),
                                lower_fn_decl(lctx, decl),
                                lower_block(lctx, body))
                }
                ExprBlock(ref blk) => hir::ExprBlock(lower_block(lctx, blk)),
                ExprAssign(ref el, ref er) => {
                    hir::ExprAssign(lower_expr(lctx, el), lower_expr(lctx, er))
                }
                ExprAssignOp(op, ref el, ref er) => {
                    hir::ExprAssignOp(lower_binop(lctx, op),
                                lower_expr(lctx, el),
                                lower_expr(lctx, er))
                }
                ExprField(ref el, ident) => {
                    hir::ExprField(lower_expr(lctx, el), respan(ident.span, ident.node.name))
                }
                ExprTupField(ref el, ident) => {
                    hir::ExprTupField(lower_expr(lctx, el), ident)
                }
                ExprIndex(ref el, ref er) => {
                    hir::ExprIndex(lower_expr(lctx, el), lower_expr(lctx, er))
                }
                ExprRange(ref e1, ref e2) => {
                    hir::ExprRange(e1.as_ref().map(|x| lower_expr(lctx, x)),
                              e2.as_ref().map(|x| lower_expr(lctx, x)))
                }
                ExprPath(ref qself, ref path) => {
                    let qself = qself.as_ref().map(|&QSelf { ref ty, position }| {
                        hir::QSelf {
                            ty: lower_ty(lctx, ty),
                            position: position
                        }
                    });
                    hir::ExprPath(qself, lower_path(lctx, path))
                }
                ExprBreak(opt_ident) => hir::ExprBreak(opt_ident),
                ExprAgain(opt_ident) => hir::ExprAgain(opt_ident),
                ExprRet(ref e) => hir::ExprRet(e.as_ref().map(|x| lower_expr(lctx, x))),
                ExprInlineAsm(InlineAsm {
                    ref inputs,
                    ref outputs,
                    ref asm,
                    asm_str_style,
                    ref clobbers,
                    volatile,
                    alignstack,
                    dialect,
                    expn_id,
                }) => hir::ExprInlineAsm(hir::InlineAsm {
                    inputs: inputs.iter().map(|&(ref c, ref input)| {
                        (c.clone(), lower_expr(lctx, input))
                    }).collect(),
                    outputs: outputs.iter().map(|&(ref c, ref out, ref is_rw)| {
                        (c.clone(), lower_expr(lctx, out), *is_rw)
                    }).collect(),
                    asm: asm.clone(),
                    asm_str_style: asm_str_style,
                    clobbers: clobbers.clone(),
                    volatile: volatile,
                    alignstack: alignstack,
                    dialect: dialect,
                    expn_id: expn_id,
                }),
                ExprStruct(ref path, ref fields, ref maybe_expr) => {
                    hir::ExprStruct(lower_path(lctx, path),
                            fields.iter().map(|x| lower_field(lctx, x)).collect(),
                            maybe_expr.as_ref().map(|x| lower_expr(lctx, x)))
                },
                ExprParen(ref ex) => {
                    return lower_expr(lctx, ex);
                }
                ExprInPlace(..) => {
                    panic!("todo");
                }

                // Desugar ExprIfLet
                // From: `if let <pat> = <sub_expr> <body> [<else_opt>]`
                ExprIfLet(ref pat, ref sub_expr, ref body, ref else_opt) => {
                    // to:
                    //
                    //   match <sub_expr> {
                    //     <pat> => <body>,
                    //     [_ if <else_opt_if_cond> => <else_opt_if_body>,]
                    //     _ => [<else_opt> | ()]
                    //   }

                    // `<pat> => <body>`
                    let pat_arm = {
                        let body_expr = expr_block(lctx, lower_block(lctx, body));
                        arm(vec![lower_pat(lctx, pat)], body_expr)
                    };

                    // `[_ if <else_opt_if_cond> => <else_opt_if_body>,]`
                    let mut else_opt = else_opt.as_ref().map(|e| lower_expr(lctx, e));
                    let else_if_arms = {
                        let mut arms = vec![];
                        loop {
                            let else_opt_continue = else_opt
                                .and_then(|els| els.and_then(|els| match els.node {
                                // else if
                                hir::ExprIf(cond, then, else_opt) => {
                                    let pat_under = pat_wild(lctx, e.span);
                                    arms.push(hir::Arm {
                                        attrs: vec![],
                                        pats: vec![pat_under],
                                        guard: Some(cond),
                                        body: expr_block(lctx, then)
                                    });
                                    else_opt.map(|else_opt| (else_opt, true))
                                }
                                _ => Some((P(els), false))
                            }));
                            match else_opt_continue {
                                Some((e, true)) => {
                                    else_opt = Some(e);
                                }
                                Some((e, false)) => {
                                    else_opt = Some(e);
                                    break;
                                }
                                None => {
                                    else_opt = None;
                                    break;
                                }
                            }
                        }
                        arms
                    };

                    let contains_else_clause = else_opt.is_some();

                    // `_ => [<else_opt> | ()]`
                    let else_arm = {
                        let pat_under = pat_wild(lctx, e.span);
                        let else_expr = else_opt.unwrap_or_else(|| expr_tuple(lctx, e.span, vec![]));
                        arm(vec![pat_under], else_expr)
                    };

                    let mut arms = Vec::with_capacity(else_if_arms.len() + 2);
                    arms.push(pat_arm);
                    arms.extend(else_if_arms);
                    arms.push(else_arm);

                    let match_expr = expr(lctx,
                                          e.span,
                                          hir::ExprMatch(lower_expr(lctx, sub_expr), arms,
                                                 hir::MatchSource::IfLetDesugar {
                                                     contains_else_clause: contains_else_clause,
                                                 }));
                    return match_expr;
                }

                // Desugar ExprWhileLet
                // From: `[opt_ident]: while let <pat> = <sub_expr> <body>`
                ExprWhileLet(ref pat, ref sub_expr, ref body, opt_ident) => {
                    // to:
                    //
                    //   [opt_ident]: loop {
                    //     match <sub_expr> {
                    //       <pat> => <body>,
                    //       _ => break
                    //     }
                    //   }

                    // `<pat> => <body>`
                    let pat_arm = {
                        let body_expr = expr_block(lctx, lower_block(lctx, body));
                        arm(vec![lower_pat(lctx, pat)], body_expr)
                    };

                    // `_ => break`
                    let break_arm = {
                        let pat_under = pat_wild(lctx, e.span);
                        let break_expr = expr_break(lctx, e.span);
                        arm(vec![pat_under], break_expr)
                    };

                    // // `match <sub_expr> { ... }`
                    let arms = vec![pat_arm, break_arm];
                    let match_expr = expr(lctx,
                                          e.span,
                                          hir::ExprMatch(lower_expr(lctx, sub_expr), arms, hir::MatchSource::WhileLetDesugar));

                    // `[opt_ident]: loop { ... }`
                    let loop_block = block_expr(lctx, match_expr);
                    return expr(lctx, e.span, hir::ExprLoop(loop_block, opt_ident));
                }

                // Desugar ExprForLoop
                // From: `[opt_ident]: for <pat> in <head> <body>`
                ExprForLoop(ref pat, ref head, ref body, opt_ident) => {
                    // to:
                    //
                    //   {
                    //     let result = match ::std::iter::IntoIterator::into_iter(<head>) {
                    //       mut iter => {
                    //         [opt_ident]: loop {
                    //           match ::std::iter::Iterator::next(&mut iter) {
                    //             ::std::option::Option::Some(<pat>) => <body>,
                    //             ::std::option::Option::None => break
                    //           }
                    //         }
                    //       }
                    //     };
                    //     result
                    //   }

                    // expand <head>
                    let head = lower_expr(lctx, head);

                    let iter = token::gensym_ident("iter");

                    // `::std::option::Option::Some(<pat>) => <body>`
                    let pat_arm = {
                        let body_block = lower_block(lctx, body);
                        let body_span = body_block.span;
                        let body_expr = P(hir::Expr {
                            id: lctx.next_id(),
                            node: hir::ExprBlock(body_block),
                            span: body_span,
                        });
                        let pat = lower_pat(lctx, pat);
                        let some_pat = pat_some(lctx, e.span, pat);

                        arm(vec![some_pat], body_expr)
                    };

                    // `::std::option::Option::None => break`
                    let break_arm = {
                        let break_expr = expr_break(lctx, e.span);

                        arm(vec![pat_none(lctx, e.span)], break_expr)
                    };

                    // `match ::std::iter::Iterator::next(&mut iter) { ... }`
                    let match_expr = {
                        let next_path = {
                            let strs = std_path(lctx, &["iter", "Iterator", "next"]);

                            path_global(e.span, strs)
                        };
                        let ref_mut_iter = expr_mut_addr_of(lctx, e.span, expr_ident(lctx, e.span, iter));
                        let next_expr =
                            expr_call(lctx, e.span, expr_path(lctx, next_path), vec![ref_mut_iter]);
                        let arms = vec![pat_arm, break_arm];

                        expr(lctx,
                             e.span,
                             hir::ExprMatch(next_expr, arms, hir::MatchSource::ForLoopDesugar))
                    };

                    // `[opt_ident]: loop { ... }`
                    let loop_block = block_expr(lctx, match_expr);
                    let loop_expr = expr(lctx, e.span, hir::ExprLoop(loop_block, opt_ident));

                    // `mut iter => { ... }`
                    let iter_arm = {
                        let iter_pat =
                            pat_ident_binding_mode(lctx, e.span, iter, hir::BindByValue(hir::MutMutable));
                        arm(vec![iter_pat], loop_expr)
                    };

                    // `match ::std::iter::IntoIterator::into_iter(<head>) { ... }`
                    let into_iter_expr = {
                        let into_iter_path = {
                            let strs = std_path(lctx, &["iter", "IntoIterator", "into_iter"]);

                            path_global(e.span, strs)
                        };

                        expr_call(lctx, e.span, expr_path(lctx, into_iter_path), vec![head])
                    };

                    let match_expr = expr_match(lctx, e.span, into_iter_expr, vec![iter_arm]);

                    // `{ let result = ...; result }`
                    let result_ident = token::gensym_ident("result");
                    return expr_block(lctx,
                                      block_all(lctx,
                                                e.span,
                                                vec![stmt_let(lctx, e.span,
                                                              false,
                                                              result_ident,
                                                              match_expr)],
                                                Some(expr_ident(lctx, e.span, result_ident))))
                }

                ExprMac(_) => panic!("Shouldn't exist here"),
            },
            span: e.span,
        })
}

pub fn lower_stmt(_lctx: &LoweringContext, s: &Stmt) -> P<hir::Stmt> {
    match s.node {
        StmtDecl(ref d, id) => {
            P(Spanned {
                node: hir::StmtDecl(lower_decl(_lctx, d), id),
                span: s.span
            })
        }
        StmtExpr(ref e, id) => {
            P(Spanned {
                node: hir::StmtExpr(lower_expr(_lctx, e), id),
                span: s.span
            })
        }
        StmtSemi(ref e, id) => {
            P(Spanned {
                node: hir::StmtSemi(lower_expr(_lctx, e), id),
                span: s.span
            })
        }
        StmtMac(..) => panic!("Shouldn't exist here"),
    }
}

pub fn lower_match_source(_lctx: &LoweringContext, m: &MatchSource) -> hir::MatchSource {
    match *m {
        MatchSource::Normal => hir::MatchSource::Normal,
        MatchSource::IfLetDesugar { contains_else_clause } => {
            hir::MatchSource::IfLetDesugar { contains_else_clause: contains_else_clause }
        }
        MatchSource::WhileLetDesugar => hir::MatchSource::WhileLetDesugar,
        MatchSource::ForLoopDesugar => hir::MatchSource::ForLoopDesugar,
    }
}

pub fn lower_capture_clause(_lctx: &LoweringContext, c: CaptureClause) -> hir::CaptureClause {
    match c {
        CaptureByValue => hir::CaptureByValue,
        CaptureByRef => hir::CaptureByRef,
    }
}

pub fn lower_visibility(_lctx: &LoweringContext, v: Visibility) -> hir::Visibility {
    match v {
        Public => hir::Public,
        Inherited => hir::Inherited,
    }
}

pub fn lower_block_check_mode(_lctx: &LoweringContext, b: &BlockCheckMode) -> hir::BlockCheckMode {
    match *b {
        DefaultBlock => hir::DefaultBlock,
        UnsafeBlock(u) => hir::UnsafeBlock(lower_unsafe_source(_lctx, u)),
        PushUnsafeBlock(u) => hir::PushUnsafeBlock(lower_unsafe_source(_lctx, u)),
        PopUnsafeBlock(u) => hir::PopUnsafeBlock(lower_unsafe_source(_lctx, u)),
    }
}

pub fn lower_pat_wild_kind(_lctx: &LoweringContext, p: PatWildKind) -> hir::PatWildKind {
    match p {
        PatWildSingle => hir::PatWildSingle,
        PatWildMulti => hir::PatWildMulti,
    }
}

pub fn lower_binding_mode(_lctx: &LoweringContext, b: &BindingMode) -> hir::BindingMode {
    match *b {
        BindByRef(m) => hir::BindByRef(lower_mutability(_lctx, m)),
        BindByValue(m) => hir::BindByValue(lower_mutability(_lctx, m)),
    }
}

pub fn lower_struct_field_kind(_lctx: &LoweringContext,
                               s: &StructFieldKind)
                               -> hir::StructFieldKind {
    match *s {
        NamedField(ident, vis) => hir::NamedField(ident.name, lower_visibility(_lctx, vis)),
        UnnamedField(vis) => hir::UnnamedField(lower_visibility(_lctx, vis)),
    }
}

pub fn lower_unsafe_source(_lctx: &LoweringContext, u: UnsafeSource) -> hir::UnsafeSource {
    match u {
        CompilerGenerated => hir::CompilerGenerated,
        UserProvided => hir::UserProvided,
    }
}

pub fn lower_impl_polarity(_lctx: &LoweringContext, i: ImplPolarity) -> hir::ImplPolarity {
    match i {
        ImplPolarity::Positive => hir::ImplPolarity::Positive,
        ImplPolarity::Negative => hir::ImplPolarity::Negative,
    }
}

pub fn lower_trait_bound_modifier(_lctx: &LoweringContext,
                                  f: TraitBoundModifier)
                                  -> hir::TraitBoundModifier {
    match f {
        TraitBoundModifier::None => hir::TraitBoundModifier::None,
        TraitBoundModifier::Maybe => hir::TraitBoundModifier::Maybe,
    }
}

// Helper methods for building HIR.

fn arm(pats: Vec<P<hir::Pat>>, expr: P<hir::Expr>) -> hir::Arm {
    hir::Arm {
        attrs: vec!(),
        pats: pats,
        guard: None,
        body: expr
    }
}

fn expr_break(lctx: &LoweringContext, span: Span) -> P<hir::Expr> {
    expr(lctx, span, hir::ExprBreak(None))
}

fn expr_call(lctx: &LoweringContext, span: Span, e: P<hir::Expr>, args: Vec<P<hir::Expr>>) -> P<hir::Expr> {
    expr(lctx, span, hir::ExprCall(e, args))
}

fn expr_ident(lctx: &LoweringContext, span: Span, id: Ident) -> P<hir::Expr> {
    expr_path(lctx, path_ident(span, id))
}

fn expr_mut_addr_of(lctx: &LoweringContext, span: Span, e: P<hir::Expr>) -> P<hir::Expr> {
    expr(lctx, span, hir::ExprAddrOf(hir::MutMutable, e))
}

fn expr_path(lctx: &LoweringContext, path: hir::Path) -> P<hir::Expr> {
    expr(lctx, path.span, hir::ExprPath(None, path))
}

fn expr_match(lctx: &LoweringContext, span: Span, arg: P<hir::Expr>, arms: Vec<hir::Arm>) -> P<hir::Expr> {
    expr(lctx, span, hir::ExprMatch(arg, arms, hir::MatchSource::Normal))
}

fn expr_block(lctx: &LoweringContext, b: P<hir::Block>) -> P<hir::Expr> {
    expr(lctx, b.span, hir::ExprBlock(b))
}

fn expr_tuple(lctx: &LoweringContext, sp: Span, exprs: Vec<P<hir::Expr>>) -> P<hir::Expr> {
    expr(lctx, sp, hir::ExprTup(exprs))
}

fn expr(lctx: &LoweringContext, span: Span, node: hir::Expr_) -> P<hir::Expr> {
    P(hir::Expr {
        id: lctx.next_id(),
        node: node,
        span: span,
    })
}

fn stmt_let(lctx: &LoweringContext, sp: Span, mutbl: bool, ident: Ident, ex: P<hir::Expr>) -> P<hir::Stmt> {
    let pat = if mutbl {
        pat_ident_binding_mode(lctx, sp, ident, hir::BindByValue(hir::MutMutable))
    } else {
        pat_ident(lctx, sp, ident)
    };
    let local = P(hir::Local {
        pat: pat,
        ty: None,
        init: Some(ex),
        id: lctx.next_id(),
        span: sp,
    });
    let decl = respan(sp, hir::DeclLocal(local));
    P(respan(sp, hir::StmtDecl(P(decl), lctx.next_id())))
}

fn block_expr(lctx: &LoweringContext, expr: P<hir::Expr>) -> P<hir::Block> {
    block_all(lctx, expr.span, Vec::new(), Some(expr))
}

fn block_all(lctx: &LoweringContext,
             span: Span,
             stmts: Vec<P<hir::Stmt>>,
             expr: Option<P<hir::Expr>>) -> P<hir::Block> {
        P(hir::Block {
            stmts: stmts,
            expr: expr,
            id: lctx.next_id(),
            rules: hir::DefaultBlock,
            span: span,
        })
}

fn pat_some(lctx: &LoweringContext, span: Span, pat: P<hir::Pat>) -> P<hir::Pat> {
    let some = std_path(lctx, &["option", "Option", "Some"]);
    let path = path_global(span, some);
    pat_enum(lctx, span, path, vec!(pat))
}

fn pat_none(lctx: &LoweringContext, span: Span) -> P<hir::Pat> {
    let none = std_path(lctx, &["option", "Option", "None"]);
    let path = path_global(span, none);
    pat_enum(lctx, span, path, vec![])
}

fn pat_enum(lctx: &LoweringContext, span: Span, path: hir::Path, subpats: Vec<P<hir::Pat>>) -> P<hir::Pat> {
    let pt = hir::PatEnum(path, Some(subpats));
    pat(lctx, span, pt)
}

fn pat_ident(lctx: &LoweringContext, span: Span, ident: Ident) -> P<hir::Pat> {
    pat_ident_binding_mode(lctx, span, ident, hir::BindByValue(hir::MutImmutable))
}

fn pat_ident_binding_mode(lctx: &LoweringContext,
                          span: Span,
                          ident: Ident,
                          bm: hir::BindingMode) -> P<hir::Pat> {
    let pat_ident = hir::PatIdent(bm, Spanned{span: span, node: ident}, None);
    pat(lctx, span, pat_ident)
}

fn pat_wild(lctx: &LoweringContext, span: Span) -> P<hir::Pat> {
    pat(lctx, span, hir::PatWild(hir::PatWildSingle))
}

fn pat(lctx: &LoweringContext, span: Span, pat: hir::Pat_) -> P<hir::Pat> {
    P(hir::Pat { id: lctx.next_id(), node: pat, span: span })
}

fn path_ident(span: Span, id: Ident) -> hir::Path {
    path(span, vec!(id))
}

fn path(span: Span, strs: Vec<Ident> ) -> hir::Path {
    path_all(span, false, strs, Vec::new(), Vec::new(), Vec::new())
}

fn path_global(span: Span, strs: Vec<Ident> ) -> hir::Path {
    path_all(span, true, strs, Vec::new(), Vec::new(), Vec::new())
}

fn path_all(sp: Span,
            global: bool,
            mut idents: Vec<Ident> ,
            lifetimes: Vec<hir::Lifetime>,
            types: Vec<P<hir::Ty>>,
            bindings: Vec<P<hir::TypeBinding>> )
            -> hir::Path {
    let last_identifier = idents.pop().unwrap();
    let mut segments: Vec<hir::PathSegment> = idents.into_iter()
                                                    .map(|ident| {
        hir::PathSegment {
            identifier: ident,
            parameters: hir::PathParameters::none(),
        }
    }).collect();
    segments.push(hir::PathSegment {
        identifier: last_identifier,
        parameters: hir::AngleBracketedParameters(hir::AngleBracketedParameterData {
            lifetimes: lifetimes,
            types: OwnedSlice::from_vec(types),
            bindings: OwnedSlice::from_vec(bindings),
        })
    });
    hir::Path {
        span: sp,
        global: global,
        segments: segments,
    }
}

fn std_path(lctx: &LoweringContext, components: &[&str]) -> Vec<Ident> {
    let mut v = Vec::new();
    if let Some(s) = lctx.crate_root {
        v.push(str_to_ident(s));
    }
    v.extend(components.iter().map(|s| str_to_ident(s)));
    return v
}
