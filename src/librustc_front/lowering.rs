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
use syntax::codemap::{respan, Spanned};
use syntax::owned_slice::OwnedSlice;


pub fn lower_view_path(view_path: &ViewPath) -> P<hir::ViewPath> {
    P(Spanned {
        node: match view_path.node {
            ViewPathSimple(ident, ref path) => {
                hir::ViewPathSimple(ident.name, lower_path(path))
            }
            ViewPathGlob(ref path) => {
                hir::ViewPathGlob(lower_path(path))
            }
            ViewPathList(ref path, ref path_list_idents) => {
                hir::ViewPathList(lower_path(path),
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

pub fn lower_arm(arm: &Arm) -> hir::Arm {
    hir::Arm {
        attrs: arm.attrs.clone(),
        pats: arm.pats.iter().map(|x| lower_pat(x)).collect(),
        guard: arm.guard.as_ref().map(|ref x| lower_expr(x)),
        body: lower_expr(&arm.body),
    }
}

pub fn lower_decl(d: &Decl) -> P<hir::Decl> {
    match d.node {
        DeclLocal(ref l) => P(Spanned {
            node: hir::DeclLocal(lower_local(l)),
            span: d.span
        }),
        DeclItem(ref it) => P(Spanned {
            node: hir::DeclItem(lower_item(it)),
            span: d.span
        }),
    }
}

pub fn lower_ty_binding(b: &TypeBinding) -> P<hir::TypeBinding> {
    P(hir::TypeBinding { id: b.id, name: b.ident.name, ty: lower_ty(&b.ty), span: b.span })
}

pub fn lower_ty(t: &Ty) -> P<hir::Ty> {
    P(hir::Ty {
        id: t.id,
        node: match t.node {
            TyInfer => hir::TyInfer,
            TyVec(ref ty) => hir::TyVec(lower_ty(ty)),
            TyPtr(ref mt) => hir::TyPtr(lower_mt(mt)),
            TyRptr(ref region, ref mt) => {
                hir::TyRptr(lower_opt_lifetime(region), lower_mt(mt))
            }
            TyBareFn(ref f) => {
                hir::TyBareFn(P(hir::BareFnTy {
                    lifetimes: lower_lifetime_defs(&f.lifetimes),
                    unsafety: lower_unsafety(f.unsafety),
                    abi: f.abi,
                    decl: lower_fn_decl(&f.decl)
                }))
            }
            TyTup(ref tys) => hir::TyTup(tys.iter().map(|ty| lower_ty(ty)).collect()),
            TyParen(ref ty) => hir::TyParen(lower_ty(ty)),
            TyPath(ref qself, ref path) => {
                let qself = qself.as_ref().map(|&QSelf { ref ty, position }| {
                    hir::QSelf {
                        ty: lower_ty(ty),
                        position: position
                    }
                });
                hir::TyPath(qself, lower_path(path))
            }
            TyObjectSum(ref ty, ref bounds) => {
                hir::TyObjectSum(lower_ty(ty),
                            lower_bounds(bounds))
            }
            TyFixedLengthVec(ref ty, ref e) => {
                hir::TyFixedLengthVec(lower_ty(ty), lower_expr(e))
            }
            TyTypeof(ref expr) => {
                hir::TyTypeof(lower_expr(expr))
            }
            TyPolyTraitRef(ref bounds) => {
                hir::TyPolyTraitRef(bounds.iter().map(|b| lower_ty_param_bound(b)).collect())
            }
            TyMac(_) => panic!("TyMac should have been expanded by now."),
        },
        span: t.span,
    })
}

pub fn lower_foreign_mod(fm: &ForeignMod) -> hir::ForeignMod {
    hir::ForeignMod {
        abi: fm.abi,
        items: fm.items.iter().map(|x| lower_foreign_item(x)).collect(),
    }
}

pub fn lower_variant(v: &Variant) -> P<hir::Variant> {
    P(Spanned {
        node: hir::Variant_ {
            id: v.node.id,
            name: v.node.name.name,
            attrs: v.node.attrs.clone(),
            kind: match v.node.kind {
                TupleVariantKind(ref variant_args) => {
                    hir::TupleVariantKind(variant_args.iter().map(|ref x|
                        lower_variant_arg(x)).collect())
                }
                StructVariantKind(ref struct_def) => {
                    hir::StructVariantKind(lower_struct_def(struct_def))
                }
            },
            disr_expr: v.node.disr_expr.as_ref().map(|e| lower_expr(e)),
        },
        span: v.span,
    })
}

pub fn lower_path(p: &Path) -> hir::Path {
    hir::Path {
        global: p.global,
        segments: p.segments.iter().map(|&PathSegment {identifier, ref parameters}|
            hir::PathSegment {
                identifier: identifier,
                parameters: lower_path_parameters(parameters),
            }).collect(),
        span: p.span,
    }
}

pub fn lower_path_parameters(path_parameters: &PathParameters) -> hir::PathParameters {
    match *path_parameters {
        AngleBracketedParameters(ref data) =>
            hir::AngleBracketedParameters(lower_angle_bracketed_parameter_data(data)),
        ParenthesizedParameters(ref data) =>
            hir::ParenthesizedParameters(lower_parenthesized_parameter_data(data)),
    }
}

pub fn lower_angle_bracketed_parameter_data(data: &AngleBracketedParameterData)
                                            -> hir::AngleBracketedParameterData {
    let &AngleBracketedParameterData { ref lifetimes, ref types, ref bindings } = data;
    hir::AngleBracketedParameterData {
        lifetimes: lower_lifetimes(lifetimes),
        types: types.iter().map(|ty| lower_ty(ty)).collect(),
        bindings: bindings.iter().map(|b| lower_ty_binding(b)).collect(),
    }
}

pub fn lower_parenthesized_parameter_data(data: &ParenthesizedParameterData)
                                          -> hir::ParenthesizedParameterData {
    let &ParenthesizedParameterData { ref inputs, ref output, span } = data;
    hir::ParenthesizedParameterData {
        inputs: inputs.iter().map(|ty| lower_ty(ty)).collect(),
        output: output.as_ref().map(|ty| lower_ty(ty)),
        span: span,
    }
}

pub fn lower_local(l: &Local) -> P<hir::Local> {
    P(hir::Local {
            id: l.id,
            ty: l.ty.as_ref().map(|t| lower_ty(t)),
            pat: lower_pat(&l.pat),
            init: l.init.as_ref().map(|e| lower_expr(e)),
            span: l.span,
        })
}

pub fn lower_explicit_self_underscore(es: &ExplicitSelf_) -> hir::ExplicitSelf_ {
    match *es {
        SelfStatic => hir::SelfStatic,
        SelfValue(v) => hir::SelfValue(v.name),
        SelfRegion(ref lifetime, m, ident) => {
            hir::SelfRegion(lower_opt_lifetime(lifetime), lower_mutability(m), ident.name)
        }
        SelfExplicit(ref typ, ident) => {
            hir::SelfExplicit(lower_ty(typ), ident.name)
        }
    }
}

pub fn lower_mutability(m: Mutability) -> hir::Mutability {
    match m {
        MutMutable => hir::MutMutable,
        MutImmutable => hir::MutImmutable,
    }
}

pub fn lower_explicit_self(s: &ExplicitSelf) -> hir::ExplicitSelf {
    Spanned { node: lower_explicit_self_underscore(&s.node), span: s.span }
}

pub fn lower_arg(arg: &Arg) -> hir::Arg {
    hir::Arg { id: arg.id, pat: lower_pat(&arg.pat), ty: lower_ty(&arg.ty) }
}

pub fn lower_fn_decl(decl: &FnDecl) -> P<hir::FnDecl> {
    P(hir::FnDecl {
        inputs: decl.inputs.iter().map(|x| lower_arg(x)).collect(),
        output: match decl.output {
            Return(ref ty) => hir::Return(lower_ty(ty)),
            DefaultReturn(span) => hir::DefaultReturn(span),
            NoReturn(span) => hir::NoReturn(span)
        },
        variadic: decl.variadic,
    })
}

pub fn lower_ty_param_bound(tpb: &TyParamBound) -> hir::TyParamBound {
    match *tpb {
        TraitTyParamBound(ref ty, modifier) => {
            hir::TraitTyParamBound(lower_poly_trait_ref(ty), lower_trait_bound_modifier(modifier))
        }
        RegionTyParamBound(ref lifetime) => hir::RegionTyParamBound(lower_lifetime(lifetime)),
    }
}

pub fn lower_ty_param(tp: &TyParam) -> hir::TyParam {
    hir::TyParam {
        id: tp.id,
        name: tp.ident.name,
        bounds: lower_bounds(&tp.bounds),
        default: tp.default.as_ref().map(|x| lower_ty(x)),
        span: tp.span,
    }
}

pub fn lower_ty_params(tps: &OwnedSlice<TyParam>) -> OwnedSlice<hir::TyParam> {
    tps.iter().map(|tp| lower_ty_param(tp)).collect()
}

pub fn lower_lifetime(l: &Lifetime) -> hir::Lifetime {
    hir::Lifetime { id: l.id, name: l.name, span: l.span }
}

pub fn lower_lifetime_def(l: &LifetimeDef) -> hir::LifetimeDef {
    hir::LifetimeDef { lifetime: lower_lifetime(&l.lifetime), bounds: lower_lifetimes(&l.bounds) }
}

pub fn lower_lifetimes(lts: &Vec<Lifetime>) -> Vec<hir::Lifetime> {
    lts.iter().map(|l| lower_lifetime(l)).collect()
}

pub fn lower_lifetime_defs(lts: &Vec<LifetimeDef>) -> Vec<hir::LifetimeDef> {
    lts.iter().map(|l| lower_lifetime_def(l)).collect()
}

pub fn lower_opt_lifetime(o_lt: &Option<Lifetime>) -> Option<hir::Lifetime> {
    o_lt.as_ref().map(|lt| lower_lifetime(lt))
}

pub fn lower_generics(g: &Generics) -> hir::Generics {
    hir::Generics {
        ty_params: lower_ty_params(&g.ty_params),
        lifetimes: lower_lifetime_defs(&g.lifetimes),
        where_clause: lower_where_clause(&g.where_clause),
    }
}

pub fn lower_where_clause(wc: &WhereClause) -> hir::WhereClause {
    hir::WhereClause {
        id: wc.id,
        predicates: wc.predicates.iter().map(|predicate|
            lower_where_predicate(predicate)).collect(),
    }
}

pub fn lower_where_predicate(pred: &WherePredicate) -> hir::WherePredicate {
    match *pred {
        WherePredicate::BoundPredicate(WhereBoundPredicate{ ref bound_lifetimes,
                                                            ref bounded_ty,
                                                            ref bounds,
                                                            span}) => {
            hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                bound_lifetimes: lower_lifetime_defs(bound_lifetimes),
                bounded_ty: lower_ty(bounded_ty),
                bounds: bounds.iter().map(|x| lower_ty_param_bound(x)).collect(),
                span: span
            })
        }
        WherePredicate::RegionPredicate(WhereRegionPredicate{ ref lifetime,
                                                              ref bounds,
                                                              span}) => {
            hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                span: span,
                lifetime: lower_lifetime(lifetime),
                bounds: bounds.iter().map(|bound| lower_lifetime(bound)).collect()
            })
        }
        WherePredicate::EqPredicate(WhereEqPredicate{ id,
                                                      ref path,
                                                      ref ty,
                                                      span}) => {
            hir::WherePredicate::EqPredicate(hir::WhereEqPredicate{
                id: id,
                path: lower_path(path),
                ty:lower_ty(ty),
                span: span
            })
        }
    }
}

pub fn lower_struct_def(sd: &StructDef) -> P<hir::StructDef> {
    P(hir::StructDef {
        fields: sd.fields.iter().map(|f| lower_struct_field(f)).collect(),
        ctor_id: sd.ctor_id,
    })
}

pub fn lower_trait_ref(p: &TraitRef) -> hir::TraitRef {
    hir::TraitRef { path: lower_path(&p.path), ref_id: p.ref_id }
}

pub fn lower_poly_trait_ref(p: &PolyTraitRef) -> hir::PolyTraitRef {
    hir::PolyTraitRef {
        bound_lifetimes: lower_lifetime_defs(&p.bound_lifetimes),
        trait_ref: lower_trait_ref(&p.trait_ref),
        span: p.span,
    }
}

pub fn lower_struct_field(f: &StructField) -> hir::StructField {
    Spanned {
        node: hir::StructField_ {
            id: f.node.id,
            kind: lower_struct_field_kind(&f.node.kind),
            ty: lower_ty(&f.node.ty),
            attrs: f.node.attrs.clone(),
        },
        span: f.span,
    }
}

pub fn lower_field(f: &Field) -> hir::Field {
    hir::Field {
        name: respan(f.ident.span, f.ident.node.name),
        expr: lower_expr(&f.expr), span: f.span
    }
}

pub fn lower_mt(mt: &MutTy) -> hir::MutTy {
    hir::MutTy { ty: lower_ty(&mt.ty), mutbl: lower_mutability(mt.mutbl) }
}

pub fn lower_opt_bounds(b: &Option<OwnedSlice<TyParamBound>>)
                        -> Option<OwnedSlice<hir::TyParamBound>> {
    b.as_ref().map(|ref bounds| lower_bounds(bounds))
}

fn lower_bounds(bounds: &TyParamBounds) -> hir::TyParamBounds {
    bounds.iter().map(|bound| lower_ty_param_bound(bound)).collect()
}

fn lower_variant_arg(va: &VariantArg) -> hir::VariantArg {
    hir::VariantArg { id: va.id, ty: lower_ty(&va.ty) }
}

pub fn lower_block(b: &Block) -> P<hir::Block> {
    P(hir::Block {
        id: b.id,
        stmts: b.stmts.iter().map(|s| lower_stmt(s)).collect(),
        expr: b.expr.as_ref().map(|ref x| lower_expr(x)),
        rules: lower_block_check_mode(&b.rules),
        span: b.span,
    })
}

pub fn lower_item_underscore(i: &Item_) -> hir::Item_ {
    match *i {
        ItemExternCrate(string) => hir::ItemExternCrate(string),
        ItemUse(ref view_path) => {
            hir::ItemUse(lower_view_path(view_path))
        }
        ItemStatic(ref t, m, ref e) => {
            hir::ItemStatic(lower_ty(t), lower_mutability(m), lower_expr(e))
        }
        ItemConst(ref t, ref e) => {
            hir::ItemConst(lower_ty(t), lower_expr(e))
        }
        ItemFn(ref decl, unsafety, constness, abi, ref generics, ref body) => {
            hir::ItemFn(
                lower_fn_decl(decl),
                lower_unsafety(unsafety),
                lower_constness(constness),
                abi,
                lower_generics(generics),
                lower_block(body)
            )
        }
        ItemMod(ref m) => hir::ItemMod(lower_mod(m)),
        ItemForeignMod(ref nm) => hir::ItemForeignMod(lower_foreign_mod(nm)),
        ItemTy(ref t, ref generics) => {
            hir::ItemTy(lower_ty(t), lower_generics(generics))
        }
        ItemEnum(ref enum_definition, ref generics) => {
            hir::ItemEnum(
                hir::EnumDef {
                    variants: enum_definition.variants.iter().map(|x| lower_variant(x)).collect(),
                },
                lower_generics(generics))
        }
        ItemStruct(ref struct_def, ref generics) => {
            let struct_def = lower_struct_def(struct_def);
            hir::ItemStruct(struct_def, lower_generics(generics))
        }
        ItemDefaultImpl(unsafety, ref trait_ref) => {
            hir::ItemDefaultImpl(lower_unsafety(unsafety), lower_trait_ref(trait_ref))
        }
        ItemImpl(unsafety, polarity, ref generics, ref ifce, ref ty, ref impl_items) => {
            let new_impl_items = impl_items.iter().map(|item| lower_impl_item(item)).collect();
            let ifce = ifce.as_ref().map(|trait_ref| lower_trait_ref(trait_ref));
            hir::ItemImpl(lower_unsafety(unsafety),
                          lower_impl_polarity(polarity),
                          lower_generics(generics),
                          ifce,
                          lower_ty(ty),
                          new_impl_items)
        }
        ItemTrait(unsafety, ref generics, ref bounds, ref items) => {
            let bounds = lower_bounds(bounds);
            let items = items.iter().map(|item| lower_trait_item(item)).collect();
            hir::ItemTrait(lower_unsafety(unsafety),
                           lower_generics(generics),
                           bounds,
                           items)
        }
        ItemMac(_) => panic!("Shouldn't still be around"),
    }
}

pub fn lower_trait_item(i: &TraitItem) -> P<hir::TraitItem> {
    P(hir::TraitItem {
            id: i.id,
            name: i.ident.name,
            attrs: i.attrs.clone(),
            node: match i.node {
            ConstTraitItem(ref ty, ref default) => {
                hir::ConstTraitItem(lower_ty(ty),
                                    default.as_ref().map(|x| lower_expr(x)))
            }
            MethodTraitItem(ref sig, ref body) => {
                hir::MethodTraitItem(lower_method_sig(sig),
                                     body.as_ref().map(|x| lower_block(x)))
            }
            TypeTraitItem(ref bounds, ref default) => {
                hir::TypeTraitItem(lower_bounds(bounds),
                                   default.as_ref().map(|x| lower_ty(x)))
            }
        },
            span: i.span,
        })
}

pub fn lower_impl_item(i: &ImplItem) -> P<hir::ImplItem> {
    P(hir::ImplItem {
            id: i.id,
            name: i.ident.name,
            attrs: i.attrs.clone(),
            vis: lower_visibility(i.vis),
            node: match i.node  {
            ConstImplItem(ref ty, ref expr) => {
                hir::ConstImplItem(lower_ty(ty), lower_expr(expr))
            }
            MethodImplItem(ref sig, ref body) => {
                hir::MethodImplItem(lower_method_sig(sig),
                                    lower_block(body))
            }
            TypeImplItem(ref ty) => hir::TypeImplItem(lower_ty(ty)),
            MacImplItem(..) => panic!("Shouldn't exist any more"),
        },
            span: i.span,
        })
}

pub fn lower_mod(m: &Mod) -> hir::Mod {
    hir::Mod { inner: m.inner, items: m.items.iter().map(|x| lower_item(x)).collect() }
}

pub fn lower_crate(c: &Crate) -> hir::Crate {
    hir::Crate {
        module: lower_mod(&c.module),
        attrs: c.attrs.clone(),
        config: c.config.clone(),
        span: c.span,
        exported_macros: c.exported_macros.iter().map(|m| lower_macro_def(m)).collect(),
    }
}

pub fn lower_macro_def(m: &MacroDef) -> hir::MacroDef {
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
pub fn lower_item(i: &Item) -> P<hir::Item> {
    P(lower_item_simple(i))
}

// fold one item into exactly one item
pub fn lower_item_simple(i: &Item) -> hir::Item {
    let node = lower_item_underscore(&i.node);

    hir::Item {
        id: i.id,
        name: i.ident.name,
        attrs: i.attrs.clone(),
        node: node,
        vis: lower_visibility(i.vis),
        span: i.span,
    }
}

pub fn lower_foreign_item(i: &ForeignItem) -> P<hir::ForeignItem> {
    P(hir::ForeignItem {
            id: i.id,
            name: i.ident.name,
            attrs: i.attrs.clone(),
            node: match i.node {
            ForeignItemFn(ref fdec, ref generics) => {
                hir::ForeignItemFn(lower_fn_decl(fdec), lower_generics(generics))
            }
            ForeignItemStatic(ref t, m) => {
                hir::ForeignItemStatic(lower_ty(t), m)
            }
        },
            vis: lower_visibility(i.vis),
            span: i.span,
        })
}

pub fn lower_method_sig(sig: &MethodSig) -> hir::MethodSig {
    hir::MethodSig {
        generics: lower_generics(&sig.generics),
        abi: sig.abi,
        explicit_self: lower_explicit_self(&sig.explicit_self),
        unsafety: lower_unsafety(sig.unsafety),
        constness: lower_constness(sig.constness),
        decl: lower_fn_decl(&sig.decl),
    }
}

pub fn lower_unsafety(u: Unsafety) -> hir::Unsafety {
    match u {
        Unsafety::Unsafe => hir::Unsafety::Unsafe,
        Unsafety::Normal => hir::Unsafety::Normal,
    }
}

pub fn lower_constness(c: Constness) -> hir::Constness {
    match c {
        Constness::Const => hir::Constness::Const,
        Constness::NotConst => hir::Constness::NotConst,
    }
}

pub fn lower_unop(u: UnOp) -> hir::UnOp {
    match u {
        UnUniq => hir::UnUniq,
        UnDeref => hir::UnDeref,
        UnNot => hir::UnNot,
        UnNeg => hir::UnNeg,
    }
}

pub fn lower_binop(b: BinOp) -> hir::BinOp {
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

pub fn lower_pat(p: &Pat) -> P<hir::Pat> {
    P(hir::Pat {
            id: p.id,
            node: match p.node {
            PatWild(k) => hir::PatWild(lower_pat_wild_kind(k)),
            PatIdent(ref binding_mode, pth1, ref sub) => {
                hir::PatIdent(lower_binding_mode(binding_mode),
                        pth1,
                        sub.as_ref().map(|x| lower_pat(x)))
            }
            PatLit(ref e) => hir::PatLit(lower_expr(e)),
            PatEnum(ref pth, ref pats) => {
                hir::PatEnum(lower_path(pth),
                        pats.as_ref().map(|pats| pats.iter().map(|x| lower_pat(x)).collect()))
            }
            PatQPath(ref qself, ref pth) => {
                let qself = hir::QSelf {
                    ty: lower_ty(&qself.ty),
                    position: qself.position,
                };
                hir::PatQPath(qself, lower_path(pth))
            }
            PatStruct(ref pth, ref fields, etc) => {
                let pth = lower_path(pth);
                let fs = fields.iter().map(|f| {
                    Spanned { span: f.span,
                              node: hir::FieldPat {
                                  name: f.node.ident.name,
                                  pat: lower_pat(&f.node.pat),
                                  is_shorthand: f.node.is_shorthand,
                              }}
                }).collect();
                hir::PatStruct(pth, fs, etc)
            }
            PatTup(ref elts) => hir::PatTup(elts.iter().map(|x| lower_pat(x)).collect()),
            PatBox(ref inner) => hir::PatBox(lower_pat(inner)),
            PatRegion(ref inner, mutbl) => hir::PatRegion(lower_pat(inner),
                                                          lower_mutability(mutbl)),
            PatRange(ref e1, ref e2) => {
                hir::PatRange(lower_expr(e1), lower_expr(e2))
            },
            PatVec(ref before, ref slice, ref after) => {
                hir::PatVec(before.iter().map(|x| lower_pat(x)).collect(),
                       slice.as_ref().map(|x| lower_pat(x)),
                       after.iter().map(|x| lower_pat(x)).collect())
            }
            PatMac(_) => panic!("Shouldn't exist here"),
        },
            span: p.span,
        })
}

pub fn lower_expr(e: &Expr) -> P<hir::Expr> {
    P(hir::Expr {
            id: e.id,
            node: match e.node {
                ExprBox(ref p, ref e) => {
                    hir::ExprBox(p.as_ref().map(|e| lower_expr(e)), lower_expr(e))
                }
                ExprVec(ref exprs) => {
                    hir::ExprVec(exprs.iter().map(|x| lower_expr(x)).collect())
                }
                ExprRepeat(ref expr, ref count) => {
                    hir::ExprRepeat(lower_expr(expr), lower_expr(count))
                }
                ExprTup(ref elts) => hir::ExprTup(elts.iter().map(|x| lower_expr(x)).collect()),
                ExprCall(ref f, ref args) => {
                    hir::ExprCall(lower_expr(f),
                             args.iter().map(|x| lower_expr(x)).collect())
                }
                ExprMethodCall(i, ref tps, ref args) => {
                    hir::ExprMethodCall(
                        respan(i.span, i.node.name),
                        tps.iter().map(|x| lower_ty(x)).collect(),
                        args.iter().map(|x| lower_expr(x)).collect())
                }
                ExprBinary(binop, ref lhs, ref rhs) => {
                    hir::ExprBinary(lower_binop(binop),
                            lower_expr(lhs),
                            lower_expr(rhs))
                }
                ExprUnary(op, ref ohs) => {
                    hir::ExprUnary(lower_unop(op), lower_expr(ohs))
                }
                ExprLit(ref l) => hir::ExprLit(P((**l).clone())),
                ExprCast(ref expr, ref ty) => {
                    hir::ExprCast(lower_expr(expr), lower_ty(ty))
                }
                ExprAddrOf(m, ref ohs) => hir::ExprAddrOf(lower_mutability(m), lower_expr(ohs)),
                ExprIf(ref cond, ref tr, ref fl) => {
                    hir::ExprIf(lower_expr(cond),
                           lower_block(tr),
                           fl.as_ref().map(|x| lower_expr(x)))
                }
                ExprWhile(ref cond, ref body, opt_ident) => {
                    hir::ExprWhile(lower_expr(cond),
                              lower_block(body),
                              opt_ident)
                }
                ExprLoop(ref body, opt_ident) => {
                    hir::ExprLoop(lower_block(body),
                            opt_ident)
                }
                ExprMatch(ref expr, ref arms, ref source) => {
                    hir::ExprMatch(lower_expr(expr),
                            arms.iter().map(|x| lower_arm(x)).collect(),
                            lower_match_source(source))
                }
                ExprClosure(capture_clause, ref decl, ref body) => {
                    hir::ExprClosure(lower_capture_clause(capture_clause),
                                lower_fn_decl(decl),
                                lower_block(body))
                }
                ExprBlock(ref blk) => hir::ExprBlock(lower_block(blk)),
                ExprAssign(ref el, ref er) => {
                    hir::ExprAssign(lower_expr(el), lower_expr(er))
                }
                ExprAssignOp(op, ref el, ref er) => {
                    hir::ExprAssignOp(lower_binop(op),
                                lower_expr(el),
                                lower_expr(er))
                }
                ExprField(ref el, ident) => {
                    hir::ExprField(lower_expr(el), respan(ident.span, ident.node.name))
                }
                ExprTupField(ref el, ident) => {
                    hir::ExprTupField(lower_expr(el), ident)
                }
                ExprIndex(ref el, ref er) => {
                    hir::ExprIndex(lower_expr(el), lower_expr(er))
                }
                ExprRange(ref e1, ref e2) => {
                    hir::ExprRange(e1.as_ref().map(|x| lower_expr(x)),
                              e2.as_ref().map(|x| lower_expr(x)))
                }
                ExprPath(ref qself, ref path) => {
                    let qself = qself.as_ref().map(|&QSelf { ref ty, position }| {
                        hir::QSelf {
                            ty: lower_ty(ty),
                            position: position
                        }
                    });
                    hir::ExprPath(qself, lower_path(path))
                }
                ExprBreak(opt_ident) => hir::ExprBreak(opt_ident),
                ExprAgain(opt_ident) => hir::ExprAgain(opt_ident),
                ExprRet(ref e) => hir::ExprRet(e.as_ref().map(|x| lower_expr(x))),
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
                        (c.clone(), lower_expr(input))
                    }).collect(),
                    outputs: outputs.iter().map(|&(ref c, ref out, ref is_rw)| {
                        (c.clone(), lower_expr(out), *is_rw)
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
                    hir::ExprStruct(lower_path(path),
                            fields.iter().map(|x| lower_field(x)).collect(),
                            maybe_expr.as_ref().map(|x| lower_expr(x)))
                },
                ExprParen(ref ex) => {
                    return lower_expr(ex);
                }
                ExprIfLet(..) |
                ExprWhileLet(..) |
                ExprForLoop(..) |
                ExprMac(_) => panic!("Shouldn't exist here"),
            },
            span: e.span,
        })
}

pub fn lower_stmt(s: &Stmt) -> P<hir::Stmt> {
    match s.node {
        StmtDecl(ref d, id) => {
            P(Spanned {
                node: hir::StmtDecl(lower_decl(d), id),
                span: s.span
            })
        }
        StmtExpr(ref e, id) => {
            P(Spanned {
                node: hir::StmtExpr(lower_expr(e), id),
                span: s.span
            })
        }
        StmtSemi(ref e, id) => {
            P(Spanned {
                node: hir::StmtSemi(lower_expr(e), id),
                span: s.span
            })
        }
        StmtMac(..) => panic!("Shouldn't exist here")
    }
}

pub fn lower_match_source(m: &MatchSource) -> hir::MatchSource {
    match *m {
        MatchSource::Normal => hir::MatchSource::Normal,
        MatchSource::IfLetDesugar { contains_else_clause } => {
            hir::MatchSource::IfLetDesugar { contains_else_clause: contains_else_clause }
        }
        MatchSource::WhileLetDesugar => hir::MatchSource::WhileLetDesugar,
        MatchSource::ForLoopDesugar => hir::MatchSource::ForLoopDesugar,
    }
}

pub fn lower_capture_clause(c: CaptureClause) -> hir::CaptureClause {
    match c {
        CaptureByValue => hir::CaptureByValue,
        CaptureByRef => hir::CaptureByRef,
    }
}

pub fn lower_visibility(v: Visibility) -> hir::Visibility {
    match v {
        Public => hir::Public,
        Inherited => hir::Inherited,
    }
}

pub fn lower_block_check_mode(b: &BlockCheckMode) -> hir::BlockCheckMode {
    match *b {
        DefaultBlock => hir::DefaultBlock,
        UnsafeBlock(u) => hir::UnsafeBlock(lower_unsafe_source(u)),
        PushUnsafeBlock(u) => hir::PushUnsafeBlock(lower_unsafe_source(u)),
        PopUnsafeBlock(u) => hir::PopUnsafeBlock(lower_unsafe_source(u)),
    }
}

pub fn lower_pat_wild_kind(p: PatWildKind) -> hir::PatWildKind {
    match p {
        PatWildSingle => hir::PatWildSingle,
        PatWildMulti => hir::PatWildMulti,
    }
}

pub fn lower_binding_mode(b: &BindingMode) -> hir::BindingMode {
    match *b {
        BindByRef(m) => hir::BindByRef(lower_mutability(m)),
        BindByValue(m) => hir::BindByValue(lower_mutability(m)),
    }
}

pub fn lower_struct_field_kind(s: &StructFieldKind) -> hir::StructFieldKind {
    match *s {
        NamedField(ident, vis) => hir::NamedField(ident.name, lower_visibility(vis)),
        UnnamedField(vis) => hir::UnnamedField(lower_visibility(vis)),
    }
}

pub fn lower_unsafe_source(u: UnsafeSource) -> hir::UnsafeSource {
    match u {
        CompilerGenerated => hir::CompilerGenerated,
        UserProvided => hir::UserProvided,
    }
}

pub fn lower_impl_polarity(i: ImplPolarity) -> hir::ImplPolarity {
    match i {
        ImplPolarity::Positive => hir::ImplPolarity::Positive,
        ImplPolarity::Negative => hir::ImplPolarity::Negative,
    }
}

pub fn lower_trait_bound_modifier(f: TraitBoundModifier) -> hir::TraitBoundModifier {
    match f {
        TraitBoundModifier::None => hir::TraitBoundModifier::None,
        TraitBoundModifier::Maybe => hir::TraitBoundModifier::Maybe,
    }
}
