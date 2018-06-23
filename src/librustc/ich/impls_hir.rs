// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains `HashStable` implementations for various HIR data
//! types in no particular order.

use hir;
use hir::map::DefPathHash;
use hir::def_id::{DefId, LocalDefId, CrateNum, CRATE_DEF_INDEX};
use ich::{StableHashingContext, NodeIdHashingMode};
use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey,
                                           StableHasher, StableHasherResult};
use std::mem;
use syntax::ast;
use syntax::attr;

impl<'a> HashStable<StableHashingContext<'a>> for DefId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.def_path_hash(*self).hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for DefId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> DefPathHash {
        hcx.def_path_hash(*self)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for LocalDefId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.def_path_hash(self.to_def_id()).hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for LocalDefId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> DefPathHash {
        hcx.def_path_hash(self.to_def_id())
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for CrateNum {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.def_path_hash(DefId {
            krate: *self,
            index: CRATE_DEF_INDEX
        }).hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for CrateNum {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> DefPathHash {
        let def_id = DefId { krate: *self, index: CRATE_DEF_INDEX };
        def_id.to_stable_hash_key(hcx)
    }
}

impl_stable_hash_for!(tuple_struct hir::ItemLocalId { index });

impl<'a> ToStableHashKey<StableHashingContext<'a>>
for hir::ItemLocalId {
    type KeyType = hir::ItemLocalId;

    #[inline]
    fn to_stable_hash_key(&self,
                          _: &StableHashingContext<'a>)
                          -> hir::ItemLocalId {
        *self
    }
}

// The following implementations of HashStable for ItemId, TraitItemId, and
// ImplItemId deserve special attention. Normally we do not hash NodeIds within
// the HIR, since they just signify a HIR nodes own path. But ItemId et al
// are used when another item in the HIR is *referenced* and we certainly
// want to pick up on a reference changing its target, so we hash the NodeIds
// in "DefPath Mode".

impl<'a> HashStable<StableHashingContext<'a>> for hir::ItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ItemId {
            id
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            id.hash_stable(hcx, hasher);
        })
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitItemId {
            node_id
        } = * self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            node_id.hash_stable(hcx, hasher);
        })
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::ImplItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ImplItemId {
            node_id
        } = * self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            node_id.hash_stable(hcx, hasher);
        })
    }
}

impl_stable_hash_for!(enum hir::ParamName {
    Plain(name),
    Fresh(index)
});

impl_stable_hash_for!(enum hir::LifetimeName {
    Param(param_name),
    Implicit,
    Underscore,
    Static,
});

impl_stable_hash_for!(struct hir::Label {
    span,
    name
});

impl_stable_hash_for!(struct hir::Lifetime {
    id,
    span,
    name
});

impl_stable_hash_for!(struct hir::Path {
    span,
    def,
    segments
});

impl_stable_hash_for!(struct hir::PathSegment {
    name,
    infer_types,
    args
});

impl_stable_hash_for!(enum hir::GenericArg {
    Lifetime(lt),
    Type(ty)
});

impl_stable_hash_for!(struct hir::GenericArgs {
    args,
    bindings,
    parenthesized
});

impl_stable_hash_for!(enum hir::GenericBound {
    Trait(poly_trait_ref, trait_bound_modifier),
    Outlives(lifetime)
});

impl_stable_hash_for!(enum hir::TraitBoundModifier {
    None,
    Maybe
});

impl_stable_hash_for!(struct hir::GenericParam {
    id,
    name,
    span,
    pure_wrt_drop,
    attrs,
    bounds,
    kind
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::GenericParamKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            hir::GenericParamKind::Lifetime { in_band } => {
                in_band.hash_stable(hcx, hasher);
            }
            hir::GenericParamKind::Type { ref default, synthetic } => {
                default.hash_stable(hcx, hasher);
                synthetic.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct hir::Generics {
    params,
    where_clause,
    span
});

impl_stable_hash_for!(enum hir::SyntheticTyParamKind {
    ImplTrait
});

impl_stable_hash_for!(struct hir::WhereClause {
    id,
    predicates
});

impl_stable_hash_for!(enum hir::WherePredicate {
    BoundPredicate(pred),
    RegionPredicate(pred),
    EqPredicate(pred)
});

impl_stable_hash_for!(struct hir::WhereBoundPredicate {
    span,
    bound_generic_params,
    bounded_ty,
    bounds
});

impl_stable_hash_for!(struct hir::WhereRegionPredicate {
    span,
    lifetime,
    bounds
});

impl_stable_hash_for!(struct hir::WhereEqPredicate {
    id,
    span,
    lhs_ty,
    rhs_ty
});

impl_stable_hash_for!(struct hir::MutTy {
    ty,
    mutbl
});

impl_stable_hash_for!(struct hir::MethodSig {
    header,
    decl
});

impl_stable_hash_for!(struct hir::TypeBinding {
    id,
    name,
    ty,
    span
});

impl_stable_hash_for!(struct hir::FnHeader {
    unsafety,
    constness,
    asyncness,
    abi
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Ty {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.while_hashing_hir_bodies(true, |hcx| {
            let hir::Ty {
                id: _,
                hir_id: _,
                ref node,
                ref span,
            } = *self;

            node.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        })
    }
}

impl_stable_hash_for!(enum hir::PrimTy {
    TyInt(int_ty),
    TyUint(uint_ty),
    TyFloat(float_ty),
    TyStr,
    TyBool,
    TyChar
});

impl_stable_hash_for!(struct hir::BareFnTy {
    unsafety,
    abi,
    generic_params,
    decl,
    arg_names
});

impl_stable_hash_for!(struct hir::ExistTy {
    generics,
    impl_trait_fn,
    bounds
});

impl_stable_hash_for!(enum hir::Ty_ {
    TySlice(t),
    TyArray(t, body_id),
    TyPtr(t),
    TyRptr(lifetime, t),
    TyBareFn(t),
    TyNever,
    TyTup(ts),
    TyPath(qpath),
    TyTraitObject(trait_refs, lifetime),
    TyImplTraitExistential(existty, def_id, lifetimes),
    TyTypeof(body_id),
    TyErr,
    TyInfer
});

impl_stable_hash_for!(struct hir::FnDecl {
    inputs,
    output,
    variadic,
    has_implicit_self
});

impl_stable_hash_for!(enum hir::FunctionRetTy {
    DefaultReturn(span),
    Return(t)
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitRef {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitRef {
            ref path,
            // Don't hash the ref_id. It is tracked via the thing it is used to access
            ref_id: _,
        } = *self;

        path.hash_stable(hcx, hasher);
    }
}


impl_stable_hash_for!(struct hir::PolyTraitRef {
    bound_generic_params,
    trait_ref,
    span
});

impl_stable_hash_for!(enum hir::QPath {
    Resolved(t, path),
    TypeRelative(t, path_segment)
});

impl_stable_hash_for!(struct hir::MacroDef {
    name,
    vis,
    attrs,
    id,
    span,
    legacy,
    body
});


impl<'a> HashStable<StableHashingContext<'a>> for hir::Block {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Block {
            ref stmts,
            ref expr,
            id: _,
            hir_id: _,
            rules,
            span,
            targeted_by_break,
            recovered,
        } = *self;

        stmts.hash_stable(hcx, hasher);
        expr.hash_stable(hcx, hasher);
        rules.hash_stable(hcx, hasher);
        span.hash_stable(hcx, hasher);
        recovered.hash_stable(hcx, hasher);
        targeted_by_break.hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::Pat {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Pat {
            id: _,
            hir_id: _,
            ref node,
            ref span
        } = *self;


        node.hash_stable(hcx, hasher);
        span.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for_spanned!(hir::FieldPat);

impl<'a> HashStable<StableHashingContext<'a>> for hir::FieldPat {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::FieldPat {
            id: _,
            ident,
            ref pat,
            is_shorthand,
        } = *self;

        ident.hash_stable(hcx, hasher);
        pat.hash_stable(hcx, hasher);
        is_shorthand.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(enum hir::BindingAnnotation {
    Unannotated,
    Mutable,
    Ref,
    RefMut
});

impl_stable_hash_for!(enum hir::RangeEnd {
    Included,
    Excluded
});

impl_stable_hash_for!(enum hir::PatKind {
    Wild,
    Binding(binding_mode, var, name, sub),
    Struct(path, field_pats, dotdot),
    TupleStruct(path, field_pats, dotdot),
    Path(path),
    Tuple(field_pats, dotdot),
    Box(sub),
    Ref(sub, mutability),
    Lit(expr),
    Range(start, end, end_kind),
    Slice(one, two, three)
});

impl_stable_hash_for!(enum hir::BinOp_ {
    BiAdd,
    BiSub,
    BiMul,
    BiDiv,
    BiRem,
    BiAnd,
    BiOr,
    BiBitXor,
    BiBitAnd,
    BiBitOr,
    BiShl,
    BiShr,
    BiEq,
    BiLt,
    BiLe,
    BiNe,
    BiGe,
    BiGt
});

impl_stable_hash_for_spanned!(hir::BinOp_);

impl_stable_hash_for!(enum hir::UnOp {
    UnDeref,
    UnNot,
    UnNeg
});

impl_stable_hash_for_spanned!(hir::Stmt_);

impl_stable_hash_for!(struct hir::Local {
    pat,
    ty,
    init,
    id,
    hir_id,
    span,
    attrs,
    source
});

impl_stable_hash_for_spanned!(hir::Decl_);
impl_stable_hash_for!(enum hir::Decl_ {
    DeclLocal(local),
    DeclItem(item_id)
});

impl_stable_hash_for!(struct hir::Arm {
    attrs,
    pats,
    guard,
    body
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Field {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Field {
            id: _,
            ident,
            ref expr,
            span,
            is_shorthand,
        } = *self;

        ident.hash_stable(hcx, hasher);
        expr.hash_stable(hcx, hasher);
        span.hash_stable(hcx, hasher);
        is_shorthand.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for_spanned!(ast::Name);


impl_stable_hash_for!(enum hir::BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(src),
    PushUnsafeBlock(src),
    PopUnsafeBlock(src)
});

impl_stable_hash_for!(enum hir::UnsafeSource {
    CompilerGenerated,
    UserProvided
});

impl_stable_hash_for!(struct hir::AnonConst {
    id,
    hir_id,
    body
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Expr {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.while_hashing_hir_bodies(true, |hcx| {
            let hir::Expr {
                id: _,
                hir_id: _,
                ref span,
                ref node,
                ref attrs
            } = *self;

            span.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
        })
    }
}

impl_stable_hash_for!(enum hir::Expr_ {
    ExprBox(sub),
    ExprArray(subs),
    ExprCall(callee, args),
    ExprMethodCall(segment, span, args),
    ExprTup(fields),
    ExprBinary(op, lhs, rhs),
    ExprUnary(op, operand),
    ExprLit(value),
    ExprCast(expr, t),
    ExprType(expr, t),
    ExprIf(cond, then, els),
    ExprWhile(cond, body, label),
    ExprLoop(body, label, loop_src),
    ExprMatch(matchee, arms, match_src),
    ExprClosure(capture_clause, decl, body_id, span, gen),
    ExprBlock(blk, label),
    ExprAssign(lhs, rhs),
    ExprAssignOp(op, lhs, rhs),
    ExprField(owner, ident),
    ExprIndex(lhs, rhs),
    ExprPath(path),
    ExprAddrOf(mutability, sub),
    ExprBreak(destination, sub),
    ExprContinue(destination),
    ExprRet(val),
    ExprInlineAsm(asm, inputs, outputs),
    ExprStruct(path, fields, base),
    ExprRepeat(val, times),
    ExprYield(val)
});

impl_stable_hash_for!(enum hir::LocalSource {
    Normal,
    ForLoopDesugar
});

impl_stable_hash_for!(enum hir::LoopSource {
    Loop,
    WhileLet,
    ForLoop
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::MatchSource {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use hir::MatchSource;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            MatchSource::Normal |
            MatchSource::WhileLetDesugar |
            MatchSource::ForLoopDesugar |
            MatchSource::TryDesugar => {
                // No fields to hash.
            }
            MatchSource::IfLetDesugar { contains_else_clause } => {
                contains_else_clause.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(enum hir::GeneratorMovability {
    Static,
    Movable
});

impl_stable_hash_for!(enum hir::CaptureClause {
    CaptureByValue,
    CaptureByRef
});

impl_stable_hash_for_spanned!(usize);

impl_stable_hash_for!(struct hir::Destination {
    label,
    target_id
});

impl_stable_hash_for_spanned!(ast::Ident);

impl_stable_hash_for!(enum hir::LoopIdError {
    OutsideLoopScope,
    UnlabeledCfInWhileCondition,
    UnresolvedLabel
});

impl<'a> HashStable<StableHashingContext<'a>> for ast::Ident {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ast::Ident {
            name,
            span,
        } = *self;

        name.hash_stable(hcx, hasher);
        span.hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitItem {
            id: _,
            hir_id: _,
            name,
            ref attrs,
            ref generics,
            ref node,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            name.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            generics.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl_stable_hash_for!(enum hir::TraitMethod {
    Required(name),
    Provided(body)
});

impl_stable_hash_for!(enum hir::TraitItemKind {
    Const(t, body),
    Method(sig, method),
    Type(bounds, rhs)
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::ImplItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ImplItem {
            id: _,
            hir_id: _,
            name,
            ref vis,
            defaultness,
            ref attrs,
            ref generics,
            ref node,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            name.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            defaultness.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            generics.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl_stable_hash_for!(enum hir::ImplItemKind {
    Const(t, body),
    Method(sig, body),
    Type(t)
});

impl_stable_hash_for!(enum ::syntax::ast::CrateSugar {
    JustCrate,
    PubCrate,
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Visibility {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            hir::Visibility::Public |
            hir::Visibility::Inherited => {
                // No fields to hash.
            }
            hir::Visibility::Crate(sugar) => {
                sugar.hash_stable(hcx, hasher);
            }
            hir::Visibility::Restricted { ref path, id } => {
                hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                    id.hash_stable(hcx, hasher);
                });
                path.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::Defaultness {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            hir::Defaultness::Final => {
                // No fields to hash.
            }
            hir::Defaultness::Default { has_value } => {
                has_value.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(enum hir::ImplPolarity {
    Positive,
    Negative
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Mod {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Mod {
            inner,
            // We are not hashing the IDs of the items contained in the module.
            // This is harmless and matches the current behavior but it's not
            // actually correct. See issue #40876.
            item_ids: _,
        } = *self;

        inner.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct hir::ForeignMod {
    abi,
    items
});

impl_stable_hash_for!(struct hir::EnumDef {
    variants
});

impl_stable_hash_for!(struct hir::Variant_ {
    name,
    attrs,
    data,
    disr_expr
});

impl_stable_hash_for_spanned!(hir::Variant_);

impl_stable_hash_for!(enum hir::UseKind {
    Single,
    Glob,
    ListStem
});

impl_stable_hash_for!(struct hir::StructField {
    span,
    ident,
    vis,
    id,
    ty,
    attrs
});

impl_stable_hash_for!(enum hir::VariantData {
    Struct(fields, id),
    Tuple(fields, id),
    Unit(id)
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Item {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Item {
            name,
            ref attrs,
            id: _,
            hir_id: _,
            ref node,
            ref vis,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            name.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl_stable_hash_for!(enum hir::Item_ {
    ItemExternCrate(orig_name),
    ItemUse(path, use_kind),
    ItemStatic(ty, mutability, body_id),
    ItemConst(ty, body_id),
    ItemFn(fn_decl, header, generics, body_id),
    ItemMod(module),
    ItemForeignMod(foreign_mod),
    ItemGlobalAsm(global_asm),
    ItemTy(ty, generics),
    ItemExistential(exist),
    ItemEnum(enum_def, generics),
    ItemStruct(variant_data, generics),
    ItemUnion(variant_data, generics),
    ItemTrait(is_auto, unsafety, generics, bounds, item_refs),
    ItemTraitAlias(generics, bounds),
    ItemImpl(unsafety, impl_polarity, impl_defaultness, generics, trait_ref, ty, impl_item_refs)
});

impl_stable_hash_for!(struct hir::TraitItemRef {
    id,
    name,
    kind,
    span,
    defaultness
});

impl_stable_hash_for!(struct hir::ImplItemRef {
    id,
    name,
    kind,
    span,
    vis,
    defaultness
});

impl<'a> HashStable<StableHashingContext<'a>>
for hir::AssociatedItemKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            hir::AssociatedItemKind::Const |
            hir::AssociatedItemKind::Type => {
                // No fields to hash.
            }
            hir::AssociatedItemKind::Method { has_self } => {
                has_self.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct hir::ForeignItem {
    name,
    attrs,
    node,
    id,
    span,
    vis
});

impl_stable_hash_for!(enum hir::ForeignItem_ {
    ForeignItemFn(fn_decl, arg_names, generics),
    ForeignItemStatic(ty, is_mutbl),
    ForeignItemType
});

impl_stable_hash_for!(enum hir::Stmt_ {
    StmtDecl(decl, id),
    StmtExpr(expr, id),
    StmtSemi(expr, id)
});

impl_stable_hash_for!(struct hir::Arg {
    pat,
    id,
    hir_id
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Body {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Body {
            ref arguments,
            ref value,
            is_generator,
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::Ignore, |hcx| {
            arguments.hash_stable(hcx, hasher);
            value.hash_stable(hcx, hasher);
            is_generator.hash_stable(hcx, hasher);
        });
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for hir::BodyId {
    type KeyType = (DefPathHash, hir::ItemLocalId);

    #[inline]
    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'a>)
                          -> (DefPathHash, hir::ItemLocalId) {
        let hir::BodyId { node_id } = *self;
        node_id.to_stable_hash_key(hcx)
    }
}

impl_stable_hash_for!(struct hir::InlineAsmOutput {
    constraint,
    is_rw,
    is_indirect
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::GlobalAsm {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::GlobalAsm {
            asm,
            ctxt: _
        } = *self;

        asm.hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::InlineAsm {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::InlineAsm {
            asm,
            asm_str_style,
            ref outputs,
            ref inputs,
            ref clobbers,
            volatile,
            alignstack,
            dialect,
            ctxt: _, // This is used for error reporting
        } = *self;

        asm.hash_stable(hcx, hasher);
        asm_str_style.hash_stable(hcx, hasher);
        outputs.hash_stable(hcx, hasher);
        inputs.hash_stable(hcx, hasher);
        clobbers.hash_stable(hcx, hasher);
        volatile.hash_stable(hcx, hasher);
        alignstack.hash_stable(hcx, hasher);
        dialect.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(enum hir::def::CtorKind {
    Fn,
    Const,
    Fictive
});

impl_stable_hash_for!(enum hir::def::Def {
    Mod(def_id),
    Struct(def_id),
    Union(def_id),
    Enum(def_id),
    Existential(def_id),
    Variant(def_id),
    Trait(def_id),
    TyAlias(def_id),
    TraitAlias(def_id),
    AssociatedTy(def_id),
    PrimTy(prim_ty),
    TyParam(def_id),
    SelfTy(trait_def_id, impl_def_id),
    TyForeign(def_id),
    Fn(def_id),
    Const(def_id),
    Static(def_id, is_mutbl),
    StructCtor(def_id, ctor_kind),
    VariantCtor(def_id, ctor_kind),
    Method(def_id),
    AssociatedConst(def_id),
    Local(def_id),
    Upvar(def_id, index, expr_id),
    Label(node_id),
    Macro(def_id, macro_kind),
    GlobalAsm(def_id),
    Err
});

impl_stable_hash_for!(enum hir::Mutability {
    MutMutable,
    MutImmutable
});

impl_stable_hash_for!(enum hir::IsAuto {
    Yes,
    No
});

impl_stable_hash_for!(enum hir::Unsafety {
    Unsafe,
    Normal
});

impl_stable_hash_for!(enum hir::IsAsync {
    Async,
    NotAsync
});

impl_stable_hash_for!(enum hir::Constness {
    Const,
    NotConst
});

impl<'a> HashStable<StableHashingContext<'a>>
for hir::def_id::DefIndex {

    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.local_def_path_hash(*self).hash_stable(hcx, hasher);
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>>
for hir::def_id::DefIndex {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> DefPathHash {
         hcx.local_def_path_hash(*self)
    }
}

impl_stable_hash_for!(struct hir::def::Export {
    ident,
    def,
    vis,
    span
});

impl<'a> HashStable<StableHashingContext<'a>>
for ::middle::lang_items::LangItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl_stable_hash_for!(struct ::middle::lang_items::LanguageItems {
    items,
    missing
});

impl<'a> HashStable<StableHashingContext<'a>>
for hir::TraitCandidate {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            let hir::TraitCandidate {
                def_id,
                import_id,
            } = *self;

            def_id.hash_stable(hcx, hasher);
            import_id.hash_stable(hcx, hasher);
        });
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for hir::TraitCandidate {
    type KeyType = (DefPathHash, Option<(DefPathHash, hir::ItemLocalId)>);

    fn to_stable_hash_key(&self,
                          hcx: &StableHashingContext<'a>)
                          -> Self::KeyType {
        let hir::TraitCandidate {
            def_id,
            import_id,
        } = *self;

        let import_id = import_id.map(|node_id| hcx.node_to_hir_id(node_id))
                                 .map(|hir_id| (hcx.local_def_path_hash(hir_id.owner),
                                                hir_id.local_id));
        (hcx.def_path_hash(def_id), import_id)
    }
}

impl<'hir> HashStable<StableHashingContext<'hir>> for hir::CodegenFnAttrs
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'hir>,
                                          hasher: &mut StableHasher<W>) {
        let hir::CodegenFnAttrs {
            flags,
            inline,
            export_name,
            ref target_features,
            linkage,
        } = *self;

        flags.hash_stable(hcx, hasher);
        inline.hash_stable(hcx, hasher);
        export_name.hash_stable(hcx, hasher);
        target_features.hash_stable(hcx, hasher);
        linkage.hash_stable(hcx, hasher);
    }
}

impl<'hir> HashStable<StableHashingContext<'hir>> for hir::CodegenFnAttrFlags
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'hir>,
                                          hasher: &mut StableHasher<W>) {
        self.bits().hash_stable(hcx, hasher);
    }
}

impl<'hir> HashStable<StableHashingContext<'hir>> for attr::InlineAttr {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'hir>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct hir::Freevar {
    def,
    span
});
