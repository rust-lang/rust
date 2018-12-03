//! This module contains `HashStable` implementations for various HIR data
//! types in no particular order.

use crate::hir;
use crate::hir::map::DefPathHash;
use crate::hir::def_id::{DefId, LocalDefId, CrateNum, CRATE_DEF_INDEX};
use crate::ich::{StableHashingContext, NodeIdHashingMode, Fingerprint};
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

impl<'a> HashStable<StableHashingContext<'a>> for hir::ItemLocalId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.as_u32().hash_stable(hcx, hasher);
    }
}

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
            hir_id
        } = * self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            hir_id.hash_stable(hcx, hasher);
        })
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for hir::ImplItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ImplItemId {
            hir_id
        } = * self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            hir_id.hash_stable(hcx, hasher);
        })
    }
}

impl_stable_hash_for!(enum hir::ParamName {
    Plain(name),
    Fresh(index),
    Error,
});

impl_stable_hash_for!(enum hir::LifetimeName {
    Param(param_name),
    Implicit,
    Underscore,
    Static,
    Error,
});

impl_stable_hash_for!(struct ast::Label {
    ident
});

impl_stable_hash_for!(struct hir::Path {
    span,
    def,
    segments
});

impl_stable_hash_for!(struct hir::PathSegment {
    ident -> (ident.name),
    id,
    hir_id,
    def,
    infer_types,
    args
});

impl_stable_hash_for!(struct hir::ConstArg {
    value,
    span,
});

impl_stable_hash_for!(enum hir::GenericArg {
    Lifetime(lt),
    Type(ty),
    Const(ct),
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
    hir_id,
    name,
    pure_wrt_drop,
    attrs,
    bounds,
    span,
    kind
});

impl_stable_hash_for!(enum hir::LifetimeParamKind {
    Explicit,
    InBand,
    Elided,
    Error,
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::GenericParamKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            hir::GenericParamKind::Lifetime { kind } => {
                kind.hash_stable(hcx, hasher);
            }
            hir::GenericParamKind::Type { ref default, synthetic } => {
                default.hash_stable(hcx, hasher);
                synthetic.hash_stable(hcx, hasher);
            }
            hir::GenericParamKind::Const { ref ty } => {
                ty.hash_stable(hcx, hasher);
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
    hir_id,
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
    hir_id,
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
    hir_id,
    ident -> (ident.name),
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
    Int(int_ty),
    Uint(uint_ty),
    Float(float_ty),
    Str,
    Bool,
    Char
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

impl_stable_hash_for!(enum hir::TyKind {
    Slice(t),
    Array(t, body_id),
    Ptr(t),
    Rptr(lifetime, t),
    BareFn(t),
    Never,
    Tup(ts),
    Path(qpath),
    Def(it, lt),
    TraitObject(trait_refs, lifetime),
    Typeof(body_id),
    Err,
    Infer,
    CVarArgs(lt),
});

impl_stable_hash_for!(struct hir::FnDecl {
    inputs,
    output,
    c_variadic,
    implicit_self
});

impl_stable_hash_for!(enum hir::FunctionRetTy {
    DefaultReturn(span),
    Return(t)
});

impl_stable_hash_for!(enum hir::ImplicitSelfKind {
    Imm,
    Mut,
    ImmRef,
    MutRef,
    None
});

impl_stable_hash_for!(struct hir::TraitRef {
    // Don't hash the hir_ref_id. It is tracked via the thing it is used to access
    hir_ref_id -> _,
    path,
});

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
    hir_id,
    span,
    legacy,
    body
});

impl_stable_hash_for!(struct hir::Block {
    stmts,
    expr,
    hir_id -> _,
    rules,
    span,
    targeted_by_break,
});

impl_stable_hash_for!(struct hir::Pat {
    hir_id -> _,
    node,
    span,
});

impl_stable_hash_for_spanned!(hir::FieldPat);

impl_stable_hash_for!(struct hir::FieldPat {
    hir_id -> _,
    ident -> (ident.name),
    pat,
    is_shorthand,
});

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
    Binding(binding_mode, var, hir_id, name, sub),
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

impl_stable_hash_for!(enum hir::BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt
});

impl_stable_hash_for_spanned!(hir::BinOpKind);

impl_stable_hash_for!(enum hir::UnOp {
    UnDeref,
    UnNot,
    UnNeg
});

impl_stable_hash_for!(struct hir::Stmt {
    hir_id,
    node,
    span,
});


impl_stable_hash_for!(struct hir::Local {
    pat,
    ty,
    init,
    hir_id,
    span,
    attrs,
    source
});

impl_stable_hash_for!(struct hir::Arm {
    attrs,
    pats,
    guard,
    body
});

impl_stable_hash_for!(enum hir::Guard {
    If(expr),
});

impl_stable_hash_for!(struct hir::Field {
    hir_id -> _,
    ident,
    expr,
    span,
    is_shorthand,
});

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
    hir_id,
    body
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Expr {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.while_hashing_hir_bodies(true, |hcx| {
            let hir::Expr {
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

impl_stable_hash_for!(enum hir::ExprKind {
    Box(sub),
    Array(subs),
    Call(callee, args),
    MethodCall(segment, span, args),
    Tup(fields),
    Binary(op, lhs, rhs),
    Unary(op, operand),
    Lit(value),
    Cast(expr, t),
    Type(expr, t),
    If(cond, then, els),
    While(cond, body, label),
    Loop(body, label, loop_src),
    Match(matchee, arms, match_src),
    Closure(capture_clause, decl, body_id, span, gen),
    Block(blk, label),
    Assign(lhs, rhs),
    AssignOp(op, lhs, rhs),
    Field(owner, ident),
    Index(lhs, rhs),
    Path(path),
    AddrOf(mutability, sub),
    Break(destination, sub),
    Continue(destination),
    Ret(val),
    InlineAsm(asm, inputs, outputs),
    Struct(path, fields, base),
    Repeat(val, times),
    Yield(val),
    Err
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
        use crate::hir::MatchSource;

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

impl_stable_hash_for!(struct ast::Ident {
    name,
    span,
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitItem {
            hir_id: _,
            ident,
            ref attrs,
            ref generics,
            ref node,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
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
            hir_id: _,
            ident,
            ref vis,
            defaultness,
            ref attrs,
            ref generics,
            ref node,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
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
    Existential(bounds),
    Type(t)
});

impl_stable_hash_for!(enum ::syntax::ast::CrateSugar {
    JustCrate,
    PubCrate,
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::VisibilityKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            hir::VisibilityKind::Public |
            hir::VisibilityKind::Inherited => {
                // No fields to hash.
            }
            hir::VisibilityKind::Crate(sugar) => {
                sugar.hash_stable(hcx, hasher);
            }
            hir::VisibilityKind::Restricted { ref path, hir_id } => {
                hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                    hir_id.hash_stable(hcx, hasher);
                });
                path.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for_spanned!(hir::VisibilityKind);

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
            inner: ref inner_span,
            ref item_ids,
        } = *self;

        inner_span.hash_stable(hcx, hasher);

        // Combining the DefPathHashes directly is faster than feeding them
        // into the hasher. Because we use a commutative combine, we also don't
        // have to sort the array.
        let item_ids_hash = item_ids
            .iter()
            .map(|id| {
                let (def_path_hash, local_id) = id.id.to_stable_hash_key(hcx);
                debug_assert_eq!(local_id, hir::ItemLocalId::from_u32(0));
                def_path_hash.0
            }).fold(Fingerprint::ZERO, |a, b| {
                a.combine_commutative(b)
            });

        item_ids.len().hash_stable(hcx, hasher);
        item_ids_hash.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct hir::ForeignMod {
    abi,
    items
});

impl_stable_hash_for!(struct hir::EnumDef {
    variants
});

impl_stable_hash_for!(struct hir::VariantKind {
    ident -> (ident.name),
    attrs,
    data,
    disr_expr
});

impl_stable_hash_for_spanned!(hir::VariantKind);

impl_stable_hash_for!(enum hir::UseKind {
    Single,
    Glob,
    ListStem
});

impl_stable_hash_for!(struct hir::StructField {
    span,
    ident -> (ident.name),
    vis,
    hir_id,
    ty,
    attrs
});

impl_stable_hash_for!(enum hir::VariantData {
    Struct(fields, hir_id),
    Tuple(fields, hir_id),
    Unit(hir_id)
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::Item {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Item {
            ident,
            ref attrs,
            hir_id: _,
            ref node,
            ref vis,
            span
        } = *self;

        hcx.hash_hir_item_like(|hcx| {
            ident.name.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            span.hash_stable(hcx, hasher);
        });
    }
}

impl_stable_hash_for!(enum hir::ItemKind {
    ExternCrate(orig_name),
    Use(path, use_kind),
    Static(ty, mutability, body_id),
    Const(ty, body_id),
    Fn(fn_decl, header, generics, body_id),
    Mod(module),
    ForeignMod(foreign_mod),
    GlobalAsm(global_asm),
    Ty(ty, generics),
    Existential(exist),
    Enum(enum_def, generics),
    Struct(variant_data, generics),
    Union(variant_data, generics),
    Trait(is_auto, unsafety, generics, bounds, item_refs),
    TraitAlias(generics, bounds),
    Impl(unsafety, impl_polarity, impl_defaultness, generics, trait_ref, ty, impl_item_refs)
});

impl_stable_hash_for!(struct hir::TraitItemRef {
    id,
    ident -> (ident.name),
    kind,
    span,
    defaultness
});

impl_stable_hash_for!(struct hir::ImplItemRef {
    id,
    ident -> (ident.name),
    kind,
    span,
    vis,
    defaultness
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::AssociatedItemKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            hir::AssociatedItemKind::Const |
            hir::AssociatedItemKind::Existential |
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
    ident -> (ident.name),
    attrs,
    node,
    hir_id,
    span,
    vis
});

impl_stable_hash_for!(enum hir::ForeignItemKind {
    Fn(fn_decl, arg_names, generics),
    Static(ty, is_mutbl),
    Type
});

impl_stable_hash_for!(enum hir::StmtKind {
    Local(local),
    Item(item_id),
    Expr(expr),
    Semi(expr)
});

impl_stable_hash_for!(struct hir::Arg {
    pat,
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
        let hir::BodyId { hir_id } = *self;
        hir_id.to_stable_hash_key(hcx)
    }
}

impl_stable_hash_for!(struct hir::InlineAsmOutput {
    constraint,
    is_rw,
    is_indirect,
    span
});

impl_stable_hash_for!(struct hir::GlobalAsm {
    asm,
    ctxt -> _, // This is used for error reporting
});

impl_stable_hash_for!(struct hir::InlineAsm {
    asm,
    asm_str_style,
    outputs,
    inputs,
    clobbers,
    volatile,
    alignstack,
    dialect,
    ctxt -> _, // This is used for error reporting
});

impl_stable_hash_for!(enum hir::def::CtorKind {
    Fn,
    Const,
    Fictive
});

impl_stable_hash_for!(enum hir::def::NonMacroAttrKind {
    Builtin,
    Tool,
    DeriveHelper,
    LegacyPluginHelper,
    Custom,
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
    AssociatedExistential(def_id),
    PrimTy(prim_ty),
    TyParam(def_id),
    ConstParam(def_id),
    SelfTy(trait_def_id, impl_def_id),
    ForeignTy(def_id),
    Fn(def_id),
    Const(def_id),
    Static(def_id, is_mutbl),
    StructCtor(def_id, ctor_kind),
    SelfCtor(impl_def_id),
    VariantCtor(def_id, ctor_kind),
    Method(def_id),
    AssociatedConst(def_id),
    Local(def_id),
    Upvar(def_id, index, expr_id),
    Label(node_id),
    Macro(def_id, macro_kind),
    ToolMod,
    NonMacroAttr(attr_kind),
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

impl<'a> HashStable<StableHashingContext<'a>> for hir::def_id::DefIndex {

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

impl_stable_hash_for!(struct crate::middle::lib_features::LibFeatures {
    stable,
    unstable
});

impl<'a> HashStable<StableHashingContext<'a>> for crate::middle::lang_items::LangItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl_stable_hash_for!(struct crate::middle::lang_items::LanguageItems {
    items,
    missing
});

impl<'a> HashStable<StableHashingContext<'a>> for hir::TraitCandidate {
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

impl_stable_hash_for!(struct hir::CodegenFnAttrs {
    flags,
    inline,
    optimize,
    export_name,
    link_name,
    target_features,
    linkage,
    link_section,
});

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

impl<'hir> HashStable<StableHashingContext<'hir>> for attr::OptimizeAttr {
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
