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
use hir::def_id::{DefId, CrateNum, CRATE_DEF_INDEX};
use ich::{StableHashingContext, NodeIdHashingMode};
use std::mem;

use syntax::ast;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for DefId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        hcx.def_path_hash(*self).hash_stable(hcx, hasher);
    }
}


impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::HirId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::HirId {
            owner,
            local_id,
        } = *self;

        hcx.def_path_hash(DefId::local(owner)).hash_stable(hcx, hasher);
        local_id.hash_stable(hcx, hasher);
    }
}


impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for CrateNum {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        hcx.def_path_hash(DefId {
            krate: *self,
            index: CRATE_DEF_INDEX
        }).hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(tuple_struct hir::ItemLocalId { index });

// The following implementations of HashStable for ItemId, TraitItemId, and
// ImplItemId deserve special attention. Normally we do not hash NodeIds within
// the HIR, since they just signify a HIR nodes own path. But ItemId et al
// are used when another item in the HIR is *referenced* and we certainly
// want to pick up on a reference changing its target, so we hash the NodeIds
// in "DefPath Mode".

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::ItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ItemId {
            id
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            id.hash_stable(hcx, hasher);
        })
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::TraitItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitItemId {
            node_id
        } = * self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            node_id.hash_stable(hcx, hasher);
        })
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::ImplItemId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ImplItemId {
            node_id
        } = * self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            node_id.hash_stable(hcx, hasher);
        })
    }
}

impl_stable_hash_for!(struct hir::Lifetime {
    id,
    span,
    name
});

impl_stable_hash_for!(struct hir::LifetimeDef {
    lifetime,
    bounds,
    pure_wrt_drop
});

impl_stable_hash_for!(struct hir::Path {
    span,
    def,
    segments
});

impl_stable_hash_for!(struct hir::PathSegment {
    name,
    parameters
});

impl_stable_hash_for!(enum hir::PathParameters {
    AngleBracketedParameters(data),
    ParenthesizedParameters(data)
});

impl_stable_hash_for!(struct hir::AngleBracketedParameterData {
    lifetimes,
    types,
    infer_types,
    bindings
});

impl_stable_hash_for!(struct hir::ParenthesizedParameterData {
    span,
    inputs,
    output
});

impl_stable_hash_for!(enum hir::TyParamBound {
    TraitTyParamBound(poly_trait_ref, trait_bound_modifier),
    RegionTyParamBound(lifetime)
});

impl_stable_hash_for!(enum hir::TraitBoundModifier {
    None,
    Maybe
});

impl_stable_hash_for!(struct hir::TyParam {
    name,
    id,
    bounds,
    default,
    span,
    pure_wrt_drop
});

impl_stable_hash_for!(struct hir::Generics {
    lifetimes,
    ty_params,
    where_clause,
    span
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
    bound_lifetimes,
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
    unsafety,
    constness,
    abi,
    decl,
    generics
});

impl_stable_hash_for!(struct hir::TypeBinding {
    id,
    name,
    ty,
    span
});

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::Ty {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let node_id_hashing_mode = match self.node {
            hir::TySlice(..)       |
            hir::TyArray(..)       |
            hir::TyPtr(..)         |
            hir::TyRptr(..)        |
            hir::TyBareFn(..)      |
            hir::TyNever           |
            hir::TyTup(..)         |
            hir::TyTraitObject(..) |
            hir::TyImplTrait(..)   |
            hir::TyTypeof(..)      |
            hir::TyErr             |
            hir::TyInfer           => {
                NodeIdHashingMode::Ignore
            }
            hir::TyPath(..) => {
                NodeIdHashingMode::HashTraitsInScope
            }
        };

        hcx.while_hashing_hir_bodies(true, |hcx| {
            let hir::Ty {
                id,
                ref node,
                ref span,
            } = *self;

            hcx.with_node_id_hashing_mode(node_id_hashing_mode, |hcx| {
                id.hash_stable(hcx, hasher);
            });
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
    lifetimes,
    decl
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
    TyImplTrait(bounds),
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::TraitRef {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitRef {
            ref path,
            ref_id,
        } = *self;

        path.hash_stable(hcx, hasher);
        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashTraitsInScope, |hcx| {
            ref_id.hash_stable(hcx, hasher);
        });
    }
}


impl_stable_hash_for!(struct hir::PolyTraitRef {
    bound_lifetimes,
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


impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::Block {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::Block {
            ref stmts,
            ref expr,
            id,
            hir_id: _,
            rules,
            span,
            targeted_by_break,
        } = *self;

        let non_item_stmts = || stmts.iter().filter(|stmt| {
            match stmt.node {
                hir::StmtDecl(ref decl, _) => {
                    match decl.node {
                        // If this is a declaration of a nested item, we don't
                        // want to leave any trace of it in the hash value, not
                        // even that it exists. Otherwise changing the position
                        // of nested items would invalidate the containing item
                        // even though that does not constitute a semantic
                        // change.
                        hir::DeclItem(_) => false,
                        hir::DeclLocal(_) => true
                    }
                }
                hir::StmtExpr(..) |
                hir::StmtSemi(..) => true
            }
        });

        let count = non_item_stmts().count();

        count.hash_stable(hcx, hasher);

        for stmt in non_item_stmts() {
            stmt.hash_stable(hcx, hasher);
        }

        expr.hash_stable(hcx, hasher);
        id.hash_stable(hcx, hasher);
        rules.hash_stable(hcx, hasher);
        span.hash_stable(hcx, hasher);
        targeted_by_break.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::Pat {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let node_id_hashing_mode = match self.node {
            hir::PatKind::Wild        |
            hir::PatKind::Binding(..) |
            hir::PatKind::Tuple(..)   |
            hir::PatKind::Box(..)     |
            hir::PatKind::Ref(..)     |
            hir::PatKind::Lit(..)     |
            hir::PatKind::Range(..)   |
            hir::PatKind::Slice(..)   => {
                NodeIdHashingMode::Ignore
            }
            hir::PatKind::Path(..)        |
            hir::PatKind::Struct(..)      |
            hir::PatKind::TupleStruct(..) => {
                NodeIdHashingMode::HashTraitsInScope
            }
        };

        let hir::Pat {
            id,
            hir_id: _,
            ref node,
            ref span
        } = *self;

        hcx.with_node_id_hashing_mode(node_id_hashing_mode, |hcx| {
            id.hash_stable(hcx, hasher);
        });
        node.hash_stable(hcx, hasher);
        span.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for_spanned!(hir::FieldPat);
impl_stable_hash_for!(struct hir::FieldPat {
    name,
    pat,
    is_shorthand
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

impl_stable_hash_for!(struct hir::Field {
    name,
    expr,
    span,
    is_shorthand
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::Expr {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        hcx.while_hashing_hir_bodies(true, |hcx| {
            let hir::Expr {
                id,
                hir_id: _,
                ref span,
                ref node,
                ref attrs
            } = *self;

            let (spans_always_on, node_id_hashing_mode) = match *node {
                hir::ExprBox(..)        |
                hir::ExprArray(..)      |
                hir::ExprCall(..)       |
                hir::ExprLit(..)        |
                hir::ExprCast(..)       |
                hir::ExprType(..)       |
                hir::ExprIf(..)         |
                hir::ExprWhile(..)      |
                hir::ExprLoop(..)       |
                hir::ExprMatch(..)      |
                hir::ExprClosure(..)    |
                hir::ExprBlock(..)      |
                hir::ExprAssign(..)     |
                hir::ExprTupField(..)   |
                hir::ExprAddrOf(..)     |
                hir::ExprBreak(..)      |
                hir::ExprAgain(..)      |
                hir::ExprRet(..)        |
                hir::ExprYield(..)    |
                hir::ExprInlineAsm(..)  |
                hir::ExprRepeat(..)     |
                hir::ExprTup(..)        => {
                    // For these we only hash the span when debuginfo is on.
                    (false, NodeIdHashingMode::Ignore)
                }
                // For the following, spans might be significant because of
                // panic messages indicating the source location.
                hir::ExprBinary(op, ..) => {
                    (hcx.binop_can_panic_at_runtime(op.node), NodeIdHashingMode::Ignore)
                }
                hir::ExprUnary(op, _) => {
                    (hcx.unop_can_panic_at_runtime(op), NodeIdHashingMode::Ignore)
                }
                hir::ExprAssignOp(op, ..) => {
                    (hcx.binop_can_panic_at_runtime(op.node), NodeIdHashingMode::Ignore)
                }
                hir::ExprIndex(..) => {
                    (true, NodeIdHashingMode::Ignore)
                }
                // For these we don't care about the span, but want to hash the
                // trait in scope
                hir::ExprMethodCall(..) |
                hir::ExprPath(..)       |
                hir::ExprStruct(..)     |
                hir::ExprField(..)      => {
                    (false, NodeIdHashingMode::HashTraitsInScope)
                }
            };

            hcx.with_node_id_hashing_mode(node_id_hashing_mode, |hcx| {
                id.hash_stable(hcx, hasher);
            });

            if spans_always_on {
                hcx.while_hashing_spans(true, |hcx| {
                    span.hash_stable(hcx, hasher);
                    node.hash_stable(hcx, hasher);
                    attrs.hash_stable(hcx, hasher);
                });
            } else {
                span.hash_stable(hcx, hasher);
                node.hash_stable(hcx, hasher);
                attrs.hash_stable(hcx, hasher);
            }
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
    ExprBlock(blk),
    ExprAssign(lhs, rhs),
    ExprAssignOp(op, lhs, rhs),
    ExprField(owner, field_name),
    ExprTupField(owner, idx),
    ExprIndex(lhs, rhs),
    ExprPath(path),
    ExprAddrOf(mutability, sub),
    ExprBreak(destination, sub),
    ExprAgain(destination),
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::MatchSource {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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

impl_stable_hash_for!(enum hir::CaptureClause {
    CaptureByValue,
    CaptureByRef
});

impl_stable_hash_for_spanned!(usize);

impl_stable_hash_for!(struct hir::Destination {
    ident,
    target_id
});

impl_stable_hash_for_spanned!(ast::Ident);

impl_stable_hash_for!(enum hir::LoopIdResult {
    Ok(node_id),
    Err(loop_id_error)
});

impl_stable_hash_for!(enum hir::LoopIdError {
    OutsideLoopScope,
    UnlabeledCfInWhileCondition,
    UnresolvedLabel
});

impl_stable_hash_for!(enum hir::ScopeTarget {
    Block(node_id),
    Loop(loop_id_result)
});

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for ast::Ident {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let ast::Ident {
            ref name,
            ctxt: _ // Ignore this
        } = *self;

        name.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::TraitItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::TraitItem {
            id,
            name,
            ref attrs,
            ref node,
            span
        } = *self;

        hcx.hash_hir_item_like(attrs, |hcx| {
            id.hash_stable(hcx, hasher);
            name.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::ImplItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::ImplItem {
            id,
            name,
            ref vis,
            defaultness,
            ref attrs,
            ref node,
            span
        } = *self;

        hcx.hash_hir_item_like(attrs, |hcx| {
            id.hash_stable(hcx, hasher);
            name.hash_stable(hcx, hasher);
            vis.hash_stable(hcx, hasher);
            defaultness.hash_stable(hcx, hasher);
            attrs.hash_stable(hcx, hasher);
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::Visibility {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            hir::Visibility::Public |
            hir::Visibility::Crate |
            hir::Visibility::Inherited => {
                // No fields to hash.
            }
            hir::Visibility::Restricted { ref path, id } => {
                hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashTraitsInScope, |hcx| {
                    id.hash_stable(hcx, hasher);
                });
                path.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::Defaultness {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::Mod {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
    name,
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::Item {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let (node_id_hashing_mode, hash_spans) = match self.node {
            hir::ItemStatic(..)      |
            hir::ItemConst(..)       |
            hir::ItemFn(..)          => {
                (NodeIdHashingMode::Ignore, hcx.hash_spans())
            }
            hir::ItemUse(..) => {
                (NodeIdHashingMode::HashTraitsInScope, false)
            }

            hir::ItemExternCrate(..) |
            hir::ItemForeignMod(..)  |
            hir::ItemGlobalAsm(..)   |
            hir::ItemMod(..)         |
            hir::ItemDefaultImpl(..) |
            hir::ItemTrait(..)       |
            hir::ItemImpl(..)        |
            hir::ItemTy(..)          |
            hir::ItemEnum(..)        |
            hir::ItemStruct(..)      |
            hir::ItemUnion(..)       => {
                (NodeIdHashingMode::Ignore, false)
            }
        };

        let hir::Item {
            name,
            ref attrs,
            id,
            ref node,
            ref vis,
            span
        } = *self;

        hcx.hash_hir_item_like(attrs, |hcx| {
            hcx.while_hashing_spans(hash_spans, |hcx| {
                hcx.with_node_id_hashing_mode(node_id_hashing_mode, |hcx| {
                    id.hash_stable(hcx, hasher);
                });
                name.hash_stable(hcx, hasher);
                attrs.hash_stable(hcx, hasher);
                node.hash_stable(hcx, hasher);
                vis.hash_stable(hcx, hasher);
                span.hash_stable(hcx, hasher);
            });
        });
    }
}

impl_stable_hash_for!(enum hir::Item_ {
    ItemExternCrate(name),
    ItemUse(path, use_kind),
    ItemStatic(ty, mutability, body_id),
    ItemConst(ty, body_id),
    ItemFn(fn_decl, unsafety, constness, abi, generics, body_id),
    ItemMod(module),
    ItemForeignMod(foreign_mod),
    ItemGlobalAsm(global_asm),
    ItemTy(ty, generics),
    ItemEnum(enum_def, generics),
    ItemStruct(variant_data, generics),
    ItemUnion(variant_data, generics),
    ItemTrait(unsafety, generics, bounds, item_refs),
    ItemDefaultImpl(unsafety, trait_ref),
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for hir::AssociatedItemKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
    ForeignItemStatic(ty, is_mutbl)
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

impl_stable_hash_for!(struct hir::Body {
    arguments,
    value,
    is_generator
});

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::BodyId {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        if hcx.hash_bodies() {
            hcx.tcx().hir.body(*self).hash_stable(hcx, hasher);
        }
    }
}

impl_stable_hash_for!(struct hir::InlineAsmOutput {
    constraint,
    is_rw,
    is_indirect
});

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::GlobalAsm {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let hir::GlobalAsm {
            asm,
            ctxt: _
        } = *self;

        asm.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for hir::InlineAsm {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
    Variant(def_id),
    Trait(def_id),
    TyAlias(def_id),
    AssociatedTy(def_id),
    PrimTy(prim_ty),
    TyParam(def_id),
    SelfTy(trait_def_id, impl_def_id),
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


impl_stable_hash_for!(enum hir::Unsafety {
    Unsafe,
    Normal
});


impl_stable_hash_for!(enum hir::Constness {
    Const,
    NotConst
});

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for hir::def_id::DefIndex {

    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        DefId::local(*self).hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct hir::def::Export {
    ident,
    def,
    span
});

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ::middle::lang_items::LangItem {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        ::std::hash::Hash::hash(self, hasher);
    }
}
