//! Utilities for manipulating and extracting information from `rustc_ast::ast`.
//!
//! - The `eq_foobar` functions test for semantic equality but ignores `NodeId`s and `Span`s.

#![allow(clippy::similar_names, clippy::wildcard_imports, clippy::enum_glob_use)]

use crate::{both, over};
use if_chain::if_chain;
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, *};
use rustc_span::symbol::Ident;
use std::mem;

pub mod ident_iter;
pub use ident_iter::IdentIter;

pub fn is_useless_with_eq_exprs(kind: BinOpKind) -> bool {
    use BinOpKind::*;
    matches!(
        kind,
        Sub | Div | Eq | Lt | Le | Gt | Ge | Ne | And | Or | BitXor | BitAnd | BitOr
    )
}

/// Checks if each element in the first slice is contained within the latter as per `eq_fn`.
pub fn unordered_over<X>(left: &[X], right: &[X], mut eq_fn: impl FnMut(&X, &X) -> bool) -> bool {
    left.len() == right.len() && left.iter().all(|l| right.iter().any(|r| eq_fn(l, r)))
}

pub fn eq_id(l: Ident, r: Ident) -> bool {
    l.name == r.name
}

pub fn eq_pat(l: &Pat, r: &Pat) -> bool {
    use PatKind::*;
    match (&l.kind, &r.kind) {
        (Paren(l), _) => eq_pat(l, r),
        (_, Paren(r)) => eq_pat(l, r),
        (Wild, Wild) | (Rest, Rest) => true,
        (Lit(l), Lit(r)) => eq_expr(l, r),
        (Ident(b1, i1, s1), Ident(b2, i2, s2)) => b1 == b2 && eq_id(*i1, *i2) && both(s1, s2, |l, r| eq_pat(l, r)),
        (Range(lf, lt, le), Range(rf, rt, re)) => {
            eq_expr_opt(lf, rf) && eq_expr_opt(lt, rt) && eq_range_end(&le.node, &re.node)
        },
        (Box(l), Box(r))
        | (Ref(l, Mutability::Not), Ref(r, Mutability::Not))
        | (Ref(l, Mutability::Mut), Ref(r, Mutability::Mut)) => eq_pat(l, r),
        (Tuple(l), Tuple(r)) | (Slice(l), Slice(r)) => over(l, r, |l, r| eq_pat(l, r)),
        (Path(lq, lp), Path(rq, rp)) => both(lq, rq, eq_qself) && eq_path(lp, rp),
        (TupleStruct(lqself, lp, lfs), TupleStruct(rqself, rp, rfs)) => {
            eq_maybe_qself(lqself, rqself) && eq_path(lp, rp) && over(lfs, rfs, |l, r| eq_pat(l, r))
        },
        (Struct(lqself, lp, lfs, lr), Struct(rqself, rp, rfs, rr)) => {
            lr == rr && eq_maybe_qself(lqself, rqself) && eq_path(lp, rp) && unordered_over(lfs, rfs, eq_field_pat)
        },
        (Or(ls), Or(rs)) => unordered_over(ls, rs, |l, r| eq_pat(l, r)),
        (MacCall(l), MacCall(r)) => eq_mac_call(l, r),
        _ => false,
    }
}

pub fn eq_range_end(l: &RangeEnd, r: &RangeEnd) -> bool {
    match (l, r) {
        (RangeEnd::Excluded, RangeEnd::Excluded) => true,
        (RangeEnd::Included(l), RangeEnd::Included(r)) => {
            matches!(l, RangeSyntax::DotDotEq) == matches!(r, RangeSyntax::DotDotEq)
        },
        _ => false,
    }
}

pub fn eq_field_pat(l: &PatField, r: &PatField) -> bool {
    l.is_placeholder == r.is_placeholder
        && eq_id(l.ident, r.ident)
        && eq_pat(&l.pat, &r.pat)
        && over(&l.attrs, &r.attrs, eq_attr)
}

pub fn eq_qself(l: &QSelf, r: &QSelf) -> bool {
    l.position == r.position && eq_ty(&l.ty, &r.ty)
}

pub fn eq_maybe_qself(l: &Option<QSelf>, r: &Option<QSelf>) -> bool {
    match (l, r) {
        (Some(l), Some(r)) => eq_qself(l, r),
        (None, None) => true,
        _ => false,
    }
}

pub fn eq_path(l: &Path, r: &Path) -> bool {
    over(&l.segments, &r.segments, eq_path_seg)
}

pub fn eq_path_seg(l: &PathSegment, r: &PathSegment) -> bool {
    eq_id(l.ident, r.ident) && both(&l.args, &r.args, |l, r| eq_generic_args(l, r))
}

pub fn eq_generic_args(l: &GenericArgs, r: &GenericArgs) -> bool {
    match (l, r) {
        (GenericArgs::AngleBracketed(l), GenericArgs::AngleBracketed(r)) => over(&l.args, &r.args, eq_angle_arg),
        (GenericArgs::Parenthesized(l), GenericArgs::Parenthesized(r)) => {
            over(&l.inputs, &r.inputs, |l, r| eq_ty(l, r)) && eq_fn_ret_ty(&l.output, &r.output)
        },
        _ => false,
    }
}

pub fn eq_angle_arg(l: &AngleBracketedArg, r: &AngleBracketedArg) -> bool {
    match (l, r) {
        (AngleBracketedArg::Arg(l), AngleBracketedArg::Arg(r)) => eq_generic_arg(l, r),
        (AngleBracketedArg::Constraint(l), AngleBracketedArg::Constraint(r)) => eq_assoc_constraint(l, r),
        _ => false,
    }
}

pub fn eq_generic_arg(l: &GenericArg, r: &GenericArg) -> bool {
    match (l, r) {
        (GenericArg::Lifetime(l), GenericArg::Lifetime(r)) => eq_id(l.ident, r.ident),
        (GenericArg::Type(l), GenericArg::Type(r)) => eq_ty(l, r),
        (GenericArg::Const(l), GenericArg::Const(r)) => eq_expr(&l.value, &r.value),
        _ => false,
    }
}

pub fn eq_expr_opt(l: &Option<P<Expr>>, r: &Option<P<Expr>>) -> bool {
    both(l, r, |l, r| eq_expr(l, r))
}

pub fn eq_struct_rest(l: &StructRest, r: &StructRest) -> bool {
    match (l, r) {
        (StructRest::Base(lb), StructRest::Base(rb)) => eq_expr(lb, rb),
        (StructRest::Rest(_), StructRest::Rest(_)) | (StructRest::None, StructRest::None) => true,
        _ => false,
    }
}

pub fn eq_expr(l: &Expr, r: &Expr) -> bool {
    use ExprKind::*;
    if !over(&l.attrs, &r.attrs, eq_attr) {
        return false;
    }
    match (&l.kind, &r.kind) {
        (Paren(l), _) => eq_expr(l, r),
        (_, Paren(r)) => eq_expr(l, r),
        (Err, Err) => true,
        (Box(l), Box(r)) | (Try(l), Try(r)) | (Await(l), Await(r)) => eq_expr(l, r),
        (Array(l), Array(r)) | (Tup(l), Tup(r)) => over(l, r, |l, r| eq_expr(l, r)),
        (Repeat(le, ls), Repeat(re, rs)) => eq_expr(le, re) && eq_expr(&ls.value, &rs.value),
        (Call(lc, la), Call(rc, ra)) => eq_expr(lc, rc) && over(la, ra, |l, r| eq_expr(l, r)),
        (MethodCall(lc, la, _), MethodCall(rc, ra, _)) => eq_path_seg(lc, rc) && over(la, ra, |l, r| eq_expr(l, r)),
        (Binary(lo, ll, lr), Binary(ro, rl, rr)) => lo.node == ro.node && eq_expr(ll, rl) && eq_expr(lr, rr),
        (Unary(lo, l), Unary(ro, r)) => mem::discriminant(lo) == mem::discriminant(ro) && eq_expr(l, r),
        (Lit(l), Lit(r)) => l.kind == r.kind,
        (Cast(l, lt), Cast(r, rt)) | (Type(l, lt), Type(r, rt)) => eq_expr(l, r) && eq_ty(lt, rt),
        (Let(lp, le, _), Let(rp, re, _)) => eq_pat(lp, rp) && eq_expr(le, re),
        (If(lc, lt, le), If(rc, rt, re)) => eq_expr(lc, rc) && eq_block(lt, rt) && eq_expr_opt(le, re),
        (While(lc, lt, ll), While(rc, rt, rl)) => eq_label(ll, rl) && eq_expr(lc, rc) && eq_block(lt, rt),
        (ForLoop(lp, li, lt, ll), ForLoop(rp, ri, rt, rl)) => {
            eq_label(ll, rl) && eq_pat(lp, rp) && eq_expr(li, ri) && eq_block(lt, rt)
        },
        (Loop(lt, ll), Loop(rt, rl)) => eq_label(ll, rl) && eq_block(lt, rt),
        (Block(lb, ll), Block(rb, rl)) => eq_label(ll, rl) && eq_block(lb, rb),
        (TryBlock(l), TryBlock(r)) => eq_block(l, r),
        (Yield(l), Yield(r)) | (Ret(l), Ret(r)) => eq_expr_opt(l, r),
        (Break(ll, le), Break(rl, re)) => eq_label(ll, rl) && eq_expr_opt(le, re),
        (Continue(ll), Continue(rl)) => eq_label(ll, rl),
        (Assign(l1, l2, _), Assign(r1, r2, _)) | (Index(l1, l2), Index(r1, r2)) => eq_expr(l1, r1) && eq_expr(l2, r2),
        (AssignOp(lo, lp, lv), AssignOp(ro, rp, rv)) => lo.node == ro.node && eq_expr(lp, rp) && eq_expr(lv, rv),
        (Field(lp, lf), Field(rp, rf)) => eq_id(*lf, *rf) && eq_expr(lp, rp),
        (Match(ls, la), Match(rs, ra)) => eq_expr(ls, rs) && over(la, ra, eq_arm),
        (Closure(lc, la, lm, lf, lb, _), Closure(rc, ra, rm, rf, rb, _)) => {
            lc == rc && la.is_async() == ra.is_async() && lm == rm && eq_fn_decl(lf, rf) && eq_expr(lb, rb)
        },
        (Async(lc, _, lb), Async(rc, _, rb)) => lc == rc && eq_block(lb, rb),
        (Range(lf, lt, ll), Range(rf, rt, rl)) => ll == rl && eq_expr_opt(lf, rf) && eq_expr_opt(lt, rt),
        (AddrOf(lbk, lm, le), AddrOf(rbk, rm, re)) => lbk == rbk && lm == rm && eq_expr(le, re),
        (Path(lq, lp), Path(rq, rp)) => both(lq, rq, eq_qself) && eq_path(lp, rp),
        (MacCall(l), MacCall(r)) => eq_mac_call(l, r),
        (Struct(lse), Struct(rse)) => {
            eq_maybe_qself(&lse.qself, &rse.qself)
                && eq_path(&lse.path, &rse.path)
                && eq_struct_rest(&lse.rest, &rse.rest)
                && unordered_over(&lse.fields, &rse.fields, eq_field)
        },
        _ => false,
    }
}

pub fn eq_field(l: &ExprField, r: &ExprField) -> bool {
    l.is_placeholder == r.is_placeholder
        && eq_id(l.ident, r.ident)
        && eq_expr(&l.expr, &r.expr)
        && over(&l.attrs, &r.attrs, eq_attr)
}

pub fn eq_arm(l: &Arm, r: &Arm) -> bool {
    l.is_placeholder == r.is_placeholder
        && eq_pat(&l.pat, &r.pat)
        && eq_expr(&l.body, &r.body)
        && eq_expr_opt(&l.guard, &r.guard)
        && over(&l.attrs, &r.attrs, eq_attr)
}

pub fn eq_label(l: &Option<Label>, r: &Option<Label>) -> bool {
    both(l, r, |l, r| eq_id(l.ident, r.ident))
}

pub fn eq_block(l: &Block, r: &Block) -> bool {
    l.rules == r.rules && over(&l.stmts, &r.stmts, eq_stmt)
}

pub fn eq_stmt(l: &Stmt, r: &Stmt) -> bool {
    use StmtKind::*;
    match (&l.kind, &r.kind) {
        (Local(l), Local(r)) => {
            eq_pat(&l.pat, &r.pat)
                && both(&l.ty, &r.ty, |l, r| eq_ty(l, r))
                && eq_local_kind(&l.kind, &r.kind)
                && over(&l.attrs, &r.attrs, eq_attr)
        },
        (Item(l), Item(r)) => eq_item(l, r, eq_item_kind),
        (Expr(l), Expr(r)) | (Semi(l), Semi(r)) => eq_expr(l, r),
        (Empty, Empty) => true,
        (MacCall(l), MacCall(r)) => {
            l.style == r.style && eq_mac_call(&l.mac, &r.mac) && over(&l.attrs, &r.attrs, eq_attr)
        },
        _ => false,
    }
}

pub fn eq_local_kind(l: &LocalKind, r: &LocalKind) -> bool {
    use LocalKind::*;
    match (l, r) {
        (Decl, Decl) => true,
        (Init(l), Init(r)) => eq_expr(l, r),
        (InitElse(li, le), InitElse(ri, re)) => eq_expr(li, ri) && eq_block(le, re),
        _ => false,
    }
}

pub fn eq_item<K>(l: &Item<K>, r: &Item<K>, mut eq_kind: impl FnMut(&K, &K) -> bool) -> bool {
    eq_id(l.ident, r.ident) && over(&l.attrs, &r.attrs, eq_attr) && eq_vis(&l.vis, &r.vis) && eq_kind(&l.kind, &r.kind)
}

pub fn eq_item_kind(l: &ItemKind, r: &ItemKind) -> bool {
    use ItemKind::*;
    match (l, r) {
        (ExternCrate(l), ExternCrate(r)) => l == r,
        (Use(l), Use(r)) => eq_use_tree(l, r),
        (Static(lt, lm, le), Static(rt, rm, re)) => lm == rm && eq_ty(lt, rt) && eq_expr_opt(le, re),
        (Const(ld, lt, le), Const(rd, rt, re)) => eq_defaultness(*ld, *rd) && eq_ty(lt, rt) && eq_expr_opt(le, re),
        (Fn(box ast::Fn { defaultness: ld, sig: lf, generics: lg, body: lb }),
         Fn(box ast::Fn { defaultness: rd, sig: rf, generics: rg, body: rb })) => {
            eq_defaultness(*ld, *rd) && eq_fn_sig(lf, rf) && eq_generics(lg, rg) && both(lb, rb, |l, r| eq_block(l, r))
        },
        (Mod(lu, lmk), Mod(ru, rmk)) => {
            lu == ru
                && match (lmk, rmk) {
                    (ModKind::Loaded(litems, linline, _), ModKind::Loaded(ritems, rinline, _)) => {
                        linline == rinline && over(litems, ritems, |l, r| eq_item(l, r, eq_item_kind))
                    },
                    (ModKind::Unloaded, ModKind::Unloaded) => true,
                    _ => false,
                }
        },
        (ForeignMod(l), ForeignMod(r)) => {
            both(&l.abi, &r.abi, eq_str_lit) && over(&l.items, &r.items, |l, r| eq_item(l, r, eq_foreign_item_kind))
        },
        (TyAlias(box ast::TyAlias { defaultness: ld, generics: lg, bounds: lb, ty: lt }),
         TyAlias(box ast::TyAlias { defaultness: rd, generics: rg, bounds: rb, ty: rt })) => {
            eq_defaultness(*ld, *rd)
                && eq_generics(lg, rg)
                && over(lb, rb, eq_generic_bound)
                && both(lt, rt, |l, r| eq_ty(l, r))
        },
        (Enum(le, lg), Enum(re, rg)) => over(&le.variants, &re.variants, eq_variant) && eq_generics(lg, rg),
        (Struct(lv, lg), Struct(rv, rg)) | (Union(lv, lg), Union(rv, rg)) => {
            eq_variant_data(lv, rv) && eq_generics(lg, rg)
        },
        (Trait(box ast::Trait { is_auto: la, unsafety: lu, generics: lg, bounds: lb, items: li }),
         Trait(box ast::Trait { is_auto: ra, unsafety: ru, generics: rg, bounds: rb, items: ri })) => {
            la == ra
                && matches!(lu, Unsafe::No) == matches!(ru, Unsafe::No)
                && eq_generics(lg, rg)
                && over(lb, rb, eq_generic_bound)
                && over(li, ri, |l, r| eq_item(l, r, eq_assoc_item_kind))
        },
        (TraitAlias(lg, lb), TraitAlias(rg, rb)) => eq_generics(lg, rg) && over(lb, rb, eq_generic_bound),
        (
            Impl(box ast::Impl {
                unsafety: lu,
                polarity: lp,
                defaultness: ld,
                constness: lc,
                generics: lg,
                of_trait: lot,
                self_ty: lst,
                items: li,
            }),
            Impl(box ast::Impl {
                unsafety: ru,
                polarity: rp,
                defaultness: rd,
                constness: rc,
                generics: rg,
                of_trait: rot,
                self_ty: rst,
                items: ri,
            }),
        ) => {
            matches!(lu, Unsafe::No) == matches!(ru, Unsafe::No)
                && matches!(lp, ImplPolarity::Positive) == matches!(rp, ImplPolarity::Positive)
                && eq_defaultness(*ld, *rd)
                && matches!(lc, ast::Const::No) == matches!(rc, ast::Const::No)
                && eq_generics(lg, rg)
                && both(lot, rot, |l, r| eq_path(&l.path, &r.path))
                && eq_ty(lst, rst)
                && over(li, ri, |l, r| eq_item(l, r, eq_assoc_item_kind))
        },
        (MacCall(l), MacCall(r)) => eq_mac_call(l, r),
        (MacroDef(l), MacroDef(r)) => l.macro_rules == r.macro_rules && eq_mac_args(&l.body, &r.body),
        _ => false,
    }
}

pub fn eq_foreign_item_kind(l: &ForeignItemKind, r: &ForeignItemKind) -> bool {
    use ForeignItemKind::*;
    match (l, r) {
        (Static(lt, lm, le), Static(rt, rm, re)) => lm == rm && eq_ty(lt, rt) && eq_expr_opt(le, re),
        (Fn(box ast::Fn { defaultness: ld, sig: lf, generics: lg, body: lb }),
         Fn(box ast::Fn { defaultness: rd, sig: rf, generics: rg, body: rb })) => {
            eq_defaultness(*ld, *rd) && eq_fn_sig(lf, rf) && eq_generics(lg, rg) && both(lb, rb, |l, r| eq_block(l, r))
        },
        (TyAlias(box ast::TyAlias { defaultness: ld, generics: lg, bounds: lb, ty: lt }),
         TyAlias(box ast::TyAlias { defaultness: rd, generics: rg, bounds: rb, ty: rt })) => {
            eq_defaultness(*ld, *rd)
                && eq_generics(lg, rg)
                && over(lb, rb, eq_generic_bound)
                && both(lt, rt, |l, r| eq_ty(l, r))
        },
        (MacCall(l), MacCall(r)) => eq_mac_call(l, r),
        _ => false,
    }
}

pub fn eq_assoc_item_kind(l: &AssocItemKind, r: &AssocItemKind) -> bool {
    use AssocItemKind::*;
    match (l, r) {
        (Const(ld, lt, le), Const(rd, rt, re)) => eq_defaultness(*ld, *rd) && eq_ty(lt, rt) && eq_expr_opt(le, re),
        (Fn(box ast::Fn { defaultness: ld, sig: lf, generics: lg, body: lb }),
         Fn(box ast::Fn { defaultness: rd, sig: rf, generics: rg, body: rb })) => {
            eq_defaultness(*ld, *rd) && eq_fn_sig(lf, rf) && eq_generics(lg, rg) && both(lb, rb, |l, r| eq_block(l, r))
        },
        (TyAlias(box ast::TyAlias { defaultness: ld, generics: lg, bounds: lb, ty: lt }),
         TyAlias(box ast::TyAlias { defaultness: rd, generics: rg, bounds: rb, ty: rt })) => {
            eq_defaultness(*ld, *rd)
                && eq_generics(lg, rg)
                && over(lb, rb, eq_generic_bound)
                && both(lt, rt, |l, r| eq_ty(l, r))
        },
        (MacCall(l), MacCall(r)) => eq_mac_call(l, r),
        _ => false,
    }
}

pub fn eq_variant(l: &Variant, r: &Variant) -> bool {
    l.is_placeholder == r.is_placeholder
        && over(&l.attrs, &r.attrs, eq_attr)
        && eq_vis(&l.vis, &r.vis)
        && eq_id(l.ident, r.ident)
        && eq_variant_data(&l.data, &r.data)
        && both(&l.disr_expr, &r.disr_expr, |l, r| eq_expr(&l.value, &r.value))
}

pub fn eq_variant_data(l: &VariantData, r: &VariantData) -> bool {
    use VariantData::*;
    match (l, r) {
        (Unit(_), Unit(_)) => true,
        (Struct(l, _), Struct(r, _)) | (Tuple(l, _), Tuple(r, _)) => over(l, r, eq_struct_field),
        _ => false,
    }
}

pub fn eq_struct_field(l: &FieldDef, r: &FieldDef) -> bool {
    l.is_placeholder == r.is_placeholder
        && over(&l.attrs, &r.attrs, eq_attr)
        && eq_vis(&l.vis, &r.vis)
        && both(&l.ident, &r.ident, |l, r| eq_id(*l, *r))
        && eq_ty(&l.ty, &r.ty)
}

pub fn eq_fn_sig(l: &FnSig, r: &FnSig) -> bool {
    eq_fn_decl(&l.decl, &r.decl) && eq_fn_header(&l.header, &r.header)
}

pub fn eq_fn_header(l: &FnHeader, r: &FnHeader) -> bool {
    matches!(l.unsafety, Unsafe::No) == matches!(r.unsafety, Unsafe::No)
        && l.asyncness.is_async() == r.asyncness.is_async()
        && matches!(l.constness, Const::No) == matches!(r.constness, Const::No)
        && eq_ext(&l.ext, &r.ext)
}

pub fn eq_generics(l: &Generics, r: &Generics) -> bool {
    over(&l.params, &r.params, eq_generic_param)
        && over(&l.where_clause.predicates, &r.where_clause.predicates, |l, r| {
            eq_where_predicate(l, r)
        })
}

pub fn eq_where_predicate(l: &WherePredicate, r: &WherePredicate) -> bool {
    use WherePredicate::*;
    match (l, r) {
        (BoundPredicate(l), BoundPredicate(r)) => {
            over(&l.bound_generic_params, &r.bound_generic_params, |l, r| {
                eq_generic_param(l, r)
            }) && eq_ty(&l.bounded_ty, &r.bounded_ty)
                && over(&l.bounds, &r.bounds, eq_generic_bound)
        },
        (RegionPredicate(l), RegionPredicate(r)) => {
            eq_id(l.lifetime.ident, r.lifetime.ident) && over(&l.bounds, &r.bounds, eq_generic_bound)
        },
        (EqPredicate(l), EqPredicate(r)) => eq_ty(&l.lhs_ty, &r.lhs_ty) && eq_ty(&l.rhs_ty, &r.rhs_ty),
        _ => false,
    }
}

pub fn eq_use_tree(l: &UseTree, r: &UseTree) -> bool {
    eq_path(&l.prefix, &r.prefix) && eq_use_tree_kind(&l.kind, &r.kind)
}

pub fn eq_anon_const(l: &AnonConst, r: &AnonConst) -> bool {
    eq_expr(&l.value, &r.value)
}

pub fn eq_use_tree_kind(l: &UseTreeKind, r: &UseTreeKind) -> bool {
    use UseTreeKind::*;
    match (l, r) {
        (Glob, Glob) => true,
        (Simple(l, _, _), Simple(r, _, _)) => both(l, r, |l, r| eq_id(*l, *r)),
        (Nested(l), Nested(r)) => over(l, r, |(l, _), (r, _)| eq_use_tree(l, r)),
        _ => false,
    }
}

pub fn eq_defaultness(l: Defaultness, r: Defaultness) -> bool {
    matches!(
        (l, r),
        (Defaultness::Final, Defaultness::Final) | (Defaultness::Default(_), Defaultness::Default(_))
    )
}

pub fn eq_vis(l: &Visibility, r: &Visibility) -> bool {
    use VisibilityKind::*;
    match (&l.kind, &r.kind) {
        (Public, Public) | (Inherited, Inherited) | (Crate(_), Crate(_)) => true,
        (Restricted { path: l, .. }, Restricted { path: r, .. }) => eq_path(l, r),
        _ => false,
    }
}

pub fn eq_fn_decl(l: &FnDecl, r: &FnDecl) -> bool {
    eq_fn_ret_ty(&l.output, &r.output)
        && over(&l.inputs, &r.inputs, |l, r| {
            l.is_placeholder == r.is_placeholder
                && eq_pat(&l.pat, &r.pat)
                && eq_ty(&l.ty, &r.ty)
                && over(&l.attrs, &r.attrs, eq_attr)
        })
}

pub fn eq_fn_ret_ty(l: &FnRetTy, r: &FnRetTy) -> bool {
    match (l, r) {
        (FnRetTy::Default(_), FnRetTy::Default(_)) => true,
        (FnRetTy::Ty(l), FnRetTy::Ty(r)) => eq_ty(l, r),
        _ => false,
    }
}

pub fn eq_ty(l: &Ty, r: &Ty) -> bool {
    use TyKind::*;
    match (&l.kind, &r.kind) {
        (Paren(l), _) => eq_ty(l, r),
        (_, Paren(r)) => eq_ty(l, r),
        (Never, Never) | (Infer, Infer) | (ImplicitSelf, ImplicitSelf) | (Err, Err) | (CVarArgs, CVarArgs) => true,
        (Slice(l), Slice(r)) => eq_ty(l, r),
        (Array(le, ls), Array(re, rs)) => eq_ty(le, re) && eq_expr(&ls.value, &rs.value),
        (Ptr(l), Ptr(r)) => l.mutbl == r.mutbl && eq_ty(&l.ty, &r.ty),
        (Rptr(ll, l), Rptr(rl, r)) => {
            both(ll, rl, |l, r| eq_id(l.ident, r.ident)) && l.mutbl == r.mutbl && eq_ty(&l.ty, &r.ty)
        },
        (BareFn(l), BareFn(r)) => {
            l.unsafety == r.unsafety
                && eq_ext(&l.ext, &r.ext)
                && over(&l.generic_params, &r.generic_params, eq_generic_param)
                && eq_fn_decl(&l.decl, &r.decl)
        },
        (Tup(l), Tup(r)) => over(l, r, |l, r| eq_ty(l, r)),
        (Path(lq, lp), Path(rq, rp)) => both(lq, rq, eq_qself) && eq_path(lp, rp),
        (TraitObject(lg, ls), TraitObject(rg, rs)) => ls == rs && over(lg, rg, eq_generic_bound),
        (ImplTrait(_, lg), ImplTrait(_, rg)) => over(lg, rg, eq_generic_bound),
        (Typeof(l), Typeof(r)) => eq_expr(&l.value, &r.value),
        (MacCall(l), MacCall(r)) => eq_mac_call(l, r),
        _ => false,
    }
}

pub fn eq_ext(l: &Extern, r: &Extern) -> bool {
    use Extern::*;
    match (l, r) {
        (None, None) | (Implicit, Implicit) => true,
        (Explicit(l), Explicit(r)) => eq_str_lit(l, r),
        _ => false,
    }
}

pub fn eq_str_lit(l: &StrLit, r: &StrLit) -> bool {
    l.style == r.style && l.symbol == r.symbol && l.suffix == r.suffix
}

pub fn eq_poly_ref_trait(l: &PolyTraitRef, r: &PolyTraitRef) -> bool {
    eq_path(&l.trait_ref.path, &r.trait_ref.path)
        && over(&l.bound_generic_params, &r.bound_generic_params, |l, r| {
            eq_generic_param(l, r)
        })
}

pub fn eq_generic_param(l: &GenericParam, r: &GenericParam) -> bool {
    use GenericParamKind::*;
    l.is_placeholder == r.is_placeholder
        && eq_id(l.ident, r.ident)
        && over(&l.bounds, &r.bounds, eq_generic_bound)
        && match (&l.kind, &r.kind) {
            (Lifetime, Lifetime) => true,
            (Type { default: l }, Type { default: r }) => both(l, r, |l, r| eq_ty(l, r)),
            (
                Const {
                    ty: lt,
                    kw_span: _,
                    default: ld,
                },
                Const {
                    ty: rt,
                    kw_span: _,
                    default: rd,
                },
            ) => eq_ty(lt, rt) && both(ld, rd, eq_anon_const),
            _ => false,
        }
        && over(&l.attrs, &r.attrs, eq_attr)
}

pub fn eq_generic_bound(l: &GenericBound, r: &GenericBound) -> bool {
    use GenericBound::*;
    match (l, r) {
        (Trait(ptr1, tbm1), Trait(ptr2, tbm2)) => tbm1 == tbm2 && eq_poly_ref_trait(ptr1, ptr2),
        (Outlives(l), Outlives(r)) => eq_id(l.ident, r.ident),
        _ => false,
    }
}

pub fn eq_assoc_constraint(l: &AssocTyConstraint, r: &AssocTyConstraint) -> bool {
    use AssocTyConstraintKind::*;
    eq_id(l.ident, r.ident)
        && match (&l.kind, &r.kind) {
            (Equality { ty: l }, Equality { ty: r }) => eq_ty(l, r),
            (Bound { bounds: l }, Bound { bounds: r }) => over(l, r, eq_generic_bound),
            _ => false,
        }
}

pub fn eq_mac_call(l: &MacCall, r: &MacCall) -> bool {
    eq_path(&l.path, &r.path) && eq_mac_args(&l.args, &r.args)
}

pub fn eq_attr(l: &Attribute, r: &Attribute) -> bool {
    use AttrKind::*;
    l.style == r.style
        && match (&l.kind, &r.kind) {
            (DocComment(l1, l2), DocComment(r1, r2)) => l1 == r1 && l2 == r2,
            (Normal(l, _), Normal(r, _)) => eq_path(&l.path, &r.path) && eq_mac_args(&l.args, &r.args),
            _ => false,
        }
}

pub fn eq_mac_args(l: &MacArgs, r: &MacArgs) -> bool {
    use MacArgs::*;
    match (l, r) {
        (Empty, Empty) => true,
        (Delimited(_, ld, lts), Delimited(_, rd, rts)) => ld == rd && lts.eq_unspanned(rts),
        (Eq(_, lt), Eq(_, rt)) => lt.kind == rt.kind,
        _ => false,
    }
}

/// Extract args from an assert-like macro.
///
/// Currently working with:
/// - `assert_eq!` and `assert_ne!`
/// - `debug_assert_eq!` and `debug_assert_ne!`
///
/// For example:
///
/// `debug_assert_eq!(a, b)` will return Some([a, b])
pub fn extract_assert_macro_args(mut expr: &Expr) -> Option<[&Expr; 2]> {
    if_chain! {
        if let ExprKind::If(_, ref block, _) = expr.kind;
        if let StmtKind::Semi(ref e) = block.stmts.get(0)?.kind;
        then {
            expr = e;
        }
    }
    if_chain! {
        if let ExprKind::Block(ref block, _) = expr.kind;
        if let StmtKind::Expr(ref expr) = block.stmts.get(0)?.kind;
        if let ExprKind::Match(ref match_expr, _) = expr.kind;
        if let ExprKind::Tup(ref tup) = match_expr.kind;
        if let [a, b, ..] = tup.as_slice();
        if let (&ExprKind::AddrOf(_, _, ref a), &ExprKind::AddrOf(_, _, ref b)) = (&a.kind, &b.kind);
        then {
            return Some([&*a, &*b]);
        }
    }
    None
}
