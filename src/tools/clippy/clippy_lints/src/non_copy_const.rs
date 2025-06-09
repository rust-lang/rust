// Implementation for lints detecting interior mutability in constants.
//
// For `declare_interior_mutable_const` there are three strategies used to
// determine if a value has interior mutability:
// * A type-based check. This is the least accurate, but can always run.
// * A const-eval based check. This is the most accurate, but this requires that the value is
//   defined and does not work with generics.
// * A HIR-tree based check. This is less accurate than const-eval, but it can be applied to generic
//   values.
//
// For `borrow_interior_mutable_const` the same three strategies are applied
// when checking a constant's value, but field and array index projections at
// the borrow site are taken into account as well. As an example: `FOO.bar` may
// have interior mutability, but `FOO.baz` may not. When borrowing `FOO.baz` no
// warning will be issued.
//
// No matter the lint or strategy, a warning should only be issued if a value
// definitely contains interior mutability.

use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::is_in_const_context;
use clippy_utils::macros::macro_backtrace;
use clippy_utils::paths::{PathNS, lookup_path_str};
use clippy_utils::ty::{get_field_idx_by_name, implements_trait};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_hir::{
    Expr, ExprKind, ImplItem, ImplItemKind, Item, ItemKind, Node, StructTailExpr, TraitItem, TraitItemKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::mir::{ConstValue, UnevaluatedConst};
use rustc_middle::ty::adjustment::{Adjust, Adjustment};
use rustc_middle::ty::{
    self, AliasTyKind, EarlyBinder, GenericArgs, GenericArgsRef, Instance, Ty, TyCtxt, TypeFolder, TypeSuperFoldable,
    TypeckResults, TypingEnv,
};
use rustc_session::impl_lint_pass;
use rustc_span::{DUMMY_SP, sym};
use std::collections::hash_map::Entry;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the declaration of named constant which contain interior mutability.
    ///
    /// ### Why is this bad?
    /// Named constants are copied at every use site which means any change to their value
    /// will be lost after the newly created value is dropped. e.g.
    ///
    /// ```rust
    /// use core::sync::atomic::{AtomicUsize, Ordering};
    /// const ATOMIC: AtomicUsize = AtomicUsize::new(0);
    /// fn add_one() -> usize {
    ///     // This will always return `0` since `ATOMIC` is copied before it's used.
    ///     ATOMIC.fetch_add(1, Ordering::AcqRel)
    /// }
    /// ```
    ///
    /// If shared modification of the value is desired, a `static` item is needed instead.
    /// If that is not desired, a `const fn` constructor should be used to make it obvious
    /// at the use site that a new value is created.
    ///
    /// ### Known problems
    /// Prior to `const fn` stabilization this was the only way to provide a value which
    /// could initialize a `static` item (e.g. the `std::sync::ONCE_INIT` constant). In
    /// this case the use of `const` is required and this lint should be suppressed.
    ///
    /// There also exists types which contain private fields with interior mutability, but
    /// no way to both create a value as a constant and modify any mutable field using the
    /// type's public interface (e.g. `bytes::Bytes`). As there is no reasonable way to
    /// scan a crate's interface to see if this is the case, all such types will be linted.
    /// If this happens use the `ignore-interior-mutability` configuration option to allow
    /// the type.
    ///
    /// ### Example
    /// ```no_run
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    ///
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    /// CONST_ATOM.store(6, SeqCst); // the content of the atomic is unchanged
    /// assert_eq!(CONST_ATOM.load(SeqCst), 12); // because the CONST_ATOM in these lines are distinct
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    /// static STATIC_ATOM: AtomicUsize = AtomicUsize::new(15);
    /// STATIC_ATOM.store(9, SeqCst);
    /// assert_eq!(STATIC_ATOM.load(SeqCst), 9); // use a `static` item to refer to the same instance
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DECLARE_INTERIOR_MUTABLE_CONST,
    style,
    "declaring `const` with interior mutability"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for a borrow of a named constant with interior mutability.
    ///
    /// ### Why is this bad?
    /// Named constants are copied at every use site which means any change to their value
    /// will be lost after the newly created value is dropped. e.g.
    ///
    /// ```rust
    /// use core::sync::atomic::{AtomicUsize, Ordering};
    /// const ATOMIC: AtomicUsize = AtomicUsize::new(0);
    /// fn add_one() -> usize {
    ///     // This will always return `0` since `ATOMIC` is copied before it's borrowed
    ///     // for use by `fetch_add`.
    ///     ATOMIC.fetch_add(1, Ordering::AcqRel)
    /// }
    /// ```
    ///
    /// ### Known problems
    /// This lint does not, and cannot in general, determine if the borrow of the constant
    /// is used in a way which causes a mutation. e.g.
    ///
    /// ```rust
    /// use core::cell::Cell;
    /// const CELL: Cell<usize> = Cell::new(0);
    /// fn get_cell() -> Cell<usize> {
    ///     // This is fine. It borrows a copy of `CELL`, but never mutates it through the
    ///     // borrow.
    ///     CELL.clone()
    /// }
    /// ```
    ///
    /// There also exists types which contain private fields with interior mutability, but
    /// no way to both create a value as a constant and modify any mutable field using the
    /// type's public interface (e.g. `bytes::Bytes`). As there is no reasonable way to
    /// scan a crate's interface to see if this is the case, all such types will be linted.
    /// If this happens use the `ignore-interior-mutability` configuration option to allow
    /// the type.
    ///
    /// ### Example
    /// ```no_run
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    ///
    /// CONST_ATOM.store(6, SeqCst); // the content of the atomic is unchanged
    /// assert_eq!(CONST_ATOM.load(SeqCst), 12); // because the CONST_ATOM in these lines are distinct
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    /// const CONST_ATOM: AtomicUsize = AtomicUsize::new(12);
    ///
    /// static STATIC_ATOM: AtomicUsize = CONST_ATOM;
    /// STATIC_ATOM.store(9, SeqCst);
    /// assert_eq!(STATIC_ATOM.load(SeqCst), 9); // use a `static` item to refer to the same instance
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BORROW_INTERIOR_MUTABLE_CONST,
    style,
    "referencing `const` with interior mutability"
}

#[derive(Clone, Copy)]
enum IsFreeze {
    /// The type and all possible values are `Freeze`
    Yes,
    /// The type itself is non-`Freeze`, but not all values are.
    Maybe,
    /// The type and all possible values are non-`Freeze`
    No,
}
impl IsFreeze {
    /// Merges the variants of a sum type (i.e. an enum).
    fn from_variants(iter: impl Iterator<Item = Self>) -> Self {
        iter.fold(Self::Yes, |x, y| match (x, y) {
            (Self::Maybe, _) | (_, Self::Maybe) | (Self::No, Self::Yes) | (Self::Yes, Self::No) => Self::Maybe,
            (Self::No, Self::No) => Self::No,
            (Self::Yes, Self::Yes) => Self::Yes,
        })
    }

    /// Merges the fields of a product type (e.g. a struct or tuple).
    fn from_fields(mut iter: impl Iterator<Item = Self>) -> Self {
        iter.try_fold(Self::Yes, |x, y| match (x, y) {
            (Self::No, _) | (_, Self::No) => None,
            (Self::Maybe, _) | (_, Self::Maybe) => Some(Self::Maybe),
            (Self::Yes, Self::Yes) => Some(Self::Yes),
        })
        .unwrap_or(Self::No)
    }

    /// Checks if this is definitely `Freeze`.
    fn is_freeze(self) -> bool {
        matches!(self, Self::Yes)
    }

    /// Checks if this is definitely not `Freeze`.
    fn is_not_freeze(self) -> bool {
        matches!(self, Self::No)
    }
}

/// What operation caused a borrow to occur.
#[derive(Clone, Copy)]
enum BorrowCause {
    Borrow,
    Deref,
    Index,
    AutoDeref,
    AutoBorrow,
    AutoDerefField,
}
impl BorrowCause {
    fn note(self) -> Option<&'static str> {
        match self {
            Self::Borrow => None,
            Self::Deref => Some("this deref expression is a call to `Deref::deref`"),
            Self::Index => Some("this index expression is a call to `Index::index`"),
            Self::AutoDeref => Some("there is a compiler inserted call to `Deref::deref` here"),
            Self::AutoBorrow => Some("there is a compiler inserted borrow here"),
            Self::AutoDerefField => {
                Some("there is a compiler inserted call to `Deref::deref` when accessing this field")
            },
        }
    }
}

/// The source of a borrow. Both what caused it and where.
struct BorrowSource<'tcx> {
    expr: &'tcx Expr<'tcx>,
    cause: BorrowCause,
}
impl<'tcx> BorrowSource<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, expr: &'tcx Expr<'tcx>, cause: BorrowCause) -> Self {
        // Custom deref and index impls will always have an auto-borrow inserted since we
        // never work with reference types.
        let (expr, cause) = if matches!(cause, BorrowCause::AutoBorrow)
            && let Node::Expr(parent) = tcx.parent_hir_node(expr.hir_id)
        {
            match parent.kind {
                ExprKind::Unary(UnOp::Deref, _) => (parent, BorrowCause::Deref),
                ExprKind::Index(..) => (parent, BorrowCause::Index),
                ExprKind::Field(..) => (parent, BorrowCause::AutoDerefField),
                _ => (expr, cause),
            }
        } else {
            (expr, cause)
        };
        Self { expr, cause }
    }
}

pub struct NonCopyConst<'tcx> {
    ignore_tys: DefIdSet,
    // Cache checked types. We can recurse through a type multiple times so this
    // can be hit quite frequently.
    freeze_tys: FxHashMap<Ty<'tcx>, IsFreeze>,
}

impl_lint_pass!(NonCopyConst<'_> => [DECLARE_INTERIOR_MUTABLE_CONST, BORROW_INTERIOR_MUTABLE_CONST]);

impl<'tcx> NonCopyConst<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, conf: &'static Conf) -> Self {
        Self {
            ignore_tys: conf
                .ignore_interior_mutability
                .iter()
                .flat_map(|ignored_ty| lookup_path_str(tcx, PathNS::Type, ignored_ty))
                .collect(),
            freeze_tys: FxHashMap::default(),
        }
    }

    /// Checks if a value of the given type is `Freeze`, or may be depending on the value.
    fn is_ty_freeze(&mut self, tcx: TyCtxt<'tcx>, typing_env: TypingEnv<'tcx>, ty: Ty<'tcx>) -> IsFreeze {
        let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);
        match self.freeze_tys.entry(ty) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let e = e.insert(IsFreeze::Yes);
                if ty.is_freeze(tcx, typing_env) {
                    return IsFreeze::Yes;
                }
                let is_freeze = match *ty.kind() {
                    ty::Adt(adt, _) if adt.is_unsafe_cell() => {
                        *e = IsFreeze::No;
                        return IsFreeze::No;
                    },
                    ty::Adt(adt, _) if self.ignore_tys.contains(&adt.did()) => return IsFreeze::Yes,
                    ty::Adt(adt, args) if adt.is_enum() => IsFreeze::from_variants(adt.variants().iter().map(|v| {
                        IsFreeze::from_fields(
                            v.fields
                                .iter()
                                .map(|f| self.is_ty_freeze(tcx, typing_env, f.ty(tcx, args))),
                        )
                    })),
                    // Workaround for `ManuallyDrop`-like unions.
                    ty::Adt(adt, args)
                        if adt.is_union()
                            && adt.non_enum_variant().fields.iter().any(|f| {
                                tcx.layout_of(typing_env.as_query_input(f.ty(tcx, args)))
                                    .is_ok_and(|l| l.layout.size().bytes() == 0)
                            }) =>
                    {
                        return IsFreeze::Yes;
                    },
                    // Rust doesn't have the concept of an active union field so we have
                    // to treat all fields as active.
                    ty::Adt(adt, args) => IsFreeze::from_fields(
                        adt.non_enum_variant()
                            .fields
                            .iter()
                            .map(|f| self.is_ty_freeze(tcx, typing_env, f.ty(tcx, args))),
                    ),
                    ty::Array(ty, _) | ty::Pat(ty, _) => self.is_ty_freeze(tcx, typing_env, ty),
                    ty::Tuple(tys) => {
                        IsFreeze::from_fields(tys.iter().map(|ty| self.is_ty_freeze(tcx, typing_env, ty)))
                    },
                    // Treat type parameters as though they were `Freeze`.
                    ty::Param(_) | ty::Alias(..) => return IsFreeze::Yes,
                    // TODO: check other types.
                    _ => {
                        *e = IsFreeze::No;
                        return IsFreeze::No;
                    },
                };
                if !is_freeze.is_freeze() {
                    self.freeze_tys.insert(ty, is_freeze);
                }
                is_freeze
            },
        }
    }

    /// Checks if the given constant value is `Freeze`. Returns `Err` if the constant
    /// cannot be read, but the result depends on the value.
    fn is_value_freeze(
        &mut self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        ty: Ty<'tcx>,
        val: ConstValue<'tcx>,
    ) -> Result<bool, ()> {
        let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);
        match self.is_ty_freeze(tcx, typing_env, ty) {
            IsFreeze::Yes => Ok(true),
            IsFreeze::Maybe if matches!(ty.kind(), ty::Adt(..) | ty::Array(..) | ty::Tuple(..)) => {
                for &(val, ty) in tcx
                    .try_destructure_mir_constant_for_user_output(val, ty)
                    .ok_or(())?
                    .fields
                {
                    if !self.is_value_freeze(tcx, typing_env, ty, val)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
            IsFreeze::Maybe | IsFreeze::No => Ok(false),
        }
    }

    /// Checks if the given expression creates a value which is `Freeze`.
    ///
    /// This will return `true` if the type is maybe `Freeze`, but it cannot be
    /// determined for certain from the value.
    ///
    /// `typing_env` and `gen_args` are from the constant's use site.
    /// `typeck` and `e` are from the constant's definition site.
    fn is_init_expr_freeze(
        &mut self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        typeck: &'tcx TypeckResults<'tcx>,
        gen_args: GenericArgsRef<'tcx>,
        e: &'tcx Expr<'tcx>,
    ) -> bool {
        // Make sure to instantiate all types coming from `typeck` with `gen_args`.
        let ty = EarlyBinder::bind(typeck.expr_ty(e)).instantiate(tcx, gen_args);
        let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);
        match self.is_ty_freeze(tcx, typing_env, ty) {
            IsFreeze::Yes => true,
            IsFreeze::No => false,
            IsFreeze::Maybe => match e.kind {
                ExprKind::Block(b, _)
                    if !b.targeted_by_break
                        && b.stmts.is_empty()
                        && let Some(e) = b.expr =>
                {
                    self.is_init_expr_freeze(tcx, typing_env, typeck, gen_args, e)
                },
                ExprKind::Path(ref p) => {
                    let res = typeck.qpath_res(p, e.hir_id);
                    let gen_args = EarlyBinder::bind(typeck.node_args(e.hir_id)).instantiate(tcx, gen_args);
                    match res {
                        Res::Def(DefKind::Const | DefKind::AssocConst, did)
                            if let Ok(val) =
                                tcx.const_eval_resolve(typing_env, UnevaluatedConst::new(did, gen_args), DUMMY_SP)
                                && let Ok(is_freeze) = self.is_value_freeze(tcx, typing_env, ty, val) =>
                        {
                            is_freeze
                        },
                        Res::Def(DefKind::Const | DefKind::AssocConst, did)
                            if let Some((typeck, init)) = get_const_hir_value(tcx, typing_env, did, gen_args) =>
                        {
                            self.is_init_expr_freeze(tcx, typing_env, typeck, gen_args, init)
                        },
                        // Either this is a unit constructor, or some unknown value.
                        // In either case we consider the value to be `Freeze`.
                        _ => true,
                    }
                },
                ExprKind::Call(callee, args)
                    if let ExprKind::Path(p) = &callee.kind
                        && let res = typeck.qpath_res(p, callee.hir_id)
                        && matches!(res, Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(_)) =>
                {
                    args.iter()
                        .all(|e| self.is_init_expr_freeze(tcx, typing_env, typeck, gen_args, e))
                },
                ExprKind::Struct(_, fields, StructTailExpr::None) => fields
                    .iter()
                    .all(|f| self.is_init_expr_freeze(tcx, typing_env, typeck, gen_args, f.expr)),
                ExprKind::Tup(exprs) | ExprKind::Array(exprs) => exprs
                    .iter()
                    .all(|e| self.is_init_expr_freeze(tcx, typing_env, typeck, gen_args, e)),
                ExprKind::Repeat(e, _) => self.is_init_expr_freeze(tcx, typing_env, typeck, gen_args, e),
                _ => true,
            },
        }
    }

    /// Checks if the given expression (or a local projection of it) is both borrowed and
    /// definitely a non-`Freeze` type.
    fn is_non_freeze_expr_borrowed(
        &mut self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        typeck: &'tcx TypeckResults<'tcx>,
        mut src_expr: &'tcx Expr<'tcx>,
    ) -> Option<BorrowSource<'tcx>> {
        let mut parents = tcx.hir_parent_iter(src_expr.hir_id);
        loop {
            let ty = typeck.expr_ty(src_expr);
            // Normalized as we need to check if this is an array later.
            let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);
            let is_freeze = self.is_ty_freeze(tcx, typing_env, ty);
            if is_freeze.is_freeze() {
                return None;
            }
            if let [adjust, ..] = typeck.expr_adjustments(src_expr) {
                return does_adjust_borrow(adjust)
                    .filter(|_| is_freeze.is_not_freeze())
                    .map(|cause| BorrowSource::new(tcx, src_expr, cause));
            }
            let Some((_, Node::Expr(use_expr))) = parents.next() else {
                return None;
            };
            match use_expr.kind {
                ExprKind::Field(..) => {},
                ExprKind::Index(..) if ty.is_array() => {},
                ExprKind::AddrOf(..) if is_freeze.is_not_freeze() => {
                    return Some(BorrowSource::new(tcx, use_expr, BorrowCause::Borrow));
                },
                // All other expressions use the value.
                _ => return None,
            }
            src_expr = use_expr;
        }
    }

    /// Checks if the given value (or a local projection of it) is both borrowed and
    /// definitely non-`Freeze`. Returns `Err` if the constant cannot be read, but the
    /// result depends on the value.
    fn is_non_freeze_val_borrowed(
        &mut self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        typeck: &'tcx TypeckResults<'tcx>,
        mut src_expr: &'tcx Expr<'tcx>,
        mut val: ConstValue<'tcx>,
    ) -> Result<Option<BorrowSource<'tcx>>, ()> {
        let mut parents = tcx.hir_parent_iter(src_expr.hir_id);
        let mut ty = typeck.expr_ty(src_expr);
        loop {
            // Normalized as we need to check if this is an array later.
            ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);
            if let [adjust, ..] = typeck.expr_adjustments(src_expr) {
                let res = if let Some(cause) = does_adjust_borrow(adjust)
                    && !self.is_value_freeze(tcx, typing_env, ty, val)?
                {
                    Some(BorrowSource::new(tcx, src_expr, cause))
                } else {
                    None
                };
                return Ok(res);
            }
            // Check only the type here as the result gets cached for each type.
            if self.is_ty_freeze(tcx, typing_env, ty).is_freeze() {
                return Ok(None);
            }
            let Some((_, Node::Expr(use_expr))) = parents.next() else {
                return Ok(None);
            };
            let next_val = match use_expr.kind {
                ExprKind::Field(_, name) => {
                    if let Some(idx) = get_field_idx_by_name(ty, name.name) {
                        tcx.try_destructure_mir_constant_for_user_output(val, ty)
                            .ok_or(())?
                            .fields
                            .get(idx)
                    } else {
                        return Ok(None);
                    }
                },
                ExprKind::Index(_, idx, _) if ty.is_array() => {
                    let val = tcx.try_destructure_mir_constant_for_user_output(val, ty).ok_or(())?;
                    if let Some(Constant::Int(idx)) = ConstEvalCtxt::with_env(tcx, typing_env, typeck).eval(idx) {
                        val.fields.get(idx as usize)
                    } else {
                        // It's some value in the array so check all of them.
                        for &(val, _) in val.fields {
                            if let Some(src) =
                                self.is_non_freeze_val_borrowed(tcx, typing_env, typeck, use_expr, val)?
                            {
                                return Ok(Some(src));
                            }
                        }
                        return Ok(None);
                    }
                },
                ExprKind::AddrOf(..) if !self.is_value_freeze(tcx, typing_env, ty, val)? => {
                    return Ok(Some(BorrowSource::new(tcx, use_expr, BorrowCause::Borrow)));
                },
                // All other expressions use the value.
                _ => return Ok(None),
            };
            src_expr = use_expr;
            if let Some(&(next_val, next_ty)) = next_val {
                ty = next_ty;
                val = next_val;
            } else {
                return Ok(None);
            }
        }
    }

    /// Checks if the given value (or a local projection of it) is both borrowed and
    /// definitely non-`Freeze`.
    ///
    /// `typing_env` and `init_args` are from the constant's use site.
    /// `init_typeck` and `init_expr` are from the constant's definition site.
    #[expect(clippy::too_many_arguments, clippy::too_many_lines)]
    fn is_non_freeze_init_borrowed(
        &mut self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        typeck: &'tcx TypeckResults<'tcx>,
        mut src_expr: &'tcx Expr<'tcx>,
        mut init_typeck: &'tcx TypeckResults<'tcx>,
        mut init_args: GenericArgsRef<'tcx>,
        mut init_expr: &'tcx Expr<'tcx>,
    ) -> Option<BorrowSource<'tcx>> {
        // Make sure to instantiate all types coming from `init_typeck` with `init_args`.
        let mut parents = tcx.hir_parent_iter(src_expr.hir_id);
        loop {
            // First handle any adjustments since they are cheap to check.
            if let [adjust, ..] = typeck.expr_adjustments(src_expr) {
                return does_adjust_borrow(adjust)
                    .filter(|_| !self.is_init_expr_freeze(tcx, typing_env, init_typeck, init_args, init_expr))
                    .map(|cause| BorrowSource::new(tcx, src_expr, cause));
            }

            // Then read through constants and blocks on the init expression before
            // applying the next use expression.
            loop {
                match init_expr.kind {
                    ExprKind::Block(b, _)
                        if !b.targeted_by_break
                            && b.stmts.is_empty()
                            && let Some(next_init) = b.expr =>
                    {
                        init_expr = next_init;
                    },
                    ExprKind::Path(ref init_path) => {
                        let next_init_args =
                            EarlyBinder::bind(init_typeck.node_args(init_expr.hir_id)).instantiate(tcx, init_args);
                        match init_typeck.qpath_res(init_path, init_expr.hir_id) {
                            Res::Def(DefKind::Ctor(..), _) => return None,
                            Res::Def(DefKind::Const | DefKind::AssocConst, did)
                                if let Ok(val) = tcx.const_eval_resolve(
                                    typing_env,
                                    UnevaluatedConst::new(did, next_init_args),
                                    DUMMY_SP,
                                ) && let Ok(res) =
                                    self.is_non_freeze_val_borrowed(tcx, typing_env, init_typeck, src_expr, val) =>
                            {
                                return res;
                            },
                            Res::Def(DefKind::Const | DefKind::AssocConst, did)
                                if let Some((next_typeck, value)) =
                                    get_const_hir_value(tcx, typing_env, did, next_init_args) =>
                            {
                                init_typeck = next_typeck;
                                init_args = next_init_args;
                                init_expr = value;
                            },
                            // There's no more that we can read from the init expression. Switch to a
                            // type based check.
                            _ => {
                                return self.is_non_freeze_expr_borrowed(tcx, typing_env, typeck, src_expr);
                            },
                        }
                    },
                    _ => break,
                }
            }

            // Then a type check. Note we only check the type here as the result
            // gets cached.
            let ty = EarlyBinder::bind(typeck.expr_ty(src_expr)).instantiate(tcx, init_args);
            // Normalized as we need to check if this is an array later.
            let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);
            if self.is_ty_freeze(tcx, typing_env, ty).is_freeze() {
                return None;
            }

            // Finally reduce the init expression using the next use expression.
            let Some((_, Node::Expr(use_expr))) = parents.next() else {
                return None;
            };
            init_expr = match &use_expr.kind {
                ExprKind::Field(_, name) => match init_expr.kind {
                    ExprKind::Struct(_, fields, _)
                        if let Some(field) = fields.iter().find(|f| f.ident.name == name.name) =>
                    {
                        field.expr
                    },
                    ExprKind::Tup(fields)
                        if let Ok(idx) = name.as_str().parse::<usize>()
                            && let Some(field) = fields.get(idx) =>
                    {
                        field
                    },
                    ExprKind::Call(callee, args)
                        if let ExprKind::Path(callee_path) = &callee.kind
                            && matches!(
                                init_typeck.qpath_res(callee_path, callee.hir_id),
                                Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(_)
                            )
                            && let Ok(idx) = name.as_str().parse::<usize>()
                            && let Some(arg) = args.get(idx) =>
                    {
                        arg
                    },
                    // Revert to a type based check as we don't know the field's value.
                    _ => return self.is_non_freeze_expr_borrowed(tcx, typing_env, typeck, use_expr),
                },
                ExprKind::Index(_, idx, _) if ty.is_array() => match init_expr.kind {
                    ExprKind::Array(fields) => {
                        if let Some(Constant::Int(idx)) = ConstEvalCtxt::with_env(tcx, typing_env, typeck).eval(idx) {
                            // If the index is out of bounds it means the code
                            // unconditionally panics. In that case there is no borrow.
                            fields.get(idx as usize)?
                        } else {
                            // Unknown index, just run the check for all values.
                            return fields.iter().find_map(|f| {
                                self.is_non_freeze_init_borrowed(
                                    tcx,
                                    typing_env,
                                    typeck,
                                    use_expr,
                                    init_typeck,
                                    init_args,
                                    f,
                                )
                            });
                        }
                    },
                    // Just assume the index expression doesn't panic here.
                    ExprKind::Repeat(field, _) => field,
                    _ => return self.is_non_freeze_expr_borrowed(tcx, typing_env, typeck, use_expr),
                },
                ExprKind::AddrOf(..)
                    if !self.is_init_expr_freeze(tcx, typing_env, init_typeck, init_args, init_expr) =>
                {
                    return Some(BorrowSource::new(tcx, use_expr, BorrowCause::Borrow));
                },
                // All other expressions use the value.
                _ => return None,
            };
            src_expr = use_expr;
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for NonCopyConst<'tcx> {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Const(ident, .., body_id) = item.kind
            && !ident.is_special()
            && let ty = cx.tcx.type_of(item.owner_id).instantiate_identity()
            && match self.is_ty_freeze(cx.tcx, cx.typing_env(), ty) {
                IsFreeze::No => true,
                IsFreeze::Yes => false,
                IsFreeze::Maybe => match cx.tcx.const_eval_poly(item.owner_id.to_def_id()) {
                    Ok(val) if let Ok(is_freeze) = self.is_value_freeze(cx.tcx, cx.typing_env(), ty, val) => !is_freeze,
                    _ => !self.is_init_expr_freeze(
                        cx.tcx,
                        cx.typing_env(),
                        cx.tcx.typeck(item.owner_id),
                        GenericArgs::identity_for_item(cx.tcx, item.owner_id),
                        cx.tcx.hir_body(body_id).value,
                    ),
                },
            }
            && !item.span.in_external_macro(cx.sess().source_map())
            // Only needed when compiling `std`
            && !is_thread_local(cx, item)
        {
            span_lint_and_then(
                cx,
                DECLARE_INTERIOR_MUTABLE_CONST,
                ident.span,
                "named constant with interior mutability",
                |diag| {
                    let Some(sync_trait) = cx.tcx.lang_items().sync_trait() else {
                        return;
                    };
                    if implements_trait(cx, ty, sync_trait, &[]) {
                        diag.help("did you mean to make this a `static` item");
                    } else {
                        diag.help("did you mean to make this a `thread_local!` item");
                    }
                },
            );
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Const(_, body_id_opt) = item.kind
            && let ty = cx.tcx.type_of(item.owner_id).instantiate_identity()
            && match self.is_ty_freeze(cx.tcx, cx.typing_env(), ty) {
                IsFreeze::No => true,
                IsFreeze::Maybe if let Some(body_id) = body_id_opt => {
                    match cx.tcx.const_eval_poly(item.owner_id.to_def_id()) {
                        Ok(val) if let Ok(is_freeze) = self.is_value_freeze(cx.tcx, cx.typing_env(), ty, val) => {
                            !is_freeze
                        },
                        _ => !self.is_init_expr_freeze(
                            cx.tcx,
                            cx.typing_env(),
                            cx.tcx.typeck(item.owner_id),
                            GenericArgs::identity_for_item(cx.tcx, item.owner_id),
                            cx.tcx.hir_body(body_id).value,
                        ),
                    }
                },
                IsFreeze::Yes | IsFreeze::Maybe => false,
            }
            && !item.span.in_external_macro(cx.sess().source_map())
        {
            span_lint(
                cx,
                DECLARE_INTERIOR_MUTABLE_CONST,
                item.ident.span,
                "named constant with interior mutability",
            );
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Const(_, body_id) = item.kind
            && let ty = cx.tcx.type_of(item.owner_id).instantiate_identity()
            && match self.is_ty_freeze(cx.tcx, cx.typing_env(), ty) {
                IsFreeze::Yes => false,
                IsFreeze::No => {
                    // If this is a trait impl, check if the trait definition is the source
                    // of the cell.
                    if let Node::Item(parent_item) = cx.tcx.parent_hir_node(item.hir_id())
                        && let ItemKind::Impl(impl_block) = parent_item.kind
                        && let Some(of_trait) = impl_block.of_trait
                        && let Some(trait_id) = of_trait.trait_def_id()
                    {
                        // Replace all instances of `<Self as Trait>::AssocType` with the
                        // unit type and check again. If the result is the same then the
                        // trait definition is the cause.
                        let ty = (ReplaceAssocFolder {
                            tcx: cx.tcx,
                            trait_id,
                            self_ty: cx.tcx.type_of(parent_item.owner_id).instantiate_identity(),
                        })
                        .fold_ty(cx.tcx.type_of(item.owner_id).instantiate_identity());
                        // `ty` may not be normalizable, but that should be fine.
                        !self.is_ty_freeze(cx.tcx, cx.typing_env(), ty).is_not_freeze()
                    } else {
                        true
                    }
                },
                // Even if this is from a trait, there are values which don't have
                // interior mutability.
                IsFreeze::Maybe => match cx.tcx.const_eval_poly(item.owner_id.to_def_id()) {
                    Ok(val) if let Ok(is_freeze) = self.is_value_freeze(cx.tcx, cx.typing_env(), ty, val) => !is_freeze,
                    _ => !self.is_init_expr_freeze(
                        cx.tcx,
                        cx.typing_env(),
                        cx.tcx.typeck(item.owner_id),
                        GenericArgs::identity_for_item(cx.tcx, item.owner_id),
                        cx.tcx.hir_body(body_id).value,
                    ),
                },
            }
            && !item.span.in_external_macro(cx.sess().source_map())
        {
            span_lint(
                cx,
                DECLARE_INTERIOR_MUTABLE_CONST,
                item.ident.span,
                "named constant with interior mutability",
            );
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Path(qpath) = &e.kind
            && let typeck = cx.typeck_results()
            && let Res::Def(DefKind::Const | DefKind::AssocConst, did) = typeck.qpath_res(qpath, e.hir_id)
            // As of `1.80` constant contexts can't borrow any type with interior mutability
            && !is_in_const_context(cx)
            && !self.is_ty_freeze(cx.tcx, cx.typing_env(), typeck.expr_ty(e)).is_freeze()
            && let Some(borrow_src) = {
                // The extra block helps formatting a lot.
                if let Ok(val) = cx.tcx.const_eval_resolve(
                    cx.typing_env(),
                    UnevaluatedConst::new(did, typeck.node_args(e.hir_id)),
                    DUMMY_SP,
                ) && let Ok(src) = self.is_non_freeze_val_borrowed(cx.tcx, cx.typing_env(), typeck, e, val)
                {
                    src
                } else if let init_args = typeck.node_args(e.hir_id)
                    && let Some((init_typeck, init)) = get_const_hir_value(cx.tcx, cx.typing_env(), did, init_args)
                {
                    self.is_non_freeze_init_borrowed(cx.tcx, cx.typing_env(), typeck, e, init_typeck, init_args, init)
                } else {
                    self.is_non_freeze_expr_borrowed(cx.tcx, cx.typing_env(), typeck, e)
                }
            }
            && !borrow_src.expr.span.in_external_macro(cx.sess().source_map())
        {
            span_lint_and_then(
                cx,
                BORROW_INTERIOR_MUTABLE_CONST,
                borrow_src.expr.span,
                "borrow of a named constant with interior mutability",
                |diag| {
                    if let Some(note) = borrow_src.cause.note() {
                        diag.note(note);
                    }
                    diag.help("this lint can be silenced by assigning the value to a local variable before borrowing");
                },
            );
        }
    }
}

struct ReplaceAssocFolder<'tcx> {
    tcx: TyCtxt<'tcx>,
    trait_id: DefId,
    self_ty: Ty<'tcx>,
}
impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceAssocFolder<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Alias(AliasTyKind::Projection, ty) = ty.kind()
            && ty.trait_def_id(self.tcx) == self.trait_id
            && ty.self_ty() == self.self_ty
        {
            self.tcx.types.unit
        } else {
            ty.super_fold_with(self)
        }
    }
}

fn is_thread_local(cx: &LateContext<'_>, it: &Item<'_>) -> bool {
    macro_backtrace(it.span).any(|macro_call| {
        matches!(
            cx.tcx.get_diagnostic_name(macro_call.def_id),
            Some(sym::thread_local_macro)
        )
    })
}

/// Checks if the adjustment causes a borrow of the original value. Returns
/// `None` if the value is consumed instead of borrowed.
fn does_adjust_borrow(adjust: &Adjustment<'_>) -> Option<BorrowCause> {
    match adjust.kind {
        Adjust::Borrow(_) => Some(BorrowCause::AutoBorrow),
        // Custom deref calls `<T as Deref>::deref(&x)` resulting in a borrow.
        Adjust::Deref(Some(_)) => Some(BorrowCause::AutoDeref),
        // All other adjustments read the value.
        _ => None,
    }
}

/// Attempts to get the value of a constant as a HIR expression. Also gets the
/// `TypeckResults` associated with the constant's body.
fn get_const_hir_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    did: DefId,
    args: GenericArgsRef<'tcx>,
) -> Option<(&'tcx TypeckResults<'tcx>, &'tcx Expr<'tcx>)> {
    let did = did.as_local()?;
    let (did, body_id) = match tcx.hir_node(tcx.local_def_id_to_hir_id(did)) {
        Node::Item(item) if let ItemKind::Const(.., body_id) = item.kind => (did, body_id),
        Node::ImplItem(item) if let ImplItemKind::Const(.., body_id) = item.kind => (did, body_id),
        Node::TraitItem(_)
            if let Ok(Some(inst)) = Instance::try_resolve(tcx, typing_env, did.into(), args)
                && let Some(did) = inst.def_id().as_local() =>
        {
            match tcx.hir_node(tcx.local_def_id_to_hir_id(did)) {
                Node::ImplItem(item) if let ImplItemKind::Const(.., body_id) = item.kind => (did, body_id),
                Node::TraitItem(item) if let TraitItemKind::Const(.., Some(body_id)) = item.kind => (did, body_id),
                _ => return None,
            }
        },
        _ => return None,
    };
    Some((tcx.typeck(did), tcx.hir_body(body_id).value))
}
