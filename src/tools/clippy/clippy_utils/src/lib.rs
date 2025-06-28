#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(macro_metavar_expr)]
#![feature(never_type)]
#![feature(rustc_private)]
#![feature(assert_matches)]
#![feature(unwrap_infallible)]
#![feature(array_windows)]
#![recursion_limit = "512"]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    rustc::diagnostic_outside_of_impl,
    rustc::untranslatable_diagnostic
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications,
    rustc::internal
)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
extern crate indexmap;
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_attr_data_structures;
extern crate rustc_attr_parsing;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
// The `rustc_driver` crate seems to be required in order to use the `rust_ast` crate.
#[allow(unused_extern_crates)]
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_hir_analysis;
extern crate rustc_hir_typeck;
extern crate rustc_index;
extern crate rustc_infer;
extern crate rustc_lexer;
extern crate rustc_lint;
extern crate rustc_middle;
extern crate rustc_mir_dataflow;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_trait_selection;
extern crate smallvec;

pub mod ast_utils;
pub mod attrs;
mod check_proc_macro;
pub mod comparisons;
pub mod consts;
pub mod diagnostics;
pub mod eager_or_lazy;
pub mod higher;
mod hir_utils;
pub mod macros;
pub mod mir;
pub mod msrvs;
pub mod numeric_literal;
pub mod paths;
pub mod ptr;
pub mod qualify_min_const_fn;
pub mod source;
pub mod str_utils;
pub mod sugg;
pub mod sym;
pub mod ty;
pub mod usage;
pub mod visitors;

pub use self::attrs::*;
pub use self::check_proc_macro::{is_from_proc_macro, is_span_if, is_span_match};
pub use self::hir_utils::{
    HirEqInterExpr, SpanlessEq, SpanlessHash, both, count_eq, eq_expr_value, hash_expr, hash_stmt, is_bool, over,
};

use core::mem;
use core::ops::ControlFlow;
use std::collections::hash_map::Entry;
use std::iter::{once, repeat_n};
use std::sync::{Mutex, MutexGuard, OnceLock};

use itertools::Itertools;
use rustc_abi::Integer;
use rustc_ast::ast::{self, LitKind, RangeLimits};
use rustc_attr_data_structures::{AttributeKind, find_attr};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::packed::Pu128;
use rustc_data_structures::unhash::UnindexMap;
use rustc_hir::LangItem::{OptionNone, OptionSome, ResultErr, ResultOk};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId, LocalModDefId};
use rustc_hir::definitions::{DefPath, DefPathData};
use rustc_hir::hir_id::{HirIdMap, HirIdSet};
use rustc_hir::intravisit::{FnKind, Visitor, walk_expr};
use rustc_hir::{
    self as hir, Arm, BindingMode, Block, BlockCheckMode, Body, ByRef, Closure, ConstArgKind, CoroutineDesugaring,
    CoroutineKind, Destination, Expr, ExprField, ExprKind, FnDecl, FnRetTy, GenericArg, GenericArgs, HirId, Impl,
    ImplItem, ImplItemKind, Item, ItemKind, LangItem, LetStmt, MatchSource, Mutability, Node, OwnerId, OwnerNode,
    Param, Pat, PatExpr, PatExprKind, PatKind, Path, PathSegment, QPath, Stmt, StmtKind, TraitFn, TraitItem,
    TraitItemKind, TraitRef, TyKind, UnOp, def,
};
use rustc_lexer::{TokenKind, tokenize};
use rustc_lint::{LateContext, Level, Lint, LintContext};
use rustc_middle::hir::nested_filter;
use rustc_middle::hir::place::PlaceBase;
use rustc_middle::lint::LevelAndSource;
use rustc_middle::mir::{AggregateKind, Operand, RETURN_PLACE, Rvalue, StatementKind, TerminatorKind};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{
    self as rustc_ty, Binder, BorrowKind, ClosureKind, EarlyBinder, GenericArgKind, GenericArgsRef, IntTy, Ty, TyCtxt,
    TypeFlags, TypeVisitableExt, UintTy, UpvarCapture,
};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::{Ident, Symbol, kw};
use rustc_span::{InnerSpan, Span};
use source::{SpanRangeExt, walk_span_to_context};
use visitors::{Visitable, for_each_unconsumed_temporary};

use crate::consts::{ConstEvalCtxt, Constant, mir_to_const};
use crate::higher::Range;
use crate::ty::{adt_and_variant_of_res, can_partially_move_ty, expr_sig, is_copy, is_recursively_primitive_type};
use crate::visitors::for_each_expr_without_closures;

#[macro_export]
macro_rules! extract_msrv_attr {
    () => {
        fn check_attributes(&mut self, cx: &rustc_lint::EarlyContext<'_>, attrs: &[rustc_ast::ast::Attribute]) {
            let sess = rustc_lint::LintContext::sess(cx);
            self.msrv.check_attributes(sess, attrs);
        }

        fn check_attributes_post(&mut self, cx: &rustc_lint::EarlyContext<'_>, attrs: &[rustc_ast::ast::Attribute]) {
            let sess = rustc_lint::LintContext::sess(cx);
            self.msrv.check_attributes_post(sess, attrs);
        }
    };
}

/// If the given expression is a local binding, find the initializer expression.
/// If that initializer expression is another local binding, find its initializer again.
///
/// This process repeats as long as possible (but usually no more than once). Initializer
/// expressions with adjustments are ignored. If this is not desired, use [`find_binding_init`]
/// instead.
///
/// Examples:
/// ```no_run
/// let abc = 1;
/// //        ^ output
/// let def = abc;
/// dbg!(def);
/// //   ^^^ input
///
/// // or...
/// let abc = 1;
/// let def = abc + 2;
/// //        ^^^^^^^ output
/// dbg!(def);
/// //   ^^^ input
/// ```
pub fn expr_or_init<'a, 'b, 'tcx: 'b>(cx: &LateContext<'tcx>, mut expr: &'a Expr<'b>) -> &'a Expr<'b> {
    while let Some(init) = path_to_local(expr)
        .and_then(|id| find_binding_init(cx, id))
        .filter(|init| cx.typeck_results().expr_adjustments(init).is_empty())
    {
        expr = init;
    }
    expr
}

/// Finds the initializer expression for a local binding. Returns `None` if the binding is mutable.
///
/// By only considering immutable bindings, we guarantee that the returned expression represents the
/// value of the binding wherever it is referenced.
///
/// Example: For `let x = 1`, if the `HirId` of `x` is provided, the `Expr` `1` is returned.
/// Note: If you have an expression that references a binding `x`, use `path_to_local` to get the
/// canonical binding `HirId`.
pub fn find_binding_init<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId) -> Option<&'tcx Expr<'tcx>> {
    if let Node::Pat(pat) = cx.tcx.hir_node(hir_id)
        && matches!(pat.kind, PatKind::Binding(BindingMode::NONE, ..))
        && let Node::LetStmt(local) = cx.tcx.parent_hir_node(hir_id)
    {
        return local.init;
    }
    None
}

/// Checks if the given local has an initializer or is from something other than a `let` statement
///
/// e.g. returns true for `x` in `fn f(x: usize) { .. }` and `let x = 1;` but false for `let x;`
pub fn local_is_initialized(cx: &LateContext<'_>, local: HirId) -> bool {
    for (_, node) in cx.tcx.hir_parent_iter(local) {
        match node {
            Node::Pat(..) | Node::PatField(..) => {},
            Node::LetStmt(let_stmt) => return let_stmt.init.is_some(),
            _ => return true,
        }
    }

    false
}

/// Checks if we are currently in a const context (e.g. `const fn`, `static`/`const` initializer).
///
/// The current context is determined based on the current body which is set before calling a lint's
/// entry point (any function on `LateLintPass`). If you need to check in a different context use
/// `tcx.hir_is_inside_const_context(_)`.
///
/// Do not call this unless the `LateContext` has an enclosing body. For release build this case
/// will safely return `false`, but debug builds will ICE. Note that `check_expr`, `check_block`,
/// `check_pat` and a few other entry points will always have an enclosing body. Some entry points
/// like `check_path` or `check_ty` may or may not have one.
pub fn is_in_const_context(cx: &LateContext<'_>) -> bool {
    debug_assert!(cx.enclosing_body.is_some(), "`LateContext` has no enclosing body");
    cx.enclosing_body.is_some_and(|id| {
        cx.tcx
            .hir_body_const_context(cx.tcx.hir_body_owner_def_id(id))
            .is_some()
    })
}

/// Returns `true` if the given `HirId` is inside an always constant context.
///
/// This context includes:
///  * const/static items
///  * const blocks (or inline consts)
///  * associated constants
pub fn is_inside_always_const_context(tcx: TyCtxt<'_>, hir_id: HirId) -> bool {
    use rustc_hir::ConstContext::{Const, ConstFn, Static};
    let Some(ctx) = tcx.hir_body_const_context(tcx.hir_enclosing_body_owner(hir_id)) else {
        return false;
    };
    match ctx {
        ConstFn => false,
        Static(_) | Const { inline: _ } => true,
    }
}

/// Checks if a `Res` refers to a constructor of a `LangItem`
/// For example, use this to check whether a function call or a pattern is `Some(..)`.
pub fn is_res_lang_ctor(cx: &LateContext<'_>, res: Res, lang_item: LangItem) -> bool {
    if let Res::Def(DefKind::Ctor(..), id) = res
        && let Some(lang_id) = cx.tcx.lang_items().get(lang_item)
        && let Some(id) = cx.tcx.opt_parent(id)
    {
        id == lang_id
    } else {
        false
    }
}

/// Checks if `{ctor_call_id}(...)` is `{enum_item}::{variant_name}(...)`.
pub fn is_enum_variant_ctor(
    cx: &LateContext<'_>,
    enum_item: Symbol,
    variant_name: Symbol,
    ctor_call_id: DefId,
) -> bool {
    let Some(enum_def_id) = cx.tcx.get_diagnostic_item(enum_item) else {
        return false;
    };

    let variants = cx.tcx.adt_def(enum_def_id).variants().iter();
    variants
        .filter(|variant| variant.name == variant_name)
        .filter_map(|variant| variant.ctor.as_ref())
        .any(|(_, ctor_def_id)| *ctor_def_id == ctor_call_id)
}

/// Checks if the `DefId` matches the given diagnostic item or it's constructor.
pub fn is_diagnostic_item_or_ctor(cx: &LateContext<'_>, did: DefId, item: Symbol) -> bool {
    let did = match cx.tcx.def_kind(did) {
        DefKind::Ctor(..) => cx.tcx.parent(did),
        // Constructors for types in external crates seem to have `DefKind::Variant`
        DefKind::Variant => match cx.tcx.opt_parent(did) {
            Some(did) if matches!(cx.tcx.def_kind(did), DefKind::Variant) => did,
            _ => did,
        },
        _ => did,
    };

    cx.tcx.is_diagnostic_item(item, did)
}

/// Checks if the `DefId` matches the given `LangItem` or it's constructor.
pub fn is_lang_item_or_ctor(cx: &LateContext<'_>, did: DefId, item: LangItem) -> bool {
    let did = match cx.tcx.def_kind(did) {
        DefKind::Ctor(..) => cx.tcx.parent(did),
        // Constructors for types in external crates seem to have `DefKind::Variant`
        DefKind::Variant => match cx.tcx.opt_parent(did) {
            Some(did) if matches!(cx.tcx.def_kind(did), DefKind::Variant) => did,
            _ => did,
        },
        _ => did,
    };

    cx.tcx.lang_items().get(item) == Some(did)
}

pub fn is_unit_expr(expr: &Expr<'_>) -> bool {
    matches!(
        expr.kind,
        ExprKind::Block(
            Block {
                stmts: [],
                expr: None,
                ..
            },
            _
        ) | ExprKind::Tup([])
    )
}

/// Checks if given pattern is a wildcard (`_`)
pub fn is_wild(pat: &Pat<'_>) -> bool {
    matches!(pat.kind, PatKind::Wild)
}

// Checks if arm has the form `None => None`
pub fn is_none_arm(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
    matches!(
        arm.pat.kind,
        PatKind::Expr(PatExpr { kind: PatExprKind::Path(qpath), .. })
            if is_res_lang_ctor(cx, cx.qpath_res(qpath, arm.pat.hir_id), OptionNone)
    )
}

/// Checks if the given `QPath` belongs to a type alias.
pub fn is_ty_alias(qpath: &QPath<'_>) -> bool {
    match *qpath {
        QPath::Resolved(_, path) => matches!(path.res, Res::Def(DefKind::TyAlias | DefKind::AssocTy, ..)),
        QPath::TypeRelative(ty, _) if let TyKind::Path(qpath) = ty.kind => is_ty_alias(&qpath),
        _ => false,
    }
}

/// Checks if the given method call expression calls an inherent method.
pub fn is_inherent_method_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
        cx.tcx.trait_of_item(method_id).is_none()
    } else {
        false
    }
}

/// Checks if a method is defined in an impl of a diagnostic item
pub fn is_diag_item_method(cx: &LateContext<'_>, def_id: DefId, diag_item: Symbol) -> bool {
    if let Some(impl_did) = cx.tcx.impl_of_method(def_id)
        && let Some(adt) = cx.tcx.type_of(impl_did).instantiate_identity().ty_adt_def()
    {
        return cx.tcx.is_diagnostic_item(diag_item, adt.did());
    }
    false
}

/// Checks if a method is in a diagnostic item trait
pub fn is_diag_trait_item(cx: &LateContext<'_>, def_id: DefId, diag_item: Symbol) -> bool {
    if let Some(trait_did) = cx.tcx.trait_of_item(def_id) {
        return cx.tcx.is_diagnostic_item(diag_item, trait_did);
    }
    false
}

/// Checks if the method call given in `expr` belongs to the given trait.
pub fn is_trait_method(cx: &LateContext<'_>, expr: &Expr<'_>, diag_item: Symbol) -> bool {
    cx.typeck_results()
        .type_dependent_def_id(expr.hir_id)
        .is_some_and(|did| is_diag_trait_item(cx, did, diag_item))
}

/// Checks if the `def_id` belongs to a function that is part of a trait impl.
pub fn is_def_id_trait_method(cx: &LateContext<'_>, def_id: LocalDefId) -> bool {
    if let Node::Item(item) = cx.tcx.parent_hir_node(cx.tcx.local_def_id_to_hir_id(def_id))
        && let ItemKind::Impl(imp) = item.kind
    {
        imp.of_trait.is_some()
    } else {
        false
    }
}

/// Checks if the given expression is a path referring an item on the trait
/// that is marked with the given diagnostic item.
///
/// For checking method call expressions instead of path expressions, use
/// [`is_trait_method`].
///
/// For example, this can be used to find if an expression like `u64::default`
/// refers to an item of the trait `Default`, which is associated with the
/// `diag_item` of `sym::Default`.
pub fn is_trait_item(cx: &LateContext<'_>, expr: &Expr<'_>, diag_item: Symbol) -> bool {
    if let ExprKind::Path(ref qpath) = expr.kind {
        cx.qpath_res(qpath, expr.hir_id)
            .opt_def_id()
            .is_some_and(|def_id| is_diag_trait_item(cx, def_id, diag_item))
    } else {
        false
    }
}

pub fn last_path_segment<'tcx>(path: &QPath<'tcx>) -> &'tcx PathSegment<'tcx> {
    match *path {
        QPath::Resolved(_, path) => path.segments.last().expect("A path must have at least one segment"),
        QPath::TypeRelative(_, seg) => seg,
        QPath::LangItem(..) => panic!("last_path_segment: lang item has no path segments"),
    }
}

pub fn qpath_generic_tys<'tcx>(qpath: &QPath<'tcx>) -> impl Iterator<Item = &'tcx hir::Ty<'tcx>> {
    last_path_segment(qpath)
        .args
        .map_or(&[][..], |a| a.args)
        .iter()
        .filter_map(|a| match a {
            GenericArg::Type(ty) => Some(ty.as_unambig_ty()),
            _ => None,
        })
}

/// If `maybe_path` is a path node which resolves to an item, resolves it to a `DefId` and checks if
/// it matches the given lang item.
pub fn is_path_lang_item<'tcx>(cx: &LateContext<'_>, maybe_path: &impl MaybePath<'tcx>, lang_item: LangItem) -> bool {
    path_def_id(cx, maybe_path).is_some_and(|id| cx.tcx.lang_items().get(lang_item) == Some(id))
}

/// If `maybe_path` is a path node which resolves to an item, resolves it to a `DefId` and checks if
/// it matches the given diagnostic item.
pub fn is_path_diagnostic_item<'tcx>(
    cx: &LateContext<'_>,
    maybe_path: &impl MaybePath<'tcx>,
    diag_item: Symbol,
) -> bool {
    path_def_id(cx, maybe_path).is_some_and(|id| cx.tcx.is_diagnostic_item(diag_item, id))
}

/// If the expression is a path to a local, returns the canonical `HirId` of the local.
pub fn path_to_local(expr: &Expr<'_>) -> Option<HirId> {
    if let ExprKind::Path(QPath::Resolved(None, path)) = expr.kind
        && let Res::Local(id) = path.res
    {
        return Some(id);
    }
    None
}

/// Returns true if the expression is a path to a local with the specified `HirId`.
/// Use this function to see if an expression matches a function argument or a match binding.
pub fn path_to_local_id(expr: &Expr<'_>, id: HirId) -> bool {
    path_to_local(expr) == Some(id)
}

pub trait MaybePath<'hir> {
    fn hir_id(&self) -> HirId;
    fn qpath_opt(&self) -> Option<&QPath<'hir>>;
}

macro_rules! maybe_path {
    ($ty:ident, $kind:ident) => {
        impl<'hir> MaybePath<'hir> for hir::$ty<'hir> {
            fn hir_id(&self) -> HirId {
                self.hir_id
            }
            fn qpath_opt(&self) -> Option<&QPath<'hir>> {
                match &self.kind {
                    hir::$kind::Path(qpath) => Some(qpath),
                    _ => None,
                }
            }
        }
    };
}
maybe_path!(Expr, ExprKind);
impl<'hir> MaybePath<'hir> for Pat<'hir> {
    fn hir_id(&self) -> HirId {
        self.hir_id
    }
    fn qpath_opt(&self) -> Option<&QPath<'hir>> {
        match &self.kind {
            PatKind::Expr(PatExpr {
                kind: PatExprKind::Path(qpath),
                ..
            }) => Some(qpath),
            _ => None,
        }
    }
}
maybe_path!(Ty, TyKind);

/// If `maybe_path` is a path node, resolves it, otherwise returns `Res::Err`
pub fn path_res<'tcx>(cx: &LateContext<'_>, maybe_path: &impl MaybePath<'tcx>) -> Res {
    match maybe_path.qpath_opt() {
        None => Res::Err,
        Some(qpath) => cx.qpath_res(qpath, maybe_path.hir_id()),
    }
}

/// If `maybe_path` is a path node which resolves to an item, retrieves the item ID
pub fn path_def_id<'tcx>(cx: &LateContext<'_>, maybe_path: &impl MaybePath<'tcx>) -> Option<DefId> {
    path_res(cx, maybe_path).opt_def_id()
}

/// Gets the `hir::TraitRef` of the trait the given method is implemented for.
///
/// Use this if you want to find the `TraitRef` of the `Add` trait in this example:
///
/// ```no_run
/// struct Point(isize, isize);
///
/// impl std::ops::Add for Point {
///     type Output = Self;
///
///     fn add(self, other: Self) -> Self {
///         Point(0, 0)
///     }
/// }
/// ```
pub fn trait_ref_of_method<'tcx>(cx: &LateContext<'tcx>, owner: OwnerId) -> Option<&'tcx TraitRef<'tcx>> {
    if let Node::Item(item) = cx.tcx.hir_node(cx.tcx.hir_owner_parent(owner))
        && let ItemKind::Impl(impl_) = &item.kind
    {
        return impl_.of_trait.as_ref();
    }
    None
}

/// This method will return tuple of projection stack and root of the expression,
/// used in `can_mut_borrow_both`.
///
/// For example, if `e` represents the `v[0].a.b[x]`
/// this method will return a tuple, composed of a `Vec`
/// containing the `Expr`s for `v[0], v[0].a, v[0].a.b, v[0].a.b[x]`
/// and an `Expr` for root of them, `v`
fn projection_stack<'a, 'hir>(mut e: &'a Expr<'hir>) -> (Vec<&'a Expr<'hir>>, &'a Expr<'hir>) {
    let mut result = vec![];
    let root = loop {
        match e.kind {
            ExprKind::Index(ep, _, _) | ExprKind::Field(ep, _) => {
                result.push(e);
                e = ep;
            },
            _ => break e,
        }
    };
    result.reverse();
    (result, root)
}

/// Gets the mutability of the custom deref adjustment, if any.
pub fn expr_custom_deref_adjustment(cx: &LateContext<'_>, e: &Expr<'_>) -> Option<Mutability> {
    cx.typeck_results()
        .expr_adjustments(e)
        .iter()
        .find_map(|a| match a.kind {
            Adjust::Deref(Some(d)) => Some(Some(d.mutbl)),
            Adjust::Deref(None) => None,
            _ => Some(None),
        })
        .and_then(|x| x)
}

/// Checks if two expressions can be mutably borrowed simultaneously
/// and they aren't dependent on borrowing same thing twice
pub fn can_mut_borrow_both(cx: &LateContext<'_>, e1: &Expr<'_>, e2: &Expr<'_>) -> bool {
    let (s1, r1) = projection_stack(e1);
    let (s2, r2) = projection_stack(e2);
    if !eq_expr_value(cx, r1, r2) {
        return true;
    }
    if expr_custom_deref_adjustment(cx, r1).is_some() || expr_custom_deref_adjustment(cx, r2).is_some() {
        return false;
    }

    for (x1, x2) in s1.iter().zip(s2.iter()) {
        if expr_custom_deref_adjustment(cx, x1).is_some() || expr_custom_deref_adjustment(cx, x2).is_some() {
            return false;
        }

        match (&x1.kind, &x2.kind) {
            (ExprKind::Field(_, i1), ExprKind::Field(_, i2)) => {
                if i1 != i2 {
                    return true;
                }
            },
            (ExprKind::Index(_, i1, _), ExprKind::Index(_, i2, _)) => {
                if !eq_expr_value(cx, i1, i2) {
                    return false;
                }
            },
            _ => return false,
        }
    }
    false
}

/// Returns true if the `def_id` associated with the `path` is recognized as a "default-equivalent"
/// constructor from the std library
fn is_default_equivalent_ctor(cx: &LateContext<'_>, def_id: DefId, path: &QPath<'_>) -> bool {
    let std_types_symbols = &[
        sym::Vec,
        sym::VecDeque,
        sym::LinkedList,
        sym::HashMap,
        sym::BTreeMap,
        sym::HashSet,
        sym::BTreeSet,
        sym::BinaryHeap,
    ];

    if let QPath::TypeRelative(_, method) = path
        && method.ident.name == sym::new
        && let Some(impl_did) = cx.tcx.impl_of_method(def_id)
        && let Some(adt) = cx.tcx.type_of(impl_did).instantiate_identity().ty_adt_def()
    {
        return std_types_symbols.iter().any(|&symbol| {
            cx.tcx.is_diagnostic_item(symbol, adt.did()) || Some(adt.did()) == cx.tcx.lang_items().string()
        });
    }
    false
}

/// Returns true if the expr is equal to `Default::default` when evaluated.
pub fn is_default_equivalent_call(
    cx: &LateContext<'_>,
    repl_func: &Expr<'_>,
    whole_call_expr: Option<&Expr<'_>>,
) -> bool {
    if let ExprKind::Path(ref repl_func_qpath) = repl_func.kind
        && let Some(repl_def_id) = cx.qpath_res(repl_func_qpath, repl_func.hir_id).opt_def_id()
        && (is_diag_trait_item(cx, repl_def_id, sym::Default)
            || is_default_equivalent_ctor(cx, repl_def_id, repl_func_qpath))
    {
        return true;
    }

    // Get the type of the whole method call expression, find the exact method definition, look at
    // its body and check if it is similar to the corresponding `Default::default()` body.
    let Some(e) = whole_call_expr else { return false };
    let Some(default_fn_def_id) = cx.tcx.get_diagnostic_item(sym::default_fn) else {
        return false;
    };
    let Some(ty) = cx.tcx.typeck(e.hir_id.owner.def_id).expr_ty_adjusted_opt(e) else {
        return false;
    };
    let args = rustc_ty::GenericArgs::for_item(cx.tcx, default_fn_def_id, |param, _| {
        if let rustc_ty::GenericParamDefKind::Lifetime = param.kind {
            cx.tcx.lifetimes.re_erased.into()
        } else if param.index == 0 && param.name == kw::SelfUpper {
            ty.into()
        } else {
            param.to_error(cx.tcx)
        }
    });
    let instance = rustc_ty::Instance::try_resolve(cx.tcx, cx.typing_env(), default_fn_def_id, args);

    let Ok(Some(instance)) = instance else { return false };
    if let rustc_ty::InstanceKind::Item(def) = instance.def
        && !cx.tcx.is_mir_available(def)
    {
        return false;
    }
    let ExprKind::Path(ref repl_func_qpath) = repl_func.kind else {
        return false;
    };
    let Some(repl_def_id) = cx.qpath_res(repl_func_qpath, repl_func.hir_id).opt_def_id() else {
        return false;
    };

    // Get the MIR Body for the `<Ty as Default>::default()` function.
    // If it is a value or call (either fn or ctor), we compare its `DefId` against the one for the
    // resolution of the expression we had in the path. This lets us identify, for example, that
    // the body of `<Vec<T> as Default>::default()` is a `Vec::new()`, and the field was being
    // initialized to `Vec::new()` as well.
    let body = cx.tcx.instance_mir(instance.def);
    for block_data in body.basic_blocks.iter() {
        if block_data.statements.len() == 1
            && let StatementKind::Assign(assign) = &block_data.statements[0].kind
            && assign.0.local == RETURN_PLACE
            && let Rvalue::Aggregate(kind, _places) = &assign.1
            && let AggregateKind::Adt(did, variant_index, _, _, _) = &**kind
            && let def = cx.tcx.adt_def(did)
            && let variant = &def.variant(*variant_index)
            && variant.fields.is_empty()
            && let Some((_, did)) = variant.ctor
            && did == repl_def_id
        {
            return true;
        } else if block_data.statements.is_empty()
            && let Some(term) = &block_data.terminator
        {
            match &term.kind {
                TerminatorKind::Call {
                    func: Operand::Constant(c),
                    ..
                } if let rustc_ty::FnDef(did, _args) = c.ty().kind()
                    && *did == repl_def_id =>
                {
                    return true;
                },
                TerminatorKind::TailCall {
                    func: Operand::Constant(c),
                    ..
                } if let rustc_ty::FnDef(did, _args) = c.ty().kind()
                    && *did == repl_def_id =>
                {
                    return true;
                },
                _ => {},
            }
        }
    }
    false
}

/// Returns true if the expr is equal to `Default::default()` of its type when evaluated.
///
/// It doesn't cover all cases, like struct literals, but it is a close approximation.
pub fn is_default_equivalent(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    match &e.kind {
        ExprKind::Lit(lit) => match lit.node {
            LitKind::Bool(false) | LitKind::Int(Pu128(0), _) => true,
            LitKind::Str(s, _) => s.is_empty(),
            _ => false,
        },
        ExprKind::Tup(items) | ExprKind::Array(items) => items.iter().all(|x| is_default_equivalent(cx, x)),
        ExprKind::Repeat(x, len) => {
            if let ConstArgKind::Anon(anon_const) = len.kind
                && let ExprKind::Lit(const_lit) = cx.tcx.hir_body(anon_const.body).value.kind
                && let LitKind::Int(v, _) = const_lit.node
                && v <= 32
                && is_default_equivalent(cx, x)
            {
                true
            } else {
                false
            }
        },
        ExprKind::Call(repl_func, []) => is_default_equivalent_call(cx, repl_func, Some(e)),
        ExprKind::Call(from_func, [arg]) => is_default_equivalent_from(cx, from_func, arg),
        ExprKind::Path(qpath) => is_res_lang_ctor(cx, cx.qpath_res(qpath, e.hir_id), OptionNone),
        ExprKind::AddrOf(rustc_hir::BorrowKind::Ref, _, expr) => matches!(expr.kind, ExprKind::Array([])),
        ExprKind::Block(Block { stmts: [], expr, .. }, _) => expr.is_some_and(|e| is_default_equivalent(cx, e)),
        _ => false,
    }
}

fn is_default_equivalent_from(cx: &LateContext<'_>, from_func: &Expr<'_>, arg: &Expr<'_>) -> bool {
    if let ExprKind::Path(QPath::TypeRelative(ty, seg)) = from_func.kind
        && seg.ident.name == sym::from
    {
        match arg.kind {
            ExprKind::Lit(hir::Lit {
                node: LitKind::Str(sym, _),
                ..
            }) => return sym.is_empty() && is_path_lang_item(cx, ty, LangItem::String),
            ExprKind::Array([]) => return is_path_diagnostic_item(cx, ty, sym::Vec),
            ExprKind::Repeat(_, len) => {
                if let ConstArgKind::Anon(anon_const) = len.kind
                    && let ExprKind::Lit(const_lit) = cx.tcx.hir_body(anon_const.body).value.kind
                    && let LitKind::Int(v, _) = const_lit.node
                {
                    return v == 0 && is_path_diagnostic_item(cx, ty, sym::Vec);
                }
            },
            _ => (),
        }
    }
    false
}

/// Checks if the top level expression can be moved into a closure as is.
/// Currently checks for:
/// * Break/Continue outside the given loop HIR ids.
/// * Yield/Return statements.
/// * Inline assembly.
/// * Usages of a field of a local where the type of the local can be partially moved.
///
/// For example, given the following function:
///
/// ```no_run
/// fn f<'a>(iter: &mut impl Iterator<Item = (usize, &'a mut String)>) {
///     for item in iter {
///         let s = item.1;
///         if item.0 > 10 {
///             continue;
///         } else {
///             s.clear();
///         }
///     }
/// }
/// ```
///
/// When called on the expression `item.0` this will return false unless the local `item` is in the
/// `ignore_locals` set. The type `(usize, &mut String)` can have the second element moved, so it
/// isn't always safe to move into a closure when only a single field is needed.
///
/// When called on the `continue` expression this will return false unless the outer loop expression
/// is in the `loop_ids` set.
///
/// Note that this check is not recursive, so passing the `if` expression will always return true
/// even though sub-expressions might return false.
pub fn can_move_expr_to_closure_no_visit<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    loop_ids: &[HirId],
    ignore_locals: &HirIdSet,
) -> bool {
    match expr.kind {
        ExprKind::Break(Destination { target_id: Ok(id), .. }, _)
        | ExprKind::Continue(Destination { target_id: Ok(id), .. })
            if loop_ids.contains(&id) =>
        {
            true
        },
        ExprKind::Break(..)
        | ExprKind::Continue(_)
        | ExprKind::Ret(_)
        | ExprKind::Yield(..)
        | ExprKind::InlineAsm(_) => false,
        // Accessing a field of a local value can only be done if the type isn't
        // partially moved.
        ExprKind::Field(
            &Expr {
                hir_id,
                kind:
                    ExprKind::Path(QPath::Resolved(
                        _,
                        Path {
                            res: Res::Local(local_id),
                            ..
                        },
                    )),
                ..
            },
            _,
        ) if !ignore_locals.contains(local_id) && can_partially_move_ty(cx, cx.typeck_results().node_type(hir_id)) => {
            // TODO: check if the local has been partially moved. Assume it has for now.
            false
        },
        _ => true,
    }
}

/// How a local is captured by a closure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    Value,
    Use,
    Ref(Mutability),
}
impl CaptureKind {
    pub fn is_imm_ref(self) -> bool {
        self == Self::Ref(Mutability::Not)
    }
}
impl std::ops::BitOr for CaptureKind {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (CaptureKind::Value, _) | (_, CaptureKind::Value) => CaptureKind::Value,
            (CaptureKind::Use, _) | (_, CaptureKind::Use) => CaptureKind::Use,
            (CaptureKind::Ref(Mutability::Mut), CaptureKind::Ref(_))
            | (CaptureKind::Ref(_), CaptureKind::Ref(Mutability::Mut)) => CaptureKind::Ref(Mutability::Mut),
            (CaptureKind::Ref(Mutability::Not), CaptureKind::Ref(Mutability::Not)) => CaptureKind::Ref(Mutability::Not),
        }
    }
}
impl std::ops::BitOrAssign for CaptureKind {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

/// Given an expression referencing a local, determines how it would be captured in a closure.
///
/// Note as this will walk up to parent expressions until the capture can be determined it should
/// only be used while making a closure somewhere a value is consumed. e.g. a block, match arm, or
/// function argument (other than a receiver).
pub fn capture_local_usage(cx: &LateContext<'_>, e: &Expr<'_>) -> CaptureKind {
    fn pat_capture_kind(cx: &LateContext<'_>, pat: &Pat<'_>) -> CaptureKind {
        let mut capture = CaptureKind::Ref(Mutability::Not);
        pat.each_binding_or_first(&mut |_, id, span, _| match cx
            .typeck_results()
            .extract_binding_mode(cx.sess(), id, span)
            .0
        {
            ByRef::No if !is_copy(cx, cx.typeck_results().node_type(id)) => {
                capture = CaptureKind::Value;
            },
            ByRef::Yes(Mutability::Mut) if capture != CaptureKind::Value => {
                capture = CaptureKind::Ref(Mutability::Mut);
            },
            _ => (),
        });
        capture
    }

    debug_assert!(matches!(
        e.kind,
        ExprKind::Path(QPath::Resolved(None, Path { res: Res::Local(_), .. }))
    ));

    let mut child_id = e.hir_id;
    let mut capture = CaptureKind::Value;
    let mut capture_expr_ty = e;

    for (parent_id, parent) in cx.tcx.hir_parent_iter(e.hir_id) {
        if let [
            Adjustment {
                kind: Adjust::Deref(_) | Adjust::Borrow(AutoBorrow::Ref(..)),
                target,
            },
            ref adjust @ ..,
        ] = *cx
            .typeck_results()
            .adjustments()
            .get(child_id)
            .map_or(&[][..], |x| &**x)
            && let rustc_ty::RawPtr(_, mutability) | rustc_ty::Ref(_, _, mutability) =
                *adjust.last().map_or(target, |a| a.target).kind()
        {
            return CaptureKind::Ref(mutability);
        }

        match parent {
            Node::Expr(e) => match e.kind {
                ExprKind::AddrOf(_, mutability, _) => return CaptureKind::Ref(mutability),
                ExprKind::Index(..) | ExprKind::Unary(UnOp::Deref, _) => capture = CaptureKind::Ref(Mutability::Not),
                ExprKind::Assign(lhs, ..) | ExprKind::AssignOp(_, lhs, _) if lhs.hir_id == child_id => {
                    return CaptureKind::Ref(Mutability::Mut);
                },
                ExprKind::Field(..) => {
                    if capture == CaptureKind::Value {
                        capture_expr_ty = e;
                    }
                },
                ExprKind::Let(let_expr) => {
                    let mutability = match pat_capture_kind(cx, let_expr.pat) {
                        CaptureKind::Value | CaptureKind::Use => Mutability::Not,
                        CaptureKind::Ref(m) => m,
                    };
                    return CaptureKind::Ref(mutability);
                },
                ExprKind::Match(_, arms, _) => {
                    let mut mutability = Mutability::Not;
                    for capture in arms.iter().map(|arm| pat_capture_kind(cx, arm.pat)) {
                        match capture {
                            CaptureKind::Value | CaptureKind::Use => break,
                            CaptureKind::Ref(Mutability::Mut) => mutability = Mutability::Mut,
                            CaptureKind::Ref(Mutability::Not) => (),
                        }
                    }
                    return CaptureKind::Ref(mutability);
                },
                _ => break,
            },
            Node::LetStmt(l) => match pat_capture_kind(cx, l.pat) {
                CaptureKind::Value | CaptureKind::Use => break,
                capture @ CaptureKind::Ref(_) => return capture,
            },
            _ => break,
        }

        child_id = parent_id;
    }

    if capture == CaptureKind::Value && is_copy(cx, cx.typeck_results().expr_ty(capture_expr_ty)) {
        // Copy types are never automatically captured by value.
        CaptureKind::Ref(Mutability::Not)
    } else {
        capture
    }
}

/// Checks if the expression can be moved into a closure as is. This will return a list of captures
/// if so, otherwise, `None`.
pub fn can_move_expr_to_closure<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<HirIdMap<CaptureKind>> {
    struct V<'cx, 'tcx> {
        cx: &'cx LateContext<'tcx>,
        // Stack of potential break targets contained in the expression.
        loops: Vec<HirId>,
        /// Local variables created in the expression. These don't need to be captured.
        locals: HirIdSet,
        /// Whether this expression can be turned into a closure.
        allow_closure: bool,
        /// Locals which need to be captured, and whether they need to be by value, reference, or
        /// mutable reference.
        captures: HirIdMap<CaptureKind>,
    }
    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if !self.allow_closure {
                return;
            }

            match e.kind {
                ExprKind::Path(QPath::Resolved(None, &Path { res: Res::Local(l), .. })) => {
                    if !self.locals.contains(&l) {
                        let cap = capture_local_usage(self.cx, e);
                        self.captures.entry(l).and_modify(|e| *e |= cap).or_insert(cap);
                    }
                },
                ExprKind::Closure(closure) => {
                    for capture in self.cx.typeck_results().closure_min_captures_flattened(closure.def_id) {
                        let local_id = match capture.place.base {
                            PlaceBase::Local(id) => id,
                            PlaceBase::Upvar(var) => var.var_path.hir_id,
                            _ => continue,
                        };
                        if !self.locals.contains(&local_id) {
                            let capture = match capture.info.capture_kind {
                                UpvarCapture::ByValue => CaptureKind::Value,
                                UpvarCapture::ByUse => CaptureKind::Use,
                                UpvarCapture::ByRef(kind) => match kind {
                                    BorrowKind::Immutable => CaptureKind::Ref(Mutability::Not),
                                    BorrowKind::UniqueImmutable | BorrowKind::Mutable => {
                                        CaptureKind::Ref(Mutability::Mut)
                                    },
                                },
                            };
                            self.captures
                                .entry(local_id)
                                .and_modify(|e| *e |= capture)
                                .or_insert(capture);
                        }
                    }
                },
                ExprKind::Loop(b, ..) => {
                    self.loops.push(e.hir_id);
                    self.visit_block(b);
                    self.loops.pop();
                },
                _ => {
                    self.allow_closure &= can_move_expr_to_closure_no_visit(self.cx, e, &self.loops, &self.locals);
                    walk_expr(self, e);
                },
            }
        }

        fn visit_pat(&mut self, p: &'tcx Pat<'tcx>) {
            p.each_binding_or_first(&mut |_, id, _, _| {
                self.locals.insert(id);
            });
        }
    }

    let mut v = V {
        cx,
        loops: Vec::new(),
        locals: HirIdSet::default(),
        allow_closure: true,
        captures: HirIdMap::default(),
    };
    v.visit_expr(expr);
    v.allow_closure.then_some(v.captures)
}

/// Arguments of a method: the receiver and all the additional arguments.
pub type MethodArguments<'tcx> = Vec<(&'tcx Expr<'tcx>, &'tcx [Expr<'tcx>])>;

/// Returns the method names and argument list of nested method call expressions that make up
/// `expr`. method/span lists are sorted with the most recent call first.
pub fn method_calls<'tcx>(expr: &'tcx Expr<'tcx>, max_depth: usize) -> (Vec<Symbol>, MethodArguments<'tcx>, Vec<Span>) {
    let mut method_names = Vec::with_capacity(max_depth);
    let mut arg_lists = Vec::with_capacity(max_depth);
    let mut spans = Vec::with_capacity(max_depth);

    let mut current = expr;
    for _ in 0..max_depth {
        if let ExprKind::MethodCall(path, receiver, args, _) = &current.kind {
            if receiver.span.from_expansion() || args.iter().any(|e| e.span.from_expansion()) {
                break;
            }
            method_names.push(path.ident.name);
            arg_lists.push((*receiver, &**args));
            spans.push(path.ident.span);
            current = receiver;
        } else {
            break;
        }
    }

    (method_names, arg_lists, spans)
}

/// Matches an `Expr` against a chain of methods, and return the matched `Expr`s.
///
/// For example, if `expr` represents the `.baz()` in `foo.bar().baz()`,
/// `method_chain_args(expr, &["bar", "baz"])` will return a `Vec`
/// containing the `Expr`s for
/// `.bar()` and `.baz()`
pub fn method_chain_args<'a>(expr: &'a Expr<'_>, methods: &[Symbol]) -> Option<Vec<(&'a Expr<'a>, &'a [Expr<'a>])>> {
    let mut current = expr;
    let mut matched = Vec::with_capacity(methods.len());
    for method_name in methods.iter().rev() {
        // method chains are stored last -> first
        if let ExprKind::MethodCall(path, receiver, args, _) = current.kind {
            if path.ident.name == *method_name {
                if receiver.span.from_expansion() || args.iter().any(|e| e.span.from_expansion()) {
                    return None;
                }
                matched.push((receiver, args)); // build up `matched` backwards
                current = receiver; // go to parent expression
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    // Reverse `matched` so that it is in the same order as `methods`.
    matched.reverse();
    Some(matched)
}

/// Returns `true` if the provided `def_id` is an entrypoint to a program.
pub fn is_entrypoint_fn(cx: &LateContext<'_>, def_id: DefId) -> bool {
    cx.tcx
        .entry_fn(())
        .is_some_and(|(entry_fn_def_id, _)| def_id == entry_fn_def_id)
}

/// Returns `true` if the expression is in the program's `#[panic_handler]`.
pub fn is_in_panic_handler(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    let parent = cx.tcx.hir_get_parent_item(e.hir_id);
    Some(parent.to_def_id()) == cx.tcx.lang_items().panic_impl()
}

/// Gets the name of the item the expression is in, if available.
pub fn parent_item_name(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<Symbol> {
    let parent_id = cx.tcx.hir_get_parent_item(expr.hir_id).def_id;
    match cx.tcx.hir_node_by_def_id(parent_id) {
        Node::Item(item) => item.kind.ident().map(|ident| ident.name),
        Node::TraitItem(TraitItem { ident, .. }) | Node::ImplItem(ImplItem { ident, .. }) => Some(ident.name),
        _ => None,
    }
}

pub struct ContainsName<'a, 'tcx> {
    pub cx: &'a LateContext<'tcx>,
    pub name: Symbol,
}

impl<'tcx> Visitor<'tcx> for ContainsName<'_, 'tcx> {
    type Result = ControlFlow<()>;
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_name(&mut self, name: Symbol) -> Self::Result {
        if self.name == name {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}

/// Checks if an `Expr` contains a certain name.
pub fn contains_name<'tcx>(name: Symbol, expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) -> bool {
    let mut cn = ContainsName { cx, name };
    cn.visit_expr(expr).is_break()
}

/// Returns `true` if `expr` contains a return expression
pub fn contains_return<'tcx>(expr: impl Visitable<'tcx>) -> bool {
    for_each_expr_without_closures(expr, |e| {
        if matches!(e.kind, ExprKind::Ret(..)) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

/// Gets the parent expression, if any –- this is useful to constrain a lint.
pub fn get_parent_expr<'tcx>(cx: &LateContext<'tcx>, e: &Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    get_parent_expr_for_hir(cx, e.hir_id)
}

/// This retrieves the parent for the given `HirId` if it's an expression. This is useful for
/// constraint lints
pub fn get_parent_expr_for_hir<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId) -> Option<&'tcx Expr<'tcx>> {
    match cx.tcx.parent_hir_node(hir_id) {
        Node::Expr(parent) => Some(parent),
        _ => None,
    }
}

/// Gets the enclosing block, if any.
pub fn get_enclosing_block<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId) -> Option<&'tcx Block<'tcx>> {
    let enclosing_node = cx
        .tcx
        .hir_get_enclosing_scope(hir_id)
        .map(|enclosing_id| cx.tcx.hir_node(enclosing_id));
    enclosing_node.and_then(|node| match node {
        Node::Block(block) => Some(block),
        Node::Item(&Item {
            kind: ItemKind::Fn { body: eid, .. },
            ..
        })
        | Node::ImplItem(&ImplItem {
            kind: ImplItemKind::Fn(_, eid),
            ..
        })
        | Node::TraitItem(&TraitItem {
            kind: TraitItemKind::Fn(_, TraitFn::Provided(eid)),
            ..
        }) => match cx.tcx.hir_body(eid).value.kind {
            ExprKind::Block(block, _) => Some(block),
            _ => None,
        },
        _ => None,
    })
}

/// Gets the loop or closure enclosing the given expression, if any.
pub fn get_enclosing_loop_or_multi_call_closure<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'_>,
) -> Option<&'tcx Expr<'tcx>> {
    for (_, node) in cx.tcx.hir_parent_iter(expr.hir_id) {
        match node {
            Node::Expr(e) => match e.kind {
                ExprKind::Closure { .. }
                    if let rustc_ty::Closure(_, subs) = cx.typeck_results().expr_ty(e).kind()
                        && subs.as_closure().kind() == ClosureKind::FnOnce => {},

                // Note: A closure's kind is determined by how it's used, not it's captures.
                ExprKind::Closure { .. } | ExprKind::Loop(..) => return Some(e),
                _ => (),
            },
            Node::Stmt(_) | Node::Block(_) | Node::LetStmt(_) | Node::Arm(_) | Node::ExprField(_) => (),
            _ => break,
        }
    }
    None
}

/// Gets the parent node if it's an impl block.
pub fn get_parent_as_impl(tcx: TyCtxt<'_>, id: HirId) -> Option<&Impl<'_>> {
    match tcx.hir_parent_iter(id).next() {
        Some((
            _,
            Node::Item(Item {
                kind: ItemKind::Impl(imp),
                ..
            }),
        )) => Some(imp),
        _ => None,
    }
}

/// Removes blocks around an expression, only if the block contains just one expression
/// and no statements. Unsafe blocks are not removed.
///
/// Examples:
///  * `{}`               -> `{}`
///  * `{ x }`            -> `x`
///  * `{{ x }}`          -> `x`
///  * `{ x; }`           -> `{ x; }`
///  * `{ x; y }`         -> `{ x; y }`
///  * `{ unsafe { x } }` -> `unsafe { x }`
pub fn peel_blocks<'a>(mut expr: &'a Expr<'a>) -> &'a Expr<'a> {
    while let ExprKind::Block(
        Block {
            stmts: [],
            expr: Some(inner),
            rules: BlockCheckMode::DefaultBlock,
            ..
        },
        _,
    ) = expr.kind
    {
        expr = inner;
    }
    expr
}

/// Removes blocks around an expression, only if the block contains just one expression
/// or just one expression statement with a semicolon. Unsafe blocks are not removed.
///
/// Examples:
///  * `{}`               -> `{}`
///  * `{ x }`            -> `x`
///  * `{ x; }`           -> `x`
///  * `{{ x; }}`         -> `x`
///  * `{ x; y }`         -> `{ x; y }`
///  * `{ unsafe { x } }` -> `unsafe { x }`
pub fn peel_blocks_with_stmt<'a>(mut expr: &'a Expr<'a>) -> &'a Expr<'a> {
    while let ExprKind::Block(
        Block {
            stmts: [],
            expr: Some(inner),
            rules: BlockCheckMode::DefaultBlock,
            ..
        }
        | Block {
            stmts:
                [
                    Stmt {
                        kind: StmtKind::Expr(inner) | StmtKind::Semi(inner),
                        ..
                    },
                ],
            expr: None,
            rules: BlockCheckMode::DefaultBlock,
            ..
        },
        _,
    ) = expr.kind
    {
        expr = inner;
    }
    expr
}

/// Checks if the given expression is the else clause of either an `if` or `if let` expression.
pub fn is_else_clause(tcx: TyCtxt<'_>, expr: &Expr<'_>) -> bool {
    let mut iter = tcx.hir_parent_iter(expr.hir_id);
    match iter.next() {
        Some((
            _,
            Node::Expr(Expr {
                kind: ExprKind::If(_, _, Some(else_expr)),
                ..
            }),
        )) => else_expr.hir_id == expr.hir_id,
        _ => false,
    }
}

/// Checks if the given expression is a part of `let else`
/// returns `true` for both the `init` and the `else` part
pub fn is_inside_let_else(tcx: TyCtxt<'_>, expr: &Expr<'_>) -> bool {
    let mut child_id = expr.hir_id;
    for (parent_id, node) in tcx.hir_parent_iter(child_id) {
        if let Node::LetStmt(LetStmt {
            init: Some(init),
            els: Some(els),
            ..
        }) = node
            && (init.hir_id == child_id || els.hir_id == child_id)
        {
            return true;
        }

        child_id = parent_id;
    }

    false
}

/// Checks if the given expression is the else clause of a `let else` expression
pub fn is_else_clause_in_let_else(tcx: TyCtxt<'_>, expr: &Expr<'_>) -> bool {
    let mut child_id = expr.hir_id;
    for (parent_id, node) in tcx.hir_parent_iter(child_id) {
        if let Node::LetStmt(LetStmt { els: Some(els), .. }) = node
            && els.hir_id == child_id
        {
            return true;
        }

        child_id = parent_id;
    }

    false
}

/// Checks whether the given `Expr` is a range equivalent to a `RangeFull`.
///
/// For the lower bound, this means that:
/// - either there is none
/// - or it is the smallest value that can be represented by the range's integer type
///
/// For the upper bound, this means that:
/// - either there is none
/// - or it is the largest value that can be represented by the range's integer type and is
///   inclusive
/// - or it is a call to some container's `len` method and is exclusive, and the range is passed to
///   a method call on that same container (e.g. `v.drain(..v.len())`)
///
/// If the given `Expr` is not some kind of range, the function returns `false`.
pub fn is_range_full(cx: &LateContext<'_>, expr: &Expr<'_>, container_path: Option<&Path<'_>>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    if let Some(Range { start, end, limits }) = Range::hir(expr) {
        let start_is_none_or_min = start.is_none_or(|start| {
            if let rustc_ty::Adt(_, subst) = ty.kind()
                && let bnd_ty = subst.type_at(0)
                && let Some(min_const) = bnd_ty.numeric_min_val(cx.tcx)
                && let Some(min_const) = mir_to_const(cx.tcx, min_const)
                && let Some(start_const) = ConstEvalCtxt::new(cx).eval(start)
            {
                start_const == min_const
            } else {
                false
            }
        });
        let end_is_none_or_max = end.is_none_or(|end| match limits {
            RangeLimits::Closed => {
                if let rustc_ty::Adt(_, subst) = ty.kind()
                    && let bnd_ty = subst.type_at(0)
                    && let Some(max_const) = bnd_ty.numeric_max_val(cx.tcx)
                    && let Some(max_const) = mir_to_const(cx.tcx, max_const)
                    && let Some(end_const) = ConstEvalCtxt::new(cx).eval(end)
                {
                    end_const == max_const
                } else {
                    false
                }
            },
            RangeLimits::HalfOpen => {
                if let Some(container_path) = container_path
                    && let ExprKind::MethodCall(name, self_arg, [], _) = end.kind
                    && name.ident.name == sym::len
                    && let ExprKind::Path(QPath::Resolved(None, path)) = self_arg.kind
                {
                    container_path.res == path.res
                } else {
                    false
                }
            },
        });
        return start_is_none_or_min && end_is_none_or_max;
    }
    false
}

/// Checks whether the given expression is a constant integer of the given value.
/// unlike `is_integer_literal`, this version does const folding
pub fn is_integer_const(cx: &LateContext<'_>, e: &Expr<'_>, value: u128) -> bool {
    if is_integer_literal(e, value) {
        return true;
    }
    let enclosing_body = cx.tcx.hir_enclosing_body_owner(e.hir_id);
    if let Some(Constant::Int(v)) =
        ConstEvalCtxt::with_env(cx.tcx, cx.typing_env(), cx.tcx.typeck(enclosing_body)).eval(e)
    {
        return value == v;
    }
    false
}

/// Checks whether the given expression is a constant literal of the given value.
pub fn is_integer_literal(expr: &Expr<'_>, value: u128) -> bool {
    // FIXME: use constant folding
    if let ExprKind::Lit(spanned) = expr.kind
        && let LitKind::Int(v, _) = spanned.node
    {
        return v == value;
    }
    false
}

/// Checks whether the given expression is a constant literal of the given value.
pub fn is_float_literal(expr: &Expr<'_>, value: f64) -> bool {
    if let ExprKind::Lit(spanned) = expr.kind
        && let LitKind::Float(v, _) = spanned.node
    {
        v.as_str().parse() == Ok(value)
    } else {
        false
    }
}

/// Returns `true` if the given `Expr` has been coerced before.
///
/// Examples of coercions can be found in the Nomicon at
/// <https://doc.rust-lang.org/nomicon/coercions.html>.
///
/// See `rustc_middle::ty::adjustment::Adjustment` and `rustc_hir_analysis::check::coercion` for
/// more information on adjustments and coercions.
pub fn is_adjusted(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    cx.typeck_results().adjustments().get(e.hir_id).is_some()
}

/// Returns the pre-expansion span if this comes from an expansion of the
/// macro `name`.
/// See also [`is_direct_expn_of`].
#[must_use]
pub fn is_expn_of(mut span: Span, name: Symbol) -> Option<Span> {
    loop {
        if span.from_expansion() {
            let data = span.ctxt().outer_expn_data();
            let new_span = data.call_site;

            if let ExpnKind::Macro(MacroKind::Bang, mac_name) = data.kind
                && mac_name == name
            {
                return Some(new_span);
            }

            span = new_span;
        } else {
            return None;
        }
    }
}

/// Returns the pre-expansion span if the span directly comes from an expansion
/// of the macro `name`.
/// The difference with [`is_expn_of`] is that in
/// ```no_run
/// # macro_rules! foo { ($name:tt!$args:tt) => { $name!$args } }
/// # macro_rules! bar { ($e:expr) => { $e } }
/// foo!(bar!(42));
/// ```
/// `42` is considered expanded from `foo!` and `bar!` by `is_expn_of` but only
/// from `bar!` by `is_direct_expn_of`.
#[must_use]
pub fn is_direct_expn_of(span: Span, name: Symbol) -> Option<Span> {
    if span.from_expansion() {
        let data = span.ctxt().outer_expn_data();
        let new_span = data.call_site;

        if let ExpnKind::Macro(MacroKind::Bang, mac_name) = data.kind
            && mac_name == name
        {
            return Some(new_span);
        }
    }

    None
}

/// Convenience function to get the return type of a function.
pub fn return_ty<'tcx>(cx: &LateContext<'tcx>, fn_def_id: OwnerId) -> Ty<'tcx> {
    let ret_ty = cx.tcx.fn_sig(fn_def_id).instantiate_identity().output();
    cx.tcx.instantiate_bound_regions_with_erased(ret_ty)
}

/// Convenience function to get the nth argument type of a function.
pub fn nth_arg<'tcx>(cx: &LateContext<'tcx>, fn_def_id: OwnerId, nth: usize) -> Ty<'tcx> {
    let arg = cx.tcx.fn_sig(fn_def_id).instantiate_identity().input(nth);
    cx.tcx.instantiate_bound_regions_with_erased(arg)
}

/// Checks if an expression is constructing a tuple-like enum variant or struct
pub fn is_ctor_or_promotable_const_function(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ExprKind::Call(fun, _) = expr.kind
        && let ExprKind::Path(ref qp) = fun.kind
    {
        let res = cx.qpath_res(qp, fun.hir_id);
        return match res {
            Res::Def(DefKind::Variant | DefKind::Ctor(..), ..) => true,
            Res::Def(_, def_id) => cx.tcx.is_promotable_const_fn(def_id),
            _ => false,
        };
    }
    false
}

/// Returns `true` if a pattern is refutable.
// TODO: should be implemented using rustc/mir_build/thir machinery
pub fn is_refutable(cx: &LateContext<'_>, pat: &Pat<'_>) -> bool {
    fn is_qpath_refutable(cx: &LateContext<'_>, qpath: &QPath<'_>, id: HirId) -> bool {
        !matches!(
            cx.qpath_res(qpath, id),
            Res::Def(DefKind::Struct, ..) | Res::Def(DefKind::Ctor(def::CtorOf::Struct, _), _)
        )
    }

    fn are_refutable<'a, I: IntoIterator<Item = &'a Pat<'a>>>(cx: &LateContext<'_>, i: I) -> bool {
        i.into_iter().any(|pat| is_refutable(cx, pat))
    }

    match pat.kind {
        PatKind::Missing => unreachable!(),
        PatKind::Wild | PatKind::Never => false, // If `!` typechecked then the type is empty, so not refutable.
        PatKind::Binding(_, _, _, pat) => pat.is_some_and(|pat| is_refutable(cx, pat)),
        PatKind::Box(pat) | PatKind::Ref(pat, _) => is_refutable(cx, pat),
        PatKind::Expr(PatExpr {
            kind: PatExprKind::Path(qpath),
            hir_id,
            ..
        }) => is_qpath_refutable(cx, qpath, *hir_id),
        PatKind::Or(pats) => {
            // TODO: should be the honest check, that pats is exhaustive set
            are_refutable(cx, pats)
        },
        PatKind::Tuple(pats, _) => are_refutable(cx, pats),
        PatKind::Struct(ref qpath, fields, _) => {
            is_qpath_refutable(cx, qpath, pat.hir_id) || are_refutable(cx, fields.iter().map(|field| field.pat))
        },
        PatKind::TupleStruct(ref qpath, pats, _) => {
            is_qpath_refutable(cx, qpath, pat.hir_id) || are_refutable(cx, pats)
        },
        PatKind::Slice(head, middle, tail) => {
            match &cx.typeck_results().node_type(pat.hir_id).kind() {
                rustc_ty::Slice(..) => {
                    // [..] is the only irrefutable slice pattern.
                    !head.is_empty() || middle.is_none() || !tail.is_empty()
                },
                rustc_ty::Array(..) => are_refutable(cx, head.iter().chain(middle).chain(tail.iter())),
                _ => {
                    // unreachable!()
                    true
                },
            }
        },
        PatKind::Expr(..) | PatKind::Range(..) | PatKind::Err(_) | PatKind::Deref(_) | PatKind::Guard(..) => true,
    }
}

/// If the pattern is an `or` pattern, call the function once for each sub pattern. Otherwise, call
/// the function once on the given pattern.
pub fn recurse_or_patterns<'tcx, F: FnMut(&'tcx Pat<'tcx>)>(pat: &'tcx Pat<'tcx>, mut f: F) {
    if let PatKind::Or(pats) = pat.kind {
        pats.iter().for_each(f);
    } else {
        f(pat);
    }
}

pub fn is_self(slf: &Param<'_>) -> bool {
    if let PatKind::Binding(.., name, _) = slf.pat.kind {
        name.name == kw::SelfLower
    } else {
        false
    }
}

pub fn is_self_ty(slf: &hir::Ty<'_>) -> bool {
    if let TyKind::Path(QPath::Resolved(None, path)) = slf.kind
        && let Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } = path.res
    {
        return true;
    }
    false
}

pub fn iter_input_pats<'tcx>(decl: &FnDecl<'_>, body: &'tcx Body<'_>) -> impl Iterator<Item = &'tcx Param<'tcx>> {
    (0..decl.inputs.len()).map(move |i| &body.params[i])
}

/// Checks if a given expression is a match expression expanded from the `?`
/// operator or the `try` macro.
pub fn is_try<'tcx>(cx: &LateContext<'_>, expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    fn is_ok(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
        if let PatKind::TupleStruct(ref path, pat, ddpos) = arm.pat.kind
            && ddpos.as_opt_usize().is_none()
            && is_res_lang_ctor(cx, cx.qpath_res(path, arm.pat.hir_id), ResultOk)
            && let PatKind::Binding(_, hir_id, _, None) = pat[0].kind
            && path_to_local_id(arm.body, hir_id)
        {
            return true;
        }
        false
    }

    fn is_err(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
        if let PatKind::TupleStruct(ref path, _, _) = arm.pat.kind {
            is_res_lang_ctor(cx, cx.qpath_res(path, arm.pat.hir_id), ResultErr)
        } else {
            false
        }
    }

    if let ExprKind::Match(_, arms, ref source) = expr.kind {
        // desugared from a `?` operator
        if let MatchSource::TryDesugar(_) = *source {
            return Some(expr);
        }

        if arms.len() == 2
            && arms[0].guard.is_none()
            && arms[1].guard.is_none()
            && ((is_ok(cx, &arms[0]) && is_err(cx, &arms[1])) || (is_ok(cx, &arms[1]) && is_err(cx, &arms[0])))
        {
            return Some(expr);
        }
    }

    None
}

/// Returns `true` if the lint is `#[allow]`ed or `#[expect]`ed at any of the `ids`, fulfilling all
/// of the expectations in `ids`
///
/// This should only be used when the lint would otherwise be emitted, for a way to check if a lint
/// is allowed early to skip work see [`is_lint_allowed`]
///
/// To emit at a lint at a different context than the one current see
/// [`span_lint_hir`](diagnostics::span_lint_hir) or
/// [`span_lint_hir_and_then`](diagnostics::span_lint_hir_and_then)
pub fn fulfill_or_allowed(cx: &LateContext<'_>, lint: &'static Lint, ids: impl IntoIterator<Item = HirId>) -> bool {
    let mut suppress_lint = false;

    for id in ids {
        let LevelAndSource { level, lint_id, .. } = cx.tcx.lint_level_at_node(lint, id);
        if let Some(expectation) = lint_id {
            cx.fulfill_expectation(expectation);
        }

        match level {
            Level::Allow | Level::Expect => suppress_lint = true,
            Level::Warn | Level::ForceWarn | Level::Deny | Level::Forbid => {},
        }
    }

    suppress_lint
}

/// Returns `true` if the lint is allowed in the current context. This is useful for
/// skipping long running code when it's unnecessary
///
/// This function should check the lint level for the same node, that the lint will
/// be emitted at. If the information is buffered to be emitted at a later point, please
/// make sure to use `span_lint_hir` functions to emit the lint. This ensures that
/// expectations at the checked nodes will be fulfilled.
pub fn is_lint_allowed(cx: &LateContext<'_>, lint: &'static Lint, id: HirId) -> bool {
    cx.tcx.lint_level_at_node(lint, id).level == Level::Allow
}

pub fn strip_pat_refs<'hir>(mut pat: &'hir Pat<'hir>) -> &'hir Pat<'hir> {
    while let PatKind::Ref(subpat, _) = pat.kind {
        pat = subpat;
    }
    pat
}

pub fn int_bits(tcx: TyCtxt<'_>, ity: IntTy) -> u64 {
    Integer::from_int_ty(&tcx, ity).size().bits()
}

#[expect(clippy::cast_possible_wrap)]
/// Turn a constant int byte representation into an i128
pub fn sext(tcx: TyCtxt<'_>, u: u128, ity: IntTy) -> i128 {
    let amt = 128 - int_bits(tcx, ity);
    ((u as i128) << amt) >> amt
}

#[expect(clippy::cast_sign_loss)]
/// clip unused bytes
pub fn unsext(tcx: TyCtxt<'_>, u: i128, ity: IntTy) -> u128 {
    let amt = 128 - int_bits(tcx, ity);
    ((u as u128) << amt) >> amt
}

/// clip unused bytes
pub fn clip(tcx: TyCtxt<'_>, u: u128, ity: UintTy) -> u128 {
    let bits = Integer::from_uint_ty(&tcx, ity).size().bits();
    let amt = 128 - bits;
    (u << amt) >> amt
}

pub fn has_attr(attrs: &[hir::Attribute], symbol: Symbol) -> bool {
    attrs.iter().any(|attr| attr.has_name(symbol))
}

pub fn has_repr_attr(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    find_attr!(cx.tcx.hir_attrs(hir_id), AttributeKind::Repr(..))
}

pub fn any_parent_has_attr(tcx: TyCtxt<'_>, node: HirId, symbol: Symbol) -> bool {
    let mut prev_enclosing_node = None;
    let mut enclosing_node = node;
    while Some(enclosing_node) != prev_enclosing_node {
        if has_attr(tcx.hir_attrs(enclosing_node), symbol) {
            return true;
        }
        prev_enclosing_node = Some(enclosing_node);
        enclosing_node = tcx.hir_get_parent_item(enclosing_node).into();
    }

    false
}

/// Checks if the given HIR node is inside an `impl` block with the `automatically_derived`
/// attribute.
pub fn in_automatically_derived(tcx: TyCtxt<'_>, id: HirId) -> bool {
    tcx.hir_parent_owner_iter(id)
        .filter(|(_, node)| matches!(node, OwnerNode::Item(item) if matches!(item.kind, ItemKind::Impl(_))))
        .any(|(id, _)| {
            has_attr(
                tcx.hir_attrs(tcx.local_def_id_to_hir_id(id.def_id)),
                sym::automatically_derived,
            )
        })
}

/// Checks if the given `DefId` matches the `libc` item.
pub fn match_libc_symbol(cx: &LateContext<'_>, did: DefId, name: Symbol) -> bool {
    // libc is meant to be used as a flat list of names, but they're all actually defined in different
    // modules based on the target platform. Ignore everything but crate name and the item name.
    cx.tcx.crate_name(did.krate) == sym::libc && cx.tcx.def_path_str(did).ends_with(name.as_str())
}

/// Returns the list of condition expressions and the list of blocks in a
/// sequence of `if/else`.
/// E.g., this returns `([a, b], [c, d, e])` for the expression
/// `if a { c } else if b { d } else { e }`.
pub fn if_sequence<'tcx>(mut expr: &'tcx Expr<'tcx>) -> (Vec<&'tcx Expr<'tcx>>, Vec<&'tcx Block<'tcx>>) {
    let mut conds = Vec::new();
    let mut blocks: Vec<&Block<'_>> = Vec::new();

    while let Some(higher::IfOrIfLet { cond, then, r#else }) = higher::IfOrIfLet::hir(expr) {
        conds.push(cond);
        if let ExprKind::Block(block, _) = then.kind {
            blocks.push(block);
        } else {
            panic!("ExprKind::If node is not an ExprKind::Block");
        }

        if let Some(else_expr) = r#else {
            expr = else_expr;
        } else {
            break;
        }
    }

    // final `else {..}`
    if !blocks.is_empty()
        && let ExprKind::Block(block, _) = expr.kind
    {
        blocks.push(block);
    }

    (conds, blocks)
}

/// Checks if the given function kind is an async function.
pub fn is_async_fn(kind: FnKind<'_>) -> bool {
    match kind {
        FnKind::ItemFn(_, _, header) => header.asyncness.is_async(),
        FnKind::Method(_, sig) => sig.header.asyncness.is_async(),
        FnKind::Closure => false,
    }
}

/// Peels away all the compiler generated code surrounding the body of an async closure.
pub fn get_async_closure_expr<'tcx>(tcx: TyCtxt<'tcx>, expr: &Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Closure(&Closure {
        body,
        kind: hir::ClosureKind::Coroutine(CoroutineKind::Desugared(CoroutineDesugaring::Async, _)),
        ..
    }) = expr.kind
        && let ExprKind::Block(
            Block {
                expr:
                    Some(Expr {
                        kind: ExprKind::DropTemps(inner_expr),
                        ..
                    }),
                ..
            },
            _,
        ) = tcx.hir_body(body).value.kind
    {
        Some(inner_expr)
    } else {
        None
    }
}

/// Peels away all the compiler generated code surrounding the body of an async function,
pub fn get_async_fn_body<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'_>) -> Option<&'tcx Expr<'tcx>> {
    get_async_closure_expr(tcx, body.value)
}

// check if expr is calling method or function with #[must_use] attribute
pub fn is_must_use_func_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let did = match expr.kind {
        ExprKind::Call(path, _) => {
            if let ExprKind::Path(ref qpath) = path.kind
                && let Res::Def(_, did) = cx.qpath_res(qpath, path.hir_id)
            {
                Some(did)
            } else {
                None
            }
        },
        ExprKind::MethodCall(..) => cx.typeck_results().type_dependent_def_id(expr.hir_id),
        _ => None,
    };

    did.is_some_and(|did| find_attr!(cx.tcx.get_all_attrs(did), AttributeKind::MustUse { .. }))
}

/// Checks if a function's body represents the identity function. Looks for bodies of the form:
/// * `|x| x`
/// * `|x| return x`
/// * `|x| { return x }`
/// * `|x| { return x; }`
/// * `|(x, y)| (x, y)`
///
/// Consider calling [`is_expr_untyped_identity_function`] or [`is_expr_identity_function`] instead.
fn is_body_identity_function(cx: &LateContext<'_>, func: &Body<'_>) -> bool {
    fn check_pat(cx: &LateContext<'_>, pat: &Pat<'_>, expr: &Expr<'_>) -> bool {
        if cx
            .typeck_results()
            .pat_binding_modes()
            .get(pat.hir_id)
            .is_some_and(|mode| matches!(mode.0, ByRef::Yes(_)))
        {
            // If a tuple `(x, y)` is of type `&(i32, i32)`, then due to match ergonomics,
            // the inner patterns become references. Don't consider this the identity function
            // as that changes types.
            return false;
        }

        match (pat.kind, expr.kind) {
            (PatKind::Binding(_, id, _, _), _) => {
                path_to_local_id(expr, id) && cx.typeck_results().expr_adjustments(expr).is_empty()
            },
            (PatKind::Tuple(pats, dotdot), ExprKind::Tup(tup))
                if dotdot.as_opt_usize().is_none() && pats.len() == tup.len() =>
            {
                pats.iter().zip(tup).all(|(pat, expr)| check_pat(cx, pat, expr))
            },
            _ => false,
        }
    }

    let [param] = func.params else {
        return false;
    };

    let mut expr = func.value;
    loop {
        match expr.kind {
            ExprKind::Block(
                &Block {
                    stmts: [],
                    expr: Some(e),
                    ..
                },
                _,
            )
            | ExprKind::Ret(Some(e)) => expr = e,
            ExprKind::Block(
                &Block {
                    stmts: [stmt],
                    expr: None,
                    ..
                },
                _,
            ) => {
                if let StmtKind::Semi(e) | StmtKind::Expr(e) = stmt.kind
                    && let ExprKind::Ret(Some(ret_val)) = e.kind
                {
                    expr = ret_val;
                } else {
                    return false;
                }
            },
            _ => return check_pat(cx, param.pat, expr),
        }
    }
}

/// This is the same as [`is_expr_identity_function`], but does not consider closures
/// with type annotations for its bindings (or similar) as identity functions:
/// * `|x: u8| x`
/// * `std::convert::identity::<u8>`
pub fn is_expr_untyped_identity_function(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Closure(&Closure { body, fn_decl, .. })
            if fn_decl.inputs.iter().all(|ty| matches!(ty.kind, TyKind::Infer(()))) =>
        {
            is_body_identity_function(cx, cx.tcx.hir_body(body))
        },
        ExprKind::Path(QPath::Resolved(_, path))
            if path.segments.iter().all(|seg| seg.infer_args)
                && let Some(did) = path.res.opt_def_id() =>
        {
            cx.tcx.is_diagnostic_item(sym::convert_identity, did)
        },
        _ => false,
    }
}

/// Checks if an expression represents the identity function
/// Only examines closures and `std::convert::identity`
///
/// NOTE: If you want to use this function to find out if a closure is unnecessary, you likely want
/// to call [`is_expr_untyped_identity_function`] instead, which makes sure that the closure doesn't
/// have type annotations. This is important because removing a closure with bindings can
/// remove type information that helped type inference before, which can then lead to compile
/// errors.
pub fn is_expr_identity_function(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Closure(&Closure { body, .. }) => is_body_identity_function(cx, cx.tcx.hir_body(body)),
        _ => path_def_id(cx, expr).is_some_and(|id| cx.tcx.is_diagnostic_item(sym::convert_identity, id)),
    }
}

/// Gets the node where an expression is either used, or it's type is unified with another branch.
/// Returns both the node and the `HirId` of the closest child node.
pub fn get_expr_use_or_unification_node<'tcx>(tcx: TyCtxt<'tcx>, expr: &Expr<'_>) -> Option<(Node<'tcx>, HirId)> {
    let mut child_id = expr.hir_id;
    let mut iter = tcx.hir_parent_iter(child_id);
    loop {
        match iter.next() {
            None => break None,
            Some((id, Node::Block(_))) => child_id = id,
            Some((id, Node::Arm(arm))) if arm.body.hir_id == child_id => child_id = id,
            Some((_, Node::Expr(expr))) => match expr.kind {
                ExprKind::Match(_, [arm], _) if arm.hir_id == child_id => child_id = expr.hir_id,
                ExprKind::Block(..) | ExprKind::DropTemps(_) => child_id = expr.hir_id,
                ExprKind::If(_, then_expr, None) if then_expr.hir_id == child_id => break None,
                _ => break Some((Node::Expr(expr), child_id)),
            },
            Some((_, node)) => break Some((node, child_id)),
        }
    }
}

/// Checks if the result of an expression is used, or it's type is unified with another branch.
pub fn is_expr_used_or_unified(tcx: TyCtxt<'_>, expr: &Expr<'_>) -> bool {
    !matches!(
        get_expr_use_or_unification_node(tcx, expr),
        None | Some((
            Node::Stmt(Stmt {
                kind: StmtKind::Expr(_)
                    | StmtKind::Semi(_)
                    | StmtKind::Let(LetStmt {
                        pat: Pat {
                            kind: PatKind::Wild,
                            ..
                        },
                        ..
                    }),
                ..
            }),
            _
        ))
    )
}

/// Checks if the expression is the final expression returned from a block.
pub fn is_expr_final_block_expr(tcx: TyCtxt<'_>, expr: &Expr<'_>) -> bool {
    matches!(tcx.parent_hir_node(expr.hir_id), Node::Block(..))
}

/// Checks if the expression is a temporary value.
// This logic is the same as the one used in rustc's `check_named_place_expr function`.
// https://github.com/rust-lang/rust/blob/3ed2a10d173d6c2e0232776af338ca7d080b1cd4/compiler/rustc_hir_typeck/src/expr.rs#L482-L499
pub fn is_expr_temporary_value(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    !expr.is_place_expr(|base| {
        cx.typeck_results()
            .adjustments()
            .get(base.hir_id)
            .is_some_and(|x| x.iter().any(|adj| matches!(adj.kind, Adjust::Deref(_))))
    })
}

pub fn std_or_core(cx: &LateContext<'_>) -> Option<&'static str> {
    if !is_no_std_crate(cx) {
        Some("std")
    } else if !is_no_core_crate(cx) {
        Some("core")
    } else {
        None
    }
}

pub fn is_no_std_crate(cx: &LateContext<'_>) -> bool {
    cx.tcx
        .hir_attrs(hir::CRATE_HIR_ID)
        .iter()
        .any(|attr| attr.has_name(sym::no_std))
}

pub fn is_no_core_crate(cx: &LateContext<'_>) -> bool {
    cx.tcx
        .hir_attrs(hir::CRATE_HIR_ID)
        .iter()
        .any(|attr| attr.has_name(sym::no_core))
}

/// Check if parent of a hir node is a trait implementation block.
/// For example, `f` in
/// ```no_run
/// # struct S;
/// # trait Trait { fn f(); }
/// impl Trait for S {
///     fn f() {}
/// }
/// ```
pub fn is_trait_impl_item(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    if let Node::Item(item) = cx.tcx.parent_hir_node(hir_id) {
        matches!(item.kind, ItemKind::Impl(Impl { of_trait: Some(_), .. }))
    } else {
        false
    }
}

/// Check if it's even possible to satisfy the `where` clause for the item.
///
/// `trivial_bounds` feature allows functions with unsatisfiable bounds, for example:
///
/// ```ignore
/// fn foo() where i32: Iterator {
///     for _ in 2i32 {}
/// }
/// ```
pub fn fn_has_unsatisfiable_preds(cx: &LateContext<'_>, did: DefId) -> bool {
    use rustc_trait_selection::traits;
    let predicates = cx
        .tcx
        .predicates_of(did)
        .predicates
        .iter()
        .filter_map(|(p, _)| if p.is_global() { Some(*p) } else { None });
    traits::impossible_predicates(cx.tcx, traits::elaborate(cx.tcx, predicates).collect::<Vec<_>>())
}

/// Returns the `DefId` of the callee if the given expression is a function or method call.
pub fn fn_def_id(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<DefId> {
    fn_def_id_with_node_args(cx, expr).map(|(did, _)| did)
}

/// Returns the `DefId` of the callee if the given expression is a function or method call,
/// as well as its node args.
pub fn fn_def_id_with_node_args<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'_>,
) -> Option<(DefId, GenericArgsRef<'tcx>)> {
    let typeck = cx.typeck_results();
    match &expr.kind {
        ExprKind::MethodCall(..) => Some((
            typeck.type_dependent_def_id(expr.hir_id)?,
            typeck.node_args(expr.hir_id),
        )),
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(qpath),
                hir_id: path_hir_id,
                ..
            },
            ..,
        ) => {
            // Only return Fn-like DefIds, not the DefIds of statics/consts/etc that contain or
            // deref to fn pointers, dyn Fn, impl Fn - #8850
            if let Res::Def(DefKind::Fn | DefKind::Ctor(..) | DefKind::AssocFn, id) =
                typeck.qpath_res(qpath, *path_hir_id)
            {
                Some((id, typeck.node_args(*path_hir_id)))
            } else {
                None
            }
        },
        _ => None,
    }
}

/// Returns `Option<String>` where String is a textual representation of the type encapsulated in
/// the slice iff the given expression is a slice of primitives.
///
/// (As defined in the `is_recursively_primitive_type` function.) Returns `None` otherwise.
pub fn is_slice_of_primitives(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<String> {
    let expr_type = cx.typeck_results().expr_ty_adjusted(expr);
    let expr_kind = expr_type.kind();
    let is_primitive = match expr_kind {
        rustc_ty::Slice(element_type) => is_recursively_primitive_type(*element_type),
        rustc_ty::Ref(_, inner_ty, _) if matches!(inner_ty.kind(), &rustc_ty::Slice(_)) => {
            if let rustc_ty::Slice(element_type) = inner_ty.kind() {
                is_recursively_primitive_type(*element_type)
            } else {
                unreachable!()
            }
        },
        _ => false,
    };

    if is_primitive {
        // if we have wrappers like Array, Slice or Tuple, print these
        // and get the type enclosed in the slice ref
        match expr_type.peel_refs().walk().nth(1).unwrap().expect_ty().kind() {
            rustc_ty::Slice(..) => return Some("slice".into()),
            rustc_ty::Array(..) => return Some("array".into()),
            rustc_ty::Tuple(..) => return Some("tuple".into()),
            _ => {
                // is_recursively_primitive_type() should have taken care
                // of the rest and we can rely on the type that is found
                let refs_peeled = expr_type.peel_refs();
                return Some(refs_peeled.walk().last().unwrap().to_string());
            },
        }
    }
    None
}

/// Returns a list of groups where elements in each group are equal according to `eq`
///
/// - Within each group the elements are sorted by the order they appear in `exprs`
/// - The groups themselves are sorted by their first element's appearence in `exprs`
///
/// Given functions `eq` and `hash` such that `eq(a, b) == true`
/// implies `hash(a) == hash(b)`
pub fn search_same<T, Hash, Eq>(exprs: &[T], mut hash: Hash, mut eq: Eq) -> Vec<Vec<&T>>
where
    Hash: FnMut(&T) -> u64,
    Eq: FnMut(&T, &T) -> bool,
{
    match exprs {
        [a, b] if eq(a, b) => return vec![vec![a, b]],
        _ if exprs.len() <= 2 => return vec![],
        _ => {},
    }

    let mut buckets: UnindexMap<u64, Vec<Vec<&T>>> = UnindexMap::default();

    for expr in exprs {
        match buckets.entry(hash(expr)) {
            indexmap::map::Entry::Occupied(mut o) => {
                let bucket = o.get_mut();
                match bucket.iter_mut().find(|group| eq(expr, group[0])) {
                    Some(group) => group.push(expr),
                    None => bucket.push(vec![expr]),
                }
            },
            indexmap::map::Entry::Vacant(v) => {
                v.insert(vec![vec![expr]]);
            },
        }
    }

    buckets
        .into_values()
        .flatten()
        .filter(|group| group.len() > 1)
        .collect()
}

/// Peels off all references on the pattern. Returns the underlying pattern and the number of
/// references removed.
pub fn peel_hir_pat_refs<'a>(pat: &'a Pat<'a>) -> (&'a Pat<'a>, usize) {
    fn peel<'a>(pat: &'a Pat<'a>, count: usize) -> (&'a Pat<'a>, usize) {
        if let PatKind::Ref(pat, _) = pat.kind {
            peel(pat, count + 1)
        } else {
            (pat, count)
        }
    }
    peel(pat, 0)
}

/// Peels of expressions while the given closure returns `Some`.
pub fn peel_hir_expr_while<'tcx>(
    mut expr: &'tcx Expr<'tcx>,
    mut f: impl FnMut(&'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>>,
) -> &'tcx Expr<'tcx> {
    while let Some(e) = f(expr) {
        expr = e;
    }
    expr
}

/// Peels off up to the given number of references on the expression. Returns the underlying
/// expression and the number of references removed.
pub fn peel_n_hir_expr_refs<'a>(expr: &'a Expr<'a>, count: usize) -> (&'a Expr<'a>, usize) {
    let mut remaining = count;
    let e = peel_hir_expr_while(expr, |e| match e.kind {
        ExprKind::AddrOf(ast::BorrowKind::Ref, _, e) if remaining != 0 => {
            remaining -= 1;
            Some(e)
        },
        _ => None,
    });
    (e, count - remaining)
}

/// Peels off all unary operators of an expression. Returns the underlying expression and the number
/// of operators removed.
pub fn peel_hir_expr_unary<'a>(expr: &'a Expr<'a>) -> (&'a Expr<'a>, usize) {
    let mut count: usize = 0;
    let mut curr_expr = expr;
    while let ExprKind::Unary(_, local_expr) = curr_expr.kind {
        count = count.wrapping_add(1);
        curr_expr = local_expr;
    }
    (curr_expr, count)
}

/// Peels off all references on the expression. Returns the underlying expression and the number of
/// references removed.
pub fn peel_hir_expr_refs<'a>(expr: &'a Expr<'a>) -> (&'a Expr<'a>, usize) {
    let mut count = 0;
    let e = peel_hir_expr_while(expr, |e| match e.kind {
        ExprKind::AddrOf(ast::BorrowKind::Ref, _, e) => {
            count += 1;
            Some(e)
        },
        _ => None,
    });
    (e, count)
}

/// Peels off all references on the type. Returns the underlying type and the number of references
/// removed.
pub fn peel_hir_ty_refs<'a>(mut ty: &'a hir::Ty<'a>) -> (&'a hir::Ty<'a>, usize) {
    let mut count = 0;
    loop {
        match &ty.kind {
            TyKind::Ref(_, ref_ty) => {
                ty = ref_ty.ty;
                count += 1;
            },
            _ => break (ty, count),
        }
    }
}

/// Peels off all references on the type. Returns the underlying type and the number of references
/// removed.
pub fn peel_middle_ty_refs(mut ty: Ty<'_>) -> (Ty<'_>, usize) {
    let mut count = 0;
    while let rustc_ty::Ref(_, dest_ty, _) = ty.kind() {
        ty = *dest_ty;
        count += 1;
    }
    (ty, count)
}

/// Removes `AddrOf` operators (`&`) or deref operators (`*`), but only if a reference type is
/// dereferenced. An overloaded deref such as `Vec` to slice would not be removed.
pub fn peel_ref_operators<'hir>(cx: &LateContext<'_>, mut expr: &'hir Expr<'hir>) -> &'hir Expr<'hir> {
    loop {
        match expr.kind {
            ExprKind::AddrOf(_, _, e) => expr = e,
            ExprKind::Unary(UnOp::Deref, e) if cx.typeck_results().expr_ty(e).is_ref() => expr = e,
            _ => break,
        }
    }
    expr
}

pub fn is_hir_ty_cfg_dependant(cx: &LateContext<'_>, ty: &hir::Ty<'_>) -> bool {
    if let TyKind::Path(QPath::Resolved(_, path)) = ty.kind
        && let Res::Def(_, def_id) = path.res
    {
        return cx.tcx.has_attr(def_id, sym::cfg) || cx.tcx.has_attr(def_id, sym::cfg_attr);
    }
    false
}

static TEST_ITEM_NAMES_CACHE: OnceLock<Mutex<FxHashMap<LocalModDefId, Vec<Symbol>>>> = OnceLock::new();

/// Apply `f()` to the set of test item names.
/// The names are sorted using the default `Symbol` ordering.
fn with_test_item_names(tcx: TyCtxt<'_>, module: LocalModDefId, f: impl FnOnce(&[Symbol]) -> bool) -> bool {
    let cache = TEST_ITEM_NAMES_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()));
    let mut map: MutexGuard<'_, FxHashMap<LocalModDefId, Vec<Symbol>>> = cache.lock().unwrap();
    let value = map.entry(module);
    match value {
        Entry::Occupied(entry) => f(entry.get()),
        Entry::Vacant(entry) => {
            let mut names = Vec::new();
            for id in tcx.hir_module_free_items(module) {
                if matches!(tcx.def_kind(id.owner_id), DefKind::Const)
                    && let item = tcx.hir_item(id)
                    && let ItemKind::Const(ident, _generics, ty, _body) = item.kind
                    && let TyKind::Path(QPath::Resolved(_, path)) = ty.kind
                        // We could also check for the type name `test::TestDescAndFn`
                        && let Res::Def(DefKind::Struct, _) = path.res
                {
                    let has_test_marker = tcx
                        .hir_attrs(item.hir_id())
                        .iter()
                        .any(|a| a.has_name(sym::rustc_test_marker));
                    if has_test_marker {
                        names.push(ident.name);
                    }
                }
            }
            names.sort_unstable();
            f(entry.insert(names))
        },
    }
}

/// Checks if the function containing the given `HirId` is a `#[test]` function
///
/// Note: Add `//@compile-flags: --test` to UI tests with a `#[test]` function
pub fn is_in_test_function(tcx: TyCtxt<'_>, id: HirId) -> bool {
    with_test_item_names(tcx, tcx.parent_module(id), |names| {
        let node = tcx.hir_node(id);
        once((id, node))
            .chain(tcx.hir_parent_iter(id))
            // Since you can nest functions we need to collect all until we leave
            // function scope
            .any(|(_id, node)| {
                if let Node::Item(item) = node
                    && let ItemKind::Fn { ident, .. } = item.kind
                {
                    // Note that we have sorted the item names in the visitor,
                    // so the binary_search gets the same as `contains`, but faster.
                    return names.binary_search(&ident.name).is_ok();
                }
                false
            })
    })
}

/// Checks if `fn_def_id` has a `#[test]` attribute applied
///
/// This only checks directly applied attributes. To see if a node has a parent function marked with
/// `#[test]` use [`is_in_test_function`].
///
/// Note: Add `//@compile-flags: --test` to UI tests with a `#[test]` function
pub fn is_test_function(tcx: TyCtxt<'_>, fn_def_id: LocalDefId) -> bool {
    let id = tcx.local_def_id_to_hir_id(fn_def_id);
    if let Node::Item(item) = tcx.hir_node(id)
        && let ItemKind::Fn { ident, .. } = item.kind
    {
        with_test_item_names(tcx, tcx.parent_module(id), |names| {
            names.binary_search(&ident.name).is_ok()
        })
    } else {
        false
    }
}

/// Checks if `id` has a `#[cfg(test)]` attribute applied
///
/// This only checks directly applied attributes, to see if a node is inside a `#[cfg(test)]` parent
/// use [`is_in_cfg_test`]
pub fn is_cfg_test(tcx: TyCtxt<'_>, id: HirId) -> bool {
    tcx.hir_attrs(id).iter().any(|attr| {
        if attr.has_name(sym::cfg_trace)
            && let Some(items) = attr.meta_item_list()
            && let [item] = &*items
            && item.has_name(sym::test)
        {
            true
        } else {
            false
        }
    })
}

/// Checks if any parent node of `HirId` has `#[cfg(test)]` attribute applied
pub fn is_in_cfg_test(tcx: TyCtxt<'_>, id: HirId) -> bool {
    tcx.hir_parent_id_iter(id).any(|parent_id| is_cfg_test(tcx, parent_id))
}

/// Checks if the node is in a `#[test]` function or has any parent node marked `#[cfg(test)]`
pub fn is_in_test(tcx: TyCtxt<'_>, hir_id: HirId) -> bool {
    is_in_test_function(tcx, hir_id) || is_in_cfg_test(tcx, hir_id)
}

/// Checks if the item of any of its parents has `#[cfg(...)]` attribute applied.
pub fn inherits_cfg(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    tcx.has_attr(def_id, sym::cfg_trace)
        || tcx
            .hir_parent_iter(tcx.local_def_id_to_hir_id(def_id))
            .flat_map(|(parent_id, _)| tcx.hir_attrs(parent_id))
            .any(|attr| attr.has_name(sym::cfg_trace))
}

/// Walks up the HIR tree from the given expression in an attempt to find where the value is
/// consumed.
///
/// Termination has three conditions:
/// - The given function returns `Break`. This function will return the value.
/// - The consuming node is found. This function will return `Continue(use_node, child_id)`.
/// - No further parent nodes are found. This will trigger a debug assert or return `None`.
///
/// This allows walking through `if`, `match`, `break`, and block expressions to find where the
/// value produced by the expression is consumed.
pub fn walk_to_expr_usage<'tcx, T>(
    cx: &LateContext<'tcx>,
    e: &Expr<'tcx>,
    mut f: impl FnMut(HirId, Node<'tcx>, HirId) -> ControlFlow<T>,
) -> Option<ControlFlow<T, (Node<'tcx>, HirId)>> {
    let mut iter = cx.tcx.hir_parent_iter(e.hir_id);
    let mut child_id = e.hir_id;

    while let Some((parent_id, parent)) = iter.next() {
        if let ControlFlow::Break(x) = f(parent_id, parent, child_id) {
            return Some(ControlFlow::Break(x));
        }
        let parent_expr = match parent {
            Node::Expr(e) => e,
            Node::Block(Block { expr: Some(body), .. }) | Node::Arm(Arm { body, .. }) if body.hir_id == child_id => {
                child_id = parent_id;
                continue;
            },
            Node::Arm(a) if a.body.hir_id == child_id => {
                child_id = parent_id;
                continue;
            },
            _ => return Some(ControlFlow::Continue((parent, child_id))),
        };
        match parent_expr.kind {
            ExprKind::If(child, ..) | ExprKind::Match(child, ..) if child.hir_id != child_id => child_id = parent_id,
            ExprKind::Break(Destination { target_id: Ok(id), .. }, _) => {
                child_id = id;
                iter = cx.tcx.hir_parent_iter(id);
            },
            ExprKind::Block(..) | ExprKind::DropTemps(_) => child_id = parent_id,
            _ => return Some(ControlFlow::Continue((parent, child_id))),
        }
    }
    debug_assert!(false, "no parent node found for `{child_id:?}`");
    None
}

/// A type definition as it would be viewed from within a function.
#[derive(Clone, Copy)]
pub enum DefinedTy<'tcx> {
    // Used for locals and closures defined within the function.
    Hir(&'tcx hir::Ty<'tcx>),
    /// Used for function signatures, and constant and static values. The type is
    /// in the context of its definition site. We also track the `def_id` of its
    /// definition site.
    ///
    /// WARNING: As the `ty` in in the scope of the definition, not of the function
    /// using it, you must be very careful with how you use it. Using it in the wrong
    /// scope easily results in ICEs.
    Mir {
        def_site_def_id: Option<DefId>,
        ty: Binder<'tcx, Ty<'tcx>>,
    },
}

/// The context an expressions value is used in.
pub struct ExprUseCtxt<'tcx> {
    /// The parent node which consumes the value.
    pub node: Node<'tcx>,
    /// The child id of the node the value came from.
    pub child_id: HirId,
    /// Any adjustments applied to the type.
    pub adjustments: &'tcx [Adjustment<'tcx>],
    /// Whether the type must unify with another code path.
    pub is_ty_unified: bool,
    /// Whether the value will be moved before it's used.
    pub moved_before_use: bool,
    /// Whether the use site has the same `SyntaxContext` as the value.
    pub same_ctxt: bool,
}
impl<'tcx> ExprUseCtxt<'tcx> {
    pub fn use_node(&self, cx: &LateContext<'tcx>) -> ExprUseNode<'tcx> {
        match self.node {
            Node::LetStmt(l) => ExprUseNode::LetStmt(l),
            Node::ExprField(field) => ExprUseNode::Field(field),

            Node::Item(&Item {
                kind: ItemKind::Static(..) | ItemKind::Const(..),
                owner_id,
                ..
            })
            | Node::TraitItem(&TraitItem {
                kind: TraitItemKind::Const(..),
                owner_id,
                ..
            })
            | Node::ImplItem(&ImplItem {
                kind: ImplItemKind::Const(..),
                owner_id,
                ..
            }) => ExprUseNode::ConstStatic(owner_id),

            Node::Item(&Item {
                kind: ItemKind::Fn { .. },
                owner_id,
                ..
            })
            | Node::TraitItem(&TraitItem {
                kind: TraitItemKind::Fn(..),
                owner_id,
                ..
            })
            | Node::ImplItem(&ImplItem {
                kind: ImplItemKind::Fn(..),
                owner_id,
                ..
            }) => ExprUseNode::Return(owner_id),

            Node::Expr(use_expr) => match use_expr.kind {
                ExprKind::Ret(_) => ExprUseNode::Return(OwnerId {
                    def_id: cx.tcx.hir_body_owner_def_id(cx.enclosing_body.unwrap()),
                }),

                ExprKind::Closure(closure) => ExprUseNode::Return(OwnerId { def_id: closure.def_id }),
                ExprKind::Call(func, args) => match args.iter().position(|arg| arg.hir_id == self.child_id) {
                    Some(i) => ExprUseNode::FnArg(func, i),
                    None => ExprUseNode::Callee,
                },
                ExprKind::MethodCall(name, _, args, _) => ExprUseNode::MethodArg(
                    use_expr.hir_id,
                    name.args,
                    args.iter()
                        .position(|arg| arg.hir_id == self.child_id)
                        .map_or(0, |i| i + 1),
                ),
                ExprKind::Field(_, name) => ExprUseNode::FieldAccess(name),
                ExprKind::AddrOf(kind, mutbl, _) => ExprUseNode::AddrOf(kind, mutbl),
                _ => ExprUseNode::Other,
            },
            _ => ExprUseNode::Other,
        }
    }
}

/// The node which consumes a value.
pub enum ExprUseNode<'tcx> {
    /// Assignment to, or initializer for, a local
    LetStmt(&'tcx LetStmt<'tcx>),
    /// Initializer for a const or static item.
    ConstStatic(OwnerId),
    /// Implicit or explicit return from a function.
    Return(OwnerId),
    /// Initialization of a struct field.
    Field(&'tcx ExprField<'tcx>),
    /// An argument to a function.
    FnArg(&'tcx Expr<'tcx>, usize),
    /// An argument to a method.
    MethodArg(HirId, Option<&'tcx GenericArgs<'tcx>>, usize),
    /// The callee of a function call.
    Callee,
    /// Access of a field.
    FieldAccess(Ident),
    /// Borrow expression.
    AddrOf(ast::BorrowKind, Mutability),
    Other,
}
impl<'tcx> ExprUseNode<'tcx> {
    /// Checks if the value is returned from the function.
    pub fn is_return(&self) -> bool {
        matches!(self, Self::Return(_))
    }

    /// Checks if the value is used as a method call receiver.
    pub fn is_recv(&self) -> bool {
        matches!(self, Self::MethodArg(_, _, 0))
    }

    /// Gets the needed type as it's defined without any type inference.
    pub fn defined_ty(&self, cx: &LateContext<'tcx>) -> Option<DefinedTy<'tcx>> {
        match *self {
            Self::LetStmt(LetStmt { ty: Some(ty), .. }) => Some(DefinedTy::Hir(ty)),
            Self::ConstStatic(id) => Some(DefinedTy::Mir {
                def_site_def_id: Some(id.def_id.to_def_id()),
                ty: Binder::dummy(cx.tcx.type_of(id).instantiate_identity()),
            }),
            Self::Return(id) => {
                if let Node::Expr(Expr {
                    kind: ExprKind::Closure(c),
                    ..
                }) = cx.tcx.hir_node_by_def_id(id.def_id)
                {
                    match c.fn_decl.output {
                        FnRetTy::DefaultReturn(_) => None,
                        FnRetTy::Return(ty) => Some(DefinedTy::Hir(ty)),
                    }
                } else {
                    let ty = cx.tcx.fn_sig(id).instantiate_identity().output();
                    Some(DefinedTy::Mir {
                        def_site_def_id: Some(id.def_id.to_def_id()),
                        ty,
                    })
                }
            },
            Self::Field(field) => match get_parent_expr_for_hir(cx, field.hir_id) {
                Some(Expr {
                    hir_id,
                    kind: ExprKind::Struct(path, ..),
                    ..
                }) => adt_and_variant_of_res(cx, cx.qpath_res(path, *hir_id))
                    .and_then(|(adt, variant)| {
                        variant
                            .fields
                            .iter()
                            .find(|f| f.name == field.ident.name)
                            .map(|f| (adt, f))
                    })
                    .map(|(adt, field_def)| DefinedTy::Mir {
                        def_site_def_id: Some(adt.did()),
                        ty: Binder::dummy(cx.tcx.type_of(field_def.did).instantiate_identity()),
                    }),
                _ => None,
            },
            Self::FnArg(callee, i) => {
                let sig = expr_sig(cx, callee)?;
                let (hir_ty, ty) = sig.input_with_hir(i)?;
                Some(match hir_ty {
                    Some(hir_ty) => DefinedTy::Hir(hir_ty),
                    None => DefinedTy::Mir {
                        def_site_def_id: sig.predicates_id(),
                        ty,
                    },
                })
            },
            Self::MethodArg(id, _, i) => {
                let id = cx.typeck_results().type_dependent_def_id(id)?;
                let sig = cx.tcx.fn_sig(id).skip_binder();
                Some(DefinedTy::Mir {
                    def_site_def_id: Some(id),
                    ty: sig.input(i),
                })
            },
            Self::LetStmt(_) | Self::FieldAccess(..) | Self::Callee | Self::Other | Self::AddrOf(..) => None,
        }
    }
}

/// Gets the context an expression's value is used in.
pub fn expr_use_ctxt<'tcx>(cx: &LateContext<'tcx>, e: &Expr<'tcx>) -> ExprUseCtxt<'tcx> {
    let mut adjustments = [].as_slice();
    let mut is_ty_unified = false;
    let mut moved_before_use = false;
    let mut same_ctxt = true;
    let ctxt = e.span.ctxt();
    let node = walk_to_expr_usage(cx, e, &mut |parent_id, parent, child_id| -> ControlFlow<!> {
        if adjustments.is_empty()
            && let Node::Expr(e) = cx.tcx.hir_node(child_id)
        {
            adjustments = cx.typeck_results().expr_adjustments(e);
        }
        same_ctxt &= cx.tcx.hir_span(parent_id).ctxt() == ctxt;
        if let Node::Expr(e) = parent {
            match e.kind {
                ExprKind::If(e, _, _) | ExprKind::Match(e, _, _) if e.hir_id != child_id => {
                    is_ty_unified = true;
                    moved_before_use = true;
                },
                ExprKind::Block(_, Some(_)) | ExprKind::Break(..) => {
                    is_ty_unified = true;
                    moved_before_use = true;
                },
                ExprKind::Block(..) => moved_before_use = true,
                _ => {},
            }
        }
        ControlFlow::Continue(())
    });
    match node {
        Some(ControlFlow::Continue((node, child_id))) => ExprUseCtxt {
            node,
            child_id,
            adjustments,
            is_ty_unified,
            moved_before_use,
            same_ctxt,
        },
        #[allow(unreachable_patterns)]
        Some(ControlFlow::Break(_)) => unreachable!("type of node is ControlFlow<!>"),
        None => ExprUseCtxt {
            node: Node::Crate(cx.tcx.hir_root_module()),
            child_id: HirId::INVALID,
            adjustments: &[],
            is_ty_unified: true,
            moved_before_use: true,
            same_ctxt: false,
        },
    }
}

/// Tokenizes the input while keeping the text associated with each token.
pub fn tokenize_with_text(s: &str) -> impl Iterator<Item = (TokenKind, &str, InnerSpan)> {
    let mut pos = 0;
    tokenize(s).map(move |t| {
        let end = pos + t.len;
        let range = pos as usize..end as usize;
        let inner = InnerSpan::new(range.start, range.end);
        pos = end;
        (t.kind, s.get(range).unwrap_or_default(), inner)
    })
}

/// Checks whether a given span has any comment token
/// This checks for all types of comment: line "//", block "/**", doc "///" "//!"
pub fn span_contains_comment(sm: &SourceMap, span: Span) -> bool {
    let Ok(snippet) = sm.span_to_snippet(span) else {
        return false;
    };
    return tokenize(&snippet).any(|token| {
        matches!(
            token.kind,
            TokenKind::BlockComment { .. } | TokenKind::LineComment { .. }
        )
    });
}

/// Checks whether a given span has any significant token. A significant token is a non-whitespace
/// token, including comments unless `skip_comments` is set.
/// This is useful to determine if there are any actual code tokens in the span that are omitted in
/// the late pass, such as platform-specific code.
pub fn span_contains_non_whitespace(cx: &impl source::HasSession, span: Span, skip_comments: bool) -> bool {
    matches!(span.get_source_text(cx), Some(snippet) if tokenize_with_text(&snippet).any(|(token, _, _)|
        match token {
            TokenKind::Whitespace => false,
            TokenKind::BlockComment { .. } | TokenKind::LineComment { .. } => !skip_comments,
            _ => true,
        }
    ))
}
/// Returns all the comments a given span contains
///
/// Comments are returned wrapped with their relevant delimiters
pub fn span_extract_comment(sm: &SourceMap, span: Span) -> String {
    span_extract_comments(sm, span).join("\n")
}

/// Returns all the comments a given span contains.
///
/// Comments are returned wrapped with their relevant delimiters.
pub fn span_extract_comments(sm: &SourceMap, span: Span) -> Vec<String> {
    let snippet = sm.span_to_snippet(span).unwrap_or_default();
    tokenize_with_text(&snippet)
        .filter(|(t, ..)| matches!(t, TokenKind::BlockComment { .. } | TokenKind::LineComment { .. }))
        .map(|(_, s, _)| s.to_string())
        .collect::<Vec<_>>()
}

pub fn span_find_starting_semi(sm: &SourceMap, span: Span) -> Span {
    sm.span_take_while(span, |&ch| ch == ' ' || ch == ';')
}

/// Returns whether the given let pattern and else body can be turned into the `?` operator
///
/// For this example:
/// ```ignore
/// let FooBar { a, b } = if let Some(a) = ex { a } else { return None };
/// ```
/// We get as parameters:
/// ```ignore
/// pat: Some(a)
/// else_body: return None
/// ```
///
/// And for this example:
/// ```ignore
/// let Some(FooBar { a, b }) = ex else { return None };
/// ```
/// We get as parameters:
/// ```ignore
/// pat: Some(FooBar { a, b })
/// else_body: return None
/// ```
///
/// We output `Some(a)` in the first instance, and `Some(FooBar { a, b })` in the second, because
/// the `?` operator is applicable here. Callers have to check whether we are in a constant or not.
pub fn pat_and_expr_can_be_question_mark<'a, 'hir>(
    cx: &LateContext<'_>,
    pat: &'a Pat<'hir>,
    else_body: &Expr<'_>,
) -> Option<&'a Pat<'hir>> {
    if let PatKind::TupleStruct(pat_path, [inner_pat], _) = pat.kind
        && is_res_lang_ctor(cx, cx.qpath_res(&pat_path, pat.hir_id), OptionSome)
        && !is_refutable(cx, inner_pat)
        && let else_body = peel_blocks(else_body)
        && let ExprKind::Ret(Some(ret_val)) = else_body.kind
        && let ExprKind::Path(ret_path) = ret_val.kind
        && is_res_lang_ctor(cx, cx.qpath_res(&ret_path, ret_val.hir_id), OptionNone)
    {
        Some(inner_pat)
    } else {
        None
    }
}

macro_rules! op_utils {
    ($($name:ident $assign:ident)*) => {
        /// Binary operation traits like `LangItem::Add`
        pub static BINOP_TRAITS: &[LangItem] = &[$(LangItem::$name,)*];

        /// Operator-Assign traits like `LangItem::AddAssign`
        pub static OP_ASSIGN_TRAITS: &[LangItem] = &[$(LangItem::$assign,)*];

        /// Converts `BinOpKind::Add` to `(LangItem::Add, LangItem::AddAssign)`, for example
        pub fn binop_traits(kind: hir::BinOpKind) -> Option<(LangItem, LangItem)> {
            match kind {
                $(hir::BinOpKind::$name => Some((LangItem::$name, LangItem::$assign)),)*
                _ => None,
            }
        }
    };
}

op_utils! {
    Add    AddAssign
    Sub    SubAssign
    Mul    MulAssign
    Div    DivAssign
    Rem    RemAssign
    BitXor BitXorAssign
    BitAnd BitAndAssign
    BitOr  BitOrAssign
    Shl    ShlAssign
    Shr    ShrAssign
}

/// Returns `true` if the pattern is a `PatWild`, or is an ident prefixed with `_`
/// that is not locally used.
pub fn pat_is_wild<'tcx>(cx: &LateContext<'tcx>, pat: &'tcx PatKind<'_>, body: impl Visitable<'tcx>) -> bool {
    match *pat {
        PatKind::Wild => true,
        PatKind::Binding(_, id, ident, None) if ident.as_str().starts_with('_') => {
            !visitors::is_local_used(cx, body, id)
        },
        _ => false,
    }
}

#[derive(Clone, Copy)]
pub enum RequiresSemi {
    Yes,
    No,
}
impl RequiresSemi {
    pub fn requires_semi(self) -> bool {
        matches!(self, Self::Yes)
    }
}

/// Check if the expression return `!`, a type coerced from `!`, or could return `!` if the final
/// expression were turned into a statement.
#[expect(clippy::too_many_lines)]
pub fn is_never_expr<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> Option<RequiresSemi> {
    struct BreakTarget {
        id: HirId,
        unused: bool,
    }

    struct V<'cx, 'tcx> {
        cx: &'cx LateContext<'tcx>,
        break_targets: Vec<BreakTarget>,
        break_targets_for_result_ty: u32,
        in_final_expr: bool,
        requires_semi: bool,
        is_never: bool,
    }

    impl V<'_, '_> {
        fn push_break_target(&mut self, id: HirId) {
            self.break_targets.push(BreakTarget { id, unused: true });
            self.break_targets_for_result_ty += u32::from(self.in_final_expr);
        }
    }

    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            // Note: Part of the complexity here comes from the fact that
            // coercions are applied to the innermost expression.
            // e.g. In `let x: u32 = { break () };` the never-to-any coercion
            // is applied to the break expression. This means we can't just
            // check the block's type as it will be `u32` despite the fact
            // that the block always diverges.

            // The rest of the complexity comes from checking blocks which
            // syntactically return a value, but will always diverge before
            // reaching that point.
            // e.g. In `let x = { foo(panic!()) };` the block's type will be the
            // return type of `foo` even though it will never actually run. This
            // can be trivially fixed by adding a semicolon after the call, but
            // we must first detect that a semicolon is needed to make that
            // suggestion.

            if self.is_never && self.break_targets.is_empty() {
                if self.in_final_expr && !self.requires_semi {
                    // This expression won't ever run, but we still need to check
                    // if it can affect the type of the final expression.
                    match e.kind {
                        ExprKind::DropTemps(e) => self.visit_expr(e),
                        ExprKind::If(_, then, Some(else_)) => {
                            self.visit_expr(then);
                            self.visit_expr(else_);
                        },
                        ExprKind::Match(_, arms, _) => {
                            for arm in arms {
                                self.visit_expr(arm.body);
                            }
                        },
                        ExprKind::Loop(b, ..) => {
                            self.push_break_target(e.hir_id);
                            self.in_final_expr = false;
                            self.visit_block(b);
                            self.break_targets.pop();
                        },
                        ExprKind::Block(b, _) => {
                            if b.targeted_by_break {
                                self.push_break_target(b.hir_id);
                                self.visit_block(b);
                                self.break_targets.pop();
                            } else {
                                self.visit_block(b);
                            }
                        },
                        _ => {
                            self.requires_semi = !self.cx.typeck_results().expr_ty(e).is_never();
                        },
                    }
                }
                return;
            }
            match e.kind {
                ExprKind::DropTemps(e) => self.visit_expr(e),
                ExprKind::Ret(None) | ExprKind::Continue(_) => self.is_never = true,
                ExprKind::Ret(Some(e)) | ExprKind::Become(e) => {
                    self.in_final_expr = false;
                    self.visit_expr(e);
                    self.is_never = true;
                },
                ExprKind::Break(dest, e) => {
                    if let Some(e) = e {
                        self.in_final_expr = false;
                        self.visit_expr(e);
                    }
                    if let Ok(id) = dest.target_id
                        && let Some((i, target)) = self
                            .break_targets
                            .iter_mut()
                            .enumerate()
                            .find(|(_, target)| target.id == id)
                    {
                        target.unused &= self.is_never;
                        if i < self.break_targets_for_result_ty as usize {
                            self.requires_semi = true;
                        }
                    }
                    self.is_never = true;
                },
                ExprKind::If(cond, then, else_) => {
                    let in_final_expr = mem::replace(&mut self.in_final_expr, false);
                    self.visit_expr(cond);
                    self.in_final_expr = in_final_expr;

                    if self.is_never {
                        self.visit_expr(then);
                        if let Some(else_) = else_ {
                            self.visit_expr(else_);
                        }
                    } else {
                        self.visit_expr(then);
                        let is_never = mem::replace(&mut self.is_never, false);
                        if let Some(else_) = else_ {
                            self.visit_expr(else_);
                            self.is_never &= is_never;
                        }
                    }
                },
                ExprKind::Match(scrutinee, arms, _) => {
                    let in_final_expr = mem::replace(&mut self.in_final_expr, false);
                    self.visit_expr(scrutinee);
                    self.in_final_expr = in_final_expr;

                    if self.is_never {
                        for arm in arms {
                            self.visit_arm(arm);
                        }
                    } else {
                        let mut is_never = true;
                        for arm in arms {
                            self.is_never = false;
                            if let Some(guard) = arm.guard {
                                let in_final_expr = mem::replace(&mut self.in_final_expr, false);
                                self.visit_expr(guard);
                                self.in_final_expr = in_final_expr;
                                // The compiler doesn't consider diverging guards as causing the arm to diverge.
                                self.is_never = false;
                            }
                            self.visit_expr(arm.body);
                            is_never &= self.is_never;
                        }
                        self.is_never = is_never;
                    }
                },
                ExprKind::Loop(b, _, _, _) => {
                    self.push_break_target(e.hir_id);
                    self.in_final_expr = false;
                    self.visit_block(b);
                    self.is_never = self.break_targets.pop().unwrap().unused;
                },
                ExprKind::Block(b, _) => {
                    if b.targeted_by_break {
                        self.push_break_target(b.hir_id);
                        self.visit_block(b);
                        self.is_never &= self.break_targets.pop().unwrap().unused;
                    } else {
                        self.visit_block(b);
                    }
                },
                _ => {
                    self.in_final_expr = false;
                    walk_expr(self, e);
                    self.is_never |= self.cx.typeck_results().expr_ty(e).is_never();
                },
            }
        }

        fn visit_block(&mut self, b: &'tcx Block<'_>) {
            let in_final_expr = mem::replace(&mut self.in_final_expr, false);
            for s in b.stmts {
                self.visit_stmt(s);
            }
            self.in_final_expr = in_final_expr;
            if let Some(e) = b.expr {
                self.visit_expr(e);
            }
        }

        fn visit_local(&mut self, l: &'tcx LetStmt<'_>) {
            if let Some(e) = l.init {
                self.visit_expr(e);
            }
            if let Some(else_) = l.els {
                let is_never = self.is_never;
                self.visit_block(else_);
                self.is_never = is_never;
            }
        }

        fn visit_arm(&mut self, arm: &Arm<'tcx>) {
            if let Some(guard) = arm.guard {
                let in_final_expr = mem::replace(&mut self.in_final_expr, false);
                self.visit_expr(guard);
                self.in_final_expr = in_final_expr;
            }
            self.visit_expr(arm.body);
        }
    }

    if cx.typeck_results().expr_ty(e).is_never() {
        Some(RequiresSemi::No)
    } else if let ExprKind::Block(b, _) = e.kind
        && !b.targeted_by_break
        && b.expr.is_none()
    {
        // If a block diverges without a final expression then it's type is `!`.
        None
    } else {
        let mut v = V {
            cx,
            break_targets: Vec::new(),
            break_targets_for_result_ty: 0,
            in_final_expr: true,
            requires_semi: false,
            is_never: false,
        };
        v.visit_expr(e);
        v.is_never
            .then_some(if v.requires_semi && matches!(e.kind, ExprKind::Block(..)) {
                RequiresSemi::Yes
            } else {
                RequiresSemi::No
            })
    }
}

/// Produces a path from a local caller to the type of the called method. Suitable for user
/// output/suggestions.
///
/// Returned path can be either absolute (for methods defined non-locally), or relative (for local
/// methods).
pub fn get_path_from_caller_to_method_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    from: LocalDefId,
    method: DefId,
    args: GenericArgsRef<'tcx>,
) -> String {
    let assoc_item = tcx.associated_item(method);
    let def_id = assoc_item.container_id(tcx);
    match assoc_item.container {
        rustc_ty::AssocItemContainer::Trait => get_path_to_callee(tcx, from, def_id),
        rustc_ty::AssocItemContainer::Impl => {
            let ty = tcx.type_of(def_id).instantiate_identity();
            get_path_to_ty(tcx, from, ty, args)
        },
    }
}

fn get_path_to_ty<'tcx>(tcx: TyCtxt<'tcx>, from: LocalDefId, ty: Ty<'tcx>, args: GenericArgsRef<'tcx>) -> String {
    match ty.kind() {
        rustc_ty::Adt(adt, _) => get_path_to_callee(tcx, from, adt.did()),
        // TODO these types need to be recursively resolved as well
        rustc_ty::Array(..)
        | rustc_ty::Dynamic(..)
        | rustc_ty::Never
        | rustc_ty::RawPtr(_, _)
        | rustc_ty::Ref(..)
        | rustc_ty::Slice(_)
        | rustc_ty::Tuple(_) => format!("<{}>", EarlyBinder::bind(ty).instantiate(tcx, args)),
        _ => ty.to_string(),
    }
}

/// Produce a path from some local caller to the callee. Suitable for user output/suggestions.
fn get_path_to_callee(tcx: TyCtxt<'_>, from: LocalDefId, callee: DefId) -> String {
    // only search for a relative path if the call is fully local
    if callee.is_local() {
        let callee_path = tcx.def_path(callee);
        let caller_path = tcx.def_path(from.to_def_id());
        maybe_get_relative_path(&caller_path, &callee_path, 2)
    } else {
        tcx.def_path_str(callee)
    }
}

/// Tries to produce a relative path from `from` to `to`; if such a path would contain more than
/// `max_super` `super` items, produces an absolute path instead. Both `from` and `to` should be in
/// the local crate.
///
/// Suitable for user output/suggestions.
///
/// This ignores use items, and assumes that the target path is visible from the source
/// path (which _should_ be a reasonable assumption since we in order to be able to use an object of
/// certain type T, T is required to be visible).
///
/// TODO make use of `use` items. Maybe we should have something more sophisticated like
/// rust-analyzer does? <https://docs.rs/ra_ap_hir_def/0.0.169/src/ra_ap_hir_def/find_path.rs.html#19-27>
fn maybe_get_relative_path(from: &DefPath, to: &DefPath, max_super: usize) -> String {
    use itertools::EitherOrBoth::{Both, Left, Right};

    // 1. skip the segments common for both paths (regardless of their type)
    let unique_parts = to
        .data
        .iter()
        .zip_longest(from.data.iter())
        .skip_while(|el| matches!(el, Both(l, r) if l == r))
        .map(|el| match el {
            Both(l, r) => Both(l.data, r.data),
            Left(l) => Left(l.data),
            Right(r) => Right(r.data),
        });

    // 2. for the remaining segments, construct relative path using only mod names and `super`
    let mut go_up_by = 0;
    let mut path = Vec::new();
    for el in unique_parts {
        match el {
            Both(l, r) => {
                // consider:
                // a::b::sym:: ::    refers to
                // c::d::e  ::f::sym
                // result should be super::super::c::d::e::f
                //
                // alternatively:
                // a::b::c  ::d::sym refers to
                // e::f::sym:: ::
                // result should be super::super::super::super::e::f
                if let DefPathData::TypeNs(s) = l {
                    path.push(s.to_string());
                }
                if let DefPathData::TypeNs(_) = r {
                    go_up_by += 1;
                }
            },
            // consider:
            // a::b::sym:: ::    refers to
            // c::d::e  ::f::sym
            // when looking at `f`
            Left(DefPathData::TypeNs(sym)) => path.push(sym.to_string()),
            // consider:
            // a::b::c  ::d::sym refers to
            // e::f::sym:: ::
            // when looking at `d`
            Right(DefPathData::TypeNs(_)) => go_up_by += 1,
            _ => {},
        }
    }

    if go_up_by > max_super {
        // `super` chain would be too long, just use the absolute path instead
        once(String::from("crate"))
            .chain(to.data.iter().filter_map(|el| {
                if let DefPathData::TypeNs(sym) = el.data {
                    Some(sym.to_string())
                } else {
                    None
                }
            }))
            .join("::")
    } else {
        repeat_n(String::from("super"), go_up_by).chain(path).join("::")
    }
}

/// Returns true if the specified `HirId` is the top-level expression of a statement or the only
/// expression in a block.
pub fn is_parent_stmt(cx: &LateContext<'_>, id: HirId) -> bool {
    matches!(
        cx.tcx.parent_hir_node(id),
        Node::Stmt(..) | Node::Block(Block { stmts: [], .. })
    )
}

/// Returns true if the given `expr` is a block or resembled as a block,
/// such as `if`, `loop`, `match` expressions etc.
pub fn is_block_like(expr: &Expr<'_>) -> bool {
    matches!(
        expr.kind,
        ExprKind::Block(..) | ExprKind::ConstBlock(..) | ExprKind::If(..) | ExprKind::Loop(..) | ExprKind::Match(..)
    )
}

/// Returns true if the given `expr` is binary expression that needs to be wrapped in parentheses.
pub fn binary_expr_needs_parentheses(expr: &Expr<'_>) -> bool {
    fn contains_block(expr: &Expr<'_>, is_operand: bool) -> bool {
        match expr.kind {
            ExprKind::Binary(_, lhs, _) | ExprKind::Cast(lhs, _) => contains_block(lhs, true),
            _ if is_block_like(expr) => is_operand,
            _ => false,
        }
    }

    contains_block(expr, false)
}

/// Returns true if the specified expression is in a receiver position.
pub fn is_receiver_of_method_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(parent_expr) = get_parent_expr(cx, expr)
        && let ExprKind::MethodCall(_, receiver, ..) = parent_expr.kind
        && receiver.hir_id == expr.hir_id
    {
        return true;
    }
    false
}

/// Returns true if `expr` creates any temporary whose type references a non-static lifetime and has
/// a significant drop and does not consume it.
pub fn leaks_droppable_temporary_with_limited_lifetime<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    for_each_unconsumed_temporary(cx, expr, |temporary_ty| {
        if temporary_ty.has_significant_drop(cx.tcx, cx.typing_env())
            && temporary_ty
                .walk()
                .any(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(re) if !re.is_static()))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_break()
}

/// Returns true if the specified `expr` requires coercion,
/// meaning that it either has a coercion or propagates a coercion from one of its sub expressions.
///
/// Similar to [`is_adjusted`], this not only checks if an expression's type was adjusted,
/// but also going through extra steps to see if it fits the description of [coercion sites].
///
/// You should used this when you want to avoid suggesting replacing an expression that is currently
/// a coercion site or coercion propagating expression with one that is not.
///
/// [coercion sites]: https://doc.rust-lang.org/stable/reference/type-coercions.html#coercion-sites
pub fn expr_requires_coercion<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> bool {
    let expr_ty_is_adjusted = cx
        .typeck_results()
        .expr_adjustments(expr)
        .iter()
        // ignore `NeverToAny` adjustments, such as `panic!` call.
        .any(|adj| !matches!(adj.kind, Adjust::NeverToAny));
    if expr_ty_is_adjusted {
        return true;
    }

    // Identify coercion sites and recursively check if those sites
    // actually have type adjustments.
    match expr.kind {
        ExprKind::Call(_, args) | ExprKind::MethodCall(_, _, args, _) if let Some(def_id) = fn_def_id(cx, expr) => {
            let fn_sig = cx.tcx.fn_sig(def_id).instantiate_identity();

            if !fn_sig.output().skip_binder().has_type_flags(TypeFlags::HAS_TY_PARAM) {
                return false;
            }

            let self_arg_count = usize::from(matches!(expr.kind, ExprKind::MethodCall(..)));
            let mut args_with_ty_param = {
                fn_sig
                    .inputs()
                    .skip_binder()
                    .iter()
                    .skip(self_arg_count)
                    .zip(args)
                    .filter_map(|(arg_ty, arg)| {
                        if arg_ty.has_type_flags(TypeFlags::HAS_TY_PARAM) {
                            Some(arg)
                        } else {
                            None
                        }
                    })
            };
            args_with_ty_param.any(|arg| expr_requires_coercion(cx, arg))
        },
        // Struct/union initialization.
        ExprKind::Struct(qpath, _, _) => {
            let res = cx.typeck_results().qpath_res(qpath, expr.hir_id);
            if let Some((_, v_def)) = adt_and_variant_of_res(cx, res) {
                let rustc_ty::Adt(_, generic_args) = cx.typeck_results().expr_ty_adjusted(expr).kind() else {
                    // This should never happen, but when it does, not linting is the better option.
                    return true;
                };
                v_def
                    .fields
                    .iter()
                    .any(|field| field.ty(cx.tcx, generic_args).has_type_flags(TypeFlags::HAS_TY_PARAM))
            } else {
                false
            }
        },
        // Function results, including the final line of a block or a `return` expression.
        ExprKind::Block(
            &Block {
                expr: Some(ret_expr), ..
            },
            _,
        )
        | ExprKind::Ret(Some(ret_expr)) => expr_requires_coercion(cx, ret_expr),

        // ===== Coercion-propagation expressions =====
        ExprKind::Array(elems) | ExprKind::Tup(elems) => elems.iter().any(|elem| expr_requires_coercion(cx, elem)),
        // Array but with repeating syntax.
        ExprKind::Repeat(rep_elem, _) => expr_requires_coercion(cx, rep_elem),
        // Others that may contain coercion sites.
        ExprKind::If(_, then, maybe_else) => {
            expr_requires_coercion(cx, then) || maybe_else.is_some_and(|e| expr_requires_coercion(cx, e))
        },
        ExprKind::Match(_, arms, _) => arms
            .iter()
            .map(|arm| arm.body)
            .any(|body| expr_requires_coercion(cx, body)),
        _ => false,
    }
}

/// Returns `true` if `expr` designates a mutable static, a mutable local binding, or an expression
/// that can be owned.
pub fn is_mutable(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(hir_id) = path_to_local(expr)
        && let Node::Pat(pat) = cx.tcx.hir_node(hir_id)
    {
        matches!(pat.kind, PatKind::Binding(BindingMode::MUT, ..))
    } else if let ExprKind::Path(p) = &expr.kind
        && let Some(mutability) = cx
            .qpath_res(p, expr.hir_id)
            .opt_def_id()
            .and_then(|id| cx.tcx.static_mutability(id))
    {
        mutability == Mutability::Mut
    } else if let ExprKind::Field(parent, _) = expr.kind {
        is_mutable(cx, parent)
    } else {
        true
    }
}

/// Peel `Option<…>` from `hir_ty` as long as the HIR name is `Option` and it corresponds to the
/// `core::Option<_>` type.
pub fn peel_hir_ty_options<'tcx>(cx: &LateContext<'tcx>, mut hir_ty: &'tcx hir::Ty<'tcx>) -> &'tcx hir::Ty<'tcx> {
    let Some(option_def_id) = cx.tcx.get_diagnostic_item(sym::Option) else {
        return hir_ty;
    };
    while let TyKind::Path(QPath::Resolved(None, path)) = hir_ty.kind
        && let Some(segment) = path.segments.last()
        && segment.ident.name == sym::Option
        && let Res::Def(DefKind::Enum, def_id) = segment.res
        && def_id == option_def_id
        && let [GenericArg::Type(arg_ty)] = segment.args().args
    {
        hir_ty = arg_ty.as_unambig_ty();
    }
    hir_ty
}

/// If `expr` is a desugared `.await`, return the original expression if it does not come from a
/// macro expansion.
pub fn desugar_await<'tcx>(expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Match(match_value, _, MatchSource::AwaitDesugar) = expr.kind
        && let ExprKind::Call(_, [into_future_arg]) = match_value.kind
        && let ctxt = expr.span.ctxt()
        && for_each_expr_without_closures(into_future_arg, |e| {
            walk_span_to_context(e.span, ctxt).map_or(ControlFlow::Break(()), |_| ControlFlow::Continue(()))
        })
        .is_none()
    {
        Some(into_future_arg)
    } else {
        None
    }
}

/// Checks if the given expression is a call to `Default::default()`.
pub fn is_expr_default<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    if let ExprKind::Call(fn_expr, []) = &expr.kind
        && let ExprKind::Path(qpath) = &fn_expr.kind
        && let Res::Def(_, def_id) = cx.qpath_res(qpath, fn_expr.hir_id)
    {
        cx.tcx.is_diagnostic_item(sym::default_fn, def_id)
    } else {
        false
    }
}
