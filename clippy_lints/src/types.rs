#![allow(rustc::default_hash_types)]

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;

use if_chain::if_chain;
use rustc_ast::ast::{FloatTy, IntTy, LitFloatType, LitIntType, LitKind, UintTy};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_body, walk_expr, walk_ty, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::{
    BinOpKind, Body, Expr, ExprKind, FnDecl, FnRetTy, FnSig, GenericArg, GenericParamKind, HirId, ImplItem,
    ImplItemKind, Item, ItemKind, Lifetime, Local, MatchSource, MutTy, Mutability, QPath, Stmt, StmtKind, TraitFn,
    TraitItem, TraitItemKind, TyKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, InferTy, Ty, TyCtxt, TypeckTables};
use rustc_session::{declare_lint_pass, declare_tool_lint, impl_lint_pass};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::source_map::Span;
use rustc_span::symbol::{sym, Symbol};
use rustc_target::abi::LayoutOf;
use rustc_target::spec::abi::Abi;
use rustc_typeck::hir_ty_to_ty;

use crate::consts::{constant, Constant};
use crate::utils::paths;
use crate::utils::{
    clip, comparisons, differing_macro_contexts, higher, in_constant, int_bits, last_path_segment, match_def_path,
    match_path, method_chain_args, multispan_sugg, numeric_literal::NumericLiteral, qpath_res, same_tys, sext, snippet,
    snippet_opt, snippet_with_applicability, snippet_with_macro_callsite, span_lint, span_lint_and_help,
    span_lint_and_sugg, span_lint_and_then, unsext,
};

declare_clippy_lint! {
    /// **What it does:** Checks for use of `Box<Vec<_>>` anywhere in the code.
    ///
    /// **Why is this bad?** `Vec` already keeps its contents in a separate area on
    /// the heap. So if you `Box` it, you just add another level of indirection
    /// without any benefit whatsoever.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// struct X {
    ///     values: Box<Vec<Foo>>,
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust,ignore
    /// struct X {
    ///     values: Vec<Foo>,
    /// }
    /// ```
    pub BOX_VEC,
    perf,
    "usage of `Box<Vec<T>>`, vector elements are already on the heap"
}

declare_clippy_lint! {
    /// **What it does:** Checks for use of `Vec<Box<T>>` where T: Sized anywhere in the code.
    ///
    /// **Why is this bad?** `Vec` already keeps its contents in a separate area on
    /// the heap. So if you `Box` its contents, you just add another level of indirection.
    ///
    /// **Known problems:** Vec<Box<T: Sized>> makes sense if T is a large type (see #3530,
    /// 1st comment).
    ///
    /// **Example:**
    /// ```rust
    /// struct X {
    ///     values: Vec<Box<i32>>,
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust
    /// struct X {
    ///     values: Vec<i32>,
    /// }
    /// ```
    pub VEC_BOX,
    complexity,
    "usage of `Vec<Box<T>>` where T: Sized, vector elements are already on the heap"
}

declare_clippy_lint! {
    /// **What it does:** Checks for use of `Option<Option<_>>` in function signatures and type
    /// definitions
    ///
    /// **Why is this bad?** `Option<_>` represents an optional value. `Option<Option<_>>`
    /// represents an optional optional value which is logically the same thing as an optional
    /// value but has an unneeded extra level of wrapping.
    ///
    /// If you have a case where `Some(Some(_))`, `Some(None)` and `None` are distinct cases,
    /// consider a custom `enum` instead, with clear names for each case.
    ///
    /// **Known problems:** None.
    ///
    /// **Example**
    /// ```rust
    /// fn get_data() -> Option<Option<u32>> {
    ///     None
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust
    /// pub enum Contents {
    ///     Data(Vec<u8>), // Was Some(Some(Vec<u8>))
    ///     NotYetFetched, // Was Some(None)
    ///     None,          // Was None
    /// }
    ///
    /// fn get_data() -> Contents {
    ///     Contents::None
    /// }
    /// ```
    pub OPTION_OPTION,
    pedantic,
    "usage of `Option<Option<T>>`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of any `LinkedList`, suggesting to use a
    /// `Vec` or a `VecDeque` (formerly called `RingBuf`).
    ///
    /// **Why is this bad?** Gankro says:
    ///
    /// > The TL;DR of `LinkedList` is that it's built on a massive amount of
    /// pointers and indirection.
    /// > It wastes memory, it has terrible cache locality, and is all-around slow.
    /// `RingBuf`, while
    /// > "only" amortized for push/pop, should be faster in the general case for
    /// almost every possible
    /// > workload, and isn't even amortized at all if you can predict the capacity
    /// you need.
    /// >
    /// > `LinkedList`s are only really good if you're doing a lot of merging or
    /// splitting of lists.
    /// > This is because they can just mangle some pointers instead of actually
    /// copying the data. Even
    /// > if you're doing a lot of insertion in the middle of the list, `RingBuf`
    /// can still be better
    /// > because of how expensive it is to seek to the middle of a `LinkedList`.
    ///
    /// **Known problems:** False positives – the instances where using a
    /// `LinkedList` makes sense are few and far between, but they can still happen.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::collections::LinkedList;
    /// let x: LinkedList<usize> = LinkedList::new();
    /// ```
    pub LINKEDLIST,
    pedantic,
    "usage of LinkedList, usually a vector is faster, or a more specialized data structure like a `VecDeque`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for use of `&Box<T>` anywhere in the code.
    ///
    /// **Why is this bad?** Any `&Box<T>` can also be a `&T`, which is more
    /// general.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// fn foo(bar: &Box<T>) { ... }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust,ignore
    /// fn foo(bar: &T) { ... }
    /// ```
    pub BORROWED_BOX,
    complexity,
    "a borrow of a boxed type"
}

declare_clippy_lint! {
    /// **What it does:** Checks for use of redundant allocations anywhere in the code.
    ///
    /// **Why is this bad?** Expressions such as `Rc<&T>`, `Rc<Rc<T>>`, `Rc<Box<T>>`, `Box<&T>`
    /// add an unnecessary level of indirection.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::rc::Rc;
    /// fn foo(bar: Rc<&usize>) {}
    /// ```
    ///
    /// Better:
    ///
    /// ```rust
    /// fn foo(bar: &usize) {}
    /// ```
    pub REDUNDANT_ALLOCATION,
    perf,
    "redundant allocation"
}

pub struct Types {
    vec_box_size_threshold: u64,
}

impl_lint_pass!(Types => [BOX_VEC, VEC_BOX, OPTION_OPTION, LINKEDLIST, BORROWED_BOX, REDUNDANT_ALLOCATION]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Types {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_, '_>,
        _: FnKind<'_>,
        decl: &FnDecl<'_>,
        _: &Body<'_>,
        _: Span,
        id: HirId,
    ) {
        // Skip trait implementations; see issue #605.
        if let Some(hir::Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_item(id)) {
            if let ItemKind::Impl { of_trait: Some(_), .. } = item.kind {
                return;
            }
        }

        self.check_fn_decl(cx, decl);
    }

    fn check_struct_field(&mut self, cx: &LateContext<'_, '_>, field: &hir::StructField<'_>) {
        self.check_ty(cx, &field.ty, false);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_, '_>, item: &TraitItem<'_>) {
        match item.kind {
            TraitItemKind::Const(ref ty, _) | TraitItemKind::Type(_, Some(ref ty)) => self.check_ty(cx, ty, false),
            TraitItemKind::Fn(ref sig, _) => self.check_fn_decl(cx, &sig.decl),
            _ => (),
        }
    }

    fn check_local(&mut self, cx: &LateContext<'_, '_>, local: &Local<'_>) {
        if let Some(ref ty) = local.ty {
            self.check_ty(cx, ty, true);
        }
    }
}

/// Checks if `qpath` has last segment with type parameter matching `path`
fn match_type_parameter(cx: &LateContext<'_, '_>, qpath: &QPath<'_>, path: &[&str]) -> Option<Span> {
    let last = last_path_segment(qpath);
    if_chain! {
        if let Some(ref params) = last.args;
        if !params.parenthesized;
        if let Some(ty) = params.args.iter().find_map(|arg| match arg {
            GenericArg::Type(ty) => Some(ty),
            _ => None,
        });
        if let TyKind::Path(ref qpath) = ty.kind;
        if let Some(did) = qpath_res(cx, qpath, ty.hir_id).opt_def_id();
        if match_def_path(cx, did, path);
        then {
            return Some(ty.span);
        }
    }
    None
}

fn match_borrows_parameter(_cx: &LateContext<'_, '_>, qpath: &QPath<'_>) -> Option<Span> {
    let last = last_path_segment(qpath);
    if_chain! {
        if let Some(ref params) = last.args;
        if !params.parenthesized;
        if let Some(ty) = params.args.iter().find_map(|arg| match arg {
            GenericArg::Type(ty) => Some(ty),
            _ => None,
        });
        if let TyKind::Rptr(..) = ty.kind;
        then {
            return Some(ty.span);
        }
    }
    None
}

impl Types {
    pub fn new(vec_box_size_threshold: u64) -> Self {
        Self { vec_box_size_threshold }
    }

    fn check_fn_decl(&mut self, cx: &LateContext<'_, '_>, decl: &FnDecl<'_>) {
        for input in decl.inputs {
            self.check_ty(cx, input, false);
        }

        if let FnRetTy::Return(ref ty) = decl.output {
            self.check_ty(cx, ty, false);
        }
    }

    /// Recursively check for `TypePass` lints in the given type. Stop at the first
    /// lint found.
    ///
    /// The parameter `is_local` distinguishes the context of the type; types from
    /// local bindings should only be checked for the `BORROWED_BOX` lint.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cognitive_complexity)]
    fn check_ty(&mut self, cx: &LateContext<'_, '_>, hir_ty: &hir::Ty<'_>, is_local: bool) {
        if hir_ty.span.from_expansion() {
            return;
        }
        match hir_ty.kind {
            TyKind::Path(ref qpath) if !is_local => {
                let hir_id = hir_ty.hir_id;
                let res = qpath_res(cx, qpath, hir_id);
                if let Some(def_id) = res.opt_def_id() {
                    if Some(def_id) == cx.tcx.lang_items().owned_box() {
                        if let Some(span) = match_borrows_parameter(cx, qpath) {
                            span_lint_and_sugg(
                                cx,
                                REDUNDANT_ALLOCATION,
                                hir_ty.span,
                                "usage of `Box<&T>`",
                                "try",
                                snippet(cx, span, "..").to_string(),
                                Applicability::MachineApplicable,
                            );
                            return; // don't recurse into the type
                        }
                        if match_type_parameter(cx, qpath, &paths::VEC).is_some() {
                            span_lint_and_help(
                                cx,
                                BOX_VEC,
                                hir_ty.span,
                                "you seem to be trying to use `Box<Vec<T>>`. Consider using just `Vec<T>`",
                                "`Vec<T>` is already on the heap, `Box<Vec<T>>` makes an extra allocation.",
                            );
                            return; // don't recurse into the type
                        }
                    } else if Some(def_id) == cx.tcx.lang_items().rc() {
                        if let Some(span) = match_type_parameter(cx, qpath, &paths::RC) {
                            span_lint_and_sugg(
                                cx,
                                REDUNDANT_ALLOCATION,
                                hir_ty.span,
                                "usage of `Rc<Rc<T>>`",
                                "try",
                                snippet(cx, span, "..").to_string(),
                                Applicability::MachineApplicable,
                            );
                            return; // don't recurse into the type
                        }
                        if let Some(span) = match_type_parameter(cx, qpath, &paths::BOX) {
                            span_lint_and_sugg(
                                cx,
                                REDUNDANT_ALLOCATION,
                                hir_ty.span,
                                "usage of `Rc<Box<T>>`",
                                "try",
                                snippet(cx, span, "..").to_string(),
                                Applicability::MachineApplicable,
                            );
                            return; // don't recurse into the type
                        }
                        if let Some(span) = match_borrows_parameter(cx, qpath) {
                            span_lint_and_sugg(
                                cx,
                                REDUNDANT_ALLOCATION,
                                hir_ty.span,
                                "usage of `Rc<&T>`",
                                "try",
                                snippet(cx, span, "..").to_string(),
                                Applicability::MachineApplicable,
                            );
                            return; // don't recurse into the type
                        }
                    } else if cx.tcx.is_diagnostic_item(Symbol::intern("vec_type"), def_id) {
                        if_chain! {
                            // Get the _ part of Vec<_>
                            if let Some(ref last) = last_path_segment(qpath).args;
                            if let Some(ty) = last.args.iter().find_map(|arg| match arg {
                                GenericArg::Type(ty) => Some(ty),
                                _ => None,
                            });
                            // ty is now _ at this point
                            if let TyKind::Path(ref ty_qpath) = ty.kind;
                            let res = qpath_res(cx, ty_qpath, ty.hir_id);
                            if let Some(def_id) = res.opt_def_id();
                            if Some(def_id) == cx.tcx.lang_items().owned_box();
                            // At this point, we know ty is Box<T>, now get T
                            if let Some(ref last) = last_path_segment(ty_qpath).args;
                            if let Some(boxed_ty) = last.args.iter().find_map(|arg| match arg {
                                GenericArg::Type(ty) => Some(ty),
                                _ => None,
                            });
                            let ty_ty = hir_ty_to_ty(cx.tcx, boxed_ty);
                            if ty_ty.is_sized(cx.tcx.at(ty.span), cx.param_env);
                            if let Ok(ty_ty_size) = cx.layout_of(ty_ty).map(|l| l.size.bytes());
                            if ty_ty_size <= self.vec_box_size_threshold;
                            then {
                                span_lint_and_sugg(
                                    cx,
                                    VEC_BOX,
                                    hir_ty.span,
                                    "`Vec<T>` is already on the heap, the boxing is unnecessary.",
                                    "try",
                                    format!("Vec<{}>", ty_ty),
                                    Applicability::MachineApplicable,
                                );
                                return; // don't recurse into the type
                            }
                        }
                    } else if match_def_path(cx, def_id, &paths::OPTION) {
                        if match_type_parameter(cx, qpath, &paths::OPTION).is_some() {
                            span_lint(
                                cx,
                                OPTION_OPTION,
                                hir_ty.span,
                                "consider using `Option<T>` instead of `Option<Option<T>>` or a custom \
                                 enum if you need to distinguish all 3 cases",
                            );
                            return; // don't recurse into the type
                        }
                    } else if match_def_path(cx, def_id, &paths::LINKED_LIST) {
                        span_lint_and_help(
                            cx,
                            LINKEDLIST,
                            hir_ty.span,
                            "I see you're using a LinkedList! Perhaps you meant some other data structure?",
                            "a `VecDeque` might work",
                        );
                        return; // don't recurse into the type
                    }
                }
                match *qpath {
                    QPath::Resolved(Some(ref ty), ref p) => {
                        self.check_ty(cx, ty, is_local);
                        for ty in p.segments.iter().flat_map(|seg| {
                            seg.args
                                .as_ref()
                                .map_or_else(|| [].iter(), |params| params.args.iter())
                                .filter_map(|arg| match arg {
                                    GenericArg::Type(ty) => Some(ty),
                                    _ => None,
                                })
                        }) {
                            self.check_ty(cx, ty, is_local);
                        }
                    },
                    QPath::Resolved(None, ref p) => {
                        for ty in p.segments.iter().flat_map(|seg| {
                            seg.args
                                .as_ref()
                                .map_or_else(|| [].iter(), |params| params.args.iter())
                                .filter_map(|arg| match arg {
                                    GenericArg::Type(ty) => Some(ty),
                                    _ => None,
                                })
                        }) {
                            self.check_ty(cx, ty, is_local);
                        }
                    },
                    QPath::TypeRelative(ref ty, ref seg) => {
                        self.check_ty(cx, ty, is_local);
                        if let Some(ref params) = seg.args {
                            for ty in params.args.iter().filter_map(|arg| match arg {
                                GenericArg::Type(ty) => Some(ty),
                                _ => None,
                            }) {
                                self.check_ty(cx, ty, is_local);
                            }
                        }
                    },
                }
            },
            TyKind::Rptr(ref lt, ref mut_ty) => self.check_ty_rptr(cx, hir_ty, is_local, lt, mut_ty),
            // recurse
            TyKind::Slice(ref ty) | TyKind::Array(ref ty, _) | TyKind::Ptr(MutTy { ref ty, .. }) => {
                self.check_ty(cx, ty, is_local)
            },
            TyKind::Tup(tys) => {
                for ty in tys {
                    self.check_ty(cx, ty, is_local);
                }
            },
            _ => {},
        }
    }

    fn check_ty_rptr(
        &mut self,
        cx: &LateContext<'_, '_>,
        hir_ty: &hir::Ty<'_>,
        is_local: bool,
        lt: &Lifetime,
        mut_ty: &MutTy<'_>,
    ) {
        match mut_ty.ty.kind {
            TyKind::Path(ref qpath) => {
                let hir_id = mut_ty.ty.hir_id;
                let def = qpath_res(cx, qpath, hir_id);
                if_chain! {
                    if let Some(def_id) = def.opt_def_id();
                    if Some(def_id) == cx.tcx.lang_items().owned_box();
                    if let QPath::Resolved(None, ref path) = *qpath;
                    if let [ref bx] = *path.segments;
                    if let Some(ref params) = bx.args;
                    if !params.parenthesized;
                    if let Some(inner) = params.args.iter().find_map(|arg| match arg {
                        GenericArg::Type(ty) => Some(ty),
                        _ => None,
                    });
                    then {
                        if is_any_trait(inner) {
                            // Ignore `Box<Any>` types; see issue #1884 for details.
                            return;
                        }

                        let ltopt = if lt.is_elided() {
                            String::new()
                        } else {
                            format!("{} ", lt.name.ident().as_str())
                        };
                        let mutopt = if mut_ty.mutbl == Mutability::Mut {
                            "mut "
                        } else {
                            ""
                        };
                        let mut applicability = Applicability::MachineApplicable;
                        span_lint_and_sugg(
                            cx,
                            BORROWED_BOX,
                            hir_ty.span,
                            "you seem to be trying to use `&Box<T>`. Consider using just `&T`",
                            "try",
                            format!(
                                "&{}{}{}",
                                ltopt,
                                mutopt,
                                &snippet_with_applicability(cx, inner.span, "..", &mut applicability)
                            ),
                            Applicability::Unspecified,
                        );
                        return; // don't recurse into the type
                    }
                };
                self.check_ty(cx, &mut_ty.ty, is_local);
            },
            _ => self.check_ty(cx, &mut_ty.ty, is_local),
        }
    }
}

// Returns true if given type is `Any` trait.
fn is_any_trait(t: &hir::Ty<'_>) -> bool {
    if_chain! {
        if let TyKind::TraitObject(ref traits, _) = t.kind;
        if !traits.is_empty();
        // Only Send/Sync can be used as additional traits, so it is enough to
        // check only the first trait.
        if match_path(&traits[0].trait_ref.path, &paths::ANY_TRAIT);
        then {
            return true;
        }
    }

    false
}

declare_clippy_lint! {
    /// **What it does:** Checks for binding a unit value.
    ///
    /// **Why is this bad?** A unit value cannot usefully be used anywhere. So
    /// binding one is kind of pointless.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = {
    ///     1;
    /// };
    /// ```
    pub LET_UNIT_VALUE,
    style,
    "creating a `let` binding to a value of unit type, which usually can't be used afterwards"
}

declare_lint_pass!(LetUnitValue => [LET_UNIT_VALUE]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LetUnitValue {
    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Local(ref local) = stmt.kind {
            if is_unit(cx.tables.pat_ty(&local.pat)) {
                if in_external_macro(cx.sess(), stmt.span) || local.pat.span.from_expansion() {
                    return;
                }
                if higher::is_from_for_desugar(local) {
                    return;
                }
                span_lint_and_then(cx, LET_UNIT_VALUE, stmt.span, "this let-binding has unit value", |db| {
                    if let Some(expr) = &local.init {
                        let snip = snippet_with_macro_callsite(cx, expr.span, "()");
                        db.span_suggestion(
                            stmt.span,
                            "omit the `let` binding",
                            format!("{};", snip),
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                });
            }
        }
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for comparisons to unit. This includes all binary
    /// comparisons (like `==` and `<`) and asserts.
    ///
    /// **Why is this bad?** Unit is always equal to itself, and thus is just a
    /// clumsily written constant. Mostly this happens when someone accidentally
    /// adds semicolons at the end of the operands.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// # fn baz() {};
    /// if {
    ///     foo();
    /// } == {
    ///     bar();
    /// } {
    ///     baz();
    /// }
    /// ```
    /// is equal to
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// # fn baz() {};
    /// {
    ///     foo();
    ///     bar();
    ///     baz();
    /// }
    /// ```
    ///
    /// For asserts:
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// assert_eq!({ foo(); }, { bar(); });
    /// ```
    /// will always succeed
    pub UNIT_CMP,
    correctness,
    "comparing unit values"
}

declare_lint_pass!(UnitCmp => [UNIT_CMP]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnitCmp {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'tcx>) {
        if expr.span.from_expansion() {
            if let Some(callee) = expr.span.source_callee() {
                if let ExpnKind::Macro(MacroKind::Bang, symbol) = callee.kind {
                    if let ExprKind::Binary(ref cmp, ref left, _) = expr.kind {
                        let op = cmp.node;
                        if op.is_comparison() && is_unit(cx.tables.expr_ty(left)) {
                            let result = match &*symbol.as_str() {
                                "assert_eq" | "debug_assert_eq" => "succeed",
                                "assert_ne" | "debug_assert_ne" => "fail",
                                _ => return,
                            };
                            span_lint(
                                cx,
                                UNIT_CMP,
                                expr.span,
                                &format!(
                                    "`{}` of unit values detected. This will always {}",
                                    symbol.as_str(),
                                    result
                                ),
                            );
                        }
                    }
                }
            }
            return;
        }
        if let ExprKind::Binary(ref cmp, ref left, _) = expr.kind {
            let op = cmp.node;
            if op.is_comparison() && is_unit(cx.tables.expr_ty(left)) {
                let result = match op {
                    BinOpKind::Eq | BinOpKind::Le | BinOpKind::Ge => "true",
                    _ => "false",
                };
                span_lint(
                    cx,
                    UNIT_CMP,
                    expr.span,
                    &format!(
                        "{}-comparison of unit values detected. This will always be {}",
                        op.as_str(),
                        result
                    ),
                );
            }
        }
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for passing a unit value as an argument to a function without using a
    /// unit literal (`()`).
    ///
    /// **Why is this bad?** This is likely the result of an accidental semicolon.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// foo({
    ///     let a = bar();
    ///     baz(a);
    /// })
    /// ```
    pub UNIT_ARG,
    complexity,
    "passing unit to a function"
}

declare_lint_pass!(UnitArg => [UNIT_ARG]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnitArg {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        // apparently stuff in the desugaring of `?` can trigger this
        // so check for that here
        // only the calls to `Try::from_error` is marked as desugared,
        // so we need to check both the current Expr and its parent.
        if is_questionmark_desugar_marked_call(expr) {
            return;
        }
        if_chain! {
            let map = &cx.tcx.hir();
            let opt_parent_node = map.find(map.get_parent_node(expr.hir_id));
            if let Some(hir::Node::Expr(parent_expr)) = opt_parent_node;
            if is_questionmark_desugar_marked_call(parent_expr);
            then {
                return;
            }
        }

        match expr.kind {
            ExprKind::Call(_, args) | ExprKind::MethodCall(_, _, args) => {
                for arg in args {
                    if is_unit(cx.tables.expr_ty(arg)) && !is_unit_literal(arg) {
                        if let ExprKind::Match(.., match_source) = &arg.kind {
                            if *match_source == MatchSource::TryDesugar {
                                continue;
                            }
                        }

                        span_lint_and_sugg(
                            cx,
                            UNIT_ARG,
                            arg.span,
                            "passing a unit value to a function",
                            "if you intended to pass a unit value, use a unit literal instead",
                            "()".to_string(),
                            Applicability::MachineApplicable,
                        );
                    }
                }
            },
            _ => (),
        }
    }
}

fn is_questionmark_desugar_marked_call(expr: &Expr<'_>) -> bool {
    use rustc_span::hygiene::DesugaringKind;
    if let ExprKind::Call(ref callee, _) = expr.kind {
        callee.span.is_desugaring(DesugaringKind::QuestionMark)
    } else {
        false
    }
}

fn is_unit(ty: Ty<'_>) -> bool {
    match ty.kind {
        ty::Tuple(slice) if slice.is_empty() => true,
        _ => false,
    }
}

fn is_unit_literal(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Tup(ref slice) if slice.is_empty() => true,
        _ => false,
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts from any numerical to a float type where
    /// the receiving type cannot store all values from the original type without
    /// rounding errors. This possible rounding is to be expected, so this lint is
    /// `Allow` by default.
    ///
    /// Basically, this warns on casting any integer with 32 or more bits to `f32`
    /// or any 64-bit integer to `f64`.
    ///
    /// **Why is this bad?** It's not bad at all. But in some applications it can be
    /// helpful to know where precision loss can take place. This lint can help find
    /// those places in the code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = std::u64::MAX;
    /// x as f64;
    /// ```
    pub CAST_PRECISION_LOSS,
    pedantic,
    "casts that cause loss of precision, e.g., `x as f32` where `x: u64`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts from a signed to an unsigned numerical
    /// type. In this case, negative values wrap around to large positive values,
    /// which can be quite surprising in practice. However, as the cast works as
    /// defined, this lint is `Allow` by default.
    ///
    /// **Why is this bad?** Possibly surprising results. You can activate this lint
    /// as a one-time check to see where numerical wrapping can arise.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let y: i8 = -1;
    /// y as u128; // will return 18446744073709551615
    /// ```
    pub CAST_SIGN_LOSS,
    pedantic,
    "casts from signed types to unsigned types, e.g., `x as u32` where `x: i32`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts between numerical types that may
    /// truncate large values. This is expected behavior, so the cast is `Allow` by
    /// default.
    ///
    /// **Why is this bad?** In some problem domains, it is good practice to avoid
    /// truncation. This lint can be activated to help assess where additional
    /// checks could be beneficial.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn as_u8(x: u64) -> u8 {
    ///     x as u8
    /// }
    /// ```
    pub CAST_POSSIBLE_TRUNCATION,
    pedantic,
    "casts that may cause truncation of the value, e.g., `x as u8` where `x: u32`, or `x as i32` where `x: f32`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts from an unsigned type to a signed type of
    /// the same size. Performing such a cast is a 'no-op' for the compiler,
    /// i.e., nothing is changed at the bit level, and the binary representation of
    /// the value is reinterpreted. This can cause wrapping if the value is too big
    /// for the target signed type. However, the cast works as defined, so this lint
    /// is `Allow` by default.
    ///
    /// **Why is this bad?** While such a cast is not bad in itself, the results can
    /// be surprising when this is not the intended behavior, as demonstrated by the
    /// example below.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// std::u32::MAX as i32; // will yield a value of `-1`
    /// ```
    pub CAST_POSSIBLE_WRAP,
    pedantic,
    "casts that may cause wrapping around the value, e.g., `x as i32` where `x: u32` and `x > i32::MAX`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts between numerical types that may
    /// be replaced by safe conversion functions.
    ///
    /// **Why is this bad?** Rust's `as` keyword will perform many kinds of
    /// conversions, including silently lossy conversions. Conversion functions such
    /// as `i32::from` will only perform lossless conversions. Using the conversion
    /// functions prevents conversions from turning into silent lossy conversions if
    /// the types of the input expressions ever change, and make it easier for
    /// people reading the code to know that the conversion is lossless.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn as_u64(x: u8) -> u64 {
    ///     x as u64
    /// }
    /// ```
    ///
    /// Using `::from` would look like this:
    ///
    /// ```rust
    /// fn as_u64(x: u8) -> u64 {
    ///     u64::from(x)
    /// }
    /// ```
    pub CAST_LOSSLESS,
    pedantic,
    "casts using `as` that are known to be lossless, e.g., `x as u64` where `x: u8`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts to the same type.
    ///
    /// **Why is this bad?** It's just unnecessary.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let _ = 2i32 as i32;
    /// ```
    pub UNNECESSARY_CAST,
    complexity,
    "cast to the same type, e.g., `x as i32` where `x: i32`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts from a less-strictly-aligned pointer to a
    /// more-strictly-aligned pointer
    ///
    /// **Why is this bad?** Dereferencing the resulting pointer may be undefined
    /// behavior.
    ///
    /// **Known problems:** Using `std::ptr::read_unaligned` and `std::ptr::write_unaligned` or similar
    /// on the resulting pointer is fine.
    ///
    /// **Example:**
    /// ```rust
    /// let _ = (&1u8 as *const u8) as *const u16;
    /// let _ = (&mut 1u8 as *mut u8) as *mut u16;
    /// ```
    pub CAST_PTR_ALIGNMENT,
    correctness,
    "cast from a pointer to a more-strictly-aligned pointer"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts of function pointers to something other than usize
    ///
    /// **Why is this bad?**
    /// Casting a function pointer to anything other than usize/isize is not portable across
    /// architectures, because you end up losing bits if the target type is too small or end up with a
    /// bunch of extra bits that waste space and add more instructions to the final binary than
    /// strictly necessary for the problem
    ///
    /// Casting to isize also doesn't make sense since there are no signed addresses.
    ///
    /// **Example**
    ///
    /// ```rust
    /// // Bad
    /// fn fun() -> i32 { 1 }
    /// let a = fun as i64;
    ///
    /// // Good
    /// fn fun2() -> i32 { 1 }
    /// let a = fun2 as usize;
    /// ```
    pub FN_TO_NUMERIC_CAST,
    style,
    "casting a function pointer to a numeric type other than usize"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts of a function pointer to a numeric type not wide enough to
    /// store address.
    ///
    /// **Why is this bad?**
    /// Such a cast discards some bits of the function's address. If this is intended, it would be more
    /// clearly expressed by casting to usize first, then casting the usize to the intended type (with
    /// a comment) to perform the truncation.
    ///
    /// **Example**
    ///
    /// ```rust
    /// // Bad
    /// fn fn1() -> i16 {
    ///     1
    /// };
    /// let _ = fn1 as i32;
    ///
    /// // Better: Cast to usize first, then comment with the reason for the truncation
    /// fn fn2() -> i16 {
    ///     1
    /// };
    /// let fn_ptr = fn2 as usize;
    /// let fn_ptr_truncated = fn_ptr as i32;
    /// ```
    pub FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
    style,
    "casting a function pointer to a numeric type not wide enough to store the address"
}

/// Returns the size in bits of an integral type.
/// Will return 0 if the type is not an int or uint variant
fn int_ty_to_nbits(typ: Ty<'_>, tcx: TyCtxt<'_>) -> u64 {
    match typ.kind {
        ty::Int(i) => match i {
            IntTy::Isize => tcx.data_layout.pointer_size.bits(),
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
        },
        ty::Uint(i) => match i {
            UintTy::Usize => tcx.data_layout.pointer_size.bits(),
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
        },
        _ => 0,
    }
}

fn is_isize_or_usize(typ: Ty<'_>) -> bool {
    match typ.kind {
        ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize) => true,
        _ => false,
    }
}

fn span_precision_loss_lint(cx: &LateContext<'_, '_>, expr: &Expr<'_>, cast_from: Ty<'_>, cast_to_f64: bool) {
    let mantissa_nbits = if cast_to_f64 { 52 } else { 23 };
    let arch_dependent = is_isize_or_usize(cast_from) && cast_to_f64;
    let arch_dependent_str = "on targets with 64-bit wide pointers ";
    let from_nbits_str = if arch_dependent {
        "64".to_owned()
    } else if is_isize_or_usize(cast_from) {
        "32 or 64".to_owned()
    } else {
        int_ty_to_nbits(cast_from, cx.tcx).to_string()
    };
    span_lint(
        cx,
        CAST_PRECISION_LOSS,
        expr.span,
        &format!(
            "casting `{0}` to `{1}` causes a loss of precision {2}(`{0}` is {3} bits wide, \
             but `{1}`'s mantissa is only {4} bits wide)",
            cast_from,
            if cast_to_f64 { "f64" } else { "f32" },
            if arch_dependent { arch_dependent_str } else { "" },
            from_nbits_str,
            mantissa_nbits
        ),
    );
}

fn should_strip_parens(op: &Expr<'_>, snip: &str) -> bool {
    if let ExprKind::Binary(_, _, _) = op.kind {
        if snip.starts_with('(') && snip.ends_with(')') {
            return true;
        }
    }
    false
}

fn span_lossless_lint(cx: &LateContext<'_, '_>, expr: &Expr<'_>, op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    // Do not suggest using From in consts/statics until it is valid to do so (see #2267).
    if in_constant(cx, expr.hir_id) {
        return;
    }
    // The suggestion is to use a function call, so if the original expression
    // has parens on the outside, they are no longer needed.
    let mut applicability = Applicability::MachineApplicable;
    let opt = snippet_opt(cx, op.span);
    let sugg = if let Some(ref snip) = opt {
        if should_strip_parens(op, snip) {
            &snip[1..snip.len() - 1]
        } else {
            snip.as_str()
        }
    } else {
        applicability = Applicability::HasPlaceholders;
        ".."
    };

    span_lint_and_sugg(
        cx,
        CAST_LOSSLESS,
        expr.span,
        &format!(
            "casting `{}` to `{}` may become silently lossy if you later change the type",
            cast_from, cast_to
        ),
        "try",
        format!("{}::from({})", cast_to, sugg),
        applicability,
    );
}

enum ArchSuffix {
    _32,
    _64,
    None,
}

fn check_loss_of_sign(cx: &LateContext<'_, '_>, expr: &Expr<'_>, op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    if !cast_from.is_signed() || cast_to.is_signed() {
        return;
    }

    // don't lint for positive constants
    let const_val = constant(cx, &cx.tables, op);
    if_chain! {
        if let Some((const_val, _)) = const_val;
        if let Constant::Int(n) = const_val;
        if let ty::Int(ity) = cast_from.kind;
        if sext(cx.tcx, n, ity) >= 0;
        then {
            return
        }
    }

    // don't lint for the result of methods that always return non-negative values
    if let ExprKind::MethodCall(ref path, _, _) = op.kind {
        let mut method_name = path.ident.name.as_str();
        let whitelisted_methods = ["abs", "checked_abs", "rem_euclid", "checked_rem_euclid"];

        if_chain! {
            if method_name == "unwrap";
            if let Some(arglist) = method_chain_args(op, &["unwrap"]);
            if let ExprKind::MethodCall(ref inner_path, _, _) = &arglist[0][0].kind;
            then {
                method_name = inner_path.ident.name.as_str();
            }
        }

        if whitelisted_methods.iter().any(|&name| method_name == name) {
            return;
        }
    }

    span_lint(
        cx,
        CAST_SIGN_LOSS,
        expr.span,
        &format!(
            "casting `{}` to `{}` may lose the sign of the value",
            cast_from, cast_to
        ),
    );
}

fn check_truncation_and_wrapping(cx: &LateContext<'_, '_>, expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let arch_64_suffix = " on targets with 64-bit wide pointers";
    let arch_32_suffix = " on targets with 32-bit wide pointers";
    let cast_unsigned_to_signed = !cast_from.is_signed() && cast_to.is_signed();
    let from_nbits = int_ty_to_nbits(cast_from, cx.tcx);
    let to_nbits = int_ty_to_nbits(cast_to, cx.tcx);
    let (span_truncation, suffix_truncation, span_wrap, suffix_wrap) =
        match (is_isize_or_usize(cast_from), is_isize_or_usize(cast_to)) {
            (true, true) | (false, false) => (
                to_nbits < from_nbits,
                ArchSuffix::None,
                to_nbits == from_nbits && cast_unsigned_to_signed,
                ArchSuffix::None,
            ),
            (true, false) => (
                to_nbits <= 32,
                if to_nbits == 32 {
                    ArchSuffix::_64
                } else {
                    ArchSuffix::None
                },
                to_nbits <= 32 && cast_unsigned_to_signed,
                ArchSuffix::_32,
            ),
            (false, true) => (
                from_nbits == 64,
                ArchSuffix::_32,
                cast_unsigned_to_signed,
                if from_nbits == 64 {
                    ArchSuffix::_64
                } else {
                    ArchSuffix::_32
                },
            ),
        };
    if span_truncation {
        span_lint(
            cx,
            CAST_POSSIBLE_TRUNCATION,
            expr.span,
            &format!(
                "casting `{}` to `{}` may truncate the value{}",
                cast_from,
                cast_to,
                match suffix_truncation {
                    ArchSuffix::_32 => arch_32_suffix,
                    ArchSuffix::_64 => arch_64_suffix,
                    ArchSuffix::None => "",
                }
            ),
        );
    }
    if span_wrap {
        span_lint(
            cx,
            CAST_POSSIBLE_WRAP,
            expr.span,
            &format!(
                "casting `{}` to `{}` may wrap around the value{}",
                cast_from,
                cast_to,
                match suffix_wrap {
                    ArchSuffix::_32 => arch_32_suffix,
                    ArchSuffix::_64 => arch_64_suffix,
                    ArchSuffix::None => "",
                }
            ),
        );
    }
}

fn check_lossless(cx: &LateContext<'_, '_>, expr: &Expr<'_>, op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let cast_signed_to_unsigned = cast_from.is_signed() && !cast_to.is_signed();
    let from_nbits = int_ty_to_nbits(cast_from, cx.tcx);
    let to_nbits = int_ty_to_nbits(cast_to, cx.tcx);
    if !is_isize_or_usize(cast_from) && !is_isize_or_usize(cast_to) && from_nbits < to_nbits && !cast_signed_to_unsigned
    {
        span_lossless_lint(cx, expr, op, cast_from, cast_to);
    }
}

declare_lint_pass!(Casts => [
    CAST_PRECISION_LOSS,
    CAST_SIGN_LOSS,
    CAST_POSSIBLE_TRUNCATION,
    CAST_POSSIBLE_WRAP,
    CAST_LOSSLESS,
    UNNECESSARY_CAST,
    CAST_PTR_ALIGNMENT,
    FN_TO_NUMERIC_CAST,
    FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
]);

// Check if the given type is either `core::ffi::c_void` or
// one of the platform specific `libc::<platform>::c_void` of libc.
fn is_c_void(cx: &LateContext<'_, '_>, ty: Ty<'_>) -> bool {
    if let ty::Adt(adt, _) = ty.kind {
        let names = cx.get_def_path(adt.did);

        if names.is_empty() {
            return false;
        }
        if names[0] == sym!(libc) || names[0] == sym::core && *names.last().unwrap() == sym!(c_void) {
            return true;
        }
    }
    false
}

/// Returns the mantissa bits wide of a fp type.
/// Will return 0 if the type is not a fp
fn fp_ty_mantissa_nbits(typ: Ty<'_>) -> u32 {
    match typ.kind {
        ty::Float(FloatTy::F32) => 23,
        ty::Float(FloatTy::F64) | ty::Infer(InferTy::FloatVar(_)) => 52,
        _ => 0,
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Casts {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }
        if let ExprKind::Cast(ref ex, _) = expr.kind {
            let (cast_from, cast_to) = (cx.tables.expr_ty(ex), cx.tables.expr_ty(expr));
            lint_fn_to_numeric_cast(cx, expr, ex, cast_from, cast_to);
            if let ExprKind::Lit(ref lit) = ex.kind {
                if_chain! {
                    if let LitKind::Int(n, _) = lit.node;
                    if let Some(src) = snippet_opt(cx, lit.span);
                    if cast_to.is_floating_point();
                    if let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node);
                    let from_nbits = 128 - n.leading_zeros();
                    let to_nbits = fp_ty_mantissa_nbits(cast_to);
                    if from_nbits != 0 && to_nbits != 0 && from_nbits <= to_nbits && num_lit.is_decimal();
                    then {
                        span_lint_and_sugg(
                            cx,
                            UNNECESSARY_CAST,
                            expr.span,
                            &format!("casting integer literal to `{}` is unnecessary", cast_to),
                            "try",
                            format!("{}_{}", n, cast_to),
                            Applicability::MachineApplicable,
                        );
                        return;
                    }
                }
                match lit.node {
                    LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed) => {},
                    _ => {
                        if cast_from.kind == cast_to.kind && !in_external_macro(cx.sess(), expr.span) {
                            span_lint(
                                cx,
                                UNNECESSARY_CAST,
                                expr.span,
                                &format!(
                                    "casting to the same type is unnecessary (`{}` -> `{}`)",
                                    cast_from, cast_to
                                ),
                            );
                        }
                    },
                }
            }
            if cast_from.is_numeric() && cast_to.is_numeric() && !in_external_macro(cx.sess(), expr.span) {
                lint_numeric_casts(cx, expr, ex, cast_from, cast_to);
            }

            lint_cast_ptr_alignment(cx, expr, cast_from, cast_to);
        }
    }
}

fn lint_numeric_casts<'tcx>(
    cx: &LateContext<'_, 'tcx>,
    expr: &Expr<'tcx>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'tcx>,
    cast_to: Ty<'tcx>,
) {
    match (cast_from.is_integral(), cast_to.is_integral()) {
        (true, false) => {
            let from_nbits = int_ty_to_nbits(cast_from, cx.tcx);
            let to_nbits = if let ty::Float(FloatTy::F32) = cast_to.kind {
                32
            } else {
                64
            };
            if is_isize_or_usize(cast_from) || from_nbits >= to_nbits {
                span_precision_loss_lint(cx, expr, cast_from, to_nbits == 64);
            }
            if from_nbits < to_nbits {
                span_lossless_lint(cx, expr, cast_expr, cast_from, cast_to);
            }
        },
        (false, true) => {
            span_lint(
                cx,
                CAST_POSSIBLE_TRUNCATION,
                expr.span,
                &format!("casting `{}` to `{}` may truncate the value", cast_from, cast_to),
            );
            if !cast_to.is_signed() {
                span_lint(
                    cx,
                    CAST_SIGN_LOSS,
                    expr.span,
                    &format!(
                        "casting `{}` to `{}` may lose the sign of the value",
                        cast_from, cast_to
                    ),
                );
            }
        },
        (true, true) => {
            check_loss_of_sign(cx, expr, cast_expr, cast_from, cast_to);
            check_truncation_and_wrapping(cx, expr, cast_from, cast_to);
            check_lossless(cx, expr, cast_expr, cast_from, cast_to);
        },
        (false, false) => {
            if let (&ty::Float(FloatTy::F64), &ty::Float(FloatTy::F32)) = (&cast_from.kind, &cast_to.kind) {
                span_lint(
                    cx,
                    CAST_POSSIBLE_TRUNCATION,
                    expr.span,
                    "casting `f64` to `f32` may truncate the value",
                );
            }
            if let (&ty::Float(FloatTy::F32), &ty::Float(FloatTy::F64)) = (&cast_from.kind, &cast_to.kind) {
                span_lossless_lint(cx, expr, cast_expr, cast_from, cast_to);
            }
        },
    }
}

fn lint_cast_ptr_alignment<'tcx>(cx: &LateContext<'_, 'tcx>, expr: &Expr<'_>, cast_from: Ty<'tcx>, cast_to: Ty<'tcx>) {
    if_chain! {
        if let ty::RawPtr(from_ptr_ty) = &cast_from.kind;
        if let ty::RawPtr(to_ptr_ty) = &cast_to.kind;
        if let Ok(from_layout) = cx.layout_of(from_ptr_ty.ty);
        if let Ok(to_layout) = cx.layout_of(to_ptr_ty.ty);
        if from_layout.align.abi < to_layout.align.abi;
        // with c_void, we inherently need to trust the user
        if !is_c_void(cx, from_ptr_ty.ty);
        // when casting from a ZST, we don't know enough to properly lint
        if !from_layout.is_zst();
        then {
            span_lint(
                cx,
                CAST_PTR_ALIGNMENT,
                expr.span,
                &format!(
                    "casting from `{}` to a more-strictly-aligned pointer (`{}`) ({} < {} bytes)",
                    cast_from,
                    cast_to,
                    from_layout.align.abi.bytes(),
                    to_layout.align.abi.bytes(),
                ),
            );
        }
    }
}

fn lint_fn_to_numeric_cast(
    cx: &LateContext<'_, '_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
) {
    // We only want to check casts to `ty::Uint` or `ty::Int`
    match cast_to.kind {
        ty::Uint(_) | ty::Int(..) => { /* continue on */ },
        _ => return,
    }
    match cast_from.kind {
        ty::FnDef(..) | ty::FnPtr(_) => {
            let mut applicability = Applicability::MaybeIncorrect;
            let from_snippet = snippet_with_applicability(cx, cast_expr.span, "x", &mut applicability);

            let to_nbits = int_ty_to_nbits(cast_to, cx.tcx);
            if to_nbits < cx.tcx.data_layout.pointer_size.bits() {
                span_lint_and_sugg(
                    cx,
                    FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
                    expr.span,
                    &format!(
                        "casting function pointer `{}` to `{}`, which truncates the value",
                        from_snippet, cast_to
                    ),
                    "try",
                    format!("{} as usize", from_snippet),
                    applicability,
                );
            } else if cast_to.kind != ty::Uint(UintTy::Usize) {
                span_lint_and_sugg(
                    cx,
                    FN_TO_NUMERIC_CAST,
                    expr.span,
                    &format!("casting function pointer `{}` to `{}`", from_snippet, cast_to),
                    "try",
                    format!("{} as usize", from_snippet),
                    applicability,
                );
            }
        },
        _ => {},
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for types used in structs, parameters and `let`
    /// declarations above a certain complexity threshold.
    ///
    /// **Why is this bad?** Too complex types make the code less readable. Consider
    /// using a `type` definition to simplify them.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::rc::Rc;
    /// struct Foo {
    ///     inner: Rc<Vec<Vec<Box<(u32, u32, u32, u32)>>>>,
    /// }
    /// ```
    pub TYPE_COMPLEXITY,
    complexity,
    "usage of very complex types that might be better factored into `type` definitions"
}

pub struct TypeComplexity {
    threshold: u64,
}

impl TypeComplexity {
    #[must_use]
    pub fn new(threshold: u64) -> Self {
        Self { threshold }
    }
}

impl_lint_pass!(TypeComplexity => [TYPE_COMPLEXITY]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TypeComplexity {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        _: &'tcx Body<'_>,
        _: Span,
        _: HirId,
    ) {
        self.check_fndecl(cx, decl);
    }

    fn check_struct_field(&mut self, cx: &LateContext<'a, 'tcx>, field: &'tcx hir::StructField<'_>) {
        // enum variants are also struct fields now
        self.check_type(cx, &field.ty);
    }

    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item<'_>) {
        match item.kind {
            ItemKind::Static(ref ty, _, _) | ItemKind::Const(ref ty, _) => self.check_type(cx, ty),
            // functions, enums, structs, impls and traits are covered
            _ => (),
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem<'_>) {
        match item.kind {
            TraitItemKind::Const(ref ty, _) | TraitItemKind::Type(_, Some(ref ty)) => self.check_type(cx, ty),
            TraitItemKind::Fn(FnSig { ref decl, .. }, TraitFn::Required(_)) => self.check_fndecl(cx, decl),
            // methods with default impl are covered by check_fn
            _ => (),
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx ImplItem<'_>) {
        match item.kind {
            ImplItemKind::Const(ref ty, _) | ImplItemKind::TyAlias(ref ty) => self.check_type(cx, ty),
            // methods are covered by check_fn
            _ => (),
        }
    }

    fn check_local(&mut self, cx: &LateContext<'a, 'tcx>, local: &'tcx Local<'_>) {
        if let Some(ref ty) = local.ty {
            self.check_type(cx, ty);
        }
    }
}

impl<'a, 'tcx> TypeComplexity {
    fn check_fndecl(&self, cx: &LateContext<'a, 'tcx>, decl: &'tcx FnDecl<'_>) {
        for arg in decl.inputs {
            self.check_type(cx, arg);
        }
        if let FnRetTy::Return(ref ty) = decl.output {
            self.check_type(cx, ty);
        }
    }

    fn check_type(&self, cx: &LateContext<'_, '_>, ty: &hir::Ty<'_>) {
        if ty.span.from_expansion() {
            return;
        }
        let score = {
            let mut visitor = TypeComplexityVisitor { score: 0, nest: 1 };
            visitor.visit_ty(ty);
            visitor.score
        };

        if score > self.threshold {
            span_lint(
                cx,
                TYPE_COMPLEXITY,
                ty.span,
                "very complex type used. Consider factoring parts into `type` definitions",
            );
        }
    }
}

/// Walks a type and assigns a complexity score to it.
struct TypeComplexityVisitor {
    /// total complexity score of the type
    score: u64,
    /// current nesting level
    nest: u64,
}

impl<'tcx> Visitor<'tcx> for TypeComplexityVisitor {
    type Map = Map<'tcx>;

    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'_>) {
        let (add_score, sub_nest) = match ty.kind {
            // _, &x and *x have only small overhead; don't mess with nesting level
            TyKind::Infer | TyKind::Ptr(..) | TyKind::Rptr(..) => (1, 0),

            // the "normal" components of a type: named types, arrays/tuples
            TyKind::Path(..) | TyKind::Slice(..) | TyKind::Tup(..) | TyKind::Array(..) => (10 * self.nest, 1),

            // function types bring a lot of overhead
            TyKind::BareFn(ref bare) if bare.abi == Abi::Rust => (50 * self.nest, 1),

            TyKind::TraitObject(ref param_bounds, _) => {
                let has_lifetime_parameters = param_bounds.iter().any(|bound| {
                    bound.bound_generic_params.iter().any(|gen| match gen.kind {
                        GenericParamKind::Lifetime { .. } => true,
                        _ => false,
                    })
                });
                if has_lifetime_parameters {
                    // complex trait bounds like A<'a, 'b>
                    (50 * self.nest, 1)
                } else {
                    // simple trait bounds like A + B
                    (20 * self.nest, 0)
                }
            },

            _ => (0, 0),
        };
        self.score += add_score;
        self.nest += sub_nest;
        walk_ty(self, ty);
        self.nest -= sub_nest;
    }
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for expressions where a character literal is cast
    /// to `u8` and suggests using a byte literal instead.
    ///
    /// **Why is this bad?** In general, casting values to smaller types is
    /// error-prone and should be avoided where possible. In the particular case of
    /// converting a character literal to u8, it is easy to avoid by just using a
    /// byte literal instead. As an added bonus, `b'a'` is even slightly shorter
    /// than `'a' as u8`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// 'x' as u8
    /// ```
    ///
    /// A better version, using the byte literal:
    ///
    /// ```rust,ignore
    /// b'x'
    /// ```
    pub CHAR_LIT_AS_U8,
    complexity,
    "casting a character literal to `u8` truncates"
}

declare_lint_pass!(CharLitAsU8 => [CHAR_LIT_AS_U8]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CharLitAsU8 {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if !expr.span.from_expansion();
            if let ExprKind::Cast(e, _) = &expr.kind;
            if let ExprKind::Lit(l) = &e.kind;
            if let LitKind::Char(c) = l.node;
            if ty::Uint(UintTy::U8) == cx.tables.expr_ty(expr).kind;
            then {
                let mut applicability = Applicability::MachineApplicable;
                let snippet = snippet_with_applicability(cx, e.span, "'x'", &mut applicability);

                span_lint_and_then(
                    cx,
                    CHAR_LIT_AS_U8,
                    expr.span,
                    "casting a character literal to `u8` truncates",
                    |db| {
                        db.note("`char` is four bytes wide, but `u8` is a single byte");

                        if c.is_ascii() {
                            db.span_suggestion(
                                expr.span,
                                "use a byte literal instead",
                                format!("b{}", snippet),
                                applicability,
                            );
                        }
                });
            }
        }
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for comparisons where one side of the relation is
    /// either the minimum or maximum value for its type and warns if it involves a
    /// case that is always true or always false. Only integer and boolean types are
    /// checked.
    ///
    /// **Why is this bad?** An expression like `min <= x` may misleadingly imply
    /// that it is possible for `x` to be less than the minimum. Expressions like
    /// `max < x` are probably mistakes.
    ///
    /// **Known problems:** For `usize` the size of the current compile target will
    /// be assumed (e.g., 64 bits on 64 bit systems). This means code that uses such
    /// a comparison to detect target pointer width will trigger this lint. One can
    /// use `mem::sizeof` and compare its value or conditional compilation
    /// attributes
    /// like `#[cfg(target_pointer_width = "64")] ..` instead.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let vec: Vec<isize> = Vec::new();
    /// if vec.len() <= 0 {}
    /// if 100 > std::i32::MAX {}
    /// ```
    pub ABSURD_EXTREME_COMPARISONS,
    correctness,
    "a comparison with a maximum or minimum value that is always true or false"
}

declare_lint_pass!(AbsurdExtremeComparisons => [ABSURD_EXTREME_COMPARISONS]);

enum ExtremeType {
    Minimum,
    Maximum,
}

struct ExtremeExpr<'a> {
    which: ExtremeType,
    expr: &'a Expr<'a>,
}

enum AbsurdComparisonResult {
    AlwaysFalse,
    AlwaysTrue,
    InequalityImpossible,
}

fn is_cast_between_fixed_and_target<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    if let ExprKind::Cast(ref cast_exp, _) = expr.kind {
        let precast_ty = cx.tables.expr_ty(cast_exp);
        let cast_ty = cx.tables.expr_ty(expr);

        return is_isize_or_usize(precast_ty) != is_isize_or_usize(cast_ty);
    }

    false
}

fn detect_absurd_comparison<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    op: BinOpKind,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
) -> Option<(ExtremeExpr<'tcx>, AbsurdComparisonResult)> {
    use crate::types::AbsurdComparisonResult::{AlwaysFalse, AlwaysTrue, InequalityImpossible};
    use crate::types::ExtremeType::{Maximum, Minimum};
    use crate::utils::comparisons::{normalize_comparison, Rel};

    // absurd comparison only makes sense on primitive types
    // primitive types don't implement comparison operators with each other
    if cx.tables.expr_ty(lhs) != cx.tables.expr_ty(rhs) {
        return None;
    }

    // comparisons between fix sized types and target sized types are considered unanalyzable
    if is_cast_between_fixed_and_target(cx, lhs) || is_cast_between_fixed_and_target(cx, rhs) {
        return None;
    }

    let (rel, normalized_lhs, normalized_rhs) = normalize_comparison(op, lhs, rhs)?;

    let lx = detect_extreme_expr(cx, normalized_lhs);
    let rx = detect_extreme_expr(cx, normalized_rhs);

    Some(match rel {
        Rel::Lt => {
            match (lx, rx) {
                (Some(l @ ExtremeExpr { which: Maximum, .. }), _) => (l, AlwaysFalse), // max < x
                (_, Some(r @ ExtremeExpr { which: Minimum, .. })) => (r, AlwaysFalse), // x < min
                _ => return None,
            }
        },
        Rel::Le => {
            match (lx, rx) {
                (Some(l @ ExtremeExpr { which: Minimum, .. }), _) => (l, AlwaysTrue), // min <= x
                (Some(l @ ExtremeExpr { which: Maximum, .. }), _) => (l, InequalityImpossible), // max <= x
                (_, Some(r @ ExtremeExpr { which: Minimum, .. })) => (r, InequalityImpossible), // x <= min
                (_, Some(r @ ExtremeExpr { which: Maximum, .. })) => (r, AlwaysTrue), // x <= max
                _ => return None,
            }
        },
        Rel::Ne | Rel::Eq => return None,
    })
}

fn detect_extreme_expr<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) -> Option<ExtremeExpr<'tcx>> {
    use crate::types::ExtremeType::{Maximum, Minimum};

    let ty = cx.tables.expr_ty(expr);

    let cv = constant(cx, cx.tables, expr)?.0;

    let which = match (&ty.kind, cv) {
        (&ty::Bool, Constant::Bool(false)) | (&ty::Uint(_), Constant::Int(0)) => Minimum,
        (&ty::Int(ity), Constant::Int(i))
            if i == unsext(cx.tcx, i128::min_value() >> (128 - int_bits(cx.tcx, ity)), ity) =>
        {
            Minimum
        },

        (&ty::Bool, Constant::Bool(true)) => Maximum,
        (&ty::Int(ity), Constant::Int(i))
            if i == unsext(cx.tcx, i128::max_value() >> (128 - int_bits(cx.tcx, ity)), ity) =>
        {
            Maximum
        },
        (&ty::Uint(uty), Constant::Int(i)) if clip(cx.tcx, u128::max_value(), uty) == i => Maximum,

        _ => return None,
    };
    Some(ExtremeExpr { which, expr })
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AbsurdExtremeComparisons {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        use crate::types::AbsurdComparisonResult::{AlwaysFalse, AlwaysTrue, InequalityImpossible};
        use crate::types::ExtremeType::{Maximum, Minimum};

        if let ExprKind::Binary(ref cmp, ref lhs, ref rhs) = expr.kind {
            if let Some((culprit, result)) = detect_absurd_comparison(cx, cmp.node, lhs, rhs) {
                if !expr.span.from_expansion() {
                    let msg = "this comparison involving the minimum or maximum element for this \
                               type contains a case that is always true or always false";

                    let conclusion = match result {
                        AlwaysFalse => "this comparison is always false".to_owned(),
                        AlwaysTrue => "this comparison is always true".to_owned(),
                        InequalityImpossible => format!(
                            "the case where the two sides are not equal never occurs, consider using `{} == {}` \
                             instead",
                            snippet(cx, lhs.span, "lhs"),
                            snippet(cx, rhs.span, "rhs")
                        ),
                    };

                    let help = format!(
                        "because `{}` is the {} value for this type, {}",
                        snippet(cx, culprit.expr.span, "x"),
                        match culprit.which {
                            Minimum => "minimum",
                            Maximum => "maximum",
                        },
                        conclusion
                    );

                    span_lint_and_help(cx, ABSURD_EXTREME_COMPARISONS, expr.span, msg, &help);
                }
            }
        }
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for comparisons where the relation is always either
    /// true or false, but where one side has been upcast so that the comparison is
    /// necessary. Only integer types are checked.
    ///
    /// **Why is this bad?** An expression like `let x : u8 = ...; (x as u32) > 300`
    /// will mistakenly imply that it is possible for `x` to be outside the range of
    /// `u8`.
    ///
    /// **Known problems:**
    /// https://github.com/rust-lang/rust-clippy/issues/886
    ///
    /// **Example:**
    /// ```rust
    /// let x: u8 = 1;
    /// (x as u32) > 300;
    /// ```
    pub INVALID_UPCAST_COMPARISONS,
    pedantic,
    "a comparison involving an upcast which is always true or false"
}

declare_lint_pass!(InvalidUpcastComparisons => [INVALID_UPCAST_COMPARISONS]);

#[derive(Copy, Clone, Debug, Eq)]
enum FullInt {
    S(i128),
    U(u128),
}

impl FullInt {
    #[allow(clippy::cast_sign_loss)]
    #[must_use]
    fn cmp_s_u(s: i128, u: u128) -> Ordering {
        if s < 0 {
            Ordering::Less
        } else if u > (i128::max_value() as u128) {
            Ordering::Greater
        } else {
            (s as u128).cmp(&u)
        }
    }
}

impl PartialEq for FullInt {
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other).expect("`partial_cmp` only returns `Some(_)`") == Ordering::Equal
    }
}

impl PartialOrd for FullInt {
    #[must_use]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(match (self, other) {
            (&Self::S(s), &Self::S(o)) => s.cmp(&o),
            (&Self::U(s), &Self::U(o)) => s.cmp(&o),
            (&Self::S(s), &Self::U(o)) => Self::cmp_s_u(s, o),
            (&Self::U(s), &Self::S(o)) => Self::cmp_s_u(o, s).reverse(),
        })
    }
}
impl Ord for FullInt {
    #[must_use]
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("`partial_cmp` for FullInt can never return `None`")
    }
}

fn numeric_cast_precast_bounds<'a>(cx: &LateContext<'_, '_>, expr: &'a Expr<'_>) -> Option<(FullInt, FullInt)> {
    use std::{i128, i16, i32, i64, i8, isize, u128, u16, u32, u64, u8, usize};

    if let ExprKind::Cast(ref cast_exp, _) = expr.kind {
        let pre_cast_ty = cx.tables.expr_ty(cast_exp);
        let cast_ty = cx.tables.expr_ty(expr);
        // if it's a cast from i32 to u32 wrapping will invalidate all these checks
        if cx.layout_of(pre_cast_ty).ok().map(|l| l.size) == cx.layout_of(cast_ty).ok().map(|l| l.size) {
            return None;
        }
        match pre_cast_ty.kind {
            ty::Int(int_ty) => Some(match int_ty {
                IntTy::I8 => (
                    FullInt::S(i128::from(i8::min_value())),
                    FullInt::S(i128::from(i8::max_value())),
                ),
                IntTy::I16 => (
                    FullInt::S(i128::from(i16::min_value())),
                    FullInt::S(i128::from(i16::max_value())),
                ),
                IntTy::I32 => (
                    FullInt::S(i128::from(i32::min_value())),
                    FullInt::S(i128::from(i32::max_value())),
                ),
                IntTy::I64 => (
                    FullInt::S(i128::from(i64::min_value())),
                    FullInt::S(i128::from(i64::max_value())),
                ),
                IntTy::I128 => (FullInt::S(i128::min_value()), FullInt::S(i128::max_value())),
                IntTy::Isize => (
                    FullInt::S(isize::min_value() as i128),
                    FullInt::S(isize::max_value() as i128),
                ),
            }),
            ty::Uint(uint_ty) => Some(match uint_ty {
                UintTy::U8 => (
                    FullInt::U(u128::from(u8::min_value())),
                    FullInt::U(u128::from(u8::max_value())),
                ),
                UintTy::U16 => (
                    FullInt::U(u128::from(u16::min_value())),
                    FullInt::U(u128::from(u16::max_value())),
                ),
                UintTy::U32 => (
                    FullInt::U(u128::from(u32::min_value())),
                    FullInt::U(u128::from(u32::max_value())),
                ),
                UintTy::U64 => (
                    FullInt::U(u128::from(u64::min_value())),
                    FullInt::U(u128::from(u64::max_value())),
                ),
                UintTy::U128 => (FullInt::U(u128::min_value()), FullInt::U(u128::max_value())),
                UintTy::Usize => (
                    FullInt::U(usize::min_value() as u128),
                    FullInt::U(usize::max_value() as u128),
                ),
            }),
            _ => None,
        }
    } else {
        None
    }
}

fn node_as_const_fullint<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) -> Option<FullInt> {
    let val = constant(cx, cx.tables, expr)?.0;
    if let Constant::Int(const_int) = val {
        match cx.tables.expr_ty(expr).kind {
            ty::Int(ity) => Some(FullInt::S(sext(cx.tcx, const_int, ity))),
            ty::Uint(_) => Some(FullInt::U(const_int)),
            _ => None,
        }
    } else {
        None
    }
}

fn err_upcast_comparison(cx: &LateContext<'_, '_>, span: Span, expr: &Expr<'_>, always: bool) {
    if let ExprKind::Cast(ref cast_val, _) = expr.kind {
        span_lint(
            cx,
            INVALID_UPCAST_COMPARISONS,
            span,
            &format!(
                "because of the numeric bounds on `{}` prior to casting, this expression is always {}",
                snippet(cx, cast_val.span, "the expression"),
                if always { "true" } else { "false" },
            ),
        );
    }
}

fn upcast_comparison_bounds_err<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    span: Span,
    rel: comparisons::Rel,
    lhs_bounds: Option<(FullInt, FullInt)>,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
    invert: bool,
) {
    use crate::utils::comparisons::Rel;

    if let Some((lb, ub)) = lhs_bounds {
        if let Some(norm_rhs_val) = node_as_const_fullint(cx, rhs) {
            if rel == Rel::Eq || rel == Rel::Ne {
                if norm_rhs_val < lb || norm_rhs_val > ub {
                    err_upcast_comparison(cx, span, lhs, rel == Rel::Ne);
                }
            } else if match rel {
                Rel::Lt => {
                    if invert {
                        norm_rhs_val < lb
                    } else {
                        ub < norm_rhs_val
                    }
                },
                Rel::Le => {
                    if invert {
                        norm_rhs_val <= lb
                    } else {
                        ub <= norm_rhs_val
                    }
                },
                Rel::Eq | Rel::Ne => unreachable!(),
            } {
                err_upcast_comparison(cx, span, lhs, true)
            } else if match rel {
                Rel::Lt => {
                    if invert {
                        norm_rhs_val >= ub
                    } else {
                        lb >= norm_rhs_val
                    }
                },
                Rel::Le => {
                    if invert {
                        norm_rhs_val > ub
                    } else {
                        lb > norm_rhs_val
                    }
                },
                Rel::Eq | Rel::Ne => unreachable!(),
            } {
                err_upcast_comparison(cx, span, lhs, false)
            }
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for InvalidUpcastComparisons {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref cmp, ref lhs, ref rhs) = expr.kind {
            let normalized = comparisons::normalize_comparison(cmp.node, lhs, rhs);
            let (rel, normalized_lhs, normalized_rhs) = if let Some(val) = normalized {
                val
            } else {
                return;
            };

            let lhs_bounds = numeric_cast_precast_bounds(cx, normalized_lhs);
            let rhs_bounds = numeric_cast_precast_bounds(cx, normalized_rhs);

            upcast_comparison_bounds_err(cx, expr.span, rel, lhs_bounds, normalized_lhs, normalized_rhs, false);
            upcast_comparison_bounds_err(cx, expr.span, rel, rhs_bounds, normalized_rhs, normalized_lhs, true);
        }
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for public `impl` or `fn` missing generalization
    /// over different hashers and implicitly defaulting to the default hashing
    /// algorithm (`SipHash`).
    ///
    /// **Why is this bad?** `HashMap` or `HashSet` with custom hashers cannot be
    /// used with them.
    ///
    /// **Known problems:** Suggestions for replacing constructors can contain
    /// false-positives. Also applying suggestions can require modification of other
    /// pieces of code, possibly including external crates.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::collections::HashMap;
    /// # use std::hash::{Hash, BuildHasher};
    /// # trait Serialize {};
    /// impl<K: Hash + Eq, V> Serialize for HashMap<K, V> { }
    ///
    /// pub fn foo(map: &mut HashMap<i32, i32>) { }
    /// ```
    /// could be rewritten as
    /// ```rust
    /// # use std::collections::HashMap;
    /// # use std::hash::{Hash, BuildHasher};
    /// # trait Serialize {};
    /// impl<K: Hash + Eq, V, S: BuildHasher> Serialize for HashMap<K, V, S> { }
    ///
    /// pub fn foo<S: BuildHasher>(map: &mut HashMap<i32, i32, S>) { }
    /// ```
    pub IMPLICIT_HASHER,
    style,
    "missing generalization over different hashers"
}

declare_lint_pass!(ImplicitHasher => [IMPLICIT_HASHER]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ImplicitHasher {
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item<'_>) {
        use rustc_span::BytePos;

        fn suggestion<'a, 'tcx>(
            cx: &LateContext<'a, 'tcx>,
            db: &mut DiagnosticBuilder<'_>,
            generics_span: Span,
            generics_suggestion_span: Span,
            target: &ImplicitHasherType<'_>,
            vis: ImplicitHasherConstructorVisitor<'_, '_, '_>,
        ) {
            let generics_snip = snippet(cx, generics_span, "");
            // trim `<` `>`
            let generics_snip = if generics_snip.is_empty() {
                ""
            } else {
                &generics_snip[1..generics_snip.len() - 1]
            };

            multispan_sugg(
                db,
                "consider adding a type parameter".to_string(),
                vec![
                    (
                        generics_suggestion_span,
                        format!(
                            "<{}{}S: ::std::hash::BuildHasher{}>",
                            generics_snip,
                            if generics_snip.is_empty() { "" } else { ", " },
                            if vis.suggestions.is_empty() {
                                ""
                            } else {
                                // request users to add `Default` bound so that generic constructors can be used
                                " + Default"
                            },
                        ),
                    ),
                    (
                        target.span(),
                        format!("{}<{}, S>", target.type_name(), target.type_arguments(),),
                    ),
                ],
            );

            if !vis.suggestions.is_empty() {
                multispan_sugg(db, "...and use generic constructor".into(), vis.suggestions);
            }
        }

        if !cx.access_levels.is_exported(item.hir_id) {
            return;
        }

        match item.kind {
            ItemKind::Impl {
                ref generics,
                self_ty: ref ty,
                ref items,
                ..
            } => {
                let mut vis = ImplicitHasherTypeVisitor::new(cx);
                vis.visit_ty(ty);

                for target in &vis.found {
                    if differing_macro_contexts(item.span, target.span()) {
                        return;
                    }

                    let generics_suggestion_span = generics.span.substitute_dummy({
                        let pos = snippet_opt(cx, item.span.until(target.span()))
                            .and_then(|snip| Some(item.span.lo() + BytePos(snip.find("impl")? as u32 + 4)));
                        if let Some(pos) = pos {
                            Span::new(pos, pos, item.span.data().ctxt)
                        } else {
                            return;
                        }
                    });

                    let mut ctr_vis = ImplicitHasherConstructorVisitor::new(cx, target);
                    for item in items.iter().map(|item| cx.tcx.hir().impl_item(item.id)) {
                        ctr_vis.visit_impl_item(item);
                    }

                    span_lint_and_then(
                        cx,
                        IMPLICIT_HASHER,
                        target.span(),
                        &format!(
                            "impl for `{}` should be generalized over different hashers",
                            target.type_name()
                        ),
                        move |db| {
                            suggestion(cx, db, generics.span, generics_suggestion_span, target, ctr_vis);
                        },
                    );
                }
            },
            ItemKind::Fn(ref sig, ref generics, body_id) => {
                let body = cx.tcx.hir().body(body_id);

                for ty in sig.decl.inputs {
                    let mut vis = ImplicitHasherTypeVisitor::new(cx);
                    vis.visit_ty(ty);

                    for target in &vis.found {
                        if in_external_macro(cx.sess(), generics.span) {
                            continue;
                        }
                        let generics_suggestion_span = generics.span.substitute_dummy({
                            let pos = snippet_opt(cx, item.span.until(body.params[0].pat.span))
                                .and_then(|snip| {
                                    let i = snip.find("fn")?;
                                    Some(item.span.lo() + BytePos((i + (&snip[i..]).find('(')?) as u32))
                                })
                                .expect("failed to create span for type parameters");
                            Span::new(pos, pos, item.span.data().ctxt)
                        });

                        let mut ctr_vis = ImplicitHasherConstructorVisitor::new(cx, target);
                        ctr_vis.visit_body(body);

                        span_lint_and_then(
                            cx,
                            IMPLICIT_HASHER,
                            target.span(),
                            &format!(
                                "parameter of type `{}` should be generalized over different hashers",
                                target.type_name()
                            ),
                            move |db| {
                                suggestion(cx, db, generics.span, generics_suggestion_span, target, ctr_vis);
                            },
                        );
                    }
                }
            },
            _ => {},
        }
    }
}

enum ImplicitHasherType<'tcx> {
    HashMap(Span, Ty<'tcx>, Cow<'static, str>, Cow<'static, str>),
    HashSet(Span, Ty<'tcx>, Cow<'static, str>),
}

impl<'tcx> ImplicitHasherType<'tcx> {
    /// Checks that `ty` is a target type without a `BuildHasher`.
    fn new<'a>(cx: &LateContext<'a, 'tcx>, hir_ty: &hir::Ty<'_>) -> Option<Self> {
        if let TyKind::Path(QPath::Resolved(None, ref path)) = hir_ty.kind {
            let params: Vec<_> = path
                .segments
                .last()
                .as_ref()?
                .args
                .as_ref()?
                .args
                .iter()
                .filter_map(|arg| match arg {
                    GenericArg::Type(ty) => Some(ty),
                    _ => None,
                })
                .collect();
            let params_len = params.len();

            let ty = hir_ty_to_ty(cx.tcx, hir_ty);

            if match_path(path, &paths::HASHMAP) && params_len == 2 {
                Some(ImplicitHasherType::HashMap(
                    hir_ty.span,
                    ty,
                    snippet(cx, params[0].span, "K"),
                    snippet(cx, params[1].span, "V"),
                ))
            } else if match_path(path, &paths::HASHSET) && params_len == 1 {
                Some(ImplicitHasherType::HashSet(
                    hir_ty.span,
                    ty,
                    snippet(cx, params[0].span, "T"),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn type_name(&self) -> &'static str {
        match *self {
            ImplicitHasherType::HashMap(..) => "HashMap",
            ImplicitHasherType::HashSet(..) => "HashSet",
        }
    }

    fn type_arguments(&self) -> String {
        match *self {
            ImplicitHasherType::HashMap(.., ref k, ref v) => format!("{}, {}", k, v),
            ImplicitHasherType::HashSet(.., ref t) => format!("{}", t),
        }
    }

    fn ty(&self) -> Ty<'tcx> {
        match *self {
            ImplicitHasherType::HashMap(_, ty, ..) | ImplicitHasherType::HashSet(_, ty, ..) => ty,
        }
    }

    fn span(&self) -> Span {
        match *self {
            ImplicitHasherType::HashMap(span, ..) | ImplicitHasherType::HashSet(span, ..) => span,
        }
    }
}

struct ImplicitHasherTypeVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
    found: Vec<ImplicitHasherType<'tcx>>,
}

impl<'a, 'tcx> ImplicitHasherTypeVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'a, 'tcx>) -> Self {
        Self { cx, found: vec![] }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ImplicitHasherTypeVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_ty(&mut self, t: &'tcx hir::Ty<'_>) {
        if let Some(target) = ImplicitHasherType::new(self.cx, t) {
            self.found.push(target);
        }

        walk_ty(self, t);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

/// Looks for default-hasher-dependent constructors like `HashMap::new`.
struct ImplicitHasherConstructorVisitor<'a, 'b, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
    body: &'a TypeckTables<'tcx>,
    target: &'b ImplicitHasherType<'tcx>,
    suggestions: BTreeMap<Span, String>,
}

impl<'a, 'b, 'tcx> ImplicitHasherConstructorVisitor<'a, 'b, 'tcx> {
    fn new(cx: &'a LateContext<'a, 'tcx>, target: &'b ImplicitHasherType<'tcx>) -> Self {
        Self {
            cx,
            body: cx.tables,
            target,
            suggestions: BTreeMap::new(),
        }
    }
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for ImplicitHasherConstructorVisitor<'a, 'b, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_body(&mut self, body: &'tcx Body<'_>) {
        let prev_body = self.body;
        self.body = self.cx.tcx.body_tables(body.id());
        walk_body(self, body);
        self.body = prev_body;
    }

    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(ref fun, ref args) = e.kind;
            if let ExprKind::Path(QPath::TypeRelative(ref ty, ref method)) = fun.kind;
            if let TyKind::Path(QPath::Resolved(None, ref ty_path)) = ty.kind;
            then {
                if !same_tys(self.cx, self.target.ty(), self.body.expr_ty(e)) {
                    return;
                }

                if match_path(ty_path, &paths::HASHMAP) {
                    if method.ident.name == sym!(new) {
                        self.suggestions
                            .insert(e.span, "HashMap::default()".to_string());
                    } else if method.ident.name == sym!(with_capacity) {
                        self.suggestions.insert(
                            e.span,
                            format!(
                                "HashMap::with_capacity_and_hasher({}, Default::default())",
                                snippet(self.cx, args[0].span, "capacity"),
                            ),
                        );
                    }
                } else if match_path(ty_path, &paths::HASHSET) {
                    if method.ident.name == sym!(new) {
                        self.suggestions
                            .insert(e.span, "HashSet::default()".to_string());
                    } else if method.ident.name == sym!(with_capacity) {
                        self.suggestions.insert(
                            e.span,
                            format!(
                                "HashSet::with_capacity_and_hasher({}, Default::default())",
                                snippet(self.cx, args[0].span, "capacity"),
                            ),
                        );
                    }
                }
            }
        }

        walk_expr(self, e);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts of `&T` to `&mut T` anywhere in the code.
    ///
    /// **Why is this bad?** It’s basically guaranteed to be undefined behaviour.
    /// `UnsafeCell` is the only way to obtain aliasable data that is considered
    /// mutable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// fn x(r: &i32) {
    ///     unsafe {
    ///         *(r as *const _ as *mut _) += 1;
    ///     }
    /// }
    /// ```
    ///
    /// Instead consider using interior mutability types.
    ///
    /// ```rust
    /// use std::cell::UnsafeCell;
    ///
    /// fn x(r: &UnsafeCell<i32>) {
    ///     unsafe {
    ///         *r.get() += 1;
    ///     }
    /// }
    /// ```
    pub CAST_REF_TO_MUT,
    correctness,
    "a cast of reference to a mutable pointer"
}

declare_lint_pass!(RefToMut => [CAST_REF_TO_MUT]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for RefToMut {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Unary(UnOp::UnDeref, e) = &expr.kind;
            if let ExprKind::Cast(e, t) = &e.kind;
            if let TyKind::Ptr(MutTy { mutbl: Mutability::Mut, .. }) = t.kind;
            if let ExprKind::Cast(e, t) = &e.kind;
            if let TyKind::Ptr(MutTy { mutbl: Mutability::Not, .. }) = t.kind;
            if let ty::Ref(..) = cx.tables.node_type(e.hir_id).kind;
            then {
                span_lint(
                    cx,
                    CAST_REF_TO_MUT,
                    expr.span,
                    "casting `&T` to `&mut T` may cause undefined behavior, consider instead using an `UnsafeCell`",
                );
            }
        }
    }
}
