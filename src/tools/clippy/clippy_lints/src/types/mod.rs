#![allow(rustc::default_hash_types)]

mod borrowed_box;
mod box_vec;
mod linked_list;
mod option_option;
mod rc_buffer;
mod redundant_allocation;
mod utils;
mod vec_box;

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;

use if_chain::if_chain;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_body, walk_expr, walk_ty, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::{
    BinOpKind, Block, Body, Expr, ExprKind, FnDecl, FnRetTy, FnSig, GenericArg, GenericParamKind, HirId, ImplItem,
    ImplItemKind, Item, ItemKind, Local, MatchSource, MutTy, Node, QPath, Stmt, StmtKind, TraitFn, TraitItem,
    TraitItemKind, TyKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, IntTy, Ty, TyS, TypeckResults, UintTy};
use rustc_session::{declare_lint_pass, declare_tool_lint, impl_lint_pass};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::source_map::Span;
use rustc_span::symbol::sym;
use rustc_target::abi::LayoutOf;
use rustc_target::spec::abi::Abi;
use rustc_typeck::hir_ty_to_ty;

use crate::consts::{constant, Constant};
use crate::utils::paths;
use crate::utils::{
    clip, comparisons, differing_macro_contexts, higher, indent_of, int_bits, is_isize_or_usize,
    is_type_diagnostic_item, match_path, multispan_sugg, reindent_multiline, sext, snippet, snippet_opt,
    snippet_with_macro_callsite, span_lint, span_lint_and_help, span_lint_and_then, unsext,
};

declare_clippy_lint! {
    /// **What it does:** Checks for use of `Box<Vec<_>>` anywhere in the code.
    /// Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information.
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
    /// Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information.
    ///
    /// **Why is this bad?** `Vec` already keeps its contents in a separate area on
    /// the heap. So if you `Box` its contents, you just add another level of indirection.
    ///
    /// **Known problems:** Vec<Box<T: Sized>> makes sense if T is a large type (see [#3530](https://github.com/rust-lang/rust-clippy/issues/3530),
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
    /// Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information.
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

declare_clippy_lint! {
    /// **What it does:** Checks for `Rc<T>` and `Arc<T>` when `T` is a mutable buffer type such as `String` or `Vec`.
    ///
    /// **Why is this bad?** Expressions such as `Rc<String>` usually have no advantage over `Rc<str>`, since
    /// it is larger and involves an extra level of indirection, and doesn't implement `Borrow<str>`.
    ///
    /// While mutating a buffer type would still be possible with `Rc::get_mut()`, it only
    /// works if there are no additional references yet, which usually defeats the purpose of
    /// enclosing it in a shared ownership type. Instead, additionally wrapping the inner
    /// type with an interior mutable container (such as `RefCell` or `Mutex`) would normally
    /// be used.
    ///
    /// **Known problems:** This pattern can be desirable to avoid the overhead of a `RefCell` or `Mutex` for
    /// cases where mutation only happens before there are any additional references.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// # use std::rc::Rc;
    /// fn foo(interned: Rc<String>) { ... }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust,ignore
    /// fn foo(interned: Rc<str>) { ... }
    /// ```
    pub RC_BUFFER,
    restriction,
    "shared ownership of a buffer type"
}

pub struct Types {
    vec_box_size_threshold: u64,
}

impl_lint_pass!(Types => [BOX_VEC, VEC_BOX, OPTION_OPTION, LINKEDLIST, BORROWED_BOX, REDUNDANT_ALLOCATION, RC_BUFFER]);

impl<'tcx> LateLintPass<'tcx> for Types {
    fn check_fn(&mut self, cx: &LateContext<'_>, _: FnKind<'_>, decl: &FnDecl<'_>, _: &Body<'_>, _: Span, id: HirId) {
        // Skip trait implementations; see issue #605.
        if let Some(hir::Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_item(id)) {
            if let ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) = item.kind {
                return;
            }
        }

        self.check_fn_decl(cx, decl);
    }

    fn check_field_def(&mut self, cx: &LateContext<'_>, field: &hir::FieldDef<'_>) {
        self.check_ty(cx, &field.ty, false);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, item: &TraitItem<'_>) {
        match item.kind {
            TraitItemKind::Const(ref ty, _) | TraitItemKind::Type(_, Some(ref ty)) => self.check_ty(cx, ty, false),
            TraitItemKind::Fn(ref sig, _) => self.check_fn_decl(cx, &sig.decl),
            _ => (),
        }
    }

    fn check_local(&mut self, cx: &LateContext<'_>, local: &Local<'_>) {
        if let Some(ref ty) = local.ty {
            self.check_ty(cx, ty, true);
        }
    }
}

impl Types {
    pub fn new(vec_box_size_threshold: u64) -> Self {
        Self { vec_box_size_threshold }
    }

    fn check_fn_decl(&mut self, cx: &LateContext<'_>, decl: &FnDecl<'_>) {
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
    /// The parameter `is_local` distinguishes the context of the type.
    fn check_ty(&mut self, cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, is_local: bool) {
        if hir_ty.span.from_expansion() {
            return;
        }
        match hir_ty.kind {
            TyKind::Path(ref qpath) if !is_local => {
                let hir_id = hir_ty.hir_id;
                let res = cx.qpath_res(qpath, hir_id);
                if let Some(def_id) = res.opt_def_id() {
                    let mut triggered = false;
                    triggered |= box_vec::check(cx, hir_ty, qpath, def_id);
                    triggered |= redundant_allocation::check(cx, hir_ty, qpath, def_id);
                    triggered |= rc_buffer::check(cx, hir_ty, qpath, def_id);
                    triggered |= vec_box::check(cx, hir_ty, qpath, def_id, self.vec_box_size_threshold);
                    triggered |= option_option::check(cx, hir_ty, qpath, def_id);
                    triggered |= linked_list::check(cx, hir_ty, def_id);

                    if triggered {
                        return;
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
                    QPath::LangItem(..) => {},
                }
            },
            TyKind::Rptr(ref lt, ref mut_ty) => {
                if !borrowed_box::check(cx, hir_ty, lt, mut_ty) {
                    self.check_ty(cx, &mut_ty.ty, is_local);
                }
            },
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
    pedantic,
    "creating a `let` binding to a value of unit type, which usually can't be used afterwards"
}

declare_lint_pass!(LetUnitValue => [LET_UNIT_VALUE]);

impl<'tcx> LateLintPass<'tcx> for LetUnitValue {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Local(ref local) = stmt.kind {
            if is_unit(cx.typeck_results().pat_ty(&local.pat)) {
                if in_external_macro(cx.sess(), stmt.span) || local.pat.span.from_expansion() {
                    return;
                }
                if higher::is_from_for_desugar(local) {
                    return;
                }
                span_lint_and_then(
                    cx,
                    LET_UNIT_VALUE,
                    stmt.span,
                    "this let-binding has unit value",
                    |diag| {
                        if let Some(expr) = &local.init {
                            let snip = snippet_with_macro_callsite(cx, expr.span, "()");
                            diag.span_suggestion(
                                stmt.span,
                                "omit the `let` binding",
                                format!("{};", snip),
                                Applicability::MachineApplicable, // snippet
                            );
                        }
                    },
                );
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

impl<'tcx> LateLintPass<'tcx> for UnitCmp {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if expr.span.from_expansion() {
            if let Some(callee) = expr.span.source_callee() {
                if let ExpnKind::Macro(MacroKind::Bang, symbol) = callee.kind {
                    if let ExprKind::Binary(ref cmp, ref left, _) = expr.kind {
                        let op = cmp.node;
                        if op.is_comparison() && is_unit(cx.typeck_results().expr_ty(left)) {
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
            if op.is_comparison() && is_unit(cx.typeck_results().expr_ty(left)) {
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

impl<'tcx> LateLintPass<'tcx> for UnitArg {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
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
            ExprKind::Call(_, args) | ExprKind::MethodCall(_, _, args, _) => {
                let args_to_recover = args
                    .iter()
                    .filter(|arg| {
                        if is_unit(cx.typeck_results().expr_ty(arg)) && !is_unit_literal(arg) {
                            !matches!(
                                &arg.kind,
                                ExprKind::Match(.., MatchSource::TryDesugar) | ExprKind::Path(..)
                            )
                        } else {
                            false
                        }
                    })
                    .collect::<Vec<_>>();
                if !args_to_recover.is_empty() {
                    lint_unit_args(cx, expr, &args_to_recover);
                }
            },
            _ => (),
        }
    }
}

fn fmt_stmts_and_call(
    cx: &LateContext<'_>,
    call_expr: &Expr<'_>,
    call_snippet: &str,
    args_snippets: &[impl AsRef<str>],
    non_empty_block_args_snippets: &[impl AsRef<str>],
) -> String {
    let call_expr_indent = indent_of(cx, call_expr.span).unwrap_or(0);
    let call_snippet_with_replacements = args_snippets
        .iter()
        .fold(call_snippet.to_owned(), |acc, arg| acc.replacen(arg.as_ref(), "()", 1));

    let mut stmts_and_call = non_empty_block_args_snippets
        .iter()
        .map(|it| it.as_ref().to_owned())
        .collect::<Vec<_>>();
    stmts_and_call.push(call_snippet_with_replacements);
    stmts_and_call = stmts_and_call
        .into_iter()
        .map(|v| reindent_multiline(v.into(), true, Some(call_expr_indent)).into_owned())
        .collect();

    let mut stmts_and_call_snippet = stmts_and_call.join(&format!("{}{}", ";\n", " ".repeat(call_expr_indent)));
    // expr is not in a block statement or result expression position, wrap in a block
    let parent_node = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(call_expr.hir_id));
    if !matches!(parent_node, Some(Node::Block(_))) && !matches!(parent_node, Some(Node::Stmt(_))) {
        let block_indent = call_expr_indent + 4;
        stmts_and_call_snippet =
            reindent_multiline(stmts_and_call_snippet.into(), true, Some(block_indent)).into_owned();
        stmts_and_call_snippet = format!(
            "{{\n{}{}\n{}}}",
            " ".repeat(block_indent),
            &stmts_and_call_snippet,
            " ".repeat(call_expr_indent)
        );
    }
    stmts_and_call_snippet
}

fn lint_unit_args(cx: &LateContext<'_>, expr: &Expr<'_>, args_to_recover: &[&Expr<'_>]) {
    let mut applicability = Applicability::MachineApplicable;
    let (singular, plural) = if args_to_recover.len() > 1 {
        ("", "s")
    } else {
        ("a ", "")
    };
    span_lint_and_then(
        cx,
        UNIT_ARG,
        expr.span,
        &format!("passing {}unit value{} to a function", singular, plural),
        |db| {
            let mut or = "";
            args_to_recover
                .iter()
                .filter_map(|arg| {
                    if_chain! {
                        if let ExprKind::Block(block, _) = arg.kind;
                        if block.expr.is_none();
                        if let Some(last_stmt) = block.stmts.iter().last();
                        if let StmtKind::Semi(last_expr) = last_stmt.kind;
                        if let Some(snip) = snippet_opt(cx, last_expr.span);
                        then {
                            Some((
                                last_stmt.span,
                                snip,
                            ))
                        }
                        else {
                            None
                        }
                    }
                })
                .for_each(|(span, sugg)| {
                    db.span_suggestion(
                        span,
                        "remove the semicolon from the last statement in the block",
                        sugg,
                        Applicability::MaybeIncorrect,
                    );
                    or = "or ";
                    applicability = Applicability::MaybeIncorrect;
                });

            let arg_snippets: Vec<String> = args_to_recover
                .iter()
                .filter_map(|arg| snippet_opt(cx, arg.span))
                .collect();
            let arg_snippets_without_empty_blocks: Vec<String> = args_to_recover
                .iter()
                .filter(|arg| !is_empty_block(arg))
                .filter_map(|arg| snippet_opt(cx, arg.span))
                .collect();

            if let Some(call_snippet) = snippet_opt(cx, expr.span) {
                let sugg = fmt_stmts_and_call(
                    cx,
                    expr,
                    &call_snippet,
                    &arg_snippets,
                    &arg_snippets_without_empty_blocks,
                );

                if arg_snippets_without_empty_blocks.is_empty() {
                    db.multipart_suggestion(
                        &format!("use {}unit literal{} instead", singular, plural),
                        args_to_recover
                            .iter()
                            .map(|arg| (arg.span, "()".to_string()))
                            .collect::<Vec<_>>(),
                        applicability,
                    );
                } else {
                    let plural = arg_snippets_without_empty_blocks.len() > 1;
                    let empty_or_s = if plural { "s" } else { "" };
                    let it_or_them = if plural { "them" } else { "it" };
                    db.span_suggestion(
                        expr.span,
                        &format!(
                            "{}move the expression{} in front of the call and replace {} with the unit literal `()`",
                            or, empty_or_s, it_or_them
                        ),
                        sugg,
                        applicability,
                    );
                }
            }
        },
    );
}

fn is_empty_block(expr: &Expr<'_>) -> bool {
    matches!(
        expr.kind,
        ExprKind::Block(
            Block {
                stmts: &[],
                expr: None,
                ..
            },
            _,
        )
    )
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
    matches!(ty.kind(), ty::Tuple(slice) if slice.is_empty())
}

fn is_unit_literal(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Tup(ref slice) if slice.is_empty())
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

impl<'tcx> LateLintPass<'tcx> for TypeComplexity {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        _: &'tcx Body<'_>,
        _: Span,
        _: HirId,
    ) {
        self.check_fndecl(cx, decl);
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, field: &'tcx hir::FieldDef<'_>) {
        // enum variants are also struct fields now
        self.check_type(cx, &field.ty);
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        match item.kind {
            ItemKind::Static(ref ty, _, _) | ItemKind::Const(ref ty, _) => self.check_type(cx, ty),
            // functions, enums, structs, impls and traits are covered
            _ => (),
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        match item.kind {
            TraitItemKind::Const(ref ty, _) | TraitItemKind::Type(_, Some(ref ty)) => self.check_type(cx, ty),
            TraitItemKind::Fn(FnSig { ref decl, .. }, TraitFn::Required(_)) => self.check_fndecl(cx, decl),
            // methods with default impl are covered by check_fn
            _ => (),
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        match item.kind {
            ImplItemKind::Const(ref ty, _) | ImplItemKind::TyAlias(ref ty) => self.check_type(cx, ty),
            // methods are covered by check_fn
            _ => (),
        }
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'_>) {
        if let Some(ref ty) = local.ty {
            self.check_type(cx, ty);
        }
    }
}

impl<'tcx> TypeComplexity {
    fn check_fndecl(&self, cx: &LateContext<'tcx>, decl: &'tcx FnDecl<'_>) {
        for arg in decl.inputs {
            self.check_type(cx, arg);
        }
        if let FnRetTy::Return(ref ty) = decl.output {
            self.check_type(cx, ty);
        }
    }

    fn check_type(&self, cx: &LateContext<'_>, ty: &hir::Ty<'_>) {
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

            TyKind::TraitObject(ref param_bounds, ..) => {
                let has_lifetime_parameters = param_bounds.iter().any(|bound| {
                    bound
                        .bound_generic_params
                        .iter()
                        .any(|gen| matches!(gen.kind, GenericParamKind::Lifetime { .. }))
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
    /// if 100 > i32::MAX {}
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

fn is_cast_between_fixed_and_target<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    if let ExprKind::Cast(ref cast_exp, _) = expr.kind {
        let precast_ty = cx.typeck_results().expr_ty(cast_exp);
        let cast_ty = cx.typeck_results().expr_ty(expr);

        return is_isize_or_usize(precast_ty) != is_isize_or_usize(cast_ty);
    }

    false
}

fn detect_absurd_comparison<'tcx>(
    cx: &LateContext<'tcx>,
    op: BinOpKind,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
) -> Option<(ExtremeExpr<'tcx>, AbsurdComparisonResult)> {
    use crate::types::AbsurdComparisonResult::{AlwaysFalse, AlwaysTrue, InequalityImpossible};
    use crate::types::ExtremeType::{Maximum, Minimum};
    use crate::utils::comparisons::{normalize_comparison, Rel};

    // absurd comparison only makes sense on primitive types
    // primitive types don't implement comparison operators with each other
    if cx.typeck_results().expr_ty(lhs) != cx.typeck_results().expr_ty(rhs) {
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

fn detect_extreme_expr<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<ExtremeExpr<'tcx>> {
    use crate::types::ExtremeType::{Maximum, Minimum};

    let ty = cx.typeck_results().expr_ty(expr);

    let cv = constant(cx, cx.typeck_results(), expr)?.0;

    let which = match (ty.kind(), cv) {
        (&ty::Bool, Constant::Bool(false)) | (&ty::Uint(_), Constant::Int(0)) => Minimum,
        (&ty::Int(ity), Constant::Int(i)) if i == unsext(cx.tcx, i128::MIN >> (128 - int_bits(cx.tcx, ity)), ity) => {
            Minimum
        },

        (&ty::Bool, Constant::Bool(true)) => Maximum,
        (&ty::Int(ity), Constant::Int(i)) if i == unsext(cx.tcx, i128::MAX >> (128 - int_bits(cx.tcx, ity)), ity) => {
            Maximum
        },
        (&ty::Uint(uty), Constant::Int(i)) if clip(cx.tcx, u128::MAX, uty) == i => Maximum,

        _ => return None,
    };
    Some(ExtremeExpr { which, expr })
}

impl<'tcx> LateLintPass<'tcx> for AbsurdExtremeComparisons {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
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

                    span_lint_and_help(cx, ABSURD_EXTREME_COMPARISONS, expr.span, msg, None, &help);
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
        } else if u > (i128::MAX as u128) {
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

fn numeric_cast_precast_bounds<'a>(cx: &LateContext<'_>, expr: &'a Expr<'_>) -> Option<(FullInt, FullInt)> {
    if let ExprKind::Cast(ref cast_exp, _) = expr.kind {
        let pre_cast_ty = cx.typeck_results().expr_ty(cast_exp);
        let cast_ty = cx.typeck_results().expr_ty(expr);
        // if it's a cast from i32 to u32 wrapping will invalidate all these checks
        if cx.layout_of(pre_cast_ty).ok().map(|l| l.size) == cx.layout_of(cast_ty).ok().map(|l| l.size) {
            return None;
        }
        match pre_cast_ty.kind() {
            ty::Int(int_ty) => Some(match int_ty {
                IntTy::I8 => (FullInt::S(i128::from(i8::MIN)), FullInt::S(i128::from(i8::MAX))),
                IntTy::I16 => (FullInt::S(i128::from(i16::MIN)), FullInt::S(i128::from(i16::MAX))),
                IntTy::I32 => (FullInt::S(i128::from(i32::MIN)), FullInt::S(i128::from(i32::MAX))),
                IntTy::I64 => (FullInt::S(i128::from(i64::MIN)), FullInt::S(i128::from(i64::MAX))),
                IntTy::I128 => (FullInt::S(i128::MIN), FullInt::S(i128::MAX)),
                IntTy::Isize => (FullInt::S(isize::MIN as i128), FullInt::S(isize::MAX as i128)),
            }),
            ty::Uint(uint_ty) => Some(match uint_ty {
                UintTy::U8 => (FullInt::U(u128::from(u8::MIN)), FullInt::U(u128::from(u8::MAX))),
                UintTy::U16 => (FullInt::U(u128::from(u16::MIN)), FullInt::U(u128::from(u16::MAX))),
                UintTy::U32 => (FullInt::U(u128::from(u32::MIN)), FullInt::U(u128::from(u32::MAX))),
                UintTy::U64 => (FullInt::U(u128::from(u64::MIN)), FullInt::U(u128::from(u64::MAX))),
                UintTy::U128 => (FullInt::U(u128::MIN), FullInt::U(u128::MAX)),
                UintTy::Usize => (FullInt::U(usize::MIN as u128), FullInt::U(usize::MAX as u128)),
            }),
            _ => None,
        }
    } else {
        None
    }
}

fn node_as_const_fullint<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<FullInt> {
    let val = constant(cx, cx.typeck_results(), expr)?.0;
    if let Constant::Int(const_int) = val {
        match *cx.typeck_results().expr_ty(expr).kind() {
            ty::Int(ity) => Some(FullInt::S(sext(cx.tcx, const_int, ity))),
            ty::Uint(_) => Some(FullInt::U(const_int)),
            _ => None,
        }
    } else {
        None
    }
}

fn err_upcast_comparison(cx: &LateContext<'_>, span: Span, expr: &Expr<'_>, always: bool) {
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

fn upcast_comparison_bounds_err<'tcx>(
    cx: &LateContext<'tcx>,
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

impl<'tcx> LateLintPass<'tcx> for InvalidUpcastComparisons {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
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
    pedantic,
    "missing generalization over different hashers"
}

declare_lint_pass!(ImplicitHasher => [IMPLICIT_HASHER]);

impl<'tcx> LateLintPass<'tcx> for ImplicitHasher {
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        use rustc_span::BytePos;

        fn suggestion<'tcx>(
            cx: &LateContext<'tcx>,
            diag: &mut DiagnosticBuilder<'_>,
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
                diag,
                "consider adding a type parameter",
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
                multispan_sugg(diag, "...and use generic constructor", vis.suggestions);
            }
        }

        if !cx.access_levels.is_exported(item.hir_id()) {
            return;
        }

        match item.kind {
            ItemKind::Impl(ref impl_) => {
                let mut vis = ImplicitHasherTypeVisitor::new(cx);
                vis.visit_ty(impl_.self_ty);

                for target in &vis.found {
                    if differing_macro_contexts(item.span, target.span()) {
                        return;
                    }

                    let generics_suggestion_span = impl_.generics.span.substitute_dummy({
                        let pos = snippet_opt(cx, item.span.until(target.span()))
                            .and_then(|snip| Some(item.span.lo() + BytePos(snip.find("impl")? as u32 + 4)));
                        if let Some(pos) = pos {
                            Span::new(pos, pos, item.span.data().ctxt)
                        } else {
                            return;
                        }
                    });

                    let mut ctr_vis = ImplicitHasherConstructorVisitor::new(cx, target);
                    for item in impl_.items.iter().map(|item| cx.tcx.hir().impl_item(item.id)) {
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
                        move |diag| {
                            suggestion(cx, diag, impl_.generics.span, generics_suggestion_span, target, ctr_vis);
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
                            move |diag| {
                                suggestion(cx, diag, generics.span, generics_suggestion_span, target, ctr_vis);
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
    fn new(cx: &LateContext<'tcx>, hir_ty: &hir::Ty<'_>) -> Option<Self> {
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

            if is_type_diagnostic_item(cx, ty, sym::hashmap_type) && params_len == 2 {
                Some(ImplicitHasherType::HashMap(
                    hir_ty.span,
                    ty,
                    snippet(cx, params[0].span, "K"),
                    snippet(cx, params[1].span, "V"),
                ))
            } else if is_type_diagnostic_item(cx, ty, sym::hashset_type) && params_len == 1 {
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
    cx: &'a LateContext<'tcx>,
    found: Vec<ImplicitHasherType<'tcx>>,
}

impl<'a, 'tcx> ImplicitHasherTypeVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
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
    cx: &'a LateContext<'tcx>,
    maybe_typeck_results: Option<&'tcx TypeckResults<'tcx>>,
    target: &'b ImplicitHasherType<'tcx>,
    suggestions: BTreeMap<Span, String>,
}

impl<'a, 'b, 'tcx> ImplicitHasherConstructorVisitor<'a, 'b, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>, target: &'b ImplicitHasherType<'tcx>) -> Self {
        Self {
            cx,
            maybe_typeck_results: cx.maybe_typeck_results(),
            target,
            suggestions: BTreeMap::new(),
        }
    }
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for ImplicitHasherConstructorVisitor<'a, 'b, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_body(&mut self, body: &'tcx Body<'_>) {
        let old_maybe_typeck_results = self.maybe_typeck_results.replace(self.cx.tcx.typeck_body(body.id()));
        walk_body(self, body);
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(ref fun, ref args) = e.kind;
            if let ExprKind::Path(QPath::TypeRelative(ref ty, ref method)) = fun.kind;
            if let TyKind::Path(QPath::Resolved(None, ty_path)) = ty.kind;
            then {
                if !TyS::same_type(self.target.ty(), self.maybe_typeck_results.unwrap().expr_ty(e)) {
                    return;
                }

                if match_path(ty_path, &paths::HASHMAP) {
                    if method.ident.name == sym::new {
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
                    if method.ident.name == sym::new {
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
