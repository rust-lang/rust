use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::is_lint_allowed;
use itertools::Itertools;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{Visitor, walk_block, walk_expr, walk_stmt};
use rustc_hir::{BlockCheckMode, Expr, ExprKind, HirId, Stmt, UnsafeSource};
use rustc_lint::{LateContext, LateLintPass, Level, LintContext};
use rustc_middle::lint::LevelAndSource;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, SyntaxContext, sym};
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

declare_clippy_lint! {
    /// ### What it does
    /// Looks for macros that expand metavariables in an unsafe block.
    ///
    /// ### Why is this bad?
    /// This hides an unsafe block and allows the user of the macro to write unsafe code without an explicit
    /// unsafe block at callsite, making it possible to perform unsafe operations in seemingly safe code.
    ///
    /// The macro should be restructured so that these metavariables are referenced outside of unsafe blocks
    /// and that the usual unsafety checks apply to the macro argument.
    ///
    /// This is usually done by binding it to a variable outside of the unsafe block
    /// and then using that variable inside of the block as shown in the example, or by referencing it a second time
    /// in a safe context, e.g. `if false { $expr }`.
    ///
    /// ### Known limitations
    /// Due to how macros are represented in the compiler at the time Clippy runs its lints,
    /// it's not possible to look for metavariables in macro definitions directly.
    ///
    /// Instead, this lint looks at expansions of macros.
    /// This leads to false negatives for macros that are never actually invoked.
    ///
    /// By default, this lint is rather conservative and will only emit warnings on publicly-exported
    /// macros from the same crate, because oftentimes private internal macros are one-off macros where
    /// this lint would just be noise (e.g. macros that generate `impl` blocks).
    /// The default behavior should help with preventing a high number of such false positives,
    /// however it can be configured to also emit warnings in private macros if desired.
    ///
    /// ### Example
    /// ```no_run
    /// /// Gets the first element of a slice
    /// macro_rules! first {
    ///     ($slice:expr) => {
    ///         unsafe {
    ///             let slice = $slice; // ⚠️ expansion inside of `unsafe {}`
    ///
    ///             assert!(!slice.is_empty());
    ///             // SAFETY: slice is checked to have at least one element
    ///             slice.first().unwrap_unchecked()
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(*first!(&[1i32]), 1);
    ///
    /// // This will compile as a consequence (note the lack of `unsafe {}`)
    /// assert_eq!(*first!(std::hint::unreachable_unchecked() as &[i32]), 1);
    /// ```
    /// Use instead:
    /// ```compile_fail
    /// macro_rules! first {
    ///     ($slice:expr) => {{
    ///         let slice = $slice; // ✅ outside of `unsafe {}`
    ///         unsafe {
    ///             assert!(!slice.is_empty());
    ///             // SAFETY: slice is checked to have at least one element
    ///             slice.first().unwrap_unchecked()
    ///         }
    ///     }}
    /// }
    ///
    /// assert_eq!(*first!(&[1]), 1);
    ///
    /// // This won't compile:
    /// assert_eq!(*first!(std::hint::unreachable_unchecked() as &[i32]), 1);
    /// ```
    #[clippy::version = "1.80.0"]
    pub MACRO_METAVARS_IN_UNSAFE,
    suspicious,
    "expanding macro metavariables in an unsafe block"
}
impl_lint_pass!(ExprMetavarsInUnsafe => [MACRO_METAVARS_IN_UNSAFE]);

#[derive(Clone, Debug)]
pub enum MetavarState {
    ReferencedInUnsafe { unsafe_blocks: Vec<HirId> },
    ReferencedInSafe,
}

pub struct ExprMetavarsInUnsafe {
    warn_unsafe_macro_metavars_in_private_macros: bool,
    /// A metavariable can be expanded more than once, potentially across multiple bodies, so it
    /// requires some state kept across HIR nodes to make it possible to delay a warning
    /// and later undo:
    ///
    /// ```ignore
    /// macro_rules! x {
    ///     ($v:expr) => {
    ///         unsafe { $v; } // unsafe context, it might be possible to emit a warning here, so add it to the map
    ///
    ///         $v;            // `$v` expanded another time but in a safe context, set to ReferencedInSafe to suppress
    ///     }
    /// }
    /// ```
    metavar_expns: BTreeMap<Span, MetavarState>,
}

impl ExprMetavarsInUnsafe {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            warn_unsafe_macro_metavars_in_private_macros: conf.warn_unsafe_macro_metavars_in_private_macros,
            metavar_expns: BTreeMap::new(),
        }
    }
}

struct BodyVisitor<'a, 'tcx> {
    /// Stack of unsafe blocks -- the top item always represents the last seen unsafe block from
    /// within a relevant macro.
    macro_unsafe_blocks: Vec<HirId>,
    /// When this is >0, it means that the node currently being visited is "within" a
    /// macro definition.
    /// This is used to detect if an expression represents a metavariable.
    ///
    /// For example, the following pre-expansion code that we want to lint
    /// ```ignore
    /// macro_rules! m { ($e:expr) => { unsafe { $e; } } }
    /// m!(1);
    /// ```
    /// would look like this post-expansion code:
    /// ```ignore
    /// unsafe { /* macro */
    ///     1 /* root */; /* macro */
    /// }
    /// ```
    /// Visiting the block and the statement will increment the `expn_depth` so that it is >0,
    /// and visiting the expression with a root context while `expn_depth > 0` tells us
    /// that it must be a metavariable.
    expn_depth: u32,
    cx: &'a LateContext<'tcx>,
    lint: &'a mut ExprMetavarsInUnsafe,
}

fn is_public_macro(cx: &LateContext<'_>, def_id: LocalDefId) -> bool {
    (cx.effective_visibilities.is_exported(def_id) || cx.tcx.has_attr(def_id, sym::macro_export))
        && !cx.tcx.is_doc_hidden(def_id)
}

impl<'tcx> Visitor<'tcx> for BodyVisitor<'_, 'tcx> {
    fn visit_stmt(&mut self, s: &'tcx Stmt<'tcx>) {
        let from_expn = s.span.from_expansion();
        if from_expn {
            self.expn_depth += 1;
        }
        walk_stmt(self, s);
        if from_expn {
            self.expn_depth -= 1;
        }
    }

    fn visit_expr(&mut self, e: &'tcx Expr<'tcx>) {
        let ctxt = e.span.ctxt();

        if let ExprKind::Block(block, _) = e.kind
            && let BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided) = block.rules
            && !ctxt.is_root()
            && let Some(macro_def_id) = ctxt.outer_expn_data().macro_def_id
            && let Some(macro_def_id) = macro_def_id.as_local()
            && (self.lint.warn_unsafe_macro_metavars_in_private_macros || is_public_macro(self.cx, macro_def_id))
        {
            self.macro_unsafe_blocks.push(block.hir_id);
            self.expn_depth += 1;
            walk_block(self, block);
            self.expn_depth -= 1;
            self.macro_unsafe_blocks.pop();
        } else if ctxt.is_root() && self.expn_depth > 0 {
            let unsafe_block = self.macro_unsafe_blocks.last().copied();

            match (self.lint.metavar_expns.entry(e.span), unsafe_block) {
                (Entry::Vacant(e), None) => {
                    e.insert(MetavarState::ReferencedInSafe);
                },
                (Entry::Vacant(e), Some(unsafe_block)) => {
                    e.insert(MetavarState::ReferencedInUnsafe {
                        unsafe_blocks: vec![unsafe_block],
                    });
                },
                (Entry::Occupied(mut e), None) => {
                    if let MetavarState::ReferencedInUnsafe { .. } = *e.get() {
                        e.insert(MetavarState::ReferencedInSafe);
                    }
                },
                (Entry::Occupied(mut e), Some(unsafe_block)) => {
                    if let MetavarState::ReferencedInUnsafe { unsafe_blocks } = e.get_mut()
                        && !unsafe_blocks.contains(&unsafe_block)
                    {
                        unsafe_blocks.push(unsafe_block);
                    }
                },
            }

            // NB: No need to visit descendant nodes. They're guaranteed to represent the same
            // metavariable
        } else {
            walk_expr(self, e);
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ExprMetavarsInUnsafe {
    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &rustc_hir::Body<'tcx>) {
        if is_lint_allowed(cx, MACRO_METAVARS_IN_UNSAFE, body.value.hir_id) {
            return;
        }

        // This BodyVisitor is separate and not part of the lint pass because there is no
        // `check_stmt_post` on `(Late)LintPass`, which we'd need to detect when we're leaving a macro span

        let mut vis = BodyVisitor {
            macro_unsafe_blocks: Vec::new(),
            #[expect(clippy::bool_to_int_with_if)] // obfuscates the meaning
            expn_depth: if body.value.span.from_expansion() { 1 } else { 0 },
            cx,
            lint: self
        };
        vis.visit_body(body);
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        // Aggregate all unsafe blocks from all spans:
        // ```
        // macro_rules! x {
        //   ($w:expr, $x:expr, $y:expr) => { $w; unsafe { $w; $x; }; unsafe { $x; $y; }; }
        // }
        // $w: []  (unsafe#0 is never added because it was referenced in a safe context)
        // $x: [unsafe#0, unsafe#1]
        // $y: [unsafe#1]
        // ```
        // We want to lint unsafe blocks #0 and #1
        let bad_unsafe_blocks = self
            .metavar_expns
            .iter()
            .filter_map(|(_, state)| match state {
                MetavarState::ReferencedInUnsafe { unsafe_blocks } => Some(unsafe_blocks.as_slice()),
                MetavarState::ReferencedInSafe => None,
            })
            .flatten()
            .copied()
            .inspect(|&unsafe_block| {
                if let LevelAndSource {
                    level: Level::Expect,
                    lint_id: Some(id),
                    ..
                } = cx.tcx.lint_level_at_node(MACRO_METAVARS_IN_UNSAFE, unsafe_block)
                {
                    // Since we're going to deduplicate expanded unsafe blocks by its enclosing macro definition soon,
                    // which would lead to unfulfilled `#[expect()]`s in all other unsafe blocks that are filtered out
                    // except for the one we emit the warning at, we must manually fulfill the lint
                    // for all unsafe blocks here.
                    cx.fulfill_expectation(id);
                }
            })
            .map(|id| {
                // Remove the syntax context to hide "in this macro invocation" in the diagnostic.
                // The invocation doesn't matter. Also we want to dedupe by the unsafe block and not by anything
                // related to the callsite.
                let span = cx.tcx.hir_span(id);

                (id, Span::new(span.lo(), span.hi(), SyntaxContext::root(), None))
            })
            .dedup_by(|&(_, a), &(_, b)| a == b);

        for (id, span) in bad_unsafe_blocks {
            span_lint_hir_and_then(
                cx,
                MACRO_METAVARS_IN_UNSAFE,
                id,
                span,
                "this macro expands metavariables in an unsafe block",
                |diag| {
                    diag.note("this allows the user of the macro to write unsafe code outside of an unsafe block");
                    diag.help(
                            "consider expanding any metavariables outside of this block, e.g. by storing them in a variable",
                        );
                    diag.help(
                            "... or also expand referenced metavariables in a safe context to require an unsafe block at callsite",
                        );
                },
            );
        }
    }
}
