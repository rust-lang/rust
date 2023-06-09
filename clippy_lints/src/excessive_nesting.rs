use clippy_utils::{diagnostics::span_lint_and_help, source::snippet};
use rustc_ast::{
    node_id::NodeSet,
    visit::{walk_block, walk_item, Visitor},
    Block, Crate, Inline, Item, ItemKind, ModKind, NodeId,
};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for blocks which are nested beyond a certain threshold.
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if a maximum nesting level is defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// It can severely hinder readability.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// excessive-nesting-threshold = 3
    /// ```
    /// ```rust,ignore
    /// // lib.rs
    /// pub mod a {
    ///     pub struct X;
    ///     impl X {
    ///         pub fn run(&self) {
    ///             if true {
    ///                 // etc...
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// // a.rs
    /// fn private_run(x: &X) {
    ///     if true {
    ///         // etc...
    ///     }
    /// }
    ///
    /// pub struct X;
    /// impl X {
    ///     pub fn run(&self) {
    ///         private_run(self);
    ///     }
    /// }
    /// ```
    /// ```rust,ignore
    /// // lib.rs
    /// pub mod a;
    /// ```
    #[clippy::version = "1.70.0"]
    pub EXCESSIVE_NESTING,
    complexity,
    "checks for blocks nested beyond a certain threshold"
}
impl_lint_pass!(ExcessiveNesting => [EXCESSIVE_NESTING]);

#[derive(Clone)]
pub struct ExcessiveNesting {
    pub excessive_nesting_threshold: u64,
    pub nodes: NodeSet,
}

impl ExcessiveNesting {
    pub fn check_node_id(&self, cx: &EarlyContext<'_>, span: Span, node_id: NodeId) {
        if self.nodes.contains(&node_id) {
            span_lint_and_help(
                cx,
                EXCESSIVE_NESTING,
                span,
                "this block is too nested",
                None,
                "try refactoring your code to minimize nesting",
            );
        }
    }
}

impl EarlyLintPass for ExcessiveNesting {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &Crate) {
        if self.excessive_nesting_threshold == 0 {
            return;
        }

        let mut visitor = NestingVisitor {
            conf: self,
            cx,
            nest_level: 0,
        };

        for item in &krate.items {
            visitor.visit_item(item);
        }
    }

    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &Block) {
        self.check_node_id(cx, block.span, block.id);
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        self.check_node_id(cx, item.span, item.id);
    }
}

struct NestingVisitor<'conf, 'cx> {
    conf: &'conf mut ExcessiveNesting,
    cx: &'cx EarlyContext<'cx>,
    nest_level: u64,
}

impl NestingVisitor<'_, '_> {
    fn check_indent(&mut self, span: Span, id: NodeId) -> bool {
        if self.nest_level > self.conf.excessive_nesting_threshold && !in_external_macro(self.cx.sess(), span) {
            self.conf.nodes.insert(id);

            return true;
        }

        false
    }
}

impl<'conf, 'cx> Visitor<'_> for NestingVisitor<'conf, 'cx> {
    fn visit_block(&mut self, block: &Block) {
        if block.span.from_expansion() {
            return;
        }

        // TODO: This should be rewritten using `LateLintPass` so we can use `is_from_proc_macro` instead,
        // but for now, this is fine.
        let snippet = snippet(self.cx, block.span, "{}").trim().to_owned();
        if !snippet.starts_with('{') || !snippet.ends_with('}') {
            return;
        }

        self.nest_level += 1;

        if !self.check_indent(block.span, block.id) {
            walk_block(self, block);
        }

        self.nest_level -= 1;
    }

    fn visit_item(&mut self, item: &Item) {
        if item.span.from_expansion() {
            return;
        }

        match &item.kind {
            ItemKind::Trait(_) | ItemKind::Impl(_) | ItemKind::Mod(.., ModKind::Loaded(_, Inline::Yes, _)) => {
                self.nest_level += 1;

                if !self.check_indent(item.span, item.id) {
                    walk_item(self, item);
                }

                self.nest_level -= 1;
            },
            // Reset nesting level for non-inline modules (since these are in another file)
            ItemKind::Mod(..) => walk_item(
                &mut NestingVisitor {
                    conf: self.conf,
                    cx: self.cx,
                    nest_level: 0,
                },
                item,
            ),
            _ => walk_item(self, item),
        }
    }
}
