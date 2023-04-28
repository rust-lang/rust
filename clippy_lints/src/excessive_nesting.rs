use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::{
    visit::{walk_block, walk_item, Visitor},
    Block, Crate, Inline, Item, ItemKind, ModKind,
};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for blocks which are indented beyond a certain threshold.
    ///
    /// ### Why is this bad?
    ///
    /// It can severely hinder readability. The default is very generous; if you
    /// exceed this, it's a sign you should refactor.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// excessive-nesting-threshold = 3
    /// ```
    /// lib.rs:
    /// ```rust,ignore
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
    /// Use instead:
    /// a.rs:
    /// ```rust,ignore
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
    #[clippy::version = "1.70.0"]
    pub EXCESSIVE_NESTING,
    restriction,
    "checks for blocks nested beyond a certain threshold"
}
impl_lint_pass!(ExcessiveNesting => [EXCESSIVE_NESTING]);

#[derive(Clone, Copy)]
pub struct ExcessiveNesting {
    pub excessive_nesting_threshold: u64,
}

impl EarlyLintPass for ExcessiveNesting {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &Crate) {
        let conf = self;
        let mut visitor = NestingVisitor {
            conf,
            cx,
            nest_level: 0,
        };

        for item in &krate.items {
            visitor.visit_item(item);
        }
    }
}

struct NestingVisitor<'conf, 'cx> {
    conf: &'conf ExcessiveNesting,
    cx: &'cx EarlyContext<'cx>,
    nest_level: u64,
}

impl NestingVisitor<'_, '_> {
    fn check_indent(&self, span: Span) -> bool {
        if self.nest_level > self.conf.excessive_nesting_threshold && !in_external_macro(self.cx.sess(), span) {
            span_lint_and_help(
                self.cx,
                EXCESSIVE_NESTING,
                span,
                "this block is too nested",
                None,
                "try refactoring your code to minimize nesting",
            );

            return true;
        }

        false
    }
}

impl<'conf, 'cx> Visitor<'_> for NestingVisitor<'conf, 'cx> {
    fn visit_block(&mut self, block: &Block) {
        self.nest_level += 1;

        if !self.check_indent(block.span) {
            walk_block(self, block);
        }

        self.nest_level -= 1;
    }

    fn visit_item(&mut self, item: &Item) {
        match &item.kind {
            ItemKind::Trait(_) | ItemKind::Impl(_) | ItemKind::Mod(.., ModKind::Loaded(_, Inline::Yes, _)) => {
                self.nest_level += 1;

                if !self.check_indent(item.span) {
                    walk_item(self, item);
                }

                self.nest_level -= 1;
            },
            // Mod: Don't visit non-inline modules
            // ForeignMod: I don't think this is necessary, but just incase let's not take any chances (don't want to
            // cause any false positives)
            ItemKind::Mod(..) | ItemKind::ForeignMod(..) => {},
            _ => walk_item(self, item),
        }
    }
}
