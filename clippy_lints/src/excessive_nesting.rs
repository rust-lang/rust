use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::{
    node_id::NodeId,
    ptr::P,
    visit::{FnKind, Visitor},
    Arm, AssocItemKind, Block, Expr, ExprKind, Inline, Item, ItemKind, Local, LocalKind, ModKind, ModSpans, Pat,
    PatKind, Stmt, StmtKind,
};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
use thin_vec::ThinVec;

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
    /// ### Known issues
    ///
    /// Nested inline modules will all be linted, rather than just the outermost one
    /// that applies. This makes the output a bit verbose.
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
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        let conf = self;
        NestingVisitor {
            conf,
            cx,
            nest_level: 0,
        }
        .visit_item(item);
    }
}

struct NestingVisitor<'conf, 'cx> {
    conf: &'conf ExcessiveNesting,
    cx: &'cx EarlyContext<'cx>,
    nest_level: u64,
}

impl<'conf, 'cx> Visitor<'_> for NestingVisitor<'conf, 'cx> {
    fn visit_local(&mut self, local: &Local) {
        self.visit_pat(&local.pat);

        match &local.kind {
            LocalKind::Init(expr) => self.visit_expr(expr),
            LocalKind::InitElse(expr, block) => {
                self.visit_expr(expr);
                self.visit_block(block);
            },
            LocalKind::Decl => (),
        }
    }

    fn visit_block(&mut self, block: &Block) {
        self.nest_level += 1;

        if !check_indent(self, block.span) {
            for stmt in &block.stmts {
                self.visit_stmt(stmt);
            }
        }

        self.nest_level -= 1;
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Local(local) => self.visit_local(local),
            StmtKind::Item(item) => self.visit_item(item),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => self.visit_expr(expr),
            _ => (),
        }
    }

    fn visit_arm(&mut self, arm: &Arm) {
        self.visit_pat(&arm.pat);
        if let Some(expr) = &arm.guard {
            self.visit_expr(expr);
        }
        self.visit_expr(&arm.body);
    }

    // TODO: Is this necessary?
    fn visit_pat(&mut self, pat: &Pat) {
        match &pat.kind {
            PatKind::Box(pat) | PatKind::Ref(pat, ..) | PatKind::Paren(pat) => self.visit_pat(pat),
            PatKind::Lit(expr) => self.visit_expr(expr),
            PatKind::Range(start, end, ..) => {
                if let Some(expr) = start {
                    self.visit_expr(expr);
                }
                if let Some(expr) = end {
                    self.visit_expr(expr);
                }
            },
            PatKind::Ident(.., pat) if let Some(pat) = pat => {
                self.visit_pat(pat);
            },
            PatKind::Struct(.., pat_fields, _) => {
                for pat_field in pat_fields {
                    self.visit_pat(&pat_field.pat);
                }
            },
            PatKind::TupleStruct(.., pats) | PatKind::Or(pats) | PatKind::Tuple(pats) | PatKind::Slice(pats) => {
                for pat in pats {
                    self.visit_pat(pat);
                }
            },
            _ => (),
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        // This is a mess, but really all it does is extract every expression from every applicable variant
        // of ExprKind until it finds a Block.
        match &expr.kind {
            ExprKind::ConstBlock(anon_const) => self.visit_expr(&anon_const.value),
            ExprKind::Call(.., args) => {
                for expr in args {
                    self.visit_expr(expr);
                }
            },
            ExprKind::MethodCall(method_call) => {
                for expr in &method_call.args {
                    self.visit_expr(expr);
                }
            },
            ExprKind::Tup(exprs) | ExprKind::Array(exprs) => {
                for expr in exprs {
                    self.visit_expr(expr);
                }
            },
            ExprKind::Binary(.., left, right)
            | ExprKind::Assign(left, right, ..)
            | ExprKind::AssignOp(.., left, right)
            | ExprKind::Index(left, right) => {
                self.visit_expr(left);
                self.visit_expr(right);
            },
            ExprKind::Let(pat, expr, ..) => {
                self.visit_pat(pat);
                self.visit_expr(expr);
            },
            ExprKind::Unary(.., expr)
            | ExprKind::Await(expr)
            | ExprKind::Field(expr, ..)
            | ExprKind::AddrOf(.., expr)
            | ExprKind::Try(expr) => {
                self.visit_expr(expr);
            },
            ExprKind::Repeat(expr, anon_const) => {
                self.visit_expr(expr);
                self.visit_expr(&anon_const.value);
            },
            ExprKind::If(expr, block, else_expr) => {
                self.visit_expr(expr);
                self.visit_block(block);

                if let Some(expr) = else_expr {
                    self.visit_expr(expr);
                }
            },
            ExprKind::While(expr, block, ..) => {
                self.visit_expr(expr);
                self.visit_block(block);
            },
            ExprKind::ForLoop(pat, expr, block, ..) => {
                self.visit_pat(pat);
                self.visit_expr(expr);
                self.visit_block(block);
            },
            ExprKind::Loop(block, ..)
            | ExprKind::Block(block, ..)
            | ExprKind::Async(.., block)
            | ExprKind::TryBlock(block) => {
                self.visit_block(block);
            },
            ExprKind::Match(expr, arms) => {
                self.visit_expr(expr);

                for arm in arms {
                    self.visit_arm(arm);
                }
            },
            ExprKind::Closure(closure) => self.visit_expr(&closure.body),
            ExprKind::Range(start, end, ..) => {
                if let Some(expr) = start {
                    self.visit_expr(expr);
                }
                if let Some(expr) = end {
                    self.visit_expr(expr);
                }
            },
            ExprKind::Break(.., expr) | ExprKind::Ret(expr) | ExprKind::Yield(expr) | ExprKind::Yeet(expr) => {
                if let Some(expr) = expr {
                    self.visit_expr(expr);
                }
            },
            ExprKind::Struct(struct_expr) => {
                for field in &struct_expr.fields {
                    self.visit_expr(&field.expr);
                }
            },
            _ => (),
        }
    }

    fn visit_fn(&mut self, fk: FnKind<'_>, _: Span, _: NodeId) {
        match fk {
            FnKind::Fn(.., block) if let Some(block) = block => self.visit_block(block),
            FnKind::Closure(.., expr) => self.visit_expr(expr),
            // :/
            FnKind::Fn(..) => (),
        }
    }

    fn visit_item(&mut self, item: &Item) {
        match &item.kind {
            ItemKind::Static(static_item) if let Some(expr) = static_item.expr.as_ref() => self.visit_expr(expr),
            ItemKind::Const(const_item) if let Some(expr) = const_item.expr.as_ref() => self.visit_expr(expr),
            ItemKind::Fn(fk) if let Some(block) = fk.body.as_ref() => self.visit_block(block),
            ItemKind::Mod(.., mod_kind)
                if let ModKind::Loaded(items, Inline::Yes, ModSpans { inner_span, ..}) = mod_kind =>
            {
                self.nest_level += 1;

                check_indent(self, *inner_span);

                self.nest_level -= 1;
            }
            ItemKind::Trait(trit) => check_trait_and_impl(self, item, &trit.items),
            ItemKind::Impl(imp) => check_trait_and_impl(self, item, &imp.items),
            _ => (),
        }
    }
}

fn check_trait_and_impl(visitor: &mut NestingVisitor<'_, '_>, item: &Item, items: &ThinVec<P<Item<AssocItemKind>>>) {
    visitor.nest_level += 1;

    if !check_indent(visitor, item.span) {
        for item in items {
            match &item.kind {
                AssocItemKind::Const(const_item) if let Some(expr) = const_item.expr.as_ref() => {
                    visitor.visit_expr(expr);
                },
                AssocItemKind::Fn(fk) if let Some(block) = fk.body.as_ref() => visitor.visit_block(block),
                _ => (),
            }
        }
    }

    visitor.nest_level -= 1;
}

fn check_indent(visitor: &NestingVisitor<'_, '_>, span: Span) -> bool {
    if visitor.nest_level > visitor.conf.excessive_nesting_threshold && !in_external_macro(visitor.cx.sess(), span) {
        span_lint_and_help(
            visitor.cx,
            EXCESSIVE_NESTING,
            span,
            "this block is too nested",
            None,
            "try refactoring your code, extraction is often both easier to read and less nested",
        );

        return true;
    }

    false
}
