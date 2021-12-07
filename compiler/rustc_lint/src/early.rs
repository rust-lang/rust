//! Implementation of lint checking.
//!
//! The lint checking is mostly consolidated into one pass which runs
//! after all other analyses. Throughout compilation, lint warnings
//! can be added via the `add_lint` method on the Session structure. This
//! requires a span and an ID of the node that the lint is being added to. The
//! lint isn't actually emitted at that time because it is unknown what the
//! actual lint level at that location is.
//!
//! To actually emit lint warnings/errors, a separate pass is used.
//! A context keeps track of the current state of all lint levels.
//! Upon entering a node of the ast which can modify the lint settings, the
//! previous lint state is pushed onto a stack and the ast is then recursed
//! upon. As the ast is traversed, this keeps track of the current lint level
//! for all lint attributes.

use crate::context::{EarlyContext, LintContext, LintStore};
use crate::passes::{EarlyLintPass, EarlyLintPassObject};
use rustc_ast::ptr::P;
use rustc_ast::visit::{self as ast_visit, Visitor};
use rustc_ast::AstLike;
use rustc_ast::{self as ast, walk_list};
use rustc_middle::ty::RegisteredTools;
use rustc_session::lint::{BufferedEarlyLint, LintBuffer, LintPass};
use rustc_session::Session;
use rustc_span::symbol::Ident;
use rustc_span::Span;

use std::slice;
use tracing::debug;

macro_rules! run_early_pass { ($cx:expr, $f:ident, $($args:expr),*) => ({
    $cx.pass.$f(&$cx.context, $($args),*);
}) }

pub struct EarlyContextAndPass<'a, T: EarlyLintPass> {
    context: EarlyContext<'a>,
    pass: T,
}

impl<'a, T: EarlyLintPass> EarlyContextAndPass<'a, T> {
    fn check_id(&mut self, id: ast::NodeId) {
        for early_lint in self.context.buffered.take(id) {
            let BufferedEarlyLint { span, msg, node_id: _, lint_id, diagnostic } = early_lint;
            self.context.lookup_with_diagnostics(
                lint_id.lint,
                Some(span),
                |lint| lint.build(&msg).emit(),
                diagnostic,
            );
        }
    }

    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self, id: ast::NodeId, attrs: &'a [ast::Attribute], f: F)
    where
        F: FnOnce(&mut Self),
    {
        let is_crate_node = id == ast::CRATE_NODE_ID;
        let push = self.context.builder.push(attrs, is_crate_node);
        self.check_id(id);
        self.enter_attrs(attrs);
        f(self);
        self.exit_attrs(attrs);
        self.context.builder.pop(push);
    }

    fn enter_attrs(&mut self, attrs: &'a [ast::Attribute]) {
        debug!("early context: enter_attrs({:?})", attrs);
        run_early_pass!(self, enter_lint_attrs, attrs);
    }

    fn exit_attrs(&mut self, attrs: &'a [ast::Attribute]) {
        debug!("early context: exit_attrs({:?})", attrs);
        run_early_pass!(self, exit_lint_attrs, attrs);
    }
}

impl<'a, T: EarlyLintPass> ast_visit::Visitor<'a> for EarlyContextAndPass<'a, T> {
    fn visit_param(&mut self, param: &'a ast::Param) {
        self.with_lint_attrs(param.id, &param.attrs, |cx| {
            run_early_pass!(cx, check_param, param);
            ast_visit::walk_param(cx, param);
        });
    }

    fn visit_item(&mut self, it: &'a ast::Item) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            run_early_pass!(cx, check_item, it);
            ast_visit::walk_item(cx, it);
            run_early_pass!(cx, check_item_post, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &'a ast::ForeignItem) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            run_early_pass!(cx, check_foreign_item, it);
            ast_visit::walk_foreign_item(cx, it);
            run_early_pass!(cx, check_foreign_item_post, it);
        })
    }

    fn visit_pat(&mut self, p: &'a ast::Pat) {
        run_early_pass!(self, check_pat, p);
        self.check_id(p.id);
        ast_visit::walk_pat(self, p);
        run_early_pass!(self, check_pat_post, p);
    }

    fn visit_anon_const(&mut self, c: &'a ast::AnonConst) {
        run_early_pass!(self, check_anon_const, c);
        self.check_id(c.id);
        ast_visit::walk_anon_const(self, c);
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        self.with_lint_attrs(e.id, &e.attrs, |cx| {
            run_early_pass!(cx, check_expr, e);
            ast_visit::walk_expr(cx, e);
        })
    }

    fn visit_expr_field(&mut self, f: &'a ast::ExprField) {
        self.with_lint_attrs(f.id, &f.attrs, |cx| {
            ast_visit::walk_expr_field(cx, f);
        })
    }

    fn visit_stmt(&mut self, s: &'a ast::Stmt) {
        // Add the statement's lint attributes to our
        // current state when checking the statement itself.
        // This allows us to handle attributes like
        // `#[allow(unused_doc_comments)]`, which apply to
        // sibling attributes on the same target
        //
        // Note that statements get their attributes from
        // the AST struct that they wrap (e.g. an item)
        self.with_lint_attrs(s.id, s.attrs(), |cx| {
            run_early_pass!(cx, check_stmt, s);
            cx.check_id(s.id);
        });
        // The visitor for the AST struct wrapped
        // by the statement (e.g. `Item`) will call
        // `with_lint_attrs`, so do this walk
        // outside of the above `with_lint_attrs` call
        ast_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'a>, span: Span, id: ast::NodeId) {
        run_early_pass!(self, check_fn, fk, span, id);
        self.check_id(id);
        ast_visit::walk_fn(self, fk, span);

        // Explicitly check for lints associated with 'closure_id', since
        // it does not have a corresponding AST node
        if let ast_visit::FnKind::Fn(_, _, sig, _, _) = fk {
            if let ast::Async::Yes { closure_id, .. } = sig.header.asyncness {
                self.check_id(closure_id);
            }
        }
        run_early_pass!(self, check_fn_post, fk, span, id);
    }

    fn visit_variant_data(&mut self, s: &'a ast::VariantData) {
        run_early_pass!(self, check_struct_def, s);
        if let Some(ctor_hir_id) = s.ctor_id() {
            self.check_id(ctor_hir_id);
        }
        ast_visit::walk_struct_def(self, s);
        run_early_pass!(self, check_struct_def_post, s);
    }

    fn visit_field_def(&mut self, s: &'a ast::FieldDef) {
        self.with_lint_attrs(s.id, &s.attrs, |cx| {
            run_early_pass!(cx, check_field_def, s);
            ast_visit::walk_field_def(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &'a ast::Variant) {
        self.with_lint_attrs(v.id, &v.attrs, |cx| {
            run_early_pass!(cx, check_variant, v);
            ast_visit::walk_variant(cx, v);
            run_early_pass!(cx, check_variant_post, v);
        })
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        run_early_pass!(self, check_ty, t);
        self.check_id(t.id);
        ast_visit::walk_ty(self, t);
    }

    fn visit_ident(&mut self, ident: Ident) {
        run_early_pass!(self, check_ident, ident);
    }

    fn visit_local(&mut self, l: &'a ast::Local) {
        self.with_lint_attrs(l.id, &l.attrs, |cx| {
            run_early_pass!(cx, check_local, l);
            ast_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'a ast::Block) {
        run_early_pass!(self, check_block, b);
        self.check_id(b.id);
        ast_visit::walk_block(self, b);
        run_early_pass!(self, check_block_post, b);
    }

    fn visit_arm(&mut self, a: &'a ast::Arm) {
        self.with_lint_attrs(a.id, &a.attrs, |cx| {
            run_early_pass!(cx, check_arm, a);
            ast_visit::walk_arm(cx, a);
        })
    }

    fn visit_expr_post(&mut self, e: &'a ast::Expr) {
        run_early_pass!(self, check_expr_post, e);

        // Explicitly check for lints associated with 'closure_id', since
        // it does not have a corresponding AST node
        match e.kind {
            ast::ExprKind::Closure(_, ast::Async::Yes { closure_id, .. }, ..)
            | ast::ExprKind::Async(_, closure_id, ..) => self.check_id(closure_id),
            _ => {}
        }
    }

    fn visit_generic_arg(&mut self, arg: &'a ast::GenericArg) {
        run_early_pass!(self, check_generic_arg, arg);
        ast_visit::walk_generic_arg(self, arg);
    }

    fn visit_generic_param(&mut self, param: &'a ast::GenericParam) {
        run_early_pass!(self, check_generic_param, param);
        ast_visit::walk_generic_param(self, param);
    }

    fn visit_generics(&mut self, g: &'a ast::Generics) {
        run_early_pass!(self, check_generics, g);
        ast_visit::walk_generics(self, g);
    }

    fn visit_where_predicate(&mut self, p: &'a ast::WherePredicate) {
        run_early_pass!(self, check_where_predicate, p);
        ast_visit::walk_where_predicate(self, p);
    }

    fn visit_poly_trait_ref(&mut self, t: &'a ast::PolyTraitRef, m: &'a ast::TraitBoundModifier) {
        run_early_pass!(self, check_poly_trait_ref, t, m);
        ast_visit::walk_poly_trait_ref(self, t, m);
    }

    fn visit_assoc_item(&mut self, item: &'a ast::AssocItem, ctxt: ast_visit::AssocCtxt) {
        self.with_lint_attrs(item.id, &item.attrs, |cx| match ctxt {
            ast_visit::AssocCtxt::Trait => {
                run_early_pass!(cx, check_trait_item, item);
                ast_visit::walk_assoc_item(cx, item, ctxt);
                run_early_pass!(cx, check_trait_item_post, item);
            }
            ast_visit::AssocCtxt::Impl => {
                run_early_pass!(cx, check_impl_item, item);
                ast_visit::walk_assoc_item(cx, item, ctxt);
                run_early_pass!(cx, check_impl_item_post, item);
            }
        });
    }

    fn visit_lifetime(&mut self, lt: &'a ast::Lifetime) {
        run_early_pass!(self, check_lifetime, lt);
        self.check_id(lt.id);
    }

    fn visit_path(&mut self, p: &'a ast::Path, id: ast::NodeId) {
        run_early_pass!(self, check_path, p, id);
        self.check_id(id);
        ast_visit::walk_path(self, p);
    }

    fn visit_attribute(&mut self, attr: &'a ast::Attribute) {
        run_early_pass!(self, check_attribute, attr);
    }

    fn visit_mac_def(&mut self, mac: &'a ast::MacroDef, id: ast::NodeId) {
        run_early_pass!(self, check_mac_def, mac, id);
        self.check_id(id);
    }

    fn visit_mac_call(&mut self, mac: &'a ast::MacCall) {
        run_early_pass!(self, check_mac, mac);
        ast_visit::walk_mac(self, mac);
    }
}

struct EarlyLintPassObjects<'a> {
    lints: &'a mut [EarlyLintPassObject],
}

#[allow(rustc::lint_pass_impl_without_macro)]
impl LintPass for EarlyLintPassObjects<'_> {
    fn name(&self) -> &'static str {
        panic!()
    }
}

macro_rules! expand_early_lint_pass_impl_methods {
    ([$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        $(fn $name(&mut self, context: &EarlyContext<'_>, $($param: $arg),*) {
            for obj in self.lints.iter_mut() {
                obj.$name(context, $($param),*);
            }
        })*
    )
}

macro_rules! early_lint_pass_impl {
    ([], [$($methods:tt)*]) => (
        impl EarlyLintPass for EarlyLintPassObjects<'_> {
            expand_early_lint_pass_impl_methods!([$($methods)*]);
        }
    )
}

crate::early_lint_methods!(early_lint_pass_impl, []);

/// Early lints work on different nodes - either on the crate root, or on freshly loaded modules.
/// This trait generalizes over those nodes.
pub trait EarlyCheckNode<'a>: Copy {
    fn id(self) -> ast::NodeId;
    fn attrs<'b>(self) -> &'b [ast::Attribute]
    where
        'a: 'b;
    fn check<'b>(self, cx: &mut EarlyContextAndPass<'b, impl EarlyLintPass>)
    where
        'a: 'b;
}

impl<'a> EarlyCheckNode<'a> for &'a ast::Crate {
    fn id(self) -> ast::NodeId {
        ast::CRATE_NODE_ID
    }
    fn attrs<'b>(self) -> &'b [ast::Attribute]
    where
        'a: 'b,
    {
        &self.attrs
    }
    fn check<'b>(self, cx: &mut EarlyContextAndPass<'b, impl EarlyLintPass>)
    where
        'a: 'b,
    {
        run_early_pass!(cx, check_crate, self);
        ast_visit::walk_crate(cx, self);
        run_early_pass!(cx, check_crate_post, self);
    }
}

impl<'a> EarlyCheckNode<'a> for (ast::NodeId, &'a [ast::Attribute], &'a [P<ast::Item>]) {
    fn id(self) -> ast::NodeId {
        self.0
    }
    fn attrs<'b>(self) -> &'b [ast::Attribute]
    where
        'a: 'b,
    {
        self.1
    }
    fn check<'b>(self, cx: &mut EarlyContextAndPass<'b, impl EarlyLintPass>)
    where
        'a: 'b,
    {
        walk_list!(cx, visit_attribute, self.1);
        walk_list!(cx, visit_item, self.2);
    }
}

fn early_lint_node<'a>(
    sess: &Session,
    warn_about_weird_lints: bool,
    lint_store: &LintStore,
    registered_tools: &RegisteredTools,
    buffered: LintBuffer,
    pass: impl EarlyLintPass,
    check_node: impl EarlyCheckNode<'a>,
) -> LintBuffer {
    let mut cx = EarlyContextAndPass {
        context: EarlyContext::new(
            sess,
            warn_about_weird_lints,
            lint_store,
            registered_tools,
            buffered,
        ),
        pass,
    };

    cx.with_lint_attrs(check_node.id(), check_node.attrs(), |cx| check_node.check(cx));
    cx.context.buffered
}

pub fn check_ast_node<'a>(
    sess: &Session,
    pre_expansion: bool,
    lint_store: &LintStore,
    registered_tools: &RegisteredTools,
    lint_buffer: Option<LintBuffer>,
    builtin_lints: impl EarlyLintPass,
    check_node: impl EarlyCheckNode<'a>,
) {
    let passes =
        if pre_expansion { &lint_store.pre_expansion_passes } else { &lint_store.early_passes };
    let mut passes: Vec<_> = passes.iter().map(|p| (p)()).collect();
    let mut buffered = lint_buffer.unwrap_or_default();

    if !sess.opts.debugging_opts.no_interleave_lints {
        buffered = early_lint_node(
            sess,
            pre_expansion,
            lint_store,
            registered_tools,
            buffered,
            builtin_lints,
            check_node,
        );

        if !passes.is_empty() {
            buffered = early_lint_node(
                sess,
                false,
                lint_store,
                registered_tools,
                buffered,
                EarlyLintPassObjects { lints: &mut passes[..] },
                check_node,
            );
        }
    } else {
        for (i, pass) in passes.iter_mut().enumerate() {
            buffered =
                sess.prof.extra_verbose_generic_activity("run_lint", pass.name()).run(|| {
                    early_lint_node(
                        sess,
                        pre_expansion && i == 0,
                        lint_store,
                        registered_tools,
                        buffered,
                        EarlyLintPassObjects { lints: slice::from_mut(pass) },
                        check_node,
                    )
                });
        }
    }

    // All of the buffered lints should have been emitted at this point.
    // If not, that means that we somehow buffered a lint for a node id
    // that was not lint-checked (perhaps it doesn't exist?). This is a bug.
    for (id, lints) in buffered.map {
        for early_lint in lints {
            sess.delay_span_bug(
                early_lint.span,
                &format!(
                    "failed to process buffered lint here (dummy = {})",
                    id == ast::DUMMY_NODE_ID
                ),
            );
        }
    }
}
