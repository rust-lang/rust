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
use rustc_ast as ast;
use rustc_ast::visit as ast_visit;
use rustc_attr::HasAttrs;
use rustc_session::lint::{BufferedEarlyLint, LintBuffer, LintPass};
use rustc_session::Session;
use rustc_span::symbol::Ident;
use rustc_span::Span;

use std::slice;
use tracing::debug;

macro_rules! run_early_pass { ($cx:expr, $f:ident, $($args:expr),*) => ({
    $cx.pass.$f(&$cx.context, $($args),*);
}) }

struct EarlyContextAndPass<'a, T: EarlyLintPass> {
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
        let push = self.context.builder.push(attrs, &self.context.lint_store, is_crate_node);
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
        ast_visit::walk_anon_const(self, c);
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        self.with_lint_attrs(e.id, &e.attrs, |cx| {
            run_early_pass!(cx, check_expr, e);
            ast_visit::walk_expr(cx, e);
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

    fn visit_struct_field(&mut self, s: &'a ast::StructField) {
        self.with_lint_attrs(s.id, &s.attrs, |cx| {
            run_early_pass!(cx, check_struct_field, s);
            ast_visit::walk_struct_field(cx, s);
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

    fn visit_mod(&mut self, m: &'a ast::Mod, s: Span, _a: &[ast::Attribute], n: ast::NodeId) {
        run_early_pass!(self, check_mod, m, s, n);
        self.check_id(n);
        ast_visit::walk_mod(self, m);
        run_early_pass!(self, check_mod_post, m, s, n);
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
        run_early_pass!(self, check_arm, a);
        ast_visit::walk_arm(self, a);
    }

    fn visit_expr_post(&mut self, e: &'a ast::Expr) {
        run_early_pass!(self, check_expr_post, e);
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

fn early_lint_crate<T: EarlyLintPass>(
    sess: &Session,
    lint_store: &LintStore,
    krate: &ast::Crate,
    pass: T,
    buffered: LintBuffer,
    warn_about_weird_lints: bool,
) -> LintBuffer {
    let mut cx = EarlyContextAndPass {
        context: EarlyContext::new(sess, lint_store, krate, buffered, warn_about_weird_lints),
        pass,
    };

    // Visit the whole crate.
    cx.with_lint_attrs(ast::CRATE_NODE_ID, &krate.attrs, |cx| {
        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_early_pass!(cx, check_crate, krate);

        ast_visit::walk_crate(cx, krate);

        run_early_pass!(cx, check_crate_post, krate);
    });
    cx.context.buffered
}

pub fn check_ast_crate<T: EarlyLintPass>(
    sess: &Session,
    lint_store: &LintStore,
    krate: &ast::Crate,
    pre_expansion: bool,
    lint_buffer: Option<LintBuffer>,
    builtin_lints: T,
) {
    let passes =
        if pre_expansion { &lint_store.pre_expansion_passes } else { &lint_store.early_passes };
    let mut passes: Vec<_> = passes.iter().map(|p| (p)()).collect();
    let mut buffered = lint_buffer.unwrap_or_default();

    if !sess.opts.debugging_opts.no_interleave_lints {
        buffered =
            early_lint_crate(sess, lint_store, krate, builtin_lints, buffered, pre_expansion);

        if !passes.is_empty() {
            buffered = early_lint_crate(
                sess,
                lint_store,
                krate,
                EarlyLintPassObjects { lints: &mut passes[..] },
                buffered,
                pre_expansion,
            );
        }
    } else {
        for pass in &mut passes {
            buffered =
                sess.prof.extra_verbose_generic_activity("run_lint", pass.name()).run(|| {
                    early_lint_crate(
                        sess,
                        lint_store,
                        krate,
                        EarlyLintPassObjects { lints: slice::from_mut(pass) },
                        buffered,
                        pre_expansion,
                    )
                });
        }
    }

    // All of the buffered lints should have been emitted at this point.
    // If not, that means that we somehow buffered a lint for a node id
    // that was not lint-checked (perhaps it doesn't exist?). This is a bug.
    //
    // Rustdoc runs everybody-loops before the early lints and removes
    // function bodies, so it's totally possible for linted
    // node ids to not exist (e.g., macros defined within functions for the
    // unused_macro lint) anymore. So we only run this check
    // when we're not in rustdoc mode. (see issue #47639)
    if !sess.opts.actually_rustdoc {
        for (_id, lints) in buffered.map {
            for early_lint in lints {
                sess.delay_span_bug(early_lint.span, "failed to process buffered lint here");
            }
        }
    }
}
