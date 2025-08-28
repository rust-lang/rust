//! Implementation of the early lint pass.
//!
//! The early lint pass works on AST nodes after macro expansion and name
//! resolution, just before AST lowering. These lints are for purely
//! syntactical lints.

use rustc_ast::visit::{self as ast_visit, Visitor, walk_list};
use rustc_ast::{self as ast, HasAttrs};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{BufferedEarlyLint, DecorateDiagCompat, LintBuffer};
use rustc_feature::Features;
use rustc_middle::ty::{RegisteredTools, TyCtxt};
use rustc_session::Session;
use rustc_session::lint::LintPass;
use rustc_span::{Ident, Span};
use tracing::debug;

use crate::context::{EarlyContext, LintContext, LintStore};
use crate::passes::{EarlyLintPass, EarlyLintPassObject};

pub(super) mod diagnostics;

macro_rules! lint_callback { ($cx:expr, $f:ident, $($args:expr),*) => ({
    $cx.pass.$f(&$cx.context, $($args),*);
}) }

/// Implements the AST traversal for early lint passes. `T` provides the
/// `check_*` methods.
pub struct EarlyContextAndPass<'ecx, 'tcx, T: EarlyLintPass> {
    context: EarlyContext<'ecx>,
    tcx: Option<TyCtxt<'tcx>>,
    pass: T,
}

impl<'ecx, 'tcx, T: EarlyLintPass> EarlyContextAndPass<'ecx, 'tcx, T> {
    #[allow(rustc::diagnostic_outside_of_impl)]
    fn check_id(&mut self, id: ast::NodeId) {
        for early_lint in self.context.buffered.take(id) {
            let BufferedEarlyLint { span, node_id: _, lint_id, diagnostic } = early_lint;
            self.context.opt_span_lint(lint_id.lint, span, |diag| match diagnostic {
                DecorateDiagCompat::Builtin(b) => {
                    diagnostics::decorate_builtin_lint(self.context.sess(), self.tcx, b, diag);
                }
                DecorateDiagCompat::Dynamic(d) => d.decorate_lint_box(diag),
            });
        }
    }

    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self, id: ast::NodeId, attrs: &'_ [ast::Attribute], f: F)
    where
        F: FnOnce(&mut Self),
    {
        let is_crate_node = id == ast::CRATE_NODE_ID;
        debug!(?id);
        let push = self.context.builder.push(attrs, is_crate_node, None);

        debug!("early context: enter_attrs({:?})", attrs);
        lint_callback!(self, check_attributes, attrs);
        ensure_sufficient_stack(|| f(self));
        debug!("early context: exit_attrs({:?})", attrs);
        lint_callback!(self, check_attributes_post, attrs);
        self.context.builder.pop(push);
    }
}

impl<'ast, 'ecx, 'tcx, T: EarlyLintPass> ast_visit::Visitor<'ast>
    for EarlyContextAndPass<'ecx, 'tcx, T>
{
    fn visit_id(&mut self, id: rustc_ast::NodeId) {
        self.check_id(id);
    }

    fn visit_param(&mut self, param: &'ast ast::Param) {
        self.with_lint_attrs(param.id, &param.attrs, |cx| {
            lint_callback!(cx, check_param, param);
            ast_visit::walk_param(cx, param);
        });
    }

    fn visit_item(&mut self, it: &'ast ast::Item) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            lint_callback!(cx, check_item, it);
            ast_visit::walk_item(cx, it);
            lint_callback!(cx, check_item_post, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &'ast ast::ForeignItem) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            ast_visit::walk_item(cx, it);
        })
    }

    fn visit_pat(&mut self, p: &'ast ast::Pat) {
        lint_callback!(self, check_pat, p);
        ast_visit::walk_pat(self, p);
        lint_callback!(self, check_pat_post, p);
    }

    fn visit_pat_field(&mut self, field: &'ast ast::PatField) {
        self.with_lint_attrs(field.id, &field.attrs, |cx| {
            ast_visit::walk_pat_field(cx, field);
        });
    }

    fn visit_expr(&mut self, e: &'ast ast::Expr) {
        self.with_lint_attrs(e.id, &e.attrs, |cx| {
            lint_callback!(cx, check_expr, e);
            ast_visit::walk_expr(cx, e);
            lint_callback!(cx, check_expr_post, e);
        })
    }

    fn visit_expr_field(&mut self, f: &'ast ast::ExprField) {
        self.with_lint_attrs(f.id, &f.attrs, |cx| {
            ast_visit::walk_expr_field(cx, f);
        })
    }

    fn visit_stmt(&mut self, s: &'ast ast::Stmt) {
        // Add the statement's lint attributes to our
        // current state when checking the statement itself.
        // This allows us to handle attributes like
        // `#[allow(unused_doc_comments)]`, which apply to
        // sibling attributes on the same target
        //
        // Note that statements get their attributes from
        // the AST struct that they wrap (e.g. an item)
        self.with_lint_attrs(s.id, s.attrs(), |cx| {
            lint_callback!(cx, check_stmt, s);
            ast_visit::walk_stmt(cx, s);
        });
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'ast>, span: Span, id: ast::NodeId) {
        lint_callback!(self, check_fn, fk, span, id);
        ast_visit::walk_fn(self, fk);
    }

    fn visit_field_def(&mut self, s: &'ast ast::FieldDef) {
        self.with_lint_attrs(s.id, &s.attrs, |cx| {
            ast_visit::walk_field_def(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &'ast ast::Variant) {
        self.with_lint_attrs(v.id, &v.attrs, |cx| {
            lint_callback!(cx, check_variant, v);
            ast_visit::walk_variant(cx, v);
        })
    }

    fn visit_ty(&mut self, t: &'ast ast::Ty) {
        lint_callback!(self, check_ty, t);
        ast_visit::walk_ty(self, t);
    }

    fn visit_ident(&mut self, ident: &Ident) {
        lint_callback!(self, check_ident, ident);
    }

    fn visit_local(&mut self, l: &'ast ast::Local) {
        self.with_lint_attrs(l.id, &l.attrs, |cx| {
            lint_callback!(cx, check_local, l);
            ast_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'ast ast::Block) {
        lint_callback!(self, check_block, b);
        ast_visit::walk_block(self, b);
    }

    fn visit_arm(&mut self, a: &'ast ast::Arm) {
        self.with_lint_attrs(a.id, &a.attrs, |cx| {
            lint_callback!(cx, check_arm, a);
            ast_visit::walk_arm(cx, a);
        })
    }

    fn visit_generic_arg(&mut self, arg: &'ast ast::GenericArg) {
        lint_callback!(self, check_generic_arg, arg);
        ast_visit::walk_generic_arg(self, arg);
    }

    fn visit_generic_param(&mut self, param: &'ast ast::GenericParam) {
        self.with_lint_attrs(param.id, &param.attrs, |cx| {
            lint_callback!(cx, check_generic_param, param);
            ast_visit::walk_generic_param(cx, param);
        });
    }

    fn visit_generics(&mut self, g: &'ast ast::Generics) {
        lint_callback!(self, check_generics, g);
        ast_visit::walk_generics(self, g);
    }

    fn visit_where_predicate(&mut self, p: &'ast ast::WherePredicate) {
        lint_callback!(self, enter_where_predicate, p);
        ast_visit::walk_where_predicate(self, p);
        lint_callback!(self, exit_where_predicate, p);
    }

    fn visit_poly_trait_ref(&mut self, t: &'ast ast::PolyTraitRef) {
        lint_callback!(self, check_poly_trait_ref, t);
        ast_visit::walk_poly_trait_ref(self, t);
    }

    fn visit_assoc_item(&mut self, item: &'ast ast::AssocItem, ctxt: ast_visit::AssocCtxt) {
        self.with_lint_attrs(item.id, &item.attrs, |cx| {
            match ctxt {
                ast_visit::AssocCtxt::Trait => {
                    lint_callback!(cx, check_trait_item, item);
                }
                ast_visit::AssocCtxt::Impl { .. } => {
                    lint_callback!(cx, check_impl_item, item);
                }
            }
            ast_visit::walk_assoc_item(cx, item, ctxt);
            match ctxt {
                ast_visit::AssocCtxt::Trait => {
                    lint_callback!(cx, check_trait_item_post, item);
                }
                ast_visit::AssocCtxt::Impl { .. } => {
                    lint_callback!(cx, check_impl_item_post, item);
                }
            }
        });
    }

    fn visit_attribute(&mut self, attr: &'ast ast::Attribute) {
        lint_callback!(self, check_attribute, attr);
        ast_visit::walk_attribute(self, attr);
    }

    fn visit_macro_def(&mut self, mac: &'ast ast::MacroDef) {
        lint_callback!(self, check_mac_def, mac);
    }

    fn visit_mac_call(&mut self, mac: &'ast ast::MacCall) {
        lint_callback!(self, check_mac, mac);
        ast_visit::walk_mac(self, mac);
    }
}

// Combines multiple lint passes into a single pass, at runtime. Each
// `check_foo` method in `$methods` within this pass simply calls `check_foo`
// once per `$pass`. Compare with `declare_combined_early_lint_pass`, which is
// similar, but combines lint passes at compile time.
struct RuntimeCombinedEarlyLintPass<'a> {
    passes: &'a mut [EarlyLintPassObject],
}

#[allow(rustc::lint_pass_impl_without_macro)]
impl LintPass for RuntimeCombinedEarlyLintPass<'_> {
    fn name(&self) -> &'static str {
        panic!()
    }
    fn get_lints(&self) -> crate::LintVec {
        panic!()
    }
}

macro_rules! impl_early_lint_pass {
    ([], [$($(#[$attr:meta])* fn $f:ident($($param:ident: $arg:ty),*);)*]) => (
        impl EarlyLintPass for RuntimeCombinedEarlyLintPass<'_> {
            $(fn $f(&mut self, context: &EarlyContext<'_>, $($param: $arg),*) {
                for pass in self.passes.iter_mut() {
                    pass.$f(context, $($param),*);
                }
            })*
        }
    )
}

crate::early_lint_methods!(impl_early_lint_pass, []);

/// Early lints work on different nodes - either on the crate root, or on freshly loaded modules.
/// This trait generalizes over those nodes.
pub trait EarlyCheckNode<'a>: Copy {
    fn id(self) -> ast::NodeId;
    fn attrs(self) -> &'a [ast::Attribute];
    fn check<'ecx, 'tcx, T: EarlyLintPass>(self, cx: &mut EarlyContextAndPass<'ecx, 'tcx, T>);
}

impl<'a> EarlyCheckNode<'a> for (&'a ast::Crate, &'a [ast::Attribute]) {
    fn id(self) -> ast::NodeId {
        ast::CRATE_NODE_ID
    }
    fn attrs(self) -> &'a [ast::Attribute] {
        self.1
    }
    fn check<'ecx, 'tcx, T: EarlyLintPass>(self, cx: &mut EarlyContextAndPass<'ecx, 'tcx, T>) {
        lint_callback!(cx, check_crate, self.0);
        ast_visit::walk_crate(cx, self.0);
        lint_callback!(cx, check_crate_post, self.0);
    }
}

impl<'a> EarlyCheckNode<'a> for (ast::NodeId, &'a [ast::Attribute], &'a [Box<ast::Item>]) {
    fn id(self) -> ast::NodeId {
        self.0
    }
    fn attrs(self) -> &'a [ast::Attribute] {
        self.1
    }
    fn check<'ecx, 'tcx, T: EarlyLintPass>(self, cx: &mut EarlyContextAndPass<'ecx, 'tcx, T>) {
        walk_list!(cx, visit_attribute, self.1);
        walk_list!(cx, visit_item, self.2);
    }
}

pub fn check_ast_node<'a>(
    sess: &Session,
    tcx: Option<TyCtxt<'_>>,
    features: &Features,
    pre_expansion: bool,
    lint_store: &LintStore,
    registered_tools: &RegisteredTools,
    lint_buffer: Option<LintBuffer>,
    builtin_lints: impl EarlyLintPass + 'static,
    check_node: impl EarlyCheckNode<'a>,
) {
    let context = EarlyContext::new(
        sess,
        features,
        !pre_expansion,
        lint_store,
        registered_tools,
        lint_buffer.unwrap_or_default(),
    );

    // Note: `passes` is often empty. In that case, it's faster to run
    // `builtin_lints` directly rather than bundling it up into the
    // `RuntimeCombinedEarlyLintPass`.
    let passes =
        if pre_expansion { &lint_store.pre_expansion_passes } else { &lint_store.early_passes };
    if passes.is_empty() {
        check_ast_node_inner(sess, tcx, check_node, context, builtin_lints);
    } else {
        let mut passes: Vec<_> = passes.iter().map(|mk_pass| (mk_pass)()).collect();
        passes.push(Box::new(builtin_lints));
        let pass = RuntimeCombinedEarlyLintPass { passes: &mut passes[..] };
        check_ast_node_inner(sess, tcx, check_node, context, pass);
    }
}

fn check_ast_node_inner<'a, T: EarlyLintPass>(
    sess: &Session,
    tcx: Option<TyCtxt<'_>>,
    check_node: impl EarlyCheckNode<'a>,
    context: EarlyContext<'_>,
    pass: T,
) {
    let mut cx = EarlyContextAndPass { context, tcx, pass };

    cx.with_lint_attrs(check_node.id(), check_node.attrs(), |cx| check_node.check(cx));

    // All of the buffered lints should have been emitted at this point.
    // If not, that means that we somehow buffered a lint for a node id
    // that was not lint-checked (perhaps it doesn't exist?). This is a bug.
    for (id, lints) in cx.context.buffered.map {
        if !lints.is_empty() {
            assert!(
                sess.dcx().has_errors().is_some(),
                "failed to process buffered lint here (dummy = {})",
                id == ast::DUMMY_NODE_ID
            );
            break;
        }
    }
}
