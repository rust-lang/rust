//! Implementation of the early lint pass.
//!
//! The early lint pass works on AST nodes after macro expansion and name
//! resolution, just before AST lowering. These lints are for purely
//! syntactical lints.

use rustc_ast::ptr::P;
use rustc_ast::visit::{self as ast_visit, Visitor, walk_list};
use rustc_ast::{self as ast, HasAttrs};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_feature::Features;
use rustc_middle::ty::RegisteredTools;
use rustc_session::Session;
use rustc_session::lint::{BufferedEarlyLint, LintBuffer, LintPass};
use rustc_span::Span;
use rustc_span::symbol::Ident;
use tracing::debug;

use crate::context::{EarlyContext, LintStore};
use crate::passes::{EarlyLintPass, EarlyLintPassObject};

macro_rules! lint_callback { ($cx:expr, $f:ident, $($args:expr),*) => ({
    $cx.pass.$f(&$cx.context, $($args),*);
}) }

/// Implements the AST traversal for early lint passes. `T` provides the
/// `check_*` methods.
pub struct EarlyContextAndPass<'a, T: EarlyLintPass> {
    context: EarlyContext<'a>,
    pass: T,
}

impl<'a, T: EarlyLintPass> EarlyContextAndPass<'a, T> {
    // This always-inlined function is for the hot call site.
    #[inline(always)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    fn inlined_check_id(&mut self, id: ast::NodeId) {
        for early_lint in self.context.buffered.take(id) {
            let BufferedEarlyLint { span, node_id: _, lint_id, diagnostic } = early_lint;
            self.context.opt_span_lint_with_diagnostics(lint_id.lint, span, diagnostic);
        }
    }

    // This non-inlined function is for the cold call sites.
    fn check_id(&mut self, id: ast::NodeId) {
        self.inlined_check_id(id)
    }

    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self, id: ast::NodeId, attrs: &'a [ast::Attribute], f: F)
    where
        F: FnOnce(&mut Self),
    {
        let is_crate_node = id == ast::CRATE_NODE_ID;
        debug!(?id);
        let push = self.context.builder.push(attrs, is_crate_node, None);

        self.inlined_check_id(id);
        debug!("early context: enter_attrs({:?})", attrs);
        lint_callback!(self, check_attributes, attrs);
        ensure_sufficient_stack(|| f(self));
        debug!("early context: exit_attrs({:?})", attrs);
        lint_callback!(self, check_attributes_post, attrs);
        self.context.builder.pop(push);
    }
}

impl<'a, T: EarlyLintPass> ast_visit::Visitor<'a> for EarlyContextAndPass<'a, T> {
    fn visit_coroutine_kind(&mut self, coroutine_kind: &'a ast::CoroutineKind) -> Self::Result {
        self.check_id(coroutine_kind.closure_id());
    }

    fn visit_param(&mut self, param: &'a ast::Param) {
        self.with_lint_attrs(param.id, &param.attrs, |cx| {
            lint_callback!(cx, check_param, param);
            ast_visit::walk_param(cx, param);
        });
    }

    fn visit_item(&mut self, it: &'a ast::Item) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            lint_callback!(cx, check_item, it);
            ast_visit::walk_item(cx, it);
            lint_callback!(cx, check_item_post, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &'a ast::ForeignItem) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            ast_visit::walk_item(cx, it);
        })
    }

    fn visit_pat(&mut self, p: &'a ast::Pat) {
        lint_callback!(self, check_pat, p);
        self.check_id(p.id);
        ast_visit::walk_pat(self, p);
        lint_callback!(self, check_pat_post, p);
    }

    fn visit_pat_field(&mut self, field: &'a ast::PatField) {
        self.with_lint_attrs(field.id, &field.attrs, |cx| {
            ast_visit::walk_pat_field(cx, field);
        });
    }

    fn visit_anon_const(&mut self, c: &'a ast::AnonConst) {
        self.check_id(c.id);
        ast_visit::walk_anon_const(self, c);
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        self.with_lint_attrs(e.id, &e.attrs, |cx| {
            lint_callback!(cx, check_expr, e);
            ast_visit::walk_expr(cx, e);
            lint_callback!(cx, check_expr_post, e);
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
            lint_callback!(cx, check_stmt, s);
            cx.check_id(s.id);
        });
        // The visitor for the AST struct wrapped
        // by the statement (e.g. `Item`) will call
        // `with_lint_attrs`, so do this walk
        // outside of the above `with_lint_attrs` call
        ast_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'a>, span: Span, id: ast::NodeId) {
        lint_callback!(self, check_fn, fk, span, id);
        self.check_id(id);
        ast_visit::walk_fn(self, fk);
    }

    fn visit_variant_data(&mut self, s: &'a ast::VariantData) {
        if let Some(ctor_node_id) = s.ctor_node_id() {
            self.check_id(ctor_node_id);
        }
        ast_visit::walk_struct_def(self, s);
    }

    fn visit_field_def(&mut self, s: &'a ast::FieldDef) {
        self.with_lint_attrs(s.id, &s.attrs, |cx| {
            ast_visit::walk_field_def(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &'a ast::Variant) {
        self.with_lint_attrs(v.id, &v.attrs, |cx| {
            lint_callback!(cx, check_variant, v);
            ast_visit::walk_variant(cx, v);
        })
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        lint_callback!(self, check_ty, t);
        self.check_id(t.id);
        ast_visit::walk_ty(self, t);
    }

    fn visit_ident(&mut self, ident: &Ident) {
        lint_callback!(self, check_ident, ident);
    }

    fn visit_local(&mut self, l: &'a ast::Local) {
        self.with_lint_attrs(l.id, &l.attrs, |cx| {
            lint_callback!(cx, check_local, l);
            ast_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'a ast::Block) {
        lint_callback!(self, check_block, b);
        self.check_id(b.id);
        ast_visit::walk_block(self, b);
    }

    fn visit_arm(&mut self, a: &'a ast::Arm) {
        self.with_lint_attrs(a.id, &a.attrs, |cx| {
            lint_callback!(cx, check_arm, a);
            ast_visit::walk_arm(cx, a);
        })
    }

    fn visit_generic_arg(&mut self, arg: &'a ast::GenericArg) {
        lint_callback!(self, check_generic_arg, arg);
        ast_visit::walk_generic_arg(self, arg);
    }

    fn visit_generic_param(&mut self, param: &'a ast::GenericParam) {
        self.with_lint_attrs(param.id, &param.attrs, |cx| {
            lint_callback!(cx, check_generic_param, param);
            ast_visit::walk_generic_param(cx, param);
        });
    }

    fn visit_generics(&mut self, g: &'a ast::Generics) {
        lint_callback!(self, check_generics, g);
        ast_visit::walk_generics(self, g);
    }

    fn visit_where_predicate(&mut self, p: &'a ast::WherePredicate) {
        lint_callback!(self, enter_where_predicate, p);
        ast_visit::walk_where_predicate(self, p);
        lint_callback!(self, exit_where_predicate, p);
    }

    fn visit_poly_trait_ref(&mut self, t: &'a ast::PolyTraitRef) {
        lint_callback!(self, check_poly_trait_ref, t);
        ast_visit::walk_poly_trait_ref(self, t);
    }

    fn visit_assoc_item(&mut self, item: &'a ast::AssocItem, ctxt: ast_visit::AssocCtxt) {
        self.with_lint_attrs(item.id, &item.attrs, |cx| {
            match ctxt {
                ast_visit::AssocCtxt::Trait => {
                    lint_callback!(cx, check_trait_item, item);
                }
                ast_visit::AssocCtxt::Impl => {
                    lint_callback!(cx, check_impl_item, item);
                }
            }
            ast_visit::walk_assoc_item(cx, item, ctxt);
        });
    }

    fn visit_lifetime(&mut self, lt: &'a ast::Lifetime, _: ast_visit::LifetimeCtxt) {
        self.check_id(lt.id);
    }

    fn visit_path(&mut self, p: &'a ast::Path, id: ast::NodeId) {
        self.check_id(id);
        ast_visit::walk_path(self, p);
    }

    fn visit_path_segment(&mut self, s: &'a ast::PathSegment) {
        self.check_id(s.id);
        ast_visit::walk_path_segment(self, s);
    }

    fn visit_attribute(&mut self, attr: &'a ast::Attribute) {
        lint_callback!(self, check_attribute, attr);
    }

    fn visit_mac_def(&mut self, mac: &'a ast::MacroDef, id: ast::NodeId) {
        lint_callback!(self, check_mac_def, mac);
        self.check_id(id);
    }

    fn visit_mac_call(&mut self, mac: &'a ast::MacCall) {
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
    fn attrs<'b>(self) -> &'b [ast::Attribute]
    where
        'a: 'b;
    fn check<'b, T: EarlyLintPass>(self, cx: &mut EarlyContextAndPass<'b, T>)
    where
        'a: 'b;
}

impl<'a> EarlyCheckNode<'a> for (&'a ast::Crate, &'a [ast::Attribute]) {
    fn id(self) -> ast::NodeId {
        ast::CRATE_NODE_ID
    }
    fn attrs<'b>(self) -> &'b [ast::Attribute]
    where
        'a: 'b,
    {
        self.1
    }
    fn check<'b, T: EarlyLintPass>(self, cx: &mut EarlyContextAndPass<'b, T>)
    where
        'a: 'b,
    {
        lint_callback!(cx, check_crate, self.0);
        ast_visit::walk_crate(cx, self.0);
        lint_callback!(cx, check_crate_post, self.0);
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
    fn check<'b, T: EarlyLintPass>(self, cx: &mut EarlyContextAndPass<'b, T>)
    where
        'a: 'b,
    {
        walk_list!(cx, visit_attribute, self.1);
        walk_list!(cx, visit_item, self.2);
    }
}

pub fn check_ast_node<'a>(
    sess: &Session,
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
        check_ast_node_inner(sess, check_node, context, builtin_lints);
    } else {
        let mut passes: Vec<_> = passes.iter().map(|mk_pass| (mk_pass)()).collect();
        passes.push(Box::new(builtin_lints));
        let pass = RuntimeCombinedEarlyLintPass { passes: &mut passes[..] };
        check_ast_node_inner(sess, check_node, context, pass);
    }
}

fn check_ast_node_inner<'a, T: EarlyLintPass>(
    sess: &Session,
    check_node: impl EarlyCheckNode<'a>,
    context: EarlyContext<'_>,
    pass: T,
) {
    let mut cx = EarlyContextAndPass { context, pass };

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
