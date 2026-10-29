//! Implementation of the early lint pass.
//!
//! The early lint pass works on AST nodes after macro expansion and name
//! resolution, just before AST lowering. These lints are for purely
//! syntactical lints.

use std::cell::Cell;

use rustc_ast::visit::{self as ast_visit, Visitor, walk_list};
use rustc_ast::{self as ast, AttrId, AttrVec, HasAttrs};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{
    BufferedEarlyLint, Diag, DiagCtxtHandle, DiagInner, Diagnostic, Level, LintBuffer, MultiSpan,
    PreExpansionLint,
};
use rustc_feature::Features;
use rustc_lint_defs::{Lint, LintPass};
use rustc_middle::ty::RegisteredTools;
use rustc_session::Session;
use rustc_span::{Ident, Span};
use tracing::debug;

use crate::DiagAndSess;
use crate::context::{EarlyContext, EarlyLintPassFactory, LintContext, LintStore};
use crate::levels::BuilderPush;
use crate::passes::{EarlyLintPass, EarlyLintPassObject};

pub(super) mod diagnostics;

pub trait EarlyLintMode: Sized + 'static {
    type Scope;

    fn enter(
        cx: &mut EarlyContext<'_, Self>,
        attrs: &[ast::Attribute],
        is_crate_node: bool,
    ) -> Self::Scope;

    fn exit(
        cx: &mut EarlyContext<'_, Self>,
        attrs: &[ast::Attribute],
        is_crate_node: bool,
        scope: Self::Scope,
    );

    fn opt_span_lint(
        cx: &EarlyContext<'_, Self>,
        lint: &'static Lint,
        span: Option<MultiSpan>,
        decorator: impl for<'a> Diagnostic<'a>,
    );
}

#[derive(Default)]
pub struct PreExpansion {
    attrs: Vec<AttrId>,
    lints: Cell<Vec<PreExpansionLint>>,
}

impl EarlyLintMode for PreExpansion {
    type Scope = usize;

    fn enter(cx: &mut EarlyContext<'_, Self>, attrs: &[ast::Attribute], _: bool) -> usize {
        let len = cx.mode.attrs.len();
        cx.mode.attrs.extend(attrs.iter().map(|attr| attr.id));
        len
    }

    fn exit(cx: &mut EarlyContext<'_, Self>, _: &[ast::Attribute], _: bool, len: usize) {
        cx.mode.attrs.truncate(len);
    }

    fn opt_span_lint(
        cx: &EarlyContext<'_, Self>,
        lint: &'static Lint,
        span: Option<MultiSpan>,
        decorator: impl for<'a> Diagnostic<'a>,
    ) {
        let diag = decorator.into_diag(cx.sess().dcx(), Level::Warning(None)).into_inner();
        let mut lints = cx.mode.lints.take();
        lints.push(PreExpansionLint { lint, span, diag, attrs: cx.mode.attrs.clone() });
        cx.mode.lints.set(lints);
    }
}

#[derive(Default)]
pub struct PostExpansion {
    lints: Vec<Option<PreExpansionLint>>,
    by_attr: FxHashMap<AttrId, Vec<usize>>,
}

impl PostExpansion {
    fn new(lints: Vec<PreExpansionLint>) -> Self {
        let mut by_attr: FxHashMap<AttrId, Vec<usize>> = Default::default();
        for (i, lint) in lints.iter().enumerate() {
            for &attr in &lint.attrs {
                by_attr.entry(attr).or_default().push(i);
            }
        }
        PostExpansion { lints: lints.into_iter().map(Some).collect(), by_attr }
    }
}

impl EarlyLintMode for PostExpansion {
    type Scope = BuilderPush;

    fn enter(
        cx: &mut EarlyContext<'_, Self>,
        attrs: &[ast::Attribute],
        is_crate_node: bool,
    ) -> BuilderPush {
        cx.builder.push(attrs, is_crate_node)
    }

    fn exit(
        cx: &mut EarlyContext<'_, Self>,
        attrs: &[ast::Attribute],
        is_crate_node: bool,
        push: BuilderPush,
    ) {
        let PostExpansion { lints, by_attr } = &mut cx.mode;
        if !by_attr.is_empty() || (is_crate_node && !lints.is_empty()) {
            let mut to_emit = Vec::new();
            if is_crate_node {
                to_emit.extend(lints.drain(..).flatten());
                by_attr.clear();
            } else {
                for attr in attrs {
                    if let Some(idxs) = by_attr.remove(&attr.id) {
                        to_emit.extend(idxs.into_iter().filter_map(|i| lints[i].take()));
                    }
                }
            }
            for PreExpansionLint { lint, span, diag, attrs: _ } in to_emit {
                cx.builder.opt_span_lint(lint, span, PreExpansionDiag(diag));
            }
        }
        cx.builder.pop(push);
    }

    fn opt_span_lint(
        cx: &EarlyContext<'_, Self>,
        lint: &'static Lint,
        span: Option<MultiSpan>,
        decorator: impl for<'a> Diagnostic<'a>,
    ) {
        cx.builder.opt_span_lint(lint, span, decorator)
    }
}

struct PreExpansionDiag(DiagInner);

impl<'a> Diagnostic<'a> for PreExpansionDiag {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a> {
        Diag::from_inner_with_level(dcx, self.0, level)
    }
}

macro_rules! lint_callback { ($cx:expr, $f:ident, $($args:expr),*) => ({
    $cx.pass.$f(&$cx.context, $($args),*);
}) }

/// Implements the AST traversal for early lint passes. `T` provides the
/// `check_*` methods.
struct EarlyContextAndPass<'ecx, T: EarlyLintPass<M>, M: EarlyLintMode> {
    context: EarlyContext<'ecx, M>,
    pass: T,
}

impl<'ecx, T: EarlyLintPass<M>, M: EarlyLintMode> EarlyContextAndPass<'ecx, T, M> {
    fn check_id(&mut self, id: ast::NodeId) {
        for early_lint in self.context.buffered.take(id) {
            let BufferedEarlyLint { span, node_id: _, lint_id, diagnostic } = early_lint;
            self.context.opt_span_lint(
                lint_id.lint,
                span,
                DiagAndSess { callback: diagnostic.0, sess: self.context.sess() },
            );
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
        let scope = M::enter(&mut self.context, attrs, is_crate_node);

        debug!("early context: enter_attrs({:?})", attrs);
        lint_callback!(self, check_attributes, attrs);
        f(self);
        debug!("early context: exit_attrs({:?})", attrs);
        lint_callback!(self, check_attributes_post, attrs);
        M::exit(&mut self.context, attrs, is_crate_node, scope);
    }
}

impl<'ast, 'ecx, T: EarlyLintPass<M>, M: EarlyLintMode> ast_visit::Visitor<'ast>
    for EarlyContextAndPass<'ecx, T, M>
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

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'ast>, _: &AttrVec, span: Span, id: ast::NodeId) {
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
        lint_callback!(self, check_where_predicate, p);
        ast_visit::walk_where_predicate(self, p);
        lint_callback!(self, check_where_predicate_post, p);
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
struct RuntimeCombinedEarlyLintPass<M: EarlyLintMode> {
    passes: Vec<EarlyLintPassObject<M>>,
}

#[allow(rustc::lint_pass_impl_without_macro)]
impl<M: EarlyLintMode> LintPass for RuntimeCombinedEarlyLintPass<M> {
    fn name(&self) -> &'static str {
        panic!()
    }
    fn get_lints(&self) -> crate::LintVec {
        panic!()
    }
}

macro_rules! impl_early_lint_pass {
    ([], [$($(#[$attr:meta])* fn $f:ident($($param:ident: $arg:ty),*);)*]) => (
        impl<M: EarlyLintMode> EarlyLintPass<M> for RuntimeCombinedEarlyLintPass<M> {
            $(fn $f(&mut self, context: &EarlyContext<'_, M>, $($param: $arg),*) {
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
pub enum EarlyCheckNode<'a> {
    CrateRoot(&'a ast::Crate, &'a [ast::Attribute]),
    LoadedMod(ast::NodeId, &'a [ast::Attribute], &'a [Box<ast::Item>]),
}

impl<'a> EarlyCheckNode<'a> {
    fn id(&self) -> ast::NodeId {
        match self {
            EarlyCheckNode::CrateRoot(_crate, _attrs) => ast::CRATE_NODE_ID,
            EarlyCheckNode::LoadedMod(id, _attrs, _items) => *id,
        }
    }
    fn attrs(&self) -> &'a [ast::Attribute] {
        match self {
            EarlyCheckNode::CrateRoot(_crate, attrs) => attrs,
            EarlyCheckNode::LoadedMod(_id, attrs, _items) => attrs,
        }
    }
    fn check<'ecx, T: EarlyLintPass<M>, M: EarlyLintMode>(
        &self,
        cx: &mut EarlyContextAndPass<'ecx, T, M>,
    ) {
        match self {
            EarlyCheckNode::CrateRoot(crate_, _attrs) => {
                lint_callback!(cx, check_crate, crate_);
                ast_visit::walk_crate(cx, crate_);
                lint_callback!(cx, check_crate_post, crate_);
            }
            EarlyCheckNode::LoadedMod(_id, attrs, items) => {
                walk_list!(cx, visit_attribute, *attrs);
                walk_list!(cx, visit_item, *items);
            }
        }
    }
}

pub fn check_ast_node_pre_expansion<'a>(
    sess: &Session,
    features: &Features,
    lint_store: &LintStore,
    registered_lint_tools: &RegisteredTools,
    check_node: EarlyCheckNode<'a>,
) -> Vec<PreExpansionLint> {
    let context = EarlyContext::new(
        sess,
        features,
        false,
        lint_store,
        registered_lint_tools,
        LintBuffer::default(),
        PreExpansion::default(),
    );
    let builtin_lints = crate::BuiltinCombinedPreExpansionLintPass::new();
    let passes = &lint_store.pre_expansion_lint_passes;
    let context = run_passes(check_node, context, builtin_lints, passes);
    check_buffered_lints_emitted(sess, context.buffered);
    context.mode.lints.into_inner()
}

pub fn check_ast_node<'a>(
    sess: &Session,
    features: &Features,
    lint_store: &LintStore,
    registered_lint_tools: &RegisteredTools,
    mut lint_buffer: LintBuffer,
    check_node: EarlyCheckNode<'a>,
) {
    let pre_expansion_lints = std::mem::take(&mut lint_buffer.pre_expansion);
    let context = EarlyContext::new(
        sess,
        features,
        true,
        lint_store,
        registered_lint_tools,
        lint_buffer,
        PostExpansion::new(pre_expansion_lints),
    );
    let builtin_lints = crate::BuiltinCombinedEarlyLintPass::new();
    let passes = &lint_store.early_lint_passes;
    let context = run_passes(check_node, context, builtin_lints, passes);
    check_buffered_lints_emitted(sess, context.buffered);
}

fn check_buffered_lints_emitted(sess: &Session, buffered: LintBuffer) {
    // All of the buffered lints should have been emitted at this point.
    // If not, that means that we somehow buffered a lint for a node id
    // that was not lint-checked (perhaps it doesn't exist?). This is a bug.
    for (id, lints) in buffered.early {
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

fn run_passes<'a, 'ecx, T: EarlyLintPass<M> + 'static, M: EarlyLintMode>(
    check_node: EarlyCheckNode<'a>,
    context: EarlyContext<'ecx, M>,
    builtin_lints: T,
    passes: &[EarlyLintPassFactory<M>],
) -> EarlyContext<'ecx, M> {
    // Note: `passes` is often empty. In that case, it's faster to run
    // `builtin_lints` directly rather than bundling it up into the
    // `RuntimeCombinedEarlyLintPass`.
    if passes.is_empty() {
        run_pass(check_node, context, builtin_lints)
    } else {
        let mut passes: Vec<_> = passes.iter().map(|mk_pass| mk_pass()).collect();
        passes.push(Box::new(builtin_lints));
        let pass = RuntimeCombinedEarlyLintPass { passes };
        run_pass(check_node, context, pass)
    }
}

fn run_pass<'a, 'ecx, T: EarlyLintPass<M>, M: EarlyLintMode>(
    check_node: EarlyCheckNode<'a>,
    context: EarlyContext<'ecx, M>,
    pass: T,
) -> EarlyContext<'ecx, M> {
    let mut cx = EarlyContextAndPass { context, pass };
    cx.with_lint_attrs(check_node.id(), check_node.attrs(), |cx| check_node.check(cx));
    cx.context
}
