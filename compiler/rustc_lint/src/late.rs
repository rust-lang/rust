//! Implementation of the late lint pass.
//!
//! The late lint pass Works on HIR nodes, towards the end of analysis (after
//! borrow checking, etc.). These lints have full type information available.

use std::any::Any;
use std::cell::Cell;

use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::join;
use rustc_hir::def_id::{LocalDefId, LocalModDefId};
use rustc_hir::{self as hir, AmbigArg, HirId, intravisit as hir_visit};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::Session;
use rustc_session::lint::LintPass;
use rustc_session::lint::builtin::HardwiredLints;
use rustc_span::Span;
use tracing::debug;

use crate::passes::LateLintPassObject;
use crate::{LateContext, LateLintPass, LintId, LintStore};

/// Extract the [`LintStore`] from [`Session`].
///
/// This function exists because [`Session::lint_store`] is type-erased.
pub fn unerased_lint_store(sess: &Session) -> &LintStore {
    let store: &dyn Any = sess.lint_store.as_deref().unwrap();
    store.downcast_ref().unwrap()
}

macro_rules! lint_callback { ($cx:expr, $f:ident, $($args:expr),*) => ({
    $cx.pass.$f(&$cx.context, $($args),*);
}) }

/// Implements the AST traversal for late lint passes. `T` provides the
/// `check_*` methods.
struct LateContextAndPass<'tcx, T: LateLintPass<'tcx>> {
    context: LateContext<'tcx>,
    pass: T,
}

impl<'tcx, T: LateLintPass<'tcx>> LateContextAndPass<'tcx, T> {
    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self, id: HirId, f: F)
    where
        F: FnOnce(&mut Self),
    {
        let attrs = self.context.tcx.hir_attrs(id);
        let prev = self.context.last_node_with_lint_attrs;
        self.context.last_node_with_lint_attrs = id;
        debug!("late context: enter_attrs({:?})", attrs);
        lint_callback!(self, check_attributes, attrs);
        for attr in attrs {
            lint_callback!(self, check_attribute, attr);
        }
        f(self);
        debug!("late context: exit_attrs({:?})", attrs);
        lint_callback!(self, check_attributes_post, attrs);
        self.context.last_node_with_lint_attrs = prev;
    }

    fn with_param_env<F>(&mut self, id: hir::OwnerId, f: F)
    where
        F: FnOnce(&mut Self),
    {
        let old_param_env = self.context.param_env;
        self.context.param_env = self.context.tcx.param_env(id);
        f(self);
        self.context.param_env = old_param_env;
    }

    fn process_mod(&mut self, m: &'tcx hir::Mod<'tcx>, n: HirId) {
        lint_callback!(self, check_mod, m, n);
        hir_visit::walk_mod(self, m);
    }
}

impl<'tcx, T: LateLintPass<'tcx>> hir_visit::Visitor<'tcx> for LateContextAndPass<'tcx, T> {
    type NestedFilter = nested_filter::All;

    /// Because lints are scoped lexically, we want to walk nested
    /// items in the context of the outer item, so enable
    /// deep-walking.
    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.context.tcx
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let old_enclosing_body = self.context.enclosing_body.replace(body_id);
        let old_cached_typeck_results = self.context.cached_typeck_results.get();

        // HACK(eddyb) avoid trashing `cached_typeck_results` when we're
        // nested in `visit_fn`, which may have already resulted in them
        // being queried.
        if old_enclosing_body != Some(body_id) {
            self.context.cached_typeck_results.set(None);
        }

        let body = self.context.tcx.hir_body(body_id);
        self.visit_body(body);
        self.context.enclosing_body = old_enclosing_body;

        // See HACK comment above.
        if old_enclosing_body != Some(body_id) {
            self.context.cached_typeck_results.set(old_cached_typeck_results);
        }
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        self.with_lint_attrs(param.hir_id, |cx| {
            hir_visit::walk_param(cx, param);
        });
    }

    fn visit_body(&mut self, body: &hir::Body<'tcx>) {
        lint_callback!(self, check_body, body);
        hir_visit::walk_body(self, body);
        lint_callback!(self, check_body_post, body);
    }

    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        let generics = self.context.generics.take();
        self.context.generics = it.kind.generics();
        let old_cached_typeck_results = self.context.cached_typeck_results.take();
        let old_enclosing_body = self.context.enclosing_body.take();
        self.with_lint_attrs(it.hir_id(), |cx| {
            cx.with_param_env(it.owner_id, |cx| {
                lint_callback!(cx, check_item, it);
                hir_visit::walk_item(cx, it);
                lint_callback!(cx, check_item_post, it);
            });
        });
        self.context.enclosing_body = old_enclosing_body;
        self.context.cached_typeck_results.set(old_cached_typeck_results);
        self.context.generics = generics;
    }

    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem<'tcx>) {
        self.with_lint_attrs(it.hir_id(), |cx| {
            cx.with_param_env(it.owner_id, |cx| {
                lint_callback!(cx, check_foreign_item, it);
                hir_visit::walk_foreign_item(cx, it);
            });
        })
    }

    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        lint_callback!(self, check_pat, p);
        hir_visit::walk_pat(self, p);
    }

    fn visit_lit(&mut self, hir_id: HirId, lit: hir::Lit, negated: bool) {
        lint_callback!(self, check_lit, hir_id, lit, negated);
    }

    fn visit_expr_field(&mut self, field: &'tcx hir::ExprField<'tcx>) {
        self.with_lint_attrs(field.hir_id, |cx| hir_visit::walk_expr_field(cx, field))
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        ensure_sufficient_stack(|| {
            self.with_lint_attrs(e.hir_id, |cx| {
                lint_callback!(cx, check_expr, e);
                hir_visit::walk_expr(cx, e);
                lint_callback!(cx, check_expr_post, e);
            })
        })
    }

    fn visit_stmt(&mut self, s: &'tcx hir::Stmt<'tcx>) {
        // See `EarlyContextAndPass::visit_stmt` for an explanation
        // of why we call `walk_stmt` outside of `with_lint_attrs`
        self.with_lint_attrs(s.hir_id, |cx| {
            lint_callback!(cx, check_stmt, s);
        });
        hir_visit::walk_stmt(self, s);
    }

    fn visit_fn(
        &mut self,
        fk: hir_visit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'tcx>,
        body_id: hir::BodyId,
        span: Span,
        id: LocalDefId,
    ) {
        // Wrap in typeck results here, not just in visit_nested_body,
        // in order for `check_fn` to be able to use them.
        let old_enclosing_body = self.context.enclosing_body.replace(body_id);
        let old_cached_typeck_results = self.context.cached_typeck_results.take();
        let body = self.context.tcx.hir_body(body_id);
        lint_callback!(self, check_fn, fk, decl, body, span, id);
        hir_visit::walk_fn(self, fk, decl, body_id, id);
        self.context.enclosing_body = old_enclosing_body;
        self.context.cached_typeck_results.set(old_cached_typeck_results);
    }

    fn visit_variant_data(&mut self, s: &'tcx hir::VariantData<'tcx>) {
        lint_callback!(self, check_struct_def, s);
        hir_visit::walk_struct_def(self, s);
    }

    fn visit_field_def(&mut self, s: &'tcx hir::FieldDef<'tcx>) {
        self.with_lint_attrs(s.hir_id, |cx| {
            lint_callback!(cx, check_field_def, s);
            hir_visit::walk_field_def(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &'tcx hir::Variant<'tcx>) {
        self.with_lint_attrs(v.hir_id, |cx| {
            lint_callback!(cx, check_variant, v);
            hir_visit::walk_variant(cx, v);
        })
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty<'tcx, AmbigArg>) {
        lint_callback!(self, check_ty, t);
        hir_visit::walk_ty(self, t);
    }

    fn visit_mod(&mut self, m: &'tcx hir::Mod<'tcx>, _: Span, n: HirId) {
        if !self.context.only_module {
            self.process_mod(m, n);
        }
    }

    fn visit_local(&mut self, l: &'tcx hir::LetStmt<'tcx>) {
        self.with_lint_attrs(l.hir_id, |cx| {
            lint_callback!(cx, check_local, l);
            hir_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'tcx hir::Block<'tcx>) {
        lint_callback!(self, check_block, b);
        hir_visit::walk_block(self, b);
        lint_callback!(self, check_block_post, b);
    }

    fn visit_arm(&mut self, a: &'tcx hir::Arm<'tcx>) {
        self.with_lint_attrs(a.hir_id, |cx| {
            lint_callback!(cx, check_arm, a);
            hir_visit::walk_arm(cx, a);
        })
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam<'tcx>) {
        lint_callback!(self, check_generic_param, p);
        hir_visit::walk_generic_param(self, p);
    }

    fn visit_generics(&mut self, g: &'tcx hir::Generics<'tcx>) {
        lint_callback!(self, check_generics, g);
        hir_visit::walk_generics(self, g);
    }

    fn visit_where_predicate(&mut self, p: &'tcx hir::WherePredicate<'tcx>) {
        hir_visit::walk_where_predicate(self, p);
    }

    fn visit_poly_trait_ref(&mut self, t: &'tcx hir::PolyTraitRef<'tcx>) {
        lint_callback!(self, check_poly_trait_ref, t);
        hir_visit::walk_poly_trait_ref(self, t);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        let generics = self.context.generics.take();
        self.context.generics = Some(trait_item.generics);
        self.with_lint_attrs(trait_item.hir_id(), |cx| {
            cx.with_param_env(trait_item.owner_id, |cx| {
                lint_callback!(cx, check_trait_item, trait_item);
                hir_visit::walk_trait_item(cx, trait_item);
            });
        });
        self.context.generics = generics;
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        let generics = self.context.generics.take();
        self.context.generics = Some(impl_item.generics);
        self.with_lint_attrs(impl_item.hir_id(), |cx| {
            cx.with_param_env(impl_item.owner_id, |cx| {
                lint_callback!(cx, check_impl_item, impl_item);
                hir_visit::walk_impl_item(cx, impl_item);
                lint_callback!(cx, check_impl_item_post, impl_item);
            });
        });
        self.context.generics = generics;
    }

    fn visit_lifetime(&mut self, lt: &'tcx hir::Lifetime) {
        hir_visit::walk_lifetime(self, lt);
    }

    fn visit_path(&mut self, p: &hir::Path<'tcx>, id: HirId) {
        lint_callback!(self, check_path, p, id);
        hir_visit::walk_path(self, p);
    }
}

// Combines multiple lint passes into a single pass, at runtime. Each
// `check_foo` method in `$methods` within this pass simply calls `check_foo`
// once per `$pass`. Compare with `declare_combined_late_lint_pass`, which is
// similar, but combines lint passes at compile time.
struct RuntimeCombinedLateLintPass<'a, 'tcx> {
    passes: &'a mut [LateLintPassObject<'tcx>],
}

#[allow(rustc::lint_pass_impl_without_macro)]
impl LintPass for RuntimeCombinedLateLintPass<'_, '_> {
    fn name(&self) -> &'static str {
        panic!()
    }
    fn get_lints(&self) -> crate::LintVec {
        panic!()
    }
}

macro_rules! impl_late_lint_pass {
    ([], [$($(#[$attr:meta])* fn $f:ident($($param:ident: $arg:ty),*);)*]) => {
        impl<'tcx> LateLintPass<'tcx> for RuntimeCombinedLateLintPass<'_, 'tcx> {
            $(fn $f(&mut self, context: &LateContext<'tcx>, $($param: $arg),*) {
                for pass in self.passes.iter_mut() {
                    pass.$f(context, $($param),*);
                }
            })*
        }
    };
}

crate::late_lint_methods!(impl_late_lint_pass, []);

pub fn late_lint_mod<'tcx, T: LateLintPass<'tcx> + 'tcx>(
    tcx: TyCtxt<'tcx>,
    module_def_id: LocalModDefId,
    builtin_lints: T,
) {
    let context = LateContext {
        tcx,
        enclosing_body: None,
        cached_typeck_results: Cell::new(None),
        param_env: ty::ParamEnv::empty(),
        effective_visibilities: tcx.effective_visibilities(()),
        last_node_with_lint_attrs: tcx.local_def_id_to_hir_id(module_def_id),
        generics: None,
        only_module: true,
    };

    // Note: `passes` is often empty. In that case, it's faster to run
    // `builtin_lints` directly rather than bundling it up into the
    // `RuntimeCombinedLateLintPass`.
    let store = unerased_lint_store(tcx.sess);

    if store.late_module_passes.is_empty() {
        // If all builtin lints can be skipped, there is no point in running `late_lint_mod_inner`
        // at all. This happens often for dependencies built with `--cap-lints=allow`.
        let dont_need_to_run = tcx.lints_that_dont_need_to_run(());
        let can_skip_lints = builtin_lints
            .get_lints()
            .iter()
            .all(|lint| dont_need_to_run.contains(&LintId::of(lint)));
        if !can_skip_lints {
            late_lint_mod_inner(tcx, module_def_id, context, builtin_lints);
        }
    } else {
        let builtin_lints = Box::new(builtin_lints) as Box<dyn LateLintPass<'tcx>>;
        let mut binding = store
            .late_module_passes
            .iter()
            .map(|mk_pass| (mk_pass)(tcx))
            .chain(std::iter::once(builtin_lints))
            .collect::<Vec<_>>();

        let pass = RuntimeCombinedLateLintPass { passes: binding.as_mut_slice() };
        late_lint_mod_inner(tcx, module_def_id, context, pass);
    }
}

fn late_lint_mod_inner<'tcx, T: LateLintPass<'tcx>>(
    tcx: TyCtxt<'tcx>,
    module_def_id: LocalModDefId,
    context: LateContext<'tcx>,
    pass: T,
) {
    let mut cx = LateContextAndPass { context, pass };

    let (module, _span, hir_id) = tcx.hir_get_module(module_def_id);

    cx.with_lint_attrs(hir_id, |cx| {
        // There is no module lint that will have the crate itself as an item, so check it here.
        if hir_id == hir::CRATE_HIR_ID {
            lint_callback!(cx, check_crate,);
        }

        cx.process_mod(module, hir_id);

        if hir_id == hir::CRATE_HIR_ID {
            lint_callback!(cx, check_crate_post,);
        }
    });
}

fn late_lint_crate<'tcx>(tcx: TyCtxt<'tcx>) {
    // Note: `passes` is often empty.
    let passes: Vec<_> =
        unerased_lint_store(tcx.sess).late_passes.iter().map(|mk_pass| (mk_pass)(tcx)).collect();

    if passes.is_empty() {
        return;
    }

    let context = LateContext {
        tcx,
        enclosing_body: None,
        cached_typeck_results: Cell::new(None),
        param_env: ty::ParamEnv::empty(),
        effective_visibilities: tcx.effective_visibilities(()),
        last_node_with_lint_attrs: hir::CRATE_HIR_ID,
        generics: None,
        only_module: false,
    };

    let lints_that_dont_need_to_run = tcx.lints_that_dont_need_to_run(());

    let mut filtered_passes: Vec<Box<dyn LateLintPass<'tcx>>> = passes
        .into_iter()
        .filter(|pass| {
            let lints = (**pass).get_lints();
            // Lintless passes are always in
            lints.is_empty() ||
            // If the pass doesn't have a single needed lint, omit it
            !lints.iter().all(|lint| lints_that_dont_need_to_run.contains(&LintId::of(lint)))
        })
        .collect();

    filtered_passes.push(Box::new(HardwiredLints));
    let pass = RuntimeCombinedLateLintPass { passes: &mut filtered_passes[..] };
    late_lint_crate_inner(tcx, context, pass);
}

fn late_lint_crate_inner<'tcx, T: LateLintPass<'tcx>>(
    tcx: TyCtxt<'tcx>,
    context: LateContext<'tcx>,
    pass: T,
) {
    let mut cx = LateContextAndPass { context, pass };

    // Visit the whole crate.
    cx.with_lint_attrs(hir::CRATE_HIR_ID, |cx| {
        // Since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        lint_callback!(cx, check_crate,);
        tcx.hir_walk_toplevel_module(cx);
        lint_callback!(cx, check_crate_post,);
    })
}

/// Performs lint checking on a crate.
pub fn check_crate<'tcx>(tcx: TyCtxt<'tcx>) {
    join(
        || {
            tcx.sess.time("crate_lints", || {
                // Run whole crate non-incremental lints
                late_lint_crate(tcx);
            });
        },
        || {
            tcx.sess.time("module_lints", || {
                // Run per-module lints
                tcx.par_hir_for_each_module(|module| tcx.ensure_ok().lint_mod(module));
            });
        },
    );
}
