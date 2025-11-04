use rustc_hir::hir;
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_middle::ty;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_infer::infer::TyCtxtInferExt;

fn array_trait_check<'tcx>(
    cx: &rustc_lint::LateContext<'tcx>,
    t_ty: ty::Ty<'tcx>,
    trait_id: rustc_hir::def_id::DefId,
    other_trait_id: rustc_hir::def_id::DefId,
) -> bool {
    cx.tcx.infer_ctxt().build().enter(|infcx| {
        let tcx = infcx.tcx;

        let array_ty = tcx.mk_array(t_ty, tcx.const_usize(0));

        // The obligation: `[T; 4]: Trait`
        let trait_ref = ty::TraitRef::new(tcx, trait_id, [array_ty]);
        let trait_pred = ty::Binder::dummy(ty::TraitPredicate { trait_ref });

        // First check `[T; 4]: Trait`
        let base_ok = infcx.predicate_must_hold_modulo_regions(
            &ty::PredicateKind::Trait(trait_pred),
            cx.param_env,
        );

        // Now assume T: OtherTrait
        let other_ref = ty::TraitRef::new(tcx, other_trait_id, [t_ty]);
        let other_pred = ty::Binder::dummy(ty::PredicateKind::Trait(
            ty::TraitPredicate { trait_ref: other_ref },
        ));

        let extended_env = cx.param_env.with_additional_predicate(other_pred);

        let with_ok = infcx.predicate_must_hold_modulo_regions(
            &ty::PredicateKind::Trait(trait_pred),
            extended_env,
        );

        base_ok && !with_ok
    })
}

declare_lint! {
    /// The `impl_default_on_zero_size_nondefault_arrays` lint 
    /// detects usage of `[T; 0]: Default where T: !Default`
    /// ### Example
    ///
    /// ```
    /// struct DoesNotImplDefault;
    ///
    /// fn main() {
    ///     println!("{}", [DoesNotImplDefault; 0]::default());
    /// }
    /// ```
    ///
    /// ### Explanation
    /// 
    /// There is a plan to make `impl Default for [T; N] where T: Default` 
    /// as per https://github.com/rust-lang/rust/issues/61415.
    /// `[T; 0]: Default where T: !Default` interferes with that plan.
    /// This will become a hard error in the future.
    pub IMPL_DEFAULT_ON_ZERO_SIZE_NONDEFAULT_ARRAYS,
    Warn,
    "detects `[T: 0]: Default` bounds without a `T: Default` bound"
    @future_incompatible = FutureIncompatibleInfo {
      reason: FutureIncompatibilityReason::FutureReleaseError,
      reference: "https://github.com/rust-lang/rust/issues/61415"
    };
}

impl_lint_pass!(ImplDefaultZeroSizeNonDefaultArrays => IMPL_DEFAULT_ON_ZERO_SIZE_NONDEFAULT_ARRAYS);

impl<'tcx> LateLintPass<'tcx> for ImplDefaultZeroSizeNonDefaultArrays {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx rustc_hir::Item<'tcx>) {
      let def_id = cx.tcx.hir().local_def_id(item.hir_id());
      let preds = cx.tcx.predicates_of(def_id);

      if let Some(default_id) = cx.tcx.get_diagnostic_item(rustc_span::symbol::sym::Default) {
          for (clause, _) in preds.predicates {
                if let Some(binder) = clause.as_trait_clause() {
                    let trait_id = binder.skip_binder().trait_ref();
                    if trait_id == default_id {
                        if let ty::GenericArgKind::Type(ty) = binder.skip_binder().trait_ref().args().as_slice()[0].kind() {
                            if let ty::tyKind::Array(t_ty, len_const) = ty.kind() {
                                if let Some(0) = len_const.try_eval_usize(cx.tcx, cx.param_env) {
                                    if !cx.tcx.infer_ctxt().type_implements_trait(default_id, [t_ty], cx.param_env)  {
                                        cx.struct_span_lint(ARRAY_TRAIT_BOUND, item.span, |lint| {
                                        lint.build(&format!(
                                            "trait bound `[T; 0]: Default where T: !Default` likely to be removed soon, see https://github.com/rust-lang/rust/issues/61415",
                                        ))
                                        .emit();
                                        });
                                    }
                               }
                          }
                     }
                }
           }
      }
  }
  fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::MethodCall(_, receiver, _, _) = expr.kind {
            let method_def_id = cx.typeck_results().type_dependent_def_id(expr.hir_id);
            if let Some(default_id) = cx.tcx.get_diagnostic_item(rustc_span::symbol::sym::Default) {
                if let Some(def_id) = method_def_id {
                    if let TraitImpl(Ok(trait_id)) = cx.tcx.associated_item(def_id).container && trait_id == default_id {
                        let recv_ty = cx.typeck_results().expr_ty(receiver);
                        if let ty::Array(t_ty, len_const) = recv_ty.kind() {
                            if let Some(0) = len_const.to_value().try_to_target_usize(cx.tcx) {
                                if !cx.tcx.infer_ctxt().type_implements_trait(default_id, [t_ty], cx.param_env) {
                                    let trait_name = cx.tcx.def_path_str(trait_id);
                                    cx.struct_span_lint(ARRAY_TRAIT_IMPL_USAGE, expr.span, |lint| {
                                        lint.build(&format!(
                                            "use of trait `{}` implemented on array type `{}`",
                                            trait_name,
                                            recv_ty
                                        ))
                                        .emit();
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
}
