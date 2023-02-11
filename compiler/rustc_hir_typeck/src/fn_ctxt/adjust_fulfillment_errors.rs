use crate::FnCtxt;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefId;
use rustc_infer::traits::ObligationCauseCode;
use rustc_middle::ty::{self, DefIdTree, Ty, TypeSuperVisitable, TypeVisitable, TypeVisitor};
use rustc_span::{self, Span};
use rustc_trait_selection::traits;

use std::ops::ControlFlow;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn adjust_fulfillment_error_for_expr_obligation(
        &self,
        error: &mut traits::FulfillmentError<'tcx>,
    ) -> bool {
        let (traits::ExprItemObligation(def_id, hir_id, idx) | traits::ExprBindingObligation(def_id, _, hir_id, idx))
            = *error.obligation.cause.code().peel_derives() else { return false; };
        let hir = self.tcx.hir();
        let hir::Node::Expr(expr) = hir.get(hir_id) else { return false; };

        let Some(unsubstituted_pred) =
            self.tcx.predicates_of(def_id).instantiate_identity(self.tcx).predicates.into_iter().nth(idx)
            else { return false; };

        let generics = self.tcx.generics_of(def_id);
        let predicate_substs = match unsubstituted_pred.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(pred)) => pred.trait_ref.substs,
            ty::PredicateKind::Clause(ty::Clause::Projection(pred)) => pred.projection_ty.substs,
            _ => ty::List::empty(),
        };

        let find_param_matching = |matches: &dyn Fn(&ty::ParamTy) -> bool| {
            predicate_substs.types().find_map(|ty| {
                ty.walk().find_map(|arg| {
                    if let ty::GenericArgKind::Type(ty) = arg.unpack()
                        && let ty::Param(param_ty) = ty.kind()
                        && matches(param_ty)
                    {
                        Some(arg)
                    } else {
                        None
                    }
                })
            })
        };

        // Prefer generics that are local to the fn item, since these are likely
        // to be the cause of the unsatisfied predicate.
        let mut param_to_point_at = find_param_matching(&|param_ty| {
            self.tcx.parent(generics.type_param(param_ty, self.tcx).def_id) == def_id
        });
        // Fall back to generic that isn't local to the fn item. This will come
        // from a trait or impl, for example.
        let mut fallback_param_to_point_at = find_param_matching(&|param_ty| {
            self.tcx.parent(generics.type_param(param_ty, self.tcx).def_id) != def_id
                && param_ty.name != rustc_span::symbol::kw::SelfUpper
        });
        // Finally, the `Self` parameter is possibly the reason that the predicate
        // is unsatisfied. This is less likely to be true for methods, because
        // method probe means that we already kinda check that the predicates due
        // to the `Self` type are true.
        let mut self_param_to_point_at =
            find_param_matching(&|param_ty| param_ty.name == rustc_span::symbol::kw::SelfUpper);

        // Finally, for ambiguity-related errors, we actually want to look
        // for a parameter that is the source of the inference type left
        // over in this predicate.
        if let traits::FulfillmentErrorCode::CodeAmbiguity = error.code {
            fallback_param_to_point_at = None;
            self_param_to_point_at = None;
            param_to_point_at =
                self.find_ambiguous_parameter_in(def_id, error.root_obligation.predicate);
        }

        if self.closure_span_overlaps_error(error, expr.span) {
            return false;
        }

        match &expr.kind {
            hir::ExprKind::Path(qpath) => {
                if let hir::Node::Expr(hir::Expr {
                    kind: hir::ExprKind::Call(callee, args),
                    hir_id: call_hir_id,
                    span: call_span,
                    ..
                }) = hir.get_parent(expr.hir_id)
                    && callee.hir_id == expr.hir_id
                {
                    if self.closure_span_overlaps_error(error, *call_span) {
                        return false;
                    }

                    for param in
                        [param_to_point_at, fallback_param_to_point_at, self_param_to_point_at]
                        .into_iter()
                        .flatten()
                    {
                        if self.blame_specific_arg_if_possible(
                                error,
                                def_id,
                                param,
                                *call_hir_id,
                                callee.span,
                                None,
                                args,
                            )
                        {
                            return true;
                        }
                    }
                }
                // Notably, we only point to params that are local to the
                // item we're checking, since those are the ones we are able
                // to look in the final `hir::PathSegment` for. Everything else
                // would require a deeper search into the `qpath` than I think
                // is worthwhile.
                if let Some(param_to_point_at) = param_to_point_at
                    && self.point_at_path_if_possible(error, def_id, param_to_point_at, qpath)
                {
                    return true;
                }
            }
            hir::ExprKind::MethodCall(segment, receiver, args, ..) => {
                for param in [param_to_point_at, fallback_param_to_point_at, self_param_to_point_at]
                    .into_iter()
                    .flatten()
                {
                    if self.blame_specific_arg_if_possible(
                        error,
                        def_id,
                        param,
                        hir_id,
                        segment.ident.span,
                        Some(receiver),
                        args,
                    ) {
                        return true;
                    }
                }
                if let Some(param_to_point_at) = param_to_point_at
                    && self.point_at_generic_if_possible(error, def_id, param_to_point_at, segment)
                {
                    return true;
                }
            }
            hir::ExprKind::Struct(qpath, fields, ..) => {
                if let Res::Def(
                    hir::def::DefKind::Struct | hir::def::DefKind::Variant,
                    variant_def_id,
                ) = self.typeck_results.borrow().qpath_res(qpath, hir_id)
                {
                    for param in
                        [param_to_point_at, fallback_param_to_point_at, self_param_to_point_at]
                    {
                        if let Some(param) = param {
                            let refined_expr = self.point_at_field_if_possible(
                                def_id,
                                param,
                                variant_def_id,
                                fields,
                            );

                            match refined_expr {
                                None => {}
                                Some((refined_expr, _)) => {
                                    error.obligation.cause.span = refined_expr
                                        .span
                                        .find_ancestor_in_same_ctxt(error.obligation.cause.span)
                                        .unwrap_or(refined_expr.span);
                                    return true;
                                }
                            }
                        }
                    }
                }
                if let Some(param_to_point_at) = param_to_point_at
                    && self.point_at_path_if_possible(error, def_id, param_to_point_at, qpath)
                {
                    return true;
                }
            }
            _ => {}
        }

        false
    }

    fn point_at_path_if_possible(
        &self,
        error: &mut traits::FulfillmentError<'tcx>,
        def_id: DefId,
        param: ty::GenericArg<'tcx>,
        qpath: &hir::QPath<'tcx>,
    ) -> bool {
        match qpath {
            hir::QPath::Resolved(_, path) => {
                if let Some(segment) = path.segments.last()
                    && self.point_at_generic_if_possible(error, def_id, param, segment)
                {
                    return true;
                }
            }
            hir::QPath::TypeRelative(_, segment) => {
                if self.point_at_generic_if_possible(error, def_id, param, segment) {
                    return true;
                }
            }
            _ => {}
        }

        false
    }

    fn point_at_generic_if_possible(
        &self,
        error: &mut traits::FulfillmentError<'tcx>,
        def_id: DefId,
        param_to_point_at: ty::GenericArg<'tcx>,
        segment: &hir::PathSegment<'tcx>,
    ) -> bool {
        let own_substs = self
            .tcx
            .generics_of(def_id)
            .own_substs(ty::InternalSubsts::identity_for_item(self.tcx, def_id));
        let Some((index, _)) = own_substs
            .iter()
            .filter(|arg| matches!(arg.unpack(), ty::GenericArgKind::Type(_)))
            .enumerate()
            .find(|(_, arg)| **arg == param_to_point_at) else { return false };
        let Some(arg) = segment
            .args()
            .args
            .iter()
            .filter(|arg| matches!(arg, hir::GenericArg::Type(_)))
            .nth(index) else { return false; };
        error.obligation.cause.span = arg
            .span()
            .find_ancestor_in_same_ctxt(error.obligation.cause.span)
            .unwrap_or(arg.span());
        true
    }

    fn find_ambiguous_parameter_in<T: TypeVisitable<'tcx>>(
        &self,
        item_def_id: DefId,
        t: T,
    ) -> Option<ty::GenericArg<'tcx>> {
        struct FindAmbiguousParameter<'a, 'tcx>(&'a FnCtxt<'a, 'tcx>, DefId);
        impl<'tcx> TypeVisitor<'tcx> for FindAmbiguousParameter<'_, 'tcx> {
            type BreakTy = ty::GenericArg<'tcx>;
            fn visit_ty(&mut self, ty: Ty<'tcx>) -> std::ops::ControlFlow<Self::BreakTy> {
                if let Some(origin) = self.0.type_var_origin(ty)
                    && let rustc_infer::infer::type_variable::TypeVariableOriginKind::TypeParameterDefinition(_, Some(def_id)) =
                        origin.kind
                    && let generics = self.0.tcx.generics_of(self.1)
                    && let Some(index) = generics.param_def_id_to_index(self.0.tcx, def_id)
                    && let Some(subst) = ty::InternalSubsts::identity_for_item(self.0.tcx, self.1)
                        .get(index as usize)
                {
                    ControlFlow::Break(*subst)
                } else {
                    ty.super_visit_with(self)
                }
            }
        }
        t.visit_with(&mut FindAmbiguousParameter(self, item_def_id)).break_value()
    }

    fn closure_span_overlaps_error(
        &self,
        error: &traits::FulfillmentError<'tcx>,
        span: Span,
    ) -> bool {
        if let traits::FulfillmentErrorCode::CodeSelectionError(
            traits::SelectionError::OutputTypeParameterMismatch(_, expected, _),
        ) = error.code
            && let ty::Closure(def_id, _) | ty::Generator(def_id, ..) = expected.skip_binder().self_ty().kind()
            && span.overlaps(self.tcx.def_span(*def_id))
        {
            true
        } else {
            false
        }
    }

    fn point_at_field_if_possible(
        &self,
        def_id: DefId,
        param_to_point_at: ty::GenericArg<'tcx>,
        variant_def_id: DefId,
        expr_fields: &[hir::ExprField<'tcx>],
    ) -> Option<(&'tcx hir::Expr<'tcx>, Ty<'tcx>)> {
        let def = self.tcx.adt_def(def_id);

        let identity_substs = ty::InternalSubsts::identity_for_item(self.tcx, def_id);
        let fields_referencing_param: Vec<_> = def
            .variant_with_id(variant_def_id)
            .fields
            .iter()
            .filter(|field| {
                let field_ty = field.ty(self.tcx, identity_substs);
                Self::find_param_in_ty(field_ty.into(), param_to_point_at)
            })
            .collect();

        if let [field] = fields_referencing_param.as_slice() {
            for expr_field in expr_fields {
                // Look for the ExprField that matches the field, using the
                // same rules that check_expr_struct uses for macro hygiene.
                if self.tcx.adjust_ident(expr_field.ident, variant_def_id) == field.ident(self.tcx)
                {
                    return Some((expr_field.expr, self.tcx.type_of(field.did)));
                }
            }
        }

        None
    }

    /// - `blame_specific_*` means that the function will recursively traverse the expression,
    /// looking for the most-specific-possible span to blame.
    ///
    /// - `point_at_*` means that the function will only go "one level", pointing at the specific
    /// expression mentioned.
    ///
    /// `blame_specific_arg_if_possible` will find the most-specific expression anywhere inside
    /// the provided function call expression, and mark it as responsible for the fullfillment
    /// error.
    fn blame_specific_arg_if_possible(
        &self,
        error: &mut traits::FulfillmentError<'tcx>,
        def_id: DefId,
        param_to_point_at: ty::GenericArg<'tcx>,
        call_hir_id: hir::HirId,
        callee_span: Span,
        receiver: Option<&'tcx hir::Expr<'tcx>>,
        args: &'tcx [hir::Expr<'tcx>],
    ) -> bool {
        let ty = self.tcx.type_of(def_id);
        if !ty.is_fn() {
            return false;
        }
        let sig = ty.fn_sig(self.tcx).skip_binder();
        let args_referencing_param: Vec<_> = sig
            .inputs()
            .iter()
            .enumerate()
            .filter(|(_, ty)| Self::find_param_in_ty((**ty).into(), param_to_point_at))
            .collect();
        // If there's one field that references the given generic, great!
        if let [(idx, _)] = args_referencing_param.as_slice()
            && let Some(arg) = receiver
                .map_or(args.get(*idx), |rcvr| if *idx == 0 { Some(rcvr) } else { args.get(*idx - 1) }) {

            error.obligation.cause.span = arg.span.find_ancestor_in_same_ctxt(error.obligation.cause.span).unwrap_or(arg.span);

            if let hir::Node::Expr(arg_expr) = self.tcx.hir().get(arg.hir_id) {
                // This is more specific than pointing at the entire argument.
                self.blame_specific_expr_if_possible(error, arg_expr)
            }

            error.obligation.cause.map_code(|parent_code| {
                ObligationCauseCode::FunctionArgumentObligation {
                    arg_hir_id: arg.hir_id,
                    call_hir_id,
                    parent_code,
                }
            });
            return true;
        } else if args_referencing_param.len() > 0 {
            // If more than one argument applies, then point to the callee span at least...
            // We have chance to fix this up further in `point_at_generics_if_possible`
            error.obligation.cause.span = callee_span;
        }

        false
    }

    /**
     * Recursively searches for the most-specific blamable expression.
     * For example, if you have a chain of constraints like:
     * - want `Vec<i32>: Copy`
     * - because `Option<Vec<i32>>: Copy` needs `Vec<i32>: Copy` because `impl <T: Copy> Copy for Option<T>`
     * - because `(Option<Vec<i32>, bool)` needs `Option<Vec<i32>>: Copy` because `impl <A: Copy, B: Copy> Copy for (A, B)`
     * then if you pass in `(Some(vec![1, 2, 3]), false)`, this helper `point_at_specific_expr_if_possible`
     * will find the expression `vec![1, 2, 3]` as the "most blameable" reason for this missing constraint.
     *
     * This function only updates the error span.
     */
    pub fn blame_specific_expr_if_possible(
        &self,
        error: &mut traits::FulfillmentError<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) {
        // Whether it succeeded or failed, it likely made some amount of progress.
        // In the very worst case, it's just the same `expr` we originally passed in.
        let expr = match self.blame_specific_expr_if_possible_for_obligation_cause_code(
            &error.obligation.cause.code(),
            expr,
        ) {
            Ok(expr) => expr,
            Err(expr) => expr,
        };

        // Either way, use this expression to update the error span.
        // If it doesn't overlap the existing span at all, use the original span.
        // FIXME: It would possibly be better to do this more continuously, at each level...
        error.obligation.cause.span = expr
            .span
            .find_ancestor_in_same_ctxt(error.obligation.cause.span)
            .unwrap_or(error.obligation.cause.span);
    }

    fn blame_specific_expr_if_possible_for_obligation_cause_code(
        &self,
        obligation_cause_code: &traits::ObligationCauseCode<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Result<&'tcx hir::Expr<'tcx>, &'tcx hir::Expr<'tcx>> {
        match obligation_cause_code {
            traits::ObligationCauseCode::ExprBindingObligation(_, _, _, _) => {
                // This is the "root"; we assume that the `expr` is already pointing here.
                // Therefore, we return `Ok` so that this `expr` can be refined further.
                Ok(expr)
            }
            traits::ObligationCauseCode::ImplDerivedObligation(impl_derived) => self
                .blame_specific_expr_if_possible_for_derived_predicate_obligation(
                    impl_derived,
                    expr,
                ),
            _ => {
                // We don't recognize this kind of constraint, so we cannot refine the expression
                // any further.
                Err(expr)
            }
        }
    }

    /// We want to achieve the error span in the following example:
    ///
    /// ```ignore (just for demonstration)
    /// struct Burrito<Filling> {
    ///   filling: Filling,
    /// }
    /// impl <Filling: Delicious> Delicious for Burrito<Filling> {}
    /// fn eat_delicious_food<Food: Delicious>(_food: Food) {}
    ///
    /// fn will_type_error() {
    ///   eat_delicious_food(Burrito { filling: Kale });
    /// } //                                    ^--- The trait bound `Kale: Delicious`
    ///   //                                         is not satisfied
    /// ```
    ///
    /// Without calling this function, the error span will cover the entire argument expression.
    ///
    /// Before we do any of this logic, we recursively call `point_at_specific_expr_if_possible` on the parent
    /// obligation. Hence we refine the `expr` "outwards-in" and bail at the first kind of expression/impl we don't recognize.
    ///
    /// This function returns a `Result<&Expr, &Expr>` - either way, it returns the `Expr` whose span should be
    /// reported as an error. If it is `Ok`, then it means it refined successfull. If it is `Err`, then it may be
    /// only a partial success - but it cannot be refined even further.
    fn blame_specific_expr_if_possible_for_derived_predicate_obligation(
        &self,
        obligation: &traits::ImplDerivedObligationCause<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Result<&'tcx hir::Expr<'tcx>, &'tcx hir::Expr<'tcx>> {
        // First, we attempt to refine the `expr` for our span using the parent obligation.
        // If this cannot be done, then we are already stuck, so we stop early (hence the use
        // of the `?` try operator here).
        let expr = self.blame_specific_expr_if_possible_for_obligation_cause_code(
            &*obligation.derived.parent_code,
            expr,
        )?;

        // This is the "trait" (meaning, the predicate "proved" by this `impl`) which provides the `Self` type we care about.
        // For the purposes of this function, we hope that it is a `struct` type, and that our current `expr` is a literal of
        // that struct type.
        let impl_trait_self_ref: Option<ty::TraitRef<'tcx>> =
            self.tcx.impl_trait_ref(obligation.impl_def_id).map(|impl_def| impl_def.skip_binder());

        let Some(impl_trait_self_ref) = impl_trait_self_ref else {
            // It is possible that this is absent. In this case, we make no progress.
            return Err(expr);
        };

        // We only really care about the `Self` type itself, which we extract from the ref.
        let impl_self_ty: Ty<'tcx> = impl_trait_self_ref.self_ty();

        let impl_predicates: ty::GenericPredicates<'tcx> =
            self.tcx.predicates_of(obligation.impl_def_id);
        let Some(impl_predicate_index) = obligation.impl_def_predicate_index else {
            // We don't have the index, so we can only guess.
            return Err(expr);
        };

        if impl_predicate_index >= impl_predicates.predicates.len() {
            // This shouldn't happen, but since this is only a diagnostic improvement, avoid breaking things.
            return Err(expr);
        }
        let relevant_broken_predicate: ty::PredicateKind<'tcx> =
            impl_predicates.predicates[impl_predicate_index].0.kind().skip_binder();

        match relevant_broken_predicate {
            ty::PredicateKind::Clause(ty::Clause::Trait(broken_trait)) => {
                // ...
                self.blame_specific_part_of_expr_corresponding_to_generic_param(
                    broken_trait.trait_ref.self_ty().into(),
                    expr,
                    impl_self_ty.into(),
                )
            }
            _ => Err(expr),
        }
    }

    /// Drills into `expr` to arrive at the equivalent location of `find_generic_param` in `in_ty`.
    /// For example, given
    /// - expr: `(Some(vec![1, 2, 3]), false)`
    /// - param: `T`
    /// - in_ty: `(Option<Vec<T>, bool)`
    /// we would drill until we arrive at `vec![1, 2, 3]`.
    ///
    /// If successful, we return `Ok(refined_expr)`. If unsuccesful, we return `Err(partially_refined_expr`),
    /// which will go as far as possible. For example, given `(foo(), false)` instead, we would drill to
    /// `foo()` and then return `Err("foo()")`.
    ///
    /// This means that you can (and should) use the `?` try operator to chain multiple calls to this
    /// function with different types, since you can only continue drilling the second time if you
    /// succeeded the first time.
    fn blame_specific_part_of_expr_corresponding_to_generic_param(
        &self,
        param: ty::GenericArg<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        in_ty: ty::GenericArg<'tcx>,
    ) -> Result<&'tcx hir::Expr<'tcx>, &'tcx hir::Expr<'tcx>> {
        if param == in_ty {
            // The types match exactly, so we have drilled as far as we can.
            return Ok(expr);
        }

        let ty::GenericArgKind::Type(in_ty) = in_ty.unpack() else {
            return Err(expr);
        };

        if let (hir::ExprKind::Tup(expr_elements), ty::Tuple(in_ty_elements)) =
            (&expr.kind, in_ty.kind())
        {
            if in_ty_elements.len() != expr_elements.len() {
                return Err(expr);
            }
            // Find out which of `in_ty_elements` refer to `param`.
            // FIXME: It may be better to take the first if there are multiple,
            // just so that the error points to a smaller expression.
            let Some((drill_expr, drill_ty)) = Self::is_iterator_singleton(expr_elements.iter().zip( in_ty_elements.iter()).filter(|(_expr_elem, in_ty_elem)| {
                Self::find_param_in_ty((*in_ty_elem).into(), param)
            })) else {
                // The param is not mentioned, or it is mentioned in multiple indexes.
                return Err(expr);
            };

            return self.blame_specific_part_of_expr_corresponding_to_generic_param(
                param,
                drill_expr,
                drill_ty.into(),
            );
        }

        if let (
            hir::ExprKind::Struct(expr_struct_path, expr_struct_fields, _expr_struct_rest),
            ty::Adt(in_ty_adt, in_ty_adt_generic_args),
        ) = (&expr.kind, in_ty.kind())
        {
            // First, confirm that this struct is the same one as in the types, and if so,
            // find the right variant.
            let Res::Def(expr_struct_def_kind, expr_struct_def_id) = self.typeck_results.borrow().qpath_res(expr_struct_path, expr.hir_id) else {
                return Err(expr);
            };

            let variant_def_id = match expr_struct_def_kind {
                hir::def::DefKind::Struct => {
                    if in_ty_adt.did() != expr_struct_def_id {
                        // FIXME: Deal with type aliases?
                        return Err(expr);
                    }
                    expr_struct_def_id
                }
                hir::def::DefKind::Variant => {
                    // If this is a variant, its parent is the type definition.
                    if in_ty_adt.did() != self.tcx.parent(expr_struct_def_id) {
                        // FIXME: Deal with type aliases?
                        return Err(expr);
                    }
                    expr_struct_def_id
                }
                _ => {
                    return Err(expr);
                }
            };

            // We need to know which of the generic parameters mentions our target param.
            // We expect that at least one of them does, since it is expected to be mentioned.
            let Some((drill_generic_index, generic_argument_type)) =
                Self::is_iterator_singleton(
                    in_ty_adt_generic_args.iter().enumerate().filter(
                        |(_index, in_ty_generic)| {
                            Self::find_param_in_ty(*in_ty_generic, param)
                        },
                    ),
                ) else {
                    return Err(expr);
                };

            let struct_generic_parameters: &ty::Generics = self.tcx.generics_of(in_ty_adt.did());
            if drill_generic_index >= struct_generic_parameters.params.len() {
                return Err(expr);
            }

            let param_to_point_at_in_struct = self.tcx.mk_param_from_def(
                struct_generic_parameters.param_at(drill_generic_index, self.tcx),
            );

            // We make 3 steps:
            // Suppose we have a type like
            // ```ignore (just for demonstration)
            // struct ExampleStruct<T> {
            //   enabled: bool,
            //   item: Option<(usize, T, bool)>,
            // }
            //
            // f(ExampleStruct {
            //   enabled: false,
            //   item: Some((0, Box::new(String::new()), 1) }, true)),
            // });
            // ```
            // Here, `f` is passed a `ExampleStruct<Box<String>>`, but it wants
            // for `String: Copy`, which isn't true here.
            //
            // (1) First, we drill into `.item` and highlight that expression
            // (2) Then we use the template type `Option<(usize, T, bool)>` to
            //     drill into the `T`, arriving at a `Box<String>` expression.
            // (3) Then we keep going, drilling into this expression using our
            //     outer contextual information.

            // (1) Find the (unique) field which mentions the type in our constraint:
            let (field_expr, field_type) = self
                .point_at_field_if_possible(
                    in_ty_adt.did(),
                    param_to_point_at_in_struct,
                    variant_def_id,
                    expr_struct_fields,
                )
                .ok_or(expr)?;

            // (2) Continue drilling into the struct, ignoring the struct's
            // generic argument types.
            let expr = self.blame_specific_part_of_expr_corresponding_to_generic_param(
                param_to_point_at_in_struct,
                field_expr,
                field_type.into(),
            )?;

            // (3) Continue drilling into the expression, having "passed
            // through" the struct entirely.
            return self.blame_specific_part_of_expr_corresponding_to_generic_param(
                param,
                expr,
                generic_argument_type,
            );
        }

        if let (
            hir::ExprKind::Call(expr_callee, expr_args),
            ty::Adt(in_ty_adt, in_ty_adt_generic_args),
        ) = (&expr.kind, in_ty.kind())
        {
            let hir::ExprKind::Path(expr_callee_path) = &expr_callee.kind else {
                // FIXME: This case overlaps with another one worth handling,
                // which should happen above since it applies to non-ADTs:
                // we can drill down into regular generic functions.
                return Err(expr);
            };
            // This is (possibly) a constructor call, like `Some(...)` or `MyStruct(a, b, c)`.

            let Res::Def(expr_struct_def_kind, expr_ctor_def_id) = self.typeck_results.borrow().qpath_res(expr_callee_path, expr_callee.hir_id) else {
                return Err(expr);
            };

            let variant_def_id = match expr_struct_def_kind {
                hir::def::DefKind::Ctor(hir::def::CtorOf::Struct, hir::def::CtorKind::Fn) => {
                    if in_ty_adt.did() != self.tcx.parent(expr_ctor_def_id) {
                        // FIXME: Deal with type aliases?
                        return Err(expr);
                    }
                    self.tcx.parent(expr_ctor_def_id)
                }
                hir::def::DefKind::Ctor(hir::def::CtorOf::Variant, hir::def::CtorKind::Fn) => {
                    // If this is a variant, its parent is the type definition.
                    if in_ty_adt.did() != self.tcx.parent(expr_ctor_def_id) {
                        // FIXME: Deal with type aliases?
                        return Err(expr);
                    }
                    expr_ctor_def_id
                }
                _ => {
                    return Err(expr);
                }
            };

            // We need to know which of the generic parameters mentions our target param.
            // We expect that at least one of them does, since it is expected to be mentioned.
            let Some((drill_generic_index, generic_argument_type)) =
                Self::is_iterator_singleton(
                    in_ty_adt_generic_args.iter().enumerate().filter(
                        |(_index, in_ty_generic)| {
                            Self::find_param_in_ty(*in_ty_generic, param)
                        },
                    ),
                ) else {
                    return Err(expr);
                };

            let struct_generic_parameters: &ty::Generics = self.tcx.generics_of(in_ty_adt.did());
            if drill_generic_index >= struct_generic_parameters.params.len() {
                return Err(expr);
            }

            let param_to_point_at_in_struct = self.tcx.mk_param_from_def(
                struct_generic_parameters.param_at(drill_generic_index, self.tcx),
            );

            // We make 3 steps:
            // Suppose we have a type like
            // ```ignore (just for demonstration)
            // struct ExampleStruct<T> {
            //   enabled: bool,
            //   item: Option<(usize, T, bool)>,
            // }
            //
            // f(ExampleStruct {
            //   enabled: false,
            //   item: Some((0, Box::new(String::new()), 1) }, true)),
            // });
            // ```
            // Here, `f` is passed a `ExampleStruct<Box<String>>`, but it wants
            // for `String: Copy`, which isn't true here.
            //
            // (1) First, we drill into `.item` and highlight that expression
            // (2) Then we use the template type `Option<(usize, T, bool)>` to
            //     drill into the `T`, arriving at a `Box<String>` expression.
            // (3) Then we keep going, drilling into this expression using our
            //     outer contextual information.

            // (1) Find the (unique) field index which mentions the type in our constraint:
            let Some((field_index, field_type)) = Self::is_iterator_singleton(
                in_ty_adt
                    .variant_with_id(variant_def_id)
                    .fields
                    .iter()
                    .map(|field| field.ty(self.tcx, *in_ty_adt_generic_args))
                    .enumerate()
                    .filter(|(_index, field_type)| Self::find_param_in_ty((*field_type).into(), param))
            ) else {
                return Err(expr);
            };

            if field_index >= expr_args.len() {
                return Err(expr);
            }

            // (2) Continue drilling into the struct, ignoring the struct's
            // generic argument types.
            let expr = self.blame_specific_part_of_expr_corresponding_to_generic_param(
                param_to_point_at_in_struct,
                &expr_args[field_index],
                field_type.into(),
            )?;

            // (3) Continue drilling into the expression, having "passed
            // through" the struct entirely.
            return self.blame_specific_part_of_expr_corresponding_to_generic_param(
                param,
                expr,
                generic_argument_type,
            );
        }

        // At this point, none of the basic patterns matched.
        // One major possibility which remains is that we have a function call.
        // In this case, it's often possible to dive deeper into the call to find something to blame,
        // but this is not always possible.

        Err(expr)
    }

    // FIXME: This can be made into a private, non-impl function later.
    /// Traverses the given ty (either a `ty::Ty` or a `ty::GenericArg`) and searches for references
    /// to the given `param_to_point_at`. Returns `true` if it finds any use of the param.
    pub fn find_param_in_ty(
        ty: ty::GenericArg<'tcx>,
        param_to_point_at: ty::GenericArg<'tcx>,
    ) -> bool {
        let mut walk = ty.walk();
        while let Some(arg) = walk.next() {
            if arg == param_to_point_at {
                return true;
            }
            if let ty::GenericArgKind::Type(ty) = arg.unpack()
                && let ty::Alias(ty::Projection, ..) = ty.kind()
            {
                // This logic may seem a bit strange, but typically when
                // we have a projection type in a function signature, the
                // argument that's being passed into that signature is
                // not actually constraining that projection's substs in
                // a meaningful way. So we skip it, and see improvements
                // in some UI tests.
                walk.skip_current_subtree();
            }
        }
        false
    }

    // FIXME: This can be made into a private, non-impl function later.
    /// Returns `Some(iterator.next())` if it has exactly one item, and `None` otherwise.
    pub fn is_iterator_singleton<T>(mut iterator: impl Iterator<Item = T>) -> Option<T> {
        match (iterator.next(), iterator.next()) {
            (_, Some(_)) => None,
            (first, _) => first,
        }
    }
}
