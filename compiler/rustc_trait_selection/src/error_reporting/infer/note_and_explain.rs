use rustc_errors::Applicability::{MachineApplicable, MaybeIncorrect};
use rustc_errors::{Diag, MultiSpan, pluralize};
use rustc_hir as hir;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::DefKind;
use rustc_hir::find_attr;
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::fast_reject::DeepRejectCtxt;
use rustc_middle::ty::print::{FmtPrinter, Printer};
use rustc_middle::ty::{self, Ty, suggest_constraining_type_param};
use rustc_span::def_id::DefId;
use rustc_span::{BytePos, Span, Symbol};
use tracing::debug;

use crate::error_reporting::TypeErrCtxt;
use crate::infer::InferCtxtExt;

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    pub fn note_and_explain_type_err(
        &self,
        diag: &mut Diag<'_>,
        err: TypeError<'tcx>,
        cause: &ObligationCause<'tcx>,
        sp: Span,
        body_owner_def_id: DefId,
    ) {
        debug!("note_and_explain_type_err err={:?} cause={:?}", err, cause);

        let tcx = self.tcx;

        match err {
            TypeError::ArgumentSorts(values, _) | TypeError::Sorts(values) => {
                match (*values.expected.kind(), *values.found.kind()) {
                    (ty::Closure(..), ty::Closure(..)) => {
                        diag.note("no two closures, even if identical, have the same type");
                        diag.help("consider boxing your closure and/or using it as a trait object");
                    }
                    (ty::Coroutine(def_id1, ..), ty::Coroutine(def_id2, ..))
                        if self.tcx.coroutine_is_async(def_id1)
                            && self.tcx.coroutine_is_async(def_id2) =>
                    {
                        diag.note("no two async blocks, even if identical, have the same type");
                        diag.help(
                            "consider pinning your async block and casting it to a trait object",
                        );
                    }
                    (ty::Alias(ty::Opaque, ..), ty::Alias(ty::Opaque, ..)) => {
                        // Issue #63167
                        diag.note("distinct uses of `impl Trait` result in different opaque types");
                    }
                    (ty::Float(_), ty::Infer(ty::IntVar(_)))
                        if let Ok(
                            // Issue #53280
                            snippet,
                        ) = tcx.sess.source_map().span_to_snippet(sp) =>
                    {
                        if snippet.chars().all(|c| c.is_digit(10) || c == '-' || c == '_') {
                            diag.span_suggestion_verbose(
                                sp.shrink_to_hi(),
                                "use a float literal",
                                ".0",
                                MachineApplicable,
                            );
                        }
                    }
                    (ty::Param(expected), ty::Param(found)) => {
                        let generics = tcx.generics_of(body_owner_def_id);
                        let e_span = tcx.def_span(generics.type_param(expected, tcx).def_id);
                        if !sp.contains(e_span) {
                            diag.span_label(e_span, "expected type parameter");
                        }
                        let f_span = tcx.def_span(generics.type_param(found, tcx).def_id);
                        if !sp.contains(f_span) {
                            diag.span_label(f_span, "found type parameter");
                        }
                        diag.note(
                            "a type parameter was expected, but a different one was found; \
                             you might be missing a type parameter or trait bound",
                        );
                        diag.note(
                            "for more information, visit \
                             https://doc.rust-lang.org/book/ch10-02-traits.html\
                             #traits-as-parameters",
                        );
                    }
                    (
                        ty::Alias(ty::Projection | ty::Inherent, _),
                        ty::Alias(ty::Projection | ty::Inherent, _),
                    ) => {
                        diag.note("an associated type was expected, but a different one was found");
                    }
                    // FIXME(inherent_associated_types): Extend this to support `ty::Inherent`, too.
                    (ty::Param(p), ty::Alias(ty::Projection, proj))
                    | (ty::Alias(ty::Projection, proj), ty::Param(p))
                        if !tcx.is_impl_trait_in_trait(proj.def_id) =>
                    {
                        let param = tcx.generics_of(body_owner_def_id).type_param(p, tcx);
                        let p_def_id = param.def_id;
                        let p_span = tcx.def_span(p_def_id);
                        let expected = match (values.expected.kind(), values.found.kind()) {
                            (ty::Param(_), _) => "expected ",
                            (_, ty::Param(_)) => "found ",
                            _ => "",
                        };
                        if !sp.contains(p_span) {
                            diag.span_label(p_span, format!("{expected}this type parameter"));
                        }
                        let parent = p_def_id.as_local().and_then(|id| {
                            let local_id = tcx.local_def_id_to_hir_id(id);
                            let generics = tcx.parent_hir_node(local_id).generics()?;
                            Some((id, generics))
                        });
                        let mut note = true;
                        if let Some((local_id, generics)) = parent {
                            // Synthesize the associated type restriction `Add<Output = Expected>`.
                            // FIXME: extract this logic for use in other diagnostics.
                            let (trait_ref, assoc_args) = proj.trait_ref_and_own_args(tcx);
                            let item_name = tcx.item_name(proj.def_id);
                            let item_args = self.format_generic_args(assoc_args);

                            // Here, we try to see if there's an existing
                            // trait implementation that matches the one that
                            // we're suggesting to restrict. If so, find the
                            // "end", whether it be at the end of the trait
                            // or the end of the generic arguments.
                            let mut matching_span = None;
                            let mut matched_end_of_args = false;
                            for bound in generics.bounds_for_param(local_id) {
                                let potential_spans = bound.bounds.iter().find_map(|bound| {
                                    let bound_trait_path = bound.trait_ref()?.path;
                                    let def_id = bound_trait_path.res.opt_def_id()?;
                                    let generic_args = bound_trait_path
                                        .segments
                                        .iter()
                                        .last()
                                        .map(|path| path.args());
                                    (def_id == trait_ref.def_id)
                                        .then_some((bound_trait_path.span, generic_args))
                                });

                                if let Some((end_of_trait, end_of_args)) = potential_spans {
                                    let args_span = end_of_args.and_then(|args| args.span());
                                    matched_end_of_args = args_span.is_some();
                                    matching_span = args_span
                                        .or_else(|| Some(end_of_trait))
                                        .map(|span| span.shrink_to_hi());
                                    break;
                                }
                            }

                            if matched_end_of_args {
                                // Append suggestion to the end of our args
                                let path = format!(", {item_name}{item_args} = {p}");
                                note = !suggest_constraining_type_param(
                                    tcx,
                                    generics,
                                    diag,
                                    &proj.self_ty().to_string(),
                                    &path,
                                    None,
                                    matching_span,
                                );
                            } else {
                                // Suggest adding a bound to an existing trait
                                // or if the trait doesn't exist, add the trait
                                // and the suggested bounds.
                                let path = format!("<{item_name}{item_args} = {p}>");
                                note = !suggest_constraining_type_param(
                                    tcx,
                                    generics,
                                    diag,
                                    &proj.self_ty().to_string(),
                                    &path,
                                    None,
                                    matching_span,
                                );
                            }
                        }
                        if note {
                            diag.note("you might be missing a type parameter or trait bound");
                        }
                    }
                    (ty::Param(p), ty::Dynamic(..) | ty::Alias(ty::Opaque, ..))
                    | (ty::Dynamic(..) | ty::Alias(ty::Opaque, ..), ty::Param(p)) => {
                        let generics = tcx.generics_of(body_owner_def_id);
                        let p_span = tcx.def_span(generics.type_param(p, tcx).def_id);
                        let expected = match (values.expected.kind(), values.found.kind()) {
                            (ty::Param(_), _) => "expected ",
                            (_, ty::Param(_)) => "found ",
                            _ => "",
                        };
                        if !sp.contains(p_span) {
                            diag.span_label(p_span, format!("{expected}this type parameter"));
                        }
                        diag.help("type parameters must be constrained to match other types");
                        if diag.code.is_some_and(|code| tcx.sess.teach(code)) {
                            diag.help(
                                "given a type parameter `T` and a method `foo`:
```
trait Trait<T> { fn foo(&self) -> T; }
```
the only ways to implement method `foo` are:
- constrain `T` with an explicit type:
```
impl Trait<String> for X {
    fn foo(&self) -> String { String::new() }
}
```
- add a trait bound to `T` and call a method on that trait that returns `Self`:
```
impl<T: std::default::Default> Trait<T> for X {
    fn foo(&self) -> T { <T as std::default::Default>::default() }
}
```
- change `foo` to return an argument of type `T`:
```
impl<T> Trait<T> for X {
    fn foo(&self, x: T) -> T { x }
}
```",
                            );
                        }
                        diag.note(
                            "for more information, visit \
                             https://doc.rust-lang.org/book/ch10-02-traits.html\
                             #traits-as-parameters",
                        );
                    }
                    (
                        ty::Param(p),
                        ty::Closure(..) | ty::CoroutineClosure(..) | ty::Coroutine(..),
                    ) => {
                        let generics = tcx.generics_of(body_owner_def_id);
                        let p_span = tcx.def_span(generics.type_param(p, tcx).def_id);
                        if !sp.contains(p_span) {
                            diag.span_label(p_span, "expected this type parameter");
                        }
                        diag.help(format!(
                            "every closure has a distinct type and so could not always match the \
                             caller-chosen type of parameter `{p}`"
                        ));
                    }
                    (ty::Param(p), _) | (_, ty::Param(p)) => {
                        let generics = tcx.generics_of(body_owner_def_id);
                        let p_span = tcx.def_span(generics.type_param(p, tcx).def_id);
                        let expected = match (values.expected.kind(), values.found.kind()) {
                            (ty::Param(_), _) => "expected ",
                            (_, ty::Param(_)) => "found ",
                            _ => "",
                        };
                        if !sp.contains(p_span) {
                            diag.span_label(p_span, format!("{expected}this type parameter"));
                        }
                    }
                    (ty::Alias(ty::Projection | ty::Inherent, proj_ty), _)
                        if !tcx.is_impl_trait_in_trait(proj_ty.def_id) =>
                    {
                        self.expected_projection(
                            diag,
                            proj_ty,
                            values,
                            body_owner_def_id,
                            cause.code(),
                        );
                    }
                    (_, ty::Alias(ty::Projection | ty::Inherent, proj_ty))
                        if !tcx.is_impl_trait_in_trait(proj_ty.def_id) =>
                    {
                        let msg = || {
                            format!(
                                "consider constraining the associated type `{}` to `{}`",
                                values.found, values.expected,
                            )
                        };
                        if !(self.suggest_constraining_opaque_associated_type(
                            diag,
                            msg,
                            proj_ty,
                            values.expected,
                        ) || self.suggest_constraint(
                            diag,
                            &msg,
                            body_owner_def_id,
                            proj_ty,
                            values.expected,
                        )) {
                            diag.help(msg());
                            diag.note(
                                "for more information, visit \
                                https://doc.rust-lang.org/book/ch19-03-advanced-traits.html",
                            );
                        }
                    }
                    (ty::Dynamic(t, _), ty::Alias(ty::Opaque, alias))
                        if let Some(def_id) = t.principal_def_id()
                            && tcx
                                .explicit_item_self_bounds(alias.def_id)
                                .skip_binder()
                                .iter()
                                .any(|(pred, _span)| match pred.kind().skip_binder() {
                                    ty::ClauseKind::Trait(trait_predicate)
                                        if trait_predicate.polarity
                                            == ty::PredicatePolarity::Positive =>
                                    {
                                        trait_predicate.def_id() == def_id
                                    }
                                    _ => false,
                                }) =>
                    {
                        diag.help(format!(
                            "you can box the `{}` to coerce it to `Box<{}>`, but you'll have to \
                             change the expected type as well",
                            values.found, values.expected,
                        ));
                    }
                    (ty::Dynamic(t, _), _) if let Some(def_id) = t.principal_def_id() => {
                        let mut has_matching_impl = false;
                        tcx.for_each_relevant_impl(def_id, values.found, |did| {
                            if DeepRejectCtxt::relate_rigid_infer(tcx)
                                .types_may_unify(values.found, tcx.type_of(did).skip_binder())
                            {
                                has_matching_impl = true;
                            }
                        });
                        if has_matching_impl {
                            let trait_name = tcx.item_name(def_id);
                            diag.help(format!(
                                "`{}` implements `{trait_name}` so you could box the found value \
                                 and coerce it to the trait object `Box<dyn {trait_name}>`, you \
                                 will have to change the expected type as well",
                                values.found,
                            ));
                        }
                    }
                    (_, ty::Dynamic(t, _)) if let Some(def_id) = t.principal_def_id() => {
                        let mut has_matching_impl = false;
                        tcx.for_each_relevant_impl(def_id, values.expected, |did| {
                            if DeepRejectCtxt::relate_rigid_infer(tcx)
                                .types_may_unify(values.expected, tcx.type_of(did).skip_binder())
                            {
                                has_matching_impl = true;
                            }
                        });
                        if has_matching_impl {
                            let trait_name = tcx.item_name(def_id);
                            diag.help(format!(
                                "`{}` implements `{trait_name}` so you could change the expected \
                                 type to `Box<dyn {trait_name}>`",
                                values.expected,
                            ));
                        }
                    }
                    (_, ty::Alias(ty::Opaque, opaque_ty))
                    | (ty::Alias(ty::Opaque, opaque_ty), _) => {
                        if opaque_ty.def_id.is_local()
                            && matches!(
                                tcx.def_kind(body_owner_def_id),
                                DefKind::Fn
                                    | DefKind::Static { .. }
                                    | DefKind::Const
                                    | DefKind::AssocFn
                                    | DefKind::AssocConst
                            )
                            && matches!(
                                tcx.opaque_ty_origin(opaque_ty.def_id),
                                hir::OpaqueTyOrigin::TyAlias { .. }
                            )
                            && !tcx
                                .opaque_types_defined_by(body_owner_def_id.expect_local())
                                .contains(&opaque_ty.def_id.expect_local())
                        {
                            let sp = tcx
                                .def_ident_span(body_owner_def_id)
                                .unwrap_or_else(|| tcx.def_span(body_owner_def_id));
                            let mut alias_def_id = opaque_ty.def_id;
                            while let DefKind::OpaqueTy = tcx.def_kind(alias_def_id) {
                                alias_def_id = tcx.parent(alias_def_id);
                            }
                            let opaque_path = tcx.def_path_str(alias_def_id);
                            // FIXME(type_alias_impl_trait): make this a structured suggestion
                            match tcx.opaque_ty_origin(opaque_ty.def_id) {
                                rustc_hir::OpaqueTyOrigin::FnReturn { .. } => {}
                                rustc_hir::OpaqueTyOrigin::AsyncFn { .. } => {}
                                rustc_hir::OpaqueTyOrigin::TyAlias {
                                    in_assoc_ty: false, ..
                                } => {
                                    diag.span_note(
                                        sp,
                                        format!("this item must have a `#[define_opaque({opaque_path})]` \
                                        attribute to be able to define hidden types"),
                                    );
                                }
                                rustc_hir::OpaqueTyOrigin::TyAlias {
                                    in_assoc_ty: true, ..
                                } => {}
                            }
                        }
                        // If two if arms can be coerced to a trait object, provide a structured
                        // suggestion.
                        let ObligationCauseCode::IfExpression { expr_id, .. } = cause.code() else {
                            return;
                        };
                        let hir::Node::Expr(&hir::Expr {
                            kind:
                                hir::ExprKind::If(
                                    _,
                                    &hir::Expr {
                                        kind:
                                            hir::ExprKind::Block(
                                                &hir::Block { expr: Some(then), .. },
                                                _,
                                            ),
                                        ..
                                    },
                                    Some(&hir::Expr {
                                        kind:
                                            hir::ExprKind::Block(
                                                &hir::Block { expr: Some(else_), .. },
                                                _,
                                            ),
                                        ..
                                    }),
                                ),
                            ..
                        }) = self.tcx.hir_node(*expr_id)
                        else {
                            return;
                        };
                        let expected = match values.found.kind() {
                            ty::Alias(..) => values.expected,
                            _ => values.found,
                        };
                        let preds = tcx.explicit_item_self_bounds(opaque_ty.def_id);
                        for (pred, _span) in preds.skip_binder() {
                            let ty::ClauseKind::Trait(trait_predicate) = pred.kind().skip_binder()
                            else {
                                continue;
                            };
                            if trait_predicate.polarity != ty::PredicatePolarity::Positive {
                                continue;
                            }
                            let def_id = trait_predicate.def_id();
                            let mut impl_def_ids = vec![];
                            tcx.for_each_relevant_impl(def_id, expected, |did| {
                                impl_def_ids.push(did)
                            });
                            if let [_] = &impl_def_ids[..] {
                                let trait_name = tcx.item_name(def_id);
                                diag.multipart_suggestion(
                                    format!(
                                        "`{expected}` implements `{trait_name}` so you can box \
                                         both arms and coerce to the trait object \
                                         `Box<dyn {trait_name}>`",
                                    ),
                                    vec![
                                        (then.span.shrink_to_lo(), "Box::new(".to_string()),
                                        (
                                            then.span.shrink_to_hi(),
                                            format!(") as Box<dyn {}>", tcx.def_path_str(def_id)),
                                        ),
                                        (else_.span.shrink_to_lo(), "Box::new(".to_string()),
                                        (else_.span.shrink_to_hi(), ")".to_string()),
                                    ],
                                    MachineApplicable,
                                );
                            }
                        }
                    }
                    (ty::FnPtr(_, hdr), ty::FnDef(def_id, _))
                    | (ty::FnDef(def_id, _), ty::FnPtr(_, hdr)) => {
                        if tcx.fn_sig(def_id).skip_binder().safety() < hdr.safety {
                            if !tcx.codegen_fn_attrs(def_id).safe_target_features {
                                diag.note(
                                "unsafe functions cannot be coerced into safe function pointers",
                                );
                            }
                        }
                    }
                    (ty::Adt(_, _), ty::Adt(def, args))
                        if let ObligationCauseCode::IfExpression { expr_id, .. } = cause.code()
                            && let hir::Node::Expr(if_expr) = self.tcx.hir_node(*expr_id)
                            && let hir::ExprKind::If(_, then_expr, _) = if_expr.kind
                            && let hir::ExprKind::Block(blk, _) = then_expr.kind
                            && let Some(then) = blk.expr
                            && def.is_box()
                            && let boxed_ty = args.type_at(0)
                            && let ty::Dynamic(t, _) = boxed_ty.kind()
                            && let Some(def_id) = t.principal_def_id()
                            && let mut impl_def_ids = vec![]
                            && let _ =
                                tcx.for_each_relevant_impl(def_id, values.expected, |did| {
                                    impl_def_ids.push(did)
                                })
                            && let [_] = &impl_def_ids[..] =>
                    {
                        // We have divergent if/else arms where the expected value is a type that
                        // implements the trait of the found boxed trait object.
                        diag.multipart_suggestion(
                            format!(
                                "`{}` implements `{}` so you can box it to coerce to the trait \
                                 object `{}`",
                                values.expected,
                                tcx.item_name(def_id),
                                values.found,
                            ),
                            vec![
                                (then.span.shrink_to_lo(), "Box::new(".to_string()),
                                (then.span.shrink_to_hi(), ")".to_string()),
                            ],
                            MachineApplicable,
                        );
                    }
                    _ => {}
                }
                debug!(
                    "note_and_explain_type_err expected={:?} ({:?}) found={:?} ({:?})",
                    values.expected,
                    values.expected.kind(),
                    values.found,
                    values.found.kind(),
                );
            }
            TypeError::CyclicTy(ty) => {
                // Watch out for various cases of cyclic types and try to explain.
                if ty.is_closure() || ty.is_coroutine() || ty.is_coroutine_closure() {
                    diag.note(
                        "closures cannot capture themselves or take themselves as argument;\n\
                         this error may be the result of a recent compiler bug-fix,\n\
                         see issue #46062 <https://github.com/rust-lang/rust/issues/46062>\n\
                         for more information",
                    );
                }
            }
            TypeError::TargetFeatureCast(def_id) => {
                let target_spans = find_attr!(tcx.get_all_attrs(def_id), AttributeKind::TargetFeature{attr_span: span, was_forced: false, ..} => *span);
                diag.note(
                    "functions with `#[target_feature]` can only be coerced to `unsafe` function pointers"
                );
                diag.span_labels(target_spans, "`#[target_feature]` added here");
            }
            _ => {}
        }
    }

    fn suggest_constraint(
        &self,
        diag: &mut Diag<'_>,
        msg: impl Fn() -> String,
        body_owner_def_id: DefId,
        proj_ty: ty::AliasTy<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;
        let assoc = tcx.associated_item(proj_ty.def_id);
        let (trait_ref, assoc_args) = proj_ty.trait_ref_and_own_args(tcx);
        let Some(item) = tcx.hir_get_if_local(body_owner_def_id) else {
            return false;
        };
        let Some(hir_generics) = item.generics() else {
            return false;
        };
        // Get the `DefId` for the type parameter corresponding to `A` in `<A as T>::Foo`.
        // This will also work for `impl Trait`.
        let ty::Param(param_ty) = *proj_ty.self_ty().kind() else {
            return false;
        };
        let generics = tcx.generics_of(body_owner_def_id);
        let def_id = generics.type_param(param_ty, tcx).def_id;
        let Some(def_id) = def_id.as_local() else {
            return false;
        };

        // First look in the `where` clause, as this might be
        // `fn foo<T>(x: T) where T: Trait`.
        for pred in hir_generics.bounds_for_param(def_id) {
            if self.constrain_generic_bound_associated_type_structured_suggestion(
                diag,
                trait_ref,
                pred.bounds,
                assoc,
                assoc_args,
                ty,
                &msg,
                false,
            ) {
                return true;
            }
        }
        if (param_ty.index as usize) >= generics.parent_count {
            // The param comes from the current item, do not look at the parent. (#117209)
            return false;
        }
        // If associated item, look to constrain the params of the trait/impl.
        let hir_id = match item {
            hir::Node::ImplItem(item) => item.hir_id(),
            hir::Node::TraitItem(item) => item.hir_id(),
            _ => return false,
        };
        let parent = tcx.hir_get_parent_item(hir_id).def_id;
        self.suggest_constraint(diag, msg, parent.into(), proj_ty, ty)
    }

    /// An associated type was expected and a different type was found.
    ///
    /// We perform a few different checks to see what we can suggest:
    ///
    ///  - In the current item, look for associated functions that return the expected type and
    ///    suggest calling them. (Not a structured suggestion.)
    ///  - If any of the item's generic bounds can be constrained, we suggest constraining the
    ///    associated type to the found type.
    ///  - If the associated type has a default type and was expected inside of a `trait`, we
    ///    mention that this is disallowed.
    ///  - If all other things fail, and the error is not because of a mismatch between the `trait`
    ///    and the `impl`, we provide a generic `help` to constrain the assoc type or call an assoc
    ///    fn that returns the type.
    fn expected_projection(
        &self,
        diag: &mut Diag<'_>,
        proj_ty: ty::AliasTy<'tcx>,
        values: ExpectedFound<Ty<'tcx>>,
        body_owner_def_id: DefId,
        cause_code: &ObligationCauseCode<'_>,
    ) {
        let tcx = self.tcx;

        // Don't suggest constraining a projection to something containing itself
        if self
            .tcx
            .erase_and_anonymize_regions(values.found)
            .contains(self.tcx.erase_and_anonymize_regions(values.expected))
        {
            return;
        }

        let msg = || {
            format!(
                "consider constraining the associated type `{}` to `{}`",
                values.expected, values.found
            )
        };

        let body_owner = tcx.hir_get_if_local(body_owner_def_id);
        let current_method_ident = body_owner.and_then(|n| n.ident()).map(|i| i.name);

        // We don't want to suggest calling an assoc fn in a scope where that isn't feasible.
        let callable_scope = matches!(
            body_owner,
            Some(
                hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn { .. }, .. })
                    | hir::Node::TraitItem(hir::TraitItem { kind: hir::TraitItemKind::Fn(..), .. })
                    | hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Fn(..), .. }),
            )
        );
        let impl_comparison = matches!(cause_code, ObligationCauseCode::CompareImplItem { .. });
        if impl_comparison {
            // We do not want to suggest calling functions when the reason of the
            // type error is a comparison of an `impl` with its `trait`.
        } else {
            let point_at_assoc_fn = if callable_scope
                && self.point_at_methods_that_satisfy_associated_type(
                    diag,
                    tcx.parent(proj_ty.def_id),
                    current_method_ident,
                    proj_ty.def_id,
                    values.expected,
                ) {
                // If we find a suitable associated function that returns the expected type, we
                // don't want the more general suggestion later in this method about "consider
                // constraining the associated type or calling a method that returns the associated
                // type".
                true
            } else {
                false
            };
            // Possibly suggest constraining the associated type to conform to the
            // found type.
            if self.suggest_constraint(diag, &msg, body_owner_def_id, proj_ty, values.found)
                || point_at_assoc_fn
            {
                return;
            }
        }

        self.suggest_constraining_opaque_associated_type(diag, &msg, proj_ty, values.found);

        if self.point_at_associated_type(diag, body_owner_def_id, values.found) {
            return;
        }

        if !impl_comparison {
            // Generic suggestion when we can't be more specific.
            if callable_scope {
                diag.help(format!(
                    "{} or calling a method that returns `{}`",
                    msg(),
                    values.expected
                ));
            } else {
                diag.help(msg());
            }
            diag.note(
                "for more information, visit \
                 https://doc.rust-lang.org/book/ch19-03-advanced-traits.html",
            );
        }
        if diag.code.is_some_and(|code| tcx.sess.teach(code)) {
            diag.help(
                "given an associated type `T` and a method `foo`:
```
trait Trait {
type T;
fn foo(&self) -> Self::T;
}
```
the only way of implementing method `foo` is to constrain `T` with an explicit associated type:
```
impl Trait for X {
type T = String;
fn foo(&self) -> Self::T { String::new() }
}
```",
            );
        }
    }

    /// When the expected `impl Trait` is not defined in the current item, it will come from
    /// a return type. This can occur when dealing with `TryStream` (#71035).
    fn suggest_constraining_opaque_associated_type(
        &self,
        diag: &mut Diag<'_>,
        msg: impl Fn() -> String,
        proj_ty: ty::AliasTy<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;

        let assoc = tcx.associated_item(proj_ty.def_id);
        if let ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) = *proj_ty.self_ty().kind() {
            let opaque_local_def_id = def_id.as_local();
            let opaque_hir_ty = if let Some(opaque_local_def_id) = opaque_local_def_id {
                tcx.hir_expect_opaque_ty(opaque_local_def_id)
            } else {
                return false;
            };

            let (trait_ref, assoc_args) = proj_ty.trait_ref_and_own_args(tcx);

            self.constrain_generic_bound_associated_type_structured_suggestion(
                diag,
                trait_ref,
                opaque_hir_ty.bounds,
                assoc,
                assoc_args,
                ty,
                msg,
                true,
            )
        } else {
            false
        }
    }

    fn point_at_methods_that_satisfy_associated_type(
        &self,
        diag: &mut Diag<'_>,
        assoc_container_id: DefId,
        current_method_ident: Option<Symbol>,
        proj_ty_item_def_id: DefId,
        expected: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;

        let items = tcx.associated_items(assoc_container_id);
        // Find all the methods in the trait that could be called to construct the
        // expected associated type.
        // FIXME: consider suggesting the use of associated `const`s.
        let methods: Vec<(Span, String)> = items
            .in_definition_order()
            .filter(|item| {
                item.is_fn()
                    && Some(item.name()) != current_method_ident
                    && !tcx.is_doc_hidden(item.def_id)
            })
            .filter_map(|item| {
                let method = tcx.fn_sig(item.def_id).instantiate_identity();
                match *method.output().skip_binder().kind() {
                    ty::Alias(ty::Projection, ty::AliasTy { def_id: item_def_id, .. })
                        if item_def_id == proj_ty_item_def_id =>
                    {
                        Some((
                            tcx.def_span(item.def_id),
                            format!("consider calling `{}`", tcx.def_path_str(item.def_id)),
                        ))
                    }
                    _ => None,
                }
            })
            .collect();
        if !methods.is_empty() {
            // Use a single `help:` to show all the methods in the trait that can
            // be used to construct the expected associated type.
            let mut span: MultiSpan =
                methods.iter().map(|(sp, _)| *sp).collect::<Vec<Span>>().into();
            let msg = format!(
                "{some} method{s} {are} available that return{r} `{ty}`",
                some = if methods.len() == 1 { "a" } else { "some" },
                s = pluralize!(methods.len()),
                are = pluralize!("is", methods.len()),
                r = if methods.len() == 1 { "s" } else { "" },
                ty = expected
            );
            for (sp, label) in methods.into_iter() {
                span.push_span_label(sp, label);
            }
            diag.span_help(span, msg);
            return true;
        }
        false
    }

    fn point_at_associated_type(
        &self,
        diag: &mut Diag<'_>,
        body_owner_def_id: DefId,
        found: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;

        let Some(def_id) = body_owner_def_id.as_local() else {
            return false;
        };

        // When `body_owner` is an `impl` or `trait` item, look in its associated types for
        // `expected` and point at it.
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        let parent_id = tcx.hir_get_parent_item(hir_id);
        let item = tcx.hir_node_by_def_id(parent_id.def_id);

        debug!("expected_projection parent item {:?}", item);

        let param_env = tcx.param_env(body_owner_def_id);

        if let DefKind::Trait | DefKind::Impl { .. } = tcx.def_kind(parent_id) {
            let assoc_items = tcx.associated_items(parent_id);
            // FIXME: account for `#![feature(specialization)]`
            for assoc_item in assoc_items.in_definition_order() {
                if assoc_item.is_type()
                    // FIXME: account for returning some type in a trait fn impl that has
                    // an assoc type as a return type (#72076).
                    && let hir::Defaultness::Default { has_value: true } = assoc_item.defaultness(tcx)
                    && let assoc_ty = tcx.type_of(assoc_item.def_id).instantiate_identity()
                    && self.infcx.can_eq(param_env, assoc_ty, found)
                {
                    let msg = match assoc_item.container {
                        ty::AssocContainer::Trait => {
                            "associated type defaults can't be assumed inside the \
                                            trait defining them"
                        }
                        ty::AssocContainer::InherentImpl | ty::AssocContainer::TraitImpl(_) => {
                            "associated type is `default` and may be overridden"
                        }
                    };
                    diag.span_label(tcx.def_span(assoc_item.def_id), msg);
                    return true;
                }
            }
        }

        false
    }

    /// Given a slice of `hir::GenericBound`s, if any of them corresponds to the `trait_ref`
    /// requirement, provide a structured suggestion to constrain it to a given type `ty`.
    ///
    /// `is_bound_surely_present` indicates whether we know the bound we're looking for is
    /// inside `bounds`. If that's the case then we can consider `bounds` containing only one
    /// trait bound as the one we're looking for. This can help in cases where the associated
    /// type is defined on a supertrait of the one present in the bounds.
    fn constrain_generic_bound_associated_type_structured_suggestion(
        &self,
        diag: &mut Diag<'_>,
        trait_ref: ty::TraitRef<'tcx>,
        bounds: hir::GenericBounds<'_>,
        assoc: ty::AssocItem,
        assoc_args: &[ty::GenericArg<'tcx>],
        ty: Ty<'tcx>,
        msg: impl Fn() -> String,
        is_bound_surely_present: bool,
    ) -> bool {
        // FIXME: we would want to call `resolve_vars_if_possible` on `ty` before suggesting.

        let trait_bounds = bounds.iter().filter_map(|bound| match bound {
            hir::GenericBound::Trait(ptr) if ptr.modifiers == hir::TraitBoundModifiers::NONE => {
                Some(ptr)
            }
            _ => None,
        });

        let matching_trait_bounds = trait_bounds
            .clone()
            .filter(|ptr| ptr.trait_ref.trait_def_id() == Some(trait_ref.def_id))
            .collect::<Vec<_>>();

        let span = match &matching_trait_bounds[..] {
            &[ptr] => ptr.span,
            &[] if is_bound_surely_present => match &trait_bounds.collect::<Vec<_>>()[..] {
                &[ptr] => ptr.span,
                _ => return false,
            },
            _ => return false,
        };

        self.constrain_associated_type_structured_suggestion(diag, span, assoc, assoc_args, ty, msg)
    }

    /// Given a span corresponding to a bound, provide a structured suggestion to set an
    /// associated type to a given type `ty`.
    fn constrain_associated_type_structured_suggestion(
        &self,
        diag: &mut Diag<'_>,
        span: Span,
        assoc: ty::AssocItem,
        assoc_args: &[ty::GenericArg<'tcx>],
        ty: Ty<'tcx>,
        msg: impl Fn() -> String,
    ) -> bool {
        let tcx = self.tcx;

        if let Ok(has_params) =
            tcx.sess.source_map().span_to_snippet(span).map(|snippet| snippet.ends_with('>'))
        {
            let (span, sugg) = if has_params {
                let pos = span.hi() - BytePos(1);
                let span = Span::new(pos, pos, span.ctxt(), span.parent());
                (span, format!(", {} = {}", assoc.ident(tcx), ty))
            } else {
                let item_args = self.format_generic_args(assoc_args);
                (span.shrink_to_hi(), format!("<{}{} = {}>", assoc.ident(tcx), item_args, ty))
            };
            diag.span_suggestion_verbose(span, msg(), sugg, MaybeIncorrect);
            return true;
        }
        false
    }

    pub fn format_generic_args(&self, args: &[ty::GenericArg<'tcx>]) -> String {
        FmtPrinter::print_string(self.tcx, hir::def::Namespace::TypeNS, |p| {
            p.print_path_with_generic_args(|_| Ok(()), args)
        })
        .expect("could not write to `String`.")
    }
}
