use super::TypeErrCtxt;
use rustc_errors::Applicability::{MachineApplicable, MaybeIncorrect};
use rustc_errors::{pluralize, Diagnostic, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::error::ExpectedFound;
use rustc_middle::ty::print::Printer;
use rustc_middle::{
    traits::ObligationCause,
    ty::{self, error::TypeError, print::FmtPrinter, suggest_constraining_type_param, Ty},
};
use rustc_span::{def_id::DefId, sym, BytePos, Span, Symbol};

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    pub fn note_and_explain_type_err(
        &self,
        diag: &mut Diagnostic,
        err: TypeError<'tcx>,
        cause: &ObligationCause<'tcx>,
        sp: Span,
        body_owner_def_id: DefId,
    ) {
        use ty::error::TypeError::*;
        debug!("note_and_explain_type_err err={:?} cause={:?}", err, cause);

        let tcx = self.tcx;

        match err {
            ArgumentSorts(values, _) | Sorts(values) => {
                match (values.expected.kind(), values.found.kind()) {
                    (ty::Closure(..), ty::Closure(..)) => {
                        diag.note("no two closures, even if identical, have the same type");
                        diag.help("consider boxing your closure and/or using it as a trait object");
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
                            diag.span_suggestion(
                                sp,
                                "use a float literal",
                                format!("{}.0", snippet),
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
                    (ty::Alias(ty::Projection | ty::Inherent, _), ty::Alias(ty::Projection | ty::Inherent, _)) => {
                        diag.note("an associated type was expected, but a different one was found");
                    }
                    // FIXME(inherent_associated_types): Extend this to support `ty::Inherent`, too.
                    (ty::Param(p), ty::Alias(ty::Projection, proj)) | (ty::Alias(ty::Projection, proj), ty::Param(p))
                        if !tcx.is_impl_trait_in_trait(proj.def_id) =>
                    {
                        let p_def_id = tcx
                            .generics_of(body_owner_def_id)
                            .type_param(p, tcx)
                            .def_id;
                        let p_span = tcx.def_span(p_def_id);
                        if !sp.contains(p_span) {
                            diag.span_label(p_span, "this type parameter");
                        }
                        let hir = tcx.hir();
                        let mut note = true;
                        let parent = p_def_id
                            .as_local()
                            .and_then(|id| {
                                let local_id = hir.local_def_id_to_hir_id(id);
                                let generics = tcx.hir().find_parent(local_id)?.generics()?;
                                Some((id, generics))
                            });
                        if let Some((local_id, generics)) = parent
                        {
                            // Synthesize the associated type restriction `Add<Output = Expected>`.
                            // FIXME: extract this logic for use in other diagnostics.
                            let (trait_ref, assoc_substs) = proj.trait_ref_and_own_substs(tcx);
                            let item_name = tcx.item_name(proj.def_id);
                            let item_args = self.format_generic_args(assoc_substs);

                            // Here, we try to see if there's an existing
                            // trait implementation that matches the one that
                            // we're suggesting to restrict. If so, find the
                            // "end", whether it be at the end of the trait
                            // or the end of the generic arguments.
                            let mut matching_span = None;
                            let mut matched_end_of_args = false;
                            for bound in generics.bounds_for_param(local_id) {
                                let potential_spans = bound
                                    .bounds
                                    .iter()
                                    .find_map(|bound| {
                                        let bound_trait_path = bound.trait_ref()?.path;
                                        let def_id = bound_trait_path.res.opt_def_id()?;
                                        let generic_args = bound_trait_path.segments.iter().last().map(|path| path.args());
                                        (def_id == trait_ref.def_id).then_some((bound_trait_path.span, generic_args))
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
                                let path = format!(", {}{} = {}",item_name, item_args, p);
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
                                let path = format!("<{}{} = {}>", item_name, item_args, p);
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
                        if !sp.contains(p_span) {
                            diag.span_label(p_span, "this type parameter");
                        }
                        diag.help("type parameters must be constrained to match other types");
                        if tcx.sess.teach(&diag.get_code().unwrap()) {
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
                    (ty::Param(p), ty::Closure(..) | ty::Generator(..)) => {
                        let generics = tcx.generics_of(body_owner_def_id);
                        let p_span = tcx.def_span(generics.type_param(p, tcx).def_id);
                        if !sp.contains(p_span) {
                            diag.span_label(p_span, "this type parameter");
                        }
                        diag.help(format!(
                            "every closure has a distinct type and so could not always match the \
                             caller-chosen type of parameter `{}`",
                            p
                        ));
                    }
                    (ty::Param(p), _) | (_, ty::Param(p)) => {
                        let generics = tcx.generics_of(body_owner_def_id);
                        let p_span = tcx.def_span(generics.type_param(p, tcx).def_id);
                        if !sp.contains(p_span) {
                            diag.span_label(p_span, "this type parameter");
                        }
                    }
                    (ty::Alias(ty::Projection | ty::Inherent, proj_ty), _) if !tcx.is_impl_trait_in_trait(proj_ty.def_id) => {
                        self.expected_projection(
                            diag,
                            proj_ty,
                            values,
                            body_owner_def_id,
                            cause.code(),
                        );
                    }
                    (_, ty::Alias(ty::Projection | ty::Inherent, proj_ty)) if !tcx.is_impl_trait_in_trait(proj_ty.def_id) => {
                        let msg = || format!(
                            "consider constraining the associated type `{}` to `{}`",
                            values.found, values.expected,
                        );
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
                    (ty::Alias(ty::Opaque, alias), _) | (_, ty::Alias(ty::Opaque, alias)) if alias.def_id.is_local() && matches!(tcx.def_kind(body_owner_def_id), DefKind::AssocFn | DefKind::AssocConst) => {
                        if tcx.is_type_alias_impl_trait(alias.def_id) {
                            if !tcx.opaque_types_defined_by(body_owner_def_id.expect_local()).contains(&alias.def_id.expect_local()) {
                                let sp = tcx.def_ident_span(body_owner_def_id).unwrap_or_else(|| tcx.def_span(body_owner_def_id));
                                diag.span_note(sp, "\
                                    this item must have the opaque type in its signature \
                                    in order to be able to register hidden types");
                            }
                        }
                    }
                    (ty::FnPtr(_), ty::FnDef(def, _))
                    if let hir::def::DefKind::Fn = tcx.def_kind(def) => {
                        diag.note(
                            "when the arguments and return types match, functions can be coerced \
                             to function pointers",
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
            CyclicTy(ty) => {
                // Watch out for various cases of cyclic types and try to explain.
                if ty.is_closure() || ty.is_generator() {
                    diag.note(
                        "closures cannot capture themselves or take themselves as argument;\n\
                         this error may be the result of a recent compiler bug-fix,\n\
                         see issue #46062 <https://github.com/rust-lang/rust/issues/46062>\n\
                         for more information",
                    );
                }
            }
            TargetFeatureCast(def_id) => {
                let target_spans = tcx.get_attrs(def_id, sym::target_feature).map(|attr| attr.span);
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
        diag: &mut Diagnostic,
        msg: impl Fn() -> String,
        body_owner_def_id: DefId,
        proj_ty: &ty::AliasTy<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;
        let assoc = tcx.associated_item(proj_ty.def_id);
        let (trait_ref, assoc_substs) = proj_ty.trait_ref_and_own_substs(tcx);
        if let Some(item) = tcx.hir().get_if_local(body_owner_def_id) {
            if let Some(hir_generics) = item.generics() {
                // Get the `DefId` for the type parameter corresponding to `A` in `<A as T>::Foo`.
                // This will also work for `impl Trait`.
                let def_id = if let ty::Param(param_ty) = proj_ty.self_ty().kind() {
                    let generics = tcx.generics_of(body_owner_def_id);
                    generics.type_param(param_ty, tcx).def_id
                } else {
                    return false;
                };
                let Some(def_id) = def_id.as_local() else {
                    return false;
                };

                // First look in the `where` clause, as this might be
                // `fn foo<T>(x: T) where T: Trait`.
                for pred in hir_generics.bounds_for_param(def_id) {
                    if self.constrain_generic_bound_associated_type_structured_suggestion(
                        diag,
                        &trait_ref,
                        pred.bounds,
                        assoc,
                        assoc_substs,
                        ty,
                        &msg,
                        false,
                    ) {
                        return true;
                    }
                }
            }
        }
        false
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
        diag: &mut Diagnostic,
        proj_ty: &ty::AliasTy<'tcx>,
        values: ExpectedFound<Ty<'tcx>>,
        body_owner_def_id: DefId,
        cause_code: &ObligationCauseCode<'_>,
    ) {
        let tcx = self.tcx;

        // Don't suggest constraining a projection to something containing itself
        if self.tcx.erase_regions(values.found).contains(self.tcx.erase_regions(values.expected)) {
            return;
        }

        let msg = || {
            format!(
                "consider constraining the associated type `{}` to `{}`",
                values.expected, values.found
            )
        };

        let body_owner = tcx.hir().get_if_local(body_owner_def_id);
        let current_method_ident = body_owner.and_then(|n| n.ident()).map(|i| i.name);

        // We don't want to suggest calling an assoc fn in a scope where that isn't feasible.
        let callable_scope = matches!(
            body_owner,
            Some(
                hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(..), .. })
                    | hir::Node::TraitItem(hir::TraitItem { kind: hir::TraitItemKind::Fn(..), .. })
                    | hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Fn(..), .. }),
            )
        );
        let impl_comparison =
            matches!(cause_code, ObligationCauseCode::CompareImplItemObligation { .. });
        let assoc = tcx.associated_item(proj_ty.def_id);
        if !callable_scope || impl_comparison {
            // We do not want to suggest calling functions when the reason of the
            // type error is a comparison of an `impl` with its `trait` or when the
            // scope is outside of a `Body`.
        } else {
            // If we find a suitable associated function that returns the expected type, we don't
            // want the more general suggestion later in this method about "consider constraining
            // the associated type or calling a method that returns the associated type".
            let point_at_assoc_fn = self.point_at_methods_that_satisfy_associated_type(
                diag,
                assoc.container_id(tcx),
                current_method_ident,
                proj_ty.def_id,
                values.expected,
            );
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
        if tcx.sess.teach(&diag.get_code().unwrap()) {
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
        diag: &mut Diagnostic,
        msg: impl Fn() -> String,
        proj_ty: &ty::AliasTy<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;

        let assoc = tcx.associated_item(proj_ty.def_id);
        if let ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) = *proj_ty.self_ty().kind() {
            let opaque_local_def_id = def_id.as_local();
            let opaque_hir_ty = if let Some(opaque_local_def_id) = opaque_local_def_id {
                tcx.hir().expect_item(opaque_local_def_id).expect_opaque_ty()
            } else {
                return false;
            };

            let (trait_ref, assoc_substs) = proj_ty.trait_ref_and_own_substs(tcx);

            self.constrain_generic_bound_associated_type_structured_suggestion(
                diag,
                &trait_ref,
                opaque_hir_ty.bounds,
                assoc,
                assoc_substs,
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
        diag: &mut Diagnostic,
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
                ty::AssocKind::Fn == item.kind
                    && Some(item.name) != current_method_ident
                    && !tcx.is_doc_hidden(item.def_id)
            })
            .filter_map(|item| {
                let method = tcx.fn_sig(item.def_id).subst_identity();
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
        diag: &mut Diagnostic,
        body_owner_def_id: DefId,
        found: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;

        let Some(hir_id) = body_owner_def_id.as_local() else {
            return false;
        };
        let Some(hir_id) = tcx.opt_local_def_id_to_hir_id(hir_id) else {
            return false;
        };
        // When `body_owner` is an `impl` or `trait` item, look in its associated types for
        // `expected` and point at it.
        let parent_id = tcx.hir().get_parent_item(hir_id);
        let item = tcx.hir().find_by_def_id(parent_id.def_id);

        debug!("expected_projection parent item {:?}", item);

        let param_env = tcx.param_env(body_owner_def_id);

        match item {
            Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Trait(.., items), .. })) => {
                // FIXME: account for `#![feature(specialization)]`
                for item in &items[..] {
                    match item.kind {
                        hir::AssocItemKind::Type => {
                            // FIXME: account for returning some type in a trait fn impl that has
                            // an assoc type as a return type (#72076).
                            if let hir::Defaultness::Default { has_value: true } =
                                tcx.defaultness(item.id.owner_id)
                            {
                                let assoc_ty = tcx.type_of(item.id.owner_id).subst_identity();
                                if self.infcx.can_eq(param_env, assoc_ty, found) {
                                    diag.span_label(
                                        item.span,
                                        "associated type defaults can't be assumed inside the \
                                            trait defining them",
                                    );
                                    return true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some(hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Impl(hir::Impl { items, .. }),
                ..
            })) => {
                for item in &items[..] {
                    if let hir::AssocItemKind::Type = item.kind {
                        let assoc_ty = tcx.type_of(item.id.owner_id).subst_identity();

                        if self.infcx.can_eq(param_env, assoc_ty, found) {
                            diag.span_label(item.span, "expected this associated type");
                            return true;
                        }
                    }
                }
            }
            _ => {}
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
        diag: &mut Diagnostic,
        trait_ref: &ty::TraitRef<'tcx>,
        bounds: hir::GenericBounds<'_>,
        assoc: ty::AssocItem,
        assoc_substs: &[ty::GenericArg<'tcx>],
        ty: Ty<'tcx>,
        msg: impl Fn() -> String,
        is_bound_surely_present: bool,
    ) -> bool {
        // FIXME: we would want to call `resolve_vars_if_possible` on `ty` before suggesting.

        let trait_bounds = bounds.iter().filter_map(|bound| match bound {
            hir::GenericBound::Trait(ptr, hir::TraitBoundModifier::None) => Some(ptr),
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

        self.constrain_associated_type_structured_suggestion(
            diag,
            span,
            assoc,
            assoc_substs,
            ty,
            msg,
        )
    }

    /// Given a span corresponding to a bound, provide a structured suggestion to set an
    /// associated type to a given type `ty`.
    fn constrain_associated_type_structured_suggestion(
        &self,
        diag: &mut Diagnostic,
        span: Span,
        assoc: ty::AssocItem,
        assoc_substs: &[ty::GenericArg<'tcx>],
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
                let item_args = self.format_generic_args(assoc_substs);
                (span.shrink_to_hi(), format!("<{}{} = {}>", assoc.ident(tcx), item_args, ty))
            };
            diag.span_suggestion_verbose(span, msg(), sugg, MaybeIncorrect);
            return true;
        }
        false
    }

    pub fn format_generic_args(&self, args: &[ty::GenericArg<'tcx>]) -> String {
        FmtPrinter::new(self.tcx, hir::def::Namespace::TypeNS)
            .path_generic_args(Ok, args)
            .expect("could not write to `String`.")
            .into_buffer()
    }
}
