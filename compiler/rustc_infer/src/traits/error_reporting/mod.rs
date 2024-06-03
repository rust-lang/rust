use super::ObjectSafetyViolation;

use crate::infer::InferCtxt;
use rustc_ast::TraitObjectSyntax;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{codes::*, struct_span_code_err, Applicability, Diag, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Span;
use std::fmt;
use std::iter;

impl<'tcx> InferCtxt<'tcx> {
    pub fn report_extra_impl_obligation(
        &self,
        error_span: Span,
        impl_item_def_id: LocalDefId,
        trait_item_def_id: DefId,
        requirement: &dyn fmt::Display,
    ) -> Diag<'tcx> {
        let mut err = struct_span_code_err!(
            self.tcx.dcx(),
            error_span,
            E0276,
            "impl has stricter requirements than trait"
        );

        if !self.tcx.is_impl_trait_in_trait(trait_item_def_id) {
            if let Some(span) = self.tcx.hir().span_if_local(trait_item_def_id) {
                let item_name = self.tcx.item_name(impl_item_def_id.to_def_id());
                err.span_label(span, format!("definition of `{item_name}` from trait"));
            }
        }

        err.span_label(error_span, format!("impl has extra requirement {requirement}"));

        err
    }
}

struct TraitObjectFinder<'tcx> {
    trait_def_id: DefId,
    result: Vec<&'tcx hir::Ty<'tcx>>,
}

impl<'v> Visitor<'v> for TraitObjectFinder<'v> {
    fn visit_ty(&mut self, ty: &'v hir::Ty<'v>) {
        if let hir::TyKind::TraitObject(traits, _, _) = ty.kind
            && traits.iter().any(|t| t.trait_ref.trait_def_id() == Some(self.trait_def_id))
        {
            self.result.push(ty);
        }
        hir::intravisit::walk_ty(self, ty);
    }
}

#[tracing::instrument(level = "debug", skip(tcx))]
pub fn report_object_safety_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    hir_id: Option<hir::HirId>,
    trait_def_id: DefId,
    violations: &[ObjectSafetyViolation],
) -> Diag<'tcx> {
    let trait_str = tcx.def_path_str(trait_def_id);
    let trait_span = tcx.hir().get_if_local(trait_def_id).and_then(|node| match node {
        hir::Node::Item(item) => Some(item.ident.span),
        _ => None,
    });
    let mut visitor = TraitObjectFinder { trait_def_id, result: vec![] };
    if let Some(hir_id) = hir_id {
        match tcx.hir_node(hir_id) {
            hir::Node::Expr(expr) => {
                visitor.visit_expr(&expr);
                if visitor.result.is_empty() {
                    match tcx.parent_hir_node(hir_id) {
                        hir::Node::Expr(expr) if let hir::ExprKind::Cast(_, ty) = expr.kind => {
                            // Special case for `<expr> as <ty>`, as we're given the `expr` instead
                            // of the whole cast expression. This will let us point at `dyn Trait`
                            // instead of `x` in `x as Box<dyn Trait>`.
                            visitor.visit_ty(ty);
                        }
                        hir::Node::LetStmt(stmt) if let Some(ty) = stmt.ty => {
                            // Special case for `let <pat>: <ty> = <expr>;`, as we're given the
                            // `expr` instead of the whole cast expression. This will let us point
                            // at `dyn Trait` instead of `x` in `let y: Box<dyn Trait> = x;`.
                            visitor.visit_ty(ty);
                        }
                        _ => {}
                    }
                }
            }
            hir::Node::Ty(ty) => {
                visitor.visit_ty(&ty);
            }
            _ => {}
        }
    }
    let mut label_span = span;
    let mut dyn_trait_spans = vec![];
    let mut trait_spans = vec![];
    let spans: MultiSpan = if visitor.result.is_empty() {
        span.into()
    } else {
        for ty in &visitor.result {
            let hir::TyKind::TraitObject(traits, ..) = ty.kind else { continue };
            dyn_trait_spans.push(ty.span);
            trait_spans.extend(
                traits
                    .iter()
                    .filter(|t| t.trait_ref.trait_def_id() == Some(trait_def_id))
                    .map(|t| t.trait_ref.path.span),
            );
        }
        match (&dyn_trait_spans[..], &trait_spans[..]) {
            ([], []) => span.into(),
            ([only], [_]) => {
                // There is a single `dyn Trait` for the expression or type that was stored in the
                // `WellFormedLoc`. We point at the whole `dyn Trait`.
                label_span = *only;
                (*only).into()
            }
            (_, [.., last]) => {
                // There are more than one trait in `dyn A + A` in the expression or type that was
                // stored in the `WellFormedLoc` that points at the relevant trait, or there are
                // more than one `dyn A`. We apply the primary span label to the last one of these.
                label_span = *last;
                trait_spans.into()
            }
            // Should be unreachable, as if one is empty, the other must be too.
            _ => span.into(),
        }
    };
    let mut err = struct_span_code_err!(
        tcx.dcx(),
        spans,
        E0038,
        "the trait `{}` cannot be made into an object",
        trait_str
    );
    err.span_label(label_span, format!("`{trait_str}` cannot be made into an object"));

    let mut reported_violations = FxIndexSet::default();
    let mut multi_span = vec![];
    let mut messages = vec![];
    for violation in violations {
        if let ObjectSafetyViolation::SizedSelf(sp) = &violation
            && !sp.is_empty()
        {
            // Do not report `SizedSelf` without spans pointing at `SizedSelf` obligations
            // with a `Span`.
            reported_violations.insert(ObjectSafetyViolation::SizedSelf(vec![].into()));
        }
        if reported_violations.insert(violation.clone()) {
            let spans = violation.spans();
            let msg = if trait_span.is_none() || spans.is_empty() {
                format!("the trait cannot be made into an object because {}", violation.error_msg())
            } else {
                format!("...because {}", violation.error_msg())
            };
            if spans.is_empty() {
                err.note(msg);
            } else {
                for span in spans {
                    multi_span.push(span);
                    messages.push(msg.clone());
                }
            }
        }
    }
    let has_multi_span = !multi_span.is_empty();
    let mut note_span = MultiSpan::from_spans(multi_span.clone());
    if let (Some(trait_span), true) = (trait_span, has_multi_span) {
        note_span.push_span_label(trait_span, "this trait cannot be made into an object...");
    }
    for (span, msg) in iter::zip(multi_span, messages) {
        note_span.push_span_label(span, msg);
    }
    err.span_note(
        note_span,
        "for a trait to be \"object safe\" it needs to allow building a vtable to allow the call \
         to be resolvable dynamically; for more information visit \
         <https://doc.rust-lang.org/reference/items/traits.html#object-safety>",
    );

    // Only provide the help if its a local trait, otherwise it's not actionable.
    if trait_span.is_some() {
        let mut reported_violations: Vec<_> = reported_violations.into_iter().collect();
        reported_violations.sort();

        let mut potential_solutions: Vec<_> =
            reported_violations.into_iter().map(|violation| violation.solution()).collect();
        potential_solutions.sort();
        // Allows us to skip suggesting that the same item should be moved to another trait multiple times.
        potential_solutions.dedup();
        for solution in potential_solutions {
            solution.add_to(&mut err);
        }
    }

    let impls_of = tcx.trait_impls_of(trait_def_id);
    let impls = if impls_of.blanket_impls().is_empty() {
        impls_of
            .non_blanket_impls()
            .values()
            .flatten()
            .filter(|def_id| {
                !matches!(tcx.type_of(*def_id).instantiate_identity().kind(), ty::Dynamic(..))
            })
            .collect::<Vec<_>>()
    } else {
        vec![]
    };
    let externally_visible = if !impls.is_empty()
        && let Some(def_id) = trait_def_id.as_local()
        // We may be executing this during typeck, which would result in cycle
        // if we used effective_visibilities query, which looks into opaque types
        // (and therefore calls typeck).
        && tcx.resolutions(()).effective_visibilities.is_exported(def_id)
    {
        true
    } else {
        false
    };
    let mut has_suggested = false;
    if let Some(hir_id) = hir_id {
        let node = tcx.hir_node(hir_id);
        if let hir::Node::Ty(ty) = node
            && let hir::TyKind::TraitObject([trait_ref, ..], ..) = ty.kind
        {
            let mut hir_id = hir_id;
            while let hir::Node::Ty(ty) = tcx.parent_hir_node(hir_id) {
                hir_id = ty.hir_id;
            }
            if tcx.parent_hir_node(hir_id).fn_sig().is_some() {
                // Do not suggest `impl Trait` when dealing with things like super-traits.
                err.span_suggestion_verbose(
                    ty.span.until(trait_ref.span),
                    "consider using an opaque type instead",
                    "impl ",
                    Applicability::MaybeIncorrect,
                );
                has_suggested = true;
            }
        }
        if let hir::Node::Expr(expr) = node
            && let hir::ExprKind::Path(hir::QPath::TypeRelative(ty, path_segment)) = expr.kind
            && let hir::TyKind::TraitObject([trait_ref, ..], _, trait_object_syntax) = ty.kind
        {
            if let TraitObjectSyntax::None = trait_object_syntax
                && !expr.span.edition().at_least_rust_2021()
            {
                err.span_note(
                    trait_ref.trait_ref.path.span,
                    format!(
                        "`{trait_str}` is the type for the trait in editions 2015 and 2018 and is \
                         equivalent to writing `dyn {trait_str}`",
                    ),
                );
            }
            let segment = path_segment.ident;
            err.help(format!(
                "when writing `<dyn {trait_str}>::{segment}` you are requiring `{trait_str}` be \
                 \"object safe\", which it isn't",
            ));
            let (only, msg, sugg, appl) = if let [only] = &impls[..] {
                // We know there's a single implementation for this trait, so we can be explicit on
                // the type they should have used.
                let ty = tcx.type_of(*only).instantiate_identity();
                (
                    true,
                    format!(
                        "specify the specific `impl` for type `{ty}` to avoid requiring \"object \
                         safety\" from `{trait_str}`",
                    ),
                    with_no_trimmed_paths!(format!("{ty} as ")),
                    Applicability::MachineApplicable,
                )
            } else {
                (
                    false,
                    format!(
                        "you might have meant to access the associated function of a specific \
                         `impl` to avoid requiring \"object safety\" from `{trait_str}`, either \
                         with some explicit type...",
                    ),
                    "/* Type */ as ".to_string(),
                    Applicability::HasPlaceholders,
                )
            };
            // `<dyn Trait>::segment()` or `<Trait>::segment()` to `<Type as Trait>::segment()`
            let sp = ty.span.until(trait_ref.trait_ref.path.span);
            err.span_suggestion_verbose(sp, msg, sugg, appl);
            if !only {
                // `<dyn Trait>::segment()` or `<Trait>::segment()` to `Trait::segment()`
                err.multipart_suggestion_verbose(
                    "...or rely on inference if the compiler has enough context to identify the \
                     desired type on its own...",
                    vec![
                        (expr.span.until(trait_ref.trait_ref.path.span), String::new()),
                        (
                            path_segment
                                .ident
                                .span
                                .shrink_to_lo()
                                .with_lo(trait_ref.trait_ref.path.span.hi()),
                            "::".to_string(),
                        ),
                    ],
                    Applicability::MaybeIncorrect,
                );
                // `<dyn Trait>::segment()` or `<Trait>::segment()` to `<_ as Trait>::segment()`
                err.span_suggestion_verbose(
                    ty.span.until(trait_ref.trait_ref.path.span),
                    "...which is equivalent to",
                    format!("_ as "),
                    Applicability::MaybeIncorrect,
                );
            }
            has_suggested = true;
        }
    }
    match &impls[..] {
        _ if has_suggested => {}
        [] => {}
        _ if impls.len() > 9 => {}
        [only] if externally_visible => {
            err.help(with_no_trimmed_paths!(format!(
                "only type `{}` is seen to implement the trait in this crate, consider using it \
                 directly instead",
                tcx.type_of(*only).instantiate_identity(),
            )));
        }
        [only] => {
            err.help(with_no_trimmed_paths!(format!(
                "only type `{}` implements the trait, consider using it directly instead",
                tcx.type_of(*only).instantiate_identity(),
            )));
        }
        impls => {
            let mut types = impls
                .iter()
                .map(|t| {
                    with_no_trimmed_paths!(format!("  {}", tcx.type_of(*t).instantiate_identity(),))
                })
                .collect::<Vec<_>>();
            types.sort();
            err.help(format!(
                "the following types implement the trait, consider defining an enum where each \
                 variant holds one of these types, implementing `{}` for this new enum and using \
                 it instead:\n{}",
                trait_str,
                types.join("\n"),
            ));
        }
    }
    if externally_visible {
        err.note(format!(
            "`{trait_str}` can be implemented in other crates; if you want to support your users \
             passing their own types here, you can't refer to a specific type",
        ));
    }

    err
}
