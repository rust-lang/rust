//! Unsafety checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_errors::{codes::*, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::Unsafety;
use rustc_middle::ty::{TraitRef, TyCtxt};
use rustc_span::def_id::LocalDefId;
use rustc_span::ErrorGuaranteed;

pub(super) fn check_item(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    trait_ref: TraitRef<'_>,
) -> Result<(), ErrorGuaranteed> {
    let item = tcx.hir().expect_item(def_id);
    let impl_ = item.expect_impl();
    let trait_def = tcx.trait_def(trait_ref.def_id);
    let unsafe_attr = impl_.generics.params.iter().find(|p| p.pure_wrt_drop).map(|_| "may_dangle");
    match (trait_def.unsafety, unsafe_attr, impl_.unsafety, impl_.polarity) {
        (Unsafety::Normal, None, Unsafety::Unsafe, hir::ImplPolarity::Positive) => {
            return Err(struct_span_code_err!(
                tcx.dcx(),
                tcx.def_span(def_id),
                E0199,
                "implementing the trait `{}` is not unsafe",
                trait_ref.print_trait_sugared()
            )
            .with_span_suggestion_verbose(
                item.span.with_hi(item.span.lo() + rustc_span::BytePos(7)),
                "remove `unsafe` from this trait implementation",
                "",
                rustc_errors::Applicability::MachineApplicable,
            )
            .emit());
        }

        (Unsafety::Unsafe, _, Unsafety::Normal, hir::ImplPolarity::Positive) => {
            return Err(struct_span_code_err!(
                tcx.dcx(),
                tcx.def_span(def_id),
                E0200,
                "the trait `{}` requires an `unsafe impl` declaration",
                trait_ref.print_trait_sugared()
            )
            .with_note(format!(
                "the trait `{}` enforces invariants that the compiler can't check. \
                    Review the trait documentation and make sure this implementation \
                    upholds those invariants before adding the `unsafe` keyword",
                trait_ref.print_trait_sugared()
            ))
            .with_span_suggestion_verbose(
                item.span.shrink_to_lo(),
                "add `unsafe` to this trait implementation",
                "unsafe ",
                rustc_errors::Applicability::MaybeIncorrect,
            )
            .emit());
        }

        (Unsafety::Normal, Some(attr_name), Unsafety::Normal, hir::ImplPolarity::Positive) => {
            return Err(struct_span_code_err!(
                tcx.dcx(),
                tcx.def_span(def_id),
                E0569,
                "requires an `unsafe impl` declaration due to `#[{}]` attribute",
                attr_name
            )
            .with_note(format!(
                "the trait `{}` enforces invariants that the compiler can't check. \
                    Review the trait documentation and make sure this implementation \
                    upholds those invariants before adding the `unsafe` keyword",
                trait_ref.print_trait_sugared()
            ))
            .with_span_suggestion_verbose(
                item.span.shrink_to_lo(),
                "add `unsafe` to this trait implementation",
                "unsafe ",
                rustc_errors::Applicability::MaybeIncorrect,
            )
            .emit());
        }

        (_, _, Unsafety::Unsafe, hir::ImplPolarity::Negative(_)) => {
            // Reported in AST validation
            tcx.dcx().span_delayed_bug(item.span, "unsafe negative impl");
            Ok(())
        }
        (_, _, Unsafety::Normal, hir::ImplPolarity::Negative(_))
        | (Unsafety::Unsafe, _, Unsafety::Unsafe, hir::ImplPolarity::Positive)
        | (Unsafety::Normal, Some(_), Unsafety::Unsafe, hir::ImplPolarity::Positive)
        | (Unsafety::Normal, None, Unsafety::Normal, _) => Ok(()),
    }
}
