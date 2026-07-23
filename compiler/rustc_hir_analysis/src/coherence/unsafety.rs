//! Unsafety checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_errors::codes::*;
use rustc_errors::struct_span_code_err;
use rustc_hir::{LangItem, Safety};
use rustc_middle::ty::ImplPolarity::*;
use rustc_middle::ty::print::PrintTraitRefExt as _;
use rustc_middle::ty::{ImplTraitHeader, TraitDef, TyCtxt};
use rustc_span::ErrorGuaranteed;
use rustc_span::def_id::LocalDefId;

pub(super) fn check_item(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    trait_header: ImplTraitHeader<'_>,
    trait_def: &TraitDef,
) -> Result<(), ErrorGuaranteed> {
    let may_dangle = tcx
        .generics_of(def_id)
        .own_params
        .iter()
        .find_map(|p| rustc_hir::find_attr!(tcx, p.def_id, MayDangle { unsafe_used, span, inner_span } => (*unsafe_used, *span, *inner_span)));
    let trait_ref = trait_header.trait_ref.instantiate_identity().skip_norm_wip();

    let is_copy = tcx.is_lang_item(trait_def.def_id, LangItem::Copy);
    let trait_def_safety = if is_copy {
        // If `Self` has unsafe fields, `Copy` is unsafe to implement.
        if trait_header.trait_ref.skip_binder().self_ty().has_unsafe_fields() {
            rustc_hir::Safety::Unsafe
        } else {
            rustc_hir::Safety::Safe
        }
    } else {
        trait_def.safety
    };

    match (trait_def_safety, may_dangle, trait_header.safety, trait_header.polarity) {
        (Safety::Safe, None, Safety::Unsafe, Positive | Reservation) => {
            // `unsafe impl SafeTrait ...`
            let span = tcx.def_span(def_id);
            return Err(struct_span_code_err!(
                tcx.dcx(),
                tcx.def_span(def_id),
                E0199,
                "implementing the trait `{}` is not unsafe",
                trait_ref.print_trait_sugared()
            )
            .with_span_suggestion_verbose(
                span.with_hi(span.lo() + rustc_span::BytePos(7)),
                "remove `unsafe` from this trait implementation",
                "",
                rustc_errors::Applicability::MachineApplicable,
            )
            .emit());
        }
        // FIXME(may_dangle migration) remove this eventually
        (Safety::Safe, Some((true, attr_span, _)), Safety::Unsafe, Positive | Reservation) => {
            // `unsafe impl<#[unsafe(may_dangle)] T> Drop..`
            let span = tcx.def_span(def_id);
            let unsafe_span = span.with_hi(span.lo() + rustc_span::BytePos(6));
            return Err(struct_span_code_err!(
                tcx.dcx(),
                unsafe_span,
                E0199,
                "implementing the `Drop` trait is not unsafe",
            )
            .with_span_label(unsafe_span, "`unsafe` is redundantly used here")
            .with_span_note(
                attr_span,
                "the safety obligation of `may_dangle` has already been discharged \
                by the `unsafe` use in the `may_dangle` attribute ",
            )
            .with_span_suggestion_verbose(
                span.with_hi(span.lo() + rustc_span::BytePos(7)),
                "remove `unsafe` from this trait implementation",
                "",
                rustc_errors::Applicability::MachineApplicable,
            )
            .emit());
        }

        (Safety::Unsafe, _, Safety::Safe, Positive | Reservation) => {
            let span = tcx.def_span(def_id);
            return Err(struct_span_code_err!(
                tcx.dcx(),
                span,
                E0200,
                "the trait `{}` requires an `unsafe impl` declaration",
                trait_ref.print_trait_sugared()
            )
            .with_note(if is_copy {
                format!(
                    "the trait `{}` cannot be safely implemented for `{}` \
                        because it has unsafe fields. Review the invariants \
                        of those fields before adding an `unsafe impl`",
                    trait_ref.print_trait_sugared(),
                    trait_ref.self_ty(),
                )
            } else {
                format!(
                    "the trait `{}` enforces invariants that the compiler can't check. \
                        Review the trait documentation and make sure this implementation \
                        upholds those invariants before adding the `unsafe` keyword",
                    trait_ref.print_trait_sugared()
                )
            })
            .with_span_suggestion_verbose(
                span.shrink_to_lo(),
                "add `unsafe` to this trait implementation",
                "unsafe ",
                rustc_errors::Applicability::MaybeIncorrect,
            )
            .emit());
        }

        // FIXME(may_dangle migration) remove this eventually
        (
            Safety::Safe,
            Some((unsafe_used, may_dangle_span, inner_span)),
            Safety::Safe,
            Positive | Reservation,
        ) => {
            if !unsafe_used {
                // `impl<#[may_dangle] T> Drop...` is used
                //
                // At this time, both `unsafe impl<#[may_dangle] T> Drop...`
                // and `impl<#[unsafe(may_dangle)] T> Drop...` are valid
                return Err(tcx
                .dcx()
                .struct_span_err(may_dangle_span, "usage of the unsafe `may_dangle` attribute")
                .with_note(
                    "the `may_dangle` attribute enforces invariants that the compiler can't check. \
                    Review its documentation and make sure this implementation \
                    upholds those invariants before adding the `unsafe` keyword",
                )
                .with_note(
                    "if you are involved with bootstrapping the compiler you must use \
                    `unsafe impl Drop<#[may_dangle]...`, for `#[unsafe(may_dangle)]` \
                    cannot yet be used in the beta compiler",
                )
                .with_span_suggestion_verbose(
                    inner_span,
                    "add `unsafe` to this attribute",
                    "unsafe(may_dangle)",
                    rustc_errors::Applicability::MaybeIncorrect,
                )
                .emit());
            } else {
                // `impl <#[unsafe(may_dangle)] T> Drop...`
                Ok(())
            }
        }

        (_, _, Safety::Unsafe, Negative) => {
            // Reported in AST validation
            assert!(tcx.dcx().has_errors().is_some(), "unsafe negative impl");
            Ok(())
        }
        (_, _, Safety::Safe, Negative)
        | (Safety::Unsafe, _, Safety::Unsafe, Positive | Reservation)
        | (Safety::Safe, Some(_), Safety::Unsafe, Positive | Reservation)
        | (Safety::Safe, None, Safety::Safe, _) => Ok(()),
    }
}
