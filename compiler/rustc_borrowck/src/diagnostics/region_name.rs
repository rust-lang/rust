#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use std::fmt::{self, Display};
use std::iter;

use rustc_data_structures::fx::IndexEntry;
use rustc_errors::{Diag, EmissionGuarantee};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_middle::ty::print::RegionHighlightMode;
use rustc_middle::ty::{self, GenericArgKind, GenericArgsRef, RegionVid, Ty};
use rustc_middle::{bug, span_bug};
use rustc_span::{DUMMY_SP, Span, Symbol, kw, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use tracing::{debug, instrument};

use crate::MirBorrowckCtxt;
use crate::universal_regions::DefiningTy;

/// A name for a particular region used in emitting diagnostics. This name could be a generated
/// name like `'1`, a name used by the user like `'a`, or a name like `'static`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RegionName {
    /// The name of the region (interned).
    pub(crate) name: Symbol,
    /// Where the region comes from.
    pub(crate) source: RegionNameSource,
}

/// Denotes the source of a region that is named by a `RegionName`. For example, a free region that
/// was named by the user would get `NamedLateParamRegion` and `'static` lifetime would get
/// `Static`. This helps to print the right kinds of diagnostics.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RegionNameSource {
    /// A bound (not free) region that was instantiated at the def site (not an HRTB).
    NamedEarlyParamRegion(Span),
    /// A free region that the user has a name (`'a`) for.
    NamedLateParamRegion(Span),
    /// The `'static` region.
    Static,
    /// The free region corresponding to the environment of a closure.
    SynthesizedFreeEnvRegion(Span, &'static str),
    /// The region corresponding to an argument.
    AnonRegionFromArgument(RegionNameHighlight),
    /// The region corresponding to a closure upvar.
    AnonRegionFromUpvar(Span, Symbol),
    /// The region corresponding to the return type of a closure.
    AnonRegionFromOutput(RegionNameHighlight, &'static str),
    /// The region from a type yielded by a coroutine.
    AnonRegionFromYieldTy(Span, Symbol),
    /// An anonymous region from an async fn.
    AnonRegionFromAsyncFn(Span),
    /// An anonymous region from an impl self type or trait
    AnonRegionFromImplSignature(Span, &'static str),
}

/// Describes what to highlight to explain to the user that we're giving an anonymous region a
/// synthesized name, and how to highlight it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RegionNameHighlight {
    /// The anonymous region corresponds to a reference that was found by traversing the type in the HIR.
    MatchedHirTy(Span),
    /// The anonymous region corresponds to a `'_` in the generics list of a struct/enum/union.
    MatchedAdtAndSegment(Span),
    /// The anonymous region corresponds to a region where the type annotation is completely missing
    /// from the code, e.g. in a closure arguments `|x| { ... }`, where `x` is a reference.
    CannotMatchHirTy(Span, Symbol),
    /// The anonymous region corresponds to a region where the type annotation is completely missing
    /// from the code, and *even if* we print out the full name of the type, the region name won't
    /// be included. This currently occurs for opaque types like `impl Future`.
    Occluded(Span, Symbol),
}

impl RegionName {
    pub(crate) fn was_named(&self) -> bool {
        match self.source {
            RegionNameSource::NamedEarlyParamRegion(..)
            | RegionNameSource::NamedLateParamRegion(..)
            | RegionNameSource::Static => true,
            RegionNameSource::SynthesizedFreeEnvRegion(..)
            | RegionNameSource::AnonRegionFromArgument(..)
            | RegionNameSource::AnonRegionFromUpvar(..)
            | RegionNameSource::AnonRegionFromOutput(..)
            | RegionNameSource::AnonRegionFromYieldTy(..)
            | RegionNameSource::AnonRegionFromAsyncFn(..)
            | RegionNameSource::AnonRegionFromImplSignature(..) => false,
        }
    }

    pub(crate) fn span(&self) -> Option<Span> {
        match self.source {
            RegionNameSource::Static => None,
            RegionNameSource::NamedEarlyParamRegion(span)
            | RegionNameSource::NamedLateParamRegion(span)
            | RegionNameSource::SynthesizedFreeEnvRegion(span, _)
            | RegionNameSource::AnonRegionFromUpvar(span, _)
            | RegionNameSource::AnonRegionFromYieldTy(span, _)
            | RegionNameSource::AnonRegionFromAsyncFn(span)
            | RegionNameSource::AnonRegionFromImplSignature(span, _) => Some(span),
            RegionNameSource::AnonRegionFromArgument(ref highlight)
            | RegionNameSource::AnonRegionFromOutput(ref highlight, _) => match *highlight {
                RegionNameHighlight::MatchedHirTy(span)
                | RegionNameHighlight::MatchedAdtAndSegment(span)
                | RegionNameHighlight::CannotMatchHirTy(span, _)
                | RegionNameHighlight::Occluded(span, _) => Some(span),
            },
        }
    }

    pub(crate) fn highlight_region_name<G: EmissionGuarantee>(&self, diag: &mut Diag<'_, G>) {
        match &self.source {
            RegionNameSource::NamedLateParamRegion(span)
            | RegionNameSource::NamedEarlyParamRegion(span) => {
                diag.span_label(*span, format!("lifetime `{self}` defined here"));
            }
            RegionNameSource::SynthesizedFreeEnvRegion(span, note) => {
                diag.span_label(*span, format!("lifetime `{self}` represents this closure's body"));
                diag.note(*note);
            }
            RegionNameSource::AnonRegionFromArgument(RegionNameHighlight::CannotMatchHirTy(
                span,
                type_name,
            )) => {
                diag.span_label(*span, format!("has type `{type_name}`"));
            }
            RegionNameSource::AnonRegionFromArgument(RegionNameHighlight::MatchedHirTy(span))
            | RegionNameSource::AnonRegionFromOutput(RegionNameHighlight::MatchedHirTy(span), _)
            | RegionNameSource::AnonRegionFromAsyncFn(span) => {
                diag.span_label(
                    *span,
                    format!("let's call the lifetime of this reference `{self}`"),
                );
            }
            RegionNameSource::AnonRegionFromArgument(
                RegionNameHighlight::MatchedAdtAndSegment(span),
            )
            | RegionNameSource::AnonRegionFromOutput(
                RegionNameHighlight::MatchedAdtAndSegment(span),
                _,
            ) => {
                diag.span_label(*span, format!("let's call this `{self}`"));
            }
            RegionNameSource::AnonRegionFromArgument(RegionNameHighlight::Occluded(
                span,
                type_name,
            )) => {
                diag.span_label(
                    *span,
                    format!("lifetime `{self}` appears in the type {type_name}"),
                );
            }
            RegionNameSource::AnonRegionFromOutput(
                RegionNameHighlight::Occluded(span, type_name),
                mir_description,
            ) => {
                diag.span_label(
                    *span,
                    format!(
                        "return type{mir_description} `{type_name}` contains a lifetime `{self}`"
                    ),
                );
            }
            RegionNameSource::AnonRegionFromUpvar(span, upvar_name) => {
                diag.span_label(
                    *span,
                    format!("lifetime `{self}` appears in the type of `{upvar_name}`"),
                );
            }
            RegionNameSource::AnonRegionFromOutput(
                RegionNameHighlight::CannotMatchHirTy(span, type_name),
                mir_description,
            ) => {
                diag.span_label(*span, format!("return type{mir_description} is {type_name}"));
            }
            RegionNameSource::AnonRegionFromYieldTy(span, type_name) => {
                diag.span_label(*span, format!("yield type is {type_name}"));
            }
            RegionNameSource::AnonRegionFromImplSignature(span, location) => {
                diag.span_label(
                    *span,
                    format!("lifetime `{self}` appears in the `impl`'s {location}"),
                );
            }
            RegionNameSource::Static => {}
        }
    }
}

impl Display for RegionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl rustc_errors::IntoDiagArg for RegionName {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        self.to_string().into_diag_arg(path)
    }
}

impl<'tcx> MirBorrowckCtxt<'_, '_, 'tcx> {
    pub(crate) fn mir_def_id(&self) -> hir::def_id::LocalDefId {
        self.body.source.def_id().expect_local()
    }

    pub(crate) fn mir_hir_id(&self) -> hir::HirId {
        self.infcx.tcx.local_def_id_to_hir_id(self.mir_def_id())
    }

    /// Generate a synthetic region named `'N`, where `N` is the next value of the counter. Then,
    /// increment the counter.
    ///
    /// This is _not_ idempotent. Call `give_region_a_name` when possible.
    pub(crate) fn synthesize_region_name(&self) -> Symbol {
        let c = self.next_region_name.replace_with(|counter| *counter + 1);
        Symbol::intern(&format!("'{c:?}"))
    }

    /// Maps from an internal MIR region vid to something that we can
    /// report to the user. In some cases, the region vids will map
    /// directly to lifetimes that the user has a name for (e.g.,
    /// `'static`). But frequently they will not, in which case we
    /// have to find some way to identify the lifetime to the user. To
    /// that end, this function takes a "diagnostic" so that it can
    /// create auxiliary notes as needed.
    ///
    /// The names are memoized, so this is both cheap to recompute and idempotent.
    ///
    /// Example (function arguments):
    ///
    /// Suppose we are trying to give a name to the lifetime of the
    /// reference `x`:
    ///
    /// ```ignore (pseudo-rust)
    /// fn foo(x: &u32) { .. }
    /// ```
    ///
    /// This function would create a label like this:
    ///
    /// ```text
    ///  | fn foo(x: &u32) { .. }
    ///           ------- fully elaborated type of `x` is `&'1 u32`
    /// ```
    ///
    /// and then return the name `'1` for us to use.
    pub(crate) fn give_region_a_name(&self, fr: RegionVid) -> Option<RegionName> {
        debug!(
            "give_region_a_name(fr={:?}, counter={:?})",
            fr,
            self.next_region_name.try_borrow().unwrap()
        );

        assert!(self.regioncx.universal_regions().is_universal_region(fr));

        match self.region_names.borrow_mut().entry(fr) {
            IndexEntry::Occupied(precomputed_name) => Some(*precomputed_name.get()),
            IndexEntry::Vacant(slot) => {
                let new_name = self
                    .give_name_from_error_region(fr)
                    .or_else(|| self.give_name_if_anonymous_region_appears_in_arguments(fr))
                    .or_else(|| self.give_name_if_anonymous_region_appears_in_upvars(fr))
                    .or_else(|| self.give_name_if_anonymous_region_appears_in_output(fr))
                    .or_else(|| self.give_name_if_anonymous_region_appears_in_yield_ty(fr))
                    .or_else(|| self.give_name_if_anonymous_region_appears_in_impl_signature(fr))
                    .or_else(|| {
                        self.give_name_if_anonymous_region_appears_in_arg_position_impl_trait(fr)
                    });

                if let Some(new_name) = new_name {
                    slot.insert(new_name);
                }
                debug!("give_region_a_name: gave name {:?}", new_name);

                new_name
            }
        }
    }

    /// Checks for the case where `fr` maps to something that the
    /// *user* has a name for. In that case, we'll be able to map
    /// `fr` to a `Region<'tcx>`, and that region will be one of
    /// named variants.
    #[instrument(level = "trace", skip(self))]
    fn give_name_from_error_region(&self, fr: RegionVid) -> Option<RegionName> {
        let error_region = self.to_error_region(fr)?;

        let tcx = self.infcx.tcx;

        debug!("give_region_a_name: error_region = {:?}", error_region);
        match error_region.kind() {
            ty::ReEarlyParam(ebr) => ebr.is_named().then(|| {
                let def_id = tcx.generics_of(self.mir_def_id()).region_param(ebr, tcx).def_id;
                let span = tcx.hir_span_if_local(def_id).unwrap_or(DUMMY_SP);
                RegionName { name: ebr.name, source: RegionNameSource::NamedEarlyParamRegion(span) }
            }),

            ty::ReStatic => {
                Some(RegionName { name: kw::StaticLifetime, source: RegionNameSource::Static })
            }

            ty::ReLateParam(late_param) => match late_param.kind {
                ty::LateParamRegionKind::Named(region_def_id) => {
                    // Get the span to point to, even if we don't use the name.
                    let span = tcx.hir_span_if_local(region_def_id).unwrap_or(DUMMY_SP);

                    if let Some(name) = late_param.kind.get_name(tcx) {
                        // A named region that is actually named.
                        Some(RegionName {
                            name,
                            source: RegionNameSource::NamedLateParamRegion(span),
                        })
                    } else if tcx.asyncness(self.mir_hir_id().owner).is_async() {
                        // If we spuriously thought that the region is named, we should let the
                        // system generate a true name for error messages. Currently this can
                        // happen if we have an elided name in an async fn for example: the
                        // compiler will generate a region named `'_`, but reporting such a name is
                        // not actually useful, so we synthesize a name for it instead.
                        let name = self.synthesize_region_name();
                        Some(RegionName {
                            name,
                            source: RegionNameSource::AnonRegionFromAsyncFn(span),
                        })
                    } else {
                        None
                    }
                }

                ty::LateParamRegionKind::ClosureEnv => {
                    let def_ty = self.regioncx.universal_regions().defining_ty;

                    let closure_kind = match def_ty {
                        DefiningTy::Closure(_, args) => args.as_closure().kind(),
                        DefiningTy::CoroutineClosure(_, args) => args.as_coroutine_closure().kind(),
                        _ => {
                            // Can't have BrEnv in functions, constants or coroutines.
                            bug!("BrEnv outside of closure.");
                        }
                    };
                    let hir::ExprKind::Closure(&hir::Closure { fn_decl_span, .. }) =
                        tcx.hir_expect_expr(self.mir_hir_id()).kind
                    else {
                        bug!("Closure is not defined by a closure expr");
                    };
                    let region_name = self.synthesize_region_name();
                    let note = match closure_kind {
                        ty::ClosureKind::Fn => {
                            "closure implements `Fn`, so references to captured variables \
                             can't escape the closure"
                        }
                        ty::ClosureKind::FnMut => {
                            "closure implements `FnMut`, so references to captured variables \
                             can't escape the closure"
                        }
                        ty::ClosureKind::FnOnce => {
                            bug!("BrEnv in a `FnOnce` closure");
                        }
                    };

                    Some(RegionName {
                        name: region_name,
                        source: RegionNameSource::SynthesizedFreeEnvRegion(fn_decl_span, note),
                    })
                }

                ty::LateParamRegionKind::Anon(_) => None,
                ty::LateParamRegionKind::NamedAnon(_, _) => bug!("only used for pretty printing"),
            },

            ty::ReBound(..)
            | ty::ReVar(..)
            | ty::RePlaceholder(..)
            | ty::ReErased
            | ty::ReError(_) => None,
        }
    }

    /// Finds an argument that contains `fr` and label it with a fully
    /// elaborated type, returning something like `'1`. Result looks
    /// like:
    ///
    /// ```text
    ///  | fn foo(x: &u32) { .. }
    ///           ------- fully elaborated type of `x` is `&'1 u32`
    /// ```
    #[instrument(level = "trace", skip(self))]
    fn give_name_if_anonymous_region_appears_in_arguments(
        &self,
        fr: RegionVid,
    ) -> Option<RegionName> {
        let implicit_inputs = self.regioncx.universal_regions().defining_ty.implicit_inputs();
        let argument_index = self.regioncx.get_argument_index_for_region(self.infcx.tcx, fr)?;

        let arg_ty = self.regioncx.universal_regions().unnormalized_input_tys
            [implicit_inputs + argument_index];
        let (_, span) = self.regioncx.get_argument_name_and_span_for_region(
            self.body,
            self.local_names(),
            argument_index,
        );

        let highlight = self
            .get_argument_hir_ty_for_highlighting(argument_index)
            .and_then(|arg_hir_ty| self.highlight_if_we_can_match_hir_ty(fr, arg_ty, arg_hir_ty))
            .unwrap_or_else(|| {
                // `highlight_if_we_cannot_match_hir_ty` needs to know the number we will give to
                // the anonymous region. If it succeeds, the `synthesize_region_name` call below
                // will increment the counter, "reserving" the number we just used.
                let counter = *self.next_region_name.try_borrow().unwrap();
                self.highlight_if_we_cannot_match_hir_ty(fr, arg_ty, span, counter)
            });

        Some(RegionName {
            name: self.synthesize_region_name(),
            source: RegionNameSource::AnonRegionFromArgument(highlight),
        })
    }

    fn get_argument_hir_ty_for_highlighting(
        &self,
        argument_index: usize,
    ) -> Option<&hir::Ty<'tcx>> {
        let fn_decl = self.infcx.tcx.hir_fn_decl_by_hir_id(self.mir_hir_id())?;
        let argument_hir_ty: &hir::Ty<'_> = fn_decl.inputs.get(argument_index)?;
        match argument_hir_ty.kind {
            // This indicates a variable with no type annotation, like
            // `|x|`... in that case, we can't highlight the type but
            // must highlight the variable.
            // NOTE(eddyb) this is handled in/by the sole caller
            // (`give_name_if_anonymous_region_appears_in_arguments`).
            hir::TyKind::Infer(()) => None,

            _ => Some(argument_hir_ty),
        }
    }

    /// Attempts to highlight the specific part of a type in an argument
    /// that has no type annotation.
    /// For example, we might produce an annotation like this:
    ///
    /// ```text
    ///  |     foo(|a, b| b)
    ///  |          -  -
    ///  |          |  |
    ///  |          |  has type `&'1 u32`
    ///  |          has type `&'2 u32`
    /// ```
    fn highlight_if_we_cannot_match_hir_ty(
        &self,
        needle_fr: RegionVid,
        ty: Ty<'tcx>,
        span: Span,
        counter: usize,
    ) -> RegionNameHighlight {
        let mut highlight = RegionHighlightMode::default();
        highlight.highlighting_region_vid(self.infcx.tcx, needle_fr, counter);
        let type_name =
            self.infcx.err_ctxt().extract_inference_diagnostics_data(ty.into(), highlight).name;

        debug!(
            "highlight_if_we_cannot_match_hir_ty: type_name={:?} needle_fr={:?}",
            type_name, needle_fr
        );
        if type_name.contains(&format!("'{counter}")) {
            // Only add a label if we can confirm that a region was labelled.
            RegionNameHighlight::CannotMatchHirTy(span, Symbol::intern(&type_name))
        } else {
            RegionNameHighlight::Occluded(span, Symbol::intern(&type_name))
        }
    }

    /// Attempts to highlight the specific part of a type annotation
    /// that contains the anonymous reference we want to give a name
    /// to. For example, we might produce an annotation like this:
    ///
    /// ```text
    ///  | fn a<T>(items: &[T]) -> Box<dyn Iterator<Item = &T>> {
    ///  |                - let's call the lifetime of this reference `'1`
    /// ```
    ///
    /// the way this works is that we match up `ty`, which is
    /// a `Ty<'tcx>` (the internal form of the type) with
    /// `hir_ty`, a `hir::Ty` (the syntax of the type
    /// annotation). We are descending through the types stepwise,
    /// looking in to find the region `needle_fr` in the internal
    /// type. Once we find that, we can use the span of the `hir::Ty`
    /// to add the highlight.
    ///
    /// This is a somewhat imperfect process, so along the way we also
    /// keep track of the **closest** type we've found. If we fail to
    /// find the exact `&` or `'_` to highlight, then we may fall back
    /// to highlighting that closest type instead.
    fn highlight_if_we_can_match_hir_ty(
        &self,
        needle_fr: RegionVid,
        ty: Ty<'tcx>,
        hir_ty: &hir::Ty<'_>,
    ) -> Option<RegionNameHighlight> {
        let search_stack: &mut Vec<(Ty<'tcx>, &hir::Ty<'_>)> = &mut vec![(ty, hir_ty)];

        while let Some((ty, hir_ty)) = search_stack.pop() {
            match (ty.kind(), &hir_ty.kind) {
                // Check if the `ty` is `&'X ..` where `'X`
                // is the region we are looking for -- if so, and we have a `&T`
                // on the RHS, then we want to highlight the `&` like so:
                //
                //     &
                //     - let's call the lifetime of this reference `'1`
                (ty::Ref(region, referent_ty, _), hir::TyKind::Ref(_lifetime, referent_hir_ty)) => {
                    if region.as_var() == needle_fr {
                        // Just grab the first character, the `&`.
                        let source_map = self.infcx.tcx.sess.source_map();
                        let ampersand_span = source_map.start_point(hir_ty.span);

                        return Some(RegionNameHighlight::MatchedHirTy(ampersand_span));
                    }

                    // Otherwise, let's descend into the referent types.
                    search_stack.push((*referent_ty, referent_hir_ty.ty));
                }

                // Match up something like `Foo<'1>`
                (ty::Adt(_adt_def, args), hir::TyKind::Path(hir::QPath::Resolved(None, path))) => {
                    match path.res {
                        // Type parameters of the type alias have no reason to
                        // be the same as those of the ADT.
                        // FIXME: We should be able to do something similar to
                        // match_adt_and_segment in this case.
                        Res::Def(DefKind::TyAlias, _) => (),
                        _ => {
                            if let Some(last_segment) = path.segments.last()
                                && let Some(highlight) = self.match_adt_and_segment(
                                    args,
                                    needle_fr,
                                    last_segment,
                                    search_stack,
                                )
                            {
                                return Some(highlight);
                            }
                        }
                    }
                }

                // The following cases don't have lifetimes, so we
                // just worry about trying to match up the rustc type
                // with the HIR types:
                (&ty::Tuple(elem_tys), hir::TyKind::Tup(elem_hir_tys)) => {
                    search_stack.extend(iter::zip(elem_tys, *elem_hir_tys));
                }

                (ty::Slice(elem_ty), hir::TyKind::Slice(elem_hir_ty))
                | (ty::Array(elem_ty, _), hir::TyKind::Array(elem_hir_ty, _)) => {
                    search_stack.push((*elem_ty, elem_hir_ty));
                }

                (ty::RawPtr(mut_ty, _), hir::TyKind::Ptr(mut_hir_ty)) => {
                    search_stack.push((*mut_ty, mut_hir_ty.ty));
                }

                _ => {
                    // FIXME there are other cases that we could trace
                }
            }
        }

        None
    }

    /// We've found an enum/struct/union type with the generic args
    /// `args` and -- in the HIR -- a path type with the final
    /// segment `last_segment`. Try to find a `'_` to highlight in
    /// the generic args (or, if not, to produce new zipped pairs of
    /// types+hir to search through).
    fn match_adt_and_segment<'hir>(
        &self,
        args: GenericArgsRef<'tcx>,
        needle_fr: RegionVid,
        last_segment: &'hir hir::PathSegment<'hir>,
        search_stack: &mut Vec<(Ty<'tcx>, &'hir hir::Ty<'hir>)>,
    ) -> Option<RegionNameHighlight> {
        // Did the user give explicit arguments? (e.g., `Foo<..>`)
        let explicit_args = last_segment.args.as_ref()?;
        let lifetime =
            self.try_match_adt_and_generic_args(args, needle_fr, explicit_args, search_stack)?;
        if lifetime.is_anonymous() {
            None
        } else {
            Some(RegionNameHighlight::MatchedAdtAndSegment(lifetime.ident.span))
        }
    }

    /// We've found an enum/struct/union type with the generic args
    /// `args` and -- in the HIR -- a path with the generic
    /// arguments `hir_args`. If `needle_fr` appears in the args, return
    /// the `hir::Lifetime` that corresponds to it. If not, push onto
    /// `search_stack` the types+hir to search through.
    fn try_match_adt_and_generic_args<'hir>(
        &self,
        args: GenericArgsRef<'tcx>,
        needle_fr: RegionVid,
        hir_args: &'hir hir::GenericArgs<'hir>,
        search_stack: &mut Vec<(Ty<'tcx>, &'hir hir::Ty<'hir>)>,
    ) -> Option<&'hir hir::Lifetime> {
        for (arg, hir_arg) in iter::zip(args, hir_args.args) {
            match (arg.kind(), hir_arg) {
                (GenericArgKind::Lifetime(r), hir::GenericArg::Lifetime(lt)) => {
                    if r.as_var() == needle_fr {
                        return Some(lt);
                    }
                }

                (GenericArgKind::Type(ty), hir::GenericArg::Type(hir_ty)) => {
                    search_stack.push((ty, hir_ty.as_unambig_ty()));
                }

                (GenericArgKind::Const(_ct), hir::GenericArg::Const(_hir_ct)) => {
                    // Lifetimes cannot be found in consts, so we don't need
                    // to search anything here.
                }

                (
                    GenericArgKind::Lifetime(_)
                    | GenericArgKind::Type(_)
                    | GenericArgKind::Const(_),
                    _,
                ) => {
                    self.dcx().span_delayed_bug(
                        hir_arg.span(),
                        format!("unmatched arg and hir arg: found {arg:?} vs {hir_arg:?}"),
                    );
                }
            }
        }

        None
    }

    /// Finds a closure upvar that contains `fr` and label it with a
    /// fully elaborated type, returning something like `'1`. Result
    /// looks like:
    ///
    /// ```text
    ///  | let x = Some(&22);
    ///        - fully elaborated type of `x` is `Option<&'1 u32>`
    /// ```
    #[instrument(level = "trace", skip(self))]
    fn give_name_if_anonymous_region_appears_in_upvars(&self, fr: RegionVid) -> Option<RegionName> {
        let upvar_index = self.regioncx.get_upvar_index_for_region(self.infcx.tcx, fr)?;
        let (upvar_name, upvar_span) = self.regioncx.get_upvar_name_and_span_for_region(
            self.infcx.tcx,
            self.upvars,
            upvar_index,
        );
        let region_name = self.synthesize_region_name();

        Some(RegionName {
            name: region_name,
            source: RegionNameSource::AnonRegionFromUpvar(upvar_span, upvar_name),
        })
    }

    /// Checks for arguments appearing in the (closure) return type. It
    /// must be a closure since, in a free fn, such an argument would
    /// have to either also appear in an argument (if using elision)
    /// or be early bound (named, not in argument).
    #[instrument(level = "trace", skip(self))]
    fn give_name_if_anonymous_region_appears_in_output(&self, fr: RegionVid) -> Option<RegionName> {
        let tcx = self.infcx.tcx;

        let return_ty = self.regioncx.universal_regions().unnormalized_output_ty;
        debug!("give_name_if_anonymous_region_appears_in_output: return_ty = {:?}", return_ty);
        if !tcx.any_free_region_meets(&return_ty, |r| r.as_var() == fr) {
            return None;
        }

        let mir_hir_id = self.mir_hir_id();

        let (return_span, mir_description, hir_ty) = match tcx.hir_node(mir_hir_id) {
            hir::Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(&hir::Closure { fn_decl, kind, fn_decl_span, .. }),
                ..
            }) => {
                let (mut span, mut hir_ty) = match fn_decl.output {
                    hir::FnRetTy::DefaultReturn(_) => {
                        (tcx.sess.source_map().end_point(fn_decl_span), None)
                    }
                    hir::FnRetTy::Return(hir_ty) => (fn_decl.output.span(), Some(hir_ty)),
                };
                let mir_description = match kind {
                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Async,
                        hir::CoroutineSource::Block,
                    )) => " of async block",

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Async,
                        hir::CoroutineSource::Closure,
                    ))
                    | hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::Async) => {
                        " of async closure"
                    }

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Async,
                        hir::CoroutineSource::Fn,
                    )) => {
                        let parent_item =
                            tcx.hir_node_by_def_id(tcx.hir_get_parent_item(mir_hir_id).def_id);
                        let output = &parent_item
                            .fn_decl()
                            .expect("coroutine lowered from async fn should be in fn")
                            .output;
                        span = output.span();
                        if let hir::FnRetTy::Return(ret) = output {
                            hir_ty = Some(self.get_future_inner_return_ty(ret));
                        }
                        " of async function"
                    }

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Gen,
                        hir::CoroutineSource::Block,
                    )) => " of gen block",

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Gen,
                        hir::CoroutineSource::Closure,
                    ))
                    | hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::Gen) => {
                        " of gen closure"
                    }

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Gen,
                        hir::CoroutineSource::Fn,
                    )) => {
                        let parent_item =
                            tcx.hir_node_by_def_id(tcx.hir_get_parent_item(mir_hir_id).def_id);
                        let output = &parent_item
                            .fn_decl()
                            .expect("coroutine lowered from gen fn should be in fn")
                            .output;
                        span = output.span();
                        " of gen function"
                    }

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::AsyncGen,
                        hir::CoroutineSource::Block,
                    )) => " of async gen block",

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::AsyncGen,
                        hir::CoroutineSource::Closure,
                    ))
                    | hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::AsyncGen) => {
                        " of async gen closure"
                    }

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::AsyncGen,
                        hir::CoroutineSource::Fn,
                    )) => {
                        let parent_item =
                            tcx.hir_node_by_def_id(tcx.hir_get_parent_item(mir_hir_id).def_id);
                        let output = &parent_item
                            .fn_decl()
                            .expect("coroutine lowered from async gen fn should be in fn")
                            .output;
                        span = output.span();
                        " of async gen function"
                    }

                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Coroutine(_)) => {
                        " of coroutine"
                    }
                    hir::ClosureKind::Closure => " of closure",
                };
                (span, mir_description, hir_ty)
            }
            node => match node.fn_decl() {
                Some(fn_decl) => {
                    let hir_ty = match fn_decl.output {
                        hir::FnRetTy::DefaultReturn(_) => None,
                        hir::FnRetTy::Return(ty) => Some(ty),
                    };
                    (fn_decl.output.span(), "", hir_ty)
                }
                None => (self.body.span, "", None),
            },
        };

        let highlight = hir_ty
            .and_then(|hir_ty| self.highlight_if_we_can_match_hir_ty(fr, return_ty, hir_ty))
            .unwrap_or_else(|| {
                // `highlight_if_we_cannot_match_hir_ty` needs to know the number we will give to
                // the anonymous region. If it succeeds, the `synthesize_region_name` call below
                // will increment the counter, "reserving" the number we just used.
                let counter = *self.next_region_name.try_borrow().unwrap();
                self.highlight_if_we_cannot_match_hir_ty(fr, return_ty, return_span, counter)
            });

        Some(RegionName {
            name: self.synthesize_region_name(),
            source: RegionNameSource::AnonRegionFromOutput(highlight, mir_description),
        })
    }

    /// From the [`hir::Ty`] of an async function's lowered return type,
    /// retrieve the `hir::Ty` representing the type the user originally wrote.
    ///
    /// e.g. given the function:
    ///
    /// ```
    /// async fn foo() -> i32 { 2 }
    /// ```
    ///
    /// this function, given the lowered return type of `foo`, an [`OpaqueDef`] that implements
    /// `Future<Output=i32>`, returns the `i32`.
    ///
    /// [`OpaqueDef`]: hir::TyKind::OpaqueDef
    fn get_future_inner_return_ty(&self, hir_ty: &'tcx hir::Ty<'tcx>) -> &'tcx hir::Ty<'tcx> {
        let hir::TyKind::OpaqueDef(opaque_ty) = hir_ty.kind else {
            span_bug!(
                hir_ty.span,
                "lowered return type of async fn is not OpaqueDef: {:?}",
                hir_ty
            );
        };
        if let hir::OpaqueTy { bounds: [hir::GenericBound::Trait(trait_ref)], .. } = opaque_ty
            && let Some(segment) = trait_ref.trait_ref.path.segments.last()
            && let Some(args) = segment.args
            && let [constraint] = args.constraints
            && constraint.ident.name == sym::Output
            && let Some(ty) = constraint.ty()
        {
            ty
        } else {
            span_bug!(
                hir_ty.span,
                "bounds from lowered return type of async fn did not match expected format: {opaque_ty:?}",
            );
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn give_name_if_anonymous_region_appears_in_yield_ty(
        &self,
        fr: RegionVid,
    ) -> Option<RegionName> {
        // Note: coroutines from `async fn` yield `()`, so we don't have to
        // worry about them here.
        let yield_ty = self.regioncx.universal_regions().yield_ty?;
        debug!("give_name_if_anonymous_region_appears_in_yield_ty: yield_ty = {:?}", yield_ty);

        let tcx = self.infcx.tcx;

        if !tcx.any_free_region_meets(&yield_ty, |r| r.as_var() == fr) {
            return None;
        }

        let mut highlight = RegionHighlightMode::default();
        highlight.highlighting_region_vid(tcx, fr, *self.next_region_name.try_borrow().unwrap());
        let type_name = self
            .infcx
            .err_ctxt()
            .extract_inference_diagnostics_data(yield_ty.into(), highlight)
            .name;

        let yield_span = match tcx.hir_node(self.mir_hir_id()) {
            hir::Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(&hir::Closure { fn_decl_span, .. }),
                ..
            }) => tcx.sess.source_map().end_point(fn_decl_span),
            _ => self.body.span,
        };

        debug!(
            "give_name_if_anonymous_region_appears_in_yield_ty: \
             type_name = {:?}, yield_span = {:?}",
            yield_span, type_name,
        );

        Some(RegionName {
            name: self.synthesize_region_name(),
            source: RegionNameSource::AnonRegionFromYieldTy(yield_span, Symbol::intern(&type_name)),
        })
    }

    fn give_name_if_anonymous_region_appears_in_impl_signature(
        &self,
        fr: RegionVid,
    ) -> Option<RegionName> {
        let ty::ReEarlyParam(region) = self.to_error_region(fr)?.kind() else {
            return None;
        };
        if region.is_named() {
            return None;
        };

        let tcx = self.infcx.tcx;
        let region_def = tcx.generics_of(self.mir_def_id()).region_param(region, tcx).def_id;
        let region_parent = tcx.parent(region_def);
        let DefKind::Impl { .. } = tcx.def_kind(region_parent) else {
            return None;
        };

        let found = tcx
            .any_free_region_meets(&tcx.type_of(region_parent).instantiate_identity(), |r| {
                r.kind() == ty::ReEarlyParam(region)
            });

        Some(RegionName {
            name: self.synthesize_region_name(),
            source: RegionNameSource::AnonRegionFromImplSignature(
                tcx.def_span(region_def),
                // FIXME(compiler-errors): Does this ever actually show up
                // anywhere other than the self type? I couldn't create an
                // example of a `'_` in the impl's trait being referenceable.
                if found { "self type" } else { "header" },
            ),
        })
    }

    fn give_name_if_anonymous_region_appears_in_arg_position_impl_trait(
        &self,
        fr: RegionVid,
    ) -> Option<RegionName> {
        let ty::ReEarlyParam(region) = self.to_error_region(fr)?.kind() else {
            return None;
        };
        if region.is_named() {
            return None;
        };

        let predicates = self
            .infcx
            .tcx
            .predicates_of(self.body.source.def_id())
            .instantiate_identity(self.infcx.tcx)
            .predicates;

        if let Some(upvar_index) = self
            .regioncx
            .universal_regions()
            .defining_ty
            .upvar_tys()
            .iter()
            .position(|ty| self.any_param_predicate_mentions(&predicates, ty, region))
        {
            let (upvar_name, upvar_span) = self.regioncx.get_upvar_name_and_span_for_region(
                self.infcx.tcx,
                self.upvars,
                upvar_index,
            );
            let region_name = self.synthesize_region_name();

            Some(RegionName {
                name: region_name,
                source: RegionNameSource::AnonRegionFromUpvar(upvar_span, upvar_name),
            })
        } else if let Some(arg_index) = self
            .regioncx
            .universal_regions()
            .unnormalized_input_tys
            .iter()
            .position(|ty| self.any_param_predicate_mentions(&predicates, *ty, region))
        {
            let (arg_name, arg_span) = self.regioncx.get_argument_name_and_span_for_region(
                self.body,
                self.local_names(),
                arg_index,
            );
            let region_name = self.synthesize_region_name();

            Some(RegionName {
                name: region_name,
                source: RegionNameSource::AnonRegionFromArgument(
                    RegionNameHighlight::CannotMatchHirTy(arg_span, arg_name?),
                ),
            })
        } else {
            None
        }
    }

    fn any_param_predicate_mentions(
        &self,
        clauses: &[ty::Clause<'tcx>],
        ty: Ty<'tcx>,
        region: ty::EarlyParamRegion,
    ) -> bool {
        let tcx = self.infcx.tcx;
        ty.walk().any(|arg| {
            if let ty::GenericArgKind::Type(ty) = arg.kind()
                && let ty::Param(_) = ty.kind()
            {
                clauses.iter().any(|pred| {
                    match pred.kind().skip_binder() {
                        ty::ClauseKind::Trait(data) if data.self_ty() == ty => {}
                        ty::ClauseKind::Projection(data)
                            if data.projection_term.self_ty() == ty => {}
                        _ => return false,
                    }
                    tcx.any_free_region_meets(pred, |r| r.kind() == ty::ReEarlyParam(region))
                })
            } else {
                false
            }
        })
    }
}
