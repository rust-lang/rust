use std::fmt::{self, Display};

use rustc_errors::DiagnosticBuilder;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_middle::ty::print::RegionHighlightMode;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, RegionVid, Ty};
use rustc_span::symbol::kw;
use rustc_span::{symbol::Symbol, Span, DUMMY_SP};

use crate::borrow_check::{nll::ToRegionVid, universal_regions::DefiningTy, MirBorrowckCtxt};

/// A name for a particular region used in emitting diagnostics. This name could be a generated
/// name like `'1`, a name used by the user like `'a`, or a name like `'static`.
#[derive(Debug, Clone)]
crate struct RegionName {
    /// The name of the region (interned).
    crate name: Symbol,
    /// Where the region comes from.
    crate source: RegionNameSource,
}

/// Denotes the source of a region that is named by a `RegionName`. For example, a free region that
/// was named by the user would get `NamedFreeRegion` and `'static` lifetime would get `Static`.
/// This helps to print the right kinds of diagnostics.
#[derive(Debug, Clone)]
crate enum RegionNameSource {
    /// A bound (not free) region that was substituted at the def site (not an HRTB).
    NamedEarlyBoundRegion(Span),
    /// A free region that the user has a name (`'a`) for.
    NamedFreeRegion(Span),
    /// The `'static` region.
    Static,
    /// The free region corresponding to the environment of a closure.
    SynthesizedFreeEnvRegion(Span, String),
    /// The region name corresponds to a region where the type annotation is completely missing
    /// from the code, e.g. in a closure arguments `|x| { ... }`, where `x` is a reference.
    CannotMatchHirTy(Span, String),
    /// The region name corresponds a reference that was found by traversing the type in the HIR.
    MatchedHirTy(Span),
    /// A region name from the generics list of a struct/enum/union.
    MatchedAdtAndSegment(Span),
    /// The region corresponding to a closure upvar.
    AnonRegionFromUpvar(Span, String),
    /// The region corresponding to the return type of a closure.
    AnonRegionFromOutput(Span, String, String),
    /// The region from a type yielded by a generator.
    AnonRegionFromYieldTy(Span, String),
    /// An anonymous region from an async fn.
    AnonRegionFromAsyncFn(Span),
}

impl RegionName {
    crate fn was_named(&self) -> bool {
        match self.source {
            RegionNameSource::NamedEarlyBoundRegion(..)
            | RegionNameSource::NamedFreeRegion(..)
            | RegionNameSource::Static => true,
            RegionNameSource::SynthesizedFreeEnvRegion(..)
            | RegionNameSource::CannotMatchHirTy(..)
            | RegionNameSource::MatchedHirTy(..)
            | RegionNameSource::MatchedAdtAndSegment(..)
            | RegionNameSource::AnonRegionFromUpvar(..)
            | RegionNameSource::AnonRegionFromOutput(..)
            | RegionNameSource::AnonRegionFromYieldTy(..)
            | RegionNameSource::AnonRegionFromAsyncFn(..) => false,
        }
    }

    crate fn highlight_region_name(&self, diag: &mut DiagnosticBuilder<'_>) {
        match &self.source {
            RegionNameSource::NamedFreeRegion(span)
            | RegionNameSource::NamedEarlyBoundRegion(span) => {
                diag.span_label(*span, format!("lifetime `{}` defined here", self));
            }
            RegionNameSource::SynthesizedFreeEnvRegion(span, note) => {
                diag.span_label(
                    *span,
                    format!("lifetime `{}` represents this closure's body", self),
                );
                diag.note(&note);
            }
            RegionNameSource::CannotMatchHirTy(span, type_name) => {
                diag.span_label(*span, format!("has type `{}`", type_name));
            }
            RegionNameSource::MatchedHirTy(span)
            | RegionNameSource::AnonRegionFromAsyncFn(span) => {
                diag.span_label(
                    *span,
                    format!("let's call the lifetime of this reference `{}`", self),
                );
            }
            RegionNameSource::MatchedAdtAndSegment(span) => {
                diag.span_label(*span, format!("let's call this `{}`", self));
            }
            RegionNameSource::AnonRegionFromUpvar(span, upvar_name) => {
                diag.span_label(
                    *span,
                    format!("lifetime `{}` appears in the type of `{}`", self, upvar_name),
                );
            }
            RegionNameSource::AnonRegionFromOutput(span, mir_description, type_name) => {
                diag.span_label(*span, format!("return type{} is {}", mir_description, type_name));
            }
            RegionNameSource::AnonRegionFromYieldTy(span, type_name) => {
                diag.span_label(*span, format!("yield type is {}", type_name));
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

impl<'tcx> MirBorrowckCtxt<'_, 'tcx> {
    /// Generate a synthetic region named `'N`, where `N` is the next value of the counter. Then,
    /// increment the counter.
    ///
    /// This is _not_ idempotent. Call `give_region_a_name` when possible.
    fn synthesize_region_name(&self) -> Symbol {
        let c = self.next_region_name.replace_with(|counter| *counter + 1);
        Symbol::intern(&format!("'{:?}", c))
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
    /// ```
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
    crate fn give_region_a_name(&self, fr: RegionVid) -> Option<RegionName> {
        debug!(
            "give_region_a_name(fr={:?}, counter={:?})",
            fr,
            self.next_region_name.try_borrow().unwrap()
        );

        assert!(self.regioncx.universal_regions().is_universal_region(fr));

        if let Some(value) = self.region_names.try_borrow_mut().unwrap().get(&fr) {
            return Some(value.clone());
        }

        let value = self
            .give_name_from_error_region(fr)
            .or_else(|| self.give_name_if_anonymous_region_appears_in_arguments(fr))
            .or_else(|| self.give_name_if_anonymous_region_appears_in_upvars(fr))
            .or_else(|| self.give_name_if_anonymous_region_appears_in_output(fr))
            .or_else(|| self.give_name_if_anonymous_region_appears_in_yield_ty(fr));

        if let Some(ref value) = value {
            self.region_names.try_borrow_mut().unwrap().insert(fr, value.clone());
        }

        debug!("give_region_a_name: gave name {:?}", value);
        value
    }

    /// Checks for the case where `fr` maps to something that the
    /// *user* has a name for. In that case, we'll be able to map
    /// `fr` to a `Region<'tcx>`, and that region will be one of
    /// named variants.
    fn give_name_from_error_region(&self, fr: RegionVid) -> Option<RegionName> {
        let error_region = self.to_error_region(fr)?;

        let tcx = self.infcx.tcx;

        debug!("give_region_a_name: error_region = {:?}", error_region);
        match error_region {
            ty::ReEarlyBound(ebr) => {
                if ebr.has_name() {
                    let span = tcx.hir().span_if_local(ebr.def_id).unwrap_or(DUMMY_SP);
                    Some(RegionName {
                        name: ebr.name,
                        source: RegionNameSource::NamedEarlyBoundRegion(span),
                    })
                } else {
                    None
                }
            }

            ty::ReStatic => {
                Some(RegionName { name: kw::StaticLifetime, source: RegionNameSource::Static })
            }

            ty::ReFree(free_region) => match free_region.bound_region {
                ty::BoundRegion::BrNamed(region_def_id, name) => {
                    // Get the span to point to, even if we don't use the name.
                    let span = tcx.hir().span_if_local(region_def_id).unwrap_or(DUMMY_SP);
                    debug!(
                        "bound region named: {:?}, is_named: {:?}",
                        name,
                        free_region.bound_region.is_named()
                    );

                    if free_region.bound_region.is_named() {
                        // A named region that is actually named.
                        Some(RegionName { name, source: RegionNameSource::NamedFreeRegion(span) })
                    } else {
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
                    }
                }

                ty::BoundRegion::BrEnv => {
                    let mir_hir_id = self
                        .infcx
                        .tcx
                        .hir()
                        .as_local_hir_id(self.mir_def_id)
                        .expect("non-local mir");
                    let def_ty = self.regioncx.universal_regions().defining_ty;

                    if let DefiningTy::Closure(_, substs) = def_ty {
                        let args_span = if let hir::ExprKind::Closure(_, _, _, span, _) =
                            tcx.hir().expect_expr(mir_hir_id).kind
                        {
                            span
                        } else {
                            bug!("Closure is not defined by a closure expr");
                        };
                        let region_name = self.synthesize_region_name();

                        let closure_kind_ty = substs.as_closure().kind_ty();
                        let note = match closure_kind_ty.to_opt_closure_kind() {
                            Some(ty::ClosureKind::Fn) => {
                                "closure implements `Fn`, so references to captured variables \
                                 can't escape the closure"
                            }
                            Some(ty::ClosureKind::FnMut) => {
                                "closure implements `FnMut`, so references to captured variables \
                                 can't escape the closure"
                            }
                            Some(ty::ClosureKind::FnOnce) => {
                                bug!("BrEnv in a `FnOnce` closure");
                            }
                            None => bug!("Closure kind not inferred in borrow check"),
                        };

                        Some(RegionName {
                            name: region_name,
                            source: RegionNameSource::SynthesizedFreeEnvRegion(
                                args_span,
                                note.to_string(),
                            ),
                        })
                    } else {
                        // Can't have BrEnv in functions, constants or generators.
                        bug!("BrEnv outside of closure.");
                    }
                }

                ty::BoundRegion::BrAnon(_) => None,
            },

            ty::ReLateBound(..)
            | ty::ReScope(..)
            | ty::ReVar(..)
            | ty::RePlaceholder(..)
            | ty::ReEmpty(_)
            | ty::ReErased => None,
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
    fn give_name_if_anonymous_region_appears_in_arguments(
        &self,
        fr: RegionVid,
    ) -> Option<RegionName> {
        let implicit_inputs = self.regioncx.universal_regions().defining_ty.implicit_inputs();
        let argument_index = self.regioncx.get_argument_index_for_region(self.infcx.tcx, fr)?;

        let arg_ty = self.regioncx.universal_regions().unnormalized_input_tys
            [implicit_inputs + argument_index];
        if let Some(region_name) =
            self.give_name_if_we_can_match_hir_ty_from_argument(fr, arg_ty, argument_index)
        {
            return Some(region_name);
        }

        self.give_name_if_we_cannot_match_hir_ty(fr, arg_ty)
    }

    fn give_name_if_we_can_match_hir_ty_from_argument(
        &self,
        needle_fr: RegionVid,
        argument_ty: Ty<'tcx>,
        argument_index: usize,
    ) -> Option<RegionName> {
        let mir_hir_id = self.infcx.tcx.hir().as_local_hir_id(self.mir_def_id)?;
        let fn_decl = self.infcx.tcx.hir().fn_decl_by_hir_id(mir_hir_id)?;
        let argument_hir_ty: &hir::Ty<'_> = fn_decl.inputs.get(argument_index)?;
        match argument_hir_ty.kind {
            // This indicates a variable with no type annotation, like
            // `|x|`... in that case, we can't highlight the type but
            // must highlight the variable.
            // NOTE(eddyb) this is handled in/by the sole caller
            // (`give_name_if_anonymous_region_appears_in_arguments`).
            hir::TyKind::Infer => None,

            _ => self.give_name_if_we_can_match_hir_ty(needle_fr, argument_ty, argument_hir_ty),
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
    fn give_name_if_we_cannot_match_hir_ty(
        &self,
        needle_fr: RegionVid,
        argument_ty: Ty<'tcx>,
    ) -> Option<RegionName> {
        let counter = *self.next_region_name.try_borrow().unwrap();
        let mut highlight = RegionHighlightMode::default();
        highlight.highlighting_region_vid(needle_fr, counter);
        let type_name = self.infcx.extract_type_name(&argument_ty, Some(highlight)).0;

        debug!(
            "give_name_if_we_cannot_match_hir_ty: type_name={:?} needle_fr={:?}",
            type_name, needle_fr
        );
        let assigned_region_name = if type_name.find(&format!("'{}", counter)).is_some() {
            // Only add a label if we can confirm that a region was labelled.
            let argument_index =
                self.regioncx.get_argument_index_for_region(self.infcx.tcx, needle_fr)?;
            let (_, span) = self.regioncx.get_argument_name_and_span_for_region(
                &self.body,
                &self.local_names,
                argument_index,
            );

            Some(RegionName {
                // This counter value will already have been used, so this function will increment
                // it so the next value will be used next and return the region name that would
                // have been used.
                name: self.synthesize_region_name(),
                source: RegionNameSource::CannotMatchHirTy(span, type_name),
            })
        } else {
            None
        };

        assigned_region_name
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
    /// the way this works is that we match up `argument_ty`, which is
    /// a `Ty<'tcx>` (the internal form of the type) with
    /// `argument_hir_ty`, a `hir::Ty` (the syntax of the type
    /// annotation). We are descending through the types stepwise,
    /// looking in to find the region `needle_fr` in the internal
    /// type. Once we find that, we can use the span of the `hir::Ty`
    /// to add the highlight.
    ///
    /// This is a somewhat imperfect process, so along the way we also
    /// keep track of the **closest** type we've found. If we fail to
    /// find the exact `&` or `'_` to highlight, then we may fall back
    /// to highlighting that closest type instead.
    fn give_name_if_we_can_match_hir_ty(
        &self,
        needle_fr: RegionVid,
        argument_ty: Ty<'tcx>,
        argument_hir_ty: &hir::Ty<'_>,
    ) -> Option<RegionName> {
        let search_stack: &mut Vec<(Ty<'tcx>, &hir::Ty<'_>)> =
            &mut vec![(argument_ty, argument_hir_ty)];

        while let Some((ty, hir_ty)) = search_stack.pop() {
            match (&ty.kind, &hir_ty.kind) {
                // Check if the `argument_ty` is `&'X ..` where `'X`
                // is the region we are looking for -- if so, and we have a `&T`
                // on the RHS, then we want to highlight the `&` like so:
                //
                //     &
                //     - let's call the lifetime of this reference `'1`
                (
                    ty::Ref(region, referent_ty, _),
                    hir::TyKind::Rptr(_lifetime, referent_hir_ty),
                ) => {
                    if region.to_region_vid() == needle_fr {
                        let region_name = self.synthesize_region_name();

                        // Just grab the first character, the `&`.
                        let source_map = self.infcx.tcx.sess.source_map();
                        let ampersand_span = source_map.start_point(hir_ty.span);

                        return Some(RegionName {
                            name: region_name,
                            source: RegionNameSource::MatchedHirTy(ampersand_span),
                        });
                    }

                    // Otherwise, let's descend into the referent types.
                    search_stack.push((referent_ty, &referent_hir_ty.ty));
                }

                // Match up something like `Foo<'1>`
                (
                    ty::Adt(_adt_def, substs),
                    hir::TyKind::Path(hir::QPath::Resolved(None, path)),
                ) => {
                    match path.res {
                        // Type parameters of the type alias have no reason to
                        // be the same as those of the ADT.
                        // FIXME: We should be able to do something similar to
                        // match_adt_and_segment in this case.
                        Res::Def(DefKind::TyAlias, _) => (),
                        _ => {
                            if let Some(last_segment) = path.segments.last() {
                                if let Some(name) = self.match_adt_and_segment(
                                    substs,
                                    needle_fr,
                                    last_segment,
                                    search_stack,
                                ) {
                                    return Some(name);
                                }
                            }
                        }
                    }
                }

                // The following cases don't have lifetimes, so we
                // just worry about trying to match up the rustc type
                // with the HIR types:
                (ty::Tuple(elem_tys), hir::TyKind::Tup(elem_hir_tys)) => {
                    search_stack.extend(elem_tys.iter().map(|k| k.expect_ty()).zip(*elem_hir_tys));
                }

                (ty::Slice(elem_ty), hir::TyKind::Slice(elem_hir_ty))
                | (ty::Array(elem_ty, _), hir::TyKind::Array(elem_hir_ty, _)) => {
                    search_stack.push((elem_ty, elem_hir_ty));
                }

                (ty::RawPtr(mut_ty), hir::TyKind::Ptr(mut_hir_ty)) => {
                    search_stack.push((mut_ty.ty, &mut_hir_ty.ty));
                }

                _ => {
                    // FIXME there are other cases that we could trace
                }
            }
        }

        None
    }

    /// We've found an enum/struct/union type with the substitutions
    /// `substs` and -- in the HIR -- a path type with the final
    /// segment `last_segment`. Try to find a `'_` to highlight in
    /// the generic args (or, if not, to produce new zipped pairs of
    /// types+hir to search through).
    fn match_adt_and_segment<'hir>(
        &self,
        substs: SubstsRef<'tcx>,
        needle_fr: RegionVid,
        last_segment: &'hir hir::PathSegment<'hir>,
        search_stack: &mut Vec<(Ty<'tcx>, &'hir hir::Ty<'hir>)>,
    ) -> Option<RegionName> {
        // Did the user give explicit arguments? (e.g., `Foo<..>`)
        let args = last_segment.args.as_ref()?;
        let lifetime =
            self.try_match_adt_and_generic_args(substs, needle_fr, args, search_stack)?;
        match lifetime.name {
            hir::LifetimeName::Param(_)
            | hir::LifetimeName::Error
            | hir::LifetimeName::Static
            | hir::LifetimeName::Underscore => {
                let region_name = self.synthesize_region_name();
                let ampersand_span = lifetime.span;
                Some(RegionName {
                    name: region_name,
                    source: RegionNameSource::MatchedAdtAndSegment(ampersand_span),
                })
            }

            hir::LifetimeName::ImplicitObjectLifetimeDefault | hir::LifetimeName::Implicit => {
                // In this case, the user left off the lifetime; so
                // they wrote something like:
                //
                // ```
                // x: Foo<T>
                // ```
                //
                // where the fully elaborated form is `Foo<'_, '1,
                // T>`. We don't consider this a match; instead we let
                // the "fully elaborated" type fallback above handle
                // it.
                None
            }
        }
    }

    /// We've found an enum/struct/union type with the substitutions
    /// `substs` and -- in the HIR -- a path with the generic
    /// arguments `args`. If `needle_fr` appears in the args, return
    /// the `hir::Lifetime` that corresponds to it. If not, push onto
    /// `search_stack` the types+hir to search through.
    fn try_match_adt_and_generic_args<'hir>(
        &self,
        substs: SubstsRef<'tcx>,
        needle_fr: RegionVid,
        args: &'hir hir::GenericArgs<'hir>,
        search_stack: &mut Vec<(Ty<'tcx>, &'hir hir::Ty<'hir>)>,
    ) -> Option<&'hir hir::Lifetime> {
        for (kind, hir_arg) in substs.iter().zip(args.args) {
            match (kind.unpack(), hir_arg) {
                (GenericArgKind::Lifetime(r), hir::GenericArg::Lifetime(lt)) => {
                    if r.to_region_vid() == needle_fr {
                        return Some(lt);
                    }
                }

                (GenericArgKind::Type(ty), hir::GenericArg::Type(hir_ty)) => {
                    search_stack.push((ty, hir_ty));
                }

                (GenericArgKind::Const(_ct), hir::GenericArg::Const(_hir_ct)) => {
                    // Lifetimes cannot be found in consts, so we don't need
                    // to search anything here.
                }

                (GenericArgKind::Lifetime(_), _)
                | (GenericArgKind::Type(_), _)
                | (GenericArgKind::Const(_), _) => {
                    // I *think* that HIR lowering should ensure this
                    // doesn't happen, even in erroneous
                    // programs. Else we should use delay-span-bug.
                    span_bug!(
                        hir_arg.span(),
                        "unmatched subst and hir arg: found {:?} vs {:?}",
                        kind,
                        hir_arg,
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
    fn give_name_if_anonymous_region_appears_in_upvars(&self, fr: RegionVid) -> Option<RegionName> {
        let upvar_index = self.regioncx.get_upvar_index_for_region(self.infcx.tcx, fr)?;
        let (upvar_name, upvar_span) = self.regioncx.get_upvar_name_and_span_for_region(
            self.infcx.tcx,
            &self.upvars,
            upvar_index,
        );
        let region_name = self.synthesize_region_name();

        Some(RegionName {
            name: region_name,
            source: RegionNameSource::AnonRegionFromUpvar(upvar_span, upvar_name.to_string()),
        })
    }

    /// Checks for arguments appearing in the (closure) return type. It
    /// must be a closure since, in a free fn, such an argument would
    /// have to either also appear in an argument (if using elision)
    /// or be early bound (named, not in argument).
    fn give_name_if_anonymous_region_appears_in_output(&self, fr: RegionVid) -> Option<RegionName> {
        let tcx = self.infcx.tcx;

        let return_ty = self.regioncx.universal_regions().unnormalized_output_ty;
        debug!("give_name_if_anonymous_region_appears_in_output: return_ty = {:?}", return_ty);
        if !tcx.any_free_region_meets(&return_ty, |r| r.to_region_vid() == fr) {
            return None;
        }

        let mut highlight = RegionHighlightMode::default();
        highlight.highlighting_region_vid(fr, *self.next_region_name.try_borrow().unwrap());
        let type_name = self.infcx.extract_type_name(&return_ty, Some(highlight)).0;

        let mir_hir_id = tcx.hir().as_local_hir_id(self.mir_def_id).expect("non-local mir");

        let (return_span, mir_description) = match tcx.hir().get(mir_hir_id) {
            hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(_, return_ty, _, span, gen_move),
                ..
            }) => (
                match return_ty.output {
                    hir::FnRetTy::DefaultReturn(_) => tcx.sess.source_map().end_point(*span),
                    hir::FnRetTy::Return(_) => return_ty.output.span(),
                },
                if gen_move.is_some() { " of generator" } else { " of closure" },
            ),
            hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(method_sig, _),
                ..
            }) => (method_sig.decl.output.span(), ""),
            _ => (self.body.span, ""),
        };

        Some(RegionName {
            // This counter value will already have been used, so this function will increment it
            // so the next value will be used next and return the region name that would have been
            // used.
            name: self.synthesize_region_name(),
            source: RegionNameSource::AnonRegionFromOutput(
                return_span,
                mir_description.to_string(),
                type_name,
            ),
        })
    }

    fn give_name_if_anonymous_region_appears_in_yield_ty(
        &self,
        fr: RegionVid,
    ) -> Option<RegionName> {
        // Note: generators from `async fn` yield `()`, so we don't have to
        // worry about them here.
        let yield_ty = self.regioncx.universal_regions().yield_ty?;
        debug!("give_name_if_anonymous_region_appears_in_yield_ty: yield_ty = {:?}", yield_ty,);

        let tcx = self.infcx.tcx;

        if !tcx.any_free_region_meets(&yield_ty, |r| r.to_region_vid() == fr) {
            return None;
        }

        let mut highlight = RegionHighlightMode::default();
        highlight.highlighting_region_vid(fr, *self.next_region_name.try_borrow().unwrap());
        let type_name = self.infcx.extract_type_name(&yield_ty, Some(highlight)).0;

        let mir_hir_id = tcx.hir().as_local_hir_id(self.mir_def_id).expect("non-local mir");

        let yield_span = match tcx.hir().get(mir_hir_id) {
            hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(_, _, _, span, _), ..
            }) => (tcx.sess.source_map().end_point(*span)),
            _ => self.body.span,
        };

        debug!(
            "give_name_if_anonymous_region_appears_in_yield_ty: \
             type_name = {:?}, yield_span = {:?}",
            yield_span, type_name,
        );

        Some(RegionName {
            name: self.synthesize_region_name(),
            source: RegionNameSource::AnonRegionFromYieldTy(yield_span, type_name),
        })
    }
}
