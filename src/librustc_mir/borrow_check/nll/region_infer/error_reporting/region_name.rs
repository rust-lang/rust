// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::region_infer::RegionInferenceContext;
use borrow_check::nll::ToRegionVid;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::mir::Mir;
use rustc::ty::subst::{Substs, UnpackedKind};
use rustc::ty::{self, RegionVid, Ty, TyCtxt};
use rustc_errors::DiagnosticBuilder;
use syntax::ast::Name;
use syntax::symbol::keywords;
use syntax_pos::symbol::InternedString;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Maps from an internal MIR region vid to something that we can
    /// report to the user. In some cases, the region vids will map
    /// directly to lifetimes that the user has a name for (e.g.,
    /// `'static`). But frequently they will not, in which case we
    /// have to find some way to identify the lifetime to the user. To
    /// that end, this function takes a "diagnostic" so that it can
    /// create auxiliary notes as needed.
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
    /// ```
    ///  | fn foo(x: &u32) { .. }
    ///           ------- fully elaborated type of `x` is `&'1 u32`
    /// ```
    ///
    /// and then return the name `'1` for us to use.
    crate fn give_region_a_name(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        mir: &Mir<'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder,
    ) -> InternedString {
        debug!("give_region_a_name(fr={:?}, counter={})", fr, counter);

        assert!(self.universal_regions.is_universal_region(fr));

        self.give_name_from_error_region(tcx, mir_def_id, fr, counter, diag)
            .or_else(|| {
                self.give_name_if_anonymous_region_appears_in_arguments(
                    tcx, mir, mir_def_id, fr, counter, diag)
            })
            .or_else(|| {
                self.give_name_if_anonymous_region_appears_in_upvars(
                    tcx, mir, fr, counter, diag)
            })
            .or_else(|| {
                self.give_name_if_anonymous_region_appears_in_output(tcx, mir, fr, counter, diag)
            })
            .unwrap_or_else(|| span_bug!(mir.span, "can't make a name for free region {:?}", fr))
    }

    /// Check for the case where `fr` maps to something that the
    /// *user* has a name for. In that case, we'll be able to map
    /// `fr` to a `Region<'tcx>`, and that region will be one of
    /// named variants.
    fn give_name_from_error_region(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder<'_>,
    ) -> Option<InternedString> {
        let error_region = self.to_error_region(fr)?;
        debug!("give_region_a_name: error_region = {:?}", error_region);
        match error_region {
            ty::ReEarlyBound(ebr) => Some(ebr.name),

            ty::ReStatic => Some(keywords::StaticLifetime.name().as_interned_str()),

            ty::ReFree(free_region) => match free_region.bound_region {
                ty::BoundRegion::BrNamed(_, name) => Some(name),

                ty::BoundRegion::BrEnv => {
                    let closure_span = tcx.hir.span_if_local(mir_def_id).unwrap();
                    let region_name = self.synthesize_region_name(counter);
                    diag.span_label(
                        closure_span,
                        format!("lifetime `{}` represents the closure body", region_name),
                    );
                    Some(region_name)
                }

                ty::BoundRegion::BrAnon(_) | ty::BoundRegion::BrFresh(_) => None,
            },

            ty::ReLateBound(..)
            | ty::ReScope(..)
            | ty::ReVar(..)
            | ty::ReSkolemized(..)
            | ty::ReEmpty
            | ty::ReErased
            | ty::ReClosureBound(..)
            | ty::ReCanonical(..) => None,
        }
    }

    /// Find an argument that contains `fr` and label it with a fully
    /// elaborated type, returning something like `'1`. Result looks
    /// like:
    ///
    /// ```
    ///  | fn foo(x: &u32) { .. }
    ///           ------- fully elaborated type of `x` is `&'1 u32`
    /// ```
    fn give_name_if_anonymous_region_appears_in_arguments(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        mir: &Mir<'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder<'_>,
    ) -> Option<InternedString> {
        let implicit_inputs = self.universal_regions.defining_ty.implicit_inputs();
        let argument_index = self.get_argument_index_for_region(tcx, fr)?;

        let arg_ty =
            self.universal_regions.unnormalized_input_tys[implicit_inputs + argument_index];
        if let Some(region_name) = self.give_name_if_we_can_match_hir_ty_from_argument(
            tcx,
            mir_def_id,
            fr,
            arg_ty,
            argument_index,
            counter,
            diag,
        ) {
            return Some(region_name);
        }

        let (_argument_name, argument_span) = self.get_argument_name_and_span_for_region(
            mir, argument_index);

        let region_name = self.synthesize_region_name(counter);

        diag.span_label(
            argument_span,
            format!("lifetime `{}` appears in this argument", region_name,),
        );

        Some(region_name)
    }

    fn give_name_if_we_can_match_hir_ty_from_argument(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        needle_fr: RegionVid,
        argument_ty: Ty<'tcx>,
        argument_index: usize,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder<'_>,
    ) -> Option<InternedString> {
        let mir_node_id = tcx.hir.as_local_node_id(mir_def_id)?;
        let fn_decl = tcx.hir.fn_decl(mir_node_id)?;
        let argument_hir_ty: &hir::Ty = &fn_decl.inputs[argument_index];
        match argument_hir_ty.node {
            // This indicates a variable with no type annotation, like
            // `|x|`... in that case, we can't highlight the type but
            // must highlight the variable.
            hir::TyKind::Infer => None,

            _ => self.give_name_if_we_can_match_hir_ty(
                tcx,
                needle_fr,
                argument_ty,
                argument_hir_ty,
                counter,
                diag,
            ),
        }
    }

    /// Attempts to highlight the specific part of a type annotation
    /// that contains the anonymous reference we want to give a name
    /// to. For example, we might produce an annotation like this:
    ///
    /// ```
    ///  | fn a<T>(items: &[T]) -> Box<dyn Iterator<Item=&T>> {
    ///  |                - let's call the lifetime of this reference `'1`
    /// ```
    ///
    /// the way this works is that we match up `argument_ty`, which is
    /// a `Ty<'tcx>` (the internal form of the type) with
    /// `argument_hir_ty`, a `hir::Ty` (the syntax of the type
    /// annotation). We are descending through the types stepwise,
    /// looking in to find the region `needle_fr` in the internal
    /// type.  Once we find that, we can use the span of the `hir::Ty`
    /// to add the highlight.
    ///
    /// This is a somewhat imperfect process, so long the way we also
    /// keep track of the **closest** type we've found. If we fail to
    /// find the exact `&` or `'_` to highlight, then we may fall back
    /// to highlighting that closest type instead.
    fn give_name_if_we_can_match_hir_ty(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        needle_fr: RegionVid,
        argument_ty: Ty<'tcx>,
        argument_hir_ty: &hir::Ty,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder<'_>,
    ) -> Option<InternedString> {
        let search_stack: &mut Vec<(Ty<'tcx>, &hir::Ty)> = &mut Vec::new();

        search_stack.push((argument_ty, argument_hir_ty));

        let mut closest_match: &hir::Ty = argument_hir_ty;

        while let Some((ty, hir_ty)) = search_stack.pop() {
            // While we search, also track the closet match.
            if tcx.any_free_region_meets(&ty, |r| r.to_region_vid() == needle_fr) {
                closest_match = hir_ty;
            }

            match (&ty.sty, &hir_ty.node) {
                // Check if the `argument_ty` is `&'X ..` where `'X`
                // is the region we are looking for -- if so, and we have a `&T`
                // on the RHS, then we want to highlight the `&` like so:
                //
                //     &
                //     - let's call the lifetime of this reference `'1`
                (
                    ty::TyRef(region, referent_ty, _),
                    hir::TyKind::Rptr(_lifetime, referent_hir_ty),
                ) => {
                    if region.to_region_vid() == needle_fr {
                        let region_name = self.synthesize_region_name(counter);

                        // Just grab the first character, the `&`.
                        let codemap = tcx.sess.codemap();
                        let ampersand_span = codemap.start_point(hir_ty.span);

                        diag.span_label(
                            ampersand_span,
                            format!(
                                "let's call the lifetime of this reference `{}`",
                                region_name
                            ),
                        );

                        return Some(region_name);
                    }

                    // Otherwise, let's descend into the referent types.
                    search_stack.push((referent_ty, &referent_hir_ty.ty));
                }

                // Match up something like `Foo<'1>`
                (
                    ty::TyAdt(_adt_def, substs),
                    hir::TyKind::Path(hir::QPath::Resolved(None, path)),
                ) => {
                    if let Some(last_segment) = path.segments.last() {
                        if let Some(name) = self.match_adt_and_segment(
                            substs,
                            needle_fr,
                            last_segment,
                            counter,
                            diag,
                            search_stack,
                        ) {
                            return Some(name);
                        }
                    }
                }

                // The following cases don't have lifetimes, so we
                // just worry about trying to match up the rustc type
                // with the HIR types:
                (ty::TyTuple(elem_tys), hir::TyKind::Tup(elem_hir_tys)) => {
                    search_stack.extend(elem_tys.iter().cloned().zip(elem_hir_tys));
                }

                (ty::TySlice(elem_ty), hir::TyKind::Slice(elem_hir_ty))
                | (ty::TyArray(elem_ty, _), hir::TyKind::Array(elem_hir_ty, _)) => {
                    search_stack.push((elem_ty, elem_hir_ty));
                }

                (ty::TyRawPtr(mut_ty), hir::TyKind::Ptr(mut_hir_ty)) => {
                    search_stack.push((mut_ty.ty, &mut_hir_ty.ty));
                }

                _ => {
                    // FIXME there are other cases that we could trace
                }
            }
        }

        let region_name = self.synthesize_region_name(counter);
        diag.span_label(
            closest_match.span,
            format!("lifetime `{}` appears in this type", region_name),
        );

        return Some(region_name);
    }

    /// We've found an enum/struct/union type with the substitutions
    /// `substs` and -- in the HIR -- a path type with the final
    /// segment `last_segment`. Try to find a `'_` to highlight in
    /// the generic args (or, if not, to produce new zipped pairs of
    /// types+hir to search through).
    fn match_adt_and_segment<'hir>(
        &self,
        substs: &'tcx Substs<'tcx>,
        needle_fr: RegionVid,
        last_segment: &'hir hir::PathSegment,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder<'_>,
        search_stack: &mut Vec<(Ty<'tcx>, &'hir hir::Ty)>,
    ) -> Option<InternedString> {
        // Did the user give explicit arguments? (e.g., `Foo<..>`)
        let args = last_segment.args.as_ref()?;
        let lifetime = self.try_match_adt_and_generic_args(substs, needle_fr, args, search_stack)?;
        match lifetime.name {
            hir::LifetimeName::Param(_)
            | hir::LifetimeName::Static
            | hir::LifetimeName::Underscore => {
                let region_name = self.synthesize_region_name(counter);
                let ampersand_span = lifetime.span;
                diag.span_label(ampersand_span, format!("let's call this `{}`", region_name));
                return Some(region_name);
            }

            hir::LifetimeName::Implicit => {
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
                return None;
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
        substs: &'tcx Substs<'tcx>,
        needle_fr: RegionVid,
        args: &'hir hir::GenericArgs,
        search_stack: &mut Vec<(Ty<'tcx>, &'hir hir::Ty)>,
    ) -> Option<&'hir hir::Lifetime> {
        for (kind, hir_arg) in substs.iter().zip(&args.args) {
            match (kind.unpack(), hir_arg) {
                (UnpackedKind::Lifetime(r), hir::GenericArg::Lifetime(lt)) => {
                    if r.to_region_vid() == needle_fr {
                        return Some(lt);
                    }
                }

                (UnpackedKind::Type(ty), hir::GenericArg::Type(hir_ty)) => {
                    search_stack.push((ty, hir_ty));
                }

                (UnpackedKind::Lifetime(_), _) | (UnpackedKind::Type(_), _) => {
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

    /// Find a closure upvar that contains `fr` and label it with a
    /// fully elaborated type, returning something like `'1`. Result
    /// looks like:
    ///
    /// ```
    ///  | let x = Some(&22);
    ///        - fully elaborated type of `x` is `Option<&'1 u32>`
    /// ```
    fn give_name_if_anonymous_region_appears_in_upvars(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        mir: &Mir<'tcx>,
        fr: RegionVid,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder<'_>,
    ) -> Option<InternedString> {
        let upvar_index = self.get_upvar_index_for_region(tcx, fr)?;
        let (upvar_name, upvar_span) = self.get_upvar_name_and_span_for_region(tcx, mir,
                                                                               upvar_index);
        let region_name = self.synthesize_region_name(counter);

        diag.span_label(
            upvar_span,
            format!(
                "lifetime `{}` appears in the type of `{}`",
                region_name, upvar_name.unwrap(),
            ),
        );

        Some(region_name)
    }

    /// Check for arguments appearing in the (closure) return type. It
    /// must be a closure since, in a free fn, such an argument would
    /// have to either also appear in an argument (if using elision)
    /// or be early bound (named, not in argument).
    fn give_name_if_anonymous_region_appears_in_output(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        mir: &Mir<'tcx>,
        fr: RegionVid,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder<'_>,
    ) -> Option<InternedString> {
        let return_ty = self.universal_regions.unnormalized_output_ty;
        debug!(
            "give_name_if_anonymous_region_appears_in_output: return_ty = {:?}",
            return_ty
        );
        if !tcx.any_free_region_meets(&return_ty, |r| r.to_region_vid() == fr) {
            return None;
        }

        let region_name = self.synthesize_region_name(counter);
        diag.span_label(
            mir.span,
            format!("lifetime `{}` appears in return type", region_name),
        );

        Some(region_name)
    }

    /// Create a synthetic region named `'1`, incrementing the
    /// counter.
    fn synthesize_region_name(&self, counter: &mut usize) -> InternedString {
        let c = *counter;
        *counter += 1;

        Name::intern(&format!("'{:?}", c)).as_interned_str()
    }
}
