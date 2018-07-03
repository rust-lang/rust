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
use rustc::hir::def_id::DefId;
use rustc::mir::{Local, Mir};
use rustc::ty::{self, RegionVid, TyCtxt};
use rustc_data_structures::indexed_vec::Idx;
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
                self.give_name_if_anonymous_region_appears_in_arguments(tcx, mir, fr, counter, diag)
            })
            .or_else(|| {
                self.give_name_if_anonymous_region_appears_in_upvars(tcx, mir, fr, counter, diag)
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
        fr: RegionVid,
        counter: &mut usize,
        diag: &mut DiagnosticBuilder<'_>,
    ) -> Option<InternedString> {
        let implicit_inputs = self.universal_regions.defining_ty.implicit_inputs();
        let argument_index = self.universal_regions
            .unnormalized_input_tys
            .iter()
            .skip(implicit_inputs)
            .position(|arg_ty| {
                debug!("give_name_if_anonymous_region_appears_in_arguments: arg_ty = {:?}", arg_ty);
                tcx.any_free_region_meets(arg_ty, |r| r.to_region_vid() == fr)
            })?
            + implicit_inputs;

        debug!(
            "give_name_if_anonymous_region_appears_in_arguments: \
             found {:?} in argument {} which has type {:?}",
            fr, argument_index, self.universal_regions.unnormalized_input_tys[argument_index],
        );

        let region_name = self.synthesize_region_name(counter);

        let argument_local = Local::new(argument_index + 1);
        let argument_span = mir.local_decls[argument_local].source_info.span;
        diag.span_label(
            argument_span,
            format!("lifetime `{}` appears in this argument", region_name,),
        );

        Some(region_name)
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
        let upvar_index = self.universal_regions
            .defining_ty
            .upvar_tys(tcx)
            .position(|upvar_ty| {
                debug!(
                    "give_name_if_anonymous_region_appears_in_upvars: upvar_ty = {:?}",
                    upvar_ty,
                );
                tcx.any_free_region_meets(&upvar_ty, |r| r.to_region_vid() == fr)
            })?;

        debug!(
            "give_name_if_anonymous_region_appears_in_upvars: \
             found {:?} in upvar {} which has type {:?}",
            fr,
            upvar_index,
            self.universal_regions
                .defining_ty
                .upvar_tys(tcx)
                .nth(upvar_index),
        );

        let region_name = self.synthesize_region_name(counter);

        let upvar_hir_id = mir.upvar_decls[upvar_index].var_hir_id.assert_crate_local();
        let upvar_node_id = tcx.hir.hir_to_node_id(upvar_hir_id);
        let upvar_span = tcx.hir.span(upvar_node_id);
        let upvar_name = tcx.hir.name(upvar_node_id);
        diag.span_label(
            upvar_span,
            format!(
                "lifetime `{}` appears in the type of `{}`",
                region_name, upvar_name,
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
        let return_ty = self.universal_regions
            .unnormalized_output_ty;
        debug!("give_name_if_anonymous_region_appears_in_output: return_ty = {:?}", return_ty);
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
