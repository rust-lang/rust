use crate::borrow_check::nll::region_infer::RegionInferenceContext;
use crate::borrow_check::nll::ToRegionVid;
use crate::borrow_check::Upvar;
use rustc::mir::{Local, Body};
use rustc::ty::{RegionVid, TyCtxt};
use rustc_data_structures::indexed_vec::Idx;
use syntax::source_map::Span;
use syntax_pos::symbol::Symbol;

impl<'tcx> RegionInferenceContext<'tcx> {
    crate fn get_var_name_and_span_for_region(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        upvars: &[Upvar],
        fr: RegionVid,
    ) -> Option<(Option<Symbol>, Span)> {
        debug!("get_var_name_and_span_for_region(fr={:?})", fr);
        assert!(self.universal_regions.is_universal_region(fr));

        debug!("get_var_name_and_span_for_region: attempting upvar");
        self.get_upvar_index_for_region(tcx, fr)
            .map(|index| {
                let (name, span) =
                    self.get_upvar_name_and_span_for_region(tcx, upvars, index);
                (Some(name), span)
            })
            .or_else(|| {
                debug!("get_var_name_and_span_for_region: attempting argument");
                self.get_argument_index_for_region(tcx, fr)
                    .map(|index| self.get_argument_name_and_span_for_region(body, index))
            })
    }

    /// Search the upvars (if any) to find one that references fr. Return its index.
    crate fn get_upvar_index_for_region(&self, tcx: TyCtxt<'tcx>, fr: RegionVid) -> Option<usize> {
        let upvar_index = self
            .universal_regions
            .defining_ty
            .upvar_tys(tcx)
            .position(|upvar_ty| {
                debug!("get_upvar_index_for_region: upvar_ty={:?}", upvar_ty);
                tcx.any_free_region_meets(&upvar_ty, |r| {
                    let r = r.to_region_vid();
                    debug!("get_upvar_index_for_region: r={:?} fr={:?}", r, fr);
                    r == fr
                })
            })?;

        let upvar_ty = self
            .universal_regions
            .defining_ty
            .upvar_tys(tcx)
            .nth(upvar_index);

        debug!(
            "get_upvar_index_for_region: found {:?} in upvar {} which has type {:?}",
            fr, upvar_index, upvar_ty,
        );

        Some(upvar_index)
    }

    /// Given the index of an upvar, finds its name and the span from where it was
    /// declared.
    crate fn get_upvar_name_and_span_for_region(
        &self,
        tcx: TyCtxt<'tcx>,
        upvars: &[Upvar],
        upvar_index: usize,
    ) -> (Symbol, Span) {
        let upvar_hir_id = upvars[upvar_index].var_hir_id;
        debug!("get_upvar_name_and_span_for_region: upvar_hir_id={:?}", upvar_hir_id);

        let upvar_name = tcx.hir().name(upvar_hir_id);
        let upvar_span = tcx.hir().span(upvar_hir_id);
        debug!("get_upvar_name_and_span_for_region: upvar_name={:?} upvar_span={:?}",
               upvar_name, upvar_span);

        (upvar_name, upvar_span)
    }

    /// Search the argument types for one that references fr (which should be a free region).
    /// Returns Some(_) with the index of the input if one is found.
    ///
    /// N.B., in the case of a closure, the index is indexing into the signature as seen by the
    /// user - in particular, index 0 is not the implicit self parameter.
    crate fn get_argument_index_for_region(
        &self,
        tcx: TyCtxt<'tcx>,
        fr: RegionVid,
    ) -> Option<usize> {
        let implicit_inputs = self.universal_regions.defining_ty.implicit_inputs();
        let argument_index = self
            .universal_regions
            .unnormalized_input_tys
            .iter()
            .skip(implicit_inputs)
            .position(|arg_ty| {
                debug!(
                    "get_argument_index_for_region: arg_ty = {:?}",
                    arg_ty
                );
                tcx.any_free_region_meets(arg_ty, |r| r.to_region_vid() == fr)
            })?;

        debug!(
            "get_argument_index_for_region: found {:?} in argument {} which has type {:?}",
            fr, argument_index, self.universal_regions.unnormalized_input_tys[argument_index],
        );

        Some(argument_index)
    }

    /// Given the index of an argument, finds its name (if any) and the span from where it was
    /// declared.
    crate fn get_argument_name_and_span_for_region(
        &self,
        body: &Body<'tcx>,
        argument_index: usize,
    ) -> (Option<Symbol>, Span) {
        let implicit_inputs = self.universal_regions.defining_ty.implicit_inputs();
        let argument_local = Local::new(implicit_inputs + argument_index + 1);
        debug!("get_argument_name_and_span_for_region: argument_local={:?}", argument_local);

        let argument_name = body.local_decls[argument_local].name;
        let argument_span = body.local_decls[argument_local].source_info.span;
        debug!("get_argument_name_and_span_for_region: argument_name={:?} argument_span={:?}",
               argument_name, argument_span);

        (argument_name, argument_span)
    }
}
