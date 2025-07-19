use core::ops::ControlFlow;

use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor, VisitorExt};
use rustc_hir::{self as hir, AmbigArg};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::resolve_bound_vars as rbv;
use rustc_middle::ty::{self, Region, TyCtxt};
use tracing::debug;

/// This function calls the `visit_ty` method for the parameters
/// corresponding to the anonymous regions. The `nested_visitor.found_type`
/// contains the anonymous type.
///
/// # Arguments
/// region - the anonymous region corresponding to the anon_anon conflict
/// br - the bound region corresponding to the above region which is of type `BrAnon(_)`
///
/// # Example
/// ```compile_fail
/// fn foo(x: &mut Vec<&u8>, y: &u8)
///    { x.push(y); }
/// ```
/// The function returns the nested type corresponding to the anonymous region
/// for e.g., `&u8` and `Vec<&u8>`.
pub fn find_anon_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    generic_param_scope: LocalDefId,
    region: Region<'tcx>,
) -> Option<(&'tcx hir::Ty<'tcx>, &'tcx hir::FnSig<'tcx>)> {
    let anon_reg = tcx.is_suitable_region(generic_param_scope, region)?;
    let fn_sig = tcx.hir_node_by_def_id(anon_reg.scope).fn_sig()?;

    fn_sig
        .decl
        .inputs
        .iter()
        .find_map(|arg| find_component_for_bound_region(tcx, arg, anon_reg.region_def_id))
        .map(|ty| (ty, fn_sig))
}

// This method creates a FindNestedTypeVisitor which returns the type corresponding
// to the anonymous region.
fn find_component_for_bound_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    arg: &'tcx hir::Ty<'tcx>,
    region_def_id: DefId,
) -> Option<&'tcx hir::Ty<'tcx>> {
    FindNestedTypeVisitor { tcx, region_def_id, current_index: ty::INNERMOST }
        .visit_ty_unambig(arg)
        .break_value()
}

// The FindNestedTypeVisitor captures the corresponding `hir::Ty` of the
// anonymous region. The example above would lead to a conflict between
// the two anonymous lifetimes for &u8 in x and y respectively. This visitor
// would be invoked twice, once for each lifetime, and would
// walk the types like &mut Vec<&u8> and &u8 looking for the HIR
// where that lifetime appears. This allows us to highlight the
// specific part of the type in the error message.
struct FindNestedTypeVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    // The `DefId` of the region we're looking for.
    region_def_id: DefId,
    current_index: ty::DebruijnIndex,
}

impl<'tcx> Visitor<'tcx> for FindNestedTypeVisitor<'tcx> {
    type Result = ControlFlow<&'tcx hir::Ty<'tcx>>;
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_ty(&mut self, arg: &'tcx hir::Ty<'tcx, AmbigArg>) -> Self::Result {
        match arg.kind {
            hir::TyKind::FnPtr(_) => {
                self.current_index.shift_in(1);
                let _ = intravisit::walk_ty(self, arg);
                self.current_index.shift_out(1);
                return ControlFlow::Continue(());
            }

            hir::TyKind::TraitObject(bounds, ..) => {
                for bound in bounds {
                    self.current_index.shift_in(1);
                    let _ = self.visit_poly_trait_ref(bound);
                    self.current_index.shift_out(1);
                }
            }

            hir::TyKind::Ref(lifetime, _) => {
                // the lifetime of the Ref
                let hir_id = lifetime.hir_id;
                match self.tcx.named_bound_var(hir_id) {
                    // Find the index of the named region that was part of the
                    // error. We will then search the function parameters for a bound
                    // region at the right depth with the same index
                    Some(rbv::ResolvedArg::EarlyBound(id)) => {
                        debug!("EarlyBound id={:?}", id);
                        if id.to_def_id() == self.region_def_id {
                            return ControlFlow::Break(arg.as_unambig_ty());
                        }
                    }

                    // Find the index of the named region that was part of the
                    // error. We will then search the function parameters for a bound
                    // region at the right depth with the same index
                    Some(rbv::ResolvedArg::LateBound(debruijn_index, _, id)) => {
                        debug!(
                            "FindNestedTypeVisitor::visit_ty: LateBound depth = {:?}",
                            debruijn_index
                        );
                        debug!("LateBound id={:?}", id);
                        if debruijn_index == self.current_index
                            && id.to_def_id() == self.region_def_id
                        {
                            return ControlFlow::Break(arg.as_unambig_ty());
                        }
                    }

                    Some(
                        rbv::ResolvedArg::StaticLifetime
                        | rbv::ResolvedArg::Free(_, _)
                        | rbv::ResolvedArg::Error(_),
                    )
                    | None => {
                        debug!("no arg found");
                    }
                }
            }
            // Checks if it is of type `hir::TyKind::Path` which corresponds to a struct.
            hir::TyKind::Path(_) => {
                // Prefer using the lifetime in type arguments rather than lifetime arguments.
                intravisit::walk_ty(self, arg)?;

                // Call `walk_ty` as `visit_ty` is empty.
                return if intravisit::walk_ty(
                    &mut TyPathVisitor {
                        tcx: self.tcx,
                        region_def_id: self.region_def_id,
                        current_index: self.current_index,
                    },
                    arg,
                )
                .is_break()
                {
                    ControlFlow::Break(arg.as_unambig_ty())
                } else {
                    ControlFlow::Continue(())
                };
            }
            _ => {}
        }
        // walk the embedded contents: e.g., if we are visiting `Vec<&Foo>`,
        // go on to visit `&Foo`
        intravisit::walk_ty(self, arg)
    }
}

// The visitor captures the corresponding `hir::Ty` of the anonymous region
// in the case of structs ie. `hir::TyKind::Path`.
// This visitor would be invoked for each lifetime corresponding to a struct,
// and would walk the types like Vec<Ref> in the above example and Ref looking for the HIR
// where that lifetime appears. This allows us to highlight the
// specific part of the type in the error message.
struct TyPathVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    region_def_id: DefId,
    current_index: ty::DebruijnIndex,
}

impl<'tcx> Visitor<'tcx> for TyPathVisitor<'tcx> {
    type Result = ControlFlow<()>;
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_lifetime(&mut self, lifetime: &hir::Lifetime) -> Self::Result {
        match self.tcx.named_bound_var(lifetime.hir_id) {
            // the lifetime of the TyPath!
            Some(rbv::ResolvedArg::EarlyBound(id)) => {
                debug!("EarlyBound id={:?}", id);
                if id.to_def_id() == self.region_def_id {
                    return ControlFlow::Break(());
                }
            }

            Some(rbv::ResolvedArg::LateBound(debruijn_index, _, id)) => {
                debug!("FindNestedTypeVisitor::visit_ty: LateBound depth = {:?}", debruijn_index,);
                debug!("id={:?}", id);
                if debruijn_index == self.current_index && id.to_def_id() == self.region_def_id {
                    return ControlFlow::Break(());
                }
            }

            Some(
                rbv::ResolvedArg::StaticLifetime
                | rbv::ResolvedArg::Free(_, _)
                | rbv::ResolvedArg::Error(_),
            )
            | None => {
                debug!("no arg found");
            }
        }
        ControlFlow::Continue(())
    }

    fn visit_ty(&mut self, arg: &'tcx hir::Ty<'tcx, AmbigArg>) -> Self::Result {
        // ignore nested types
        //
        // If you have a type like `Foo<'a, &Ty>` we
        // are only interested in the immediate lifetimes ('a).
        //
        // Making `visit_ty` empty will ignore the `&Ty` embedded
        // inside, it will get reached by the outer visitor.
        debug!("`Ty` corresponding to a struct is {:?}", arg);
        ControlFlow::Continue(())
    }
}
