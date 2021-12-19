use rustc_hir as hir;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::Node;
use rustc_middle::hir::map::Map;
use rustc_middle::middle::resolve_lifetime as rl;
use rustc_middle::ty::{self, Region, TyCtxt};

/// This function calls the `visit_ty` method for the parameters
/// corresponding to the anonymous regions. The `nested_visitor.found_type`
/// contains the anonymous type.
///
/// # Arguments
/// region - the anonymous region corresponding to the anon_anon conflict
/// br - the bound region corresponding to the above region which is of type `BrAnon(_)`
///
/// # Example
/// ```
/// fn foo(x: &mut Vec<&u8>, y: &u8)
///    { x.push(y); }
/// ```
/// The function returns the nested type corresponding to the anonymous region
/// for e.g., `&u8` and `Vec<&u8>`.
pub(crate) fn find_anon_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    region: Region<'tcx>,
    br: &ty::BoundRegionKind,
) -> Option<(&'tcx hir::Ty<'tcx>, &'tcx hir::FnDecl<'tcx>)> {
    if let Some(anon_reg) = tcx.is_suitable_region(region) {
        let hir_id = tcx.hir().local_def_id_to_hir_id(anon_reg.def_id);
        let fndecl = match tcx.hir().get(hir_id) {
            Node::Item(&hir::Item { kind: hir::ItemKind::Fn(ref m, ..), .. })
            | Node::TraitItem(&hir::TraitItem {
                kind: hir::TraitItemKind::Fn(ref m, ..), ..
            })
            | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Fn(ref m, ..), .. }) => {
                &m.decl
            }
            _ => return None,
        };

        fndecl
            .inputs
            .iter()
            .find_map(|arg| find_component_for_bound_region(tcx, arg, br))
            .map(|ty| (ty, &**fndecl))
    } else {
        None
    }
}

// This method creates a FindNestedTypeVisitor which returns the type corresponding
// to the anonymous region.
fn find_component_for_bound_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    arg: &'tcx hir::Ty<'tcx>,
    br: &ty::BoundRegionKind,
) -> Option<&'tcx hir::Ty<'tcx>> {
    let mut nested_visitor = FindNestedTypeVisitor {
        tcx,
        bound_region: *br,
        found_type: None,
        current_index: ty::INNERMOST,
    };
    nested_visitor.visit_ty(arg);
    nested_visitor.found_type
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
    // The bound_region corresponding to the Refree(freeregion)
    // associated with the anonymous region we are looking for.
    bound_region: ty::BoundRegionKind,
    // The type where the anonymous lifetime appears
    // for e.g., Vec<`&u8`> and <`&u8`>
    found_type: Option<&'tcx hir::Ty<'tcx>>,
    current_index: ty::DebruijnIndex,
}

impl<'tcx> Visitor<'tcx> for FindNestedTypeVisitor<'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_ty(&mut self, arg: &'tcx hir::Ty<'tcx>) {
        match arg.kind {
            hir::TyKind::BareFn(_) => {
                self.current_index.shift_in(1);
                intravisit::walk_ty(self, arg);
                self.current_index.shift_out(1);
                return;
            }

            hir::TyKind::TraitObject(bounds, ..) => {
                for bound in bounds {
                    self.current_index.shift_in(1);
                    self.visit_poly_trait_ref(bound, hir::TraitBoundModifier::None);
                    self.current_index.shift_out(1);
                }
            }

            hir::TyKind::Rptr(ref lifetime, _) => {
                // the lifetime of the TyRptr
                let hir_id = lifetime.hir_id;
                match (self.tcx.named_region(hir_id), self.bound_region) {
                    // Find the index of the anonymous region that was part of the
                    // error. We will then search the function parameters for a bound
                    // region at the right depth with the same index
                    (
                        Some(rl::Region::LateBoundAnon(debruijn_index, _, anon_index)),
                        ty::BrAnon(br_index),
                    ) => {
                        debug!(
                            "LateBoundAnon depth = {:?} anon_index = {:?} br_index={:?}",
                            debruijn_index, anon_index, br_index
                        );
                        if debruijn_index == self.current_index && anon_index == br_index {
                            self.found_type = Some(arg);
                            return; // we can stop visiting now
                        }
                    }

                    // Find the index of the named region that was part of the
                    // error. We will then search the function parameters for a bound
                    // region at the right depth with the same index
                    (Some(rl::Region::EarlyBound(_, id, _)), ty::BrNamed(def_id, _)) => {
                        debug!("EarlyBound id={:?} def_id={:?}", id, def_id);
                        if id == def_id {
                            self.found_type = Some(arg);
                            return; // we can stop visiting now
                        }
                    }

                    // Find the index of the named region that was part of the
                    // error. We will then search the function parameters for a bound
                    // region at the right depth with the same index
                    (
                        Some(rl::Region::LateBound(debruijn_index, _, id, _)),
                        ty::BrNamed(def_id, _),
                    ) => {
                        debug!(
                            "FindNestedTypeVisitor::visit_ty: LateBound depth = {:?}",
                            debruijn_index
                        );
                        debug!("LateBound id={:?} def_id={:?}", id, def_id);
                        if debruijn_index == self.current_index && id == def_id {
                            self.found_type = Some(arg);
                            return; // we can stop visiting now
                        }
                    }

                    (
                        Some(
                            rl::Region::Static
                            | rl::Region::Free(_, _)
                            | rl::Region::EarlyBound(_, _, _)
                            | rl::Region::LateBound(_, _, _, _)
                            | rl::Region::LateBoundAnon(_, _, _),
                        )
                        | None,
                        _,
                    ) => {
                        debug!("no arg found");
                    }
                }
            }
            // Checks if it is of type `hir::TyKind::Path` which corresponds to a struct.
            hir::TyKind::Path(_) => {
                let subvisitor = &mut TyPathVisitor {
                    tcx: self.tcx,
                    found_it: false,
                    bound_region: self.bound_region,
                    current_index: self.current_index,
                };
                intravisit::walk_ty(subvisitor, arg); // call walk_ty; as visit_ty is empty,
                // this will visit only outermost type
                if subvisitor.found_it {
                    self.found_type = Some(arg);
                }
            }
            _ => {}
        }
        // walk the embedded contents: e.g., if we are visiting `Vec<&Foo>`,
        // go on to visit `&Foo`
        intravisit::walk_ty(self, arg);
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
    found_it: bool,
    bound_region: ty::BoundRegionKind,
    current_index: ty::DebruijnIndex,
}

impl<'tcx> Visitor<'tcx> for TyPathVisitor<'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Map<'tcx>> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_lifetime(&mut self, lifetime: &hir::Lifetime) {
        match (self.tcx.named_region(lifetime.hir_id), self.bound_region) {
            // the lifetime of the TyPath!
            (
                Some(rl::Region::LateBoundAnon(debruijn_index, _, anon_index)),
                ty::BrAnon(br_index),
            ) => {
                if debruijn_index == self.current_index && anon_index == br_index {
                    self.found_it = true;
                    return;
                }
            }

            (Some(rl::Region::EarlyBound(_, id, _)), ty::BrNamed(def_id, _)) => {
                debug!("EarlyBound id={:?} def_id={:?}", id, def_id);
                if id == def_id {
                    self.found_it = true;
                    return; // we can stop visiting now
                }
            }

            (Some(rl::Region::LateBound(debruijn_index, _, id, _)), ty::BrNamed(def_id, _)) => {
                debug!("FindNestedTypeVisitor::visit_ty: LateBound depth = {:?}", debruijn_index,);
                debug!("id={:?}", id);
                debug!("def_id={:?}", def_id);
                if debruijn_index == self.current_index && id == def_id {
                    self.found_it = true;
                    return; // we can stop visiting now
                }
            }

            (
                Some(
                    rl::Region::Static
                    | rl::Region::EarlyBound(_, _, _)
                    | rl::Region::LateBound(_, _, _, _)
                    | rl::Region::LateBoundAnon(_, _, _)
                    | rl::Region::Free(_, _),
                )
                | None,
                _,
            ) => {
                debug!("no arg found");
            }
        }
    }

    fn visit_ty(&mut self, arg: &'tcx hir::Ty<'tcx>) {
        // ignore nested types
        //
        // If you have a type like `Foo<'a, &Ty>` we
        // are only interested in the immediate lifetimes ('a).
        //
        // Making `visit_ty` empty will ignore the `&Ty` embedded
        // inside, it will get reached by the outer visitor.
        debug!("`Ty` corresponding to a struct is {:?}", arg);
    }
}
