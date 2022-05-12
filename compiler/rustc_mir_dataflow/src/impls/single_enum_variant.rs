use super::*;
//use crate::lattice::PackedU8JoinSemiLattice as Fact;

//use crate::lattice::FactArray;
use crate::lattice::FactCache;
use rustc_target::abi::VariantIdx;

use rustc_middle::mir::*;

use crate::{Analysis, AnalysisDomain};

/// A dataflow analysis that tracks whether an enum can hold 0, 1, or more than one variants.
pub struct SingleEnumVariant<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
}

impl<'tcx> AnalysisDomain<'tcx> for SingleEnumVariant<'_, 'tcx> {
    /// For each local, keep track of which enum index it is, if its uninhabited, or unknown.
    //type Domain = FactArray<Fact, 128>;
    type Domain = FactCache<Local, Location, VariantIdx, 16>;

    const NAME: &'static str = "single_enum_variant";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        //FactArray { arr: [Fact::TOP; 128] }
        FactCache::new(Local::from_u32(0), Location::START, VariantIdx::MAX)
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut Self::Domain) {
        // assume everything is top initially.
        let local_decls = body.local_decls();
        for (l, _) in local_decls.iter_enumerated() {
            state.remove(l);
            //state.insert(l, Fact::TOP);
            //state[l] = Fact::TOP;
        }
    }
}

impl<'tcx> SingleEnumVariant<'_, 'tcx> {
    pub fn new<'a>(tcx: TyCtxt<'tcx>, body: &'a mir::Body<'tcx>) -> SingleEnumVariant<'a, 'tcx> {
        SingleEnumVariant { tcx, body }
    }
    #[inline]
    pub fn is_tracked(&self, place: &Place<'tcx>) -> bool {
        place.ty(self.body, self.tcx).ty.is_enum()
    }
    fn assign(
        &self,
        state: &mut <Self as AnalysisDomain<'tcx>>::Domain,
        lhs: &Place<'tcx>,
        rhs: &Operand<'tcx>,
        location: Location,
    ) {
        let _: Option<_> = try {
            if !self.is_tracked(lhs) {
                return;
            }
            let lhs_local = lhs.local_or_deref_local()?;

            let new_fact = match rhs {
                Operand::Copy(rhs) | Operand::Move(rhs) => {
                    if let Some(rhs_local) = rhs.local_or_deref_local() {
                        state.get(rhs_local).map(|f| f.1).copied()
                    } else {
                        rhs.ty(self.body, self.tcx).variant_index.map(|var_idx| var_idx)
                    }
                }
                // Assigning a constant does not affect discriminant?
                Operand::Constant(_c) => return,
            };
            if let Some(new_fact) = new_fact {
                state.insert(lhs_local, location, new_fact);
            } else {
                state.remove(lhs_local);
            }
        };
    }
}

impl<'tcx> Analysis<'tcx> for SingleEnumVariant<'_, 'tcx> {
    fn apply_statement_effect(
        &self,
        state: &mut Self::Domain,
        statement: &Statement<'tcx>,
        loc: Location,
    ) {
        let (place, fact) = match &statement.kind {
            StatementKind::Deinit(box place) => (place, None),
            StatementKind::SetDiscriminant { box place, variant_index } => {
                (place, Some(*variant_index))
            }
            StatementKind::Assign(box (lhs, Rvalue::Use(op))) => {
                return self.assign(state, lhs, op, loc);
            }
            /* may alias/mutate RHS need to specify that it is no longer a single value */
            StatementKind::Assign(box (
                _,
                Rvalue::Ref(_, BorrowKind::Mut { .. }, rhs)
                | Rvalue::AddressOf(Mutability::Mut, rhs),
            )) => (rhs, None),
            StatementKind::CopyNonOverlapping(box ref copy) => {
                let place = match &copy.dst {
                    Operand::Copy(p) | Operand::Move(p) => p,
                    _ => return,
                };
                (place, None)
            }
            _ => return,
        };
        if !self.is_tracked(place) {
            return;
        }
        let Some(local) = place.local_or_deref_local() else { return };
        if let Some(fact) = fact {
            state.insert(local, loc, fact);
        } else {
            state.remove(local);
        }
    }
    fn apply_terminator_effect(
        &self,
        state: &mut Self::Domain,
        terminator: &Terminator<'tcx>,
        loc: Location,
    ) {
        match &terminator.kind {
            TerminatorKind::DropAndReplace { place, value, .. } => {
                self.assign(state, place, value, loc)
            }
            TerminatorKind::Drop { place, .. } if self.is_tracked(place) => {
                let Some(local) = place.local_or_deref_local() else { return };
                state.remove(local);
            }
            _ => {}
        }
    }

    fn apply_call_return_effect(
        &self,
        _: &mut Self::Domain,
        _: BasicBlock,
        _: CallReturnPlaces<'_, 'tcx>,
    ) {
    }

    fn apply_switch_int_edge_effects(
        &self,
        _block: BasicBlock,
        discr: &Operand<'tcx>,
        apply_edge_effects: &mut impl SwitchIntEdgeEffects<Self::Domain>,
    ) {
        let Some(place) = discr.place() else { return };
        if !self.is_tracked(&place) {
            return;
        }
        let Some(local) = place.local_or_deref_local() else { return };
        apply_edge_effects.apply(|state, target| {
            // This probably isn't right, need to check that it fits.
            let new_fact = target.value.map(|v| VariantIdx::from_u32(v as u32));

            if let Some(new_fact) = new_fact {
                let loc = Location { block: target.target, statement_index: 0 };
                state.insert(local, loc, new_fact);
            } else {
                state.remove(local);
            }
        });
    }
}
