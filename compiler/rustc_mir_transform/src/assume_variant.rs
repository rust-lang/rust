//! This pass provides the active variant fact to the downcast projection.
//! The fact can be used by other passes.
//! Even if we know the active variant is different from the variant of the downcast projection,
//! this plain downcast is not UB.
//! ```rust
//! #![feature(custom_mir, core_intrinsics)]
//!
//! use std::intrinsics::mir::*;
//!
//! enum Foo {
//!     A(u32),
//!     B(u32),
//! }
//!
//! #[custom_mir(dialect = "runtime")]
//! pub fn unknown(x: Foo) -> u32 {
//!     mir! {
//!         {
//!             RET = Field(Variant(x, 0), 0);
//!             RET = Field(Variant(x, 1), 0);
//!             Return()
//!         }
//!     }
//! }
//! ```

use std::borrow::Cow;

use rustc_abi::VariantIdx;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_middle::mir::basic_blocks::Predecessors;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::{BasicBlock, Body, Location, Operand, Place, TerminatorKind, *};
use rustc_middle::ty::{self, TyCtxt};

use crate::PassPolicy;
use crate::ssa::SsaLocals;

pub(super) struct AssumeVariant;

impl<'tcx> crate::MirPass<'tcx> for AssumeVariant {
    fn policy(&self, ctx: &crate::PassCtx<'_>) -> PassPolicy {
        PassPolicy::optional(ctx.mir_opt_level() >= 2)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let typing_env = body.typing_env(tcx);
        let ssa = SsaLocals::new(tcx, body, typing_env);
        let cfg_cache = body.basic_blocks.cfg_cache();
        let dominators = cfg_cache.dominators(&body.basic_blocks);
        let predecessors = cfg_cache.predecessors(&body.basic_blocks);
        let mut collector = AssumeVariantCollector::new(tcx, body, &ssa, dominators, predecessors);

        let reverse_postorder = body.basic_blocks.reverse_postorder().to_vec();
        for bb in reverse_postorder {
            let data = &body.basic_blocks[bb];
            collector.visit_basic_block_data(bb, data);
        }
        for (loc, ref replacements) in collector.pending_replacements {
            let mut apply = ApplyAssumeVariant { tcx, replacements };
            let block = &mut body.basic_blocks.as_mut_preserves_cfg()[loc.block];
            if loc.statement_index == block.statements.len() {
                apply.visit_terminator(block.terminator_mut(), loc);
            } else {
                let stmt = &mut block.statements[loc.statement_index];
                apply.visit_statement(stmt, loc);
            }
        }
    }
}

struct AssumeVariantCollector<'tcx, 'body, 'a> {
    tcx: TyCtxt<'tcx>,
    ssa: &'a SsaLocals,
    body: &'body Body<'tcx>,
    dominators: &'a Dominators<BasicBlock>,
    predecessors: &'a Predecessors,
    /// Known discrs at each location.
    known_discrs: FxHashMap<Place<'tcx>, Vec<(Location, u128)>>,
    pending_replacements: Vec<(Location, Vec<(Place<'tcx>, Vec<PlaceElem<'tcx>>)>)>,
}

impl<'tcx, 'body, 'a> AssumeVariantCollector<'tcx, 'body, 'a> {
    fn new(
        tcx: TyCtxt<'tcx>,
        body: &'body Body<'tcx>,
        ssa: &'a SsaLocals,
        dominators: &'a Dominators<BasicBlock>,
        predecessors: &'a Predecessors,
    ) -> Self {
        AssumeVariantCollector {
            tcx,
            ssa,
            body,
            dominators,
            predecessors,
            known_discrs: FxHashMap::default(),
            pending_replacements: Vec::new(),
        }
    }

    fn has_unique_predecessor(&self, bb: BasicBlock) -> bool {
        self.predecessors[bb].len() == 1
    }

    /// Inserts the known discr value at the location.
    fn insert_discr(&mut self, place: Place<'tcx>, location: Location, discr: u128) {
        assert!(self.has_stable_discr(place));
        self.known_discrs.entry(place).or_default().push((location, discr));
    }

    fn is_variant_known(
        &mut self,
        place_ref: PlaceRef<'tcx>,
        variant_idx: VariantIdx,
        location: Location,
    ) -> bool {
        let enum_place = place_ref.to_place(self.tcx);
        let ty = enum_place.ty(self.body, self.tcx).ty;
        let ty::Adt(enum_def, _) = ty.kind() else {
            return false;
        };
        if enum_def.variants().iter().count() == 1 {
            // FIXME: Should we also handle uninhabited?
            return true;
        }
        let discr = enum_def.discriminant_for_variant(self.tcx, variant_idx);
        let Some(known_discrs) = self.known_discrs.get(&enum_place) else {
            return false;
        };
        for &(known_loc, val) in known_discrs {
            if discr.val == val && known_loc.dominates(location, &self.dominators) {
                return true;
            }
        }
        false
    }

    fn has_stable_discr(&self, place: Place<'tcx>) -> bool {
        if !self.ssa.is_ssa(place.local) {
            return false;
        }
        if place.is_stable_offset() {
            return true;
        }
        if place.is_indirect_first_projection()
            && let Some(Mutability::Not) = self.body.local_decls[place.local].ty.ref_mutability()
            && self.body.local_kind(place.local) == LocalKind::Arg
        {
            return true;
        }
        false
    }

    fn find_enum_place(&self, discr: &Operand<'tcx>) -> Option<Place<'tcx>> {
        let local = discr.place()?.as_local()?;
        let enum_place = if let Some(def_loc) = self.ssa.assignment_location(local)
            && let Some(&(_, Rvalue::Discriminant(enum_place))) =
                self.body.basic_blocks[def_loc.block].statements[def_loc.statement_index]
                    .kind
                    .as_assign()
        {
            enum_place
        } else if let Some(def_block) = self.ssa.call_block(local)
            && let TerminatorKind::Call { destination, .. } =
                self.body.basic_blocks[def_block].terminator().kind
        {
            destination
        } else {
            return None;
        };
        if !self.has_stable_discr(enum_place) {
            return None;
        }
        Some(enum_place)
    }
}

impl<'tcx> Visitor<'tcx> for AssumeVariantCollector<'tcx, '_, '_> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if !matches!(context, PlaceContext::NonMutatingUse(..)) || !self.has_stable_discr(*place) {
            return;
        }
        let mut projection = Cow::Borrowed(place.projection.as_slice());
        for (i, (place_ref, elem)) in place.iter_projections().enumerate() {
            let PlaceElem::Downcast { name, variant_index, assume_variant: false } = elem else {
                continue;
            };
            if self.is_variant_known(place_ref, variant_index, location) {
                let projection_mut = projection.to_mut();
                projection_mut[i] =
                    PlaceElem::Downcast { name, variant_index, assume_variant: true };
            }
        }
        if let Cow::Owned(new_projection) = projection {
            let replacements = if let Some((loc, replacements)) =
                self.pending_replacements.last_mut()
                && *loc == location
            {
                replacements
            } else {
                &mut self.pending_replacements.push_mut((location, Vec::new())).1
            };
            replacements.push((*place, new_projection));
        }
        self.super_place(place, context, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        let TerminatorKind::SwitchInt { ref discr, ref targets } = terminator.kind else {
            return;
        };
        let Some(enum_place) = self.find_enum_place(discr) else {
            return;
        };
        let unique_targets = targets.unique_targets();
        for (val, target) in targets.iter() {
            if !unique_targets.contains(&target) {
                continue;
            }
            let successor = target.start_location();
            if self.has_unique_predecessor(successor.block) {
                assert_ne!(location.block, successor.block);
                self.insert_discr(enum_place, successor, val);
            }
        }
    }
}

struct ApplyAssumeVariant<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    replacements: &'a [(Place<'tcx>, Vec<PlaceElem<'tcx>>)],
}

impl<'tcx, 'a> MutVisitor<'tcx> for ApplyAssumeVariant<'tcx, 'a> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, _: Location) {
        if !matches!(context, PlaceContext::NonMutatingUse(..)) {
            return;
        }
        let Some(((target_place, new_projection), rest)) = self.replacements.split_first() else {
            return;
        };
        if target_place != place {
            return;
        }
        place.projection = self.tcx().mk_place_elems(&new_projection);
        self.replacements = rest;
    }
}
