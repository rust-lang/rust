//! Performs local value numbering.
//!
//! Local value numbering is a powerful analysis that subsumes constant and copy propagation.  Since
//! it can reason about immutability of memory, it can do optimizations that LLVM won't. Because
//! it's local to a basic block, however, it doesn't handle all the cases that our constant and copy
//! propagation passes do. Eventually, when we have SSA, we should be able to upgrade this to global
//! value numbering.

use crate::simplify;
use crate::ty::{Region, Ty};
use crate::MirPass;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BasicBlockData, BinOp, Body, BorrowKind, CastKind, Constant,
    InlineAsmOperand, Local, LocalDecls, Location, Mutability, NullOp, Operand, Place, PlaceElem,
    ProjectionElem, Rvalue, SourceInfo, Statement, StatementKind, TerminatorKind, UnOp,
};
use rustc_middle::ty::{self, Const, ParamEnv, TyCtxt};
use rustc_span::def_id::DefId;
use std::collections::hash_map::Entry;
use std::mem;

/// Performs copy and constant propagation via value numbering.
pub struct ValueNumbering;

impl<'tcx> MirPass<'tcx> for ValueNumbering {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        // We run on MIR opt level 2 or greater, if `-Z number-values` is passed.
        //
        // FIXME(pcwalton): Enable by default.
        sess.mir_opt_level() >= 2 && sess.opts.debugging_opts.number_values
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // First, promote mutable locals to immutable ones. This helps our rudimentary alias
        // analysis a lot.
        //
        // FIXME(pcwalton): Should the pass manager be doing this?
        simplify::promote_mutable_locals_to_immutable(body, tcx);

        // Now loop over and optimize all basic blocks.
        let param_env = tcx.param_env(body.source.def_id());
        let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();
        for block_index in 0..basic_blocks.len() {
            let block_index = BasicBlock::from_usize(block_index);
            if basic_blocks[block_index].statements.is_empty() {
                // This is necessary to avoid indexing underflows in `update_storage_dead_markers`,
                // and it's probably faster anyway.
                continue;
            }

            let mut numberer = ValueNumberer::new(tcx, param_env, block_index);
            numberer.init_locals(block_index, &mut basic_blocks[block_index]);
            numberer.process_statements(&mut basic_blocks[block_index], local_decls);
            numberer.process_terminator(&mut basic_blocks[block_index]);
            numberer.update_storage_dead_markers(block_index, basic_blocks);
        }

        // Finally, prune dead locals, as value numbering frequently makes dead stores.
        //
        // FIXME(pcwalton): As above, should the pass manager be doing this?
        simplify::remove_unused_locals(body, tcx);
    }
}

struct ValueNumberer<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,

    // A mapping from computed rvalues to the place they live.
    //
    // When we encounter an already-computed rvalue, we replace it with a reference to the
    // corresponding operand if it's safe to do so.
    values: FxHashMap<VnRvalue<'tcx>, VnOperand<'tcx>>,

    // Stores information about each local, most notably its current live range.
    locals: FxHashMap<Local, VnLocalInfo>,

    // The location of the statement we're processing.
    location: Location,
}

// Generally identical to Rvalue in MIR, except it uses SSA locals instead of MIR locals.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
enum VnRvalue<'tcx> {
    Use(VnOperand<'tcx>),
    Repeat(VnOperand<'tcx>, Const<'tcx>),
    Ref(Region<'tcx>, BorrowKind, VnPlace<'tcx>),
    ThreadLocalRef(DefId),
    AddressOf(Mutability, VnPlace<'tcx>),
    Len(VnPlace<'tcx>),
    Cast(CastKind, VnOperand<'tcx>, Ty<'tcx>),
    BinaryOp(BinOp, Box<(VnOperand<'tcx>, VnOperand<'tcx>)>),
    CheckedBinaryOp(BinOp, Box<(VnOperand<'tcx>, VnOperand<'tcx>)>),
    NullaryOp(NullOp, Ty<'tcx>),
    UnaryOp(UnOp, VnOperand<'tcx>),
    Discriminant(VnPlace<'tcx>),
    Aggregate(Box<AggregateKind<'tcx>>, Vec<VnOperand<'tcx>>),
    ShallowInitBox(VnOperand<'tcx>, Ty<'tcx>),
}

// Generally identical to Operand in MIR, except it uses SSA locals instead of MIR locals.
//
// FIXME(pcwalton): Add support for moves.
#[derive(Clone, PartialEq, Debug, Hash)]
enum VnOperand<'tcx> {
    Copy(VnPlace<'tcx>),
    Constant(Box<Constant<'tcx>>),
}

impl<'tcx> Eq for VnOperand<'tcx> {}

// Generally identical to Place in MIR, except it uses SSA locals instead of MIR locals.
#[derive(Clone, PartialEq, Debug, Hash)]
struct VnPlace<'tcx> {
    local: VnLocal,
    projection: Vec<VnPlaceElem<'tcx>>,
}

impl<'tcx> Eq for VnPlace<'tcx> {}

// Generally identical to PlaceElem in MIR, except it uses SSA locals instead of MIR locals.
type VnPlaceElem<'tcx> = ProjectionElem<VnLocal, Ty<'tcx>>;

// An local in mostly-SSA form.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
struct VnLocal {
    // The index of the MIR local.
    local: Local,
    // This starts at 0 and is bumped whenever we detect a possible mutation to the local.
    live_range_index: usize,
}

// Information that we track about each local.
#[derive(Default)]
struct VnLocalInfo {
    // The index of the last statement that potentially mutated this local.
    statement_index_of_last_kill: usize,
    // This starts at 0 and is bumped whenever we detect a possible mutation to the local.
    current_live_range_index: usize,
}

impl<'tcx> ValueNumberer<'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        block_index: BasicBlock,
    ) -> ValueNumberer<'tcx> {
        ValueNumberer {
            tcx,
            param_env,
            values: FxHashMap::default(),
            locals: FxHashMap::default(),
            location: Location { block: block_index, statement_index: 0 },
        }
    }

    // Gathers up all locals and initializes VnLocalInfo for them.
    fn init_locals(&mut self, block_index: BasicBlock, block: &mut BasicBlockData<'tcx>) {
        let mut visitor = LocalInitVisitor { locals: FxHashMap::default() };
        visitor.visit_basic_block_data(block_index, block);
        self.locals = visitor.locals;

        struct LocalInitVisitor {
            locals: FxHashMap<Local, VnLocalInfo>,
        }

        impl<'tcx> Visitor<'tcx> for LocalInitVisitor {
            fn visit_local(&mut self, local: &Local, _: PlaceContext, _: Location) {
                if let Entry::Vacant(entry) = self.locals.entry((*local).clone()) {
                    entry.insert(VnLocalInfo {
                        statement_index_of_last_kill: 0,
                        current_live_range_index: 0,
                    });
                }
            }
        }
    }

    fn process_statements(
        &mut self,
        block: &mut BasicBlockData<'tcx>,
        local_decls: &LocalDecls<'tcx>,
    ) {
        for (statement_index, statement) in block.statements.iter_mut().enumerate() {
            self.location.statement_index = statement_index;

            match statement.kind {
                StatementKind::Assign(box (ref place, ref mut rvalue)) => {
                    self.process_assignment(place, rvalue, &statement.source_info, local_decls);
                }
                _ => {
                    // Non-assignment statements are only analyzed for live range kills.
                    VnAliasAnalysis::new(local_decls, &mut self.locals)
                        .visit_statement(statement, self.location)
                }
            }
        }
    }

    fn process_terminator(&mut self, block: &mut BasicBlockData<'tcx>) {
        self.location.statement_index = block.statements.len();

        let terminator = match block.terminator {
            None => return,
            Some(ref mut terminator) => terminator,
        };

        // Certain terminator statements, the most important of which are calls, are eligible for
        // value numbering optimizations.
        match terminator.kind {
            TerminatorKind::SwitchInt { ref mut discr, .. } => {
                self.substitute_values_in_operand(discr);
            }
            TerminatorKind::Call { ref mut func, ref mut args, .. } => {
                self.substitute_values_in_operand(func);
                for arg in args {
                    self.substitute_values_in_operand(arg);
                }
            }
            TerminatorKind::Assert { ref mut cond, .. } => {
                self.substitute_values_in_operand(cond);
            }
            TerminatorKind::Yield { ref mut value, .. } => {
                self.substitute_values_in_operand(value);
            }
            TerminatorKind::DropAndReplace { ref mut value, .. } => {
                self.substitute_values_in_operand(value);
            }
            TerminatorKind::InlineAsm { ref mut operands, .. } => {
                for operand in operands {
                    match *operand {
                        InlineAsmOperand::In { value: ref mut operand, .. }
                        | InlineAsmOperand::InOut { in_value: ref mut operand, .. } => {
                            self.substitute_values_in_operand(operand);
                        }
                        InlineAsmOperand::Out { .. }
                        | InlineAsmOperand::Const { .. }
                        | InlineAsmOperand::SymFn { .. }
                        | InlineAsmOperand::SymStatic { .. } => {}
                    }
                }
            }
            TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {}
        }
    }

    // Pushes down all StorageDead markers as far as they need to go.
    //
    // This is a necessary pass because we may extend ranges of various Locals as we perform copy
    // propagation.
    fn update_storage_dead_markers(
        &mut self,
        block_index: BasicBlock,
        basic_blocks: &mut IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    ) {
        // First, find all StorageDead statements in this basic block.
        let block = &mut basic_blocks[block_index];
        let mut visitor = MarkerFinderVisitor {
            storage_dead_statements: FxHashMap::default(),
            storage_dead_statements_to_insert: vec![],
        };
        for statement in block.statements.iter_mut() {
            if let StatementKind::StorageDead(local) = statement.kind {
                visitor.storage_dead_statements.insert(local, (*statement).clone());
            }
        }

        // Push all the StorageDeads we found right up to where they need to go. Start with the
        // terminator, pushing all new StorageDeads into all successor blocks (duplicating them if
        // necessary).
        let mut successors = vec![];
        if let Some(ref mut terminator) = block.terminator {
            visitor.visit_terminator(
                terminator,
                Location { block: block_index, statement_index: block.statements.len() },
            );
            successors.extend(terminator.successors().cloned());
        }
        for successor_index in successors {
            let mut new_statements = visitor.storage_dead_statements_to_insert.clone();
            let mut successor = &mut basic_blocks[successor_index];
            new_statements.extend(successor.statements.drain(..));
            successor.statements = new_statements;
        }
        visitor.storage_dead_statements_to_insert.clear();

        // Now process the statements in reverse order, building up a new statement list as we do.
        let block = &mut basic_blocks[block_index];
        for (src_statement_index, src_statement) in
            mem::replace(&mut block.statements, vec![]).into_iter().enumerate().rev()
        {
            if let StatementKind::StorageDead(_) = src_statement.kind {
                continue;
            }
            visitor.visit_statement(
                &src_statement,
                Location { block: block_index, statement_index: src_statement_index },
            );
            for storage_dead_statement in visitor.storage_dead_statements_to_insert.drain(..) {
                block.statements.push(storage_dead_statement);
            }
            block.statements.push(src_statement);
        }

        // Push any missing StorageDeads.
        for (_, storage_dead_statement) in visitor.storage_dead_statements.into_iter() {
            block.statements.push(storage_dead_statement);
        }

        // Since we processed the statements in reverse order, reverse them to produce the correct
        // ordering.
        block.statements.reverse();

        // Searches for StorageDead statements.
        struct MarkerFinderVisitor<'tcx> {
            storage_dead_statements: FxHashMap<Local, Statement<'tcx>>,
            storage_dead_statements_to_insert: Vec<Statement<'tcx>>,
        }

        impl<'tcx> Visitor<'tcx> for MarkerFinderVisitor<'tcx> {
            fn visit_local(&mut self, local: &Local, _: PlaceContext, _: Location) {
                if let Some(storage_dead_statement) = self.storage_dead_statements.remove(local) {
                    self.storage_dead_statements_to_insert.push(storage_dead_statement);
                }
            }
        }
    }

    fn process_assignment(
        &mut self,
        place: &Place<'tcx>,
        rvalue: &mut Rvalue<'tcx>,
        source_info: &SourceInfo,
        local_decls: &LocalDecls<'tcx>,
    ) {
        // Perform subsitutions in the rvalue.
        let mut vn_rvalue = self.vn_rvalue_for(rvalue);
        if let Some(ref mut vn_rvalue) = vn_rvalue {
            if self.substitute_values_in_vn_rvalue(vn_rvalue) {
                *rvalue = vn_rvalue.to_rvalue(self.tcx);
            }
        }

        // Kill possibly-aliasing values in `place`.
        VnAliasAnalysis::new(local_decls, &mut self.locals).visit_place(
            place,
            PlaceContext::MutatingUse(MutatingUseContext::Store),
            self.location,
        );

        // If the rvalue involves locals we don't support yet, bail.
        let vn_rvalue = match vn_rvalue {
            None => return,
            Some(vn_rvalue) => vn_rvalue,
        };

        // FIXME(pcwalton): Support a subset of projections!
        if !place.projection.is_empty() {
            return;
        }
        // FIXME(pcwalton): Support non-Copy types!
        if !local_decls[place.local]
            .ty
            .is_copy_modulo_regions(self.tcx.at(source_info.span), self.param_env)
        {
            return;
        }

        // Update the value table.
        let vn_local = self
            .vn_local_for(place.local)
            .expect("We should have created a local info for this local by now!");
        match vn_rvalue {
            VnRvalue::Use(ref target @ VnOperand::Copy(ref place @ VnPlace { .. }))
                if self.vn_place_is_immutable(place, local_decls) =>
            {
                debug!("adding use rvalue {:?} -> {:?}", vn_local, target);
                self.values.insert(VnRvalue::use_of_local(&vn_local), (*target).clone());
            }
            VnRvalue::Use(ref target @ VnOperand::Constant(_)) => {
                debug!("adding use rvalue {:?} -> {:?}", vn_local, target);
                self.values.insert(VnRvalue::use_of_local(&vn_local), (*target).clone());
            }
            _ => {
                if let Entry::Vacant(entry) = self.values.entry(vn_rvalue.clone()) {
                    debug!("adding general rvalue {:?} -> {:?}", vn_rvalue, vn_local);
                    entry.insert(VnOperand::copy_of_local(&vn_local));
                }
            }
        }
    }

    // Converts a Local to the corresponding VnLocal. The location of the local is assumed to
    // be `self.location`.
    fn vn_local_for(&self, local: Local) -> Option<VnLocal> {
        self.locals.get(&local).map(|local_info| VnLocal {
            local,
            live_range_index: local_info.current_live_range_index,
        })
    }

    // Converts an Operand to the corresponding VnOperand. The location of the operand is
    // assumed to be `self.location`.
    fn vn_operand_for(&self, operand: &Operand<'tcx>) -> Option<VnOperand<'tcx>> {
        match *operand {
            Operand::Copy(ref place) | Operand::Move(ref place) => {
                self.vn_place_for(place).map(VnOperand::Copy)
            }
            Operand::Constant(ref constant) => Some(VnOperand::Constant((*constant).clone())),
        }
    }

    // Converts a Place to the corresponding VnPlace. The location of the place is assumed to
    // be `self.location`.
    fn vn_place_for(&self, place: &Place<'tcx>) -> Option<VnPlace<'tcx>> {
        let local = self.vn_local_for(place.local)?;
        let mut projection = vec![];
        for place_elem in place.projection.iter() {
            projection.push(self.vn_place_elem_for(&place_elem)?);
        }
        Some(VnPlace { local, projection })
    }

    // Converts a PlaceElem to the corresponding VnPlaceElem. The location of the place element is
    // assumed to be `self.location`.
    fn vn_place_elem_for(&self, place_elem: &PlaceElem<'tcx>) -> Option<VnPlaceElem<'tcx>> {
        match *place_elem {
            ProjectionElem::Deref => Some(ProjectionElem::Deref),
            ProjectionElem::Field(field, ty) => Some(ProjectionElem::Field(field, ty)),
            ProjectionElem::Index(local) => self.vn_local_for(local).map(ProjectionElem::Index),
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                Some(ProjectionElem::ConstantIndex { offset, min_length, from_end })
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                Some(ProjectionElem::Subslice { from, to, from_end })
            }
            ProjectionElem::Downcast(symbol, variant) => {
                Some(ProjectionElem::Downcast(symbol, variant))
            }
        }
    }

    // Converts an Rvalue to the corresponding VnRvalue. The location of the rvalue is assumed to be
    // `self.location`.
    fn vn_rvalue_for(&self, rvalue: &Rvalue<'tcx>) -> Option<VnRvalue<'tcx>> {
        match *rvalue {
            Rvalue::Use(ref operand) => self.vn_operand_for(operand).map(VnRvalue::Use),
            Rvalue::Repeat(ref operand, constant) => {
                self.vn_operand_for(operand).map(|operand| VnRvalue::Repeat(operand, constant))
            }
            Rvalue::Ref(region, borrow_kind, ref place) => {
                self.vn_place_for(place).map(|place| VnRvalue::Ref(region, borrow_kind, place))
            }
            Rvalue::ThreadLocalRef(def_id) => Some(VnRvalue::ThreadLocalRef(def_id)),
            Rvalue::AddressOf(mutability, ref place) => {
                self.vn_place_for(place).map(|place| VnRvalue::AddressOf(mutability, place))
            }
            Rvalue::Len(ref place) => self.vn_place_for(place).map(VnRvalue::Len),
            Rvalue::Cast(kind, ref operand, ty) => {
                self.vn_operand_for(operand).map(|operand| VnRvalue::Cast(kind, operand, ty))
            }
            Rvalue::BinaryOp(binop, box (ref lhs, ref rhs)) => {
                let lhs = self.vn_operand_for(lhs)?;
                let rhs = self.vn_operand_for(rhs)?;
                Some(VnRvalue::BinaryOp(binop, Box::new((lhs, rhs))))
            }
            Rvalue::CheckedBinaryOp(binop, box (ref lhs, ref rhs)) => {
                let lhs = self.vn_operand_for(lhs)?;
                let rhs = self.vn_operand_for(rhs)?;
                Some(VnRvalue::CheckedBinaryOp(binop, Box::new((lhs, rhs))))
            }
            Rvalue::NullaryOp(nullop, ty) => Some(VnRvalue::NullaryOp(nullop, ty)),
            Rvalue::UnaryOp(unop, ref operand) => {
                self.vn_operand_for(operand).map(|operand| VnRvalue::UnaryOp(unop, operand))
            }
            Rvalue::Discriminant(ref place) => self.vn_place_for(place).map(VnRvalue::Discriminant),
            Rvalue::Aggregate(ref aggregate_kind, ref operands) => {
                let mut vn_operands = vec![];
                for operand in operands {
                    vn_operands.push(self.vn_operand_for(operand)?);
                }
                Some(VnRvalue::Aggregate((*aggregate_kind).clone(), vn_operands))
            }
            Rvalue::ShallowInitBox(ref operand, ty) => {
                self.vn_operand_for(operand).map(|operand| VnRvalue::ShallowInitBox(operand, ty))
            }
        }
    }

    // `*_may_be_materialized_here`
    //
    // These functions are called when we want to perform a replacement of some rvalue component A
    // with B. We need to make sure that B can be validly materialized at this location.
    //
    // A VN local may not be materialized at a statement if it has been killed (in the dataflow
    // sense). For example:
    //
    //      let mut x = 1;
    //      let mut x_ref = &mut x;
    //      let z = x + 2;
    //      *x_ref = 3;
    //      let w = x + 2;
    //
    // In this case, the `x_ref = 3` statement will kill `x` and we need to make sure `z` and
    // `w` get different values. How much this limits our optimization potential depends on how
    // good our alias information is. (Right now it's not very good.)

    fn local_may_be_materialized_here(&self, vn_local: &VnLocal) -> bool {
        self.locals[&vn_local.local].current_live_range_index == vn_local.live_range_index
    }

    fn rvalue_may_be_materialized_here(&self, vn_rvalue: &VnRvalue<'tcx>) -> bool {
        match *vn_rvalue {
            VnRvalue::Use(ref operand)
            | VnRvalue::Repeat(ref operand, _)
            | VnRvalue::Cast(_, ref operand, _)
            | VnRvalue::UnaryOp(_, ref operand)
            | VnRvalue::ShallowInitBox(ref operand, _) => {
                self.operand_may_be_materialized_here(operand)
            }
            VnRvalue::Ref(_, _, ref place)
            | VnRvalue::AddressOf(_, ref place)
            | VnRvalue::Len(ref place)
            | VnRvalue::Discriminant(ref place) => self.place_may_be_materialized_here(place),
            VnRvalue::BinaryOp(_, box (ref lhs, ref rhs))
            | VnRvalue::CheckedBinaryOp(_, box (ref lhs, ref rhs)) => {
                self.operand_may_be_materialized_here(lhs)
                    && self.operand_may_be_materialized_here(rhs)
            }
            VnRvalue::Aggregate(_, ref operands) => {
                operands.iter().all(|operand| self.operand_may_be_materialized_here(operand))
            }
            VnRvalue::NullaryOp(_, _) | VnRvalue::ThreadLocalRef(_) => true,
        }
    }

    fn operand_may_be_materialized_here(&self, vn_operand: &VnOperand<'tcx>) -> bool {
        match *vn_operand {
            VnOperand::Copy(ref place) => self.place_may_be_materialized_here(place),
            VnOperand::Constant(_) => true,
        }
    }

    fn place_may_be_materialized_here(&self, vn_place: &VnPlace<'tcx>) -> bool {
        self.local_may_be_materialized_here(&vn_place.local)
            && vn_place
                .projection
                .iter()
                .all(|place_elem| self.place_elem_may_be_materialized_here(place_elem))
    }

    fn place_elem_may_be_materialized_here(&self, vn_place_elem: &VnPlaceElem<'tcx>) -> bool {
        match *vn_place_elem {
            ProjectionElem::Deref
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Downcast(_, _)
            | ProjectionElem::Field(_, _) => true,
            ProjectionElem::Index(ref local) => self.local_may_be_materialized_here(local),
        }
    }

    // `substitute_values_in_*`
    //
    // These functions perform the actual copy/constant propagation optimizations. Whenever we see
    // a value that the program has already computed in the current basic block, we replace that
    // value with a copy of that value, if it's safe to do so.

    fn substitute_values_in_vn_rvalue(&self, vn_rvalue: &mut VnRvalue<'tcx>) -> bool {
        let mut modified = false;
        match *vn_rvalue {
            VnRvalue::Use(ref mut operand)
            | VnRvalue::Repeat(ref mut operand, _)
            | VnRvalue::Cast(_, ref mut operand, _)
            | VnRvalue::UnaryOp(_, ref mut operand)
            | VnRvalue::ShallowInitBox(ref mut operand, _) => {
                modified = self.substitute_values_in_vn_operand(operand) || modified;
            }
            VnRvalue::Len(ref mut place) | VnRvalue::Discriminant(ref mut place) => {
                modified = self.substitute_values_in_vn_place(place) || modified;
            }
            VnRvalue::BinaryOp(_, box (ref mut lhs, ref mut rhs))
            | VnRvalue::CheckedBinaryOp(_, box (ref mut lhs, ref mut rhs)) => {
                modified = self.substitute_values_in_vn_operand(lhs) || modified;
                modified = self.substitute_values_in_vn_operand(rhs) || modified;
            }
            VnRvalue::Aggregate(_, ref mut operands) => {
                for operand in operands.iter_mut() {
                    modified = self.substitute_values_in_vn_operand(operand) || modified
                }
            }
            // FIXME(pcwalton): We should be able to do better with `Ref`/`AddressOf`, for example
            // `x + y` is eligible for value numbering in `&a[x + y]`. But for now, ignore them
            // because we have no idea whether the address is significant; e.g. we definitely don't
            // want to make `let a = 1; let b = 1; (&a) as usize == (&b) as usize` true!
            VnRvalue::ThreadLocalRef(_)
            | VnRvalue::NullaryOp(_, _)
            | VnRvalue::Ref(_, _, _)
            | VnRvalue::AddressOf(_, _) => {}
        }

        if let Some(vn_operand) = self.values.get(&vn_rvalue) {
            if self.operand_may_be_materialized_here(vn_operand)
                && self.rvalue_may_be_materialized_here(&vn_rvalue)
            {
                // FIXME(pcwalton): Move around `StorageLive` and `StorageDead`!
                *vn_rvalue = VnRvalue::Use((*vn_operand).clone());
                modified = true;
            }
        }

        modified
    }

    fn substitute_values_in_vn_place(&self, vn_place: &mut VnPlace<'tcx>) -> bool {
        let mut modified = false;
        if let Some(vn_operand) = self.values.get(&VnRvalue::use_of_local(&vn_place.local)) {
            match *vn_operand {
                VnOperand::Copy(ref new_place) => {
                    if self.place_may_be_materialized_here(new_place) {
                        let original_projections = mem::replace(&mut vn_place.projection, vec![]);
                        *vn_place = (*new_place).clone();
                        vn_place.projection.extend(original_projections.into_iter());
                        modified = true;
                    }
                }
                VnOperand::Constant(_) => {}
            }
        }

        for vn_place_elem in vn_place.projection.iter_mut() {
            modified = self.substitute_values_in_vn_place_elem(vn_place_elem) || modified;
        }
        modified
    }

    fn substitute_values_in_vn_place_elem(&self, vn_place_elem: &mut VnPlaceElem<'tcx>) -> bool {
        match *vn_place_elem {
            ProjectionElem::Index(ref mut orig_vn_local) => {
                if let Some(vn_operand) = self.values.get(&VnRvalue::use_of_local(orig_vn_local)) {
                    if let Some(vn_local) = vn_operand.get_single_local() {
                        if self.local_may_be_materialized_here(&vn_local) {
                            *orig_vn_local = vn_local;
                            return true;
                        }
                    }
                }
            }
            ProjectionElem::Deref
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Downcast(_, _)
            | ProjectionElem::Field(_, _) => {}
        }
        false
    }

    fn substitute_values_in_vn_operand(&self, vn_operand: &mut VnOperand<'tcx>) -> bool {
        let mut modified = false;
        if let Some(vn_new_operand) = self.values.get(&VnRvalue::Use((*vn_operand).clone())) {
            if self.operand_may_be_materialized_here(vn_new_operand) {
                *vn_operand = (*vn_new_operand).clone();
                modified = true;
            }
        }
        modified
    }

    // Directly performs copy/constant propagation optimizations in the given MIR Operand.
    fn substitute_values_in_operand(&self, operand: &mut Operand<'tcx>) -> bool {
        if let Some(mut vn_operand) = self.vn_operand_for(operand) {
            if self.substitute_values_in_vn_operand(&mut vn_operand) {
                *operand = vn_operand.to_operand(self.tcx);
                return true;
            }
        }
        false
    }

    // Returns true if the given VnPlace describes an immutable location. Immutable locations are
    // guaranteed to be immutable for the life of the place.
    fn vn_place_is_immutable(&self, place: &VnPlace<'tcx>, local_decls: &LocalDecls<'tcx>) -> bool {
        debug!("vn_place_is_immutable({:?})", place);
        let local_decl = &local_decls[place.local.local];
        if let Mutability::Mut = local_decl.mutability {
            return false;
        }

        let mut ty_kind = local_decl.ty.kind();
        for place_elem in &place.projection {
            match *place_elem {
                ProjectionElem::Deref => match ty_kind {
                    ty::Ref(_, inner_ty, Mutability::Not) => {
                        ty_kind = inner_ty.kind();
                    }
                    _ => return false,
                },
                ProjectionElem::Field(_, _) => {
                    match ty_kind {
                        ty::Adt(ref adt, _)
                            if Some(adt.did()) != self.tcx.lang_items().unsafe_cell_type() =>
                        {
                            // The field of an `UnsafeCell` is considered mutable.
                        }
                        _ => return false,
                    }
                }
                // FIXME(pcwalton): Handle some of these.
                ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Downcast(_, _)
                | ProjectionElem::Index(_)
                | ProjectionElem::Subslice { .. } => return false,
            }
        }

        true
    }
}

// A very conservative alias analysis that detects when locals might have been mutated by a
// statement.
//
// FIXME(pcwalton): Improve precision.
struct VnAliasAnalysis<'a, 'vn, 'tcx> {
    local_decls: &'a LocalDecls<'tcx>,
    locals: &'vn mut FxHashMap<Local, VnLocalInfo>,
}

impl<'a, 'vn, 'tcx> VnAliasAnalysis<'a, 'vn, 'tcx> {
    fn new(
        local_decls: &'a LocalDecls<'tcx>,
        locals: &'vn mut FxHashMap<Local, VnLocalInfo>,
    ) -> Self {
        Self { local_decls, locals }
    }
}

impl<'a, 'vn, 'tcx> Visitor<'tcx> for VnAliasAnalysis<'a, 'vn, 'tcx> {
    fn visit_place(
        &mut self,
        place: &Place<'tcx>,
        place_context: PlaceContext,
        location: Location,
    ) {
        match place_context {
            PlaceContext::NonUse(_) | PlaceContext::NonMutatingUse(_) => return,
            PlaceContext::MutatingUse(_) => {}
        }

        let mut local_info = self.locals.get_mut(&place.local).unwrap();
        local_info.statement_index_of_last_kill = location.statement_index;
        local_info.current_live_range_index += 1;

        // Kill every mutable local if we see any projection (most importantly, a dereference).
        // This includes mutating through references, etc.
        //
        // FIXME(pcwalton): This is super conservative.
        if !place.projection.is_empty() {
            for (local, local_info) in self.locals.iter_mut() {
                if *local != place.local && self.local_decls[*local].mutability == Mutability::Mut {
                    local_info.statement_index_of_last_kill = location.statement_index;
                    local_info.current_live_range_index += 1;
                }
            }
        }
    }
}

impl<'tcx> VnRvalue<'tcx> {
    // Creates a VnRvalue that simply describes a copy of the given VnLocal.
    fn use_of_local(vn_local: &VnLocal) -> VnRvalue<'tcx> {
        // FIXME(pcwalton): Copy or move as appropriate, once we have the ability to handle moves!
        VnRvalue::Use(VnOperand::copy_of_local(vn_local))
    }

    // Converts this VnRvalue back to a MIR Rvalue.
    //
    // It is the caller's responsibility to ensure that all local references are valid in the live
    // range where this rvalue is used.
    fn to_rvalue(&self, tcx: TyCtxt<'tcx>) -> Rvalue<'tcx> {
        match *self {
            VnRvalue::Use(ref operand) => Rvalue::Use(operand.to_operand(tcx)),
            VnRvalue::Repeat(ref operand, constant) => {
                Rvalue::Repeat(operand.to_operand(tcx), constant)
            }
            VnRvalue::Ref(region, borrow_kind, ref place) => {
                Rvalue::Ref(region, borrow_kind, place.to_place(tcx))
            }
            VnRvalue::ThreadLocalRef(def_id) => Rvalue::ThreadLocalRef(def_id),
            VnRvalue::AddressOf(mutability, ref place) => {
                Rvalue::AddressOf(mutability, place.to_place(tcx))
            }
            VnRvalue::Len(ref place) => Rvalue::Len(place.to_place(tcx)),
            VnRvalue::Cast(kind, ref operand, ty) => {
                Rvalue::Cast(kind, operand.to_operand(tcx), ty)
            }
            VnRvalue::BinaryOp(binop, box (ref lhs, ref rhs)) => {
                Rvalue::BinaryOp(binop, Box::new((lhs.to_operand(tcx), rhs.to_operand(tcx))))
            }
            VnRvalue::CheckedBinaryOp(binop, box (ref lhs, ref rhs)) => {
                Rvalue::CheckedBinaryOp(binop, Box::new((lhs.to_operand(tcx), rhs.to_operand(tcx))))
            }
            VnRvalue::NullaryOp(nullop, ty) => Rvalue::NullaryOp(nullop, ty),
            VnRvalue::UnaryOp(unop, ref operand) => Rvalue::UnaryOp(unop, operand.to_operand(tcx)),
            VnRvalue::Discriminant(ref place) => Rvalue::Discriminant(place.to_place(tcx)),
            VnRvalue::Aggregate(ref kind, ref vn_operands) => {
                let operands: Vec<_> =
                    vn_operands.iter().map(|operand| operand.to_operand(tcx)).collect();
                Rvalue::Aggregate((*kind).clone(), operands)
            }
            VnRvalue::ShallowInitBox(ref operand, ty) => {
                Rvalue::ShallowInitBox(operand.to_operand(tcx), ty)
            }
        }
    }
}

impl<'tcx> VnOperand<'tcx> {
    // Creates a VnOperand that simply describes a copy of the given VnLocal.
    fn copy_of_local(vn_local: &VnLocal) -> VnOperand<'tcx> {
        VnOperand::Copy(VnPlace { local: *vn_local, projection: vec![] })
    }

    // If this VnOperand simply describes a copy of the given VnLocal, returns that VnLocal.
    fn get_single_local(&self) -> Option<VnLocal> {
        match *self {
            VnOperand::Copy(VnPlace { local, ref projection }) if projection.is_empty() => {
                Some(local)
            }
            _ => None,
        }
    }

    // Converts this VnOperand back to a MIR Operand.
    //
    // It is the caller's responsibility to ensure that all local references are valid in the live
    // range where this operand is used.
    fn to_operand(&self, tcx: TyCtxt<'tcx>) -> Operand<'tcx> {
        match *self {
            VnOperand::Copy(ref place) => Operand::Copy(place.to_place(tcx)),
            VnOperand::Constant(ref constant) => Operand::Constant((*constant).clone()),
        }
    }
}

impl<'tcx> VnPlace<'tcx> {
    // Converts this VnPlace back to a MIR Place.
    //
    // It is the caller's responsibility to ensure that all local references are valid in the live
    // range where this place is used.
    fn to_place(&self, tcx: TyCtxt<'tcx>) -> Place<'tcx> {
        let projection: Vec<_> =
            self.projection.iter().map(|place_elem| place_elem.to_place_elem()).collect();
        Place { local: self.local.to_local(), projection: tcx.intern_place_elems(&projection) }
    }
}

trait VnPlaceElemExt {
    type PlaceElem;
    fn to_place_elem(&self) -> Self::PlaceElem;
}

impl<'tcx> VnPlaceElemExt for VnPlaceElem<'tcx> {
    type PlaceElem = PlaceElem<'tcx>;

    // Converts this VnPlaceElem back to a MIR PlaceElem.
    //
    // It is the caller's responsibility to ensure that all local references are valid in the live
    // range where this place element is used.
    fn to_place_elem(&self) -> PlaceElem<'tcx> {
        match *self {
            ProjectionElem::Deref => ProjectionElem::Deref,
            ProjectionElem::Field(field, ty) => ProjectionElem::Field(field, ty),
            ProjectionElem::Index(value) => ProjectionElem::Index(value.to_local()),
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                ProjectionElem::ConstantIndex { offset, min_length, from_end }
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                ProjectionElem::Subslice { from, to, from_end }
            }
            ProjectionElem::Downcast(symbol, variant_index) => {
                ProjectionElem::Downcast(symbol, variant_index)
            }
        }
    }
}

impl VnLocal {
    // Converts this VnLocal back to a MIR Local.
    //
    // It is the caller's responsibility to ensure that all local references are valid in the live
    // range where this local is used.
    fn to_local(&self) -> Local {
        self.local
    }
}
