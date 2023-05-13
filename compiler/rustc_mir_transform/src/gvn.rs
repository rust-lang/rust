//! Global value numbering.
//!
//! MIR may contain repeated and/or redundant computations. The objective of this pass is to detect
//! such redundancies and re-use the already-computed result when possible.
//!
//! In a first pass, we compute a symbolic representation of values that are assigned to SSA
//! locals. This symbolic representation is defined by the `Value` enum. Each produced instance of
//! `Value` is interned as a `VnIndex`, which allows us to cheaply compute identical values.
//!
//! From those assignments, we construct a mapping `VnIndex -> Vec<(Local, Location)>` of available
//! values, the locals in which they are stored, and a the assignment location.
//!
//! In a second pass, we traverse all (non SSA) assignments `x = rvalue` and operands. For each
//! one, we compute the `VnIndex` of the rvalue. If this `VnIndex` is associated to a constant, we
//! replace the rvalue/operand by that constant. Otherwise, if there is an SSA local `y`
//! associated to this `VnIndex`, and if its definition location strictly dominates the assignment
//! to `x`, we replace the assignment by `x = y`.
//!
//! By opportunity, this pass simplifies some `Rvalue`s based on the accumulated knowledge.
//!
//! # Operational semantic
//!
//! Operationally, this pass attempts to prove bitwise equality between locals. Given this MIR:
//! ```ignore (MIR)
//! _a = some value // has VnIndex i
//! // some MIR
//! _b = some other value // also has VnIndex i
//! ```
//!
//! We consider it to be replacable by:
//! ```ignore (MIR)
//! _a = some value // has VnIndex i
//! // some MIR
//! _c = some other value // also has VnIndex i
//! assume(_a bitwise equal to _c) // follows from having the same VnIndex
//! _b = _a // follows from the `assume`
//! ```
//!
//! Which is simplifiable to:
//! ```ignore (MIR)
//! _a = some value // has VnIndex i
//! // some MIR
//! _b = _a
//! ```
//!
//! # Handling of references
//!
//! We handle references by assigning a different "provenance" index to each Ref/AddressOf rvalue.
//! This ensure that we do not spuriously merge borrows that should not be merged. Meanwhile, we
//! consider all the derefs of an immutable reference to a freeze type to give the same value:
//! ```ignore (MIR)
//! _a = *_b // _b is &Freeze
//! _c = *_b // replaced by _c = _a
//! ```

use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_macros::newtype_index;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_target::abi::{VariantIdx, FIRST_VARIANT};

use crate::ssa::SsaLocals;
use crate::MirPass;

pub struct GVN;

impl<'tcx> MirPass<'tcx> for GVN {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 4
    }

    #[instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());
        propagate_ssa(tcx, body);
    }
}

fn propagate_ssa<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
    let ssa = SsaLocals::new(body);
    // Clone dominators as we need them while mutating the body.
    let dominators = body.basic_blocks.dominators().clone();

    let mut state = VnState::new(tcx, param_env, &body.local_decls);
    for arg in body.args_iter() {
        if ssa.is_ssa(arg) {
            let value = state.new_opaque().unwrap();
            state.assign(arg, value);
        }
    }

    for (local, rvalue, _) in ssa.assignments(body) {
        let value = state.insert_rvalue(rvalue).or_else(|| state.new_opaque()).unwrap();
        state.assign(local, value);
    }

    // Stop creating opaques during replacement as it is useless.
    state.next_opaque = None;

    let mut any_replacement = false;
    let mut replacer = Replacer {
        tcx,
        ssa,
        dominators,
        state,
        reused_locals: BitSet::new_empty(body.local_decls.len()),
        any_replacement: &mut any_replacement,
    };

    let reverse_postorder = body.basic_blocks.reverse_postorder().to_vec();
    for bb in reverse_postorder {
        let data = &mut body.basic_blocks.as_mut_preserves_cfg()[bb];
        replacer.visit_basic_block_data(bb, data);
    }

    // For each local that is reused (`y` above), we remove its storage statements do avoid any
    // difficulty. Those locals are SSA, so should be easy to optimize by LLVM without storage
    // statements.
    StorageRemover { tcx, reused_locals: replacer.reused_locals }.visit_body_preserves_cfg(body);

    if any_replacement {
        crate::simplify::remove_unused_definitions(body);
    }
}

newtype_index! {
    struct VnIndex {}
}

#[derive(Debug, PartialEq, Eq, Hash)]
enum Value<'tcx> {
    // Root values.
    /// Used to represent values we know nothing about.
    /// The `usize` is a counter incremented by `new_opaque`.
    Opaque(usize),
    /// Evaluated or unevaluated constant value.
    Constant(Const<'tcx>),
    /// An aggregate value, either tuple/closure/struct/enum.
    /// This does not contain unions, as we cannot reason with the value.
    Aggregate(Ty<'tcx>, VariantIdx, Vec<VnIndex>),
    /// This corresponds to a `[value; count]` expression.
    Repeat(VnIndex, ty::Const<'tcx>),
    /// The address of a place.
    Address {
        place: Place<'tcx>,
        /// Give each borrow and pointer a different provenance, so we don't merge them.
        provenance: usize,
    },

    // Extractions.
    /// This is the *value* obtained by projecting another value.
    Projection(VnIndex, ProjectionElem<VnIndex, Ty<'tcx>>),
    /// Discriminant of the given value.
    Discriminant(VnIndex),
    /// Length of an array or slice.
    Len(VnIndex),

    // Operations.
    NullaryOp(NullOp<'tcx>, Ty<'tcx>),
    UnaryOp(UnOp, VnIndex),
    BinaryOp(BinOp, VnIndex, VnIndex),
    CheckedBinaryOp(BinOp, VnIndex, VnIndex),
    Cast {
        kind: CastKind,
        value: VnIndex,
        from: Ty<'tcx>,
        to: Ty<'tcx>,
    },
}

struct VnState<'body, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    local_decls: &'body LocalDecls<'tcx>,
    /// Value stored in each local.
    locals: IndexVec<Local, Option<VnIndex>>,
    /// First local to be assigned that value.
    rev_locals: FxHashMap<VnIndex, Vec<Local>>,
    values: FxIndexSet<Value<'tcx>>,
    /// Counter to generate different values.
    /// This is an option to stop creating opaques during replacement.
    next_opaque: Option<usize>,
}

impl<'body, 'tcx> VnState<'body, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        local_decls: &'body LocalDecls<'tcx>,
    ) -> Self {
        VnState {
            tcx,
            param_env,
            local_decls,
            locals: IndexVec::from_elem(None, local_decls),
            rev_locals: FxHashMap::default(),
            values: FxIndexSet::default(),
            next_opaque: Some(0),
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn insert(&mut self, value: Value<'tcx>) -> VnIndex {
        let (index, _) = self.values.insert_full(value);
        VnIndex::from_usize(index)
    }

    /// Create a new `Value` for which we have no information at all, except that it is distinct
    /// from all the others.
    #[instrument(level = "trace", skip(self), ret)]
    fn new_opaque(&mut self) -> Option<VnIndex> {
        let next_opaque = self.next_opaque.as_mut()?;
        let value = Value::Opaque(*next_opaque);
        *next_opaque += 1;
        Some(self.insert(value))
    }

    /// Create a new `Value::Address` distinct from all the others.
    #[instrument(level = "trace", skip(self), ret)]
    fn new_pointer(&mut self, place: Place<'tcx>) -> Option<VnIndex> {
        let next_opaque = self.next_opaque.as_mut()?;
        let value = Value::Address { place, provenance: *next_opaque };
        *next_opaque += 1;
        Some(self.insert(value))
    }

    fn get(&self, index: VnIndex) -> &Value<'tcx> {
        self.values.get_index(index.as_usize()).unwrap()
    }

    /// Record that `local` is assigned `value`. `local` must be SSA.
    #[instrument(level = "trace", skip(self))]
    fn assign(&mut self, local: Local, value: VnIndex) {
        self.locals[local] = Some(value);

        // Only register the value if its type is `Sized`, as we will emit copies of it.
        let is_sized = !self.tcx.features().unsized_locals
            || self.local_decls[local].ty.is_sized(self.tcx, self.param_env);
        if is_sized {
            self.rev_locals.entry(value).or_default().push(local);
        }
    }

    /// Represent the *value* which would be read from `place`.
    #[instrument(level = "trace", skip(self), ret)]
    fn insert_place(&mut self, place: Place<'tcx>) -> Option<VnIndex> {
        let mut value = self.locals[place.local]?;

        for (index, proj) in place.projection.iter().enumerate() {
            let proj = match proj {
                ProjectionElem::Deref => {
                    let ty = Place::ty_from(
                        place.local,
                        &place.projection[..index],
                        self.local_decls,
                        self.tcx,
                    )
                    .ty;
                    if let Some(Mutability::Not) = ty.ref_mutability()
                        && let Some(pointee_ty) = ty.builtin_deref(true)
                        && pointee_ty.ty.is_freeze(self.tcx, self.param_env)
                    {
                        // An immutable borrow `_x` always points to the same value for the
                        // lifetime of the borrow, so we can merge all instances of `*_x`.
                        ProjectionElem::Deref
                    } else {
                        return None;
                    }
                }
                ProjectionElem::Field(f, ty) => ProjectionElem::Field(f, ty),
                ProjectionElem::Index(idx) => {
                    let idx = self.locals[idx]?;
                    ProjectionElem::Index(idx)
                }
                ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                    ProjectionElem::ConstantIndex { offset, min_length, from_end }
                }
                ProjectionElem::Subslice { from, to, from_end } => {
                    ProjectionElem::Subslice { from, to, from_end }
                }
                ProjectionElem::Downcast(name, index) => ProjectionElem::Downcast(name, index),
                ProjectionElem::OpaqueCast(ty) => ProjectionElem::OpaqueCast(ty),
            };
            value = self.insert(Value::Projection(value, proj));
        }

        Some(value)
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn insert_operand(&mut self, operand: &Operand<'tcx>) -> Option<VnIndex> {
        match *operand {
            Operand::Constant(ref constant) => Some(self.insert(Value::Constant(constant.const_))),
            Operand::Copy(place) | Operand::Move(place) => self.insert_place(place),
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn insert_rvalue(&mut self, rvalue: &Rvalue<'tcx>) -> Option<VnIndex> {
        let value = match *rvalue {
            // Forward values.
            Rvalue::Use(ref operand) => return self.insert_operand(operand),
            Rvalue::CopyForDeref(place) => return self.insert_operand(&Operand::Copy(place)),

            // Roots.
            Rvalue::Repeat(ref op, amount) => {
                let op = self.insert_operand(op)?;
                Value::Repeat(op, amount)
            }
            Rvalue::NullaryOp(op, ty) => Value::NullaryOp(op, ty),
            Rvalue::Aggregate(box ref kind, ref fields) => {
                let variant_index = match *kind {
                    AggregateKind::Array(..)
                    | AggregateKind::Tuple
                    | AggregateKind::Closure(..)
                    | AggregateKind::Generator(..) => FIRST_VARIANT,
                    AggregateKind::Adt(_, variant_index, _, _, None) => variant_index,
                    // Do not track unions.
                    AggregateKind::Adt(_, _, _, _, Some(_)) => return None,
                };
                let fields: Option<Vec<_>> = fields
                    .iter()
                    .map(|op| self.insert_operand(op).or_else(|| self.new_opaque()))
                    .collect();
                let ty = rvalue.ty(self.local_decls, self.tcx);
                Value::Aggregate(ty, variant_index, fields?)
            }
            Rvalue::Ref(.., place) | Rvalue::AddressOf(_, place) => return self.new_pointer(place),

            // Operations.
            Rvalue::Len(place) => {
                let place = self.insert_place(place)?;
                Value::Len(place)
            }
            Rvalue::Cast(kind, ref value, to) => {
                let from = value.ty(self.local_decls, self.tcx);
                let value = self.insert_operand(value)?;
                Value::Cast { kind, value, from, to }
            }
            Rvalue::BinaryOp(op, box (ref lhs, ref rhs)) => {
                let lhs = self.insert_operand(lhs)?;
                let rhs = self.insert_operand(rhs)?;
                Value::BinaryOp(op, lhs, rhs)
            }
            Rvalue::CheckedBinaryOp(op, box (ref lhs, ref rhs)) => {
                let lhs = self.insert_operand(lhs)?;
                let rhs = self.insert_operand(rhs)?;
                Value::CheckedBinaryOp(op, lhs, rhs)
            }
            Rvalue::UnaryOp(op, ref arg) => {
                let arg = self.insert_operand(arg)?;
                Value::UnaryOp(op, arg)
            }
            Rvalue::Discriminant(place) => {
                let place = self.insert_place(place)?;
                Value::Discriminant(place)
            }

            // Unsupported values.
            Rvalue::ThreadLocalRef(..) | Rvalue::ShallowInitBox(..) => return None,
        };
        debug!(?value);
        Some(self.insert(value))
    }
}

struct Replacer<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    ssa: SsaLocals,
    dominators: Dominators<BasicBlock>,
    state: VnState<'a, 'tcx>,
    /// Set of locals that are reused, and for which we should remove storage statements to avoid a
    /// use-after-StorageDead.
    reused_locals: BitSet<Local>,
    any_replacement: &'a mut bool,
}

impl<'tcx> Replacer<'_, 'tcx> {
    /// If `index` is a `Value::Constant`, return the `Constant` to be put in the MIR.
    fn try_as_constant(&mut self, index: VnIndex) -> Option<ConstOperand<'tcx>> {
        if let Value::Constant(const_) = self.state.get(index) {
            Some(ConstOperand { span: rustc_span::DUMMY_SP, user_ty: None, const_: const_.clone() })
        } else {
            None
        }
    }

    /// If there is a local which is assigned `index`, and its assignment strictly dominates `loc`,
    /// return it.
    fn try_as_local(&mut self, index: VnIndex, loc: Location) -> Option<Local> {
        let other = self.state.rev_locals.get(&index)?;
        other
            .iter()
            .copied()
            .find(|&other| self.ssa.assignment_dominates(&self.dominators, other, loc))
    }
}

impl<'tcx> MutVisitor<'tcx> for Replacer<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        if let Some(place) = operand.place()
            && let Some(value) = self.state.insert_place(place)
        {
            if let Some(const_) = self.try_as_constant(value) {
                *operand = Operand::Constant(Box::new(const_));
                *self.any_replacement = true;
            } else if let Some(local) = self.try_as_local(value, location)
                && *operand != Operand::Move(local.into())
            {
                *operand = Operand::Copy(local.into());
                self.reused_locals.insert(local);
                *self.any_replacement = true;
            }
        }
    }

    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, location: Location) {
        self.super_statement(stmt, location);
        if let StatementKind::Assign(box (_, ref mut rvalue)) = stmt.kind
            && let Some(value) = self.state.insert_rvalue(rvalue)
        {
            if let Some(const_) = self.try_as_constant(value) {
                *rvalue = Rvalue::Use(Operand::Constant(Box::new(const_)));
                *self.any_replacement = true;
            } else if let Some(local) = self.try_as_local(value, location)
                && *rvalue != Rvalue::Use(Operand::Move(local.into()))
            {
                *rvalue = Rvalue::Use(Operand::Copy(local.into()));
                self.reused_locals.insert(local);
                *self.any_replacement = true;
            }
        }
    }
}

struct StorageRemover<'tcx> {
    tcx: TyCtxt<'tcx>,
    reused_locals: BitSet<Local>,
}

impl<'tcx> MutVisitor<'tcx> for StorageRemover<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, _: Location) {
        if let Operand::Move(place) = *operand
            && let Some(local) = place.as_local()
            && self.reused_locals.contains(local)
        {
            *operand = Operand::Copy(place);
        }
    }

    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, loc: Location) {
        match stmt.kind {
            // When removing storage statements, we need to remove both (#107511).
            StatementKind::StorageLive(l) | StatementKind::StorageDead(l)
                if self.reused_locals.contains(l) =>
            {
                stmt.make_nop()
            }
            _ => self.super_statement(stmt, loc),
        }
    }
}
