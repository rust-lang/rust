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

use crate::ssa::{AssignedValue, SsaLocals};
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

    let mut state = VnState::new(tcx, param_env, &ssa, &dominators, &body.local_decls);
    ssa.for_each_assignment_mut(
        body.basic_blocks.as_mut_preserves_cfg(),
        |local, value, location| {
            let value = match value {
                // We do not know anything of this assigned value.
                AssignedValue::Arg | AssignedValue::Terminator(_) => None,
                // Try to get some insight.
                AssignedValue::Rvalue(rvalue) => {
                    let value = state.simplify_rvalue(rvalue, location);
                    // FIXME(#112651) `rvalue` may have a subtype to `local`. We can only mark `local` as
                    // reusable if we have an exact type match.
                    if state.local_decls[local].ty != rvalue.ty(state.local_decls, tcx) {
                        return;
                    }
                    value
                }
            };
            // `next_opaque` is `Some`, so `new_opaque` must return `Some`.
            let value = value.or_else(|| state.new_opaque()).unwrap();
            state.assign(local, value);
        },
    );

    // Stop creating opaques during replacement as it is useless.
    state.next_opaque = None;

    let reverse_postorder = body.basic_blocks.reverse_postorder().to_vec();
    for bb in reverse_postorder {
        let data = &mut body.basic_blocks.as_mut_preserves_cfg()[bb];
        state.visit_basic_block_data(bb, data);
    }
    let any_replacement = state.any_replacement;

    // For each local that is reused (`y` above), we remove its storage statements do avoid any
    // difficulty. Those locals are SSA, so should be easy to optimize by LLVM without storage
    // statements.
    StorageRemover { tcx, reused_locals: state.reused_locals }.visit_body_preserves_cfg(body);

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
    ssa: &'body SsaLocals,
    dominators: &'body Dominators<BasicBlock>,
    reused_locals: BitSet<Local>,
    any_replacement: bool,
}

impl<'body, 'tcx> VnState<'body, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ssa: &'body SsaLocals,
        dominators: &'body Dominators<BasicBlock>,
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
            ssa,
            dominators,
            reused_locals: BitSet::new_empty(local_decls.len()),
            any_replacement: false,
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

    /// Represent the *value* which would be read from `place`, and point `place` to a preexisting
    /// place with the same value (if that already exists).
    #[instrument(level = "trace", skip(self), ret)]
    fn simplify_place_value(
        &mut self,
        place: &mut Place<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        // Invariant: `place` and `place_ref` point to the same value, even if they point to
        // different memory locations.
        let mut place_ref = place.as_ref();

        // Invariant: `value` holds the value up-to the `index`th projection excluded.
        let mut value = self.locals[place.local]?;
        for (index, proj) in place.projection.iter().enumerate() {
            if let Some(local) = self.try_as_local(value, location) {
                // Both `local` and `Place { local: place.local, projection: projection[..index] }`
                // hold the same value. Therefore, following place holds the value in the original
                // `place`.
                place_ref = PlaceRef { local, projection: &place.projection[index..] };
            }

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
                ProjectionElem::Subtype(ty) => ProjectionElem::Subtype(ty),
            };
            value = self.insert(Value::Projection(value, proj));
        }

        if let Some(local) = self.try_as_local(value, location)
            && local != place.local
        // in case we had no projection to begin with.
        {
            *place = local.into();
            self.reused_locals.insert(local);
            self.any_replacement = true;
        } else if place_ref.local != place.local
            || place_ref.projection.len() < place.projection.len()
        {
            // By the invariant on `place_ref`.
            *place = place_ref.project_deeper(&[], self.tcx);
            self.reused_locals.insert(place_ref.local);
            self.any_replacement = true;
        }

        Some(value)
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn simplify_operand(
        &mut self,
        operand: &mut Operand<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        match *operand {
            Operand::Constant(ref constant) => Some(self.insert(Value::Constant(constant.const_))),
            Operand::Copy(ref mut place) | Operand::Move(ref mut place) => {
                let value = self.simplify_place_value(place, location)?;
                if let Some(const_) = self.try_as_constant(value) {
                    *operand = Operand::Constant(Box::new(const_));
                    self.any_replacement = true;
                }
                Some(value)
            }
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn simplify_rvalue(
        &mut self,
        rvalue: &mut Rvalue<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        let value = match *rvalue {
            // Forward values.
            Rvalue::Use(ref mut operand) => return self.simplify_operand(operand, location),
            Rvalue::CopyForDeref(place) => {
                let mut operand = Operand::Copy(place);
                let val = self.simplify_operand(&mut operand, location);
                *rvalue = Rvalue::Use(operand);
                return val;
            }

            // Roots.
            Rvalue::Repeat(ref mut op, amount) => {
                let op = self.simplify_operand(op, location)?;
                Value::Repeat(op, amount)
            }
            Rvalue::NullaryOp(op, ty) => Value::NullaryOp(op, ty),
            Rvalue::Aggregate(box ref kind, ref mut fields) => {
                let variant_index = match *kind {
                    AggregateKind::Array(..)
                    | AggregateKind::Tuple
                    | AggregateKind::Closure(..)
                    | AggregateKind::Coroutine(..) => FIRST_VARIANT,
                    AggregateKind::Adt(_, variant_index, _, _, None) => variant_index,
                    // Do not track unions.
                    AggregateKind::Adt(_, _, _, _, Some(_)) => return None,
                };
                let fields: Option<Vec<_>> = fields
                    .iter_mut()
                    .map(|op| self.simplify_operand(op, location).or_else(|| self.new_opaque()))
                    .collect();
                let ty = rvalue.ty(self.local_decls, self.tcx);
                Value::Aggregate(ty, variant_index, fields?)
            }
            Rvalue::Ref(.., place) | Rvalue::AddressOf(_, place) => return self.new_pointer(place),

            // Operations.
            Rvalue::Len(ref mut place) => {
                let place = self.simplify_place_value(place, location)?;
                Value::Len(place)
            }
            Rvalue::Cast(kind, ref mut value, to) => {
                let from = value.ty(self.local_decls, self.tcx);
                let value = self.simplify_operand(value, location)?;
                Value::Cast { kind, value, from, to }
            }
            Rvalue::BinaryOp(op, box (ref mut lhs, ref mut rhs)) => {
                let lhs = self.simplify_operand(lhs, location);
                let rhs = self.simplify_operand(rhs, location);
                Value::BinaryOp(op, lhs?, rhs?)
            }
            Rvalue::CheckedBinaryOp(op, box (ref mut lhs, ref mut rhs)) => {
                let lhs = self.simplify_operand(lhs, location);
                let rhs = self.simplify_operand(rhs, location);
                Value::CheckedBinaryOp(op, lhs?, rhs?)
            }
            Rvalue::UnaryOp(op, ref mut arg) => {
                let arg = self.simplify_operand(arg, location)?;
                Value::UnaryOp(op, arg)
            }
            Rvalue::Discriminant(ref mut place) => {
                let place = self.simplify_place_value(place, location)?;
                Value::Discriminant(place)
            }

            // Unsupported values.
            Rvalue::ThreadLocalRef(..) | Rvalue::ShallowInitBox(..) => return None,
        };
        debug!(?value);
        Some(self.insert(value))
    }
}

impl<'tcx> VnState<'_, 'tcx> {
    /// If `index` is a `Value::Constant`, return the `Constant` to be put in the MIR.
    fn try_as_constant(&mut self, index: VnIndex) -> Option<ConstOperand<'tcx>> {
        if let Value::Constant(const_) = *self.get(index) {
            // Some constants may contain pointers. We need to preserve the provenance of these
            // pointers, but not all constants guarantee this:
            // - valtrees purposefully do not;
            // - ConstValue::Slice does not either.
            match const_ {
                Const::Ty(c) => match c.kind() {
                    ty::ConstKind::Value(valtree) => match valtree {
                        // This is just an integer, keep it.
                        ty::ValTree::Leaf(_) => {}
                        ty::ValTree::Branch(_) => return None,
                    },
                    ty::ConstKind::Param(..)
                    | ty::ConstKind::Unevaluated(..)
                    | ty::ConstKind::Expr(..) => {}
                    // Should not appear in runtime MIR.
                    ty::ConstKind::Infer(..)
                    | ty::ConstKind::Bound(..)
                    | ty::ConstKind::Placeholder(..)
                    | ty::ConstKind::Error(..) => bug!(),
                },
                Const::Unevaluated(..) => {}
                // If the same slice appears twice in the MIR, we cannot guarantee that we will
                // give the same `AllocId` to the data.
                Const::Val(ConstValue::Slice { .. }, _) => return None,
                Const::Val(
                    ConstValue::ZeroSized | ConstValue::Scalar(_) | ConstValue::Indirect { .. },
                    _,
                ) => {}
            }
            Some(ConstOperand { span: rustc_span::DUMMY_SP, user_ty: None, const_ })
        } else {
            None
        }
    }

    /// If there is a local which is assigned `index`, and its assignment strictly dominates `loc`,
    /// return it.
    fn try_as_local(&mut self, index: VnIndex, loc: Location) -> Option<Local> {
        let other = self.rev_locals.get(&index)?;
        other
            .iter()
            .copied()
            .find(|&other| self.ssa.assignment_dominates(self.dominators, other, loc))
    }
}

impl<'tcx> MutVisitor<'tcx> for VnState<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        self.simplify_operand(operand, location);
    }

    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, location: Location) {
        self.super_statement(stmt, location);
        if let StatementKind::Assign(box (_, ref mut rvalue)) = stmt.kind
            // Do not try to simplify a constant, it's already in canonical shape.
            && !matches!(rvalue, Rvalue::Use(Operand::Constant(_)))
            && let Some(value) = self.simplify_rvalue(rvalue, location)
        {
            if let Some(const_) = self.try_as_constant(value) {
                *rvalue = Rvalue::Use(Operand::Constant(Box::new(const_)));
                self.any_replacement = true;
            } else if let Some(local) = self.try_as_local(value, location)
                && *rvalue != Rvalue::Use(Operand::Move(local.into()))
            {
                *rvalue = Rvalue::Use(Operand::Copy(local.into()));
                self.reused_locals.insert(local);
                self.any_replacement = true;
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
