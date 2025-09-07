//! Global value numbering.
//!
//! MIR may contain repeated and/or redundant computations. The objective of this pass is to detect
//! such redundancies and re-use the already-computed result when possible.
//!
//! From those assignments, we construct a mapping `VnIndex -> Vec<(Local, Location)>` of available
//! values, the locals in which they are stored, and the assignment location.
//!
//! We traverse all assignments `x = rvalue` and operands.
//!
//! For each SSA one, we compute a symbolic representation of values that are assigned to SSA
//! locals. This symbolic representation is defined by the `Value` enum. Each produced instance of
//! `Value` is interned as a `VnIndex`, which allows us to cheaply compute identical values.
//!
//! For each non-SSA
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
//! We consider it to be replaceable by:
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
//! We handle references by assigning a different "provenance" index to each Ref/RawPtr rvalue.
//! This ensure that we do not spuriously merge borrows that should not be merged. Meanwhile, we
//! consider all the derefs of an immutable reference to a freeze type to give the same value:
//! ```ignore (MIR)
//! _a = *_b // _b is &Freeze
//! _c = *_b // replaced by _c = _a
//! ```
//!
//! # Determinism of constant propagation
//!
//! When registering a new `Value`, we attempt to opportunistically evaluate it as a constant.
//! The evaluated form is inserted in `evaluated` as an `OpTy` or `None` if evaluation failed.
//!
//! The difficulty is non-deterministic evaluation of MIR constants. Some `Const` can have
//! different runtime values each time they are evaluated. This is the case with
//! `Const::Slice` which have a new pointer each time they are evaluated, and constants that
//! contain a fn pointer (`AllocId` pointing to a `GlobalAlloc::Function`) pointing to a different
//! symbol in each codegen unit.
//!
//! Meanwhile, we want to be able to read indirect constants. For instance:
//! ```
//! static A: &'static &'static u8 = &&63;
//! fn foo() -> u8 {
//!     **A // We want to replace by 63.
//! }
//! fn bar() -> u8 {
//!     b"abc"[1] // We want to replace by 'b'.
//! }
//! ```
//!
//! The `Value::Constant` variant stores a possibly unevaluated constant. Evaluating that constant
//! may be non-deterministic. When that happens, we assign a disambiguator to ensure that we do not
//! merge the constants. See `duplicate_slice` test in `gvn.rs`.
//!
//! Second, when writing constants in MIR, we do not write `Const::Slice` or `Const`
//! that contain `AllocId`s.

use std::borrow::Cow;
use std::hash::{Hash, Hasher};

use either::Either;
use hashbrown::hash_table::{Entry, HashTable};
use itertools::Itertools as _;
use rustc_abi::{self as abi, BackendRepr, FIRST_VARIANT, FieldIdx, Primitive, Size, VariantIdx};
use rustc_arena::DroplessArena;
use rustc_const_eval::const_eval::DummyMachine;
use rustc_const_eval::interpret::{
    ImmTy, Immediate, InterpCx, MemPlaceMeta, MemoryKind, OpTy, Projectable, Scalar,
    intern_const_alloc_for_constprop,
};
use rustc_data_structures::fx::FxHasher;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_hir::def::DefKind;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexVec, newtype_index};
use rustc_middle::bug;
use rustc_middle::mir::interpret::GlobalAlloc;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::layout::HasTypingEnv;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;
use smallvec::SmallVec;
use tracing::{debug, instrument, trace};

use crate::ssa::SsaLocals;

pub(super) struct GVN;

impl<'tcx> crate::MirPass<'tcx> for GVN {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    #[instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());

        let typing_env = body.typing_env(tcx);
        let ssa = SsaLocals::new(tcx, body, typing_env);
        // Clone dominators because we need them while mutating the body.
        let dominators = body.basic_blocks.dominators().clone();

        let arena = DroplessArena::default();
        let mut state =
            VnState::new(tcx, body, typing_env, &ssa, dominators, &body.local_decls, &arena);

        for local in body.args_iter().filter(|&local| ssa.is_ssa(local)) {
            let opaque = state.new_opaque(body.local_decls[local].ty);
            state.assign(local, opaque);
        }

        let reverse_postorder = body.basic_blocks.reverse_postorder().to_vec();
        for bb in reverse_postorder {
            let data = &mut body.basic_blocks.as_mut_preserves_cfg()[bb];
            state.visit_basic_block_data(bb, data);
        }

        // For each local that is reused (`y` above), we remove its storage statements do avoid any
        // difficulty. Those locals are SSA, so should be easy to optimize by LLVM without storage
        // statements.
        StorageRemover { tcx, reused_locals: state.reused_locals }.visit_body_preserves_cfg(body);
    }

    fn is_required(&self) -> bool {
        false
    }
}

newtype_index! {
    /// This represents a `Value` in the symbolic execution.
    #[debug_format = "_v{}"]
    struct VnIndex {}
}

/// Marker type to forbid hashing and comparing opaque values.
/// This struct should only be constructed by `ValueSet::insert_unique` to ensure we use that
/// method to create non-unifiable values. It will ICE if used in `ValueSet::insert`.
#[derive(Copy, Clone, Debug, Eq)]
struct VnOpaque;
impl PartialEq for VnOpaque {
    fn eq(&self, _: &VnOpaque) -> bool {
        // ICE if we try to compare unique values
        unreachable!()
    }
}
impl Hash for VnOpaque {
    fn hash<T: Hasher>(&self, _: &mut T) {
        // ICE if we try to hash unique values
        unreachable!()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum AddressKind {
    Ref(BorrowKind),
    Address(RawPtrKind),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum AddressBase {
    /// This address is based on this local.
    Local(Local),
    /// This address is based on the deref of this pointer.
    Deref(VnIndex),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Value<'a, 'tcx> {
    // Root values.
    /// Used to represent values we know nothing about.
    /// The `usize` is a counter incremented by `new_opaque`.
    Opaque(VnOpaque),
    /// Evaluated or unevaluated constant value.
    Constant {
        value: Const<'tcx>,
        /// Some constants do not have a deterministic value. To avoid merging two instances of the
        /// same `Const`, we assign them an additional integer index.
        // `disambiguator` is `None` iff the constant is deterministic.
        disambiguator: Option<VnOpaque>,
    },

    // Aggregates.
    /// An aggregate value, either tuple/closure/struct/enum.
    /// This does not contain unions, as we cannot reason with the value.
    Aggregate(VariantIdx, &'a [VnIndex]),
    /// A union aggregate value.
    Union(FieldIdx, VnIndex),
    /// A raw pointer aggregate built from a thin pointer and metadata.
    RawPtr {
        /// Thin pointer component. This is field 0 in MIR.
        pointer: VnIndex,
        /// Metadata component. This is field 1 in MIR.
        metadata: VnIndex,
    },
    /// This corresponds to a `[value; count]` expression.
    Repeat(VnIndex, ty::Const<'tcx>),
    /// The address of a place.
    Address {
        base: AddressBase,
        // We do not use a plain `Place` as we want to be able to reason about indices.
        // This does not contain any `Deref` projection.
        projection: &'a [ProjectionElem<VnIndex, Ty<'tcx>>],
        kind: AddressKind,
        /// Give each borrow and pointer a different provenance, so we don't merge them.
        provenance: VnOpaque,
    },

    // Extractions.
    /// This is the *value* obtained by projecting another value.
    Projection(VnIndex, ProjectionElem<VnIndex, ()>),
    /// Discriminant of the given value.
    Discriminant(VnIndex),

    // Operations.
    NullaryOp(NullOp<'tcx>, Ty<'tcx>),
    UnaryOp(UnOp, VnIndex),
    BinaryOp(BinOp, VnIndex, VnIndex),
    Cast {
        kind: CastKind,
        value: VnIndex,
    },
}

/// Stores and deduplicates pairs of `(Value, Ty)` into in `VnIndex` numbered values.
///
/// This data structure is mostly a partial reimplementation of `FxIndexMap<VnIndex, (Value, Ty)>`.
/// We do not use a regular `FxIndexMap` to skip hashing values that are unique by construction,
/// like opaque values, address with provenance and non-deterministic constants.
struct ValueSet<'a, 'tcx> {
    indices: HashTable<VnIndex>,
    hashes: IndexVec<VnIndex, u64>,
    values: IndexVec<VnIndex, Value<'a, 'tcx>>,
    types: IndexVec<VnIndex, Ty<'tcx>>,
}

impl<'a, 'tcx> ValueSet<'a, 'tcx> {
    fn new(num_values: usize) -> ValueSet<'a, 'tcx> {
        ValueSet {
            indices: HashTable::with_capacity(num_values),
            hashes: IndexVec::with_capacity(num_values),
            values: IndexVec::with_capacity(num_values),
            types: IndexVec::with_capacity(num_values),
        }
    }

    /// Insert a `(Value, Ty)` pair without hashing or deduplication.
    /// This always creates a new `VnIndex`.
    #[inline]
    fn insert_unique(
        &mut self,
        ty: Ty<'tcx>,
        value: impl FnOnce(VnOpaque) -> Value<'a, 'tcx>,
    ) -> VnIndex {
        let value = value(VnOpaque);

        debug_assert!(match value {
            Value::Opaque(_) | Value::Address { .. } => true,
            Value::Constant { disambiguator, .. } => disambiguator.is_some(),
            _ => false,
        });

        let index = self.hashes.push(0);
        let _index = self.types.push(ty);
        debug_assert_eq!(index, _index);
        let _index = self.values.push(value);
        debug_assert_eq!(index, _index);
        index
    }

    /// Insert a `(Value, Ty)` pair to be deduplicated.
    /// Returns `true` as second tuple field if this value did not exist previously.
    #[allow(rustc::pass_by_value)] // closures take `&VnIndex`
    fn insert(&mut self, ty: Ty<'tcx>, value: Value<'a, 'tcx>) -> (VnIndex, bool) {
        debug_assert!(match value {
            Value::Opaque(_) | Value::Address { .. } => false,
            Value::Constant { disambiguator, .. } => disambiguator.is_none(),
            _ => true,
        });

        let hash: u64 = {
            let mut h = FxHasher::default();
            value.hash(&mut h);
            ty.hash(&mut h);
            h.finish()
        };

        let eq = |index: &VnIndex| self.values[*index] == value && self.types[*index] == ty;
        let hasher = |index: &VnIndex| self.hashes[*index];
        match self.indices.entry(hash, eq, hasher) {
            Entry::Occupied(entry) => {
                let index = *entry.get();
                (index, false)
            }
            Entry::Vacant(entry) => {
                let index = self.hashes.push(hash);
                entry.insert(index);
                let _index = self.values.push(value);
                debug_assert_eq!(index, _index);
                let _index = self.types.push(ty);
                debug_assert_eq!(index, _index);
                (index, true)
            }
        }
    }

    /// Return the `Value` associated with the given `VnIndex`.
    #[inline]
    fn value(&self, index: VnIndex) -> Value<'a, 'tcx> {
        self.values[index]
    }

    /// Return the type associated with the given `VnIndex`.
    #[inline]
    fn ty(&self, index: VnIndex) -> Ty<'tcx> {
        self.types[index]
    }

    /// Replace the value associated with `index` with an opaque value.
    #[inline]
    fn forget(&mut self, index: VnIndex) {
        self.values[index] = Value::Opaque(VnOpaque);
    }
}

struct VnState<'body, 'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    ecx: InterpCx<'tcx, DummyMachine>,
    local_decls: &'body LocalDecls<'tcx>,
    is_coroutine: bool,
    /// Value stored in each local.
    locals: IndexVec<Local, Option<VnIndex>>,
    /// Locals that are assigned that value.
    // This vector does not hold all the values of `VnIndex` that we create.
    rev_locals: IndexVec<VnIndex, SmallVec<[Local; 1]>>,
    values: ValueSet<'a, 'tcx>,
    /// Values evaluated as constants if possible.
    evaluated: IndexVec<VnIndex, Option<OpTy<'tcx>>>,
    /// Cache the deref values.
    derefs: Vec<VnIndex>,
    ssa: &'body SsaLocals,
    dominators: Dominators<BasicBlock>,
    reused_locals: DenseBitSet<Local>,
    arena: &'a DroplessArena,
}

impl<'body, 'a, 'tcx> VnState<'body, 'a, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        ssa: &'body SsaLocals,
        dominators: Dominators<BasicBlock>,
        local_decls: &'body LocalDecls<'tcx>,
        arena: &'a DroplessArena,
    ) -> Self {
        // Compute a rough estimate of the number of values in the body from the number of
        // statements. This is meant to reduce the number of allocations, but it's all right if
        // we miss the exact amount. We estimate based on 2 values per statement (one in LHS and
        // one in RHS) and 4 values per terminator (for call operands).
        let num_values =
            2 * body.basic_blocks.iter().map(|bbdata| bbdata.statements.len()).sum::<usize>()
                + 4 * body.basic_blocks.len();
        VnState {
            tcx,
            ecx: InterpCx::new(tcx, DUMMY_SP, typing_env, DummyMachine),
            local_decls,
            is_coroutine: body.coroutine.is_some(),
            locals: IndexVec::from_elem(None, local_decls),
            rev_locals: IndexVec::with_capacity(num_values),
            values: ValueSet::new(num_values),
            evaluated: IndexVec::with_capacity(num_values),
            derefs: Vec::new(),
            ssa,
            dominators,
            reused_locals: DenseBitSet::new_empty(local_decls.len()),
            arena,
        }
    }

    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.ecx.typing_env()
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn insert(&mut self, ty: Ty<'tcx>, value: Value<'a, 'tcx>) -> VnIndex {
        let (index, new) = self.values.insert(ty, value);
        if new {
            // Grow `evaluated` and `rev_locals` here to amortize the allocations.
            let evaluated = self.eval_to_const(index);
            let _index = self.evaluated.push(evaluated);
            debug_assert_eq!(index, _index);
            let _index = self.rev_locals.push(SmallVec::new());
            debug_assert_eq!(index, _index);
        }
        index
    }

    /// Create a new `Value` for which we have no information at all, except that it is distinct
    /// from all the others.
    #[instrument(level = "trace", skip(self), ret)]
    fn new_opaque(&mut self, ty: Ty<'tcx>) -> VnIndex {
        let index = self.values.insert_unique(ty, Value::Opaque);
        let _index = self.evaluated.push(None);
        debug_assert_eq!(index, _index);
        let _index = self.rev_locals.push(SmallVec::new());
        debug_assert_eq!(index, _index);
        index
    }

    /// Create a new `Value::Address` distinct from all the others.
    #[instrument(level = "trace", skip(self), ret)]
    fn new_pointer(&mut self, place: Place<'tcx>, kind: AddressKind) -> Option<VnIndex> {
        let pty = place.ty(self.local_decls, self.tcx).ty;
        let ty = match kind {
            AddressKind::Ref(bk) => {
                Ty::new_ref(self.tcx, self.tcx.lifetimes.re_erased, pty, bk.to_mutbl_lossy())
            }
            AddressKind::Address(mutbl) => Ty::new_ptr(self.tcx, pty, mutbl.to_mutbl_lossy()),
        };

        let mut projection = place.projection.iter();
        let base = if place.is_indirect_first_projection() {
            let base = self.locals[place.local]?;
            // Skip the initial `Deref`.
            projection.next();
            AddressBase::Deref(base)
        } else {
            AddressBase::Local(place.local)
        };
        // Do not try evaluating inside `Index`, this has been done by `simplify_place_projection`.
        let projection =
            projection.map(|proj| proj.try_map(|index| self.locals[index], |ty| ty).ok_or(()));
        let projection = self.arena.try_alloc_from_iter(projection).ok()?;

        let index = self.values.insert_unique(ty, |provenance| Value::Address {
            base,
            projection,
            kind,
            provenance,
        });
        let evaluated = self.eval_to_const(index);
        let _index = self.evaluated.push(evaluated);
        debug_assert_eq!(index, _index);
        let _index = self.rev_locals.push(SmallVec::new());
        debug_assert_eq!(index, _index);

        Some(index)
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn insert_constant(&mut self, value: Const<'tcx>) -> VnIndex {
        let (index, new) = if value.is_deterministic() {
            // The constant is deterministic, no need to disambiguate.
            let constant = Value::Constant { value, disambiguator: None };
            self.values.insert(value.ty(), constant)
        } else {
            // Multiple mentions of this constant will yield different values,
            // so assign a different `disambiguator` to ensure they do not get the same `VnIndex`.
            let index = self.values.insert_unique(value.ty(), |disambiguator| Value::Constant {
                value,
                disambiguator: Some(disambiguator),
            });
            (index, true)
        };
        if new {
            let evaluated = self.eval_to_const(index);
            let _index = self.evaluated.push(evaluated);
            debug_assert_eq!(index, _index);
            let _index = self.rev_locals.push(SmallVec::new());
            debug_assert_eq!(index, _index);
        }
        index
    }

    #[inline]
    fn get(&self, index: VnIndex) -> Value<'a, 'tcx> {
        self.values.value(index)
    }

    #[inline]
    fn ty(&self, index: VnIndex) -> Ty<'tcx> {
        self.values.ty(index)
    }

    /// Record that `local` is assigned `value`. `local` must be SSA.
    #[instrument(level = "trace", skip(self))]
    fn assign(&mut self, local: Local, value: VnIndex) {
        debug_assert!(self.ssa.is_ssa(local));
        self.locals[local] = Some(value);
        self.rev_locals[value].push(local);
    }

    fn insert_bool(&mut self, flag: bool) -> VnIndex {
        // Booleans are deterministic.
        let value = Const::from_bool(self.tcx, flag);
        debug_assert!(value.is_deterministic());
        self.insert(self.tcx.types.bool, Value::Constant { value, disambiguator: None })
    }

    fn insert_scalar(&mut self, ty: Ty<'tcx>, scalar: Scalar) -> VnIndex {
        // Scalars are deterministic.
        let value = Const::from_scalar(self.tcx, scalar, ty);
        debug_assert!(value.is_deterministic());
        self.insert(ty, Value::Constant { value, disambiguator: None })
    }

    fn insert_tuple(&mut self, ty: Ty<'tcx>, values: &[VnIndex]) -> VnIndex {
        self.insert(ty, Value::Aggregate(VariantIdx::ZERO, self.arena.alloc_slice(values)))
    }

    fn insert_deref(&mut self, ty: Ty<'tcx>, value: VnIndex) -> VnIndex {
        let value = self.insert(ty, Value::Projection(value, ProjectionElem::Deref));
        self.derefs.push(value);
        value
    }

    fn invalidate_derefs(&mut self) {
        for deref in std::mem::take(&mut self.derefs) {
            self.values.forget(deref);
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn eval_to_const(&mut self, value: VnIndex) -> Option<OpTy<'tcx>> {
        use Value::*;
        let ty = self.ty(value);
        // Avoid computing layouts inside a coroutine, as that can cause cycles.
        let ty = if !self.is_coroutine || ty.is_scalar() {
            self.ecx.layout_of(ty).ok()?
        } else {
            return None;
        };
        let op = match self.get(value) {
            _ if ty.is_zst() => ImmTy::uninit(ty).into(),

            Opaque(_) => return None,
            // Do not bother evaluating repeat expressions. This would uselessly consume memory.
            Repeat(..) => return None,

            Constant { ref value, disambiguator: _ } => {
                self.ecx.eval_mir_constant(value, DUMMY_SP, None).discard_err()?
            }
            Aggregate(variant, ref fields) => {
                let fields = fields
                    .iter()
                    .map(|&f| self.evaluated[f].as_ref())
                    .collect::<Option<Vec<_>>>()?;
                let variant = if ty.ty.is_enum() { Some(variant) } else { None };
                if matches!(ty.backend_repr, BackendRepr::Scalar(..) | BackendRepr::ScalarPair(..))
                {
                    let dest = self.ecx.allocate(ty, MemoryKind::Stack).discard_err()?;
                    let variant_dest = if let Some(variant) = variant {
                        self.ecx.project_downcast(&dest, variant).discard_err()?
                    } else {
                        dest.clone()
                    };
                    for (field_index, op) in fields.into_iter().enumerate() {
                        let field_dest = self
                            .ecx
                            .project_field(&variant_dest, FieldIdx::from_usize(field_index))
                            .discard_err()?;
                        self.ecx.copy_op(op, &field_dest).discard_err()?;
                    }
                    self.ecx
                        .write_discriminant(variant.unwrap_or(FIRST_VARIANT), &dest)
                        .discard_err()?;
                    self.ecx
                        .alloc_mark_immutable(dest.ptr().provenance.unwrap().alloc_id())
                        .discard_err()?;
                    dest.into()
                } else {
                    return None;
                }
            }
            Union(active_field, field) => {
                let field = self.evaluated[field].as_ref()?;
                if matches!(ty.backend_repr, BackendRepr::Scalar(..) | BackendRepr::ScalarPair(..))
                {
                    let dest = self.ecx.allocate(ty, MemoryKind::Stack).discard_err()?;
                    let field_dest = self.ecx.project_field(&dest, active_field).discard_err()?;
                    self.ecx.copy_op(field, &field_dest).discard_err()?;
                    self.ecx
                        .alloc_mark_immutable(dest.ptr().provenance.unwrap().alloc_id())
                        .discard_err()?;
                    dest.into()
                } else {
                    return None;
                }
            }
            RawPtr { pointer, metadata } => {
                let pointer = self.evaluated[pointer].as_ref()?;
                let metadata = self.evaluated[metadata].as_ref()?;

                // Pointers don't have fields, so don't `project_field` them.
                let data = self.ecx.read_pointer(pointer).discard_err()?;
                let meta = if metadata.layout.is_zst() {
                    MemPlaceMeta::None
                } else {
                    MemPlaceMeta::Meta(self.ecx.read_scalar(metadata).discard_err()?)
                };
                let ptr_imm = Immediate::new_pointer_with_meta(data, meta, &self.ecx);
                ImmTy::from_immediate(ptr_imm, ty).into()
            }

            Projection(base, elem) => {
                let base = self.evaluated[base].as_ref()?;
                // `Index` by constants should have been replaced by `ConstantIndex` by
                // `simplify_place_projection`.
                let elem = elem.try_map(|_| None, |()| ty.ty)?;
                self.ecx.project(base, elem).discard_err()?
            }
            Address { base, projection, .. } => {
                debug_assert!(!projection.contains(&ProjectionElem::Deref));
                let pointer = match base {
                    AddressBase::Deref(pointer) => self.evaluated[pointer].as_ref()?,
                    // We have no stack to point to.
                    AddressBase::Local(_) => return None,
                };
                let mut mplace = self.ecx.deref_pointer(pointer).discard_err()?;
                for elem in projection {
                    // `Index` by constants should have been replaced by `ConstantIndex` by
                    // `simplify_place_projection`.
                    let elem = elem.try_map(|_| None, |ty| ty)?;
                    mplace = self.ecx.project(&mplace, elem).discard_err()?;
                }
                let pointer = mplace.to_ref(&self.ecx);
                ImmTy::from_immediate(pointer, ty).into()
            }

            Discriminant(base) => {
                let base = self.evaluated[base].as_ref()?;
                let variant = self.ecx.read_discriminant(base).discard_err()?;
                let discr_value =
                    self.ecx.discriminant_for_variant(base.layout.ty, variant).discard_err()?;
                discr_value.into()
            }
            NullaryOp(null_op, arg_ty) => {
                let arg_layout = self.ecx.layout_of(arg_ty).ok()?;
                if let NullOp::SizeOf | NullOp::AlignOf = null_op
                    && arg_layout.is_unsized()
                {
                    return None;
                }
                let val = match null_op {
                    NullOp::SizeOf => arg_layout.size.bytes(),
                    NullOp::AlignOf => arg_layout.align.bytes(),
                    NullOp::OffsetOf(fields) => self
                        .ecx
                        .tcx
                        .offset_of_subfield(self.typing_env(), arg_layout, fields.iter())
                        .bytes(),
                    NullOp::UbChecks => return None,
                    NullOp::ContractChecks => return None,
                };
                ImmTy::from_uint(val, ty).into()
            }
            UnaryOp(un_op, operand) => {
                let operand = self.evaluated[operand].as_ref()?;
                let operand = self.ecx.read_immediate(operand).discard_err()?;
                let val = self.ecx.unary_op(un_op, &operand).discard_err()?;
                val.into()
            }
            BinaryOp(bin_op, lhs, rhs) => {
                let lhs = self.evaluated[lhs].as_ref()?;
                let lhs = self.ecx.read_immediate(lhs).discard_err()?;
                let rhs = self.evaluated[rhs].as_ref()?;
                let rhs = self.ecx.read_immediate(rhs).discard_err()?;
                let val = self.ecx.binary_op(bin_op, &lhs, &rhs).discard_err()?;
                val.into()
            }
            Cast { kind, value } => match kind {
                CastKind::IntToInt | CastKind::IntToFloat => {
                    let value = self.evaluated[value].as_ref()?;
                    let value = self.ecx.read_immediate(value).discard_err()?;
                    let res = self.ecx.int_to_int_or_float(&value, ty).discard_err()?;
                    res.into()
                }
                CastKind::FloatToFloat | CastKind::FloatToInt => {
                    let value = self.evaluated[value].as_ref()?;
                    let value = self.ecx.read_immediate(value).discard_err()?;
                    let res = self.ecx.float_to_float_or_int(&value, ty).discard_err()?;
                    res.into()
                }
                CastKind::Transmute | CastKind::Subtype => {
                    let value = self.evaluated[value].as_ref()?;
                    // `offset` for immediates generally only supports projections that match the
                    // type of the immediate. However, as a HACK, we exploit that it can also do
                    // limited transmutes: it only works between types with the same layout, and
                    // cannot transmute pointers to integers.
                    if value.as_mplace_or_imm().is_right() {
                        let can_transmute = match (value.layout.backend_repr, ty.backend_repr) {
                            (BackendRepr::Scalar(s1), BackendRepr::Scalar(s2)) => {
                                s1.size(&self.ecx) == s2.size(&self.ecx)
                                    && !matches!(s1.primitive(), Primitive::Pointer(..))
                            }
                            (BackendRepr::ScalarPair(a1, b1), BackendRepr::ScalarPair(a2, b2)) => {
                                a1.size(&self.ecx) == a2.size(&self.ecx) &&
                                b1.size(&self.ecx) == b2.size(&self.ecx) &&
                                // The alignment of the second component determines its offset, so that also needs to match.
                                b1.align(&self.ecx) == b2.align(&self.ecx) &&
                                // None of the inputs may be a pointer.
                                !matches!(a1.primitive(), Primitive::Pointer(..))
                                    && !matches!(b1.primitive(), Primitive::Pointer(..))
                            }
                            _ => false,
                        };
                        if !can_transmute {
                            return None;
                        }
                    }
                    value.offset(Size::ZERO, ty, &self.ecx).discard_err()?
                }
                CastKind::PointerCoercion(ty::adjustment::PointerCoercion::Unsize, _) => {
                    let src = self.evaluated[value].as_ref()?;
                    let dest = self.ecx.allocate(ty, MemoryKind::Stack).discard_err()?;
                    self.ecx.unsize_into(src, ty, &dest).discard_err()?;
                    self.ecx
                        .alloc_mark_immutable(dest.ptr().provenance.unwrap().alloc_id())
                        .discard_err()?;
                    dest.into()
                }
                CastKind::FnPtrToPtr | CastKind::PtrToPtr => {
                    let src = self.evaluated[value].as_ref()?;
                    let src = self.ecx.read_immediate(src).discard_err()?;
                    let ret = self.ecx.ptr_to_ptr(&src, ty).discard_err()?;
                    ret.into()
                }
                CastKind::PointerCoercion(ty::adjustment::PointerCoercion::UnsafeFnPointer, _) => {
                    let src = self.evaluated[value].as_ref()?;
                    let src = self.ecx.read_immediate(src).discard_err()?;
                    ImmTy::from_immediate(*src, ty).into()
                }
                _ => return None,
            },
        };
        Some(op)
    }

    /// Represent the *value* we obtain by dereferencing an `Address` value.
    #[instrument(level = "trace", skip(self), ret)]
    fn dereference_address(
        &mut self,
        base: AddressBase,
        projection: &[ProjectionElem<VnIndex, Ty<'tcx>>],
    ) -> Option<VnIndex> {
        let (mut place_ty, mut value) = match base {
            // The base is a local, so we take the local's value and project from it.
            AddressBase::Local(local) => {
                let local = self.locals[local]?;
                let place_ty = PlaceTy::from_ty(self.ty(local));
                (place_ty, local)
            }
            // The base is a pointer's deref, so we introduce the implicit deref.
            AddressBase::Deref(reborrow) => {
                let place_ty = PlaceTy::from_ty(self.ty(reborrow));
                self.project(place_ty, reborrow, ProjectionElem::Deref)?
            }
        };
        for &proj in projection {
            (place_ty, value) = self.project(place_ty, value, proj)?;
        }
        Some(value)
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn project(
        &mut self,
        place_ty: PlaceTy<'tcx>,
        value: VnIndex,
        proj: ProjectionElem<VnIndex, Ty<'tcx>>,
    ) -> Option<(PlaceTy<'tcx>, VnIndex)> {
        let projection_ty = place_ty.projection_ty(self.tcx, proj);
        let proj = match proj {
            ProjectionElem::Deref => {
                if let Some(Mutability::Not) = place_ty.ty.ref_mutability()
                    && projection_ty.ty.is_freeze(self.tcx, self.typing_env())
                {
                    if let Value::Address { base, projection, .. } = self.get(value)
                        && let Some(value) = self.dereference_address(base, projection)
                    {
                        return Some((projection_ty, value));
                    }

                    // An immutable borrow `_x` always points to the same value for the
                    // lifetime of the borrow, so we can merge all instances of `*_x`.
                    return Some((projection_ty, self.insert_deref(projection_ty.ty, value)));
                } else {
                    return None;
                }
            }
            ProjectionElem::Downcast(name, index) => ProjectionElem::Downcast(name, index),
            ProjectionElem::Field(f, _) => match self.get(value) {
                Value::Aggregate(_, fields) => return Some((projection_ty, fields[f.as_usize()])),
                Value::Union(active, field) if active == f => return Some((projection_ty, field)),
                Value::Projection(outer_value, ProjectionElem::Downcast(_, read_variant))
                    if let Value::Aggregate(written_variant, fields) = self.get(outer_value)
                    // This pass is not aware of control-flow, so we do not know whether the
                    // replacement we are doing is actually reachable. We could be in any arm of
                    // ```
                    // match Some(x) {
                    //     Some(y) => /* stuff */,
                    //     None => /* other */,
                    // }
                    // ```
                    //
                    // In surface rust, the current statement would be unreachable.
                    //
                    // However, from the reference chapter on enums and RFC 2195,
                    // accessing the wrong variant is not UB if the enum has repr.
                    // So it's not impossible for a series of MIR opts to generate
                    // a downcast to an inactive variant.
                    && written_variant == read_variant =>
                {
                    return Some((projection_ty, fields[f.as_usize()]));
                }
                _ => ProjectionElem::Field(f, ()),
            },
            ProjectionElem::Index(idx) => {
                if let Value::Repeat(inner, _) = self.get(value) {
                    return Some((projection_ty, inner));
                }
                ProjectionElem::Index(idx)
            }
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                match self.get(value) {
                    Value::Repeat(inner, _) => {
                        return Some((projection_ty, inner));
                    }
                    Value::Aggregate(_, operands) => {
                        let offset = if from_end {
                            operands.len() - offset as usize
                        } else {
                            offset as usize
                        };
                        let value = operands.get(offset).copied()?;
                        return Some((projection_ty, value));
                    }
                    _ => {}
                };
                ProjectionElem::ConstantIndex { offset, min_length, from_end }
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                ProjectionElem::Subslice { from, to, from_end }
            }
            ProjectionElem::OpaqueCast(_) => ProjectionElem::OpaqueCast(()),
            ProjectionElem::UnwrapUnsafeBinder(_) => ProjectionElem::UnwrapUnsafeBinder(()),
        };

        let value = self.insert(projection_ty.ty, Value::Projection(value, proj));
        Some((projection_ty, value))
    }

    /// Simplify the projection chain if we know better.
    #[instrument(level = "trace", skip(self))]
    fn simplify_place_projection(&mut self, place: &mut Place<'tcx>, location: Location) {
        // If the projection is indirect, we treat the local as a value, so can replace it with
        // another local.
        if place.is_indirect_first_projection()
            && let Some(base) = self.locals[place.local]
            && let Some(new_local) = self.try_as_local(base, location)
            && place.local != new_local
        {
            place.local = new_local;
            self.reused_locals.insert(new_local);
        }

        let mut projection = Cow::Borrowed(&place.projection[..]);

        for i in 0..projection.len() {
            let elem = projection[i];
            if let ProjectionElem::Index(idx_local) = elem
                && let Some(idx) = self.locals[idx_local]
            {
                if let Some(offset) = self.evaluated[idx].as_ref()
                    && let Some(offset) = self.ecx.read_target_usize(offset).discard_err()
                    && let Some(min_length) = offset.checked_add(1)
                {
                    projection.to_mut()[i] =
                        ProjectionElem::ConstantIndex { offset, min_length, from_end: false };
                } else if let Some(new_idx_local) = self.try_as_local(idx, location)
                    && idx_local != new_idx_local
                {
                    projection.to_mut()[i] = ProjectionElem::Index(new_idx_local);
                    self.reused_locals.insert(new_idx_local);
                }
            }
        }

        if projection.is_owned() {
            place.projection = self.tcx.mk_place_elems(&projection);
        }

        trace!(?place);
    }

    /// Represent the *value* which would be read from `place`. If we succeed, return it.
    /// If we fail, return a `PlaceRef` that contains the same value.
    #[instrument(level = "trace", skip(self), ret)]
    fn compute_place_value(
        &mut self,
        place: Place<'tcx>,
        location: Location,
    ) -> Result<VnIndex, PlaceRef<'tcx>> {
        // Invariant: `place` and `place_ref` point to the same value, even if they point to
        // different memory locations.
        let mut place_ref = place.as_ref();

        // Invariant: `value` holds the value up-to the `index`th projection excluded.
        let Some(mut value) = self.locals[place.local] else { return Err(place_ref) };
        // Invariant: `value` has type `place_ty`, with optional downcast variant if needed.
        let mut place_ty = PlaceTy::from_ty(self.local_decls[place.local].ty);
        for (index, proj) in place.projection.iter().enumerate() {
            if let Some(local) = self.try_as_local(value, location) {
                // Both `local` and `Place { local: place.local, projection: projection[..index] }`
                // hold the same value. Therefore, following place holds the value in the original
                // `place`.
                place_ref = PlaceRef { local, projection: &place.projection[index..] };
            }

            let Some(proj) = proj.try_map(|value| self.locals[value], |ty| ty) else {
                return Err(place_ref);
            };
            let Some(ty_and_value) = self.project(place_ty, value, proj) else {
                return Err(place_ref);
            };
            (place_ty, value) = ty_and_value;
        }

        Ok(value)
    }

    /// Represent the *value* which would be read from `place`, and point `place` to a preexisting
    /// place with the same value (if that already exists).
    #[instrument(level = "trace", skip(self), ret)]
    fn simplify_place_value(
        &mut self,
        place: &mut Place<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        self.simplify_place_projection(place, location);

        match self.compute_place_value(*place, location) {
            Ok(value) => {
                if let Some(new_place) = self.try_as_place(value, location, true)
                    && (new_place.local != place.local
                        || new_place.projection.len() < place.projection.len())
                {
                    *place = new_place;
                    self.reused_locals.insert(new_place.local);
                }
                Some(value)
            }
            Err(place_ref) => {
                if place_ref.local != place.local
                    || place_ref.projection.len() < place.projection.len()
                {
                    // By the invariant on `place_ref`.
                    *place = place_ref.project_deeper(&[], self.tcx);
                    self.reused_locals.insert(place_ref.local);
                }
                None
            }
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn simplify_operand(
        &mut self,
        operand: &mut Operand<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        match *operand {
            Operand::Constant(ref constant) => Some(self.insert_constant(constant.const_)),
            Operand::Copy(ref mut place) | Operand::Move(ref mut place) => {
                let value = self.simplify_place_value(place, location)?;
                if let Some(const_) = self.try_as_constant(value) {
                    *operand = Operand::Constant(Box::new(const_));
                }
                Some(value)
            }
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn simplify_rvalue(
        &mut self,
        lhs: &Place<'tcx>,
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
            Rvalue::Aggregate(..) => return self.simplify_aggregate(lhs, rvalue, location),
            Rvalue::Ref(_, borrow_kind, ref mut place) => {
                self.simplify_place_projection(place, location);
                return self.new_pointer(*place, AddressKind::Ref(borrow_kind));
            }
            Rvalue::RawPtr(mutbl, ref mut place) => {
                self.simplify_place_projection(place, location);
                return self.new_pointer(*place, AddressKind::Address(mutbl));
            }
            Rvalue::WrapUnsafeBinder(ref mut op, _) => {
                let value = self.simplify_operand(op, location)?;
                Value::Cast { kind: CastKind::Transmute, value }
            }

            // Operations.
            Rvalue::Cast(ref mut kind, ref mut value, to) => {
                return self.simplify_cast(kind, value, to, location);
            }
            Rvalue::BinaryOp(op, box (ref mut lhs, ref mut rhs)) => {
                return self.simplify_binary(op, lhs, rhs, location);
            }
            Rvalue::UnaryOp(op, ref mut arg_op) => {
                return self.simplify_unary(op, arg_op, location);
            }
            Rvalue::Discriminant(ref mut place) => {
                let place = self.simplify_place_value(place, location)?;
                if let Some(discr) = self.simplify_discriminant(place) {
                    return Some(discr);
                }
                Value::Discriminant(place)
            }

            // Unsupported values.
            Rvalue::ThreadLocalRef(..) | Rvalue::ShallowInitBox(..) => return None,
        };
        let ty = rvalue.ty(self.local_decls, self.tcx);
        Some(self.insert(ty, value))
    }

    fn simplify_discriminant(&mut self, place: VnIndex) -> Option<VnIndex> {
        let enum_ty = self.ty(place);
        if enum_ty.is_enum()
            && let Value::Aggregate(variant, _) = self.get(place)
        {
            let discr = self.ecx.discriminant_for_variant(enum_ty, variant).discard_err()?;
            return Some(self.insert_scalar(discr.layout.ty, discr.to_scalar()));
        }

        None
    }

    fn try_as_place_elem(
        &mut self,
        ty: Ty<'tcx>,
        proj: ProjectionElem<VnIndex, ()>,
        loc: Location,
    ) -> Option<PlaceElem<'tcx>> {
        proj.try_map(
            |value| {
                let local = self.try_as_local(value, loc)?;
                self.reused_locals.insert(local);
                Some(local)
            },
            |()| ty,
        )
    }

    fn simplify_aggregate_to_copy(
        &mut self,
        ty: Ty<'tcx>,
        variant_index: VariantIdx,
        fields: &[VnIndex],
    ) -> Option<VnIndex> {
        let Some(&first_field) = fields.first() else { return None };
        let Value::Projection(copy_from_value, _) = self.get(first_field) else { return None };

        // All fields must correspond one-to-one and come from the same aggregate value.
        if fields.iter().enumerate().any(|(index, &v)| {
            if let Value::Projection(pointer, ProjectionElem::Field(from_index, _)) = self.get(v)
                && copy_from_value == pointer
                && from_index.index() == index
            {
                return false;
            }
            true
        }) {
            return None;
        }

        let mut copy_from_local_value = copy_from_value;
        if let Value::Projection(pointer, proj) = self.get(copy_from_value)
            && let ProjectionElem::Downcast(_, read_variant) = proj
        {
            if variant_index == read_variant {
                // When copying a variant, there is no need to downcast.
                copy_from_local_value = pointer;
            } else {
                // The copied variant must be identical.
                return None;
            }
        }

        // Both must be variants of the same type.
        if self.ty(copy_from_local_value) == ty { Some(copy_from_local_value) } else { None }
    }

    fn simplify_aggregate(
        &mut self,
        lhs: &Place<'tcx>,
        rvalue: &mut Rvalue<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        let tcx = self.tcx;
        let ty = rvalue.ty(self.local_decls, tcx);

        let Rvalue::Aggregate(box ref kind, ref mut field_ops) = *rvalue else { bug!() };

        if field_ops.is_empty() {
            let is_zst = match *kind {
                AggregateKind::Array(..)
                | AggregateKind::Tuple
                | AggregateKind::Closure(..)
                | AggregateKind::CoroutineClosure(..) => true,
                // Only enums can be non-ZST.
                AggregateKind::Adt(did, ..) => tcx.def_kind(did) != DefKind::Enum,
                // Coroutines are never ZST, as they at least contain the implicit states.
                AggregateKind::Coroutine(..) => false,
                AggregateKind::RawPtr(..) => bug!("MIR for RawPtr aggregate must have 2 fields"),
            };

            if is_zst {
                return Some(self.insert_constant(Const::zero_sized(ty)));
            }
        }

        let fields = self.arena.alloc_from_iter(field_ops.iter_mut().map(|op| {
            self.simplify_operand(op, location)
                .unwrap_or_else(|| self.new_opaque(op.ty(self.local_decls, self.tcx)))
        }));

        let variant_index = match *kind {
            AggregateKind::Array(..) | AggregateKind::Tuple => {
                assert!(!field_ops.is_empty());
                FIRST_VARIANT
            }
            AggregateKind::Closure(..)
            | AggregateKind::CoroutineClosure(..)
            | AggregateKind::Coroutine(..) => FIRST_VARIANT,
            AggregateKind::Adt(_, variant_index, _, _, None) => variant_index,
            // Do not track unions.
            AggregateKind::Adt(_, _, _, _, Some(active_field)) => {
                let field = *fields.first()?;
                return Some(self.insert(ty, Value::Union(active_field, field)));
            }
            AggregateKind::RawPtr(..) => {
                assert_eq!(field_ops.len(), 2);
                let [mut pointer, metadata] = fields.try_into().unwrap();

                // Any thin pointer of matching mutability is fine as the data pointer.
                let mut was_updated = false;
                while let Value::Cast { kind: CastKind::PtrToPtr, value: cast_value } =
                    self.get(pointer)
                    && let ty::RawPtr(from_pointee_ty, from_mtbl) = self.ty(cast_value).kind()
                    && let ty::RawPtr(_, output_mtbl) = ty.kind()
                    && from_mtbl == output_mtbl
                    && from_pointee_ty.is_sized(self.tcx, self.typing_env())
                {
                    pointer = cast_value;
                    was_updated = true;
                }

                if was_updated && let Some(op) = self.try_as_operand(pointer, location) {
                    field_ops[FieldIdx::ZERO] = op;
                }

                return Some(self.insert(ty, Value::RawPtr { pointer, metadata }));
            }
        };

        if ty.is_array()
            && fields.len() > 4
            && let Ok(&first) = fields.iter().all_equal_value()
        {
            let len = ty::Const::from_target_usize(self.tcx, fields.len().try_into().unwrap());
            if let Some(op) = self.try_as_operand(first, location) {
                *rvalue = Rvalue::Repeat(op, len);
            }
            return Some(self.insert(ty, Value::Repeat(first, len)));
        }

        if let Some(value) = self.simplify_aggregate_to_copy(ty, variant_index, &fields) {
            // Allow introducing places with non-constant offsets, as those are still better than
            // reconstructing an aggregate. But avoid creating `*a = copy (*b)`, as they might be
            // aliases resulting in overlapping assignments.
            let allow_complex_projection =
                lhs.projection[..].iter().all(PlaceElem::is_stable_offset);
            if let Some(place) = self.try_as_place(value, location, allow_complex_projection) {
                self.reused_locals.insert(place.local);
                *rvalue = Rvalue::Use(Operand::Copy(place));
            }
            return Some(value);
        }

        Some(self.insert(ty, Value::Aggregate(variant_index, fields)))
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn simplify_unary(
        &mut self,
        op: UnOp,
        arg_op: &mut Operand<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        let mut arg_index = self.simplify_operand(arg_op, location)?;
        let arg_ty = self.ty(arg_index);
        let ret_ty = op.ty(self.tcx, arg_ty);

        // PtrMetadata doesn't care about *const vs *mut vs & vs &mut,
        // so start by removing those distinctions so we can update the `Operand`
        if op == UnOp::PtrMetadata {
            let mut was_updated = false;
            loop {
                arg_index = match self.get(arg_index) {
                    // Pointer casts that preserve metadata, such as
                    // `*const [i32]` <-> `*mut [i32]` <-> `*mut [f32]`.
                    // It's critical that this not eliminate cases like
                    // `*const [T]` -> `*const T` which remove metadata.
                    // We run on potentially-generic MIR, though, so unlike codegen
                    // we can't always know exactly what the metadata are.
                    // To allow things like `*mut (?A, ?T)` <-> `*mut (?B, ?T)`,
                    // it's fine to get a projection as the type.
                    Value::Cast { kind: CastKind::PtrToPtr, value: inner }
                        if self.pointers_have_same_metadata(self.ty(inner), arg_ty) =>
                    {
                        inner
                    }

                    // We have an unsizing cast, which assigns the length to wide pointer metadata.
                    Value::Cast {
                        kind: CastKind::PointerCoercion(ty::adjustment::PointerCoercion::Unsize, _),
                        value: from,
                    } if let Some(from) = self.ty(from).builtin_deref(true)
                        && let ty::Array(_, len) = from.kind()
                        && let Some(to) = self.ty(arg_index).builtin_deref(true)
                        && let ty::Slice(..) = to.kind() =>
                    {
                        return Some(self.insert_constant(Const::Ty(self.tcx.types.usize, *len)));
                    }

                    // `&mut *p`, `&raw *p`, etc don't change metadata.
                    Value::Address { base: AddressBase::Deref(reborrowed), projection, .. }
                        if projection.is_empty() =>
                    {
                        reborrowed
                    }

                    _ => break,
                };
                was_updated = true;
            }

            if was_updated && let Some(op) = self.try_as_operand(arg_index, location) {
                *arg_op = op;
            }
        }

        let value = match (op, self.get(arg_index)) {
            (UnOp::Not, Value::UnaryOp(UnOp::Not, inner)) => return Some(inner),
            (UnOp::Neg, Value::UnaryOp(UnOp::Neg, inner)) => return Some(inner),
            (UnOp::Not, Value::BinaryOp(BinOp::Eq, lhs, rhs)) => {
                Value::BinaryOp(BinOp::Ne, lhs, rhs)
            }
            (UnOp::Not, Value::BinaryOp(BinOp::Ne, lhs, rhs)) => {
                Value::BinaryOp(BinOp::Eq, lhs, rhs)
            }
            (UnOp::PtrMetadata, Value::RawPtr { metadata, .. }) => return Some(metadata),
            // We have an unsizing cast, which assigns the length to wide pointer metadata.
            (
                UnOp::PtrMetadata,
                Value::Cast {
                    kind: CastKind::PointerCoercion(ty::adjustment::PointerCoercion::Unsize, _),
                    value: inner,
                },
            ) if let ty::Slice(..) = arg_ty.builtin_deref(true).unwrap().kind()
                && let ty::Array(_, len) = self.ty(inner).builtin_deref(true).unwrap().kind() =>
            {
                return Some(self.insert_constant(Const::Ty(self.tcx.types.usize, *len)));
            }
            _ => Value::UnaryOp(op, arg_index),
        };
        Some(self.insert(ret_ty, value))
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn simplify_binary(
        &mut self,
        op: BinOp,
        lhs_operand: &mut Operand<'tcx>,
        rhs_operand: &mut Operand<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        let lhs = self.simplify_operand(lhs_operand, location);
        let rhs = self.simplify_operand(rhs_operand, location);

        // Only short-circuit options after we called `simplify_operand`
        // on both operands for side effect.
        let mut lhs = lhs?;
        let mut rhs = rhs?;

        let lhs_ty = self.ty(lhs);

        // If we're comparing pointers, remove `PtrToPtr` casts if the from
        // types of both casts and the metadata all match.
        if let BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge = op
            && lhs_ty.is_any_ptr()
            && let Value::Cast { kind: CastKind::PtrToPtr, value: lhs_value } = self.get(lhs)
            && let Value::Cast { kind: CastKind::PtrToPtr, value: rhs_value } = self.get(rhs)
            && let lhs_from = self.ty(lhs_value)
            && lhs_from == self.ty(rhs_value)
            && self.pointers_have_same_metadata(lhs_from, lhs_ty)
        {
            lhs = lhs_value;
            rhs = rhs_value;
            if let Some(lhs_op) = self.try_as_operand(lhs, location)
                && let Some(rhs_op) = self.try_as_operand(rhs, location)
            {
                *lhs_operand = lhs_op;
                *rhs_operand = rhs_op;
            }
        }

        if let Some(value) = self.simplify_binary_inner(op, lhs_ty, lhs, rhs) {
            return Some(value);
        }
        let ty = op.ty(self.tcx, lhs_ty, self.ty(rhs));
        let value = Value::BinaryOp(op, lhs, rhs);
        Some(self.insert(ty, value))
    }

    fn simplify_binary_inner(
        &mut self,
        op: BinOp,
        lhs_ty: Ty<'tcx>,
        lhs: VnIndex,
        rhs: VnIndex,
    ) -> Option<VnIndex> {
        // Floats are weird enough that none of the logic below applies.
        let reasonable_ty =
            lhs_ty.is_integral() || lhs_ty.is_bool() || lhs_ty.is_char() || lhs_ty.is_any_ptr();
        if !reasonable_ty {
            return None;
        }

        let layout = self.ecx.layout_of(lhs_ty).ok()?;

        let as_bits = |value: VnIndex| {
            let constant = self.evaluated[value].as_ref()?;
            if layout.backend_repr.is_scalar() {
                let scalar = self.ecx.read_scalar(constant).discard_err()?;
                scalar.to_bits(constant.layout.size).discard_err()
            } else {
                // `constant` is a wide pointer. Do not evaluate to bits.
                None
            }
        };

        // Represent the values as `Left(bits)` or `Right(VnIndex)`.
        use Either::{Left, Right};
        let a = as_bits(lhs).map_or(Right(lhs), Left);
        let b = as_bits(rhs).map_or(Right(rhs), Left);

        let result = match (op, a, b) {
            // Neutral elements.
            (
                BinOp::Add
                | BinOp::AddWithOverflow
                | BinOp::AddUnchecked
                | BinOp::BitOr
                | BinOp::BitXor,
                Left(0),
                Right(p),
            )
            | (
                BinOp::Add
                | BinOp::AddWithOverflow
                | BinOp::AddUnchecked
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Sub
                | BinOp::SubWithOverflow
                | BinOp::SubUnchecked
                | BinOp::Offset
                | BinOp::Shl
                | BinOp::Shr,
                Right(p),
                Left(0),
            )
            | (BinOp::Mul | BinOp::MulWithOverflow | BinOp::MulUnchecked, Left(1), Right(p))
            | (
                BinOp::Mul | BinOp::MulWithOverflow | BinOp::MulUnchecked | BinOp::Div,
                Right(p),
                Left(1),
            ) => p,
            // Attempt to simplify `x & ALL_ONES` to `x`, with `ALL_ONES` depending on type size.
            (BinOp::BitAnd, Right(p), Left(ones)) | (BinOp::BitAnd, Left(ones), Right(p))
                if ones == layout.size.truncate(u128::MAX)
                    || (layout.ty.is_bool() && ones == 1) =>
            {
                p
            }
            // Absorbing elements.
            (
                BinOp::Mul | BinOp::MulWithOverflow | BinOp::MulUnchecked | BinOp::BitAnd,
                _,
                Left(0),
            )
            | (BinOp::Rem, _, Left(1))
            | (
                BinOp::Mul
                | BinOp::MulWithOverflow
                | BinOp::MulUnchecked
                | BinOp::Div
                | BinOp::Rem
                | BinOp::BitAnd
                | BinOp::Shl
                | BinOp::Shr,
                Left(0),
                _,
            ) => self.insert_scalar(lhs_ty, Scalar::from_uint(0u128, layout.size)),
            // Attempt to simplify `x | ALL_ONES` to `ALL_ONES`.
            (BinOp::BitOr, _, Left(ones)) | (BinOp::BitOr, Left(ones), _)
                if ones == layout.size.truncate(u128::MAX)
                    || (layout.ty.is_bool() && ones == 1) =>
            {
                self.insert_scalar(lhs_ty, Scalar::from_uint(ones, layout.size))
            }
            // Sub/Xor with itself.
            (BinOp::Sub | BinOp::SubWithOverflow | BinOp::SubUnchecked | BinOp::BitXor, a, b)
                if a == b =>
            {
                self.insert_scalar(lhs_ty, Scalar::from_uint(0u128, layout.size))
            }
            // Comparison:
            // - if both operands can be computed as bits, just compare the bits;
            // - if we proved that both operands have the same value, we can insert true/false;
            // - otherwise, do nothing, as we do not try to prove inequality.
            (BinOp::Eq, Left(a), Left(b)) => self.insert_bool(a == b),
            (BinOp::Eq, a, b) if a == b => self.insert_bool(true),
            (BinOp::Ne, Left(a), Left(b)) => self.insert_bool(a != b),
            (BinOp::Ne, a, b) if a == b => self.insert_bool(false),
            _ => return None,
        };

        if op.is_overflowing() {
            let ty = Ty::new_tup(self.tcx, &[self.ty(result), self.tcx.types.bool]);
            let false_val = self.insert_bool(false);
            Some(self.insert_tuple(ty, &[result, false_val]))
        } else {
            Some(result)
        }
    }

    fn simplify_cast(
        &mut self,
        initial_kind: &mut CastKind,
        initial_operand: &mut Operand<'tcx>,
        to: Ty<'tcx>,
        location: Location,
    ) -> Option<VnIndex> {
        use CastKind::*;
        use rustc_middle::ty::adjustment::PointerCoercion::*;

        let mut kind = *initial_kind;
        let mut value = self.simplify_operand(initial_operand, location)?;
        let mut from = self.ty(value);
        if from == to {
            return Some(value);
        }

        if let CastKind::PointerCoercion(ReifyFnPointer | ClosureFnPointer(_), _) = kind {
            // Each reification of a generic fn may get a different pointer.
            // Do not try to merge them.
            return Some(self.new_opaque(to));
        }

        let mut was_ever_updated = false;
        loop {
            let mut was_updated_this_iteration = false;

            // Transmuting between raw pointers is just a pointer cast so long as
            // they have the same metadata type (like `*const i32` <=> `*mut u64`
            // or `*mut [i32]` <=> `*const [u64]`), including the common special
            // case of `*const T` <=> `*mut T`.
            if let Transmute = kind
                && from.is_raw_ptr()
                && to.is_raw_ptr()
                && self.pointers_have_same_metadata(from, to)
            {
                kind = PtrToPtr;
                was_updated_this_iteration = true;
            }

            // If a cast just casts away the metadata again, then we can get it by
            // casting the original thin pointer passed to `from_raw_parts`
            if let PtrToPtr = kind
                && let Value::RawPtr { pointer, .. } = self.get(value)
                && let ty::RawPtr(to_pointee, _) = to.kind()
                && to_pointee.is_sized(self.tcx, self.typing_env())
            {
                from = self.ty(pointer);
                value = pointer;
                was_updated_this_iteration = true;
                if from == to {
                    return Some(pointer);
                }
            }

            // Aggregate-then-Transmute can just transmute the original field value,
            // so long as the bytes of a value from only from a single field.
            if let Transmute = kind
                && let Value::Aggregate(variant_idx, field_values) = self.get(value)
                && let Some((field_idx, field_ty)) =
                    self.value_is_all_in_one_field(from, variant_idx)
            {
                from = field_ty;
                value = field_values[field_idx.as_usize()];
                was_updated_this_iteration = true;
                if field_ty == to {
                    return Some(value);
                }
            }

            // Various cast-then-cast cases can be simplified.
            if let Value::Cast { kind: inner_kind, value: inner_value } = self.get(value) {
                let inner_from = self.ty(inner_value);
                let new_kind = match (inner_kind, kind) {
                    // Even if there's a narrowing cast in here that's fine, because
                    // things like `*mut [i32] -> *mut i32 -> *const i32` and
                    // `*mut [i32] -> *const [i32] -> *const i32` can skip the middle in MIR.
                    (PtrToPtr, PtrToPtr) => Some(PtrToPtr),
                    // PtrToPtr-then-Transmute is fine so long as the pointer cast is identity:
                    // `*const T -> *mut T -> NonNull<T>` is fine, but we need to check for narrowing
                    // to skip things like `*const [i32] -> *const i32 -> NonNull<T>`.
                    (PtrToPtr, Transmute) if self.pointers_have_same_metadata(inner_from, from) => {
                        Some(Transmute)
                    }
                    // Similarly, for Transmute-then-PtrToPtr. Note that we need to check different
                    // variables for their metadata, and thus this can't merge with the previous arm.
                    (Transmute, PtrToPtr) if self.pointers_have_same_metadata(from, to) => {
                        Some(Transmute)
                    }
                    // If would be legal to always do this, but we don't want to hide information
                    // from the backend that it'd otherwise be able to use for optimizations.
                    (Transmute, Transmute)
                        if !self.type_may_have_niche_of_interest_to_backend(from) =>
                    {
                        Some(Transmute)
                    }
                    _ => None,
                };
                if let Some(new_kind) = new_kind {
                    kind = new_kind;
                    from = inner_from;
                    value = inner_value;
                    was_updated_this_iteration = true;
                    if inner_from == to {
                        return Some(inner_value);
                    }
                }
            }

            if was_updated_this_iteration {
                was_ever_updated = true;
            } else {
                break;
            }
        }

        if was_ever_updated && let Some(op) = self.try_as_operand(value, location) {
            *initial_operand = op;
            *initial_kind = kind;
        }

        Some(self.insert(to, Value::Cast { kind, value }))
    }

    fn pointers_have_same_metadata(&self, left_ptr_ty: Ty<'tcx>, right_ptr_ty: Ty<'tcx>) -> bool {
        let left_meta_ty = left_ptr_ty.pointee_metadata_ty_or_projection(self.tcx);
        let right_meta_ty = right_ptr_ty.pointee_metadata_ty_or_projection(self.tcx);
        if left_meta_ty == right_meta_ty {
            true
        } else if let Ok(left) =
            self.tcx.try_normalize_erasing_regions(self.typing_env(), left_meta_ty)
            && let Ok(right) =
                self.tcx.try_normalize_erasing_regions(self.typing_env(), right_meta_ty)
        {
            left == right
        } else {
            false
        }
    }

    /// Returns `false` if we know for sure that this type has no interesting niche,
    /// and thus we can skip transmuting through it without worrying.
    ///
    /// The backend will emit `assume`s when transmuting between types with niches,
    /// so we want to preserve `i32 -> char -> u32` so that that data is around,
    /// but it's fine to skip whole-range-is-value steps like `A -> u32 -> B`.
    fn type_may_have_niche_of_interest_to_backend(&self, ty: Ty<'tcx>) -> bool {
        let Ok(layout) = self.ecx.layout_of(ty) else {
            // If it's too generic or something, then assume it might be interesting later.
            return true;
        };

        if layout.uninhabited {
            return true;
        }

        match layout.backend_repr {
            BackendRepr::Scalar(a) => !a.is_always_valid(&self.ecx),
            BackendRepr::ScalarPair(a, b) => {
                !a.is_always_valid(&self.ecx) || !b.is_always_valid(&self.ecx)
            }
            BackendRepr::SimdVector { .. } | BackendRepr::Memory { .. } => false,
        }
    }

    fn value_is_all_in_one_field(
        &self,
        ty: Ty<'tcx>,
        variant: VariantIdx,
    ) -> Option<(FieldIdx, Ty<'tcx>)> {
        if let Ok(layout) = self.ecx.layout_of(ty)
            && let abi::Variants::Single { index } = layout.variants
            && index == variant
            && let Some((field_idx, field_layout)) = layout.non_1zst_field(&self.ecx)
            && layout.size == field_layout.size
        {
            // We needed to check the variant to avoid trying to read the tag
            // field from an enum where no fields have variants, since that tag
            // field isn't in the `Aggregate` from which we're getting values.
            Some((field_idx, field_layout.ty))
        } else if let ty::Adt(adt, args) = ty.kind()
            && adt.is_struct()
            && adt.repr().transparent()
            && let [single_field] = adt.non_enum_variant().fields.raw.as_slice()
        {
            Some((FieldIdx::ZERO, single_field.ty(self.tcx, args)))
        } else {
            None
        }
    }
}

fn op_to_prop_const<'tcx>(
    ecx: &mut InterpCx<'tcx, DummyMachine>,
    op: &OpTy<'tcx>,
) -> Option<ConstValue> {
    // Do not attempt to propagate unsized locals.
    if op.layout.is_unsized() {
        return None;
    }

    // This constant is a ZST, just return an empty value.
    if op.layout.is_zst() {
        return Some(ConstValue::ZeroSized);
    }

    // Do not synthetize too large constants. Codegen will just memcpy them, which we'd like to
    // avoid.
    if !matches!(op.layout.backend_repr, BackendRepr::Scalar(..) | BackendRepr::ScalarPair(..)) {
        return None;
    }

    // If this constant has scalar ABI, return it as a `ConstValue::Scalar`.
    if let BackendRepr::Scalar(abi::Scalar::Initialized { .. }) = op.layout.backend_repr
        && let Some(scalar) = ecx.read_scalar(op).discard_err()
    {
        if !scalar.try_to_scalar_int().is_ok() {
            // Check that we do not leak a pointer.
            // Those pointers may lose part of their identity in codegen.
            // FIXME: remove this hack once https://github.com/rust-lang/rust/issues/79738 is fixed.
            return None;
        }
        return Some(ConstValue::Scalar(scalar));
    }

    // If this constant is already represented as an `Allocation`,
    // try putting it into global memory to return it.
    if let Either::Left(mplace) = op.as_mplace_or_imm() {
        let (size, _align) = ecx.size_and_align_of_val(&mplace).discard_err()??;

        // Do not try interning a value that contains provenance.
        // Due to https://github.com/rust-lang/rust/issues/79738, doing so could lead to bugs.
        // FIXME: remove this hack once that issue is fixed.
        let alloc_ref = ecx.get_ptr_alloc(mplace.ptr(), size).discard_err()??;
        if alloc_ref.has_provenance() {
            return None;
        }

        let pointer = mplace.ptr().into_pointer_or_addr().ok()?;
        let (prov, offset) = pointer.prov_and_relative_offset();
        let alloc_id = prov.alloc_id();
        intern_const_alloc_for_constprop(ecx, alloc_id).discard_err()?;

        // `alloc_id` may point to a static. Codegen will choke on an `Indirect` with anything
        // by `GlobalAlloc::Memory`, so do fall through to copying if needed.
        // FIXME: find a way to treat this more uniformly (probably by fixing codegen)
        if let GlobalAlloc::Memory(alloc) = ecx.tcx.global_alloc(alloc_id)
            // Transmuting a constant is just an offset in the allocation. If the alignment of the
            // allocation is not enough, fallback to copying into a properly aligned value.
            && alloc.inner().align >= op.layout.align.abi
        {
            return Some(ConstValue::Indirect { alloc_id, offset });
        }
    }

    // Everything failed: create a new allocation to hold the data.
    let alloc_id =
        ecx.intern_with_temp_alloc(op.layout, |ecx, dest| ecx.copy_op(op, dest)).discard_err()?;
    let value = ConstValue::Indirect { alloc_id, offset: Size::ZERO };

    // Check that we do not leak a pointer.
    // Those pointers may lose part of their identity in codegen.
    // FIXME: remove this hack once https://github.com/rust-lang/rust/issues/79738 is fixed.
    if ecx.tcx.global_alloc(alloc_id).unwrap_memory().inner().provenance().ptrs().is_empty() {
        return Some(value);
    }

    None
}

impl<'tcx> VnState<'_, '_, 'tcx> {
    /// If either [`Self::try_as_constant`] as [`Self::try_as_place`] succeeds,
    /// returns that result as an [`Operand`].
    fn try_as_operand(&mut self, index: VnIndex, location: Location) -> Option<Operand<'tcx>> {
        if let Some(const_) = self.try_as_constant(index) {
            Some(Operand::Constant(Box::new(const_)))
        } else if let Some(place) = self.try_as_place(index, location, false) {
            self.reused_locals.insert(place.local);
            Some(Operand::Copy(place))
        } else {
            None
        }
    }

    /// If `index` is a `Value::Constant`, return the `Constant` to be put in the MIR.
    fn try_as_constant(&mut self, index: VnIndex) -> Option<ConstOperand<'tcx>> {
        // This was already constant in MIR, do not change it. If the constant is not
        // deterministic, adding an additional mention of it in MIR will not give the same value as
        // the former mention.
        if let Value::Constant { value, disambiguator: None } = self.get(index) {
            debug_assert!(value.is_deterministic());
            return Some(ConstOperand { span: DUMMY_SP, user_ty: None, const_: value });
        }

        let op = self.evaluated[index].as_ref()?;
        if op.layout.is_unsized() {
            // Do not attempt to propagate unsized locals.
            return None;
        }

        let value = op_to_prop_const(&mut self.ecx, op)?;

        // Check that we do not leak a pointer.
        // Those pointers may lose part of their identity in codegen.
        // FIXME: remove this hack once https://github.com/rust-lang/rust/issues/79738 is fixed.
        assert!(!value.may_have_provenance(self.tcx, op.layout.size));

        let const_ = Const::Val(value, op.layout.ty);
        Some(ConstOperand { span: DUMMY_SP, user_ty: None, const_ })
    }

    /// Construct a place which holds the same value as `index` and for which all locals strictly
    /// dominate `loc`. If you used this place, add its base local to `reused_locals` to remove
    /// storage statements.
    #[instrument(level = "trace", skip(self), ret)]
    fn try_as_place(
        &mut self,
        mut index: VnIndex,
        loc: Location,
        allow_complex_projection: bool,
    ) -> Option<Place<'tcx>> {
        let mut projection = SmallVec::<[PlaceElem<'tcx>; 1]>::new();
        loop {
            if let Some(local) = self.try_as_local(index, loc) {
                projection.reverse();
                let place =
                    Place { local, projection: self.tcx.mk_place_elems(projection.as_slice()) };
                return Some(place);
            } else if projection.last() == Some(&PlaceElem::Deref) {
                // `Deref` can only be the first projection in a place.
                // If we are here, we failed to find a local, and we already have a `Deref`.
                // Trying to add projections will only result in an ill-formed place.
                return None;
            } else if let Value::Projection(pointer, proj) = self.get(index)
                && (allow_complex_projection || proj.is_stable_offset())
                && let Some(proj) = self.try_as_place_elem(self.ty(index), proj, loc)
            {
                projection.push(proj);
                index = pointer;
            } else {
                return None;
            }
        }
    }

    /// If there is a local which is assigned `index`, and its assignment strictly dominates `loc`,
    /// return it. If you used this local, add it to `reused_locals` to remove storage statements.
    fn try_as_local(&mut self, index: VnIndex, loc: Location) -> Option<Local> {
        let other = self.rev_locals.get(index)?;
        other
            .iter()
            .find(|&&other| self.ssa.assignment_dominates(&self.dominators, other, loc))
            .copied()
    }
}

impl<'tcx> MutVisitor<'tcx> for VnState<'_, '_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        self.simplify_place_projection(place, location);
        if context.is_mutating_use() && place.is_indirect() {
            // Non-local mutation maybe invalidate deref.
            self.invalidate_derefs();
        }
        self.super_place(place, context, location);
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        self.simplify_operand(operand, location);
        self.super_operand(operand, location);
    }

    fn visit_assign(
        &mut self,
        lhs: &mut Place<'tcx>,
        rvalue: &mut Rvalue<'tcx>,
        location: Location,
    ) {
        self.simplify_place_projection(lhs, location);

        let value = self.simplify_rvalue(lhs, rvalue, location);
        if let Some(value) = value {
            if let Some(const_) = self.try_as_constant(value) {
                *rvalue = Rvalue::Use(Operand::Constant(Box::new(const_)));
            } else if let Some(place) = self.try_as_place(value, location, false)
                && *rvalue != Rvalue::Use(Operand::Move(place))
                && *rvalue != Rvalue::Use(Operand::Copy(place))
            {
                *rvalue = Rvalue::Use(Operand::Copy(place));
                self.reused_locals.insert(place.local);
            }
        }

        if lhs.is_indirect() {
            // Non-local mutation maybe invalidate deref.
            self.invalidate_derefs();
        }

        if let Some(local) = lhs.as_local()
            && self.ssa.is_ssa(local)
            && let rvalue_ty = rvalue.ty(self.local_decls, self.tcx)
            // FIXME(#112651) `rvalue` may have a subtype to `local`. We can only mark
            // `local` as reusable if we have an exact type match.
            && self.local_decls[local].ty == rvalue_ty
        {
            let value = value.unwrap_or_else(|| self.new_opaque(rvalue_ty));
            self.assign(local, value);
        }
    }

    fn visit_terminator(&mut self, terminator: &mut Terminator<'tcx>, location: Location) {
        if let Terminator { kind: TerminatorKind::Call { destination, .. }, .. } = terminator {
            if let Some(local) = destination.as_local()
                && self.ssa.is_ssa(local)
            {
                let ty = self.local_decls[local].ty;
                let opaque = self.new_opaque(ty);
                self.assign(local, opaque);
            }
        }
        // Function calls and ASM may invalidate (nested) derefs. We must handle them carefully.
        // Currently, only preserving derefs for trivial terminators like SwitchInt and Goto.
        let safe_to_preserve_derefs = matches!(
            terminator.kind,
            TerminatorKind::SwitchInt { .. } | TerminatorKind::Goto { .. }
        );
        if !safe_to_preserve_derefs {
            self.invalidate_derefs();
        }
        self.super_terminator(terminator, location);
    }
}

struct StorageRemover<'tcx> {
    tcx: TyCtxt<'tcx>,
    reused_locals: DenseBitSet<Local>,
}

impl<'tcx> MutVisitor<'tcx> for StorageRemover<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, _: Location) {
        if let Operand::Move(place) = *operand
            && !place.is_indirect_first_projection()
            && self.reused_locals.contains(place.local)
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
                stmt.make_nop(true)
            }
            _ => self.super_statement(stmt, loc),
        }
    }
}
