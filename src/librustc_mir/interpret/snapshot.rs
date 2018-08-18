use std::hash::{Hash, Hasher};

use rustc::ich::{StableHashingContext, StableHashingContextProvider};
use rustc::mir;
use rustc::mir::interpret::{AllocId, Pointer, Scalar, ScalarMaybeUndef, Relocations, Allocation, UndefMask};
use rustc::ty;
use rustc::ty::layout::Align;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableHasherResult};
use syntax::ast::Mutability;
use syntax::source_map::Span;

use super::eval_context::{LocalValue, StackPopCleanup};
use super::{Frame, Memory, Machine, Operand, MemPlace, Place, PlaceExtra, Value};

trait SnapshotContext<'a> {
    type To;
    type From;
    fn resolve(&'a self, id: &Self::From) -> Option<&'a Self::To>;
}

trait Snapshot<'a, Ctx: SnapshotContext<'a>> {
    type Item;
    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item;
}

#[derive(Eq, PartialEq)]
struct AllocIdSnapshot<'a>(Option<AllocationSnapshot<'a>>);

impl<'a, Ctx> Snapshot<'a, Ctx> for AllocId
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = AllocIdSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        AllocIdSnapshot(ctx.resolve(self).map(|alloc| alloc.snapshot(ctx)))
    }
}

type PointerSnapshot<'a> = Pointer<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for Pointer
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = PointerSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        let Pointer{ alloc_id, offset } = self;

        Pointer {
            alloc_id: alloc_id.snapshot(ctx),
            offset: *offset,
        }
    }
}

type ScalarSnapshot<'a> = Scalar<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for Scalar
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = ScalarSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            Scalar::Ptr(p) => Scalar::Ptr(p.snapshot(ctx)),
            Scalar::Bits{ size, bits } => Scalar::Bits{
                size: *size,
                bits: *bits,
            },
        }
    }
}

type ScalarMaybeUndefSnapshot<'a> = ScalarMaybeUndef<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for ScalarMaybeUndef
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = ScalarMaybeUndefSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            ScalarMaybeUndef::Scalar(s) => ScalarMaybeUndef::Scalar(s.snapshot(ctx)),
            ScalarMaybeUndef::Undef => ScalarMaybeUndef::Undef,
        }
    }
}

type MemPlaceSnapshot<'a> = MemPlace<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for MemPlace
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = MemPlaceSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        let MemPlace{ ptr, extra, align } = self;

        MemPlaceSnapshot{
            ptr: ptr.snapshot(ctx),
            extra: extra.snapshot(ctx),
            align: *align,
        }
    }
}

type PlaceSnapshot<'a> = Place<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for Place
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = PlaceSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            Place::Ptr(p) => Place::Ptr(p.snapshot(ctx)),

            Place::Local{ frame, local } => Place::Local{
                frame: *frame,
                local: *local,
            },
        }
    }
}

type PlaceExtraSnapshot<'a> = PlaceExtra<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for PlaceExtra
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = PlaceExtraSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            PlaceExtra::Vtable(p) => PlaceExtra::Vtable(p.snapshot(ctx)),
            PlaceExtra::Length(l) => PlaceExtra::Length(*l),
            PlaceExtra::None => PlaceExtra::None,
        }
    }
}

type ValueSnapshot<'a> = Value<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for Value
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = ValueSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            Value::Scalar(s) => Value::Scalar(s.snapshot(ctx)),
            Value::ScalarPair(a, b) => Value::ScalarPair(a.snapshot(ctx), b.snapshot(ctx)),
        }
    }
}

type OperandSnapshot<'a> = Operand<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for Operand
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = OperandSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            Operand::Immediate(v) => Operand::Immediate(v.snapshot(ctx)),
            Operand::Indirect(m) => Operand::Indirect(m.snapshot(ctx)),
        }
    }
}

type LocalValueSnapshot<'a> = LocalValue<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for LocalValue
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = LocalValueSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            LocalValue::Live(v) => LocalValue::Live(v.snapshot(ctx)),
            LocalValue::Dead => LocalValue::Dead,
        }
    }
}

type RelocationsSnapshot<'a> = Relocations<AllocIdSnapshot<'a>>;

impl<'a, Ctx> Snapshot<'a, Ctx> for Relocations
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = RelocationsSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        Relocations::from_presorted(self.iter().map(|(size, id)| (*size, id.snapshot(ctx))).collect())
    }
}

#[derive(Eq, PartialEq)]
struct AllocationSnapshot<'a> {
    bytes: &'a [u8],
    relocations: RelocationsSnapshot<'a>,
    undef_mask: &'a UndefMask,
    align: &'a Align,
    runtime_mutability: &'a Mutability,
}

impl<'a, Ctx> Snapshot<'a, Ctx> for &'a Allocation
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = AllocationSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        let Allocation { bytes, relocations, undef_mask, align, runtime_mutability } = self;

        AllocationSnapshot {
            bytes,
            undef_mask,
            align,
            runtime_mutability,
            relocations: relocations.snapshot(ctx),
        }
    }
}

#[derive(Eq, PartialEq)]
struct FrameSnapshot<'a, 'tcx> {
    instance: &'a ty::Instance<'tcx>,
    span: &'a Span,
    return_to_block: &'a StackPopCleanup,
    return_place: PlaceSnapshot<'a>,
    locals: IndexVec<mir::Local, LocalValueSnapshot<'a>>,
    block: &'a mir::BasicBlock,
    stmt: usize,
}

impl<'a, 'mir, 'tcx, Ctx> Snapshot<'a, Ctx> for &'a Frame<'mir, 'tcx>
    where Ctx: SnapshotContext<'a, To=Allocation, From=AllocId>,
{
    type Item = FrameSnapshot<'a, 'tcx>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        let Frame {
            mir: _,
            instance,
            span,
            return_to_block,
            return_place,
            locals,
            block,
            stmt,
        } = self;

        FrameSnapshot {
            instance,
            span,
            return_to_block,
            block,
            stmt: *stmt,
            return_place: return_place.snapshot(ctx),
            locals: locals.iter().map(|local| local.snapshot(ctx)).collect(),
        }
    }
}

#[derive(Eq, PartialEq)]
struct MemorySnapshot<'a, 'mir: 'a, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx> + 'a> {
    data: &'a M::MemoryData,
}

/// The virtual machine state during const-evaluation at a given point in time.
#[derive(Eq, PartialEq)]
pub struct EvalSnapshot<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    machine: M,
    memory: Memory<'a, 'mir, 'tcx, M>,
    stack: Vec<Frame<'mir, 'tcx>>,
}

impl<'a, 'mir, 'tcx, M> EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    pub fn new(machine: &M, memory: &Memory<'a, 'mir, 'tcx, M>, stack: &[Frame<'mir, 'tcx>]) -> Self {
        EvalSnapshot {
            machine: machine.clone(),
            memory: memory.clone(),
            stack: stack.into(),
        }
    }
}

impl<'a, 'mir, 'tcx, M> Hash for EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Implement in terms of hash stable, so that k1 == k2 -> hash(k1) == hash(k2)
        let mut hcx = self.memory.tcx.get_stable_hashing_context();
        let mut hasher = StableHasher::<u64>::new();
        self.hash_stable(&mut hcx, &mut hasher);
        hasher.finish().hash(state)
    }
}

impl<'a, 'b, 'mir, 'tcx, M> HashStable<StableHashingContext<'b>> for EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn hash_stable<W: StableHasherResult>(&self, hcx: &mut StableHashingContext<'b>, hasher: &mut StableHasher<W>) {
        let EvalSnapshot{ machine, memory, stack } = self;
        (machine, &memory.data, stack).hash_stable(hcx, hasher);
    }
}
