use std::hash::{Hash, Hasher};

use rustc::ich::{StableHashingContext, StableHashingContextProvider};
use rustc::mir;
use rustc::mir::interpret::{
    AllocId, Pointer, Scalar, ScalarMaybeUndef,
    Relocations, Allocation, UndefMask,
    EvalResult, EvalErrorKind,
};

use rustc::ty::{self, TyCtxt};
use rustc::ty::layout::Align;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableHasherResult};
use syntax::ast::Mutability;
use syntax::source_map::Span;

use super::eval_context::{LocalValue, StackPopCleanup};
use super::{Frame, Memory, Machine, Operand, MemPlace, Place, Value};

pub(super) struct InfiniteLoopDetector<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    /// The set of all `EvalSnapshot` *hashes* observed by this detector.
    ///
    /// When a collision occurs in this table, we store the full snapshot in
    /// `snapshots`.
    hashes: FxHashSet<u64>,

    /// The set of all `EvalSnapshot`s observed by this detector.
    ///
    /// An `EvalSnapshot` will only be fully cloned once it has caused a
    /// collision in `hashes`. As a result, the detector must observe at least
    /// *two* full cycles of an infinite loop before it triggers.
    snapshots: FxHashSet<EvalSnapshot<'a, 'mir, 'tcx, M>>,
}

impl<'a, 'mir, 'tcx, M> Default for InfiniteLoopDetector<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
          'tcx: 'a + 'mir,
{
    fn default() -> Self {
        InfiniteLoopDetector {
            hashes: FxHashSet::default(),
            snapshots: FxHashSet::default(),
        }
    }
}

impl<'a, 'mir, 'tcx, M> InfiniteLoopDetector<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
          'tcx: 'a + 'mir,
{
    /// Returns `true` if the loop detector has not yet observed a snapshot.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    pub fn observe_and_analyze(
        &mut self,
        tcx: &TyCtxt<'b, 'tcx, 'tcx>,
        machine: &M,
        memory: &Memory<'a, 'mir, 'tcx, M>,
        stack: &[Frame<'mir, 'tcx>],
    ) -> EvalResult<'tcx, ()> {

        let mut hcx = tcx.get_stable_hashing_context();
        let mut hasher = StableHasher::<u64>::new();
        (machine, stack).hash_stable(&mut hcx, &mut hasher);
        let hash = hasher.finish();

        if self.hashes.insert(hash) {
            // No collision
            return Ok(())
        }

        info!("snapshotting the state of the interpreter");

        if self.snapshots.insert(EvalSnapshot::new(machine, memory, stack)) {
            // Spurious collision or first cycle
            return Ok(())
        }

        // Second cycle
        Err(EvalErrorKind::InfiniteLoop.into())
    }
}

trait SnapshotContext<'a> {
    fn resolve(&'a self, id: &AllocId) -> Option<&'a Allocation>;
}

trait Snapshot<'a, Ctx: SnapshotContext<'a>> {
    type Item;
    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item;
}

macro_rules! __impl_snapshot_field {
    ($field:ident, $ctx:expr) => ($field.snapshot($ctx));
    ($field:ident, $ctx:expr, $delegate:expr) => ($delegate);
}

macro_rules! impl_snapshot_for {
    // FIXME(mark-i-m): Some of these should be `?` rather than `*`.
    (enum $enum_name:ident {
        $( $variant:ident $( ( $($field:ident $(-> $delegate:expr)*),* ) )* ),* $(,)*
    }) => {

        impl<'a, Ctx> self::Snapshot<'a, Ctx> for $enum_name
            where Ctx: self::SnapshotContext<'a>,
        {
            type Item = $enum_name<AllocIdSnapshot<'a>>;

            #[inline]
            fn snapshot(&self, __ctx: &'a Ctx) -> Self::Item {
                match *self {
                    $(
                        $enum_name::$variant $( ( $(ref $field),* ) )* =>
                            $enum_name::$variant $(
                                ( $( __impl_snapshot_field!($field, __ctx $(, $delegate)*) ),* ),
                            )*
                    )*
                }
            }
        }
    };

    // FIXME(mark-i-m): same here.
    (struct $struct_name:ident { $($field:ident $(-> $delegate:expr)*),*  $(,)* }) => {
        impl<'a, Ctx> self::Snapshot<'a, Ctx> for $struct_name
            where Ctx: self::SnapshotContext<'a>,
        {
            type Item = $struct_name<AllocIdSnapshot<'a>>;

            #[inline]
            fn snapshot(&self, __ctx: &'a Ctx) -> Self::Item {
                let $struct_name {
                    $(ref $field),*
                } = *self;

                $struct_name {
                    $( $field: __impl_snapshot_field!($field, __ctx $(, $delegate)*) ),*
                }
            }
        }
    };
}

impl<'a, Ctx, T> Snapshot<'a, Ctx> for Option<T>
    where Ctx: SnapshotContext<'a>,
          T: Snapshot<'a, Ctx>
{
    type Item = Option<<T as Snapshot<'a, Ctx>>::Item>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            Some(x) => Some(x.snapshot(ctx)),
            None => None,
        }
    }
}

#[derive(Eq, PartialEq)]
struct AllocIdSnapshot<'a>(Option<AllocationSnapshot<'a>>);

impl<'a, Ctx> Snapshot<'a, Ctx> for AllocId
    where Ctx: SnapshotContext<'a>,
{
    type Item = AllocIdSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        AllocIdSnapshot(ctx.resolve(self).map(|alloc| alloc.snapshot(ctx)))
    }
}

impl_snapshot_for!(struct Pointer {
    alloc_id,
    offset -> *offset,
});

impl<'a, Ctx> Snapshot<'a, Ctx> for Scalar
    where Ctx: SnapshotContext<'a>,
{
    type Item = Scalar<AllocIdSnapshot<'a>>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            Scalar::Ptr(p) => Scalar::Ptr(p.snapshot(ctx)),
            Scalar::Bits{ size, bits } => Scalar::Bits {
                size: *size,
                bits: *bits,
            },
        }
    }
}

impl_snapshot_for!(enum ScalarMaybeUndef {
    Scalar(s),
    Undef,
});

impl_snapshot_for!(struct MemPlace {
    ptr,
    extra,
    align -> *align,
});

impl<'a, Ctx> Snapshot<'a, Ctx> for Place
    where Ctx: SnapshotContext<'a>,
{
    type Item = Place<AllocIdSnapshot<'a>>;

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

impl_snapshot_for!(enum Value {
    Scalar(s),
    ScalarPair(s, t),
});

impl_snapshot_for!(enum Operand {
    Immediate(v),
    Indirect(m),
});

impl_snapshot_for!(enum LocalValue {
    Live(v),
    Dead,
});

impl<'a, Ctx> Snapshot<'a, Ctx> for Relocations
    where Ctx: SnapshotContext<'a>,
{
    type Item = Relocations<AllocIdSnapshot<'a>>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        Relocations::from_presorted(self.iter()
            .map(|(size, id)| (*size, id.snapshot(ctx)))
            .collect())
    }
}

#[derive(Eq, PartialEq)]
struct AllocationSnapshot<'a> {
    bytes: &'a [u8],
    relocations: Relocations<AllocIdSnapshot<'a>>,
    undef_mask: &'a UndefMask,
    align: &'a Align,
    mutability: &'a Mutability,
}

impl<'a, Ctx> Snapshot<'a, Ctx> for &'a Allocation
    where Ctx: SnapshotContext<'a>,
{
    type Item = AllocationSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        let Allocation { bytes, relocations, undef_mask, align, mutability } = self;

        AllocationSnapshot {
            bytes,
            undef_mask,
            align,
            mutability,
            relocations: relocations.snapshot(ctx),
        }
    }
}

#[derive(Eq, PartialEq)]
struct FrameSnapshot<'a, 'tcx: 'a> {
    instance: &'a ty::Instance<'tcx>,
    span: &'a Span,
    return_to_block: &'a StackPopCleanup,
    return_place: Place<AllocIdSnapshot<'a>>,
    locals: IndexVec<mir::Local, LocalValue<AllocIdSnapshot<'a>>>,
    block: &'a mir::BasicBlock,
    stmt: usize,
}

impl<'a, 'mir, 'tcx, Ctx> Snapshot<'a, Ctx> for &'a Frame<'mir, 'tcx>
    where Ctx: SnapshotContext<'a>,
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

impl<'a, 'mir, 'tcx, M> Memory<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn snapshot<'b: 'a>(&'b self) -> MemorySnapshot<'b, 'mir, 'tcx, M> {
        let Memory { data, .. } = self;
        MemorySnapshot { data }
    }
}

impl<'a, 'b, 'mir, 'tcx, M> SnapshotContext<'b> for Memory<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn resolve(&'b self, id: &AllocId) -> Option<&'b Allocation> {
        self.get(*id).ok()
    }
}

/// The virtual machine state during const-evaluation at a given point in time.
struct EvalSnapshot<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    machine: M,
    memory: Memory<'a, 'mir, 'tcx, M>,
    stack: Vec<Frame<'mir, 'tcx>>,
}

impl<'a, 'mir, 'tcx, M> EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn new(
        machine: &M,
        memory: &Memory<'a, 'mir, 'tcx, M>,
        stack: &[Frame<'mir, 'tcx>]) -> Self {

        EvalSnapshot {
            machine: machine.clone(),
            memory: memory.clone(),
            stack: stack.into(),
        }
    }

    fn snapshot<'b: 'a>(&'b self)
        -> (&'b M, MemorySnapshot<'b, 'mir, 'tcx, M>, Vec<FrameSnapshot<'a, 'tcx>>) {
        let EvalSnapshot{ machine, memory, stack } = self;
        (&machine, memory.snapshot(), stack.iter().map(|frame| frame.snapshot(memory)).collect())
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

impl<'a, 'b, 'mir, 'tcx, M> HashStable<StableHashingContext<'b>>
    for EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn hash_stable<W: StableHasherResult>(
        &self,
        hcx: &mut StableHashingContext<'b>,
        hasher: &mut StableHasher<W>) {

        let EvalSnapshot{ machine, memory, stack } = self;
        (machine, &memory.data, stack).hash_stable(hcx, hasher);
    }
}

impl<'a, 'mir, 'tcx, M> Eq for EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{}

impl<'a, 'mir, 'tcx, M> PartialEq for EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn eq(&self, other: &Self) -> bool {
        self.snapshot() == other.snapshot()
    }
}
