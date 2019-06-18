//! This module contains the machinery necessary to detect infinite loops
//! during const-evaluation by taking snapshots of the state of the interpreter
//! at regular intervals.

// This lives in `interpret` because it needs access to all sots of private state.  However,
// it is not used by the general miri engine, just by CTFE.

use std::hash::{Hash, Hasher};

use rustc::ich::StableHashingContextProvider;
use rustc::mir;
use rustc::mir::interpret::{
    AllocId, Pointer, Scalar,
    Relocations, Allocation, UndefMask,
    InterpResult, InterpError,
};

use rustc::ty::{self, TyCtxt};
use rustc::ty::layout::Align;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use syntax::ast::Mutability;
use syntax::source_map::Span;

use super::eval_context::{LocalState, StackPopCleanup};
use super::{Frame, Memory, Operand, MemPlace, Place, Immediate, ScalarMaybeUndef, LocalValue};
use crate::const_eval::CompileTimeInterpreter;

#[derive(Default)]
pub(crate) struct InfiniteLoopDetector<'mir, 'tcx> {
    /// The set of all `InterpSnapshot` *hashes* observed by this detector.
    ///
    /// When a collision occurs in this table, we store the full snapshot in
    /// `snapshots`.
    hashes: FxHashSet<u64>,

    /// The set of all `InterpSnapshot`s observed by this detector.
    ///
    /// An `InterpSnapshot` will only be fully cloned once it has caused a
    /// collision in `hashes`. As a result, the detector must observe at least
    /// *two* full cycles of an infinite loop before it triggers.
    snapshots: FxHashSet<InterpSnapshot<'mir, 'tcx>>,
}

impl<'mir, 'tcx> InfiniteLoopDetector<'mir, 'tcx> {
    pub fn observe_and_analyze(
        &mut self,
        tcx: TyCtxt<'tcx>,
        span: Span,
        memory: &Memory<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>,
        stack: &[Frame<'mir, 'tcx>],
    ) -> InterpResult<'tcx, ()> {
        // Compute stack's hash before copying anything
        let mut hcx = tcx.get_stable_hashing_context();
        let mut hasher = StableHasher::<u64>::new();
        stack.hash_stable(&mut hcx, &mut hasher);
        let hash = hasher.finish();

        // Check if we know that hash already
        if self.hashes.is_empty() {
            // FIXME(#49980): make this warning a lint
            tcx.sess.span_warn(span,
                "Constant evaluating a complex constant, this might take some time");
        }
        if self.hashes.insert(hash) {
            // No collision
            return Ok(())
        }

        // We need to make a full copy. NOW things that to get really expensive.
        info!("snapshotting the state of the interpreter");

        if self.snapshots.insert(InterpSnapshot::new(memory, stack)) {
            // Spurious collision or first cycle
            return Ok(())
        }

        // Second cycle
        Err(InterpError::InfiniteLoop.into())
    }
}

trait SnapshotContext<'a> {
    fn resolve(&'a self, id: &AllocId) -> Option<&'a Allocation>;
}

/// Taking a snapshot of the evaluation context produces a view of
/// the state of the interpreter that is invariant to `AllocId`s.
trait Snapshot<'a, Ctx: SnapshotContext<'a>> {
    type Item;
    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item;
}

macro_rules! __impl_snapshot_field {
    ($field:ident, $ctx:expr) => ($field.snapshot($ctx));
    ($field:ident, $ctx:expr, $delegate:expr) => ($delegate);
}

// This assumes the type has two type parameters, first for the tag (set to `()`),
// then for the id
macro_rules! impl_snapshot_for {
    (enum $enum_name:ident {
        $( $variant:ident $( ( $($field:ident $(-> $delegate:expr)?),* ) )? ),* $(,)?
    }) => {

        impl<'a, Ctx> self::Snapshot<'a, Ctx> for $enum_name
            where Ctx: self::SnapshotContext<'a>,
        {
            type Item = $enum_name<(), AllocIdSnapshot<'a>>;

            #[inline]
            fn snapshot(&self, __ctx: &'a Ctx) -> Self::Item {
                match *self {
                    $(
                        $enum_name::$variant $( ( $(ref $field),* ) )? => {
                            $enum_name::$variant $(
                                ( $( __impl_snapshot_field!($field, __ctx $(, $delegate)?) ),* )
                            )?
                        }
                    )*
                }
            }
        }
    };

    (struct $struct_name:ident { $($field:ident $(-> $delegate:expr)?),*  $(,)? }) => {
        impl<'a, Ctx> self::Snapshot<'a, Ctx> for $struct_name
            where Ctx: self::SnapshotContext<'a>,
        {
            type Item = $struct_name<(), AllocIdSnapshot<'a>>;

            #[inline]
            fn snapshot(&self, __ctx: &'a Ctx) -> Self::Item {
                let $struct_name {
                    $(ref $field),*
                } = *self;

                $struct_name {
                    $( $field: __impl_snapshot_field!($field, __ctx $(, $delegate)?) ),*
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
    offset -> *offset, // just copy offset verbatim
    tag -> *tag, // just copy tag
});

impl<'a, Ctx> Snapshot<'a, Ctx> for Scalar
    where Ctx: SnapshotContext<'a>,
{
    type Item = Scalar<(), AllocIdSnapshot<'a>>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        match self {
            Scalar::Ptr(p) => Scalar::Ptr(p.snapshot(ctx)),
            Scalar::Raw{ size, data } => Scalar::Raw {
                data: *data,
                size: *size,
            },
        }
    }
}

impl_snapshot_for!(enum ScalarMaybeUndef {
    Scalar(s),
    Undef,
});

impl_stable_hash_for!(struct crate::interpret::MemPlace {
    ptr,
    align,
    meta,
});
impl_snapshot_for!(struct MemPlace {
    ptr,
    meta,
    align -> *align, // just copy alignment verbatim
});

impl_stable_hash_for!(enum crate::interpret::Place {
    Ptr(mem_place),
    Local { frame, local },
});
impl<'a, Ctx> Snapshot<'a, Ctx> for Place
    where Ctx: SnapshotContext<'a>,
{
    type Item = Place<(), AllocIdSnapshot<'a>>;

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

impl_stable_hash_for!(enum crate::interpret::Immediate {
    Scalar(x),
    ScalarPair(x, y),
});
impl_snapshot_for!(enum Immediate {
    Scalar(s),
    ScalarPair(s, t),
});

impl_stable_hash_for!(enum crate::interpret::Operand {
    Immediate(x),
    Indirect(x),
});
impl_snapshot_for!(enum Operand {
    Immediate(v),
    Indirect(m),
});

impl_stable_hash_for!(enum crate::interpret::LocalValue {
    Dead,
    Uninitialized,
    Live(x),
});
impl_snapshot_for!(enum LocalValue {
    Dead,
    Uninitialized,
    Live(v),
});

impl<'a, Ctx> Snapshot<'a, Ctx> for Relocations
    where Ctx: SnapshotContext<'a>,
{
    type Item = Relocations<(), AllocIdSnapshot<'a>>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        Relocations::from_presorted(self.iter()
            .map(|(size, ((), id))| (*size, ((), id.snapshot(ctx))))
            .collect())
    }
}

#[derive(Eq, PartialEq)]
struct AllocationSnapshot<'a> {
    bytes: &'a [u8],
    relocations: Relocations<(), AllocIdSnapshot<'a>>,
    undef_mask: &'a UndefMask,
    align: &'a Align,
    mutability: &'a Mutability,
}

impl<'a, Ctx> Snapshot<'a, Ctx> for &'a Allocation
    where Ctx: SnapshotContext<'a>,
{
    type Item = AllocationSnapshot<'a>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        let Allocation { bytes, relocations, undef_mask, align, mutability, extra: () } = self;

        AllocationSnapshot {
            bytes,
            undef_mask,
            align,
            mutability,
            relocations: relocations.snapshot(ctx),
        }
    }
}

impl_stable_hash_for!(enum crate::interpret::eval_context::StackPopCleanup {
    Goto(block),
    None { cleanup },
});

#[derive(Eq, PartialEq)]
struct FrameSnapshot<'a, 'tcx> {
    instance: &'a ty::Instance<'tcx>,
    span: &'a Span,
    return_to_block: &'a StackPopCleanup,
    return_place: Option<Place<(), AllocIdSnapshot<'a>>>,
    locals: IndexVec<mir::Local, LocalValue<(), AllocIdSnapshot<'a>>>,
    block: &'a mir::BasicBlock,
    stmt: usize,
}

impl_stable_hash_for!(impl<> for struct Frame<'mir, 'tcx> {
    body,
    instance,
    span,
    return_to_block,
    return_place -> (return_place.as_ref().map(|r| &**r)),
    locals,
    block,
    stmt,
    extra,
});

impl<'a, 'mir, 'tcx, Ctx> Snapshot<'a, Ctx> for &'a Frame<'mir, 'tcx>
    where Ctx: SnapshotContext<'a>,
{
    type Item = FrameSnapshot<'a, 'tcx>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        let Frame {
            body: _,
            instance,
            span,
            return_to_block,
            return_place,
            locals,
            block,
            stmt,
            extra: _,
        } = self;

        FrameSnapshot {
            instance,
            span,
            return_to_block,
            block,
            stmt: *stmt,
            return_place: return_place.map(|r| r.snapshot(ctx)),
            locals: locals.iter().map(|local| local.snapshot(ctx)).collect(),
        }
    }
}

impl<'a, 'tcx, Ctx> Snapshot<'a, Ctx> for &'a LocalState<'tcx>
    where Ctx: SnapshotContext<'a>,
{
    type Item = LocalValue<(), AllocIdSnapshot<'a>>;

    fn snapshot(&self, ctx: &'a Ctx) -> Self::Item {
        let LocalState { value, layout: _ } = self;
        value.snapshot(ctx)
    }
}

impl_stable_hash_for!(struct LocalState<'tcx> {
    value,
    layout -> _,
});

impl<'b, 'mir, 'tcx> SnapshotContext<'b>
    for Memory<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>
{
    fn resolve(&'b self, id: &AllocId) -> Option<&'b Allocation> {
        self.get(*id).ok()
    }
}

/// The virtual machine state during const-evaluation at a given point in time.
/// We assume the `CompileTimeInterpreter` has no interesting extra state that
/// is worth considering here.
struct InterpSnapshot<'mir, 'tcx> {
    memory: Memory<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>,
    stack: Vec<Frame<'mir, 'tcx>>,
}

impl InterpSnapshot<'mir, 'tcx> {
    fn new(
        memory: &Memory<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>,
        stack: &[Frame<'mir, 'tcx>],
    ) -> Self {
        InterpSnapshot {
            memory: memory.clone(),
            stack: stack.into(),
        }
    }

    // Used to compare two snapshots
    fn snapshot(&'b self)
        -> Vec<FrameSnapshot<'b, 'tcx>>
    {
        // Start with the stack, iterate and recursively snapshot
        self.stack.iter().map(|frame| frame.snapshot(&self.memory)).collect()
    }
}

impl<'mir, 'tcx> Hash for InterpSnapshot<'mir, 'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Implement in terms of hash stable, so that k1 == k2 -> hash(k1) == hash(k2)
        let mut hcx = self.memory.tcx.get_stable_hashing_context();
        let mut hasher = StableHasher::<u64>::new();
        self.hash_stable(&mut hcx, &mut hasher);
        hasher.finish().hash(state)
    }
}

impl_stable_hash_for!(impl<> for struct InterpSnapshot<'mir, 'tcx> {
    // Not hashing memory: Avoid hashing memory all the time during execution
    memory -> _,
    stack,
});

impl<'mir, 'tcx> Eq for InterpSnapshot<'mir, 'tcx> {}

impl<'mir, 'tcx> PartialEq for InterpSnapshot<'mir, 'tcx> {
    fn eq(&self, other: &Self) -> bool {
        // FIXME: This looks to be a *ridiculously expensive* comparison operation.
        // Doesn't this make tons of copies?  Either `snapshot` is very badly named,
        // or it does!
        self.snapshot() == other.snapshot()
    }
}
