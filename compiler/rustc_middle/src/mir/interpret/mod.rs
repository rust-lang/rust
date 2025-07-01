//! An interpreter for MIR used in CTFE and by miri.

#[macro_use]
mod error;

mod allocation;
mod pointer;
mod queries;
mod value;

use std::io::{Read, Write};
use std::num::NonZero;
use std::{fmt, io};

use rustc_abi::{AddressSpace, Align, Endian, HasDataLayout, Size};
use rustc_ast::{LitKind, Mutability};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sharded::ShardedHashMap;
use rustc_data_structures::sync::{AtomicU64, Lock};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_serialize::{Decodable, Encodable};
use tracing::{debug, trace};
// Also make the error macros available from this module.
pub use {
    err_exhaust, err_inval, err_machine_stop, err_ub, err_ub_custom, err_ub_format, err_unsup,
    err_unsup_format, throw_exhaust, throw_inval, throw_machine_stop, throw_ub, throw_ub_custom,
    throw_ub_format, throw_unsup, throw_unsup_format,
};

pub use self::allocation::{
    AllocBytes, AllocError, AllocInit, AllocRange, AllocResult, Allocation, ConstAllocation,
    InitChunk, InitChunkIter, alloc_range,
};
pub use self::error::{
    BadBytesAccess, CheckAlignMsg, CheckInAllocMsg, ErrorHandled, EvalStaticInitializerRawResult,
    EvalToAllocationRawResult, EvalToConstValueResult, EvalToValTreeResult, ExpectedKind,
    InterpErrorInfo, InterpErrorKind, InterpResult, InvalidMetaKind, InvalidProgramInfo,
    MachineStopType, Misalignment, PointerKind, ReportedErrorInfo, ResourceExhaustionInfo,
    ScalarSizeMismatch, UndefinedBehaviorInfo, UnsupportedOpInfo, ValTreeCreationError,
    ValidationErrorInfo, ValidationErrorKind, interp_ok,
};
pub use self::pointer::{CtfeProvenance, Pointer, PointerArithmetic, Provenance};
pub use self::value::Scalar;
use crate::mir;
use crate::ty::codec::{TyDecoder, TyEncoder};
use crate::ty::print::with_no_trimmed_paths;
use crate::ty::{self, Instance, Ty, TyCtxt};

/// Uniquely identifies one of the following:
/// - A constant
/// - A static
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct GlobalId<'tcx> {
    /// For a constant or static, the `Instance` of the item itself.
    /// For a promoted global, the `Instance` of the function they belong to.
    pub instance: ty::Instance<'tcx>,

    /// The index for promoted globals within their function's `mir::Body`.
    pub promoted: Option<mir::Promoted>,
}

impl<'tcx> GlobalId<'tcx> {
    pub fn display(self, tcx: TyCtxt<'tcx>) -> String {
        let instance_name = with_no_trimmed_paths!(tcx.def_path_str(self.instance.def.def_id()));
        if let Some(promoted) = self.promoted {
            format!("{instance_name}::{promoted:?}")
        } else {
            instance_name
        }
    }
}

/// Input argument for `tcx.lit_to_const`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, HashStable)]
pub struct LitToConstInput<'tcx> {
    /// The absolute value of the resultant constant.
    pub lit: LitKind,
    /// The type of the constant.
    pub ty: Ty<'tcx>,
    /// If the constant is negative.
    pub neg: bool,
}

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocId(pub NonZero<u64>);

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
impl fmt::Debug for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() { write!(f, "a{}", self.0) } else { write!(f, "alloc{}", self.0) }
    }
}

// No "Display" since AllocIds are not usually user-visible.

#[derive(TyDecodable, TyEncodable)]
enum AllocDiscriminant {
    Alloc,
    Fn,
    VTable,
    Static,
}

pub fn specialized_encode_alloc_id<'tcx, E: TyEncoder<'tcx>>(
    encoder: &mut E,
    tcx: TyCtxt<'tcx>,
    alloc_id: AllocId,
) {
    match tcx.global_alloc(alloc_id) {
        GlobalAlloc::Memory(alloc) => {
            trace!("encoding {:?} with {:#?}", alloc_id, alloc);
            AllocDiscriminant::Alloc.encode(encoder);
            alloc.encode(encoder);
        }
        GlobalAlloc::Function { instance } => {
            trace!("encoding {:?} with {:#?}", alloc_id, instance);
            AllocDiscriminant::Fn.encode(encoder);
            instance.encode(encoder);
        }
        GlobalAlloc::VTable(ty, poly_trait_ref) => {
            trace!("encoding {:?} with {ty:#?}, {poly_trait_ref:#?}", alloc_id);
            AllocDiscriminant::VTable.encode(encoder);
            ty.encode(encoder);
            poly_trait_ref.encode(encoder);
        }
        GlobalAlloc::Static(did) => {
            assert!(!tcx.is_thread_local_static(did));
            // References to statics doesn't need to know about their allocations,
            // just about its `DefId`.
            AllocDiscriminant::Static.encode(encoder);
            // Cannot use `did.encode(encoder)` because of a bug around
            // specializations and method calls.
            Encodable::<E>::encode(&did, encoder);
        }
    }
}

#[derive(Clone)]
enum State {
    Empty,
    Done(AllocId),
}

pub struct AllocDecodingState {
    // For each `AllocId`, we keep track of which decoding state it's currently in.
    decoding_state: Vec<Lock<State>>,
    // The offsets of each allocation in the data stream.
    data_offsets: Vec<u64>,
}

impl AllocDecodingState {
    #[inline]
    pub fn new_decoding_session(&self) -> AllocDecodingSession<'_> {
        AllocDecodingSession { state: self }
    }

    pub fn new(data_offsets: Vec<u64>) -> Self {
        let decoding_state =
            std::iter::repeat_with(|| Lock::new(State::Empty)).take(data_offsets.len()).collect();

        Self { decoding_state, data_offsets }
    }
}

#[derive(Copy, Clone)]
pub struct AllocDecodingSession<'s> {
    state: &'s AllocDecodingState,
}

impl<'s> AllocDecodingSession<'s> {
    /// Decodes an `AllocId` in a thread-safe way.
    pub fn decode_alloc_id<'tcx, D>(&self, decoder: &mut D) -> AllocId
    where
        D: TyDecoder<'tcx>,
    {
        // Read the index of the allocation.
        let idx = usize::try_from(decoder.read_u32()).unwrap();
        let pos = usize::try_from(self.state.data_offsets[idx]).unwrap();

        // Decode the `AllocDiscriminant` now so that we know if we have to reserve an
        // `AllocId`.
        let (alloc_kind, pos) = decoder.with_position(pos, |decoder| {
            let alloc_kind = AllocDiscriminant::decode(decoder);
            (alloc_kind, decoder.position())
        });

        // We are going to hold this lock during the entire decoding of this allocation, which may
        // require that we decode other allocations. This cannot deadlock for two reasons:
        //
        // At the time of writing, it is only possible to create an allocation that contains a pointer
        // to itself using the const_allocate intrinsic (which is for testing only), and even attempting
        // to evaluate such consts blows the stack. If we ever grow a mechanism for producing
        // cyclic allocations, we will need a new strategy for decoding that doesn't bring back
        // https://github.com/rust-lang/rust/issues/126741.
        //
        // It is also impossible to create two allocations (call them A and B) where A is a pointer to B, and B
        // is a pointer to A, because attempting to evaluate either of those consts will produce a
        // query cycle, failing compilation.
        let mut entry = self.state.decoding_state[idx].lock();
        // Check the decoding state to see if it's already decoded or if we should
        // decode it here.
        if let State::Done(alloc_id) = *entry {
            return alloc_id;
        }

        // Now decode the actual data.
        let alloc_id = decoder.with_position(pos, |decoder| match alloc_kind {
            AllocDiscriminant::Alloc => {
                trace!("creating memory alloc ID");
                let alloc = <ConstAllocation<'tcx> as Decodable<_>>::decode(decoder);
                trace!("decoded alloc {:?}", alloc);
                decoder.interner().reserve_and_set_memory_alloc(alloc)
            }
            AllocDiscriminant::Fn => {
                trace!("creating fn alloc ID");
                let instance = ty::Instance::decode(decoder);
                trace!("decoded fn alloc instance: {:?}", instance);
                decoder.interner().reserve_and_set_fn_alloc(instance, CTFE_ALLOC_SALT)
            }
            AllocDiscriminant::VTable => {
                trace!("creating vtable alloc ID");
                let ty = Decodable::decode(decoder);
                let poly_trait_ref = Decodable::decode(decoder);
                trace!("decoded vtable alloc instance: {ty:?}, {poly_trait_ref:?}");
                decoder.interner().reserve_and_set_vtable_alloc(ty, poly_trait_ref, CTFE_ALLOC_SALT)
            }
            AllocDiscriminant::Static => {
                trace!("creating extern static alloc ID");
                let did = <DefId as Decodable<D>>::decode(decoder);
                trace!("decoded static def-ID: {:?}", did);
                decoder.interner().reserve_and_set_static_alloc(did)
            }
        });

        *entry = State::Done(alloc_id);

        alloc_id
    }
}

/// An allocation in the global (tcx-managed) memory can be either a function pointer,
/// a static, or a "real" allocation with some data in it.
#[derive(Debug, Clone, Eq, PartialEq, Hash, TyDecodable, TyEncodable, HashStable)]
pub enum GlobalAlloc<'tcx> {
    /// The alloc ID is used as a function pointer.
    Function { instance: Instance<'tcx> },
    /// This alloc ID points to a symbolic (not-reified) vtable.
    /// We remember the full dyn type, not just the principal trait, so that
    /// const-eval and Miri can detect UB due to invalid transmutes of
    /// `dyn Trait` types.
    VTable(Ty<'tcx>, &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>),
    /// The alloc ID points to a "lazy" static variable that did not get computed (yet).
    /// This is also used to break the cycle in recursive statics.
    Static(DefId),
    /// The alloc ID points to memory.
    Memory(ConstAllocation<'tcx>),
}

impl<'tcx> GlobalAlloc<'tcx> {
    /// Panics if the `GlobalAlloc` does not refer to an `GlobalAlloc::Memory`
    #[track_caller]
    #[inline]
    pub fn unwrap_memory(&self) -> ConstAllocation<'tcx> {
        match *self {
            GlobalAlloc::Memory(mem) => mem,
            _ => bug!("expected memory, got {:?}", self),
        }
    }

    /// Panics if the `GlobalAlloc` is not `GlobalAlloc::Function`
    #[track_caller]
    #[inline]
    pub fn unwrap_fn(&self) -> Instance<'tcx> {
        match *self {
            GlobalAlloc::Function { instance, .. } => instance,
            _ => bug!("expected function, got {:?}", self),
        }
    }

    /// Panics if the `GlobalAlloc` is not `GlobalAlloc::VTable`
    #[track_caller]
    #[inline]
    pub fn unwrap_vtable(&self) -> (Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>) {
        match *self {
            GlobalAlloc::VTable(ty, dyn_ty) => (ty, dyn_ty.principal()),
            _ => bug!("expected vtable, got {:?}", self),
        }
    }

    /// The address space that this `GlobalAlloc` should be placed in.
    #[inline]
    pub fn address_space(&self, cx: &impl HasDataLayout) -> AddressSpace {
        match self {
            GlobalAlloc::Function { .. } => cx.data_layout().instruction_address_space,
            GlobalAlloc::Static(..) | GlobalAlloc::Memory(..) | GlobalAlloc::VTable(..) => {
                AddressSpace::DATA
            }
        }
    }

    pub fn mutability(&self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> Mutability {
        // Let's see what kind of memory we are.
        match self {
            GlobalAlloc::Static(did) => {
                let DefKind::Static { safety: _, mutability, nested } = tcx.def_kind(did) else {
                    bug!()
                };
                if nested {
                    // Nested statics in a `static` are never interior mutable,
                    // so just use the declared mutability.
                    if cfg!(debug_assertions) {
                        let alloc = tcx.eval_static_initializer(did).unwrap();
                        assert_eq!(alloc.0.mutability, mutability);
                    }
                    mutability
                } else {
                    let mutability = match mutability {
                        Mutability::Not
                            if !tcx
                                .type_of(did)
                                .no_bound_vars()
                                .expect("statics should not have generic parameters")
                                .is_freeze(tcx, typing_env) =>
                        {
                            Mutability::Mut
                        }
                        _ => mutability,
                    };
                    mutability
                }
            }
            GlobalAlloc::Memory(alloc) => alloc.inner().mutability,
            GlobalAlloc::Function { .. } | GlobalAlloc::VTable(..) => {
                // These are immutable.
                Mutability::Not
            }
        }
    }

    pub fn size_and_align(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> (Size, Align) {
        match self {
            GlobalAlloc::Static(def_id) => {
                let DefKind::Static { nested, .. } = tcx.def_kind(def_id) else {
                    bug!("GlobalAlloc::Static is not a static")
                };

                if nested {
                    // Nested anonymous statics are untyped, so let's get their
                    // size and alignment from the allocation itself. This always
                    // succeeds, as the query is fed at DefId creation time, so no
                    // evaluation actually occurs.
                    let alloc = tcx.eval_static_initializer(def_id).unwrap();
                    (alloc.0.size(), alloc.0.align)
                } else {
                    // Use size and align of the type for everything else. We need
                    // to do that to
                    // * avoid cycle errors in case of self-referential statics,
                    // * be able to get information on extern statics.
                    let ty = tcx
                        .type_of(def_id)
                        .no_bound_vars()
                        .expect("statics should not have generic parameters");
                    let layout = tcx.layout_of(typing_env.as_query_input(ty)).unwrap();
                    assert!(layout.is_sized());
                    (layout.size, layout.align.abi)
                }
            }
            GlobalAlloc::Memory(alloc) => {
                let alloc = alloc.inner();
                (alloc.size(), alloc.align)
            }
            GlobalAlloc::Function { .. } => (Size::ZERO, Align::ONE),
            GlobalAlloc::VTable(..) => {
                // No data to be accessed here. But vtables are pointer-aligned.
                return (Size::ZERO, tcx.data_layout.pointer_align.abi);
            }
        }
    }
}

pub const CTFE_ALLOC_SALT: usize = 0;

pub(crate) struct AllocMap<'tcx> {
    /// Maps `AllocId`s to their corresponding allocations.
    // Note that this map on rustc workloads seems to be rather dense, but in miri workloads should
    // be pretty sparse. In #136105 we considered replacing it with a (dense) Vec-based map, but
    // since there are workloads where it can be sparse we decided to go with sharding for now. At
    // least up to 32 cores the one workload tested didn't exhibit much difference between the two.
    //
    // Should be locked *after* locking dedup if locking both to avoid deadlocks.
    to_alloc: ShardedHashMap<AllocId, GlobalAlloc<'tcx>>,

    /// Used to deduplicate global allocations: functions, vtables, string literals, ...
    ///
    /// The `usize` is a "salt" used by Miri to make deduplication imperfect, thus better emulating
    /// the actual guarantees.
    dedup: Lock<FxHashMap<(GlobalAlloc<'tcx>, usize), AllocId>>,

    /// The `AllocId` to assign to the next requested ID.
    /// Always incremented; never gets smaller.
    next_id: AtomicU64,
}

impl<'tcx> AllocMap<'tcx> {
    pub(crate) fn new() -> Self {
        AllocMap {
            to_alloc: Default::default(),
            dedup: Default::default(),
            next_id: AtomicU64::new(1),
        }
    }
    fn reserve(&self) -> AllocId {
        // Technically there is a window here where we overflow and then another thread
        // increments `next_id` *again* and uses it before we panic and tear down the entire session.
        // We consider this fine since such overflows cannot realistically occur.
        let next_id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        AllocId(NonZero::new(next_id).unwrap())
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Obtains a new allocation ID that can be referenced but does not
    /// yet have an allocation backing it.
    ///
    /// Make sure to call `set_alloc_id_memory` or `set_alloc_id_same_memory` before returning such
    /// an `AllocId` from a query.
    pub fn reserve_alloc_id(self) -> AllocId {
        self.alloc_map.reserve()
    }

    /// Reserves a new ID *if* this allocation has not been dedup-reserved before.
    /// Should not be used for mutable memory.
    fn reserve_and_set_dedup(self, alloc: GlobalAlloc<'tcx>, salt: usize) -> AllocId {
        if let GlobalAlloc::Memory(mem) = alloc {
            if mem.inner().mutability.is_mut() {
                bug!("trying to dedup-reserve mutable memory");
            }
        }
        let alloc_salt = (alloc, salt);
        // Locking this *before* `to_alloc` also to ensure correct lock order.
        let mut dedup = self.alloc_map.dedup.lock();
        if let Some(&alloc_id) = dedup.get(&alloc_salt) {
            return alloc_id;
        }
        let id = self.alloc_map.reserve();
        debug!("creating alloc {:?} with id {id:?}", alloc_salt.0);
        let had_previous = self.alloc_map.to_alloc.insert(id, alloc_salt.0.clone()).is_some();
        // We just reserved, so should always be unique.
        assert!(!had_previous);
        dedup.insert(alloc_salt, id);
        id
    }

    /// Generates an `AllocId` for a memory allocation. If the exact same memory has been
    /// allocated before, this will return the same `AllocId`.
    pub fn reserve_and_set_memory_dedup(self, mem: ConstAllocation<'tcx>, salt: usize) -> AllocId {
        self.reserve_and_set_dedup(GlobalAlloc::Memory(mem), salt)
    }

    /// Generates an `AllocId` for a static or return a cached one in case this function has been
    /// called on the same static before.
    pub fn reserve_and_set_static_alloc(self, static_id: DefId) -> AllocId {
        let salt = 0; // Statics have a guaranteed unique address, no salt added.
        self.reserve_and_set_dedup(GlobalAlloc::Static(static_id), salt)
    }

    /// Generates an `AllocId` for a function. Will get deduplicated.
    pub fn reserve_and_set_fn_alloc(self, instance: Instance<'tcx>, salt: usize) -> AllocId {
        self.reserve_and_set_dedup(GlobalAlloc::Function { instance }, salt)
    }

    /// Generates an `AllocId` for a (symbolic, not-reified) vtable. Will get deduplicated.
    pub fn reserve_and_set_vtable_alloc(
        self,
        ty: Ty<'tcx>,
        dyn_ty: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        salt: usize,
    ) -> AllocId {
        self.reserve_and_set_dedup(GlobalAlloc::VTable(ty, dyn_ty), salt)
    }

    /// Interns the `Allocation` and return a new `AllocId`, even if there's already an identical
    /// `Allocation` with a different `AllocId`.
    /// Statics with identical content will still point to the same `Allocation`, i.e.,
    /// their data will be deduplicated through `Allocation` interning -- but they
    /// are different places in memory and as such need different IDs.
    pub fn reserve_and_set_memory_alloc(self, mem: ConstAllocation<'tcx>) -> AllocId {
        let id = self.reserve_alloc_id();
        self.set_alloc_id_memory(id, mem);
        id
    }

    /// Returns `None` in case the `AllocId` is dangling. An `InterpretCx` can still have a
    /// local `Allocation` for that `AllocId`, but having such an `AllocId` in a constant is
    /// illegal and will likely ICE.
    /// This function exists to allow const eval to detect the difference between evaluation-
    /// local dangling pointers and allocations in constants/statics.
    #[inline]
    pub fn try_get_global_alloc(self, id: AllocId) -> Option<GlobalAlloc<'tcx>> {
        self.alloc_map.to_alloc.get(&id)
    }

    #[inline]
    #[track_caller]
    /// Panics in case the `AllocId` is dangling. Since that is impossible for `AllocId`s in
    /// constants (as all constants must pass interning and validation that check for dangling
    /// ids), this function is frequently used throughout rustc, but should not be used within
    /// the interpreter.
    pub fn global_alloc(self, id: AllocId) -> GlobalAlloc<'tcx> {
        match self.try_get_global_alloc(id) {
            Some(alloc) => alloc,
            None => bug!("could not find allocation for {id:?}"),
        }
    }

    /// Freezes an `AllocId` created with `reserve` by pointing it at an `Allocation`. Trying to
    /// call this function twice, even with the same `Allocation` will ICE the compiler.
    pub fn set_alloc_id_memory(self, id: AllocId, mem: ConstAllocation<'tcx>) {
        if let Some(old) = self.alloc_map.to_alloc.insert(id, GlobalAlloc::Memory(mem)) {
            bug!("tried to set allocation ID {id:?}, but it was already existing as {old:#?}");
        }
    }

    /// Freezes an `AllocId` created with `reserve` by pointing it at a static item. Trying to
    /// call this function twice, even with the same `DefId` will ICE the compiler.
    pub fn set_nested_alloc_id_static(self, id: AllocId, def_id: LocalDefId) {
        if let Some(old) =
            self.alloc_map.to_alloc.insert(id, GlobalAlloc::Static(def_id.to_def_id()))
        {
            bug!("tried to set allocation ID {id:?}, but it was already existing as {old:#?}");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Methods to access integers in the target endianness
////////////////////////////////////////////////////////////////////////////////

#[inline]
pub fn write_target_uint(
    endianness: Endian,
    mut target: &mut [u8],
    data: u128,
) -> Result<(), io::Error> {
    // This u128 holds an "any-size uint" (since smaller uints can fits in it)
    // So we do not write all bytes of the u128, just the "payload".
    match endianness {
        Endian::Little => target.write(&data.to_le_bytes())?,
        Endian::Big => target.write(&data.to_be_bytes()[16 - target.len()..])?,
    };
    debug_assert!(target.len() == 0); // We should have filled the target buffer.
    Ok(())
}

#[inline]
pub fn read_target_uint(endianness: Endian, mut source: &[u8]) -> Result<u128, io::Error> {
    // This u128 holds an "any-size uint" (since smaller uints can fits in it)
    let mut buf = [0u8; size_of::<u128>()];
    // So we do not read exactly 16 bytes into the u128, just the "payload".
    let uint = match endianness {
        Endian::Little => {
            source.read_exact(&mut buf[..source.len()])?;
            Ok(u128::from_le_bytes(buf))
        }
        Endian::Big => {
            source.read_exact(&mut buf[16 - source.len()..])?;
            Ok(u128::from_be_bytes(buf))
        }
    };
    debug_assert!(source.len() == 0); // We should have consumed the source buffer.
    uint
}
