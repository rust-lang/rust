//! An interpreter for MIR used in CTFE and by miri.

#[macro_export]
macro_rules! err_unsup {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpError::Unsupported(
            $crate::mir::interpret::UnsupportedOpInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! err_unsup_format {
    ($($tt:tt)*) => { err_unsup!(Unsupported(format!($($tt)*))) };
}

#[macro_export]
macro_rules! err_inval {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpError::InvalidProgram(
            $crate::mir::interpret::InvalidProgramInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! err_ub {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpError::UndefinedBehavior(
            $crate::mir::interpret::UndefinedBehaviorInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! err_ub_format {
    ($($tt:tt)*) => { err_ub!(Ub(format!($($tt)*))) };
}

#[macro_export]
macro_rules! err_exhaust {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpError::ResourceExhaustion(
            $crate::mir::interpret::ResourceExhaustionInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! err_machine_stop {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpError::MachineStop(Box::new($($tt)*))
    };
}

// In the `throw_*` macros, avoid `return` to make them work with `try {}`.
#[macro_export]
macro_rules! throw_unsup {
    ($($tt:tt)*) => { do yeet err_unsup!($($tt)*) };
}

#[macro_export]
macro_rules! throw_unsup_format {
    ($($tt:tt)*) => { throw_unsup!(Unsupported(format!($($tt)*))) };
}

#[macro_export]
macro_rules! throw_inval {
    ($($tt:tt)*) => { do yeet err_inval!($($tt)*) };
}

#[macro_export]
macro_rules! throw_ub {
    ($($tt:tt)*) => { do yeet err_ub!($($tt)*) };
}

#[macro_export]
macro_rules! throw_ub_format {
    ($($tt:tt)*) => { throw_ub!(Ub(format!($($tt)*))) };
}

#[macro_export]
macro_rules! throw_exhaust {
    ($($tt:tt)*) => { do yeet err_exhaust!($($tt)*) };
}

#[macro_export]
macro_rules! throw_machine_stop {
    ($($tt:tt)*) => { do yeet err_machine_stop!($($tt)*) };
}

mod allocation;
mod error;
mod pointer;
mod queries;
mod value;

use std::convert::TryFrom;
use std::fmt;
use std::io;
use std::io::{Read, Write};
use std::num::{NonZeroU32, NonZeroU64};
use std::sync::atomic::{AtomicU32, Ordering};

use rustc_ast::LitKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::{HashMapExt, Lock};
use rustc_data_structures::tiny_list::TinyList;
use rustc_hir::def_id::DefId;
use rustc_macros::HashStable;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_serialize::{Decodable, Encodable};
use rustc_target::abi::Endian;

use crate::mir;
use crate::ty::codec::{TyDecoder, TyEncoder};
use crate::ty::subst::GenericArgKind;
use crate::ty::{self, Instance, Ty, TyCtxt};

pub use self::error::{
    struct_error, CheckInAllocMsg, ErrorHandled, EvalToAllocationRawResult, EvalToConstValueResult,
    EvalToValTreeResult, InterpError, InterpErrorInfo, InterpResult, InvalidProgramInfo,
    MachineStopType, ResourceExhaustionInfo, ScalarSizeMismatch, UndefinedBehaviorInfo,
    UninitBytesAccess, UnsupportedOpInfo,
};

pub use self::value::{get_slice_bytes, ConstAlloc, ConstValue, Scalar};

pub use self::allocation::{
    alloc_range, AllocRange, Allocation, ConstAllocation, InitChunk, InitChunkIter, InitMask,
    Relocations,
};

pub use self::pointer::{Pointer, PointerArithmetic, Provenance};

/// Uniquely identifies one of the following:
/// - A constant
/// - A static
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, Lift)]
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
            format!("{}::{:?}", instance_name, promoted)
        } else {
            instance_name
        }
    }
}

/// Input argument for `tcx.lit_to_const`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, HashStable)]
pub struct LitToConstInput<'tcx> {
    /// The absolute value of the resultant constant.
    pub lit: &'tcx LitKind,
    /// The type of the constant.
    pub ty: Ty<'tcx>,
    /// If the constant is negative.
    pub neg: bool,
}

/// Error type for `tcx.lit_to_const`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, HashStable)]
pub enum LitToConstError {
    /// The literal's inferred type did not match the expected `ty` in the input.
    /// This is used for graceful error handling (`delay_span_bug`) in
    /// type checking (`Const::from_anon_const`).
    TypeError,
    Reported,
}

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocId(pub NonZeroU64);

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

pub fn specialized_encode_alloc_id<'tcx, E: TyEncoder<I = TyCtxt<'tcx>>>(
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
        GlobalAlloc::Function(fn_instance) => {
            trace!("encoding {:?} with {:#?}", alloc_id, fn_instance);
            AllocDiscriminant::Fn.encode(encoder);
            fn_instance.encode(encoder);
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
            did.encode(encoder);
        }
    }
}

// Used to avoid infinite recursion when decoding cyclic allocations.
type DecodingSessionId = NonZeroU32;

#[derive(Clone)]
enum State {
    Empty,
    InProgressNonAlloc(TinyList<DecodingSessionId>),
    InProgress(TinyList<DecodingSessionId>, AllocId),
    Done(AllocId),
}

pub struct AllocDecodingState {
    // For each `AllocId`, we keep track of which decoding state it's currently in.
    decoding_state: Vec<Lock<State>>,
    // The offsets of each allocation in the data stream.
    data_offsets: Vec<u32>,
}

impl AllocDecodingState {
    #[inline]
    pub fn new_decoding_session(&self) -> AllocDecodingSession<'_> {
        static DECODER_SESSION_ID: AtomicU32 = AtomicU32::new(0);
        let counter = DECODER_SESSION_ID.fetch_add(1, Ordering::SeqCst);

        // Make sure this is never zero.
        let session_id = DecodingSessionId::new((counter & 0x7FFFFFFF) + 1).unwrap();

        AllocDecodingSession { state: self, session_id }
    }

    pub fn new(data_offsets: Vec<u32>) -> Self {
        let decoding_state = vec![Lock::new(State::Empty); data_offsets.len()];

        Self { decoding_state, data_offsets }
    }
}

#[derive(Copy, Clone)]
pub struct AllocDecodingSession<'s> {
    state: &'s AllocDecodingState,
    session_id: DecodingSessionId,
}

impl<'s> AllocDecodingSession<'s> {
    /// Decodes an `AllocId` in a thread-safe way.
    pub fn decode_alloc_id<'tcx, D>(&self, decoder: &mut D) -> AllocId
    where
        D: TyDecoder<I = TyCtxt<'tcx>>,
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

        // Check the decoding state to see if it's already decoded or if we should
        // decode it here.
        let alloc_id = {
            let mut entry = self.state.decoding_state[idx].lock();

            match *entry {
                State::Done(alloc_id) => {
                    return alloc_id;
                }
                ref mut entry @ State::Empty => {
                    // We are allowed to decode.
                    match alloc_kind {
                        AllocDiscriminant::Alloc => {
                            // If this is an allocation, we need to reserve an
                            // `AllocId` so we can decode cyclic graphs.
                            let alloc_id = decoder.interner().reserve_alloc_id();
                            *entry =
                                State::InProgress(TinyList::new_single(self.session_id), alloc_id);
                            Some(alloc_id)
                        }
                        AllocDiscriminant::Fn
                        | AllocDiscriminant::Static
                        | AllocDiscriminant::VTable => {
                            // Fns and statics cannot be cyclic, and their `AllocId`
                            // is determined later by interning.
                            *entry =
                                State::InProgressNonAlloc(TinyList::new_single(self.session_id));
                            None
                        }
                    }
                }
                State::InProgressNonAlloc(ref mut sessions) => {
                    if sessions.contains(&self.session_id) {
                        bug!("this should be unreachable");
                    } else {
                        // Start decoding concurrently.
                        sessions.insert(self.session_id);
                        None
                    }
                }
                State::InProgress(ref mut sessions, alloc_id) => {
                    if sessions.contains(&self.session_id) {
                        // Don't recurse.
                        return alloc_id;
                    } else {
                        // Start decoding concurrently.
                        sessions.insert(self.session_id);
                        Some(alloc_id)
                    }
                }
            }
        };

        // Now decode the actual data.
        let alloc_id = decoder.with_position(pos, |decoder| {
            match alloc_kind {
                AllocDiscriminant::Alloc => {
                    let alloc = <ConstAllocation<'tcx> as Decodable<_>>::decode(decoder);
                    // We already have a reserved `AllocId`.
                    let alloc_id = alloc_id.unwrap();
                    trace!("decoded alloc {:?}: {:#?}", alloc_id, alloc);
                    decoder.interner().set_alloc_id_same_memory(alloc_id, alloc);
                    alloc_id
                }
                AllocDiscriminant::Fn => {
                    assert!(alloc_id.is_none());
                    trace!("creating fn alloc ID");
                    let instance = ty::Instance::decode(decoder);
                    trace!("decoded fn alloc instance: {:?}", instance);
                    let alloc_id = decoder.interner().create_fn_alloc(instance);
                    alloc_id
                }
                AllocDiscriminant::VTable => {
                    assert!(alloc_id.is_none());
                    trace!("creating vtable alloc ID");
                    let ty = <Ty<'_> as Decodable<D>>::decode(decoder);
                    let poly_trait_ref =
                        <Option<ty::PolyExistentialTraitRef<'_>> as Decodable<D>>::decode(decoder);
                    trace!("decoded vtable alloc instance: {ty:?}, {poly_trait_ref:?}");
                    let alloc_id = decoder.interner().create_vtable_alloc(ty, poly_trait_ref);
                    alloc_id
                }
                AllocDiscriminant::Static => {
                    assert!(alloc_id.is_none());
                    trace!("creating extern static alloc ID");
                    let did = <DefId as Decodable<D>>::decode(decoder);
                    trace!("decoded static def-ID: {:?}", did);
                    let alloc_id = decoder.interner().create_static_alloc(did);
                    alloc_id
                }
            }
        });

        self.state.decoding_state[idx].with_lock(|entry| {
            *entry = State::Done(alloc_id);
        });

        alloc_id
    }
}

/// An allocation in the global (tcx-managed) memory can be either a function pointer,
/// a static, or a "real" allocation with some data in it.
#[derive(Debug, Clone, Eq, PartialEq, Hash, TyDecodable, TyEncodable, HashStable)]
pub enum GlobalAlloc<'tcx> {
    /// The alloc ID is used as a function pointer.
    Function(Instance<'tcx>),
    /// This alloc ID points to a symbolic (not-reified) vtable.
    VTable(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>),
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
            GlobalAlloc::Function(instance) => instance,
            _ => bug!("expected function, got {:?}", self),
        }
    }

    /// Panics if the `GlobalAlloc` is not `GlobalAlloc::VTable`
    #[track_caller]
    #[inline]
    pub fn unwrap_vtable(&self) -> (Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>) {
        match *self {
            GlobalAlloc::VTable(ty, poly_trait_ref) => (ty, poly_trait_ref),
            _ => bug!("expected vtable, got {:?}", self),
        }
    }
}

pub(crate) struct AllocMap<'tcx> {
    /// Maps `AllocId`s to their corresponding allocations.
    alloc_map: FxHashMap<AllocId, GlobalAlloc<'tcx>>,

    /// Used to ensure that statics and functions only get one associated `AllocId`.
    /// Should never contain a `GlobalAlloc::Memory`!
    //
    // FIXME: Should we just have two separate dedup maps for statics and functions each?
    dedup: FxHashMap<GlobalAlloc<'tcx>, AllocId>,

    /// The `AllocId` to assign to the next requested ID.
    /// Always incremented; never gets smaller.
    next_id: AllocId,
}

impl<'tcx> AllocMap<'tcx> {
    pub(crate) fn new() -> Self {
        AllocMap {
            alloc_map: Default::default(),
            dedup: Default::default(),
            next_id: AllocId(NonZeroU64::new(1).unwrap()),
        }
    }
    fn reserve(&mut self) -> AllocId {
        let next = self.next_id;
        self.next_id.0 = self.next_id.0.checked_add(1).expect(
            "You overflowed a u64 by incrementing by 1... \
             You've just earned yourself a free drink if we ever meet. \
             Seriously, how did you do that?!",
        );
        next
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Obtains a new allocation ID that can be referenced but does not
    /// yet have an allocation backing it.
    ///
    /// Make sure to call `set_alloc_id_memory` or `set_alloc_id_same_memory` before returning such
    /// an `AllocId` from a query.
    pub fn reserve_alloc_id(self) -> AllocId {
        self.alloc_map.lock().reserve()
    }

    /// Reserves a new ID *if* this allocation has not been dedup-reserved before.
    /// Should only be used for "symbolic" allocations (function pointers, vtables, statics), we
    /// don't want to dedup IDs for "real" memory!
    fn reserve_and_set_dedup(self, alloc: GlobalAlloc<'tcx>) -> AllocId {
        let mut alloc_map = self.alloc_map.lock();
        match alloc {
            GlobalAlloc::Function(..) | GlobalAlloc::Static(..) | GlobalAlloc::VTable(..) => {}
            GlobalAlloc::Memory(..) => bug!("Trying to dedup-reserve memory with real data!"),
        }
        if let Some(&alloc_id) = alloc_map.dedup.get(&alloc) {
            return alloc_id;
        }
        let id = alloc_map.reserve();
        debug!("creating alloc {alloc:?} with id {id:?}");
        alloc_map.alloc_map.insert(id, alloc.clone());
        alloc_map.dedup.insert(alloc, id);
        id
    }

    /// Generates an `AllocId` for a static or return a cached one in case this function has been
    /// called on the same static before.
    pub fn create_static_alloc(self, static_id: DefId) -> AllocId {
        self.reserve_and_set_dedup(GlobalAlloc::Static(static_id))
    }

    /// Generates an `AllocId` for a function.  Depending on the function type,
    /// this might get deduplicated or assigned a new ID each time.
    pub fn create_fn_alloc(self, instance: Instance<'tcx>) -> AllocId {
        // Functions cannot be identified by pointers, as asm-equal functions can get deduplicated
        // by the linker (we set the "unnamed_addr" attribute for LLVM) and functions can be
        // duplicated across crates.
        // We thus generate a new `AllocId` for every mention of a function. This means that
        // `main as fn() == main as fn()` is false, while `let x = main as fn(); x == x` is true.
        // However, formatting code relies on function identity (see #58320), so we only do
        // this for generic functions.  Lifetime parameters are ignored.
        let is_generic = instance
            .substs
            .into_iter()
            .any(|kind| !matches!(kind.unpack(), GenericArgKind::Lifetime(_)));
        if is_generic {
            // Get a fresh ID.
            let mut alloc_map = self.alloc_map.lock();
            let id = alloc_map.reserve();
            alloc_map.alloc_map.insert(id, GlobalAlloc::Function(instance));
            id
        } else {
            // Deduplicate.
            self.reserve_and_set_dedup(GlobalAlloc::Function(instance))
        }
    }

    /// Generates an `AllocId` for a (symbolic, not-reified) vtable.  Will get deduplicated.
    pub fn create_vtable_alloc(
        self,
        ty: Ty<'tcx>,
        poly_trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
    ) -> AllocId {
        self.reserve_and_set_dedup(GlobalAlloc::VTable(ty, poly_trait_ref))
    }

    /// Interns the `Allocation` and return a new `AllocId`, even if there's already an identical
    /// `Allocation` with a different `AllocId`.
    /// Statics with identical content will still point to the same `Allocation`, i.e.,
    /// their data will be deduplicated through `Allocation` interning -- but they
    /// are different places in memory and as such need different IDs.
    pub fn create_memory_alloc(self, mem: ConstAllocation<'tcx>) -> AllocId {
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
        self.alloc_map.lock().alloc_map.get(&id).cloned()
    }

    #[inline]
    #[track_caller]
    /// Panics in case the `AllocId` is dangling. Since that is impossible for `AllocId`s in
    /// constants (as all constants must pass interning and validation that check for dangling
    /// ids), this function is frequently used throughout rustc, but should not be used within
    /// the miri engine.
    pub fn global_alloc(self, id: AllocId) -> GlobalAlloc<'tcx> {
        match self.try_get_global_alloc(id) {
            Some(alloc) => alloc,
            None => bug!("could not find allocation for {id:?}"),
        }
    }

    /// Freezes an `AllocId` created with `reserve` by pointing it at an `Allocation`. Trying to
    /// call this function twice, even with the same `Allocation` will ICE the compiler.
    pub fn set_alloc_id_memory(self, id: AllocId, mem: ConstAllocation<'tcx>) {
        if let Some(old) = self.alloc_map.lock().alloc_map.insert(id, GlobalAlloc::Memory(mem)) {
            bug!("tried to set allocation ID {id:?}, but it was already existing as {old:#?}");
        }
    }

    /// Freezes an `AllocId` created with `reserve` by pointing it at an `Allocation`. May be called
    /// twice for the same `(AllocId, Allocation)` pair.
    fn set_alloc_id_same_memory(self, id: AllocId, mem: ConstAllocation<'tcx>) {
        self.alloc_map.lock().alloc_map.insert_same(id, GlobalAlloc::Memory(mem));
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
    let mut buf = [0u8; std::mem::size_of::<u128>()];
    // So we do not read exactly 16 bytes into the u128, just the "payload".
    let uint = match endianness {
        Endian::Little => {
            source.read(&mut buf)?;
            Ok(u128::from_le_bytes(buf))
        }
        Endian::Big => {
            source.read(&mut buf[16 - source.len()..])?;
            Ok(u128::from_be_bytes(buf))
        }
    };
    debug_assert!(source.len() == 0); // We should have consumed the source buffer.
    uint
}
