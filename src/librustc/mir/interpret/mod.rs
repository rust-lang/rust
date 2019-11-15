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
macro_rules! err_panic {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpError::Panic(
            $crate::mir::interpret::PanicInfo::$($tt)*
        )
    };
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
macro_rules! throw_unsup {
    ($($tt:tt)*) => { return Err(err_unsup!($($tt)*).into()) };
}

#[macro_export]
macro_rules! throw_unsup_format {
    ($($tt:tt)*) => { throw_unsup!(Unsupported(format!($($tt)*))) };
}

#[macro_export]
macro_rules! throw_inval {
    ($($tt:tt)*) => { return Err(err_inval!($($tt)*).into()) };
}

#[macro_export]
macro_rules! throw_ub {
    ($($tt:tt)*) => { return Err(err_ub!($($tt)*).into()) };
}

#[macro_export]
macro_rules! throw_ub_format {
    ($($tt:tt)*) => { throw_ub!(Ub(format!($($tt)*))) };
}

#[macro_export]
macro_rules! throw_panic {
    ($($tt:tt)*) => { return Err(err_panic!($($tt)*).into()) };
}

#[macro_export]
macro_rules! throw_exhaust {
    ($($tt:tt)*) => { return Err(err_exhaust!($($tt)*).into()) };
}

mod error;
mod value;
mod allocation;
mod pointer;

pub use self::error::{
    InterpErrorInfo, InterpResult, InterpError, AssertMessage, ConstEvalErr, struct_error,
    FrameInfo, ConstEvalRawResult, ConstEvalResult, ErrorHandled, PanicInfo, UnsupportedOpInfo,
    InvalidProgramInfo, ResourceExhaustionInfo, UndefinedBehaviorInfo,
};

pub use self::value::{Scalar, ScalarMaybeUndef, RawConst, ConstValue, get_slice_bytes};

pub use self::allocation::{Allocation, AllocationExtra, Relocations, UndefMask};

pub use self::pointer::{Pointer, PointerArithmetic, CheckInAllocMsg};

use crate::mir;
use crate::hir::def_id::DefId;
use crate::ty::{self, TyCtxt, Instance, subst::GenericArgKind};
use crate::ty::codec::TyDecoder;
use crate::ty::layout::{self, Size};
use std::io;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use rustc_serialize::{Encoder, Decodable, Encodable};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::{Lock, HashMapExt};
use rustc_data_structures::tiny_list::TinyList;
use rustc_macros::HashStable;
use byteorder::{WriteBytesExt, ReadBytesExt, LittleEndian, BigEndian};

/// Uniquely identifies a specific constant or static.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, RustcEncodable, RustcDecodable)]
#[derive(HashStable, Lift)]
pub struct GlobalId<'tcx> {
    /// For a constant or static, the `Instance` of the item itself.
    /// For a promoted global, the `Instance` of the function they belong to.
    pub instance: ty::Instance<'tcx>,

    /// The index for promoted globals within their function's `mir::Body`.
    pub promoted: Option<mir::Promoted>,
}

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Debug)]
pub struct AllocId(pub u64);

impl rustc_serialize::UseSpecializedEncodable for AllocId {}
impl rustc_serialize::UseSpecializedDecodable for AllocId {}

#[derive(RustcDecodable, RustcEncodable)]
enum AllocDiscriminant {
    Alloc,
    Fn,
    Static,
}

pub fn specialized_encode_alloc_id<'tcx, E: Encoder>(
    encoder: &mut E,
    tcx: TyCtxt<'tcx>,
    alloc_id: AllocId,
) -> Result<(), E::Error> {
    let alloc: GlobalAlloc<'tcx> = tcx.alloc_map.lock().get(alloc_id)
        .expect("no value for given alloc ID");
    match alloc {
        GlobalAlloc::Memory(alloc) => {
            trace!("encoding {:?} with {:#?}", alloc_id, alloc);
            AllocDiscriminant::Alloc.encode(encoder)?;
            alloc.encode(encoder)?;
        }
        GlobalAlloc::Function(fn_instance) => {
            trace!("encoding {:?} with {:#?}", alloc_id, fn_instance);
            AllocDiscriminant::Fn.encode(encoder)?;
            fn_instance.encode(encoder)?;
        }
        GlobalAlloc::Static(did) => {
            // References to statics doesn't need to know about their allocations,
            // just about its `DefId`.
            AllocDiscriminant::Static.encode(encoder)?;
            did.encode(encoder)?;
        }
    }
    Ok(())
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
    pub fn new_decoding_session(&self) -> AllocDecodingSession<'_> {
        static DECODER_SESSION_ID: AtomicU32 = AtomicU32::new(0);
        let counter = DECODER_SESSION_ID.fetch_add(1, Ordering::SeqCst);

        // Make sure this is never zero.
        let session_id = DecodingSessionId::new((counter & 0x7FFFFFFF) + 1).unwrap();

        AllocDecodingSession {
            state: self,
            session_id,
        }
    }

    pub fn new(data_offsets: Vec<u32>) -> Self {
        let decoding_state = vec![Lock::new(State::Empty); data_offsets.len()];

        Self {
            decoding_state,
            data_offsets,
        }
    }
}

#[derive(Copy, Clone)]
pub struct AllocDecodingSession<'s> {
    state: &'s AllocDecodingState,
    session_id: DecodingSessionId,
}

impl<'s> AllocDecodingSession<'s> {
    /// Decodes an `AllocId` in a thread-safe way.
    pub fn decode_alloc_id<D>(&self, decoder: &mut D) -> Result<AllocId, D::Error>
    where
        D: TyDecoder<'tcx>,
    {
        // Read the index of the allocation.
        let idx = decoder.read_u32()? as usize;
        let pos = self.state.data_offsets[idx] as usize;

        // Decode the `AllocDiscriminant` now so that we know if we have to reserve an
        // `AllocId`.
        let (alloc_kind, pos) = decoder.with_position(pos, |decoder| {
            let alloc_kind = AllocDiscriminant::decode(decoder)?;
            Ok((alloc_kind, decoder.position()))
        })?;

        // Check the decoding state to see if it's already decoded or if we should
        // decode it here.
        let alloc_id = {
            let mut entry = self.state.decoding_state[idx].lock();

            match *entry {
                State::Done(alloc_id) => {
                    return Ok(alloc_id);
                }
                ref mut entry @ State::Empty => {
                    // We are allowed to decode.
                    match alloc_kind {
                        AllocDiscriminant::Alloc => {
                            // If this is an allocation, we need to reserve an
                            // `AllocId` so we can decode cyclic graphs.
                            let alloc_id = decoder.tcx().alloc_map.lock().reserve();
                            *entry = State::InProgress(
                                TinyList::new_single(self.session_id),
                                alloc_id);
                            Some(alloc_id)
                        },
                        AllocDiscriminant::Fn | AllocDiscriminant::Static => {
                            // Fns and statics cannot be cyclic, and their `AllocId`
                            // is determined later by interning.
                            *entry = State::InProgressNonAlloc(
                                TinyList::new_single(self.session_id));
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
                        return Ok(alloc_id)
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
                    let alloc = <&'tcx Allocation as Decodable>::decode(decoder)?;
                    // We already have a reserved `AllocId`.
                    let alloc_id = alloc_id.unwrap();
                    trace!("decoded alloc {:?}: {:#?}", alloc_id, alloc);
                    decoder.tcx().alloc_map.lock().set_alloc_id_same_memory(alloc_id, alloc);
                    Ok(alloc_id)
                },
                AllocDiscriminant::Fn => {
                    assert!(alloc_id.is_none());
                    trace!("creating fn alloc ID");
                    let instance = ty::Instance::decode(decoder)?;
                    trace!("decoded fn alloc instance: {:?}", instance);
                    let alloc_id = decoder.tcx().alloc_map.lock().create_fn_alloc(instance);
                    Ok(alloc_id)
                },
                AllocDiscriminant::Static => {
                    assert!(alloc_id.is_none());
                    trace!("creating extern static alloc ID");
                    let did = DefId::decode(decoder)?;
                    trace!("decoded static def-ID: {:?}", did);
                    let alloc_id = decoder.tcx().alloc_map.lock().create_static_alloc(did);
                    Ok(alloc_id)
                }
            }
        })?;

        self.state.decoding_state[idx].with_lock(|entry| {
            *entry = State::Done(alloc_id);
        });

        Ok(alloc_id)
    }
}

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An allocation in the global (tcx-managed) memory can be either a function pointer,
/// a static, or a "real" allocation with some data in it.
#[derive(Debug, Clone, Eq, PartialEq, Hash, RustcDecodable, RustcEncodable, HashStable)]
pub enum GlobalAlloc<'tcx> {
    /// The alloc ID is used as a function pointer.
    Function(Instance<'tcx>),
    /// The alloc ID points to a "lazy" static variable that did not get computed (yet).
    /// This is also used to break the cycle in recursive statics.
    Static(DefId),
    /// The alloc ID points to memory.
    Memory(&'tcx Allocation),
}

pub struct AllocMap<'tcx> {
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
    pub fn new() -> Self {
        AllocMap {
            alloc_map: Default::default(),
            dedup: Default::default(),
            next_id: AllocId(0),
        }
    }

    /// Obtains a new allocation ID that can be referenced but does not
    /// yet have an allocation backing it.
    ///
    /// Make sure to call `set_alloc_id_memory` or `set_alloc_id_same_memory` before returning such
    /// an `AllocId` from a query.
    pub fn reserve(
        &mut self,
    ) -> AllocId {
        let next = self.next_id;
        self.next_id.0 = self.next_id.0
            .checked_add(1)
            .expect("You overflowed a u64 by incrementing by 1... \
                     You've just earned yourself a free drink if we ever meet. \
                     Seriously, how did you do that?!");
        next
    }

    /// Reserves a new ID *if* this allocation has not been dedup-reserved before.
    /// Should only be used for function pointers and statics, we don't want
    /// to dedup IDs for "real" memory!
    fn reserve_and_set_dedup(&mut self, alloc: GlobalAlloc<'tcx>) -> AllocId {
        match alloc {
            GlobalAlloc::Function(..) | GlobalAlloc::Static(..) => {},
            GlobalAlloc::Memory(..) => bug!("Trying to dedup-reserve memory with real data!"),
        }
        if let Some(&alloc_id) = self.dedup.get(&alloc) {
            return alloc_id;
        }
        let id = self.reserve();
        debug!("creating alloc {:?} with id {}", alloc, id);
        self.alloc_map.insert(id, alloc.clone());
        self.dedup.insert(alloc, id);
        id
    }

    /// Generates an `AllocId` for a static or return a cached one in case this function has been
    /// called on the same static before.
    pub fn create_static_alloc(&mut self, static_id: DefId) -> AllocId {
        self.reserve_and_set_dedup(GlobalAlloc::Static(static_id))
    }

    /// Generates an `AllocId` for a function.  Depending on the function type,
    /// this might get deduplicated or assigned a new ID each time.
    pub fn create_fn_alloc(&mut self, instance: Instance<'tcx>) -> AllocId {
        // Functions cannot be identified by pointers, as asm-equal functions can get deduplicated
        // by the linker (we set the "unnamed_addr" attribute for LLVM) and functions can be
        // duplicated across crates.
        // We thus generate a new `AllocId` for every mention of a function. This means that
        // `main as fn() == main as fn()` is false, while `let x = main as fn(); x == x` is true.
        // However, formatting code relies on function identity (see #58320), so we only do
        // this for generic functions.  Lifetime parameters are ignored.
        let is_generic = instance.substs.into_iter().any(|kind| {
            match kind.unpack() {
                GenericArgKind::Lifetime(_) => false,
                _ => true,
            }
        });
        if is_generic {
            // Get a fresh ID.
            let id = self.reserve();
            self.alloc_map.insert(id, GlobalAlloc::Function(instance));
            id
        } else {
            // Deduplicate.
            self.reserve_and_set_dedup(GlobalAlloc::Function(instance))
        }
    }

    /// Interns the `Allocation` and return a new `AllocId`, even if there's already an identical
    /// `Allocation` with a different `AllocId`.
    /// Statics with identical content will still point to the same `Allocation`, i.e.,
    /// their data will be deduplicated through `Allocation` interning -- but they
    /// are different places in memory and as such need different IDs.
    pub fn create_memory_alloc(&mut self, mem: &'tcx Allocation) -> AllocId {
        let id = self.reserve();
        self.set_alloc_id_memory(id, mem);
        id
    }

    /// Returns `None` in case the `AllocId` is dangling. An `InterpretCx` can still have a
    /// local `Allocation` for that `AllocId`, but having such an `AllocId` in a constant is
    /// illegal and will likely ICE.
    /// This function exists to allow const eval to detect the difference between evaluation-
    /// local dangling pointers and allocations in constants/statics.
    #[inline]
    pub fn get(&self, id: AllocId) -> Option<GlobalAlloc<'tcx>> {
        self.alloc_map.get(&id).cloned()
    }

    /// Panics if the `AllocId` does not refer to an `Allocation`
    pub fn unwrap_memory(&self, id: AllocId) -> &'tcx Allocation {
        match self.get(id) {
            Some(GlobalAlloc::Memory(mem)) => mem,
            _ => bug!("expected allocation ID {} to point to memory", id),
        }
    }

    /// Panics if the `AllocId` does not refer to a function
    pub fn unwrap_fn(&self, id: AllocId) -> Instance<'tcx> {
        match self.get(id) {
            Some(GlobalAlloc::Function(instance)) => instance,
            _ => bug!("expected allocation ID {} to point to a function", id),
        }
    }

    /// Freezes an `AllocId` created with `reserve` by pointing it at an `Allocation`. Trying to
    /// call this function twice, even with the same `Allocation` will ICE the compiler.
    pub fn set_alloc_id_memory(&mut self, id: AllocId, mem: &'tcx Allocation) {
        if let Some(old) = self.alloc_map.insert(id, GlobalAlloc::Memory(mem)) {
            bug!("tried to set allocation ID {}, but it was already existing as {:#?}", id, old);
        }
    }

    /// Freezes an `AllocId` created with `reserve` by pointing it at an `Allocation`. May be called
    /// twice for the same `(AllocId, Allocation)` pair.
    fn set_alloc_id_same_memory(&mut self, id: AllocId, mem: &'tcx Allocation) {
        self.alloc_map.insert_same(id, GlobalAlloc::Memory(mem));
    }
}

////////////////////////////////////////////////////////////////////////////////
// Methods to access integers in the target endianness
////////////////////////////////////////////////////////////////////////////////

#[inline]
pub fn write_target_uint(
    endianness: layout::Endian,
    mut target: &mut [u8],
    data: u128,
) -> Result<(), io::Error> {
    let len = target.len();
    match endianness {
        layout::Endian::Little => target.write_uint128::<LittleEndian>(data, len),
        layout::Endian::Big => target.write_uint128::<BigEndian>(data, len),
    }
}

#[inline]
pub fn read_target_uint(endianness: layout::Endian, mut source: &[u8]) -> Result<u128, io::Error> {
    match endianness {
        layout::Endian::Little => source.read_uint128::<LittleEndian>(source.len()),
        layout::Endian::Big => source.read_uint128::<BigEndian>(source.len()),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Methods to facilitate working with signed integers stored in a u128
////////////////////////////////////////////////////////////////////////////////

/// Truncates `value` to `size` bits and then sign-extend it to 128 bits
/// (i.e., if it is negative, fill with 1's on the left).
#[inline]
pub fn sign_extend(value: u128, size: Size) -> u128 {
    let size = size.bits();
    if size == 0 {
        // Truncated until nothing is left.
        return 0;
    }
    // Sign-extend it.
    let shift = 128 - size;
    // Shift the unsigned value to the left, then shift back to the right as signed
    // (essentially fills with FF on the left).
    (((value << shift) as i128) >> shift) as u128
}

/// Truncates `value` to `size` bits.
#[inline]
pub fn truncate(value: u128, size: Size) -> u128 {
    let size = size.bits();
    if size == 0 {
        // Truncated until nothing is left.
        return 0;
    }
    let shift = 128 - size;
    // Truncate (shift left to drop out leftover values, shift right to fill with zeroes).
    (value << shift) >> shift
}
