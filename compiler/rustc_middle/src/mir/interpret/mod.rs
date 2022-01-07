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
    ($($tt:tt)*) => { Err::<!, _>(err_unsup!($($tt)*))? };
}

#[macro_export]
macro_rules! throw_unsup_format {
    ($($tt:tt)*) => { throw_unsup!(Unsupported(format!($($tt)*))) };
}

#[macro_export]
macro_rules! throw_inval {
    ($($tt:tt)*) => { Err::<!, _>(err_inval!($($tt)*))? };
}

#[macro_export]
macro_rules! throw_ub {
    ($($tt:tt)*) => { Err::<!, _>(err_ub!($($tt)*))? };
}

#[macro_export]
macro_rules! throw_ub_format {
    ($($tt:tt)*) => { throw_ub!(Ub(format!($($tt)*))) };
}

#[macro_export]
macro_rules! throw_exhaust {
    ($($tt:tt)*) => { Err::<!, _>(err_exhaust!($($tt)*))? };
}

#[macro_export]
macro_rules! throw_machine_stop {
    ($($tt:tt)*) => { Err::<!, _>(err_machine_stop!($($tt)*))? };
}

mod allocation;
mod error;
mod pointer;
mod queries;
mod value;

use std::cell::RefCell;
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
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::definitions::DefPathData;
use rustc_middle::traits::Reveal;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_serialize::{Decodable, Encodable};
use rustc_span::{Pos, Span, DUMMY_SP};
use rustc_target::abi::Endian;

use crate::mir;
use crate::ty::codec::{TyDecoder, TyEncoder};
use crate::ty::layout::LayoutError;
use crate::ty::subst::GenericArgKind;
use crate::ty::{self, Instance, Ty, TyCtxt, WithOptConstParam};

use std::borrow::Cow;

pub use self::error::{
    struct_error, CheckInAllocMsg, ConstErrorEmitted, ConstEvalErr, ErrorHandled,
    EvalToAllocationRawResult, EvalToConstValueResult, InterpError, InterpErrorInfo, InterpResult,
    InvalidProgramInfo, MachineStopType, ResourceExhaustionInfo, UndefinedBehaviorInfo,
    UninitBytesAccess, UnsupportedOpInfo,
};

pub use self::value::{get_slice_bytes, ConstAlloc, ConstValue, Scalar, ScalarMaybeUninit};

pub use self::allocation::{
    alloc_range, AllocRange, Allocation, InitChunk, InitChunkIter, InitMask, Relocations,
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
        let instance_name = with_no_trimmed_paths(|| tcx.def_path_str(self.instance.def.def_id()));
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

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[derive(TyDecodable, TyEncodable)]
enum AllocDiscriminant {
    Alloc,
    Fn,
    Static,
}

pub fn specialized_encode_alloc_id<'tcx, E: TyEncoder<'tcx>>(
    encoder: &mut E,
    tcx: TyCtxt<'tcx>,
    alloc_id: AllocId,
) -> Result<(), E::Error> {
    match tcx.global_alloc(alloc_id) {
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
            assert!(!tcx.is_thread_local_static(did));
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
    pub fn decode_alloc_id<'tcx, D>(&self, decoder: &mut D) -> Result<AllocId, D::Error>
    where
        D: TyDecoder<'tcx>,
    {
        // Read the index of the allocation.
        let idx = usize::try_from(decoder.read_u32()?).unwrap();
        let pos = usize::try_from(self.state.data_offsets[idx]).unwrap();

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
                            let alloc_id = decoder.tcx().reserve_alloc_id();
                            *entry =
                                State::InProgress(TinyList::new_single(self.session_id), alloc_id);
                            Some(alloc_id)
                        }
                        AllocDiscriminant::Fn | AllocDiscriminant::Static => {
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
                        return Ok(alloc_id);
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
                    let alloc = <&'tcx Allocation as Decodable<_>>::decode(decoder)?;
                    // We already have a reserved `AllocId`.
                    let alloc_id = alloc_id.unwrap();
                    trace!("decoded alloc {:?}: {:#?}", alloc_id, alloc);
                    decoder.tcx().set_alloc_id_same_memory(alloc_id, alloc);
                    Ok(alloc_id)
                }
                AllocDiscriminant::Fn => {
                    assert!(alloc_id.is_none());
                    trace!("creating fn alloc ID");
                    let instance = ty::Instance::decode(decoder)?;
                    trace!("decoded fn alloc instance: {:?}", instance);
                    let alloc_id = decoder.tcx().create_fn_alloc(instance);
                    Ok(alloc_id)
                }
                AllocDiscriminant::Static => {
                    assert!(alloc_id.is_none());
                    trace!("creating extern static alloc ID");
                    let did = <DefId as Decodable<D>>::decode(decoder)?;
                    trace!("decoded static def-ID: {:?}", did);
                    let alloc_id = decoder.tcx().create_static_alloc(did);
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

/// An allocation in the global (tcx-managed) memory can be either a function pointer,
/// a static, or a "real" allocation with some data in it.
#[derive(Debug, Clone, Eq, PartialEq, Hash, TyDecodable, TyEncodable, HashStable)]
pub enum GlobalAlloc<'tcx> {
    /// The alloc ID is used as a function pointer.
    Function(Instance<'tcx>),
    /// The alloc ID points to a "lazy" static variable that did not get computed (yet).
    /// This is also used to break the cycle in recursive statics.
    Static(DefId),
    /// The alloc ID points to memory.
    Memory(&'tcx Allocation),
}

impl<'tcx> GlobalAlloc<'tcx> {
    /// Panics if the `GlobalAlloc` does not refer to an `GlobalAlloc::Memory`
    #[track_caller]
    #[inline]
    pub fn unwrap_memory(&self) -> &'tcx Allocation {
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
}

crate struct AllocMap<'tcx> {
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
    crate fn new() -> Self {
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

/// What we store about a frame in an interpreter backtrace.
#[derive(Debug)]
pub struct FrameInfo<'tcx> {
    pub instance: ty::Instance<'tcx>,
    pub span: Span,
    pub lint_root: Option<hir::HirId>,
}

impl<'tcx> fmt::Display for FrameInfo<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            if tcx.def_key(self.instance.def_id()).disambiguated_data.data
                == DefPathData::ClosureExpr
            {
                write!(f, "inside closure")?;
            } else {
                write!(f, "inside `{}`", self.instance)?;
            }
            if !self.span.is_dummy() {
                let sm = tcx.sess.source_map();
                let lo = sm.lookup_char_pos(self.span.lo());
                write!(
                    f,
                    " at {}:{}:{}",
                    sm.filename_for_diagnostics(&lo.file.name),
                    lo.line,
                    lo.col.to_usize() + 1
                )?;
            }
            Ok(())
        })
    }
}

/// Errors that were returned from calls to `eval_to_allocation_raw` or
/// `eval_to_const_value_raw` with `Reveal::Selection`. Since we want to
/// deduplicate from `Reveal::Selection` results, we store the returned
/// `ConstEvalErr` in order to allow us to report those errors on calls
/// with `Reveal::UserFacing` or `Reveal::All`.
#[derive(Debug)]
pub enum SilentError<'tcx> {
    ConstErr(ConstEvalErr<'tcx>),
    Handled(ErrorHandled<'tcx>),
}

#[derive(Debug)]
pub enum ConstDedupError<'tcx> {
    /// used for errors found in `eval_to_allocation_raw` with `Reveal::Selection`
    /// in order to allow deduplication.
    Silent(SilentError<'tcx>),

    /// error that was reported in call of `eval_to_allocation_raw`.
    Handled(ErrorHandled<'tcx>),
}

impl<'tcx> ConstDedupError<'tcx> {
    pub fn new_handled(e: ErrorHandled<'tcx>, reveal: Reveal) -> Self {
        match reveal {
            Reveal::Selection => ConstDedupError::Silent(SilentError::Handled(e)),
            _ => ConstDedupError::Handled(e),
        }
    }

    pub fn new_silent(e: SilentError<'tcx>, reveal: Reveal) -> Self {
        match reveal {
            Reveal::Selection => ConstDedupError::Silent(e),
            _ => bug!("can only create a `ConstDedupError::Silent` with `Reveal::Selection`"),
        }
    }

    pub fn get_const_err(self) -> ConstEvalErr<'tcx> {
        match self {
            ConstDedupError::Silent(e) => match e {
                SilentError::ConstErr(e) => e,
                _ => bug!(
                    "cannot call `get_const_err` on `ConstDedupError::Silent(SilentError::Handled)`"
                ),
            },
            ConstDedupError::Handled(_) => {
                bug!("get_const_err called on ConstDedupError::Handled")
            }
        }
    }

    pub fn get_handled_err(&self) -> ErrorHandled<'tcx> {
        match self {
            ConstDedupError::Handled(e) => *e,
            ConstDedupError::Silent(e) => match e {
                SilentError::Handled(e) => *e,
                SilentError::ConstErr(_) => {
                    bug!("get_handled_err called on ConstDedupError::Silent(SilentError::ConstErr)")
                }
            },
        }
    }
}

#[derive(Debug)]
pub enum ConstDedupResult<'tcx, T: 'tcx> {
    // include a Span so that we can report errors when we try to deduplicate calls
    // with Reveal::UserFacing from calls with Reveal::Selection, for which we have
    // to report the error during the dedup call
    Selection((Result<T, SilentError<'tcx>>, Span)),
    UserFacing(Result<T, ErrorHandled<'tcx>>),
    All(Result<T, ErrorHandled<'tcx>>),
}

impl<'tcx, T: 'tcx> ConstDedupResult<'tcx, T> {
    pub fn new(
        reveal: Reveal,
        val: Result<T, ConstDedupError<'tcx>>,
        opt_span: Option<Span>,
    ) -> Self {
        match reveal {
            Reveal::Selection => ConstDedupResult::Selection((
                val.map_err(|e| match e {
                    ConstDedupError::Silent(err) => err,
                    ConstDedupError::Handled(_) => bug!("expected ConstDedupError::Silent"),
                }),
                opt_span.unwrap_or(DUMMY_SP),
            )),
            Reveal::UserFacing => {
                ConstDedupResult::UserFacing(val.or_else(|e| Err(e.get_handled_err())))
            }
            Reveal::All => ConstDedupResult::All(val.or_else(|e| Err(e.get_handled_err()))),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ConstOrigin {
    ConstValue,
    Alloc,
}

/// Used to store results of calls to `eval_to_allocation_raw` and
/// `eval_to_const_value_raw`.
///
/// Depending on the value of `Reveal` of the `ParamEnv` with which the queries
/// are executed we handle errors differently. We suppress errors with `Reveal::Selection`,
/// and report errors otherwise.
/// Since a large portion of the calls with different `Reveal` arguments leads to
/// duplicate results, we try to only store the result of a call with one specific `Reveal`
/// and use that result for queries with other `Reveal` arguments.
#[derive(Debug)]
pub struct ConstDedupMap<'tcx> {
    // interning for deduplication of `eval_to_allocation_raw`
    pub alloc_map: RefCell<FxHashMap<GlobalId<'tcx>, ConstDedupResult<'tcx, ConstAlloc<'tcx>>>>,

    // interning for deduplication of `eval_to_const_value_raw`
    pub const_val_map: RefCell<FxHashMap<GlobalId<'tcx>, ConstDedupResult<'tcx, ConstValue<'tcx>>>>,

    // Used to tell whether an error needs to be reported when trying to deduplicate a
    // call to `eval_to_allocation_raw` or `eval_to_const_value_raw` with
    // `Reveal::UserFacing` or `Reveal::All` from a call with `Reveal::Selection`
    pub error_reported_map: RefCell<FxHashMap<GlobalId<'tcx>, ErrorHandled<'tcx>>>,
}

impl<'tcx> ConstDedupMap<'tcx> {
    pub fn new() -> Self {
        ConstDedupMap {
            alloc_map: Default::default(),
            const_val_map: Default::default(),
            error_reported_map: Default::default(),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn add_error_reported(
        &self,
        id: GlobalId<'tcx>,
        err: ConstErrorEmitted<'tcx>,
    ) -> ConstErrorEmitted<'tcx> {
        match err {
            ConstErrorEmitted::Emitted(err_handled) => {
                let mut error_reported_map = self.error_reported_map.borrow_mut();
                error_reported_map.insert(id, err_handled);
                debug!("error_reported_map after update: {:#?}", self.error_reported_map);

                err
            }
            ConstErrorEmitted::NotEmitted(_) => err,
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn insert_alloc(&self, id: GlobalId<'tcx>, val: ConstDedupResult<'tcx, ConstAlloc<'tcx>>) {
        let mut alloc_map = self.alloc_map.borrow_mut();
        alloc_map.insert(id, val);
        debug!("alloc_map after update: {:#?}", alloc_map);
    }

    #[instrument(skip(self), level = "debug")]
    fn insert_const_val(&self, id: GlobalId<'tcx>, val: ConstDedupResult<'tcx, ConstValue<'tcx>>) {
        let mut const_val_map = self.const_val_map.borrow_mut();
        const_val_map.insert(id, val);
        debug!("const_val_map after update: {:#?}", const_val_map);
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
    /// Should only be used for function pointers and statics, we don't want
    /// to dedup IDs for "real" memory!
    fn reserve_and_set_dedup(self, alloc: GlobalAlloc<'tcx>) -> AllocId {
        let mut alloc_map = self.alloc_map.lock();
        match alloc {
            GlobalAlloc::Function(..) | GlobalAlloc::Static(..) => {}
            GlobalAlloc::Memory(..) => bug!("Trying to dedup-reserve memory with real data!"),
        }
        if let Some(&alloc_id) = alloc_map.dedup.get(&alloc) {
            return alloc_id;
        }
        let id = alloc_map.reserve();
        debug!("creating alloc {:?} with id {}", alloc, id);
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

    /// Interns the `Allocation` and return a new `AllocId`, even if there's already an identical
    /// `Allocation` with a different `AllocId`.
    /// Statics with identical content will still point to the same `Allocation`, i.e.,
    /// their data will be deduplicated through `Allocation` interning -- but they
    /// are different places in memory and as such need different IDs.
    pub fn create_memory_alloc(self, mem: &'tcx Allocation) -> AllocId {
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
    pub fn get_global_alloc(self, id: AllocId) -> Option<GlobalAlloc<'tcx>> {
        self.alloc_map.lock().alloc_map.get(&id).cloned()
    }

    #[inline]
    #[track_caller]
    /// Panics in case the `AllocId` is dangling. Since that is impossible for `AllocId`s in
    /// constants (as all constants must pass interning and validation that check for dangling
    /// ids), this function is frequently used throughout rustc, but should not be used within
    /// the miri engine.
    pub fn global_alloc(self, id: AllocId) -> GlobalAlloc<'tcx> {
        match self.get_global_alloc(id) {
            Some(alloc) => alloc,
            None => bug!("could not find allocation for {}", id),
        }
    }

    /// Freezes an `AllocId` created with `reserve` by pointing it at an `Allocation`. Trying to
    /// call this function twice, even with the same `Allocation` will ICE the compiler.
    pub fn set_alloc_id_memory(self, id: AllocId, mem: &'tcx Allocation) {
        if let Some(old) = self.alloc_map.lock().alloc_map.insert(id, GlobalAlloc::Memory(mem)) {
            bug!("tried to set allocation ID {}, but it was already existing as {:#?}", id, old);
        }
    }

    /// Freezes an `AllocId` created with `reserve` by pointing it at an `Allocation`. May be called
    /// twice for the same `(AllocId, Allocation)` pair.
    fn set_alloc_id_same_memory(self, id: AllocId, mem: &'tcx Allocation) {
        self.alloc_map.lock().alloc_map.insert_same(id, GlobalAlloc::Memory(mem));
    }

    /// Called after an error has been returned from `eval_to_allocation_raw`
    /// or `eval_to_const_value_raw`. We handle this differently based on
    /// the `Reveal` argument. With `Reveal::Selection` we don't report the
    /// error, otherwise we do. In all cases we cache the error.
    #[instrument(skip(self, report_fn), level = "debug")]
    pub fn handle_err_for_dedup<F>(
        self,
        id: GlobalId<'tcx>,
        origin: ConstOrigin,
        error: ConstEvalErr<'tcx>,
        reveal: Reveal,
        span: Span,
        report_fn: F,
    ) -> ErrorHandled<'tcx>
    where
        F: Fn(&ConstEvalErr<'tcx>) -> ErrorHandled<'tcx>,
    {
        match reveal {
            Reveal::Selection => {
                match origin {
                    ConstOrigin::ConstValue => {
                        let res = ConstDedupResult::new(
                            reveal,
                            Err(ConstDedupError::new_silent(SilentError::ConstErr(error), reveal)),
                            Some(span),
                        );

                        self.save_const_value_for_dedup(id, res);
                    }
                    ConstOrigin::Alloc => {
                        let res = ConstDedupResult::new(
                            reveal,
                            Err(ConstDedupError::new_silent(SilentError::ConstErr(error), reveal)),
                            Some(span),
                        );
                        self.save_alloc_for_dedup(id, res);
                    }
                }

                return ErrorHandled::Silent(id);
            }
            _ => {
                let error_handled = report_fn(&error);
                debug!(?error_handled);

                match origin {
                    ConstOrigin::ConstValue => {
                        let res = ConstDedupResult::new(
                            reveal,
                            Err(ConstDedupError::new_handled(error_handled, reveal)),
                            None,
                        );

                        self.save_const_value_for_dedup(id, res);
                    }
                    ConstOrigin::Alloc => {
                        let res = ConstDedupResult::new(
                            reveal,
                            Err(ConstDedupError::new_handled(error_handled, reveal)),
                            None,
                        );

                        self.save_alloc_for_dedup(id, res);
                    }
                }

                error_handled
            }
        }
    }

    /// Stores returned errors from `eval_to_allocation_raw` or `eval_to_const_value_raw`
    /// calls that were already reported.
    #[instrument(skip(self), level = "debug")]
    pub fn handle_reported_error_for_dedup(
        self,
        id: GlobalId<'tcx>,
        origin: ConstOrigin,
        err: ConstErrorEmitted<'tcx>,
        reveal: Reveal,
    ) -> ErrorHandled<'tcx> {
        match err {
            ConstErrorEmitted::Emitted(error_handled) => {
                let dedup_err = ConstDedupError::new_handled(error_handled, reveal);

                match origin {
                    ConstOrigin::ConstValue => {
                        let val = ConstDedupResult::new(reveal, Err(dedup_err), None);
                        self.save_const_value_for_dedup(id, val);
                    }
                    ConstOrigin::Alloc => {
                        let val = ConstDedupResult::new(reveal, Err(dedup_err), None);
                        self.save_alloc_for_dedup(id, val);
                    }
                }

                let error_handled = self.add_error_reported(id, err).get_error();

                error_handled
            }
            ConstErrorEmitted::NotEmitted(_) => bug!(
                "`handle_reported_error_for_dedup` should only be called on emitted `ConstErr`s"
            ),
        }
    }

    fn add_error_reported(
        self,
        id: GlobalId<'tcx>,
        err: ConstErrorEmitted<'tcx>,
    ) -> ConstErrorEmitted<'tcx> {
        let const_dedup_map = self.dedup_const_map.lock();
        const_dedup_map.add_error_reported(id, err)
    }

    #[instrument(skip(self), level = "debug")]
    pub fn report_and_add_error(
        self,
        id: GlobalId<'tcx>,
        err: &ConstEvalErr<'tcx>,
        span: Span,
        msg: &str,
    ) -> ErrorHandled<'tcx> {
        let error_emitted = err.report_as_error(self.at(span), msg);
        debug!(?error_emitted);

        let const_dedup_map = self.dedup_const_map.lock();
        let error_handled = const_dedup_map.add_error_reported(id, error_emitted).get_error();
        debug!(?error_handled);

        error_handled
    }

    /// Store the result of a call to `eval_to_allocation_raw` in order to
    /// allow deduplication.
    #[instrument(skip(self), level = "debug")]
    pub fn save_alloc_for_dedup(
        self,
        id: GlobalId<'tcx>,
        val: ConstDedupResult<'tcx, ConstAlloc<'tcx>>,
    ) {
        let dedup_const_map = self.dedup_const_map.lock();
        dedup_const_map.insert_alloc(id, val);
        debug!("dedup_const_map after insert: {:#?}", dedup_const_map);
    }

    /// Store the result of a call to `eval_to_const_value_raw` in order to deduplicate it.
    #[instrument(skip(self), level = "debug")]
    pub fn save_const_value_for_dedup(
        self,
        id: GlobalId<'tcx>,
        val: ConstDedupResult<'tcx, ConstValue<'tcx>>,
    ) {
        let dedup_const_map = self.dedup_const_map.lock();
        dedup_const_map.insert_const_val(id, val);
        debug!("dedup_const_map after insert: {:#?}", dedup_const_map);
    }

    /// This function reports errors that were returned from calls of
    /// `eval_to_allocation_raw` and stores them in order to allow
    /// errors to be retrieved in deduplication.
    #[instrument(skip(self, dedup_const_map), level = "debug")]
    pub fn report_const_alloc_error(
        self,
        dedup_const_map: &ConstDedupMap<'tcx>,
        id: GlobalId<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        err: &ConstEvalErr<'tcx>,
        is_static: bool,
        def: WithOptConstParam<DefId>,
        span: Span,
    ) -> ConstErrorEmitted<'tcx> {
        // Some CTFE errors raise just a lint, not a hard error; see
        // <https://github.com/rust-lang/rust/issues/71800>.
        let is_hard_err = if let Some(def) = def.as_local() {
            // (Associated) consts only emit a lint, since they might be unused.
            !matches!(self.def_kind(def.did.to_def_id()), DefKind::Const | DefKind::AssocConst)
                // check if the inner InterpError is hard
                || err.error.is_hard_err()
        } else {
            // use of broken constant from other crate: always an error
            true
        };

        if is_hard_err {
            let msg = if is_static {
                Cow::from("could not evaluate static initializer")
            } else {
                // If the current item has generics, we'd like to enrich the message with the
                // instance and its substs: to show the actual compile-time values, in addition to
                // the expression, leading to the const eval error.
                let instance = &id.instance;
                if !instance.substs.is_empty() {
                    let instance = with_no_trimmed_paths(|| instance.to_string());
                    let msg = format!("evaluation of `{}` failed", instance);
                    Cow::from(msg)
                } else {
                    Cow::from("evaluation of constant value failed")
                }
            };

            let e = err.report_as_error(self.at(span), &msg);
            dedup_const_map.add_error_reported(id, e).get_error();

            e
        } else {
            let hir_id = self.hir().local_def_id_to_hir_id(def.as_local().unwrap().did);
            let e = err.report_as_lint(
                self.at(self.def_span(def.did)),
                "any use of this value will cause an error",
                hir_id,
                Some(err.span),
            );

            dedup_const_map.add_error_reported(id, e).get_error();

            e
        }
    }

    /// This function reports errors that were returned from calls of
    /// `eval_to_const_value_raw` and stores them in order to allow
    /// errors to be retrieved in deduplication.
    #[instrument(skip(self), level = "debug")]
    fn report_const_val_error_if_not_already_reported(
        self,
        dedup_const_map: &ConstDedupMap<'tcx>,
        id: GlobalId<'tcx>,
        error: &ConstEvalErr<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        span: Span,
    ) -> ErrorHandled<'tcx> {
        // FIXME: This is really problematic in that it is tightly coupled to the
        // implementation of `eval_to_const_value_raw`. Introducing new errors
        // in that function would have to be considered here. Need to find an
        // abstraction that makes this coupling explicit.
        if let ty::InstanceDef::Intrinsic(_) = id.instance.def {
            let contained = dedup_const_map.error_reported_map.borrow().contains_key(&id);

            if !contained {
                let error_emitted =
                    error.report_as_error(self.at(span), "could not evaluate nullary intrinsic");
                let error_handled =
                    dedup_const_map.add_error_reported(id, error_emitted).get_error();

                error_handled
            } else {
                let error_handled = *dedup_const_map.error_reported_map.borrow().get(&id).unwrap();
                error_handled
            }
        } else {
            let def = id.instance.def.with_opt_param();

            self.report_alloc_error_if_not_already_reported(
                dedup_const_map,
                id,
                error,
                def,
                param_env,
                span,
            )
        }
    }

    /// Tries to deduplicate a call to `eval_to_allocation_raw`. If deduplication isn't
    /// successful `eval_to_allocation_raw` query is executed.
    #[instrument(skip(self, opt_span), level = "debug")]
    pub fn dedup_eval_alloc_raw(
        self,
        key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
        opt_span: Option<Span>,
    ) -> EvalToAllocationRawResult<'tcx> {
        use ConstDedupResult::*;

        let (param_env, id) = key.into_parts();
        let def = id.instance.def.with_opt_param();

        let dedup_const_map = self.dedup_const_map.lock();
        debug!("dedup_const_map: {:#?}", dedup_const_map);
        let alloc_map = dedup_const_map.alloc_map.borrow();
        debug!("alloc_map: {:#?}", alloc_map);

        let dedup_result = alloc_map.get(&id);
        debug!(?dedup_result);

        match param_env.reveal() {
            Reveal::Selection => match dedup_result {
                Some(Selection((Ok(alloc), _))) | Some(UserFacing(Ok(alloc))) => {
                    return Ok(*alloc);
                }
                Some(Selection((Err(err), span))) => {
                    match err {
                        SilentError::ConstErr(const_eval_err) => {
                            match const_eval_err.error {
                                err_inval!(Layout(LayoutError::Unknown(_)))
                                | err_inval!(TooGeneric)
                                | err_inval!(AlreadyReported(_)) => {
                                    // We do want to report these errors even with `Reveal::Selection`

                                    let is_static = self.is_static(def.did);
                                    let err_handled = self
                                        .report_const_alloc_error(
                                            &dedup_const_map,
                                            id,
                                            param_env,
                                            const_eval_err,
                                            is_static,
                                            def,
                                            *span,
                                        )
                                        .get_error();

                                    debug!(?err_handled);

                                    return Err(err_handled);
                                }
                                _ => return Err(ErrorHandled::Silent(id)),
                            }
                        }
                        SilentError::Handled(err) => {
                            return Err(*err);
                        }
                    }
                }
                Some(UserFacing(Err(_)) | All(Err(_))) => {
                    // these errors were previously reported, so we stay silent here
                    // and later access the reported errors using `id`.
                    return Err(ErrorHandled::Silent(id));
                }
                _ => {}
            },
            Reveal::UserFacing => match dedup_result {
                Some(Selection((Ok(alloc), _))) | Some(UserFacing(Ok(alloc))) => {
                    return Ok(*alloc);
                }
                Some(UserFacing(Err(e)) | All(Err(e))) => {
                    return Err(*e);
                }
                Some(Selection((Err(e), span))) => match e {
                    SilentError::ConstErr(const_err) => {
                        let error_handled = self.report_alloc_error_if_not_already_reported(
                            &dedup_const_map,
                            id,
                            const_err,
                            def,
                            param_env,
                            *span,
                        );

                        return Err(error_handled);
                    }
                    SilentError::Handled(error_handled) => return Err(*error_handled),
                },
                _ => {}
            },
            Reveal::All => match dedup_result {
                Some(Selection((Ok(alloc), _)) | UserFacing(Ok(alloc)) | All(Ok(alloc))) => {
                    return Ok(*alloc);
                }
                Some(All(Err(e))) => return Err(*e),
                Some(UserFacing(Err(e))) => match e {
                    ErrorHandled::TooGeneric => {} // run query again with Reveal::All
                    _ => return Err(*e),
                },
                _ => {}
            },
        }

        // Important to drop the lock here
        drop(alloc_map);
        drop(dedup_const_map);

        debug!("unable to deduplicate");

        // We weren't able to deduplicate
        match opt_span {
            Some(span) => self.at(span).eval_to_allocation_raw(key),
            None => self.eval_to_allocation_raw(key),
        }
    }

    /// Tries to deduplicate a call to `eval_to_const_value_raw`. If deduplication isn't
    /// successful, `eval_to_const_value_raw` query is executed.
    #[instrument(skip(self), level = "debug")]
    pub fn dedup_eval_const_value_raw(
        self,
        key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
    ) -> EvalToConstValueResult<'tcx> {
        use ConstDedupResult::*;

        let (param_env, id) = key.into_parts();
        let dedup_const_map = self.dedup_const_map.lock();
        debug!("dedup_const_map: {:#?}", dedup_const_map);
        let const_val_map = dedup_const_map.const_val_map.borrow();
        debug!("const_val_map: {:#?}", const_val_map);

        let dedup_result = const_val_map.get(&id);
        debug!(?dedup_result);

        match param_env.reveal() {
            Reveal::Selection => match dedup_result {
                Some(Selection((Ok(val), _))) | Some(UserFacing(Ok(val))) => {
                    return Ok(*val);
                }
                Some(Selection((Err(_), _)) | UserFacing(Err(_)) | All(Err(_))) => {
                    return Err(ErrorHandled::Silent(id));
                }
                _ => {}
            },
            Reveal::UserFacing => match dedup_result {
                Some(Selection((Ok(val), _))) | Some(UserFacing(Ok(val))) => {
                    return Ok(*val);
                }
                Some(UserFacing(Err(e)) | All(Err(e))) => {
                    return Err(*e);
                }
                Some(Selection((Err(e), span))) => match e {
                    SilentError::ConstErr(const_err) => {
                        let error_handled = self.report_const_val_error_if_not_already_reported(
                            &dedup_const_map,
                            id,
                            const_err,
                            param_env,
                            *span,
                        );

                        return Err(error_handled);
                    }
                    SilentError::Handled(error_handled) => return Err(*error_handled),
                },
                _ => {}
            },
            Reveal::All => match dedup_result {
                Some(Selection((Ok(val), _)) | UserFacing(Ok(val)) | All(Ok(val))) => {
                    return Ok(*val);
                }
                Some(All(Err(e))) => return Err(*e),
                Some(UserFacing(Err(e))) => match e {
                    ErrorHandled::TooGeneric => {} // run query again with Reveal::All
                    _ => return Err(*e),
                },
                _ => {}
            },
        }

        // Important to drop the lock here
        drop(const_val_map);
        drop(dedup_const_map);

        debug!("unable to deduplicate");

        // We weren't able to deduplicate
        self.eval_to_const_value_raw(key)
    }

    #[instrument(skip(self), level = "debug")]
    pub fn report_alloc_error(
        self,
        id: GlobalId<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        err: &ConstEvalErr<'tcx>,
        is_static: bool,
        def: WithOptConstParam<DefId>,
        span: Span,
    ) -> ErrorHandled<'tcx> {
        let dedup_const_map = self.dedup_const_map.lock();
        let error_handled = self
            .report_const_alloc_error(&dedup_const_map, id, param_env, err, is_static, def, span)
            .get_error();

        error_handled
    }

    #[instrument(skip(self, dedup_const_map), level = "debug")]
    fn report_alloc_error_if_not_already_reported(
        self,
        dedup_const_map: &ConstDedupMap<'tcx>,
        id: GlobalId<'tcx>,
        e: &ConstEvalErr<'tcx>,
        def: WithOptConstParam<DefId>,
        param_env: ty::ParamEnv<'tcx>,
        span: Span,
    ) -> ErrorHandled<'tcx> {
        let stored_error = dedup_const_map.error_reported_map.borrow().get(&id).cloned();
        match stored_error {
            Some(err) => err,
            None => {
                let is_static = self.is_static(def.did);
                let def = id.instance.def.with_opt_param();

                let error_handled = self
                    .report_const_alloc_error(
                        dedup_const_map,
                        id,
                        param_env,
                        e,
                        is_static,
                        def,
                        span,
                    )
                    .get_error();

                error_handled
            }
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
