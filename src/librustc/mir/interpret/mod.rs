// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An interpreter for MIR used in CTFE and by miri

#[macro_export]
macro_rules! err {
    ($($tt:tt)*) => { Err($crate::mir::interpret::EvalErrorKind::$($tt)*.into()) };
}

mod error;
mod value;

pub use self::error::{
    EvalError, EvalResult, EvalErrorKind, AssertMessage, ConstEvalErr, struct_error,
    FrameInfo, ConstEvalResult,
};

pub use self::value::{Scalar, ConstValue, ScalarMaybeUndef};

use std::fmt;
use mir;
use hir::def_id::DefId;
use ty::{self, TyCtxt, Instance};
use ty::layout::{self, Align, HasDataLayout, Size};
use middle::region;
use std::iter;
use std::io;
use std::ops::{Deref, DerefMut};
use std::hash::Hash;
use syntax::ast::Mutability;
use rustc_serialize::{Encoder, Decodable, Encodable};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::{Lock as Mutex, HashMapExt};
use rustc_data_structures::tiny_list::TinyList;
use byteorder::{WriteBytesExt, ReadBytesExt, LittleEndian, BigEndian};
use ty::codec::TyDecoder;
use std::sync::atomic::{AtomicU32, Ordering};
use std::num::NonZeroU32;

#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum Lock {
    NoLock,
    WriteLock(DynamicLifetime),
    /// This should never be empty -- that would be a read lock held and nobody
    /// there to release it...
    ReadLock(Vec<DynamicLifetime>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct DynamicLifetime {
    pub frame: usize,
    pub region: Option<region::Scope>, // "None" indicates "until the function ends"
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum AccessKind {
    Read,
    Write,
}

/// Uniquely identifies a specific constant or static.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub struct GlobalId<'tcx> {
    /// For a constant or static, the `Instance` of the item itself.
    /// For a promoted global, the `Instance` of the function they belong to.
    pub instance: ty::Instance<'tcx>,

    /// The index for promoted globals within their function's `Mir`.
    pub promoted: Option<mir::Promoted>,
}

////////////////////////////////////////////////////////////////////////////////
// Pointer arithmetic
////////////////////////////////////////////////////////////////////////////////

pub trait PointerArithmetic: layout::HasDataLayout {
    // These are not supposed to be overridden.

    //// Trunace the given value to the pointer size; also return whether there was an overflow
    fn truncate_to_ptr(self, val: u128) -> (u64, bool) {
        let max_ptr_plus_1 = 1u128 << self.data_layout().pointer_size.bits();
        ((val % max_ptr_plus_1) as u64, val >= max_ptr_plus_1)
    }

    // Overflow checking only works properly on the range from -u64 to +u64.
    fn overflowing_signed_offset(self, val: u64, i: i128) -> (u64, bool) {
        // FIXME: is it possible to over/underflow here?
        if i < 0 {
            // trickery to ensure that i64::min_value() works fine
            // this formula only works for true negative values, it panics for zero!
            let n = u64::max_value() - (i as u64) + 1;
            val.overflowing_sub(n)
        } else {
            self.overflowing_offset(val, i as u64)
        }
    }

    fn overflowing_offset(self, val: u64, i: u64) -> (u64, bool) {
        let (res, over1) = val.overflowing_add(i);
        let (res, over2) = self.truncate_to_ptr(res as u128);
        (res, over1 || over2)
    }

    fn signed_offset<'tcx>(self, val: u64, i: i64) -> EvalResult<'tcx, u64> {
        let (res, over) = self.overflowing_signed_offset(val, i as i128);
        if over { err!(Overflow(mir::BinOp::Add)) } else { Ok(res) }
    }

    fn offset<'tcx>(self, val: u64, i: u64) -> EvalResult<'tcx, u64> {
        let (res, over) = self.overflowing_offset(val, i);
        if over { err!(Overflow(mir::BinOp::Add)) } else { Ok(res) }
    }

    fn wrapping_signed_offset(self, val: u64, i: i64) -> u64 {
        self.overflowing_signed_offset(val, i as i128).0
    }
}

impl<T: layout::HasDataLayout> PointerArithmetic for T {}


#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub struct Pointer {
    pub alloc_id: AllocId,
    pub offset: Size,
}

/// Produces a `Pointer` which points to the beginning of the Allocation
impl From<AllocId> for Pointer {
    fn from(alloc_id: AllocId) -> Self {
        Pointer::new(alloc_id, Size::ZERO)
    }
}

impl<'tcx> Pointer {
    pub fn new(alloc_id: AllocId, offset: Size) -> Self {
        Pointer { alloc_id, offset }
    }

    pub fn wrapping_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> Self {
        Pointer::new(
            self.alloc_id,
            Size::from_bytes(cx.data_layout().wrapping_signed_offset(self.offset.bytes(), i)),
        )
    }

    pub fn overflowing_signed_offset<C: HasDataLayout>(self, i: i128, cx: C) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_signed_offset(self.offset.bytes(), i);
        (Pointer::new(self.alloc_id, Size::from_bytes(res)), over)
    }

    pub fn signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        Ok(Pointer::new(
            self.alloc_id,
            Size::from_bytes(cx.data_layout().signed_offset(self.offset.bytes(), i)?),
        ))
    }

    pub fn overflowing_offset<C: HasDataLayout>(self, i: Size, cx: C) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_offset(self.offset.bytes(), i.bytes());
        (Pointer::new(self.alloc_id, Size::from_bytes(res)), over)
    }

    pub fn offset<C: HasDataLayout>(self, i: Size, cx: C) -> EvalResult<'tcx, Self> {
        Ok(Pointer::new(
            self.alloc_id,
            Size::from_bytes(cx.data_layout().offset(self.offset.bytes(), i.bytes())?),
        ))
    }
}


#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Debug)]
pub struct AllocId(pub u64);

impl ::rustc_serialize::UseSpecializedEncodable for AllocId {}
impl ::rustc_serialize::UseSpecializedDecodable for AllocId {}

#[derive(RustcDecodable, RustcEncodable)]
enum AllocKind {
    Alloc,
    Fn,
    Static,
}

pub fn specialized_encode_alloc_id<
    'a, 'tcx,
    E: Encoder,
>(
    encoder: &mut E,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    alloc_id: AllocId,
) -> Result<(), E::Error> {
    let alloc_type: AllocType<'tcx, &'tcx Allocation> =
        tcx.alloc_map.lock().get(alloc_id).expect("no value for AllocId");
    match alloc_type {
        AllocType::Memory(alloc) => {
            trace!("encoding {:?} with {:#?}", alloc_id, alloc);
            AllocKind::Alloc.encode(encoder)?;
            alloc.encode(encoder)?;
        }
        AllocType::Function(fn_instance) => {
            trace!("encoding {:?} with {:#?}", alloc_id, fn_instance);
            AllocKind::Fn.encode(encoder)?;
            fn_instance.encode(encoder)?;
        }
        AllocType::Static(did) => {
            // referring to statics doesn't need to know about their allocations,
            // just about its DefId
            AllocKind::Static.encode(encoder)?;
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
    // For each AllocId we keep track of which decoding state it's currently in.
    decoding_state: Vec<Mutex<State>>,
    // The offsets of each allocation in the data stream.
    data_offsets: Vec<u32>,
}

impl AllocDecodingState {

    pub fn new_decoding_session(&self) -> AllocDecodingSession {
        static DECODER_SESSION_ID: AtomicU32 = AtomicU32::new(0);
        let counter = DECODER_SESSION_ID.fetch_add(1, Ordering::SeqCst);

        // Make sure this is never zero
        let session_id = DecodingSessionId::new((counter & 0x7FFFFFFF) + 1).unwrap();

        AllocDecodingSession {
            state: self,
            session_id,
        }
    }

    pub fn new(data_offsets: Vec<u32>) -> AllocDecodingState {
        let decoding_state: Vec<_> = ::std::iter::repeat(Mutex::new(State::Empty))
            .take(data_offsets.len())
            .collect();

        AllocDecodingState {
            decoding_state: decoding_state,
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

    // Decodes an AllocId in a thread-safe way.
    pub fn decode_alloc_id<'a, 'tcx, D>(&self,
                                        decoder: &mut D)
                                        -> Result<AllocId, D::Error>
        where D: TyDecoder<'a, 'tcx>,
              'tcx: 'a,
    {
        // Read the index of the allocation
        let idx = decoder.read_u32()? as usize;
        let pos = self.state.data_offsets[idx] as usize;

        // Decode the AllocKind now so that we know if we have to reserve an
        // AllocId.
        let (alloc_kind, pos) = decoder.with_position(pos, |decoder| {
            let alloc_kind = AllocKind::decode(decoder)?;
            Ok((alloc_kind, decoder.position()))
        })?;

        // Check the decoding state, see if it's already decoded or if we should
        // decode it here.
        let alloc_id = {
            let mut entry = self.state.decoding_state[idx].lock();

            match *entry {
                State::Done(alloc_id) => {
                    return Ok(alloc_id);
                }
                ref mut entry @ State::Empty => {
                    // We are allowed to decode
                    match alloc_kind {
                        AllocKind::Alloc => {
                            // If this is an allocation, we need to reserve an
                            // AllocId so we can decode cyclic graphs.
                            let alloc_id = decoder.tcx().alloc_map.lock().reserve();
                            *entry = State::InProgress(
                                TinyList::new_single(self.session_id),
                                alloc_id);
                            Some(alloc_id)
                        },
                        AllocKind::Fn | AllocKind::Static => {
                            // Fns and statics cannot be cyclic and their AllocId
                            // is determined later by interning
                            *entry = State::InProgressNonAlloc(
                                TinyList::new_single(self.session_id));
                            None
                        }
                    }
                }
                State::InProgressNonAlloc(ref mut sessions) => {
                    if sessions.contains(&self.session_id) {
                        bug!("This should be unreachable")
                    } else {
                        // Start decoding concurrently
                        sessions.insert(self.session_id);
                        None
                    }
                }
                State::InProgress(ref mut sessions, alloc_id) => {
                    if sessions.contains(&self.session_id) {
                        // Don't recurse.
                        return Ok(alloc_id)
                    } else {
                        // Start decoding concurrently
                        sessions.insert(self.session_id);
                        Some(alloc_id)
                    }
                }
            }
        };

        // Now decode the actual data
        let alloc_id = decoder.with_position(pos, |decoder| {
            match alloc_kind {
                AllocKind::Alloc => {
                    let allocation = <&'tcx Allocation as Decodable>::decode(decoder)?;
                    // We already have a reserved AllocId.
                    let alloc_id = alloc_id.unwrap();
                    trace!("decoded alloc {:?} {:#?}", alloc_id, allocation);
                    decoder.tcx().alloc_map.lock().set_id_same_memory(alloc_id, allocation);
                    Ok(alloc_id)
                },
                AllocKind::Fn => {
                    assert!(alloc_id.is_none());
                    trace!("creating fn alloc id");
                    let instance = ty::Instance::decode(decoder)?;
                    trace!("decoded fn alloc instance: {:?}", instance);
                    let alloc_id = decoder.tcx().alloc_map.lock().create_fn_alloc(instance);
                    Ok(alloc_id)
                },
                AllocKind::Static => {
                    assert!(alloc_id.is_none());
                    trace!("creating extern static alloc id at");
                    let did = DefId::decode(decoder)?;
                    let alloc_id = decoder.tcx().alloc_map.lock().intern_static(did);
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, RustcDecodable, RustcEncodable)]
pub enum AllocType<'tcx, M> {
    /// The alloc id is used as a function pointer
    Function(Instance<'tcx>),
    /// The alloc id points to a static variable
    Static(DefId),
    /// The alloc id points to memory
    Memory(M)
}

pub struct AllocMap<'tcx, M> {
    /// Lets you know what an AllocId refers to
    id_to_type: FxHashMap<AllocId, AllocType<'tcx, M>>,

    /// Used to ensure that functions and statics only get one associated AllocId
    type_interner: FxHashMap<AllocType<'tcx, M>, AllocId>,

    /// The AllocId to assign to the next requested id.
    /// Always incremented, never gets smaller.
    next_id: AllocId,
}

impl<'tcx, M: fmt::Debug + Eq + Hash + Clone> AllocMap<'tcx, M> {
    pub fn new() -> Self {
        AllocMap {
            id_to_type: FxHashMap(),
            type_interner: FxHashMap(),
            next_id: AllocId(0),
        }
    }

    /// obtains a new allocation ID that can be referenced but does not
    /// yet have an allocation backing it.
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

    fn intern(&mut self, alloc_type: AllocType<'tcx, M>) -> AllocId {
        if let Some(&alloc_id) = self.type_interner.get(&alloc_type) {
            return alloc_id;
        }
        let id = self.reserve();
        debug!("creating alloc_type {:?} with id {}", alloc_type, id);
        self.id_to_type.insert(id, alloc_type.clone());
        self.type_interner.insert(alloc_type, id);
        id
    }

    // FIXME: Check if functions have identity. If not, we should not intern these,
    // but instead create a new id per use.
    // Alternatively we could just make comparing function pointers an error.
    pub fn create_fn_alloc(&mut self, instance: Instance<'tcx>) -> AllocId {
        self.intern(AllocType::Function(instance))
    }

    pub fn get(&self, id: AllocId) -> Option<AllocType<'tcx, M>> {
        self.id_to_type.get(&id).cloned()
    }

    pub fn unwrap_memory(&self, id: AllocId) -> M {
        match self.get(id) {
            Some(AllocType::Memory(mem)) => mem,
            _ => bug!("expected allocation id {} to point to memory", id),
        }
    }

    pub fn intern_static(&mut self, static_id: DefId) -> AllocId {
        self.intern(AllocType::Static(static_id))
    }

    pub fn allocate(&mut self, mem: M) -> AllocId {
        let id = self.reserve();
        self.set_id_memory(id, mem);
        id
    }

    pub fn set_id_memory(&mut self, id: AllocId, mem: M) {
        if let Some(old) = self.id_to_type.insert(id, AllocType::Memory(mem)) {
            bug!("tried to set allocation id {}, but it was already existing as {:#?}", id, old);
        }
    }

    pub fn set_id_same_memory(&mut self, id: AllocId, mem: M) {
       self.id_to_type.insert_same(id, AllocType::Memory(mem));
    }
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub struct Allocation {
    /// The actual bytes of the allocation.
    /// Note that the bytes of a pointer represent the offset of the pointer
    pub bytes: Vec<u8>,
    /// Maps from byte addresses to allocations.
    /// Only the first byte of a pointer is inserted into the map.
    pub relocations: Relocations,
    /// Denotes undefined memory. Reading from undefined memory is forbidden in miri
    pub undef_mask: UndefMask,
    /// The alignment of the allocation to detect unaligned reads.
    pub align: Align,
    /// Whether the allocation (of a static) should be put into mutable memory when codegenning
    ///
    /// Only happens for `static mut` or `static` with interior mutability
    pub runtime_mutability: Mutability,
}

impl Allocation {
    pub fn from_bytes(slice: &[u8], align: Align) -> Self {
        let mut undef_mask = UndefMask::new(Size::ZERO);
        undef_mask.grow(Size::from_bytes(slice.len() as u64), true);
        Self {
            bytes: slice.to_owned(),
            relocations: Relocations::new(),
            undef_mask,
            align,
            runtime_mutability: Mutability::Immutable,
        }
    }

    pub fn from_byte_aligned_bytes(slice: &[u8]) -> Self {
        Allocation::from_bytes(slice, Align::from_bytes(1, 1).unwrap())
    }

    pub fn undef(size: Size, align: Align) -> Self {
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        Allocation {
            bytes: vec![0; size.bytes() as usize],
            relocations: Relocations::new(),
            undef_mask: UndefMask::new(size),
            align,
            runtime_mutability: Mutability::Immutable,
        }
    }
}

impl<'tcx> ::serialize::UseSpecializedDecodable for &'tcx Allocation {}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct Relocations(SortedMap<Size, AllocId>);

impl Relocations {
    pub fn new() -> Relocations {
        Relocations(SortedMap::new())
    }

    // The caller must guarantee that the given relocations are already sorted
    // by address and contain no duplicates.
    pub fn from_presorted(r: Vec<(Size, AllocId)>) -> Relocations {
        Relocations(SortedMap::from_presorted_elements(r))
    }
}

impl Deref for Relocations {
    type Target = SortedMap<Size, AllocId>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Relocations {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Methods to access integers in the target endianness
////////////////////////////////////////////////////////////////////////////////

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

pub fn read_target_uint(endianness: layout::Endian, mut source: &[u8]) -> Result<u128, io::Error> {
    match endianness {
        layout::Endian::Little => source.read_uint128::<LittleEndian>(source.len()),
        layout::Endian::Big => source.read_uint128::<BigEndian>(source.len()),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Methods to faciliate working with signed integers stored in a u128
////////////////////////////////////////////////////////////////////////////////

pub fn sign_extend(value: u128, size: Size) -> u128 {
    let size = size.bits();
    // sign extend
    let shift = 128 - size;
    // shift the unsigned value to the left
    // and back to the right as signed (essentially fills with FF on the left)
    (((value << shift) as i128) >> shift) as u128
}

pub fn truncate(value: u128, size: Size) -> u128 {
    let size = size.bits();
    let shift = 128 - size;
    // truncate (shift left to drop out leftover values, shift right to fill with zeroes)
    (value << shift) >> shift
}

////////////////////////////////////////////////////////////////////////////////
// Undefined byte tracking
////////////////////////////////////////////////////////////////////////////////

type Block = u64;
const BLOCK_SIZE: u64 = 64;

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub struct UndefMask {
    blocks: Vec<Block>,
    len: Size,
}

impl_stable_hash_for!(struct mir::interpret::UndefMask{blocks, len});

impl UndefMask {
    pub fn new(size: Size) -> Self {
        let mut m = UndefMask {
            blocks: vec![],
            len: Size::ZERO,
        };
        m.grow(size, false);
        m
    }

    /// Check whether the range `start..end` (end-exclusive) is entirely defined.
    pub fn is_range_defined(&self, start: Size, end: Size) -> bool {
        if end > self.len {
            return false;
        }
        for i in start.bytes()..end.bytes() {
            if !self.get(Size::from_bytes(i)) {
                return false;
            }
        }
        true
    }

    pub fn set_range(&mut self, start: Size, end: Size, new_state: bool) {
        let len = self.len;
        if end > len {
            self.grow(end - len, new_state);
        }
        self.set_range_inbounds(start, end, new_state);
    }

    pub fn set_range_inbounds(&mut self, start: Size, end: Size, new_state: bool) {
        for i in start.bytes()..end.bytes() {
            self.set(Size::from_bytes(i), new_state);
        }
    }

    #[inline]
    pub fn get(&self, i: Size) -> bool {
        let (block, bit) = bit_index(i);
        (self.blocks[block] & 1 << bit) != 0
    }

    #[inline]
    pub fn set(&mut self, i: Size, new_state: bool) {
        let (block, bit) = bit_index(i);
        if new_state {
            self.blocks[block] |= 1 << bit;
        } else {
            self.blocks[block] &= !(1 << bit);
        }
    }

    pub fn grow(&mut self, amount: Size, new_state: bool) {
        let unused_trailing_bits = self.blocks.len() as u64 * BLOCK_SIZE - self.len.bytes();
        if amount.bytes() > unused_trailing_bits {
            let additional_blocks = amount.bytes() / BLOCK_SIZE + 1;
            assert_eq!(additional_blocks as usize as u64, additional_blocks);
            self.blocks.extend(
                iter::repeat(0).take(additional_blocks as usize),
            );
        }
        let start = self.len;
        self.len += amount;
        self.set_range_inbounds(start, start + amount, new_state);
    }
}

#[inline]
fn bit_index(bits: Size) -> (usize, usize) {
    let bits = bits.bytes();
    let a = bits / BLOCK_SIZE;
    let b = bits % BLOCK_SIZE;
    assert_eq!(a as usize as u64, a);
    assert_eq!(b as usize as u64, b);
    (a as usize, b as usize)
}
