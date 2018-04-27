//! An interpreter for MIR used in CTFE and by miri

#[macro_export]
macro_rules! err {
    ($($tt:tt)*) => { Err($crate::mir::interpret::EvalErrorKind::$($tt)*.into()) };
}

mod error;
mod value;

pub use self::error::{EvalError, EvalResult, EvalErrorKind, AssertMessage};

pub use self::value::{PrimVal, PrimValKind, Value, Pointer};

use std::collections::BTreeMap;
use std::fmt;
use mir;
use hir::def_id::DefId;
use ty::{self, TyCtxt};
use ty::layout::{self, Align, HasDataLayout};
use middle::region;
use std::iter;
use syntax::ast::Mutability;
use rustc_serialize::{Encoder, Decoder, Decodable, Encodable};

#[derive(Clone, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub enum Lock {
    NoLock,
    WriteLock(DynamicLifetime),
    /// This should never be empty -- that would be a read lock held and nobody there to release it...
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


#[derive(Copy, Clone, Debug, Eq, PartialEq, RustcEncodable, RustcDecodable, Hash)]
pub struct MemoryPointer {
    pub alloc_id: AllocId,
    pub offset: u64,
}

impl<'tcx> MemoryPointer {
    pub fn new(alloc_id: AllocId, offset: u64) -> Self {
        MemoryPointer { alloc_id, offset }
    }

    pub(crate) fn wrapping_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> Self {
        MemoryPointer::new(
            self.alloc_id,
            cx.data_layout().wrapping_signed_offset(self.offset, i),
        )
    }

    pub fn overflowing_signed_offset<C: HasDataLayout>(self, i: i128, cx: C) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_signed_offset(self.offset, i);
        (MemoryPointer::new(self.alloc_id, res), over)
    }

    pub(crate) fn signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        Ok(MemoryPointer::new(
            self.alloc_id,
            cx.data_layout().signed_offset(self.offset, i)?,
        ))
    }

    pub fn overflowing_offset<C: HasDataLayout>(self, i: u64, cx: C) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_offset(self.offset, i);
        (MemoryPointer::new(self.alloc_id, res), over)
    }

    pub fn offset<C: HasDataLayout>(self, i: u64, cx: C) -> EvalResult<'tcx, Self> {
        Ok(MemoryPointer::new(
            self.alloc_id,
            cx.data_layout().offset(self.offset, i)?,
        ))
    }
}


#[derive(Copy, Clone, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Debug)]
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
    if let Some(alloc) = tcx.interpret_interner.get_alloc(alloc_id) {
        trace!("encoding {:?} with {:#?}", alloc_id, alloc);
        AllocKind::Alloc.encode(encoder)?;
        alloc.encode(encoder)?;
    } else if let Some(fn_instance) = tcx.interpret_interner.get_fn(alloc_id) {
        trace!("encoding {:?} with {:#?}", alloc_id, fn_instance);
        AllocKind::Fn.encode(encoder)?;
        fn_instance.encode(encoder)?;
    } else if let Some(did) = tcx.interpret_interner.get_static(alloc_id) {
        // referring to statics doesn't need to know about their allocations, just about its DefId
        AllocKind::Static.encode(encoder)?;
        did.encode(encoder)?;
    } else {
        bug!("alloc id without corresponding allocation: {}", alloc_id);
    }
    Ok(())
}

pub fn specialized_decode_alloc_id<
    'a, 'tcx,
    D: Decoder,
    CACHE: FnOnce(&mut D, AllocId),
>(
    decoder: &mut D,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cache: CACHE,
) -> Result<AllocId, D::Error> {
    match AllocKind::decode(decoder)? {
        AllocKind::Alloc => {
            let alloc_id = tcx.interpret_interner.reserve();
            trace!("creating alloc id {:?}", alloc_id);
            // insert early to allow recursive allocs
            cache(decoder, alloc_id);

            let allocation = Allocation::decode(decoder)?;
            trace!("decoded alloc {:?} {:#?}", alloc_id, allocation);
            let allocation = tcx.intern_const_alloc(allocation);
            tcx.interpret_interner.intern_at_reserved(alloc_id, allocation);

            Ok(alloc_id)
        },
        AllocKind::Fn => {
            trace!("creating fn alloc id");
            let instance = ty::Instance::decode(decoder)?;
            trace!("decoded fn alloc instance: {:?}", instance);
            let id = tcx.interpret_interner.create_fn_alloc(instance);
            trace!("created fn alloc id: {:?}", id);
            cache(decoder, id);
            Ok(id)
        },
        AllocKind::Static => {
            trace!("creating extern static alloc id at");
            let did = DefId::decode(decoder)?;
            let alloc_id = tcx.interpret_interner.cache_static(did);
            cache(decoder, alloc_id);
            Ok(alloc_id)
        },
    }
}

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Eq, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub struct Allocation {
    /// The actual bytes of the allocation.
    /// Note that the bytes of a pointer represent the offset of the pointer
    pub bytes: Vec<u8>,
    /// Maps from byte addresses to allocations.
    /// Only the first byte of a pointer is inserted into the map.
    pub relocations: BTreeMap<u64, AllocId>,
    /// Denotes undefined memory. Reading from undefined memory is forbidden in miri
    pub undef_mask: UndefMask,
    /// The alignment of the allocation to detect unaligned reads.
    pub align: Align,
    /// Whether the allocation (of a static) should be put into mutable memory when translating
    ///
    /// Only happens for `static mut` or `static` with interior mutability
    pub runtime_mutability: Mutability,
}

impl Allocation {
    pub fn from_bytes(slice: &[u8]) -> Self {
        let mut undef_mask = UndefMask::new(0);
        undef_mask.grow(slice.len() as u64, true);
        Self {
            bytes: slice.to_owned(),
            relocations: BTreeMap::new(),
            undef_mask,
            align: Align::from_bytes(1, 1).unwrap(),
            runtime_mutability: Mutability::Immutable,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Undefined byte tracking
////////////////////////////////////////////////////////////////////////////////

type Block = u64;
const BLOCK_SIZE: u64 = 64;

#[derive(Clone, Debug, Eq, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub struct UndefMask {
    blocks: Vec<Block>,
    len: u64,
}

impl_stable_hash_for!(struct mir::interpret::UndefMask{blocks, len});

impl UndefMask {
    pub fn new(size: u64) -> Self {
        let mut m = UndefMask {
            blocks: vec![],
            len: 0,
        };
        m.grow(size, false);
        m
    }

    /// Check whether the range `start..end` (end-exclusive) is entirely defined.
    pub fn is_range_defined(&self, start: u64, end: u64) -> bool {
        if end > self.len {
            return false;
        }
        for i in start..end {
            if !self.get(i) {
                return false;
            }
        }
        true
    }

    pub fn set_range(&mut self, start: u64, end: u64, new_state: bool) {
        let len = self.len;
        if end > len {
            self.grow(end - len, new_state);
        }
        self.set_range_inbounds(start, end, new_state);
    }

    pub fn set_range_inbounds(&mut self, start: u64, end: u64, new_state: bool) {
        for i in start..end {
            self.set(i, new_state);
        }
    }

    pub fn get(&self, i: u64) -> bool {
        let (block, bit) = bit_index(i);
        (self.blocks[block] & 1 << bit) != 0
    }

    pub fn set(&mut self, i: u64, new_state: bool) {
        let (block, bit) = bit_index(i);
        if new_state {
            self.blocks[block] |= 1 << bit;
        } else {
            self.blocks[block] &= !(1 << bit);
        }
    }

    pub fn grow(&mut self, amount: u64, new_state: bool) {
        let unused_trailing_bits = self.blocks.len() as u64 * BLOCK_SIZE - self.len;
        if amount > unused_trailing_bits {
            let additional_blocks = amount / BLOCK_SIZE + 1;
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

fn bit_index(bits: u64) -> (usize, usize) {
    let a = bits / BLOCK_SIZE;
    let b = bits % BLOCK_SIZE;
    assert_eq!(a as usize as u64, a);
    assert_eq!(b as usize as u64, b);
    (a as usize, b as usize)
}
