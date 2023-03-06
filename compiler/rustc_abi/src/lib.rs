#![cfg_attr(feature = "nightly", feature(step_trait, rustc_attrs, min_specialization))]

use std::fmt;
#[cfg(feature = "nightly")]
use std::iter::Step;
use std::num::{NonZeroUsize, ParseIntError};
use std::ops::{Add, AddAssign, Mul, RangeInclusive, Sub};
use std::str::FromStr;

use bitflags::bitflags;
use rustc_data_structures::intern::Interned;
#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::StableOrd;
use rustc_index::vec::{Idx, IndexVec};
#[cfg(feature = "nightly")]
use rustc_macros::HashStable_Generic;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable, Encodable};

mod layout;

pub use layout::LayoutCalculator;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext {}

use Integer::*;
use Primitive::*;

bitflags! {
    #[derive(Default)]
    #[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_Generic))]
    pub struct ReprFlags: u8 {
        const IS_C               = 1 << 0;
        const IS_SIMD            = 1 << 1;
        const IS_TRANSPARENT     = 1 << 2;
        // Internal only for now. If true, don't reorder fields.
        const IS_LINEAR          = 1 << 3;
        // If true, the type's layout can be randomized using
        // the seed stored in `ReprOptions.layout_seed`
        const RANDOMIZE_LAYOUT   = 1 << 4;
        // Any of these flags being set prevent field reordering optimisation.
        const IS_UNOPTIMISABLE   = ReprFlags::IS_C.bits
                                 | ReprFlags::IS_SIMD.bits
                                 | ReprFlags::IS_LINEAR.bits;
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_Generic))]
pub enum IntegerType {
    /// Pointer sized integer type, i.e. isize and usize. The field shows signedness, that
    /// is, `Pointer(true)` is isize.
    Pointer(bool),
    /// Fix sized integer type, e.g. i8, u32, i128 The bool field shows signedness, `Fixed(I8, false)` means `u8`
    Fixed(Integer, bool),
}

impl IntegerType {
    pub fn is_signed(&self) -> bool {
        match self {
            IntegerType::Pointer(b) => *b,
            IntegerType::Fixed(_, b) => *b,
        }
    }
}

/// Represents the repr options provided by the user,
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_Generic))]
pub struct ReprOptions {
    pub int: Option<IntegerType>,
    pub align: Option<Align>,
    pub pack: Option<Align>,
    pub flags: ReprFlags,
    /// The seed to be used for randomizing a type's layout
    ///
    /// Note: This could technically be a `[u8; 16]` (a `u128`) which would
    /// be the "most accurate" hash as it'd encompass the item and crate
    /// hash without loss, but it does pay the price of being larger.
    /// Everything's a tradeoff, a `u64` seed should be sufficient for our
    /// purposes (primarily `-Z randomize-layout`)
    pub field_shuffle_seed: u64,
}

impl ReprOptions {
    #[inline]
    pub fn simd(&self) -> bool {
        self.flags.contains(ReprFlags::IS_SIMD)
    }

    #[inline]
    pub fn c(&self) -> bool {
        self.flags.contains(ReprFlags::IS_C)
    }

    #[inline]
    pub fn packed(&self) -> bool {
        self.pack.is_some()
    }

    #[inline]
    pub fn transparent(&self) -> bool {
        self.flags.contains(ReprFlags::IS_TRANSPARENT)
    }

    #[inline]
    pub fn linear(&self) -> bool {
        self.flags.contains(ReprFlags::IS_LINEAR)
    }

    /// Returns the discriminant type, given these `repr` options.
    /// This must only be called on enums!
    pub fn discr_type(&self) -> IntegerType {
        self.int.unwrap_or(IntegerType::Pointer(true))
    }

    /// Returns `true` if this `#[repr()]` should inhabit "smart enum
    /// layout" optimizations, such as representing `Foo<&T>` as a
    /// single pointer.
    pub fn inhibit_enum_layout_opt(&self) -> bool {
        self.c() || self.int.is_some()
    }

    /// Returns `true` if this `#[repr()]` should inhibit struct field reordering
    /// optimizations, such as with `repr(C)`, `repr(packed(1))`, or `repr(<int>)`.
    pub fn inhibit_struct_field_reordering_opt(&self) -> bool {
        if let Some(pack) = self.pack {
            if pack.bytes() == 1 {
                return true;
            }
        }

        self.flags.intersects(ReprFlags::IS_UNOPTIMISABLE) || self.int.is_some()
    }

    /// Returns `true` if this type is valid for reordering and `-Z randomize-layout`
    /// was enabled for its declaration crate
    pub fn can_randomize_type_layout(&self) -> bool {
        !self.inhibit_struct_field_reordering_opt()
            && self.flags.contains(ReprFlags::RANDOMIZE_LAYOUT)
    }

    /// Returns `true` if this `#[repr()]` should inhibit union ABI optimisations.
    pub fn inhibit_union_abi_opt(&self) -> bool {
        self.c()
    }
}

/// Parsed [Data layout](https://llvm.org/docs/LangRef.html#data-layout)
/// for a target, which contains everything needed to compute layouts.
#[derive(Debug, PartialEq, Eq)]
pub struct TargetDataLayout {
    pub endian: Endian,
    pub i1_align: AbiAndPrefAlign,
    pub i8_align: AbiAndPrefAlign,
    pub i16_align: AbiAndPrefAlign,
    pub i32_align: AbiAndPrefAlign,
    pub i64_align: AbiAndPrefAlign,
    pub i128_align: AbiAndPrefAlign,
    pub f32_align: AbiAndPrefAlign,
    pub f64_align: AbiAndPrefAlign,
    pub pointer_size: Size,
    pub pointer_align: AbiAndPrefAlign,
    pub aggregate_align: AbiAndPrefAlign,

    /// Alignments for vector types.
    pub vector_align: Vec<(Size, AbiAndPrefAlign)>,

    pub instruction_address_space: AddressSpace,

    /// Minimum size of #[repr(C)] enums (default c_int::BITS, usually 32)
    /// Note: This isn't in LLVM's data layout string, it is `short_enum`
    /// so the only valid spec for LLVM is c_int::BITS or 8
    pub c_enum_min_size: Integer,
}

impl Default for TargetDataLayout {
    /// Creates an instance of `TargetDataLayout`.
    fn default() -> TargetDataLayout {
        let align = |bits| Align::from_bits(bits).unwrap();
        TargetDataLayout {
            endian: Endian::Big,
            i1_align: AbiAndPrefAlign::new(align(8)),
            i8_align: AbiAndPrefAlign::new(align(8)),
            i16_align: AbiAndPrefAlign::new(align(16)),
            i32_align: AbiAndPrefAlign::new(align(32)),
            i64_align: AbiAndPrefAlign { abi: align(32), pref: align(64) },
            i128_align: AbiAndPrefAlign { abi: align(32), pref: align(64) },
            f32_align: AbiAndPrefAlign::new(align(32)),
            f64_align: AbiAndPrefAlign::new(align(64)),
            pointer_size: Size::from_bits(64),
            pointer_align: AbiAndPrefAlign::new(align(64)),
            aggregate_align: AbiAndPrefAlign { abi: align(0), pref: align(64) },
            vector_align: vec![
                (Size::from_bits(64), AbiAndPrefAlign::new(align(64))),
                (Size::from_bits(128), AbiAndPrefAlign::new(align(128))),
            ],
            instruction_address_space: AddressSpace::DATA,
            c_enum_min_size: Integer::I32,
        }
    }
}

pub enum TargetDataLayoutErrors<'a> {
    InvalidAddressSpace { addr_space: &'a str, cause: &'a str, err: ParseIntError },
    InvalidBits { kind: &'a str, bit: &'a str, cause: &'a str, err: ParseIntError },
    MissingAlignment { cause: &'a str },
    InvalidAlignment { cause: &'a str, err: String },
    InconsistentTargetArchitecture { dl: &'a str, target: &'a str },
    InconsistentTargetPointerWidth { pointer_size: u64, target: u32 },
    InvalidBitsSize { err: String },
}

impl TargetDataLayout {
    /// Parse data layout from an [llvm data layout string](https://llvm.org/docs/LangRef.html#data-layout)
    ///
    /// This function doesn't fill `c_enum_min_size` and it will always be `I32` since it can not be
    /// determined from llvm string.
    pub fn parse_from_llvm_datalayout_string<'a>(
        input: &'a str,
    ) -> Result<TargetDataLayout, TargetDataLayoutErrors<'a>> {
        // Parse an address space index from a string.
        let parse_address_space = |s: &'a str, cause: &'a str| {
            s.parse::<u32>().map(AddressSpace).map_err(|err| {
                TargetDataLayoutErrors::InvalidAddressSpace { addr_space: s, cause, err }
            })
        };

        // Parse a bit count from a string.
        let parse_bits = |s: &'a str, kind: &'a str, cause: &'a str| {
            s.parse::<u64>().map_err(|err| TargetDataLayoutErrors::InvalidBits {
                kind,
                bit: s,
                cause,
                err,
            })
        };

        // Parse a size string.
        let size = |s: &'a str, cause: &'a str| parse_bits(s, "size", cause).map(Size::from_bits);

        // Parse an alignment string.
        let align = |s: &[&'a str], cause: &'a str| {
            if s.is_empty() {
                return Err(TargetDataLayoutErrors::MissingAlignment { cause });
            }
            let align_from_bits = |bits| {
                Align::from_bits(bits)
                    .map_err(|err| TargetDataLayoutErrors::InvalidAlignment { cause, err })
            };
            let abi = parse_bits(s[0], "alignment", cause)?;
            let pref = s.get(1).map_or(Ok(abi), |pref| parse_bits(pref, "alignment", cause))?;
            Ok(AbiAndPrefAlign { abi: align_from_bits(abi)?, pref: align_from_bits(pref)? })
        };

        let mut dl = TargetDataLayout::default();
        let mut i128_align_src = 64;
        for spec in input.split('-') {
            let spec_parts = spec.split(':').collect::<Vec<_>>();

            match &*spec_parts {
                ["e"] => dl.endian = Endian::Little,
                ["E"] => dl.endian = Endian::Big,
                [p] if p.starts_with('P') => {
                    dl.instruction_address_space = parse_address_space(&p[1..], "P")?
                }
                ["a", ref a @ ..] => dl.aggregate_align = align(a, "a")?,
                ["f32", ref a @ ..] => dl.f32_align = align(a, "f32")?,
                ["f64", ref a @ ..] => dl.f64_align = align(a, "f64")?,
                // FIXME(erikdesjardins): we should be parsing nonzero address spaces
                // this will require replacing TargetDataLayout::{pointer_size,pointer_align}
                // with e.g. `fn pointer_size_in(AddressSpace)`
                [p @ "p", s, ref a @ ..] | [p @ "p0", s, ref a @ ..] => {
                    dl.pointer_size = size(s, p)?;
                    dl.pointer_align = align(a, p)?;
                }
                [s, ref a @ ..] if s.starts_with('i') => {
                    let Ok(bits) = s[1..].parse::<u64>() else {
                        size(&s[1..], "i")?; // For the user error.
                        continue;
                    };
                    let a = align(a, s)?;
                    match bits {
                        1 => dl.i1_align = a,
                        8 => dl.i8_align = a,
                        16 => dl.i16_align = a,
                        32 => dl.i32_align = a,
                        64 => dl.i64_align = a,
                        _ => {}
                    }
                    if bits >= i128_align_src && bits <= 128 {
                        // Default alignment for i128 is decided by taking the alignment of
                        // largest-sized i{64..=128}.
                        i128_align_src = bits;
                        dl.i128_align = a;
                    }
                }
                [s, ref a @ ..] if s.starts_with('v') => {
                    let v_size = size(&s[1..], "v")?;
                    let a = align(a, s)?;
                    if let Some(v) = dl.vector_align.iter_mut().find(|v| v.0 == v_size) {
                        v.1 = a;
                        continue;
                    }
                    // No existing entry, add a new one.
                    dl.vector_align.push((v_size, a));
                }
                _ => {} // Ignore everything else.
            }
        }
        Ok(dl)
    }

    /// Returns exclusive upper bound on object size.
    ///
    /// The theoretical maximum object size is defined as the maximum positive `isize` value.
    /// This ensures that the `offset` semantics remain well-defined by allowing it to correctly
    /// index every address within an object along with one byte past the end, along with allowing
    /// `isize` to store the difference between any two pointers into an object.
    ///
    /// The upper bound on 64-bit currently needs to be lower because LLVM uses a 64-bit integer
    /// to represent object size in bits. It would need to be 1 << 61 to account for this, but is
    /// currently conservatively bounded to 1 << 47 as that is enough to cover the current usable
    /// address space on 64-bit ARMv8 and x86_64.
    #[inline]
    pub fn obj_size_bound(&self) -> u64 {
        match self.pointer_size.bits() {
            16 => 1 << 15,
            32 => 1 << 31,
            64 => 1 << 47,
            bits => panic!("obj_size_bound: unknown pointer bit size {}", bits),
        }
    }

    #[inline]
    pub fn ptr_sized_integer(&self) -> Integer {
        match self.pointer_size.bits() {
            16 => I16,
            32 => I32,
            64 => I64,
            bits => panic!("ptr_sized_integer: unknown pointer bit size {}", bits),
        }
    }

    #[inline]
    pub fn vector_align(&self, vec_size: Size) -> AbiAndPrefAlign {
        for &(size, align) in &self.vector_align {
            if size == vec_size {
                return align;
            }
        }
        // Default to natural alignment, which is what LLVM does.
        // That is, use the size, rounded up to a power of 2.
        AbiAndPrefAlign::new(Align::from_bytes(vec_size.bytes().next_power_of_two()).unwrap())
    }
}

pub trait HasDataLayout {
    fn data_layout(&self) -> &TargetDataLayout;
}

impl HasDataLayout for TargetDataLayout {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        self
    }
}

/// Endianness of the target, which must match cfg(target-endian).
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Endian {
    Little,
    Big,
}

impl Endian {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Little => "little",
            Self::Big => "big",
        }
    }
}

impl fmt::Debug for Endian {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Endian {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "little" => Ok(Self::Little),
            "big" => Ok(Self::Big),
            _ => Err(format!(r#"unknown endian: "{}""#, s)),
        }
    }
}

/// Size of a type in bytes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_Generic))]
pub struct Size {
    raw: u64,
}

// Safety: Ord is implement as just comparing numerical values and numerical values
// are not changed by (de-)serialization.
#[cfg(feature = "nightly")]
unsafe impl StableOrd for Size {}

// This is debug-printed a lot in larger structs, don't waste too much space there
impl fmt::Debug for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Size({} bytes)", self.bytes())
    }
}

impl Size {
    pub const ZERO: Size = Size { raw: 0 };

    /// Rounds `bits` up to the next-higher byte boundary, if `bits` is
    /// not a multiple of 8.
    pub fn from_bits(bits: impl TryInto<u64>) -> Size {
        let bits = bits.try_into().ok().unwrap();
        // Avoid potential overflow from `bits + 7`.
        Size { raw: bits / 8 + ((bits % 8) + 7) / 8 }
    }

    #[inline]
    pub fn from_bytes(bytes: impl TryInto<u64>) -> Size {
        let bytes: u64 = bytes.try_into().ok().unwrap();
        Size { raw: bytes }
    }

    #[inline]
    pub fn bytes(self) -> u64 {
        self.raw
    }

    #[inline]
    pub fn bytes_usize(self) -> usize {
        self.bytes().try_into().unwrap()
    }

    #[inline]
    pub fn bits(self) -> u64 {
        #[cold]
        fn overflow(bytes: u64) -> ! {
            panic!("Size::bits: {} bytes in bits doesn't fit in u64", bytes)
        }

        self.bytes().checked_mul(8).unwrap_or_else(|| overflow(self.bytes()))
    }

    #[inline]
    pub fn bits_usize(self) -> usize {
        self.bits().try_into().unwrap()
    }

    #[inline]
    pub fn align_to(self, align: Align) -> Size {
        let mask = align.bytes() - 1;
        Size::from_bytes((self.bytes() + mask) & !mask)
    }

    #[inline]
    pub fn is_aligned(self, align: Align) -> bool {
        let mask = align.bytes() - 1;
        self.bytes() & mask == 0
    }

    #[inline]
    pub fn checked_add<C: HasDataLayout>(self, offset: Size, cx: &C) -> Option<Size> {
        let dl = cx.data_layout();

        let bytes = self.bytes().checked_add(offset.bytes())?;

        if bytes < dl.obj_size_bound() { Some(Size::from_bytes(bytes)) } else { None }
    }

    #[inline]
    pub fn checked_mul<C: HasDataLayout>(self, count: u64, cx: &C) -> Option<Size> {
        let dl = cx.data_layout();

        let bytes = self.bytes().checked_mul(count)?;
        if bytes < dl.obj_size_bound() { Some(Size::from_bytes(bytes)) } else { None }
    }

    /// Truncates `value` to `self` bits and then sign-extends it to 128 bits
    /// (i.e., if it is negative, fill with 1's on the left).
    #[inline]
    pub fn sign_extend(self, value: u128) -> u128 {
        let size = self.bits();
        if size == 0 {
            // Truncated until nothing is left.
            return 0;
        }
        // Sign-extend it.
        let shift = 128 - size;
        // Shift the unsigned value to the left, then shift back to the right as signed
        // (essentially fills with sign bit on the left).
        (((value << shift) as i128) >> shift) as u128
    }

    /// Truncates `value` to `self` bits.
    #[inline]
    pub fn truncate(self, value: u128) -> u128 {
        let size = self.bits();
        if size == 0 {
            // Truncated until nothing is left.
            return 0;
        }
        let shift = 128 - size;
        // Truncate (shift left to drop out leftover values, shift right to fill with zeroes).
        (value << shift) >> shift
    }

    #[inline]
    pub fn signed_int_min(&self) -> i128 {
        self.sign_extend(1_u128 << (self.bits() - 1)) as i128
    }

    #[inline]
    pub fn signed_int_max(&self) -> i128 {
        i128::MAX >> (128 - self.bits())
    }

    #[inline]
    pub fn unsigned_int_max(&self) -> u128 {
        u128::MAX >> (128 - self.bits())
    }
}

// Panicking addition, subtraction and multiplication for convenience.
// Avoid during layout computation, return `LayoutError` instead.

impl Add for Size {
    type Output = Size;
    #[inline]
    fn add(self, other: Size) -> Size {
        Size::from_bytes(self.bytes().checked_add(other.bytes()).unwrap_or_else(|| {
            panic!("Size::add: {} + {} doesn't fit in u64", self.bytes(), other.bytes())
        }))
    }
}

impl Sub for Size {
    type Output = Size;
    #[inline]
    fn sub(self, other: Size) -> Size {
        Size::from_bytes(self.bytes().checked_sub(other.bytes()).unwrap_or_else(|| {
            panic!("Size::sub: {} - {} would result in negative size", self.bytes(), other.bytes())
        }))
    }
}

impl Mul<Size> for u64 {
    type Output = Size;
    #[inline]
    fn mul(self, size: Size) -> Size {
        size * self
    }
}

impl Mul<u64> for Size {
    type Output = Size;
    #[inline]
    fn mul(self, count: u64) -> Size {
        match self.bytes().checked_mul(count) {
            Some(bytes) => Size::from_bytes(bytes),
            None => panic!("Size::mul: {} * {} doesn't fit in u64", self.bytes(), count),
        }
    }
}

impl AddAssign for Size {
    #[inline]
    fn add_assign(&mut self, other: Size) {
        *self = *self + other;
    }
}

#[cfg(feature = "nightly")]
impl Step for Size {
    #[inline]
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        u64::steps_between(&start.bytes(), &end.bytes())
    }

    #[inline]
    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        u64::forward_checked(start.bytes(), count).map(Self::from_bytes)
    }

    #[inline]
    fn forward(start: Self, count: usize) -> Self {
        Self::from_bytes(u64::forward(start.bytes(), count))
    }

    #[inline]
    unsafe fn forward_unchecked(start: Self, count: usize) -> Self {
        Self::from_bytes(u64::forward_unchecked(start.bytes(), count))
    }

    #[inline]
    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        u64::backward_checked(start.bytes(), count).map(Self::from_bytes)
    }

    #[inline]
    fn backward(start: Self, count: usize) -> Self {
        Self::from_bytes(u64::backward(start.bytes(), count))
    }

    #[inline]
    unsafe fn backward_unchecked(start: Self, count: usize) -> Self {
        Self::from_bytes(u64::backward_unchecked(start.bytes(), count))
    }
}

/// Alignment of a type in bytes (always a power of two).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_Generic))]
pub struct Align {
    pow2: u8,
}

// This is debug-printed a lot in larger structs, don't waste too much space there
impl fmt::Debug for Align {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Align({} bytes)", self.bytes())
    }
}

impl Align {
    pub const ONE: Align = Align { pow2: 0 };
    pub const MAX: Align = Align { pow2: 29 };

    #[inline]
    pub fn from_bits(bits: u64) -> Result<Align, String> {
        Align::from_bytes(Size::from_bits(bits).bytes())
    }

    #[inline]
    pub fn from_bytes(align: u64) -> Result<Align, String> {
        // Treat an alignment of 0 bytes like 1-byte alignment.
        if align == 0 {
            return Ok(Align::ONE);
        }

        #[cold]
        fn not_power_of_2(align: u64) -> String {
            format!("`{}` is not a power of 2", align)
        }

        #[cold]
        fn too_large(align: u64) -> String {
            format!("`{}` is too large", align)
        }

        let mut bytes = align;
        let mut pow2: u8 = 0;
        while (bytes & 1) == 0 {
            pow2 += 1;
            bytes >>= 1;
        }
        if bytes != 1 {
            return Err(not_power_of_2(align));
        }
        if pow2 > Self::MAX.pow2 {
            return Err(too_large(align));
        }

        Ok(Align { pow2 })
    }

    #[inline]
    pub fn bytes(self) -> u64 {
        1 << self.pow2
    }

    #[inline]
    pub fn bits(self) -> u64 {
        self.bytes() * 8
    }

    /// Computes the best alignment possible for the given offset
    /// (the largest power of two that the offset is a multiple of).
    ///
    /// N.B., for an offset of `0`, this happens to return `2^64`.
    #[inline]
    pub fn max_for_offset(offset: Size) -> Align {
        Align { pow2: offset.bytes().trailing_zeros() as u8 }
    }

    /// Lower the alignment, if necessary, such that the given offset
    /// is aligned to it (the offset is a multiple of the alignment).
    #[inline]
    pub fn restrict_for_offset(self, offset: Size) -> Align {
        self.min(Align::max_for_offset(offset))
    }
}

/// A pair of alignments, ABI-mandated and preferred.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]

pub struct AbiAndPrefAlign {
    pub abi: Align,
    pub pref: Align,
}

impl AbiAndPrefAlign {
    #[inline]
    pub fn new(align: Align) -> AbiAndPrefAlign {
        AbiAndPrefAlign { abi: align, pref: align }
    }

    #[inline]
    pub fn min(self, other: AbiAndPrefAlign) -> AbiAndPrefAlign {
        AbiAndPrefAlign { abi: self.abi.min(other.abi), pref: self.pref.min(other.pref) }
    }

    #[inline]
    pub fn max(self, other: AbiAndPrefAlign) -> AbiAndPrefAlign {
        AbiAndPrefAlign { abi: self.abi.max(other.abi), pref: self.pref.max(other.pref) }
    }
}

/// Integers, also used for enum discriminants.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_Generic))]

pub enum Integer {
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl Integer {
    #[inline]
    pub fn size(self) -> Size {
        match self {
            I8 => Size::from_bytes(1),
            I16 => Size::from_bytes(2),
            I32 => Size::from_bytes(4),
            I64 => Size::from_bytes(8),
            I128 => Size::from_bytes(16),
        }
    }

    /// Gets the Integer type from an IntegerType.
    pub fn from_attr<C: HasDataLayout>(cx: &C, ity: IntegerType) -> Integer {
        let dl = cx.data_layout();

        match ity {
            IntegerType::Pointer(_) => dl.ptr_sized_integer(),
            IntegerType::Fixed(x, _) => x,
        }
    }

    pub fn align<C: HasDataLayout>(self, cx: &C) -> AbiAndPrefAlign {
        let dl = cx.data_layout();

        match self {
            I8 => dl.i8_align,
            I16 => dl.i16_align,
            I32 => dl.i32_align,
            I64 => dl.i64_align,
            I128 => dl.i128_align,
        }
    }

    /// Returns the largest signed value that can be represented by this Integer.
    #[inline]
    pub fn signed_max(self) -> i128 {
        match self {
            I8 => i8::MAX as i128,
            I16 => i16::MAX as i128,
            I32 => i32::MAX as i128,
            I64 => i64::MAX as i128,
            I128 => i128::MAX,
        }
    }

    /// Finds the smallest Integer type which can represent the signed value.
    #[inline]
    pub fn fit_signed(x: i128) -> Integer {
        match x {
            -0x0000_0000_0000_0080..=0x0000_0000_0000_007f => I8,
            -0x0000_0000_0000_8000..=0x0000_0000_0000_7fff => I16,
            -0x0000_0000_8000_0000..=0x0000_0000_7fff_ffff => I32,
            -0x8000_0000_0000_0000..=0x7fff_ffff_ffff_ffff => I64,
            _ => I128,
        }
    }

    /// Finds the smallest Integer type which can represent the unsigned value.
    #[inline]
    pub fn fit_unsigned(x: u128) -> Integer {
        match x {
            0..=0x0000_0000_0000_00ff => I8,
            0..=0x0000_0000_0000_ffff => I16,
            0..=0x0000_0000_ffff_ffff => I32,
            0..=0xffff_ffff_ffff_ffff => I64,
            _ => I128,
        }
    }

    /// Finds the smallest integer with the given alignment.
    pub fn for_align<C: HasDataLayout>(cx: &C, wanted: Align) -> Option<Integer> {
        let dl = cx.data_layout();

        [I8, I16, I32, I64, I128].into_iter().find(|&candidate| {
            wanted == candidate.align(dl).abi && wanted.bytes() == candidate.size().bytes()
        })
    }

    /// Find the largest integer with the given alignment or less.
    pub fn approximate_align<C: HasDataLayout>(cx: &C, wanted: Align) -> Integer {
        let dl = cx.data_layout();

        // FIXME(eddyb) maybe include I128 in the future, when it works everywhere.
        for candidate in [I64, I32, I16] {
            if wanted >= candidate.align(dl).abi && wanted.bytes() >= candidate.size().bytes() {
                return candidate;
            }
        }
        I8
    }

    // FIXME(eddyb) consolidate this and other methods that find the appropriate
    // `Integer` given some requirements.
    #[inline]
    pub fn from_size(size: Size) -> Result<Self, String> {
        match size.bits() {
            8 => Ok(Integer::I8),
            16 => Ok(Integer::I16),
            32 => Ok(Integer::I32),
            64 => Ok(Integer::I64),
            128 => Ok(Integer::I128),
            _ => Err(format!("rust does not support integers with {} bits", size.bits())),
        }
    }
}

/// Fundamental unit of memory access and layout.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum Primitive {
    /// The `bool` is the signedness of the `Integer` type.
    ///
    /// One would think we would not care about such details this low down,
    /// but some ABIs are described in terms of C types and ISAs where the
    /// integer arithmetic is done on {sign,zero}-extended registers, e.g.
    /// a negative integer passed by zero-extension will appear positive in
    /// the callee, and most operations on it will produce the wrong values.
    Int(Integer, bool),
    F32,
    F64,
    Pointer(AddressSpace),
}

impl Primitive {
    pub fn size<C: HasDataLayout>(self, cx: &C) -> Size {
        let dl = cx.data_layout();

        match self {
            Int(i, _) => i.size(),
            F32 => Size::from_bits(32),
            F64 => Size::from_bits(64),
            // FIXME(erikdesjardins): ignoring address space is technically wrong, pointers in
            // different address spaces can have different sizes
            // (but TargetDataLayout doesn't currently parse that part of the DL string)
            Pointer(_) => dl.pointer_size,
        }
    }

    pub fn align<C: HasDataLayout>(self, cx: &C) -> AbiAndPrefAlign {
        let dl = cx.data_layout();

        match self {
            Int(i, _) => i.align(dl),
            F32 => dl.f32_align,
            F64 => dl.f64_align,
            // FIXME(erikdesjardins): ignoring address space is technically wrong, pointers in
            // different address spaces can have different alignments
            // (but TargetDataLayout doesn't currently parse that part of the DL string)
            Pointer(_) => dl.pointer_align,
        }
    }
}

/// Inclusive wrap-around range of valid values, that is, if
/// start > end, it represents `start..=MAX`,
/// followed by `0..=end`.
///
/// That is, for an i8 primitive, a range of `254..=2` means following
/// sequence:
///
///    254 (-2), 255 (-1), 0, 1, 2
///
/// This is intended specifically to mirror LLVM’s `!range` metadata semantics.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub struct WrappingRange {
    pub start: u128,
    pub end: u128,
}

impl WrappingRange {
    pub fn full(size: Size) -> Self {
        Self { start: 0, end: size.unsigned_int_max() }
    }

    /// Returns `true` if `v` is contained in the range.
    #[inline(always)]
    pub fn contains(&self, v: u128) -> bool {
        if self.start <= self.end {
            self.start <= v && v <= self.end
        } else {
            self.start <= v || v <= self.end
        }
    }

    /// Returns `self` with replaced `start`
    #[inline(always)]
    pub fn with_start(mut self, start: u128) -> Self {
        self.start = start;
        self
    }

    /// Returns `self` with replaced `end`
    #[inline(always)]
    pub fn with_end(mut self, end: u128) -> Self {
        self.end = end;
        self
    }

    /// Returns `true` if `size` completely fills the range.
    #[inline]
    pub fn is_full_for(&self, size: Size) -> bool {
        let max_value = size.unsigned_int_max();
        debug_assert!(self.start <= max_value && self.end <= max_value);
        self.start == (self.end.wrapping_add(1) & max_value)
    }
}

impl fmt::Debug for WrappingRange {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start > self.end {
            write!(fmt, "(..={}) | ({}..)", self.end, self.start)?;
        } else {
            write!(fmt, "{}..={}", self.start, self.end)?;
        }
        Ok(())
    }
}

/// Information about one scalar component of a Rust type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum Scalar {
    Initialized {
        value: Primitive,

        // FIXME(eddyb) always use the shortest range, e.g., by finding
        // the largest space between two consecutive valid values and
        // taking everything else as the (shortest) valid range.
        valid_range: WrappingRange,
    },
    Union {
        /// Even for unions, we need to use the correct registers for the kind of
        /// values inside the union, so we keep the `Primitive` type around. We
        /// also use it to compute the size of the scalar.
        /// However, unions never have niches and even allow undef,
        /// so there is no `valid_range`.
        value: Primitive,
    },
}

impl Scalar {
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(
            self,
            Scalar::Initialized {
                value: Int(I8, false),
                valid_range: WrappingRange { start: 0, end: 1 }
            }
        )
    }

    /// Get the primitive representation of this type, ignoring the valid range and whether the
    /// value is allowed to be undefined (due to being a union).
    pub fn primitive(&self) -> Primitive {
        match *self {
            Scalar::Initialized { value, .. } | Scalar::Union { value } => value,
        }
    }

    pub fn align(self, cx: &impl HasDataLayout) -> AbiAndPrefAlign {
        self.primitive().align(cx)
    }

    pub fn size(self, cx: &impl HasDataLayout) -> Size {
        self.primitive().size(cx)
    }

    #[inline]
    pub fn to_union(&self) -> Self {
        Self::Union { value: self.primitive() }
    }

    #[inline]
    pub fn valid_range(&self, cx: &impl HasDataLayout) -> WrappingRange {
        match *self {
            Scalar::Initialized { valid_range, .. } => valid_range,
            Scalar::Union { value } => WrappingRange::full(value.size(cx)),
        }
    }

    #[inline]
    /// Allows the caller to mutate the valid range. This operation will panic if attempted on a union.
    pub fn valid_range_mut(&mut self) -> &mut WrappingRange {
        match self {
            Scalar::Initialized { valid_range, .. } => valid_range,
            Scalar::Union { .. } => panic!("cannot change the valid range of a union"),
        }
    }

    /// Returns `true` if all possible numbers are valid, i.e `valid_range` covers the whole layout
    #[inline]
    pub fn is_always_valid<C: HasDataLayout>(&self, cx: &C) -> bool {
        match *self {
            Scalar::Initialized { valid_range, .. } => valid_range.is_full_for(self.size(cx)),
            Scalar::Union { .. } => true,
        }
    }

    /// Returns `true` if this type can be left uninit.
    #[inline]
    pub fn is_uninit_valid(&self) -> bool {
        match *self {
            Scalar::Initialized { .. } => false,
            Scalar::Union { .. } => true,
        }
    }
}

/// Describes how the fields of a type are located in memory.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum FieldsShape {
    /// Scalar primitives and `!`, which never have fields.
    Primitive,

    /// All fields start at no offset. The `usize` is the field count.
    Union(NonZeroUsize),

    /// Array/vector-like placement, with all fields of identical types.
    Array { stride: Size, count: u64 },

    /// Struct-like placement, with precomputed offsets.
    ///
    /// Fields are guaranteed to not overlap, but note that gaps
    /// before, between and after all the fields are NOT always
    /// padding, and as such their contents may not be discarded.
    /// For example, enum variants leave a gap at the start,
    /// where the discriminant field in the enum layout goes.
    Arbitrary {
        /// Offsets for the first byte of each field,
        /// ordered to match the source definition order.
        /// This vector does not go in increasing order.
        // FIXME(eddyb) use small vector optimization for the common case.
        offsets: Vec<Size>,

        /// Maps source order field indices to memory order indices,
        /// depending on how the fields were reordered (if at all).
        /// This is a permutation, with both the source order and the
        /// memory order using the same (0..n) index ranges.
        ///
        /// Note that during computation of `memory_index`, sometimes
        /// it is easier to operate on the inverse mapping (that is,
        /// from memory order to source order), and that is usually
        /// named `inverse_memory_index`.
        ///
        // FIXME(eddyb) build a better abstraction for permutations, if possible.
        // FIXME(camlorn) also consider small vector optimization here.
        memory_index: Vec<u32>,
    },
}

impl FieldsShape {
    #[inline]
    pub fn count(&self) -> usize {
        match *self {
            FieldsShape::Primitive => 0,
            FieldsShape::Union(count) => count.get(),
            FieldsShape::Array { count, .. } => count.try_into().unwrap(),
            FieldsShape::Arbitrary { ref offsets, .. } => offsets.len(),
        }
    }

    #[inline]
    pub fn offset(&self, i: usize) -> Size {
        match *self {
            FieldsShape::Primitive => {
                unreachable!("FieldsShape::offset: `Primitive`s have no fields")
            }
            FieldsShape::Union(count) => {
                assert!(
                    i < count.get(),
                    "tried to access field {} of union with {} fields",
                    i,
                    count
                );
                Size::ZERO
            }
            FieldsShape::Array { stride, count } => {
                let i = u64::try_from(i).unwrap();
                assert!(i < count);
                stride * i
            }
            FieldsShape::Arbitrary { ref offsets, .. } => offsets[i],
        }
    }

    #[inline]
    pub fn memory_index(&self, i: usize) -> usize {
        match *self {
            FieldsShape::Primitive => {
                unreachable!("FieldsShape::memory_index: `Primitive`s have no fields")
            }
            FieldsShape::Union(_) | FieldsShape::Array { .. } => i,
            FieldsShape::Arbitrary { ref memory_index, .. } => memory_index[i].try_into().unwrap(),
        }
    }

    /// Gets source indices of the fields by increasing offsets.
    #[inline]
    pub fn index_by_increasing_offset<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        let mut inverse_small = [0u8; 64];
        let mut inverse_big = vec![];
        let use_small = self.count() <= inverse_small.len();

        // We have to write this logic twice in order to keep the array small.
        if let FieldsShape::Arbitrary { ref memory_index, .. } = *self {
            if use_small {
                for i in 0..self.count() {
                    inverse_small[memory_index[i] as usize] = i as u8;
                }
            } else {
                inverse_big = vec![0; self.count()];
                for i in 0..self.count() {
                    inverse_big[memory_index[i] as usize] = i as u32;
                }
            }
        }

        (0..self.count()).map(move |i| match *self {
            FieldsShape::Primitive | FieldsShape::Union(_) | FieldsShape::Array { .. } => i,
            FieldsShape::Arbitrary { .. } => {
                if use_small {
                    inverse_small[i] as usize
                } else {
                    inverse_big[i] as usize
                }
            }
        })
    }
}

/// An identifier that specifies the address space that some operation
/// should operate on. Special address spaces have an effect on code generation,
/// depending on the target and the address spaces it implements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub struct AddressSpace(pub u32);

impl AddressSpace {
    /// The default address space, corresponding to data space.
    pub const DATA: Self = AddressSpace(0);
}

/// Describes how values of the type are passed by target ABIs,
/// in terms of categories of C types there are ABI rules for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]

pub enum Abi {
    Uninhabited,
    Scalar(Scalar),
    ScalarPair(Scalar, Scalar),
    Vector {
        element: Scalar,
        count: u64,
    },
    Aggregate {
        /// If true, the size is exact, otherwise it's only a lower bound.
        sized: bool,
    },
}

impl Abi {
    /// Returns `true` if the layout corresponds to an unsized type.
    #[inline]
    pub fn is_unsized(&self) -> bool {
        match *self {
            Abi::Uninhabited | Abi::Scalar(_) | Abi::ScalarPair(..) | Abi::Vector { .. } => false,
            Abi::Aggregate { sized } => !sized,
        }
    }

    #[inline]
    pub fn is_sized(&self) -> bool {
        !self.is_unsized()
    }

    /// Returns `true` if this is a single signed integer scalar
    #[inline]
    pub fn is_signed(&self) -> bool {
        match self {
            Abi::Scalar(scal) => match scal.primitive() {
                Primitive::Int(_, signed) => signed,
                _ => false,
            },
            _ => panic!("`is_signed` on non-scalar ABI {:?}", self),
        }
    }

    /// Returns `true` if this is an uninhabited type
    #[inline]
    pub fn is_uninhabited(&self) -> bool {
        matches!(*self, Abi::Uninhabited)
    }

    /// Returns `true` is this is a scalar type
    #[inline]
    pub fn is_scalar(&self) -> bool {
        matches!(*self, Abi::Scalar(_))
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum Variants {
    /// Single enum variants, structs/tuples, unions, and all non-ADTs.
    Single { index: VariantIdx },

    /// Enum-likes with more than one inhabited variant: each variant comes with
    /// a *discriminant* (usually the same as the variant index but the user can
    /// assign explicit discriminant values). That discriminant is encoded
    /// as a *tag* on the machine. The layout of each variant is
    /// a struct, and they all have space reserved for the tag.
    /// For enums, the tag is the sole field of the layout.
    Multiple {
        tag: Scalar,
        tag_encoding: TagEncoding,
        tag_field: usize,
        variants: IndexVec<VariantIdx, LayoutS>,
    },
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum TagEncoding {
    /// The tag directly stores the discriminant, but possibly with a smaller layout
    /// (so converting the tag to the discriminant can require sign extension).
    Direct,

    /// Niche (values invalid for a type) encoding the discriminant:
    /// Discriminant and variant index coincide.
    /// The variant `untagged_variant` contains a niche at an arbitrary
    /// offset (field `tag_field` of the enum), which for a variant with
    /// discriminant `d` is set to
    /// `(d - niche_variants.start).wrapping_add(niche_start)`.
    ///
    /// For example, `Option<(usize, &T)>`  is represented such that
    /// `None` has a null pointer for the second tuple field, and
    /// `Some` is the identity function (with a non-null reference).
    Niche {
        untagged_variant: VariantIdx,
        niche_variants: RangeInclusive<VariantIdx>,
        niche_start: u128,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub struct Niche {
    pub offset: Size,
    pub value: Primitive,
    pub valid_range: WrappingRange,
}

impl Niche {
    pub fn from_scalar<C: HasDataLayout>(cx: &C, offset: Size, scalar: Scalar) -> Option<Self> {
        let Scalar::Initialized { value, valid_range } = scalar else { return None };
        let niche = Niche { offset, value, valid_range };
        if niche.available(cx) > 0 { Some(niche) } else { None }
    }

    pub fn available<C: HasDataLayout>(&self, cx: &C) -> u128 {
        let Self { value, valid_range: v, .. } = *self;
        let size = value.size(cx);
        assert!(size.bits() <= 128);
        let max_value = size.unsigned_int_max();

        // Find out how many values are outside the valid range.
        let niche = v.end.wrapping_add(1)..v.start;
        niche.end.wrapping_sub(niche.start) & max_value
    }

    pub fn reserve<C: HasDataLayout>(&self, cx: &C, count: u128) -> Option<(u128, Scalar)> {
        assert!(count > 0);

        let Self { value, valid_range: v, .. } = *self;
        let size = value.size(cx);
        assert!(size.bits() <= 128);
        let max_value = size.unsigned_int_max();

        let niche = v.end.wrapping_add(1)..v.start;
        let available = niche.end.wrapping_sub(niche.start) & max_value;
        if count > available {
            return None;
        }

        // Extend the range of valid values being reserved by moving either `v.start` or `v.end` bound.
        // Given an eventual `Option<T>`, we try to maximize the chance for `None` to occupy the niche of zero.
        // This is accomplished by preferring enums with 2 variants(`count==1`) and always taking the shortest path to niche zero.
        // Having `None` in niche zero can enable some special optimizations.
        //
        // Bound selection criteria:
        // 1. Select closest to zero given wrapping semantics.
        // 2. Avoid moving past zero if possible.
        //
        // In practice this means that enums with `count > 1` are unlikely to claim niche zero, since they have to fit perfectly.
        // If niche zero is already reserved, the selection of bounds are of little interest.
        let move_start = |v: WrappingRange| {
            let start = v.start.wrapping_sub(count) & max_value;
            Some((start, Scalar::Initialized { value, valid_range: v.with_start(start) }))
        };
        let move_end = |v: WrappingRange| {
            let start = v.end.wrapping_add(1) & max_value;
            let end = v.end.wrapping_add(count) & max_value;
            Some((start, Scalar::Initialized { value, valid_range: v.with_end(end) }))
        };
        let distance_end_zero = max_value - v.end;
        if v.start > v.end {
            // zero is unavailable because wrapping occurs
            move_end(v)
        } else if v.start <= distance_end_zero {
            if count <= v.start {
                move_start(v)
            } else {
                // moved past zero, use other bound
                move_end(v)
            }
        } else {
            let end = v.end.wrapping_add(count) & max_value;
            let overshot_zero = (1..=v.end).contains(&end);
            if overshot_zero {
                // moved past zero, use other bound
                move_start(v)
            } else {
                move_end(v)
            }
        }
    }
}

rustc_index::newtype_index! {
    #[derive(HashStable_Generic)]
    pub struct VariantIdx {}
}

#[derive(PartialEq, Eq, Hash, Clone)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub struct LayoutS {
    /// Says where the fields are located within the layout.
    pub fields: FieldsShape,

    /// Encodes information about multi-variant layouts.
    /// Even with `Multiple` variants, a layout still has its own fields! Those are then
    /// shared between all variants. One of them will be the discriminant,
    /// but e.g. generators can have more.
    ///
    /// To access all fields of this layout, both `fields` and the fields of the active variant
    /// must be taken into account.
    pub variants: Variants,

    /// The `abi` defines how this data is passed between functions, and it defines
    /// value restrictions via `valid_range`.
    ///
    /// Note that this is entirely orthogonal to the recursive structure defined by
    /// `variants` and `fields`; for example, `ManuallyDrop<Result<isize, isize>>` has
    /// `Abi::ScalarPair`! So, even with non-`Aggregate` `abi`, `fields` and `variants`
    /// have to be taken into account to find all fields of this layout.
    pub abi: Abi,

    /// The leaf scalar with the largest number of invalid values
    /// (i.e. outside of its `valid_range`), if it exists.
    pub largest_niche: Option<Niche>,

    pub align: AbiAndPrefAlign,
    pub size: Size,
}

impl LayoutS {
    pub fn scalar<C: HasDataLayout>(cx: &C, scalar: Scalar) -> Self {
        let largest_niche = Niche::from_scalar(cx, Size::ZERO, scalar);
        let size = scalar.size(cx);
        let align = scalar.align(cx);
        LayoutS {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Primitive,
            abi: Abi::Scalar(scalar),
            largest_niche,
            size,
            align,
        }
    }
}

impl fmt::Debug for LayoutS {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This is how `Layout` used to print before it become
        // `Interned<LayoutS>`. We print it like this to avoid having to update
        // expected output in a lot of tests.
        let LayoutS { size, align, abi, fields, largest_niche, variants } = self;
        f.debug_struct("Layout")
            .field("size", size)
            .field("align", align)
            .field("abi", abi)
            .field("fields", fields)
            .field("largest_niche", largest_niche)
            .field("variants", variants)
            .finish()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable_Generic)]
#[rustc_pass_by_value]
pub struct Layout<'a>(pub Interned<'a, LayoutS>);

impl<'a> fmt::Debug for Layout<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // See comment on `<LayoutS as Debug>::fmt` above.
        self.0.0.fmt(f)
    }
}

impl<'a> Layout<'a> {
    pub fn fields(self) -> &'a FieldsShape {
        &self.0.0.fields
    }

    pub fn variants(self) -> &'a Variants {
        &self.0.0.variants
    }

    pub fn abi(self) -> Abi {
        self.0.0.abi
    }

    pub fn largest_niche(self) -> Option<Niche> {
        self.0.0.largest_niche
    }

    pub fn align(self) -> AbiAndPrefAlign {
        self.0.0.align
    }

    pub fn size(self) -> Size {
        self.0.0.size
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PointerKind {
    /// Shared reference. `frozen` indicates the absence of any `UnsafeCell`.
    SharedRef { frozen: bool },
    /// Mutable reference. `unpin` indicates the absence of any pinned data.
    MutableRef { unpin: bool },
    /// Box. `unpin` indicates the absence of any pinned data.
    Box { unpin: bool },
}

/// Note that this information is advisory only, and backends are free to ignore it.
/// It can only be used to encode potential optimizations, but no critical information.
#[derive(Copy, Clone, Debug)]
pub struct PointeeInfo {
    pub size: Size,
    pub align: Align,
    pub safe: Option<PointerKind>,
}

impl LayoutS {
    /// Returns `true` if the layout corresponds to an unsized type.
    pub fn is_unsized(&self) -> bool {
        self.abi.is_unsized()
    }

    pub fn is_sized(&self) -> bool {
        self.abi.is_sized()
    }

    /// Returns `true` if the type is a ZST and not unsized.
    pub fn is_zst(&self) -> bool {
        match self.abi {
            Abi::Scalar(_) | Abi::ScalarPair(..) | Abi::Vector { .. } => false,
            Abi::Uninhabited => self.size.bytes() == 0,
            Abi::Aggregate { sized } => sized && self.size.bytes() == 0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum StructKind {
    /// A tuple, closure, or univariant which cannot be coerced to unsized.
    AlwaysSized,
    /// A univariant, the last field of which may be coerced to unsized.
    MaybeUnsized,
    /// A univariant, but with a prefix of an arbitrary size & alignment (e.g., enum tag).
    Prefixed(Size, Align),
}
