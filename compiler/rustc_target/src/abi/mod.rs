pub use Integer::*;
pub use Primitive::*;

use crate::spec::Target;

use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::iter::Step;
use std::num::NonZeroUsize;
use std::ops::{Add, AddAssign, Deref, Mul, RangeInclusive, Sub};
use std::str::FromStr;

use rustc_index::vec::{Idx, IndexVec};
use rustc_macros::HashStable_Generic;
use rustc_serialize::json::{Json, ToJson};

pub mod call;

/// Parsed [Data layout](https://llvm.org/docs/LangRef.html#data-layout)
/// for a target, which contains everything needed to compute layouts.
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

    /// Minimum size of #[repr(C)] enums (default I32 bits)
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

impl TargetDataLayout {
    pub fn parse(target: &Target) -> Result<TargetDataLayout, String> {
        // Parse an address space index from a string.
        let parse_address_space = |s: &str, cause: &str| {
            s.parse::<u32>().map(AddressSpace).map_err(|err| {
                format!("invalid address space `{}` for `{}` in \"data-layout\": {}", s, cause, err)
            })
        };

        // Parse a bit count from a string.
        let parse_bits = |s: &str, kind: &str, cause: &str| {
            s.parse::<u64>().map_err(|err| {
                format!("invalid {} `{}` for `{}` in \"data-layout\": {}", kind, s, cause, err)
            })
        };

        // Parse a size string.
        let size = |s: &str, cause: &str| parse_bits(s, "size", cause).map(Size::from_bits);

        // Parse an alignment string.
        let align = |s: &[&str], cause: &str| {
            if s.is_empty() {
                return Err(format!("missing alignment for `{}` in \"data-layout\"", cause));
            }
            let align_from_bits = |bits| {
                Align::from_bits(bits).map_err(|err| {
                    format!("invalid alignment for `{}` in \"data-layout\": {}", cause, err)
                })
            };
            let abi = parse_bits(s[0], "alignment", cause)?;
            let pref = s.get(1).map_or(Ok(abi), |pref| parse_bits(pref, "alignment", cause))?;
            Ok(AbiAndPrefAlign { abi: align_from_bits(abi)?, pref: align_from_bits(pref)? })
        };

        let mut dl = TargetDataLayout::default();
        let mut i128_align_src = 64;
        for spec in target.data_layout.split('-') {
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
                [p @ "p", s, ref a @ ..] | [p @ "p0", s, ref a @ ..] => {
                    dl.pointer_size = size(s, p)?;
                    dl.pointer_align = align(a, p)?;
                }
                [s, ref a @ ..] if s.starts_with('i') => {
                    let bits = match s[1..].parse::<u64>() {
                        Ok(bits) => bits,
                        Err(_) => {
                            size(&s[1..], "i")?; // For the user error.
                            continue;
                        }
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

        // Perform consistency checks against the Target information.
        if dl.endian != target.endian {
            return Err(format!(
                "inconsistent target specification: \"data-layout\" claims \
                 architecture is {}-endian, while \"target-endian\" is `{}`",
                dl.endian.as_str(),
                target.endian.as_str(),
            ));
        }

        if dl.pointer_size.bits() != target.pointer_width.into() {
            return Err(format!(
                "inconsistent target specification: \"data-layout\" claims \
                 pointers are {}-bit, while \"target-pointer-width\" is `{}`",
                dl.pointer_size.bits(),
                target.pointer_width
            ));
        }

        dl.c_enum_min_size = Integer::from_size(Size::from_bits(target.c_enum_min_bits))?;

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
#[derive(Copy, Clone, PartialEq)]
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

impl ToJson for Endian {
    fn to_json(&self) -> Json {
        self.as_str().to_json()
    }
}

/// Size of a type in bytes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub struct Size {
    // The top 3 bits are ALWAYS zero.
    raw: u64,
}

impl Size {
    pub const ZERO: Size = Size { raw: 0 };

    /// Rounds `bits` up to the next-higher byte boundary, if `bits` is
    /// is not aligned.
    pub fn from_bits(bits: impl TryInto<u64>) -> Size {
        let bits = bits.try_into().ok().unwrap();

        #[cold]
        fn overflow(bits: u64) -> ! {
            panic!("Size::from_bits({}) has overflowed", bits);
        }

        // This is the largest value of `bits` that does not cause overflow
        // during rounding, and guarantees that the resulting number of bytes
        // cannot cause overflow when multiplied by 8.
        if bits > 0xffff_ffff_ffff_fff8 {
            overflow(bits);
        }

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
        self.raw << 3
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
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub struct Align {
    pow2: u8,
}

impl Align {
    pub const ONE: Align = Align { pow2: 0 };

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
        if pow2 > 29 {
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
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encodable, Decodable)]
#[derive(HashStable_Generic)]
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
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, HashStable_Generic)]
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

        for candidate in [I8, I16, I32, I64, I128] {
            if wanted == candidate.align(dl).abi && wanted.bytes() == candidate.size().bytes() {
                return Some(candidate);
            }
        }
        None
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
    fn from_size(size: Size) -> Result<Self, String> {
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
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
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
    Pointer,
}

impl Primitive {
    pub fn size<C: HasDataLayout>(self, cx: &C) -> Size {
        let dl = cx.data_layout();

        match self {
            Int(i, _) => i.size(),
            F32 => Size::from_bits(32),
            F64 => Size::from_bits(64),
            Pointer => dl.pointer_size,
        }
    }

    pub fn align<C: HasDataLayout>(self, cx: &C) -> AbiAndPrefAlign {
        let dl = cx.data_layout();

        match self {
            Int(i, _) => i.align(dl),
            F32 => dl.f32_align,
            F64 => dl.f64_align,
            Pointer => dl.pointer_align,
        }
    }

    // FIXME(eddyb) remove, it's trivial thanks to `matches!`.
    #[inline]
    pub fn is_float(self) -> bool {
        matches!(self, F32 | F64)
    }

    // FIXME(eddyb) remove, it's completely unused.
    #[inline]
    pub fn is_int(self) -> bool {
        matches!(self, Int(..))
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
#[derive(HashStable_Generic)]
pub struct WrappingRange {
    pub start: u128,
    pub end: u128,
}

impl WrappingRange {
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
#[derive(HashStable_Generic)]
pub struct Scalar {
    pub value: Primitive,

    // FIXME(eddyb) always use the shortest range, e.g., by finding
    // the largest space between two consecutive valid values and
    // taking everything else as the (shortest) valid range.
    pub valid_range: WrappingRange,
}

impl Scalar {
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(
            self,
            Scalar { value: Int(I8, false), valid_range: WrappingRange { start: 0, end: 1 } }
        )
    }

    /// Returns `true` if all possible numbers are valid, i.e `valid_range` covers the whole layout
    #[inline]
    pub fn is_always_valid<C: HasDataLayout>(&self, cx: &C) -> bool {
        self.valid_range.is_full_for(self.value.size(cx))
    }
}

/// Describes how the fields of a type are located in memory.
#[derive(PartialEq, Eq, Hash, Debug, HashStable_Generic)]
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
        // FIXME(camlorn) also consider small vector  optimization here.
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddressSpace(pub u32);

impl AddressSpace {
    /// The default address space, corresponding to data space.
    pub const DATA: Self = AddressSpace(0);
}

/// Describes how values of the type are passed by target ABIs,
/// in terms of categories of C types there are ABI rules for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
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

    /// Returns `true` if this is a single signed integer scalar
    #[inline]
    pub fn is_signed(&self) -> bool {
        match self {
            Abi::Scalar(scal) => match scal.value {
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

rustc_index::newtype_index! {
    pub struct VariantIdx {
        derive [HashStable_Generic]
    }
}

#[derive(PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub enum Variants {
    /// Single enum variants, structs/tuples, unions, and all non-ADTs.
    Single { index: VariantIdx },

    /// Enum-likes with more than one inhabited variant: each variant comes with
    /// a *discriminant* (usually the same as the variant index but the user can
    /// assign explicit discriminant values).  That discriminant is encoded
    /// as a *tag* on the machine.  The layout of each variant is
    /// a struct, and they all have space reserved for the tag.
    /// For enums, the tag is the sole field of the layout.
    Multiple {
        tag: Scalar,
        tag_encoding: TagEncoding,
        tag_field: usize,
        variants: IndexVec<VariantIdx, Layout>,
    },
}

#[derive(PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub enum TagEncoding {
    /// The tag directly stores the discriminant, but possibly with a smaller layout
    /// (so converting the tag to the discriminant can require sign extension).
    Direct,

    /// Niche (values invalid for a type) encoding the discriminant:
    /// Discriminant and variant index coincide.
    /// The variant `dataful_variant` contains a niche at an arbitrary
    /// offset (field `tag_field` of the enum), which for a variant with
    /// discriminant `d` is set to
    /// `(d - niche_variants.start).wrapping_add(niche_start)`.
    ///
    /// For example, `Option<(usize, &T)>`  is represented such that
    /// `None` has a null pointer for the second tuple field, and
    /// `Some` is the identity function (with a non-null reference).
    Niche {
        dataful_variant: VariantIdx,
        niche_variants: RangeInclusive<VariantIdx>,
        niche_start: u128,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct Niche {
    pub offset: Size,
    pub scalar: Scalar,
}

impl Niche {
    pub fn from_scalar<C: HasDataLayout>(cx: &C, offset: Size, scalar: Scalar) -> Option<Self> {
        let niche = Niche { offset, scalar };
        if niche.available(cx) > 0 { Some(niche) } else { None }
    }

    pub fn available<C: HasDataLayout>(&self, cx: &C) -> u128 {
        let Scalar { value, valid_range: v } = self.scalar;
        let size = value.size(cx);
        assert!(size.bits() <= 128);
        let max_value = size.unsigned_int_max();

        // Find out how many values are outside the valid range.
        let niche = v.end.wrapping_add(1)..v.start;
        niche.end.wrapping_sub(niche.start) & max_value
    }

    pub fn reserve<C: HasDataLayout>(&self, cx: &C, count: u128) -> Option<(u128, Scalar)> {
        assert!(count > 0);

        let Scalar { value, valid_range: v } = self.scalar;
        let size = value.size(cx);
        assert!(size.bits() <= 128);
        let max_value = size.unsigned_int_max();

        if count > max_value {
            return None;
        }

        // Compute the range of invalid values being reserved.
        let start = v.end.wrapping_add(1) & max_value;
        let end = v.end.wrapping_add(count) & max_value;

        if v.contains(end) {
            return None;
        }

        Some((start, Scalar { value, valid_range: v.with_end(end) }))
    }
}

#[derive(PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct Layout {
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

impl Layout {
    pub fn scalar<C: HasDataLayout>(cx: &C, scalar: Scalar) -> Self {
        let largest_niche = Niche::from_scalar(cx, Size::ZERO, scalar);
        let size = scalar.value.size(cx);
        let align = scalar.value.align(cx);
        Layout {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Primitive,
            abi: Abi::Scalar(scalar),
            largest_niche,
            size,
            align,
        }
    }
}

/// The layout of a type, alongside the type itself.
/// Provides various type traversal APIs (e.g., recursing into fields).
///
/// Note that the layout is NOT guaranteed to always be identical
/// to that obtained from `layout_of(ty)`, as we need to produce
/// layouts for which Rust types do not exist, such as enum variants
/// or synthetic fields of enums (i.e., discriminants) and fat pointers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable_Generic)]
pub struct TyAndLayout<'a, Ty> {
    pub ty: Ty,
    pub layout: &'a Layout,
}

impl<'a, Ty> Deref for TyAndLayout<'a, Ty> {
    type Target = &'a Layout;
    fn deref(&self) -> &&'a Layout {
        &self.layout
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PointerKind {
    /// Most general case, we know no restrictions to tell LLVM.
    Shared,

    /// `&T` where `T` contains no `UnsafeCell`, is `noalias` and `readonly`.
    Frozen,

    /// `&mut T` which is `noalias` but not `readonly`.
    UniqueBorrowed,

    /// `Box<T>`, unlike `UniqueBorrowed`, it also has `noalias` on returns.
    UniqueOwned,
}

#[derive(Copy, Clone, Debug)]
pub struct PointeeInfo {
    pub size: Size,
    pub align: Align,
    pub safe: Option<PointerKind>,
    pub address_space: AddressSpace,
}

/// Trait that needs to be implemented by the higher-level type representation
/// (e.g. `rustc_middle::ty::Ty`), to provide `rustc_target::abi` functionality.
pub trait TyAbiInterface<'a, C>: Sized {
    fn ty_and_layout_for_variant(
        this: TyAndLayout<'a, Self>,
        cx: &C,
        variant_index: VariantIdx,
    ) -> TyAndLayout<'a, Self>;
    fn ty_and_layout_field(this: TyAndLayout<'a, Self>, cx: &C, i: usize) -> TyAndLayout<'a, Self>;
    fn ty_and_layout_pointee_info_at(
        this: TyAndLayout<'a, Self>,
        cx: &C,
        offset: Size,
    ) -> Option<PointeeInfo>;
}

impl<'a, Ty> TyAndLayout<'a, Ty> {
    pub fn for_variant<C>(self, cx: &C, variant_index: VariantIdx) -> Self
    where
        Ty: TyAbiInterface<'a, C>,
    {
        Ty::ty_and_layout_for_variant(self, cx, variant_index)
    }

    pub fn field<C>(self, cx: &C, i: usize) -> Self
    where
        Ty: TyAbiInterface<'a, C>,
    {
        Ty::ty_and_layout_field(self, cx, i)
    }

    pub fn pointee_info_at<C>(self, cx: &C, offset: Size) -> Option<PointeeInfo>
    where
        Ty: TyAbiInterface<'a, C>,
    {
        Ty::ty_and_layout_pointee_info_at(self, cx, offset)
    }
}

impl<'a, Ty> TyAndLayout<'a, Ty> {
    /// Returns `true` if the layout corresponds to an unsized type.
    pub fn is_unsized(&self) -> bool {
        self.abi.is_unsized()
    }

    /// Returns `true` if the type is a ZST and not unsized.
    pub fn is_zst(&self) -> bool {
        match self.abi {
            Abi::Scalar(_) | Abi::ScalarPair(..) | Abi::Vector { .. } => false,
            Abi::Uninhabited => self.size.bytes() == 0,
            Abi::Aggregate { sized } => sized && self.size.bytes() == 0,
        }
    }

    /// Determines if this type permits "raw" initialization by just transmuting some
    /// memory into an instance of `T`.
    /// `zero` indicates if the memory is zero-initialized, or alternatively
    /// left entirely uninitialized.
    /// This is conservative: in doubt, it will answer `true`.
    ///
    /// FIXME: Once we removed all the conservatism, we could alternatively
    /// create an all-0/all-undef constant and run the const value validator to see if
    /// this is a valid value for the given type.
    pub fn might_permit_raw_init<C>(self, cx: &C, zero: bool) -> bool
    where
        Self: Copy,
        Ty: TyAbiInterface<'a, C>,
        C: HasDataLayout,
    {
        let scalar_allows_raw_init = move |s: Scalar| -> bool {
            if zero {
                // The range must contain 0.
                s.valid_range.contains(0)
            } else {
                // The range must include all values.
                s.is_always_valid(cx)
            }
        };

        // Check the ABI.
        let valid = match self.abi {
            Abi::Uninhabited => false, // definitely UB
            Abi::Scalar(s) => scalar_allows_raw_init(s),
            Abi::ScalarPair(s1, s2) => scalar_allows_raw_init(s1) && scalar_allows_raw_init(s2),
            Abi::Vector { element: s, count } => count == 0 || scalar_allows_raw_init(s),
            Abi::Aggregate { .. } => true, // Fields are checked below.
        };
        if !valid {
            // This is definitely not okay.
            return false;
        }

        // If we have not found an error yet, we need to recursively descend into fields.
        match &self.fields {
            FieldsShape::Primitive | FieldsShape::Union { .. } => {}
            FieldsShape::Array { .. } => {
                // FIXME(#66151): For now, we are conservative and do not check arrays.
            }
            FieldsShape::Arbitrary { offsets, .. } => {
                for idx in 0..offsets.len() {
                    if !self.field(cx, idx).might_permit_raw_init(cx, zero) {
                        // We found a field that is unhappy with this kind of initialization.
                        return false;
                    }
                }
            }
        }

        // FIXME(#66151): For now, we are conservative and do not check `self.variants`.
        true
    }
}
