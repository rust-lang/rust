// tidy-alphabetical-start
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(feature = "nightly", doc(rust_logo))]
#![cfg_attr(feature = "nightly", feature(assert_matches))]
#![cfg_attr(feature = "nightly", feature(rustc_attrs))]
#![cfg_attr(feature = "nightly", feature(rustdoc_internals))]
#![cfg_attr(feature = "nightly", feature(step_trait))]
// tidy-alphabetical-end

/*! ABI handling for rustc

## What is an "ABI"?

Literally, "application binary interface", which means it is everything about how code interacts,
at the machine level, with other code. This means it technically covers all of the following:
- object binary format for e.g. relocations or offset tables
- in-memory layout of types
- procedure calling conventions

When we discuss "ABI" in the context of rustc, we are probably discussing calling conventions.
To describe those `rustc_abi` also covers type layout, as it must for values passed on the stack.
Despite `rustc_abi` being about calling conventions, it is good to remember these usages exist.
You will encounter all of them and more if you study target-specific codegen enough!
Even in general conversation, when someone says "the Rust ABI is unstable", it may allude to
either or both of
- `repr(Rust)` types have a mostly-unspecified layout
- `extern "Rust" fn(A) -> R` has an unspecified calling convention

## Crate Goal

ABI is a foundational concept, so the `rustc_abi` crate serves as an equally foundational crate.
It cannot carry all details relevant to an ABI: those permeate code generation and linkage.
Instead, `rustc_abi` is intended to provide the interface for reasoning about the binary interface.
It should contain traits and types that other crates then use in their implementation.
For example, a platform's `extern "C" fn` calling convention will be implemented in `rustc_target`
but `rustc_abi` contains the types for calculating layout and describing register-passing.
This makes it easier to describe things in the same way across targets, codegen backends, and
even other Rust compilers, such as rust-analyzer!

*/

use std::fmt;
#[cfg(feature = "nightly")]
use std::iter::Step;
use std::num::{NonZeroUsize, ParseIntError};
use std::ops::{Add, AddAssign, Mul, RangeInclusive, Sub};
use std::str::FromStr;

use bitflags::bitflags;
#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::StableOrd;
use rustc_hashes::Hash64;
use rustc_index::{Idx, IndexSlice, IndexVec};
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_Generic};

mod callconv;
mod canon_abi;
mod extern_abi;
mod layout;
#[cfg(test)]
mod tests;

pub use callconv::{Heterogeneous, HomogeneousAggregate, Reg, RegKind};
pub use canon_abi::{ArmCall, CanonAbi, InterruptKind, X86Call};
pub use extern_abi::{ExternAbi, all_names};
#[cfg(feature = "nightly")]
pub use layout::{FIRST_VARIANT, FieldIdx, Layout, TyAbiInterface, TyAndLayout, VariantIdx};
pub use layout::{LayoutCalculator, LayoutCalculatorError};

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
#[cfg(feature = "nightly")]
pub trait HashStableContext {}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_Generic)
)]
pub struct ReprFlags(u8);

bitflags! {
    impl ReprFlags: u8 {
        const IS_C               = 1 << 0;
        const IS_SIMD            = 1 << 1;
        const IS_TRANSPARENT     = 1 << 2;
        // Internal only for now. If true, don't reorder fields.
        // On its own it does not prevent ABI optimizations.
        const IS_LINEAR          = 1 << 3;
        // If true, the type's crate has opted into layout randomization.
        // Other flags can still inhibit reordering and thus randomization.
        // The seed stored in `ReprOptions.field_shuffle_seed`.
        const RANDOMIZE_LAYOUT   = 1 << 4;
        // Any of these flags being set prevent field reordering optimisation.
        const FIELD_ORDER_UNOPTIMIZABLE   = ReprFlags::IS_C.bits()
                                 | ReprFlags::IS_SIMD.bits()
                                 | ReprFlags::IS_LINEAR.bits();
        const ABI_UNOPTIMIZABLE = ReprFlags::IS_C.bits() | ReprFlags::IS_SIMD.bits();
    }
}

// This is the same as `rustc_data_structures::external_bitflags_debug` but without the
// `rustc_data_structures` to make it build on stable.
impl std::fmt::Debug for ReprFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_Generic)
)]
pub enum IntegerType {
    /// Pointer-sized integer type, i.e. `isize` and `usize`. The field shows signedness, e.g.
    /// `Pointer(true)` means `isize`.
    Pointer(bool),
    /// Fixed-sized integer type, e.g. `i8`, `u32`, `i128`. The bool field shows signedness, e.g.
    /// `Fixed(I8, false)` means `u8`.
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

/// Represents the repr options provided by the user.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_Generic)
)]
pub struct ReprOptions {
    pub int: Option<IntegerType>,
    pub align: Option<Align>,
    pub pack: Option<Align>,
    pub flags: ReprFlags,
    /// The seed to be used for randomizing a type's layout
    ///
    /// Note: This could technically be a `u128` which would
    /// be the "most accurate" hash as it'd encompass the item and crate
    /// hash without loss, but it does pay the price of being larger.
    /// Everything's a tradeoff, a 64-bit seed should be sufficient for our
    /// purposes (primarily `-Z randomize-layout`)
    pub field_shuffle_seed: Hash64,
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

    pub fn inhibit_newtype_abi_optimization(&self) -> bool {
        self.flags.intersects(ReprFlags::ABI_UNOPTIMIZABLE)
    }

    /// Returns `true` if this `#[repr()]` guarantees a fixed field order,
    /// e.g. `repr(C)` or `repr(<int>)`.
    pub fn inhibit_struct_field_reordering(&self) -> bool {
        self.flags.intersects(ReprFlags::FIELD_ORDER_UNOPTIMIZABLE) || self.int.is_some()
    }

    /// Returns `true` if this type is valid for reordering and `-Z randomize-layout`
    /// was enabled for its declaration crate.
    pub fn can_randomize_type_layout(&self) -> bool {
        !self.inhibit_struct_field_reordering() && self.flags.contains(ReprFlags::RANDOMIZE_LAYOUT)
    }

    /// Returns `true` if this `#[repr()]` should inhibit union ABI optimisations.
    pub fn inhibits_union_abi_opt(&self) -> bool {
        self.c()
    }
}

/// The maximum supported number of lanes in a SIMD vector.
///
/// This value is selected based on backend support:
/// * LLVM does not appear to have a vector width limit.
/// * Cranelift stores the base-2 log of the lane count in a 4 bit integer.
pub const MAX_SIMD_LANES: u64 = 1 << 0xF;

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
    pub f16_align: AbiAndPrefAlign,
    pub f32_align: AbiAndPrefAlign,
    pub f64_align: AbiAndPrefAlign,
    pub f128_align: AbiAndPrefAlign,
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
            f16_align: AbiAndPrefAlign::new(align(16)),
            f32_align: AbiAndPrefAlign::new(align(32)),
            f64_align: AbiAndPrefAlign::new(align(64)),
            f128_align: AbiAndPrefAlign::new(align(128)),
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
    InvalidAlignment { cause: &'a str, err: AlignFromBytesError },
    InconsistentTargetArchitecture { dl: &'a str, target: &'a str },
    InconsistentTargetPointerWidth { pointer_size: u64, target: u32 },
    InvalidBitsSize { err: String },
}

impl TargetDataLayout {
    /// Parse data layout from an
    /// [llvm data layout string](https://llvm.org/docs/LangRef.html#data-layout)
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
        let parse_size =
            |s: &'a str, cause: &'a str| parse_bits(s, "size", cause).map(Size::from_bits);

        // Parse an alignment string.
        let parse_align = |s: &[&'a str], cause: &'a str| {
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
                ["a", a @ ..] => dl.aggregate_align = parse_align(a, "a")?,
                ["f16", a @ ..] => dl.f16_align = parse_align(a, "f16")?,
                ["f32", a @ ..] => dl.f32_align = parse_align(a, "f32")?,
                ["f64", a @ ..] => dl.f64_align = parse_align(a, "f64")?,
                ["f128", a @ ..] => dl.f128_align = parse_align(a, "f128")?,
                // FIXME(erikdesjardins): we should be parsing nonzero address spaces
                // this will require replacing TargetDataLayout::{pointer_size,pointer_align}
                // with e.g. `fn pointer_size_in(AddressSpace)`
                [p @ "p", s, a @ ..] | [p @ "p0", s, a @ ..] => {
                    dl.pointer_size = parse_size(s, p)?;
                    dl.pointer_align = parse_align(a, p)?;
                }
                [s, a @ ..] if s.starts_with('i') => {
                    let Ok(bits) = s[1..].parse::<u64>() else {
                        parse_size(&s[1..], "i")?; // For the user error.
                        continue;
                    };
                    let a = parse_align(a, s)?;
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
                [s, a @ ..] if s.starts_with('v') => {
                    let v_size = parse_size(&s[1..], "v")?;
                    let a = parse_align(a, s)?;
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

    /// Returns **exclusive** upper bound on object size in bytes.
    ///
    /// The theoretical maximum object size is defined as the maximum positive `isize` value.
    /// This ensures that the `offset` semantics remain well-defined by allowing it to correctly
    /// index every address within an object along with one byte past the end, along with allowing
    /// `isize` to store the difference between any two pointers into an object.
    ///
    /// LLVM uses a 64-bit integer to represent object size in *bits*, but we care only for bytes,
    /// so we adopt such a more-constrained size bound due to its technical limitations.
    #[inline]
    pub fn obj_size_bound(&self) -> u64 {
        match self.pointer_size.bits() {
            16 => 1 << 15,
            32 => 1 << 31,
            64 => 1 << 61,
            bits => panic!("obj_size_bound: unknown pointer bit size {bits}"),
        }
    }

    #[inline]
    pub fn ptr_sized_integer(&self) -> Integer {
        use Integer::*;
        match self.pointer_size.bits() {
            16 => I16,
            32 => I32,
            64 => I64,
            bits => panic!("ptr_sized_integer: unknown pointer bit size {bits}"),
        }
    }

    /// psABI-mandated alignment for a vector type, if any
    #[inline]
    fn cabi_vector_align(&self, vec_size: Size) -> Option<AbiAndPrefAlign> {
        self.vector_align
            .iter()
            .find(|(size, _align)| *size == vec_size)
            .map(|(_size, align)| *align)
    }

    /// an alignment resembling the one LLVM would pick for a vector
    #[inline]
    pub fn llvmlike_vector_align(&self, vec_size: Size) -> AbiAndPrefAlign {
        self.cabi_vector_align(vec_size).unwrap_or(AbiAndPrefAlign::new(
            Align::from_bytes(vec_size.bytes().next_power_of_two()).unwrap(),
        ))
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

// used by rust-analyzer
impl HasDataLayout for &TargetDataLayout {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        (**self).data_layout()
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
            _ => Err(format!(r#"unknown endian: "{s}""#)),
        }
    }
}

/// Size of a type in bytes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_Generic)
)]
pub struct Size {
    raw: u64,
}

#[cfg(feature = "nightly")]
impl StableOrd for Size {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // `Ord` is implemented as just comparing numerical values and numerical values
    // are not changed by (de-)serialization.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

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
            panic!("Size::bits: {bytes} bytes in bits doesn't fit in u64")
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
    pub fn sign_extend(self, value: u128) -> i128 {
        let size = self.bits();
        if size == 0 {
            // Truncated until nothing is left.
            return 0;
        }
        // Sign-extend it.
        let shift = 128 - size;
        // Shift the unsigned value to the left, then shift back to the right as signed
        // (essentially fills with sign bit on the left).
        ((value << shift) as i128) >> shift
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
        self.sign_extend(1_u128 << (self.bits() - 1))
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
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
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
        Self::from_bytes(unsafe { u64::forward_unchecked(start.bytes(), count) })
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
        Self::from_bytes(unsafe { u64::backward_unchecked(start.bytes(), count) })
    }
}

/// Alignment of a type in bytes (always a power of two).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_Generic)
)]
pub struct Align {
    pow2: u8,
}

// This is debug-printed a lot in larger structs, don't waste too much space there
impl fmt::Debug for Align {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Align({} bytes)", self.bytes())
    }
}

#[derive(Clone, Copy)]
pub enum AlignFromBytesError {
    NotPowerOfTwo(u64),
    TooLarge(u64),
}

impl AlignFromBytesError {
    pub fn diag_ident(self) -> &'static str {
        match self {
            Self::NotPowerOfTwo(_) => "not_power_of_two",
            Self::TooLarge(_) => "too_large",
        }
    }

    pub fn align(self) -> u64 {
        let (Self::NotPowerOfTwo(align) | Self::TooLarge(align)) = self;
        align
    }
}

impl fmt::Debug for AlignFromBytesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for AlignFromBytesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlignFromBytesError::NotPowerOfTwo(align) => write!(f, "`{align}` is not a power of 2"),
            AlignFromBytesError::TooLarge(align) => write!(f, "`{align}` is too large"),
        }
    }
}

impl Align {
    pub const ONE: Align = Align { pow2: 0 };
    pub const EIGHT: Align = Align { pow2: 3 };
    // LLVM has a maximal supported alignment of 2^29, we inherit that.
    pub const MAX: Align = Align { pow2: 29 };

    #[inline]
    pub fn from_bits(bits: u64) -> Result<Align, AlignFromBytesError> {
        Align::from_bytes(Size::from_bits(bits).bytes())
    }

    #[inline]
    pub const fn from_bytes(align: u64) -> Result<Align, AlignFromBytesError> {
        // Treat an alignment of 0 bytes like 1-byte alignment.
        if align == 0 {
            return Ok(Align::ONE);
        }

        #[cold]
        const fn not_power_of_2(align: u64) -> AlignFromBytesError {
            AlignFromBytesError::NotPowerOfTwo(align)
        }

        #[cold]
        const fn too_large(align: u64) -> AlignFromBytesError {
            AlignFromBytesError::TooLarge(align)
        }

        let tz = align.trailing_zeros();
        if align != (1 << tz) {
            return Err(not_power_of_2(align));
        }

        let pow2 = tz as u8;
        if pow2 > Self::MAX.pow2 {
            return Err(too_large(align));
        }

        Ok(Align { pow2 })
    }

    #[inline]
    pub const fn bytes(self) -> u64 {
        1 << self.pow2
    }

    #[inline]
    pub fn bytes_usize(self) -> usize {
        self.bytes().try_into().unwrap()
    }

    #[inline]
    pub const fn bits(self) -> u64 {
        self.bytes() * 8
    }

    #[inline]
    pub fn bits_usize(self) -> usize {
        self.bits().try_into().unwrap()
    }

    /// Obtain the greatest factor of `size` that is an alignment
    /// (the largest power of two the Size is a multiple of).
    ///
    /// Note that all numbers are factors of 0
    #[inline]
    pub fn max_aligned_factor(size: Size) -> Align {
        Align { pow2: size.bytes().trailing_zeros() as u8 }
    }

    /// Reduces Align to an aligned factor of `size`.
    #[inline]
    pub fn restrict_for_offset(self, size: Size) -> Align {
        self.min(Align::max_aligned_factor(size))
    }
}

/// A pair of alignments, ABI-mandated and preferred.
///
/// The "preferred" alignment is an LLVM concept that is virtually meaningless to Rust code:
/// it is not exposed semantically to programmers nor can they meaningfully affect it.
/// The only concern for us is that preferred alignment must not be less than the mandated alignment
/// and thus in practice the two values are almost always identical.
///
/// An example of a rare thing actually affected by preferred alignment is aligning of statics.
/// It is of effectively no consequence for layout in structs and on the stack.
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
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_Generic)
)]
pub enum Integer {
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl Integer {
    pub fn int_ty_str(self) -> &'static str {
        use Integer::*;
        match self {
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            I128 => "i128",
        }
    }

    pub fn uint_ty_str(self) -> &'static str {
        use Integer::*;
        match self {
            I8 => "u8",
            I16 => "u16",
            I32 => "u32",
            I64 => "u64",
            I128 => "u128",
        }
    }

    #[inline]
    pub fn size(self) -> Size {
        use Integer::*;
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
        use Integer::*;
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
        use Integer::*;
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
        use Integer::*;
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
        use Integer::*;
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
        use Integer::*;
        let dl = cx.data_layout();

        [I8, I16, I32, I64, I128].into_iter().find(|&candidate| {
            wanted == candidate.align(dl).abi && wanted.bytes() == candidate.size().bytes()
        })
    }

    /// Find the largest integer with the given alignment or less.
    pub fn approximate_align<C: HasDataLayout>(cx: &C, wanted: Align) -> Integer {
        use Integer::*;
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

/// Floating-point types.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum Float {
    F16,
    F32,
    F64,
    F128,
}

impl Float {
    pub fn size(self) -> Size {
        use Float::*;

        match self {
            F16 => Size::from_bits(16),
            F32 => Size::from_bits(32),
            F64 => Size::from_bits(64),
            F128 => Size::from_bits(128),
        }
    }

    pub fn align<C: HasDataLayout>(self, cx: &C) -> AbiAndPrefAlign {
        use Float::*;
        let dl = cx.data_layout();

        match self {
            F16 => dl.f16_align,
            F32 => dl.f32_align,
            F64 => dl.f64_align,
            F128 => dl.f128_align,
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
    Float(Float),
    Pointer(AddressSpace),
}

impl Primitive {
    pub fn size<C: HasDataLayout>(self, cx: &C) -> Size {
        use Primitive::*;
        let dl = cx.data_layout();

        match self {
            Int(i, _) => i.size(),
            Float(f) => f.size(),
            // FIXME(erikdesjardins): ignoring address space is technically wrong, pointers in
            // different address spaces can have different sizes
            // (but TargetDataLayout doesn't currently parse that part of the DL string)
            Pointer(_) => dl.pointer_size,
        }
    }

    pub fn align<C: HasDataLayout>(self, cx: &C) -> AbiAndPrefAlign {
        use Primitive::*;
        let dl = cx.data_layout();

        match self {
            Int(i, _) => i.align(dl),
            Float(f) => f.align(dl),
            // FIXME(erikdesjardins): ignoring address space is technically wrong, pointers in
            // different address spaces can have different alignments
            // (but TargetDataLayout doesn't currently parse that part of the DL string)
            Pointer(_) => dl.pointer_align,
        }
    }
}

/// Inclusive wrap-around range of valid values, that is, if
/// start > end, it represents `start..=MAX`, followed by `0..=end`.
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
    fn with_start(mut self, start: u128) -> Self {
        self.start = start;
        self
    }

    /// Returns `self` with replaced `end`
    #[inline(always)]
    fn with_end(mut self, end: u128) -> Self {
        self.end = end;
        self
    }

    /// Returns `true` if `size` completely fills the range.
    #[inline]
    fn is_full_for(&self, size: Size) -> bool {
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
        use Integer::*;
        matches!(
            self,
            Scalar::Initialized {
                value: Primitive::Int(I8, false),
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
    /// Allows the caller to mutate the valid range. This operation will panic if attempted on a
    /// union.
    pub fn valid_range_mut(&mut self) -> &mut WrappingRange {
        match self {
            Scalar::Initialized { valid_range, .. } => valid_range,
            Scalar::Union { .. } => panic!("cannot change the valid range of a union"),
        }
    }

    /// Returns `true` if all possible numbers are valid, i.e `valid_range` covers the whole
    /// layout.
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

    /// Returns `true` if this is a signed integer scalar
    #[inline]
    pub fn is_signed(&self) -> bool {
        match self.primitive() {
            Primitive::Int(_, signed) => signed,
            _ => false,
        }
    }
}

// NOTE: This struct is generic over the FieldIdx for rust-analyzer usage.
/// Describes how the fields of a type are located in memory.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum FieldsShape<FieldIdx: Idx> {
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
        offsets: IndexVec<FieldIdx, Size>,

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
        memory_index: IndexVec<FieldIdx, u32>,
    },
}

impl<FieldIdx: Idx> FieldsShape<FieldIdx> {
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
                assert!(i < count.get(), "tried to access field {i} of union with {count} fields");
                Size::ZERO
            }
            FieldsShape::Array { stride, count } => {
                let i = u64::try_from(i).unwrap();
                assert!(i < count, "tried to access field {i} of array with {count} fields");
                stride * i
            }
            FieldsShape::Arbitrary { ref offsets, .. } => offsets[FieldIdx::new(i)],
        }
    }

    #[inline]
    pub fn memory_index(&self, i: usize) -> usize {
        match *self {
            FieldsShape::Primitive => {
                unreachable!("FieldsShape::memory_index: `Primitive`s have no fields")
            }
            FieldsShape::Union(_) | FieldsShape::Array { .. } => i,
            FieldsShape::Arbitrary { ref memory_index, .. } => {
                memory_index[FieldIdx::new(i)].try_into().unwrap()
            }
        }
    }

    /// Gets source indices of the fields by increasing offsets.
    #[inline]
    pub fn index_by_increasing_offset(&self) -> impl ExactSizeIterator<Item = usize> {
        let mut inverse_small = [0u8; 64];
        let mut inverse_big = IndexVec::new();
        let use_small = self.count() <= inverse_small.len();

        // We have to write this logic twice in order to keep the array small.
        if let FieldsShape::Arbitrary { ref memory_index, .. } = *self {
            if use_small {
                for (field_idx, &mem_idx) in memory_index.iter_enumerated() {
                    inverse_small[mem_idx as usize] = field_idx.index() as u8;
                }
            } else {
                inverse_big = memory_index.invert_bijective_mapping();
            }
        }

        // Primitives don't really have fields in the way that structs do,
        // but having this return an empty iterator for them is unhelpful
        // since that makes them look kinda like ZSTs, which they're not.
        let pseudofield_count = if let FieldsShape::Primitive = self { 1 } else { self.count() };

        (0..pseudofield_count).map(move |i| match *self {
            FieldsShape::Primitive | FieldsShape::Union(_) | FieldsShape::Array { .. } => i,
            FieldsShape::Arbitrary { .. } => {
                if use_small {
                    inverse_small[i] as usize
                } else {
                    inverse_big[i as u32].index()
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

/// The way we represent values to the backend
///
/// Previously this was conflated with the "ABI" a type is given, as in the platform-specific ABI.
/// In reality, this implies little about that, but is mostly used to describe the syntactic form
/// emitted for the backend, as most backends handle SSA values and blobs of memory differently.
/// The psABI may need consideration in doing so, but this enum does not constitute a promise for
/// how the value will be lowered to the calling convention, in itself.
///
/// Generally, a codegen backend will prefer to handle smaller values as a scalar or short vector,
/// and larger values will usually prefer to be represented as memory.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum BackendRepr {
    Scalar(Scalar),
    ScalarPair(Scalar, Scalar),
    SimdVector {
        element: Scalar,
        count: u64,
    },
    // FIXME: I sometimes use memory, sometimes use an IR aggregate!
    Memory {
        /// If true, the size is exact, otherwise it's only a lower bound.
        sized: bool,
    },
}

impl BackendRepr {
    /// Returns `true` if the layout corresponds to an unsized type.
    #[inline]
    pub fn is_unsized(&self) -> bool {
        match *self {
            BackendRepr::Scalar(_)
            | BackendRepr::ScalarPair(..)
            | BackendRepr::SimdVector { .. } => false,
            BackendRepr::Memory { sized } => !sized,
        }
    }

    #[inline]
    pub fn is_sized(&self) -> bool {
        !self.is_unsized()
    }

    /// Returns `true` if this is a single signed integer scalar.
    /// Sanity check: panics if this is not a scalar type (see PR #70189).
    #[inline]
    pub fn is_signed(&self) -> bool {
        match self {
            BackendRepr::Scalar(scal) => scal.is_signed(),
            _ => panic!("`is_signed` on non-scalar ABI {self:?}"),
        }
    }

    /// Returns `true` if this is a scalar type
    #[inline]
    pub fn is_scalar(&self) -> bool {
        matches!(*self, BackendRepr::Scalar(_))
    }

    /// Returns `true` if this is a bool
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(*self, BackendRepr::Scalar(s) if s.is_bool())
    }

    /// The psABI alignment for a `Scalar` or `ScalarPair`
    ///
    /// `None` for other variants.
    pub fn scalar_align<C: HasDataLayout>(&self, cx: &C) -> Option<Align> {
        match *self {
            BackendRepr::Scalar(s) => Some(s.align(cx).abi),
            BackendRepr::ScalarPair(s1, s2) => Some(s1.align(cx).max(s2.align(cx)).abi),
            // The align of a Vector can vary in surprising ways
            BackendRepr::SimdVector { .. } | BackendRepr::Memory { .. } => None,
        }
    }

    /// The psABI size for a `Scalar` or `ScalarPair`
    ///
    /// `None` for other variants
    pub fn scalar_size<C: HasDataLayout>(&self, cx: &C) -> Option<Size> {
        match *self {
            // No padding in scalars.
            BackendRepr::Scalar(s) => Some(s.size(cx)),
            // May have some padding between the pair.
            BackendRepr::ScalarPair(s1, s2) => {
                let field2_offset = s1.size(cx).align_to(s2.align(cx).abi);
                let size = (field2_offset + s2.size(cx)).align_to(
                    self.scalar_align(cx)
                        // We absolutely must have an answer here or everything is FUBAR.
                        .unwrap(),
                );
                Some(size)
            }
            // The size of a Vector can vary in surprising ways
            BackendRepr::SimdVector { .. } | BackendRepr::Memory { .. } => None,
        }
    }

    /// Discard validity range information and allow undef.
    pub fn to_union(&self) -> Self {
        match *self {
            BackendRepr::Scalar(s) => BackendRepr::Scalar(s.to_union()),
            BackendRepr::ScalarPair(s1, s2) => {
                BackendRepr::ScalarPair(s1.to_union(), s2.to_union())
            }
            BackendRepr::SimdVector { element, count } => {
                BackendRepr::SimdVector { element: element.to_union(), count }
            }
            BackendRepr::Memory { .. } => BackendRepr::Memory { sized: true },
        }
    }

    pub fn eq_up_to_validity(&self, other: &Self) -> bool {
        match (self, other) {
            // Scalar, Vector, ScalarPair have `Scalar` in them where we ignore validity ranges.
            // We do *not* ignore the sign since it matters for some ABIs (e.g. s390x).
            (BackendRepr::Scalar(l), BackendRepr::Scalar(r)) => l.primitive() == r.primitive(),
            (
                BackendRepr::SimdVector { element: element_l, count: count_l },
                BackendRepr::SimdVector { element: element_r, count: count_r },
            ) => element_l.primitive() == element_r.primitive() && count_l == count_r,
            (BackendRepr::ScalarPair(l1, l2), BackendRepr::ScalarPair(r1, r2)) => {
                l1.primitive() == r1.primitive() && l2.primitive() == r2.primitive()
            }
            // Everything else must be strictly identical.
            _ => self == other,
        }
    }
}

// NOTE: This struct is generic over the FieldIdx and VariantIdx for rust-analyzer usage.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum Variants<FieldIdx: Idx, VariantIdx: Idx> {
    /// A type with no valid variants. Must be uninhabited.
    Empty,

    /// Single enum variants, structs/tuples, unions, and all non-ADTs.
    Single {
        /// Always `0` for types that cannot have multiple variants.
        index: VariantIdx,
    },

    /// Enum-likes with more than one variant: each variant comes with
    /// a *discriminant* (usually the same as the variant index but the user can
    /// assign explicit discriminant values). That discriminant is encoded
    /// as a *tag* on the machine. The layout of each variant is
    /// a struct, and they all have space reserved for the tag.
    /// For enums, the tag is the sole field of the layout.
    Multiple {
        tag: Scalar,
        tag_encoding: TagEncoding<VariantIdx>,
        tag_field: FieldIdx,
        variants: IndexVec<VariantIdx, LayoutData<FieldIdx, VariantIdx>>,
    },
}

// NOTE: This struct is generic over the VariantIdx for rust-analyzer usage.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum TagEncoding<VariantIdx: Idx> {
    /// The tag directly stores the discriminant, but possibly with a smaller layout
    /// (so converting the tag to the discriminant can require sign extension).
    Direct,

    /// Niche (values invalid for a type) encoding the discriminant:
    /// Discriminant and variant index coincide.
    /// The variant `untagged_variant` contains a niche at an arbitrary
    /// offset (field `tag_field` of the enum), which for a variant with
    /// discriminant `d` is set to
    /// `(d - niche_variants.start).wrapping_add(niche_start)`
    /// (this is wrapping arithmetic using the type of the niche field).
    ///
    /// For example, `Option<(usize, &T)>`  is represented such that
    /// `None` has a null pointer for the second tuple field, and
    /// `Some` is the identity function (with a non-null reference).
    ///
    /// Other variants that are not `untagged_variant` and that are outside the `niche_variants`
    /// range cannot be represented; they must be uninhabited.
    Niche {
        untagged_variant: VariantIdx,
        /// This range *may* contain `untagged_variant`; that is then just a "dead value" and
        /// not used to encode anything.
        niche_variants: RangeInclusive<VariantIdx>,
        /// This is inbounds of the type of the niche field
        /// (not sign-extended, i.e., all bits beyond the niche field size are 0).
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

        // Extend the range of valid values being reserved by moving either `v.start` or `v.end`
        // bound. Given an eventual `Option<T>`, we try to maximize the chance for `None` to occupy
        // the niche of zero. This is accomplished by preferring enums with 2 variants(`count==1`)
        // and always taking the shortest path to niche zero. Having `None` in niche zero can
        // enable some special optimizations.
        //
        // Bound selection criteria:
        // 1. Select closest to zero given wrapping semantics.
        // 2. Avoid moving past zero if possible.
        //
        // In practice this means that enums with `count > 1` are unlikely to claim niche zero,
        // since they have to fit perfectly. If niche zero is already reserved, the selection of
        // bounds are of little interest.
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

// NOTE: This struct is generic over the FieldIdx and VariantIdx for rust-analyzer usage.
#[derive(PartialEq, Eq, Hash, Clone)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub struct LayoutData<FieldIdx: Idx, VariantIdx: Idx> {
    /// Says where the fields are located within the layout.
    pub fields: FieldsShape<FieldIdx>,

    /// Encodes information about multi-variant layouts.
    /// Even with `Multiple` variants, a layout still has its own fields! Those are then
    /// shared between all variants. One of them will be the discriminant,
    /// but e.g. coroutines can have more.
    ///
    /// To access all fields of this layout, both `fields` and the fields of the active variant
    /// must be taken into account.
    pub variants: Variants<FieldIdx, VariantIdx>,

    /// The `backend_repr` defines how this data will be represented to the codegen backend,
    /// and encodes value restrictions via `valid_range`.
    ///
    /// Note that this is entirely orthogonal to the recursive structure defined by
    /// `variants` and `fields`; for example, `ManuallyDrop<Result<isize, isize>>` has
    /// `IrForm::ScalarPair`! So, even with non-`Memory` `backend_repr`, `fields` and `variants`
    /// have to be taken into account to find all fields of this layout.
    pub backend_repr: BackendRepr,

    /// The leaf scalar with the largest number of invalid values
    /// (i.e. outside of its `valid_range`), if it exists.
    pub largest_niche: Option<Niche>,
    /// Is this type known to be uninhabted?
    ///
    /// This is separate from BackendRepr because uninhabited return types can affect ABI,
    /// especially in the case of by-pointer struct returns, which allocate stack even when unused.
    pub uninhabited: bool,

    pub align: AbiAndPrefAlign,
    pub size: Size,

    /// The largest alignment explicitly requested with `repr(align)` on this type or any field.
    /// Only used on i686-windows, where the argument passing ABI is different when alignment is
    /// requested, even if the requested alignment is equal to the natural alignment.
    pub max_repr_align: Option<Align>,

    /// The alignment the type would have, ignoring any `repr(align)` but including `repr(packed)`.
    /// Only used on aarch64-linux, where the argument passing ABI ignores the requested alignment
    /// in some cases.
    pub unadjusted_abi_align: Align,

    /// The randomization seed based on this type's own repr and its fields.
    ///
    /// Since randomization is toggled on a per-crate basis even crates that do not have randomization
    /// enabled should still calculate a seed so that downstream uses can use it to distinguish different
    /// types.
    ///
    /// For every T and U for which we do not guarantee that a repr(Rust) `Foo<T>` can be coerced or
    /// transmuted to `Foo<U>` we aim to create probalistically distinct seeds so that Foo can choose
    /// to reorder its fields based on that information. The current implementation is a conservative
    /// approximation of this goal.
    pub randomization_seed: Hash64,
}

impl<FieldIdx: Idx, VariantIdx: Idx> LayoutData<FieldIdx, VariantIdx> {
    /// Returns `true` if this is an aggregate type (including a ScalarPair!)
    pub fn is_aggregate(&self) -> bool {
        match self.backend_repr {
            BackendRepr::Scalar(_) | BackendRepr::SimdVector { .. } => false,
            BackendRepr::ScalarPair(..) | BackendRepr::Memory { .. } => true,
        }
    }

    /// Returns `true` if this is an uninhabited type
    pub fn is_uninhabited(&self) -> bool {
        self.uninhabited
    }
}

impl<FieldIdx: Idx, VariantIdx: Idx> fmt::Debug for LayoutData<FieldIdx, VariantIdx>
where
    FieldsShape<FieldIdx>: fmt::Debug,
    Variants<FieldIdx, VariantIdx>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This is how `Layout` used to print before it become
        // `Interned<LayoutS>`. We print it like this to avoid having to update
        // expected output in a lot of tests.
        let LayoutData {
            size,
            align,
            backend_repr,
            fields,
            largest_niche,
            uninhabited,
            variants,
            max_repr_align,
            unadjusted_abi_align,
            randomization_seed,
        } = self;
        f.debug_struct("Layout")
            .field("size", size)
            .field("align", align)
            .field("backend_repr", backend_repr)
            .field("fields", fields)
            .field("largest_niche", largest_niche)
            .field("uninhabited", uninhabited)
            .field("variants", variants)
            .field("max_repr_align", max_repr_align)
            .field("unadjusted_abi_align", unadjusted_abi_align)
            .field("randomization_seed", randomization_seed)
            .finish()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PointerKind {
    /// Shared reference. `frozen` indicates the absence of any `UnsafeCell`.
    SharedRef { frozen: bool },
    /// Mutable reference. `unpin` indicates the absence of any pinned data.
    MutableRef { unpin: bool },
    /// Box. `unpin` indicates the absence of any pinned data. `global` indicates whether this box
    /// uses the global allocator or a custom one.
    Box { unpin: bool, global: bool },
}

/// Encodes extra information we have about a pointer.
/// Note that this information is advisory only, and backends are free to ignore it:
/// if the information is wrong, that can cause UB, but if the information is absent,
/// that must always be okay.
#[derive(Copy, Clone, Debug)]
pub struct PointeeInfo {
    /// If this is `None`, then this is a raw pointer, so size and alignment are not guaranteed to
    /// be reliable.
    pub safe: Option<PointerKind>,
    /// If `safe` is `Some`, then the pointer is either null or dereferenceable for this many bytes.
    /// On a function argument, "dereferenceable" here means "dereferenceable for the entire duration
    /// of this function call", i.e. it is UB for the memory that this pointer points to be freed
    /// while this function is still running.
    /// The size can be zero if the pointer is not dereferenceable.
    pub size: Size,
    /// If `safe` is `Some`, then the pointer is aligned as indicated.
    pub align: Align,
}

impl<FieldIdx: Idx, VariantIdx: Idx> LayoutData<FieldIdx, VariantIdx> {
    /// Returns `true` if the layout corresponds to an unsized type.
    #[inline]
    pub fn is_unsized(&self) -> bool {
        self.backend_repr.is_unsized()
    }

    #[inline]
    pub fn is_sized(&self) -> bool {
        self.backend_repr.is_sized()
    }

    /// Returns `true` if the type is sized and a 1-ZST (meaning it has size 0 and alignment 1).
    pub fn is_1zst(&self) -> bool {
        self.is_sized() && self.size.bytes() == 0 && self.align.abi.bytes() == 1
    }

    /// Returns `true` if the type is a ZST and not unsized.
    ///
    /// Note that this does *not* imply that the type is irrelevant for layout! It can still have
    /// non-trivial alignment constraints. You probably want to use `is_1zst` instead.
    pub fn is_zst(&self) -> bool {
        match self.backend_repr {
            BackendRepr::Scalar(_)
            | BackendRepr::ScalarPair(..)
            | BackendRepr::SimdVector { .. } => false,
            BackendRepr::Memory { sized } => sized && self.size.bytes() == 0,
        }
    }

    /// Checks if these two `Layout` are equal enough to be considered "the same for all function
    /// call ABIs". Note however that real ABIs depend on more details that are not reflected in the
    /// `Layout`; the `PassMode` need to be compared as well. Also note that we assume
    /// aggregates are passed via `PassMode::Indirect` or `PassMode::Cast`; more strict
    /// checks would otherwise be required.
    pub fn eq_abi(&self, other: &Self) -> bool {
        // The one thing that we are not capturing here is that for unsized types, the metadata must
        // also have the same ABI, and moreover that the same metadata leads to the same size. The
        // 2nd point is quite hard to check though.
        self.size == other.size
            && self.is_sized() == other.is_sized()
            && self.backend_repr.eq_up_to_validity(&other.backend_repr)
            && self.backend_repr.is_bool() == other.backend_repr.is_bool()
            && self.align.abi == other.align.abi
            && self.max_repr_align == other.max_repr_align
            && self.unadjusted_abi_align == other.unadjusted_abi_align
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

#[derive(Clone, Debug)]
pub enum AbiFromStrErr {
    /// not a known ABI
    Unknown,
    /// no "-unwind" variant can be used here
    NoExplicitUnwind,
}
