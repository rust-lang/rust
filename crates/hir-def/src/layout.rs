//! Definitions related to binary representations of types

use bitflags::bitflags;
use either::Either;
use std::{
    cmp, fmt,
    num::NonZeroUsize,
    ops::{Add, AddAssign, Mul, Sub},
};

use crate::{
    builtin_type::{BuiltinInt, BuiltinUint},
    LocalEnumVariantId,
};
use la_arena::ArenaMap;

/// Size of a type in bytes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Size {
    raw: u64,
}

// This is debug-printed a lot in larger structs, don't waste too much space there
impl fmt::Debug for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Size({} bytes)", self.raw)
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
    pub fn checked_add(self, offset: Size, dl: &TargetDataLayout) -> Option<Size> {
        let bytes = self.bytes().checked_add(offset.bytes())?;

        if bytes < dl.obj_size_bound() {
            Some(Size::from_bytes(bytes))
        } else {
            None
        }
    }

    #[inline]
    pub fn checked_mul(self, count: u64, dl: &TargetDataLayout) -> Option<Size> {
        let bytes = self.bytes().checked_mul(count)?;
        if bytes < dl.obj_size_bound() {
            Some(Size::from_bytes(bytes))
        } else {
            None
        }
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

#[derive(Copy, Clone, Debug)]
pub enum StructKind {
    /// A tuple, closure, or univariant which cannot be coerced to unsized.
    AlwaysSized,
    /// A univariant, the last field of which may be coerced to unsized.
    MaybeUnsized,
    /// A univariant, but with a prefix of an arbitrary size & alignment (e.g., enum tag).
    Prefixed(Size, Align),
}

/// Describes how the fields of a type are located in memory.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
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
    pub fn offset(&self, i: usize, dl: &TargetDataLayout) -> Size {
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
                stride.checked_mul(i, dl).unwrap()
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

/// Integers, also used for enum discriminants.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
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
            Integer::I8 => Size::from_bytes(1),
            Integer::I16 => Size::from_bytes(2),
            Integer::I32 => Size::from_bytes(4),
            Integer::I64 => Size::from_bytes(8),
            Integer::I128 => Size::from_bytes(16),
        }
    }

    pub fn align(self, dl: &TargetDataLayout) -> AbiAndPrefAlign {
        match self {
            Integer::I8 => dl.i8_align,
            Integer::I16 => dl.i16_align,
            Integer::I32 => dl.i32_align,
            Integer::I64 => dl.i64_align,
            Integer::I128 => dl.i128_align,
        }
    }

    /// Finds the smallest integer with the given alignment.
    pub fn for_align(dl: &TargetDataLayout, wanted: Align) -> Option<Integer> {
        use Integer::*;
        for candidate in [I8, I16, I32, I64, I128] {
            if wanted == candidate.align(dl).abi && wanted.bytes() == candidate.size().bytes() {
                return Some(candidate);
            }
        }
        None
    }

    /// Finds the smallest Integer type which can represent the signed value.
    #[inline]
    pub fn fit_signed(x: i128) -> Integer {
        match x {
            -0x0000_0000_0000_0080..=0x0000_0000_0000_007f => Integer::I8,
            -0x0000_0000_0000_8000..=0x0000_0000_0000_7fff => Integer::I16,
            -0x0000_0000_8000_0000..=0x0000_0000_7fff_ffff => Integer::I32,
            -0x8000_0000_0000_0000..=0x7fff_ffff_ffff_ffff => Integer::I64,
            _ => Integer::I128,
        }
    }

    /// Finds the smallest Integer type which can represent the unsigned value.
    #[inline]
    pub fn fit_unsigned(x: u128) -> Integer {
        match x {
            0..=0x0000_0000_0000_00ff => Integer::I8,
            0..=0x0000_0000_0000_ffff => Integer::I16,
            0..=0x0000_0000_ffff_ffff => Integer::I32,
            0..=0xffff_ffff_ffff_ffff => Integer::I64,
            _ => Integer::I128,
        }
    }

    /// Gets the Integer type from an attr::IntType.
    pub fn from_attr(dl: &TargetDataLayout, ity: Either<BuiltinInt, BuiltinUint>) -> Integer {
        match ity {
            Either::Left(BuiltinInt::I8) | Either::Right(BuiltinUint::U8) => Integer::I8,
            Either::Left(BuiltinInt::I16) | Either::Right(BuiltinUint::U16) => Integer::I16,
            Either::Left(BuiltinInt::I32) | Either::Right(BuiltinUint::U32) => Integer::I32,
            Either::Left(BuiltinInt::I64) | Either::Right(BuiltinUint::U64) => Integer::I64,
            Either::Left(BuiltinInt::I128) | Either::Right(BuiltinUint::U128) => Integer::I128,
            Either::Left(BuiltinInt::Isize) | Either::Right(BuiltinUint::Usize) => {
                dl.ptr_sized_integer()
            }
        }
    }

    /// Finds the appropriate Integer type and signedness for the given
    /// signed discriminant range and `#[repr]` attribute.
    /// N.B.: `u128` values above `i128::MAX` will be treated as signed, but
    /// that shouldn't affect anything, other than maybe debuginfo.
    pub fn repr_discr(
        dl: &TargetDataLayout,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> Result<(Integer, bool), LayoutError> {
        // Theoretically, negative values could be larger in unsigned representation
        // than the unsigned representation of the signed minimum. However, if there
        // are any negative values, the only valid unsigned representation is u128
        // which can fit all i128 values, so the result remains unaffected.
        let unsigned_fit = Integer::fit_unsigned(cmp::max(min as u128, max as u128));
        let signed_fit = cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

        if let Some(ity) = repr.int {
            let discr = Integer::from_attr(dl, ity);
            let fit = if ity.is_left() { signed_fit } else { unsigned_fit };
            if discr < fit {
                return Err(LayoutError::UserError(
                    "Integer::repr_discr: `#[repr]` hint too small for \
                      discriminant range of enum "
                        .to_string(),
                ));
            }
            return Ok((discr, ity.is_left()));
        }

        let at_least = if repr.c() {
            // This is usually I32, however it can be different on some platforms,
            // notably hexagon and arm-none/thumb-none
            dl.c_enum_min_size
        } else {
            // repr(Rust) enums try to be as small as possible
            Integer::I8
        };

        // If there are no negative values, we can use the unsigned fit.
        Ok(if min >= 0 {
            (cmp::max(unsigned_fit, at_least), false)
        } else {
            (cmp::max(signed_fit, at_least), true)
        })
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

/// An identifier that specifies the address space that some operation
/// should operate on. Special address spaces have an effect on code generation,
/// depending on the target and the address spaces it implements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddressSpace(pub u32);

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

    /// Minimum size of #[repr(C)] enums (default I32 bits)
    pub c_enum_min_size: Integer,
}

impl TargetDataLayout {
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
            16 => Integer::I16,
            32 => Integer::I32,
            64 => Integer::I64,
            bits => panic!("ptr_sized_integer: unknown pointer bit size {}", bits),
        }
    }
}

/// Fundamental unit of memory access and layout.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
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
    pub fn size(self, dl: &TargetDataLayout) -> Size {
        match self {
            Primitive::Int(i, _) => i.size(),
            Primitive::F32 => Size::from_bits(32),
            Primitive::F64 => Size::from_bits(64),
            Primitive::Pointer => dl.pointer_size,
        }
    }

    pub fn align(self, dl: &TargetDataLayout) -> AbiAndPrefAlign {
        match self {
            Primitive::Int(i, _) => i.align(dl),
            Primitive::F32 => dl.f32_align,
            Primitive::F64 => dl.f64_align,
            Primitive::Pointer => dl.pointer_align,
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
                value: Primitive::Int(Integer::I8, false),
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

    pub fn align(self, cx: &TargetDataLayout) -> AbiAndPrefAlign {
        self.primitive().align(cx)
    }

    pub fn size(self, cx: &TargetDataLayout) -> Size {
        self.primitive().size(cx)
    }

    #[inline]
    pub fn to_union(&self) -> Self {
        Self::Union { value: self.primitive() }
    }

    #[inline]
    pub fn valid_range(&self, cx: &TargetDataLayout) -> WrappingRange {
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
    pub fn is_always_valid(&self, cx: &TargetDataLayout) -> bool {
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

/// Describes how values of the type are passed by target ABIs,
/// in terms of categories of C types there are ABI rules for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

/// Alignment of a type in bytes (always a power of two).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Niche {
    pub offset: Size,
    pub value: Primitive,
    pub valid_range: WrappingRange,
}

impl Niche {
    pub fn from_scalar(cx: &TargetDataLayout, offset: Size, scalar: Scalar) -> Option<Self> {
        let (value, valid_range) = match scalar {
            Scalar::Initialized { value, valid_range } => (value, valid_range),
            _ => return None,
        };
        let niche = Niche { offset, value, valid_range };
        if niche.available(cx) > 0 {
            Some(niche)
        } else {
            None
        }
    }

    pub fn available(&self, cx: &TargetDataLayout) -> u128 {
        let Self { value, valid_range: v, .. } = *self;
        let size = value.size(cx);
        assert!(size.bits() <= 128);
        let max_value = size.unsigned_int_max();

        // Find out how many values are outside the valid range.
        let niche = v.end.wrapping_add(1)..v.start;
        niche.end.wrapping_sub(niche.start) & max_value
    }

    pub fn reserve(&self, cx: &TargetDataLayout, count: u128) -> Option<(u128, Scalar)> {
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

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
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
    Niche { untagged_variant: LocalEnumVariantId, niche_start: u128 },
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub enum Variants {
    /// Single enum variants, structs/tuples, unions, and all non-ADTs.
    Single,

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
        variants: ArenaMap<LocalEnumVariantId, Layout>,
    },
}

bitflags! {
    #[derive(Default)]
    pub struct ReprFlags: u8 {
        const IS_C               = 1 << 0;
        const IS_SIMD            = 1 << 1;
        const IS_TRANSPARENT     = 1 << 2;
        // Internal only for now. If true, don't reorder fields.
        const IS_LINEAR          = 1 << 3;
        // Any of these flags being set prevent field reordering optimisation.
        const IS_UNOPTIMISABLE   = ReprFlags::IS_C.bits
                                 | ReprFlags::IS_SIMD.bits
                                 | ReprFlags::IS_LINEAR.bits;
    }
}

/// Represents the repr options provided by the user,
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct ReprOptions {
    pub int: Option<Either<BuiltinInt, BuiltinUint>>,
    pub align: Option<Align>,
    pub pack: Option<Align>,
    pub flags: ReprFlags,
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
    pub fn discr_type(&self) -> Either<BuiltinInt, BuiltinUint> {
        self.int.unwrap_or(Either::Left(BuiltinInt::Isize))
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

    /// Returns `true` if this `#[repr()]` should inhibit union ABI optimisations.
    pub fn inhibit_union_abi_opt(&self) -> bool {
        self.c()
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
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
    pub fn scalar(dl: &TargetDataLayout, scalar: Scalar) -> Self {
        let largest_niche = Niche::from_scalar(dl, Size::ZERO, scalar);
        let size = scalar.size(dl);
        let align = scalar.align(dl);
        Layout {
            variants: Variants::Single,
            fields: FieldsShape::Primitive,
            abi: Abi::Scalar(scalar),
            largest_niche,
            size,
            align,
        }
    }
}

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This is how `Layout` used to print before it become
        // `Interned<LayoutS>`. We print it like this to avoid having to update
        // expected output in a lot of tests.
        let Layout { size, align, abi, fields, largest_niche, variants } = self;
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

impl Layout {
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
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LayoutError {
    UserError(String),
    SizeOverflow,
    HasPlaceholder,
    NotImplemented,
}
