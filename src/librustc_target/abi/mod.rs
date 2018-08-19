// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Integer::*;
pub use self::Primitive::*;

use spec::Target;

use std::{cmp, fmt};
use std::ops::{Add, Deref, Sub, Mul, AddAssign, Range, RangeInclusive};

pub mod call;

/// Parsed [Data layout](http://llvm.org/docs/LangRef.html#data-layout)
/// for a target, which contains everything needed to compute layouts.
pub struct TargetDataLayout {
    pub endian: Endian,
    pub i1_align: Align,
    pub i8_align: Align,
    pub i16_align: Align,
    pub i32_align: Align,
    pub i64_align: Align,
    pub i128_align: Align,
    pub f32_align: Align,
    pub f64_align: Align,
    pub pointer_size: Size,
    pub pointer_align: Align,
    pub aggregate_align: Align,

    /// Alignments for vector types.
    pub vector_align: Vec<(Size, Align)>
}

impl Default for TargetDataLayout {
    /// Creates an instance of `TargetDataLayout`.
    fn default() -> TargetDataLayout {
        TargetDataLayout {
            endian: Endian::Big,
            i1_align: Align::from_bits(8, 8).unwrap(),
            i8_align: Align::from_bits(8, 8).unwrap(),
            i16_align: Align::from_bits(16, 16).unwrap(),
            i32_align: Align::from_bits(32, 32).unwrap(),
            i64_align: Align::from_bits(32, 64).unwrap(),
            i128_align: Align::from_bits(32, 64).unwrap(),
            f32_align: Align::from_bits(32, 32).unwrap(),
            f64_align: Align::from_bits(64, 64).unwrap(),
            pointer_size: Size::from_bits(64),
            pointer_align: Align::from_bits(64, 64).unwrap(),
            aggregate_align: Align::from_bits(0, 64).unwrap(),
            vector_align: vec![
                (Size::from_bits(64), Align::from_bits(64, 64).unwrap()),
                (Size::from_bits(128), Align::from_bits(128, 128).unwrap())
            ]
        }
    }
}

impl TargetDataLayout {
    pub fn parse(target: &Target) -> Result<TargetDataLayout, String> {
        // Parse a bit count from a string.
        let parse_bits = |s: &str, kind: &str, cause: &str| {
            s.parse::<u64>().map_err(|err| {
                format!("invalid {} `{}` for `{}` in \"data-layout\": {}",
                        kind, s, cause, err)
            })
        };

        // Parse a size string.
        let size = |s: &str, cause: &str| {
            parse_bits(s, "size", cause).map(Size::from_bits)
        };

        // Parse an alignment string.
        let align = |s: &[&str], cause: &str| {
            if s.is_empty() {
                return Err(format!("missing alignment for `{}` in \"data-layout\"", cause));
            }
            let abi = parse_bits(s[0], "alignment", cause)?;
            let pref = s.get(1).map_or(Ok(abi), |pref| parse_bits(pref, "alignment", cause))?;
            Align::from_bits(abi, pref).map_err(|err| {
                format!("invalid alignment for `{}` in \"data-layout\": {}",
                        cause, err)
            })
        };

        let mut dl = TargetDataLayout::default();
        let mut i128_align_src = 64;
        for spec in target.data_layout.split('-') {
            match spec.split(':').collect::<Vec<_>>()[..] {
                ["e"] => dl.endian = Endian::Little,
                ["E"] => dl.endian = Endian::Big,
                ["a", ref a..] => dl.aggregate_align = align(a, "a")?,
                ["f32", ref a..] => dl.f32_align = align(a, "f32")?,
                ["f64", ref a..] => dl.f64_align = align(a, "f64")?,
                [p @ "p", s, ref a..] | [p @ "p0", s, ref a..] => {
                    dl.pointer_size = size(s, p)?;
                    dl.pointer_align = align(a, p)?;
                }
                [s, ref a..] if s.starts_with("i") => {
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
                        // largest-sized i{64...128}.
                        i128_align_src = bits;
                        dl.i128_align = a;
                    }
                }
                [s, ref a..] if s.starts_with("v") => {
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
        let endian_str = match dl.endian {
            Endian::Little => "little",
            Endian::Big => "big"
        };
        if endian_str != target.target_endian {
            return Err(format!("inconsistent target specification: \"data-layout\" claims \
                                architecture is {}-endian, while \"target-endian\" is `{}`",
                               endian_str, target.target_endian));
        }

        if dl.pointer_size.bits().to_string() != target.target_pointer_width {
            return Err(format!("inconsistent target specification: \"data-layout\" claims \
                                pointers are {}-bit, while \"target-pointer-width\" is `{}`",
                               dl.pointer_size.bits(), target.target_pointer_width));
        }

        Ok(dl)
    }

    /// Return exclusive upper bound on object size.
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
    pub fn obj_size_bound(&self) -> u64 {
        match self.pointer_size.bits() {
            16 => 1 << 15,
            32 => 1 << 31,
            64 => 1 << 47,
            bits => panic!("obj_size_bound: unknown pointer bit size {}", bits)
        }
    }

    pub fn ptr_sized_integer(&self) -> Integer {
        match self.pointer_size.bits() {
            16 => I16,
            32 => I32,
            64 => I64,
            bits => panic!("ptr_sized_integer: unknown pointer bit size {}", bits)
        }
    }

    pub fn vector_align(&self, vec_size: Size) -> Align {
        for &(size, align) in &self.vector_align {
            if size == vec_size {
                return align;
            }
        }
        // Default to natural alignment, which is what LLVM does.
        // That is, use the size, rounded up to a power of 2.
        let align = vec_size.bytes().next_power_of_two();
        Align::from_bytes(align, align).unwrap()
    }
}

pub trait HasDataLayout: Copy {
    fn data_layout(&self) -> &TargetDataLayout;
}

impl<'a> HasDataLayout for &'a TargetDataLayout {
    fn data_layout(&self) -> &TargetDataLayout {
        self
    }
}

/// Endianness of the target, which must match cfg(target-endian).
#[derive(Copy, Clone)]
pub enum Endian {
    Little,
    Big
}

/// Size of a type in bytes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct Size {
    raw: u64
}

impl Size {
    pub const ZERO: Size = Self::from_bytes(0);

    #[inline]
    pub fn from_bits(bits: u64) -> Size {
        // Avoid potential overflow from `bits + 7`.
        Size::from_bytes(bits / 8 + ((bits % 8) + 7) / 8)
    }

    #[inline]
    pub const fn from_bytes(bytes: u64) -> Size {
        Size {
            raw: bytes
        }
    }

    #[inline]
    pub fn bytes(self) -> u64 {
        self.raw
    }

    #[inline]
    pub fn bits(self) -> u64 {
        self.bytes().checked_mul(8).unwrap_or_else(|| {
            panic!("Size::bits: {} bytes in bits doesn't fit in u64", self.bytes())
        })
    }

    #[inline]
    pub fn abi_align(self, align: Align) -> Size {
        let mask = align.abi() - 1;
        Size::from_bytes((self.bytes() + mask) & !mask)
    }

    #[inline]
    pub fn is_abi_aligned(self, align: Align) -> bool {
        let mask = align.abi() - 1;
        self.bytes() & mask == 0
    }

    #[inline]
    pub fn checked_add<C: HasDataLayout>(self, offset: Size, cx: C) -> Option<Size> {
        let dl = cx.data_layout();

        let bytes = self.bytes().checked_add(offset.bytes())?;

        if bytes < dl.obj_size_bound() {
            Some(Size::from_bytes(bytes))
        } else {
            None
        }
    }

    #[inline]
    pub fn checked_mul<C: HasDataLayout>(self, count: u64, cx: C) -> Option<Size> {
        let dl = cx.data_layout();

        let bytes = self.bytes().checked_mul(count)?;
        if bytes < dl.obj_size_bound() {
            Some(Size::from_bytes(bytes))
        } else {
            None
        }
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
            None => {
                panic!("Size::mul: {} * {} doesn't fit in u64", self.bytes(), count)
            }
        }
    }
}

impl AddAssign for Size {
    #[inline]
    fn add_assign(&mut self, other: Size) {
        *self = *self + other;
    }
}

/// Alignment of a type in bytes, both ABI-mandated and preferred.
/// Each field is a power of two, giving the alignment a maximum value
/// of 2<sup>(2<sup>8</sup> - 1)</sup>, which is limited by LLVM to a
/// maximum capacity of 2<sup>29</sup> or 536870912.
#[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct Align {
    abi_pow2: u8,
    pref_pow2: u8,
}

impl Align {
    pub fn from_bits(abi: u64, pref: u64) -> Result<Align, String> {
        Align::from_bytes(Size::from_bits(abi).bytes(),
                          Size::from_bits(pref).bytes())
    }

    pub fn from_bytes(abi: u64, pref: u64) -> Result<Align, String> {
        let log2 = |align: u64| {
            // Treat an alignment of 0 bytes like 1-byte alignment.
            if align == 0 {
                return Ok(0);
            }

            let mut bytes = align;
            let mut pow: u8 = 0;
            while (bytes & 1) == 0 {
                pow += 1;
                bytes >>= 1;
            }
            if bytes != 1 {
                Err(format!("`{}` is not a power of 2", align))
            } else if pow > 29 {
                Err(format!("`{}` is too large", align))
            } else {
                Ok(pow)
            }
        };

        Ok(Align {
            abi_pow2: log2(abi)?,
            pref_pow2: log2(pref)?,
        })
    }

    pub fn abi(self) -> u64 {
        1 << self.abi_pow2
    }

    pub fn pref(self) -> u64 {
        1 << self.pref_pow2
    }

    pub fn abi_bits(self) -> u64 {
        self.abi() * 8
    }

    pub fn pref_bits(self) -> u64 {
        self.pref() * 8
    }

    pub fn min(self, other: Align) -> Align {
        Align {
            abi_pow2: cmp::min(self.abi_pow2, other.abi_pow2),
            pref_pow2: cmp::min(self.pref_pow2, other.pref_pow2),
        }
    }

    pub fn max(self, other: Align) -> Align {
        Align {
            abi_pow2: cmp::max(self.abi_pow2, other.abi_pow2),
            pref_pow2: cmp::max(self.pref_pow2, other.pref_pow2),
        }
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
    pub fn size(self) -> Size {
        match self {
            I8 => Size::from_bytes(1),
            I16 => Size::from_bytes(2),
            I32 => Size::from_bytes(4),
            I64  => Size::from_bytes(8),
            I128  => Size::from_bytes(16),
        }
    }

    pub fn align<C: HasDataLayout>(self, cx: C) -> Align {
        let dl = cx.data_layout();

        match self {
            I8 => dl.i8_align,
            I16 => dl.i16_align,
            I32 => dl.i32_align,
            I64 => dl.i64_align,
            I128 => dl.i128_align,
        }
    }

    /// Find the smallest Integer type which can represent the signed value.
    pub fn fit_signed(x: i128) -> Integer {
        match x {
            -0x0000_0000_0000_0080..=0x0000_0000_0000_007f => I8,
            -0x0000_0000_0000_8000..=0x0000_0000_0000_7fff => I16,
            -0x0000_0000_8000_0000..=0x0000_0000_7fff_ffff => I32,
            -0x8000_0000_0000_0000..=0x7fff_ffff_ffff_ffff => I64,
            _ => I128
        }
    }

    /// Find the smallest Integer type which can represent the unsigned value.
    pub fn fit_unsigned(x: u128) -> Integer {
        match x {
            0..=0x0000_0000_0000_00ff => I8,
            0..=0x0000_0000_0000_ffff => I16,
            0..=0x0000_0000_ffff_ffff => I32,
            0..=0xffff_ffff_ffff_ffff => I64,
            _ => I128,
        }
    }

    /// Find the smallest integer with the given alignment.
    pub fn for_abi_align<C: HasDataLayout>(cx: C, align: Align) -> Option<Integer> {
        let dl = cx.data_layout();

        let wanted = align.abi();
        for &candidate in &[I8, I16, I32, I64, I128] {
            if wanted == candidate.align(dl).abi() && wanted == candidate.size().bytes() {
                return Some(candidate);
            }
        }
        None
    }

    /// Find the largest integer with the given alignment or less.
    pub fn approximate_abi_align<C: HasDataLayout>(cx: C, align: Align) -> Integer {
        let dl = cx.data_layout();

        let wanted = align.abi();
        // FIXME(eddyb) maybe include I128 in the future, when it works everywhere.
        for &candidate in &[I64, I32, I16] {
            if wanted >= candidate.align(dl).abi() && wanted >= candidate.size().bytes() {
                return candidate;
            }
        }
        I8
    }
}


#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy,
         PartialOrd, Ord)]
pub enum FloatTy {
    F32,
    F64,
}

impl fmt::Debug for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ty_to_string())
    }
}

impl FloatTy {
    pub fn ty_to_string(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }

    pub fn bit_width(self) -> usize {
        match self {
            FloatTy::F32 => 32,
            FloatTy::F64 => 64,
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
    Float(FloatTy),
    Pointer
}

impl<'a, 'tcx> Primitive {
    pub fn size<C: HasDataLayout>(self, cx: C) -> Size {
        let dl = cx.data_layout();

        match self {
            Int(i, _) => i.size(),
            Float(FloatTy::F32) => Size::from_bits(32),
            Float(FloatTy::F64) => Size::from_bits(64),
            Pointer => dl.pointer_size
        }
    }

    pub fn align<C: HasDataLayout>(self, cx: C) -> Align {
        let dl = cx.data_layout();

        match self {
            Int(i, _) => i.align(dl),
            Float(FloatTy::F32) => dl.f32_align,
            Float(FloatTy::F64) => dl.f64_align,
            Pointer => dl.pointer_align
        }
    }

    pub fn is_float(self) -> bool {
        match self {
            Float(_) => true,
            _ => false
        }
    }

    pub fn is_int(self) -> bool {
        match self {
            Int(..) => true,
            _ => false,
        }
    }
}

/// Information about one scalar component of a Rust type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Scalar {
    pub value: Primitive,

    /// Inclusive wrap-around range of valid values, that is, if
    /// start > end, it represents `start..=max_value()`,
    /// followed by `0..=end`.
    ///
    /// That is, for an i8 primitive, a range of `254..=2` means following
    /// sequence:
    ///
    ///    254 (-2), 255 (-1), 0, 1, 2
    ///
    /// This is intended specifically to mirror LLVM’s `!range` metadata,
    /// semantics.
    // FIXME(eddyb) always use the shortest range, e.g. by finding
    // the largest space between two consecutive valid values and
    // taking everything else as the (shortest) valid range.
    pub valid_range: RangeInclusive<u128>,
}

impl Scalar {
    pub fn is_bool(&self) -> bool {
        if let Int(I8, _) = self.value {
            self.valid_range == (0..=1)
        } else {
            false
        }
    }

    /// Returns the valid range as a `x..y` range.
    ///
    /// If `x` and `y` are equal, the range is full, not empty.
    pub fn valid_range_exclusive<C: HasDataLayout>(&self, cx: C) -> Range<u128> {
        // For a (max) value of -1, max will be `-1 as usize`, which overflows.
        // However, that is fine here (it would still represent the full range),
        // i.e., if the range is everything.
        let bits = self.value.size(cx).bits();
        assert!(bits <= 128);
        let mask = !0u128 >> (128 - bits);
        let start = *self.valid_range.start();
        let end = *self.valid_range.end();
        assert_eq!(start, start & mask);
        assert_eq!(end, end & mask);
        start..(end.wrapping_add(1) & mask)
    }
}

/// Describes how the fields of a type are located in memory.
#[derive(PartialEq, Eq, Hash, Debug)]
pub enum FieldPlacement {
    /// All fields start at no offset. The `usize` is the field count.
    ///
    /// In the case of primitives the number of fields is `0`.
    Union(usize),

    /// Array/vector-like placement, with all fields of identical types.
    Array {
        stride: Size,
        count: u64
    },

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
        /// depending how fields were permuted.
        // FIXME(camlorn) also consider small vector  optimization here.
        memory_index: Vec<u32>
    }
}

impl FieldPlacement {
    pub fn count(&self) -> usize {
        match *self {
            FieldPlacement::Union(count) => count,
            FieldPlacement::Array { count, .. } => {
                let usize_count = count as usize;
                assert_eq!(usize_count as u64, count);
                usize_count
            }
            FieldPlacement::Arbitrary { ref offsets, .. } => offsets.len()
        }
    }

    pub fn offset(&self, i: usize) -> Size {
        match *self {
            FieldPlacement::Union(_) => Size::ZERO,
            FieldPlacement::Array { stride, count } => {
                let i = i as u64;
                assert!(i < count);
                stride * i
            }
            FieldPlacement::Arbitrary { ref offsets, .. } => offsets[i]
        }
    }

    pub fn memory_index(&self, i: usize) -> usize {
        match *self {
            FieldPlacement::Union(_) |
            FieldPlacement::Array { .. } => i,
            FieldPlacement::Arbitrary { ref memory_index, .. } => {
                let r = memory_index[i];
                assert_eq!(r as usize as u32, r);
                r as usize
            }
        }
    }

    /// Get source indices of the fields by increasing offsets.
    #[inline]
    pub fn index_by_increasing_offset<'a>(&'a self) -> impl Iterator<Item=usize>+'a {
        let mut inverse_small = [0u8; 64];
        let mut inverse_big = vec![];
        let use_small = self.count() <= inverse_small.len();

        // We have to write this logic twice in order to keep the array small.
        if let FieldPlacement::Arbitrary { ref memory_index, .. } = *self {
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

        (0..self.count()).map(move |i| {
            match *self {
                FieldPlacement::Union(_) |
                FieldPlacement::Array { .. } => i,
                FieldPlacement::Arbitrary { .. } => {
                    if use_small { inverse_small[i] as usize }
                    else { inverse_big[i] as usize }
                }
            }
        })
    }
}

/// Describes how values of the type are passed by target ABIs,
/// in terms of categories of C types there are ABI rules for.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Abi {
    Uninhabited,
    Scalar(Scalar),
    ScalarPair(Scalar, Scalar),
    Vector {
        element: Scalar,
        count: u64
    },
    Aggregate {
        /// If true, the size is exact, otherwise it's only a lower bound.
        sized: bool,
    }
}

impl Abi {
    /// Returns true if the layout corresponds to an unsized type.
    pub fn is_unsized(&self) -> bool {
        match *self {
            Abi::Uninhabited |
            Abi::Scalar(_) |
            Abi::ScalarPair(..) |
            Abi::Vector { .. } => false,
            Abi::Aggregate { sized } => !sized
        }
    }

    /// Returns true if this is a single signed integer scalar
    pub fn is_signed(&self) -> bool {
        match *self {
            Abi::Scalar(ref scal) => match scal.value {
                Primitive::Int(_, signed) => signed,
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub enum Variants {
    /// Single enum variants, structs/tuples, unions, and all non-ADTs.
    Single {
        index: usize
    },

    /// General-case enums: for each case there is a struct, and they all have
    /// all space reserved for the tag, and their first field starts
    /// at a non-0 offset, after where the tag would go.
    Tagged {
        tag: Scalar,
        variants: Vec<LayoutDetails>,
    },

    /// Multiple cases distinguished by a niche (values invalid for a type):
    /// the variant `dataful_variant` contains a niche at an arbitrary
    /// offset (field 0 of the enum), which for a variant with discriminant
    /// `d` is set to `(d - niche_variants.start).wrapping_add(niche_start)`.
    ///
    /// For example, `Option<(usize, &T)>`  is represented such that
    /// `None` has a null pointer for the second tuple field, and
    /// `Some` is the identity function (with a non-null reference).
    NicheFilling {
        dataful_variant: usize,
        niche_variants: RangeInclusive<usize>,
        niche: Scalar,
        niche_start: u128,
        variants: Vec<LayoutDetails>,
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct LayoutDetails {
    pub variants: Variants,
    pub fields: FieldPlacement,
    pub abi: Abi,
    pub align: Align,
    pub size: Size
}

impl LayoutDetails {
    pub fn scalar<C: HasDataLayout>(cx: C, scalar: Scalar) -> Self {
        let size = scalar.value.size(cx);
        let align = scalar.value.align(cx);
        LayoutDetails {
            variants: Variants::Single { index: 0 },
            fields: FieldPlacement::Union(0),
            abi: Abi::Scalar(scalar),
            size,
            align,
        }
    }
}

/// The details of the layout of a type, alongside the type itself.
/// Provides various type traversal APIs (e.g. recursing into fields).
///
/// Note that the details are NOT guaranteed to always be identical
/// to those obtained from `layout_of(ty)`, as we need to produce
/// layouts for which Rust types do not exist, such as enum variants
/// or synthetic fields of enums (i.e. discriminants) and fat pointers.
#[derive(Copy, Clone, Debug)]
pub struct TyLayout<'a, Ty> {
    pub ty: Ty,
    pub details: &'a LayoutDetails
}

impl<'a, Ty> Deref for TyLayout<'a, Ty> {
    type Target = &'a LayoutDetails;
    fn deref(&self) -> &&'a LayoutDetails {
        &self.details
    }
}

pub trait LayoutOf {
    type Ty;
    type TyLayout;

    fn layout_of(self, ty: Self::Ty) -> Self::TyLayout;
}

pub trait TyLayoutMethods<'a, C: LayoutOf<Ty = Self>>: Sized {
    fn for_variant(this: TyLayout<'a, Self>, cx: C, variant_index: usize) -> TyLayout<'a, Self>;
    fn field(this: TyLayout<'a, Self>, cx: C, i: usize) -> C::TyLayout;
}

impl<'a, Ty> TyLayout<'a, Ty> {
    pub fn for_variant<C>(self, cx: C, variant_index: usize) -> Self
    where Ty: TyLayoutMethods<'a, C>, C: LayoutOf<Ty = Ty> {
        Ty::for_variant(self, cx, variant_index)
    }
    pub fn field<C>(self, cx: C, i: usize) -> C::TyLayout
    where Ty: TyLayoutMethods<'a, C>, C: LayoutOf<Ty = Ty> {
        Ty::field(self, cx, i)
    }
}

impl<'a, Ty> TyLayout<'a, Ty> {
    /// Returns true if the layout corresponds to an unsized type.
    pub fn is_unsized(&self) -> bool {
        self.abi.is_unsized()
    }

    /// Returns true if the type is a ZST and not unsized.
    pub fn is_zst(&self) -> bool {
        match self.abi {
            Abi::Scalar(_) |
            Abi::ScalarPair(..) |
            Abi::Vector { .. } => false,
            Abi::Uninhabited => self.size.bytes() == 0,
            Abi::Aggregate { sized } => sized && self.size.bytes() == 0
        }
    }

    pub fn size_and_align(&self) -> (Size, Align) {
        (self.size, self.align)
    }
}
