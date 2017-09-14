// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Integer::*;
pub use self::Layout::*;
pub use self::Primitive::*;

use session::{self, DataTypeKind, Session};
use ty::{self, Ty, TyCtxt, TypeFoldable, ReprOptions, ReprFlags};

use syntax::ast::{self, FloatTy, IntTy, UintTy};
use syntax::attr;
use syntax_pos::DUMMY_SP;

use std::cmp;
use std::fmt;
use std::i64;
use std::iter;
use std::mem;
use std::ops::Deref;

use ich::StableHashingContext;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};

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
    pub fn parse(sess: &Session) -> TargetDataLayout {
        // Parse a bit count from a string.
        let parse_bits = |s: &str, kind: &str, cause: &str| {
            s.parse::<u64>().unwrap_or_else(|err| {
                sess.err(&format!("invalid {} `{}` for `{}` in \"data-layout\": {}",
                                  kind, s, cause, err));
                0
            })
        };

        // Parse a size string.
        let size = |s: &str, cause: &str| {
            Size::from_bits(parse_bits(s, "size", cause))
        };

        // Parse an alignment string.
        let align = |s: &[&str], cause: &str| {
            if s.is_empty() {
                sess.err(&format!("missing alignment for `{}` in \"data-layout\"", cause));
            }
            let abi = parse_bits(s[0], "alignment", cause);
            let pref = s.get(1).map_or(abi, |pref| parse_bits(pref, "alignment", cause));
            Align::from_bits(abi, pref).unwrap_or_else(|err| {
                sess.err(&format!("invalid alignment for `{}` in \"data-layout\": {}",
                                  cause, err));
                Align::from_bits(8, 8).unwrap()
            })
        };

        let mut dl = TargetDataLayout::default();
        let mut i128_align_src = 64;
        for spec in sess.target.target.data_layout.split("-") {
            match &spec.split(":").collect::<Vec<_>>()[..] {
                &["e"] => dl.endian = Endian::Little,
                &["E"] => dl.endian = Endian::Big,
                &["a", ref a..] => dl.aggregate_align = align(a, "a"),
                &["f32", ref a..] => dl.f32_align = align(a, "f32"),
                &["f64", ref a..] => dl.f64_align = align(a, "f64"),
                &[p @ "p", s, ref a..] | &[p @ "p0", s, ref a..] => {
                    dl.pointer_size = size(s, p);
                    dl.pointer_align = align(a, p);
                }
                &[s, ref a..] if s.starts_with("i") => {
                    let bits = match s[1..].parse::<u64>() {
                        Ok(bits) => bits,
                        Err(_) => {
                            size(&s[1..], "i"); // For the user error.
                            continue;
                        }
                    };
                    let a = align(a, s);
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
                &[s, ref a..] if s.starts_with("v") => {
                    let v_size = size(&s[1..], "v");
                    let a = align(a, s);
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
        if endian_str != sess.target.target.target_endian {
            sess.err(&format!("inconsistent target specification: \"data-layout\" claims \
                               architecture is {}-endian, while \"target-endian\" is `{}`",
                              endian_str, sess.target.target.target_endian));
        }

        if dl.pointer_size.bits().to_string() != sess.target.target.target_pointer_width {
            sess.err(&format!("inconsistent target specification: \"data-layout\" claims \
                               pointers are {}-bit, while \"target-pointer-width\" is `{}`",
                              dl.pointer_size.bits(), sess.target.target.target_pointer_width));
        }

        dl
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
            bits => bug!("obj_size_bound: unknown pointer bit size {}", bits)
        }
    }

    pub fn ptr_sized_integer(&self) -> Integer {
        match self.pointer_size.bits() {
            16 => I16,
            32 => I32,
            64 => I64,
            bits => bug!("ptr_sized_integer: unknown pointer bit size {}", bits)
        }
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

impl<'a, 'tcx> HasDataLayout for TyCtxt<'a, 'tcx, 'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        &self.data_layout
    }
}

/// Endianness of the target, which must match cfg(target-endian).
#[derive(Copy, Clone)]
pub enum Endian {
    Little,
    Big
}

/// Size of a type in bytes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Size {
    raw: u64
}

impl Size {
    pub fn from_bits(bits: u64) -> Size {
        Size::from_bytes((bits + 7) / 8)
    }

    pub fn from_bytes(bytes: u64) -> Size {
        if bytes >= (1 << 61) {
            bug!("Size::from_bytes: {} bytes in bits doesn't fit in u64", bytes)
        }
        Size {
            raw: bytes
        }
    }

    pub fn bytes(self) -> u64 {
        self.raw
    }

    pub fn bits(self) -> u64 {
        self.bytes() * 8
    }

    pub fn abi_align(self, align: Align) -> Size {
        let mask = align.abi() - 1;
        Size::from_bytes((self.bytes() + mask) & !mask)
    }

    pub fn checked_add<C: HasDataLayout>(self, offset: Size, cx: C) -> Option<Size> {
        let dl = cx.data_layout();

        // Each Size is less than dl.obj_size_bound(), so the sum is
        // also less than 1 << 62 (and therefore can't overflow).
        let bytes = self.bytes() + offset.bytes();

        if bytes < dl.obj_size_bound() {
            Some(Size::from_bytes(bytes))
        } else {
            None
        }
    }

    pub fn checked_mul<C: HasDataLayout>(self, count: u64, cx: C) -> Option<Size> {
        let dl = cx.data_layout();

        // Each Size is less than dl.obj_size_bound(), so the sum is
        // also less than 1 << 62 (and therefore can't overflow).
        match self.bytes().checked_mul(count) {
            Some(bytes) if bytes < dl.obj_size_bound() => {
                Some(Size::from_bytes(bytes))
            }
            _ => None
        }
    }
}

/// Alignment of a type in bytes, both ABI-mandated and preferred.
/// Each field is a power of two, giving the alignment a maximum
/// value of 2^(2^8 - 1), which is limited by LLVM to a i32, with
/// a maximum capacity of 2^31 - 1 or 2147483647.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Align {
    abi: u8,
    pref: u8,
}

impl Align {
    pub fn from_bits(abi: u64, pref: u64) -> Result<Align, String> {
        Align::from_bytes((abi + 7) / 8, (pref + 7) / 8)
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
            } else if pow > 30 {
                Err(format!("`{}` is too large", align))
            } else {
                Ok(pow)
            }
        };

        Ok(Align {
            abi: log2(abi)?,
            pref: log2(pref)?,
        })
    }

    pub fn abi(self) -> u64 {
        1 << self.abi
    }

    pub fn pref(self) -> u64 {
        1 << self.pref
    }

    pub fn min(self, other: Align) -> Align {
        Align {
            abi: cmp::min(self.abi, other.abi),
            pref: cmp::min(self.pref, other.pref),
        }
    }

    pub fn max(self, other: Align) -> Align {
        Align {
            abi: cmp::max(self.abi, other.abi),
            pref: cmp::max(self.pref, other.pref),
        }
    }
}

/// Integers, also used for enum discriminants.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Integer {
    I1,
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl Integer {
    pub fn size(&self) -> Size {
        match *self {
            I1 => Size::from_bits(1),
            I8 => Size::from_bytes(1),
            I16 => Size::from_bytes(2),
            I32 => Size::from_bytes(4),
            I64  => Size::from_bytes(8),
            I128  => Size::from_bytes(16),
        }
    }

    pub fn align<C: HasDataLayout>(&self, cx: C) -> Align {
        let dl = cx.data_layout();

        match *self {
            I1 => dl.i1_align,
            I8 => dl.i8_align,
            I16 => dl.i16_align,
            I32 => dl.i32_align,
            I64 => dl.i64_align,
            I128 => dl.i128_align,
        }
    }

    pub fn to_ty<'a, 'tcx>(&self, tcx: &TyCtxt<'a, 'tcx, 'tcx>,
                           signed: bool) -> Ty<'tcx> {
        match (*self, signed) {
            (I1, false) => tcx.types.u8,
            (I8, false) => tcx.types.u8,
            (I16, false) => tcx.types.u16,
            (I32, false) => tcx.types.u32,
            (I64, false) => tcx.types.u64,
            (I128, false) => tcx.types.u128,
            (I1, true) => tcx.types.i8,
            (I8, true) => tcx.types.i8,
            (I16, true) => tcx.types.i16,
            (I32, true) => tcx.types.i32,
            (I64, true) => tcx.types.i64,
            (I128, true) => tcx.types.i128,
        }
    }

    /// Find the smallest Integer type which can represent the signed value.
    pub fn fit_signed(x: i64) -> Integer {
        match x {
            -0x0000_0000_0000_0001...0x0000_0000_0000_0000 => I1,
            -0x0000_0000_0000_0080...0x0000_0000_0000_007f => I8,
            -0x0000_0000_0000_8000...0x0000_0000_0000_7fff => I16,
            -0x0000_0000_8000_0000...0x0000_0000_7fff_ffff => I32,
            -0x8000_0000_0000_0000...0x7fff_ffff_ffff_ffff => I64,
            _ => I128
        }
    }

    /// Find the smallest Integer type which can represent the unsigned value.
    pub fn fit_unsigned(x: u64) -> Integer {
        match x {
            0...0x0000_0000_0000_0001 => I1,
            0...0x0000_0000_0000_00ff => I8,
            0...0x0000_0000_0000_ffff => I16,
            0...0x0000_0000_ffff_ffff => I32,
            0...0xffff_ffff_ffff_ffff => I64,
            _ => I128,
        }
    }

    /// Find the smallest integer with the given alignment.
    pub fn for_abi_align<C: HasDataLayout>(cx: C, align: Align) -> Option<Integer> {
        let dl = cx.data_layout();

        let wanted = align.abi();
        for &candidate in &[I8, I16, I32, I64] {
            let ty = Int(candidate);
            if wanted == ty.align(dl).abi() && wanted == ty.size(dl).bytes() {
                return Some(candidate);
            }
        }
        None
    }

    /// Get the Integer type from an attr::IntType.
    pub fn from_attr<C: HasDataLayout>(cx: C, ity: attr::IntType) -> Integer {
        let dl = cx.data_layout();

        match ity {
            attr::SignedInt(IntTy::I8) | attr::UnsignedInt(UintTy::U8) => I8,
            attr::SignedInt(IntTy::I16) | attr::UnsignedInt(UintTy::U16) => I16,
            attr::SignedInt(IntTy::I32) | attr::UnsignedInt(UintTy::U32) => I32,
            attr::SignedInt(IntTy::I64) | attr::UnsignedInt(UintTy::U64) => I64,
            attr::SignedInt(IntTy::I128) | attr::UnsignedInt(UintTy::U128) => I128,
            attr::SignedInt(IntTy::Is) | attr::UnsignedInt(UintTy::Us) => {
                dl.ptr_sized_integer()
            }
        }
    }

    /// Find the appropriate Integer type and signedness for the given
    /// signed discriminant range and #[repr] attribute.
    /// N.B.: u64 values above i64::MAX will be treated as signed, but
    /// that shouldn't affect anything, other than maybe debuginfo.
    fn repr_discr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            ty: Ty<'tcx>,
                            repr: &ReprOptions,
                            min: i64,
                            max: i64)
                            -> (Integer, bool) {
        // Theoretically, negative values could be larger in unsigned representation
        // than the unsigned representation of the signed minimum. However, if there
        // are any negative values, the only valid unsigned representation is u64
        // which can fit all i64 values, so the result remains unaffected.
        let unsigned_fit = Integer::fit_unsigned(cmp::max(min as u64, max as u64));
        let signed_fit = cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

        let mut min_from_extern = None;
        let min_default = I8;

        if let Some(ity) = repr.int {
            let discr = Integer::from_attr(tcx, ity);
            let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
            if discr < fit {
                bug!("Integer::repr_discr: `#[repr]` hint too small for \
                  discriminant range of enum `{}", ty)
            }
            return (discr, ity.is_signed());
        }

        if repr.c() {
            match &tcx.sess.target.target.arch[..] {
                // WARNING: the ARM EABI has two variants; the one corresponding
                // to `at_least == I32` appears to be used on Linux and NetBSD,
                // but some systems may use the variant corresponding to no
                // lower bound.  However, we don't run on those yet...?
                "arm" => min_from_extern = Some(I32),
                _ => min_from_extern = Some(I32),
            }
        }

        let at_least = min_from_extern.unwrap_or(min_default);

        // If there are no negative values, we can use the unsigned fit.
        if min >= 0 {
            (cmp::max(unsigned_fit, at_least), false)
        } else {
            (cmp::max(signed_fit, at_least), true)
        }
    }
}

/// Fundamental unit of memory access and layout.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Primitive {
    Int(Integer),
    F32,
    F64,
    Pointer
}

impl Primitive {
    pub fn size<C: HasDataLayout>(self, cx: C) -> Size {
        let dl = cx.data_layout();

        match self {
            Int(I1) | Int(I8) => Size::from_bits(8),
            Int(I16) => Size::from_bits(16),
            Int(I32) | F32 => Size::from_bits(32),
            Int(I64) | F64 => Size::from_bits(64),
            Int(I128) => Size::from_bits(128),
            Pointer => dl.pointer_size
        }
    }

    pub fn align<C: HasDataLayout>(self, cx: C) -> Align {
        let dl = cx.data_layout();

        match self {
            Int(I1) => dl.i1_align,
            Int(I8) => dl.i8_align,
            Int(I16) => dl.i16_align,
            Int(I32) => dl.i32_align,
            Int(I64) => dl.i64_align,
            Int(I128) => dl.i128_align,
            F32 => dl.f32_align,
            F64 => dl.f64_align,
            Pointer => dl.pointer_align
        }
    }
}

/// Path through fields of nested structures.
// FIXME(eddyb) use small vector optimization for the common case.
pub type FieldPath = Vec<u32>;

/// A structure, a product type in ADT terms.
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Struct {
    /// Maximum alignment of fields and repr alignment.
    pub align: Align,

    /// Primitive alignment of fields without repr alignment.
    pub primitive_align: Align,

    /// If true, no alignment padding is used.
    pub packed: bool,

    /// If true, the size is exact, otherwise it's only a lower bound.
    pub sized: bool,

    /// Offsets for the first byte of each field, ordered to match the source definition order.
    /// This vector does not go in increasing order.
    /// FIXME(eddyb) use small vector optimization for the common case.
    pub offsets: Vec<Size>,

    /// Maps source order field indices to memory order indices, depending how fields were permuted.
    /// FIXME (camlorn) also consider small vector  optimization here.
    pub memory_index: Vec<u32>,

    pub min_size: Size,
}

/// Info required to optimize struct layout.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
enum StructKind {
    /// A tuple, closure, or univariant which cannot be coerced to unsized.
    AlwaysSizedUnivariant,
    /// A univariant, the last field of which may be coerced to unsized.
    MaybeUnsizedUnivariant,
    /// A univariant, but part of an enum.
    EnumVariant,
}

impl<'a, 'tcx> Struct {
    fn new(dl: &TargetDataLayout,
           fields: &Vec<&'a Layout>,
           repr: &ReprOptions,
           kind: StructKind,
           scapegoat: Ty<'tcx>)
           -> Result<Struct, LayoutError<'tcx>> {
        if repr.packed() && repr.align > 0 {
            bug!("Struct cannot be packed and aligned");
        }

        let align = if repr.packed() {
            dl.i8_align
        } else {
            dl.aggregate_align
        };

        let mut ret = Struct {
            align,
            primitive_align: align,
            packed: repr.packed(),
            sized: true,
            offsets: vec![],
            memory_index: vec![],
            min_size: Size::from_bytes(0),
        };

        // Anything with repr(C) or repr(packed) doesn't optimize.
        // Neither do  1-member and 2-member structs.
        // In addition, code in trans assume that 2-element structs can become pairs.
        // It's easier to just short-circuit here.
        let can_optimize = (fields.len() > 2 || StructKind::EnumVariant == kind)
            && (repr.flags & ReprFlags::IS_UNOPTIMISABLE).is_empty();

        let (optimize, sort_ascending) = match kind {
            StructKind::AlwaysSizedUnivariant => (can_optimize, false),
            StructKind::MaybeUnsizedUnivariant => (can_optimize, false),
            StructKind::EnumVariant => {
                assert!(fields.len() >= 1, "Enum variants must have discriminants.");
                (can_optimize && fields[0].size(dl).bytes() == 1, true)
            }
        };

        ret.offsets = vec![Size::from_bytes(0); fields.len()];
        let mut inverse_memory_index: Vec<u32> = (0..fields.len() as u32).collect();

        if optimize {
            let start = if let StructKind::EnumVariant = kind { 1 } else { 0 };
            let end = if let StructKind::MaybeUnsizedUnivariant = kind {
                fields.len() - 1
            } else {
                fields.len()
            };
            if end > start {
                let optimizing  = &mut inverse_memory_index[start..end];
                if sort_ascending {
                    optimizing.sort_by_key(|&x| fields[x as usize].align(dl).abi());
                } else {
                    optimizing.sort_by(| &a, &b | {
                        let a = fields[a as usize].align(dl).abi();
                        let b = fields[b as usize].align(dl).abi();
                        b.cmp(&a)
                    });
                }
            }
        }

        // inverse_memory_index holds field indices by increasing memory offset.
        // That is, if field 5 has offset 0, the first element of inverse_memory_index is 5.
        // We now write field offsets to the corresponding offset slot;
        // field 5 with offset 0 puts 0 in offsets[5].
        // At the bottom of this function, we use inverse_memory_index to produce memory_index.

        if let StructKind::EnumVariant = kind {
            assert_eq!(inverse_memory_index[0], 0,
              "Enum variant discriminants must have the lowest offset.");
        }

        let mut offset = Size::from_bytes(0);

        for i in inverse_memory_index.iter() {
            let field = fields[*i as usize];
            if !ret.sized {
                bug!("Struct::new: field #{} of `{}` comes after unsized field",
                     ret.offsets.len(), scapegoat);
            }

            if field.is_unsized() {
                ret.sized = false;
            }

            // Invariant: offset < dl.obj_size_bound() <= 1<<61
            if !ret.packed {
                let align = field.align(dl);
                let primitive_align = field.primitive_align(dl);
                ret.align = ret.align.max(align);
                ret.primitive_align = ret.primitive_align.max(primitive_align);
                offset = offset.abi_align(align);
            }

            debug!("Struct::new offset: {:?} field: {:?} {:?}", offset, field, field.size(dl));
            ret.offsets[*i as usize] = offset;

            offset = offset.checked_add(field.size(dl), dl)
                           .map_or(Err(LayoutError::SizeOverflow(scapegoat)), Ok)?;
        }

        if repr.align > 0 {
            let repr_align = repr.align as u64;
            ret.align = ret.align.max(Align::from_bytes(repr_align, repr_align).unwrap());
            debug!("Struct::new repr_align: {:?}", repr_align);
        }

        debug!("Struct::new min_size: {:?}", offset);
        ret.min_size = offset;

        // As stated above, inverse_memory_index holds field indices by increasing offset.
        // This makes it an already-sorted view of the offsets vec.
        // To invert it, consider:
        // If field 5 has offset 0, offsets[0] is 5, and memory_index[5] should be 0.
        // Field 5 would be the first element, so memory_index is i:
        // Note: if we didn't optimize, it's already right.

        if optimize {
            ret.memory_index = vec![0; inverse_memory_index.len()];

            for i in 0..inverse_memory_index.len() {
                ret.memory_index[inverse_memory_index[i] as usize]  = i as u32;
            }
        } else {
            ret.memory_index = inverse_memory_index;
        }

        Ok(ret)
    }

    /// Get the size with trailing alignment padding.
    pub fn stride(&self) -> Size {
        self.min_size.abi_align(self.align)
    }

    /// Determine whether a structure would be zero-sized, given its fields.
    fn would_be_zero_sized<I>(dl: &TargetDataLayout, fields: I)
                              -> Result<bool, LayoutError<'tcx>>
    where I: Iterator<Item=Result<&'a Layout, LayoutError<'tcx>>> {
        for field in fields {
            let field = field?;
            if field.is_unsized() || field.size(dl).bytes() > 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Get indices of the tys that made this struct by increasing offset.
    #[inline]
    pub fn field_index_by_increasing_offset<'b>(&'b self) -> impl iter::Iterator<Item=usize>+'b {
        let mut inverse_small = [0u8; 64];
        let mut inverse_big = vec![];
        let use_small = self.memory_index.len() <= inverse_small.len();

        // We have to write this logic twice in order to keep the array small.
        if use_small {
            for i in 0..self.memory_index.len() {
                inverse_small[self.memory_index[i] as usize] = i as u8;
            }
        } else {
            inverse_big = vec![0; self.memory_index.len()];
            for i in 0..self.memory_index.len() {
                inverse_big[self.memory_index[i] as usize] = i as u32;
            }
        }

        (0..self.memory_index.len()).map(move |i| {
            if use_small { inverse_small[i] as usize }
            else { inverse_big[i] as usize }
        })
    }

    /// Find the path leading to a non-zero leaf field, starting from
    /// the given type and recursing through aggregates.
    /// The tuple is `(path, source_path)`,
    /// where `path` is in memory order and `source_path` in source order.
    // FIXME(eddyb) track value ranges and traverse already optimized enums.
    fn non_zero_field_in_type(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              param_env: ty::ParamEnv<'tcx>,
                              ty: Ty<'tcx>)
                              -> Result<Option<(FieldPath, FieldPath)>, LayoutError<'tcx>> {
        match (ty.layout(tcx, param_env)?, &ty.sty) {
            (&Scalar { non_zero: true, .. }, _) |
            (&CEnum { non_zero: true, .. }, _) => Ok(Some((vec![], vec![]))),
            (&FatPointer { non_zero: true, .. }, _) => {
                Ok(Some((vec![FAT_PTR_ADDR as u32], vec![FAT_PTR_ADDR as u32])))
            }

            // Is this the NonZero lang item wrapping a pointer or integer type?
            (&Univariant { non_zero: true, .. }, &ty::TyAdt(def, substs)) => {
                let fields = &def.struct_variant().fields;
                assert_eq!(fields.len(), 1);
                match *fields[0].ty(tcx, substs).layout(tcx, param_env)? {
                    // FIXME(eddyb) also allow floating-point types here.
                    Scalar { value: Int(_), non_zero: false } |
                    Scalar { value: Pointer, non_zero: false } => {
                        Ok(Some((vec![0], vec![0])))
                    }
                    FatPointer { non_zero: false, .. } => {
                        let tmp = vec![FAT_PTR_ADDR as u32, 0];
                        Ok(Some((tmp.clone(), tmp)))
                    }
                    _ => Ok(None)
                }
            }

            // Perhaps one of the fields of this struct is non-zero
            // let's recurse and find out
            (&Univariant { ref variant, .. }, &ty::TyAdt(def, substs)) if def.is_struct() => {
                Struct::non_zero_field_paths(
                    tcx,
                    param_env,
                    def.struct_variant().fields.iter().map(|field| {
                        field.ty(tcx, substs)
                    }),
                    Some(&variant.memory_index[..]))
            }

            // Perhaps one of the upvars of this closure is non-zero
            (&Univariant { ref variant, .. }, &ty::TyClosure(def, substs)) => {
                let upvar_tys = substs.upvar_tys(def, tcx);
                Struct::non_zero_field_paths(
                    tcx,
                    param_env,
                    upvar_tys,
                    Some(&variant.memory_index[..]))
            }
            // Can we use one of the fields in this tuple?
            (&Univariant { ref variant, .. }, &ty::TyTuple(tys, _)) => {
                Struct::non_zero_field_paths(
                    tcx,
                    param_env,
                    tys.iter().cloned(),
                    Some(&variant.memory_index[..]))
            }

            // Is this a fixed-size array of something non-zero
            // with at least one element?
            (_, &ty::TyArray(ety, mut count)) => {
                if count.has_projections() {
                    count = tcx.normalize_associated_type_in_env(&count, param_env);
                    if count.has_projections() {
                        return Err(LayoutError::Unknown(ty));
                    }
                }
                if count.val.to_const_int().unwrap().to_u64().unwrap() != 0 {
                    Struct::non_zero_field_paths(
                        tcx,
                        param_env,
                        Some(ety).into_iter(),
                        None)
                } else {
                    Ok(None)
                }
            }

            (_, &ty::TyProjection(_)) | (_, &ty::TyAnon(..)) => {
                let normalized = tcx.normalize_associated_type_in_env(&ty, param_env);
                if ty == normalized {
                    return Ok(None);
                }
                return Struct::non_zero_field_in_type(tcx, param_env, normalized);
            }

            // Anything else is not a non-zero type.
            _ => Ok(None)
        }
    }

    /// Find the path leading to a non-zero leaf field, starting from
    /// the given set of fields and recursing through aggregates.
    /// Returns Some((path, source_path)) on success.
    /// `path` is translated to memory order. `source_path` is not.
    fn non_zero_field_paths<I>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                               param_env: ty::ParamEnv<'tcx>,
                               fields: I,
                               permutation: Option<&[u32]>)
                               -> Result<Option<(FieldPath, FieldPath)>, LayoutError<'tcx>>
    where I: Iterator<Item=Ty<'tcx>> {
        for (i, ty) in fields.enumerate() {
            let r = Struct::non_zero_field_in_type(tcx, param_env, ty)?;
            if let Some((mut path, mut source_path)) = r {
                source_path.push(i as u32);
                let index = if let Some(p) = permutation {
                    p[i] as usize
                } else {
                    i
                };
                path.push(index as u32);
                return Ok(Some((path, source_path)));
            }
        }
        Ok(None)
    }

    pub fn over_align(&self) -> Option<u32> {
        let align = self.align.abi();
        let primitive_align = self.primitive_align.abi();
        if align > primitive_align {
            Some(align as u32)
        } else {
            None
        }
    }
}

/// An untagged union.
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Union {
    pub align: Align,
    pub primitive_align: Align,

    pub min_size: Size,

    /// If true, no alignment padding is used.
    pub packed: bool,
}

impl<'a, 'tcx> Union {
    fn new(dl: &TargetDataLayout, repr: &ReprOptions) -> Union {
        if repr.packed() && repr.align > 0 {
            bug!("Union cannot be packed and aligned");
        }

        let primitive_align = if repr.packed() {
            dl.i8_align
        } else {
            dl.aggregate_align
        };

        let align = if repr.align > 0 {
            let repr_align = repr.align as u64;
            debug!("Union::new repr_align: {:?}", repr_align);
            primitive_align.max(Align::from_bytes(repr_align, repr_align).unwrap())
        } else {
            primitive_align
        };

        Union {
            align,
            primitive_align,
            min_size: Size::from_bytes(0),
            packed: repr.packed(),
        }
    }

    /// Extend the Struct with more fields.
    fn extend<I>(&mut self, dl: &TargetDataLayout,
                 fields: I,
                 scapegoat: Ty<'tcx>)
                 -> Result<(), LayoutError<'tcx>>
    where I: Iterator<Item=Result<&'a Layout, LayoutError<'tcx>>> {
        for (index, field) in fields.enumerate() {
            let field = field?;
            if field.is_unsized() {
                bug!("Union::extend: field #{} of `{}` is unsized",
                     index, scapegoat);
            }

            debug!("Union::extend field: {:?} {:?}", field, field.size(dl));

            if !self.packed {
                self.align = self.align.max(field.align(dl));
                self.primitive_align = self.primitive_align.max(field.primitive_align(dl));
            }
            self.min_size = cmp::max(self.min_size, field.size(dl));
        }

        debug!("Union::extend min-size: {:?}", self.min_size);

        Ok(())
    }

    /// Get the size with trailing alignment padding.
    pub fn stride(&self) -> Size {
        self.min_size.abi_align(self.align)
    }

    pub fn over_align(&self) -> Option<u32> {
        let align = self.align.abi();
        let primitive_align = self.primitive_align.abi();
        if align > primitive_align {
            Some(align as u32)
        } else {
            None
        }
    }
}

/// The first half of a fat pointer.
/// - For a trait object, this is the address of the box.
/// - For a slice, this is the base address.
pub const FAT_PTR_ADDR: usize = 0;

/// The second half of a fat pointer.
/// - For a trait object, this is the address of the vtable.
/// - For a slice, this is the length.
pub const FAT_PTR_EXTRA: usize = 1;

/// Type layout, from which size and alignment can be cheaply computed.
/// For ADTs, it also includes field placement and enum optimizations.
/// NOTE: Because Layout is interned, redundant information should be
/// kept to a minimum, e.g. it includes no sub-component Ty or Layout.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// TyBool, TyChar, TyInt, TyUint, TyFloat, TyRawPtr, TyRef or TyFnPtr.
    Scalar {
        value: Primitive,
        // If true, the value cannot represent a bit pattern of all zeroes.
        non_zero: bool
    },

    /// SIMD vectors, from structs marked with #[repr(simd)].
    Vector {
        element: Primitive,
        count: u64
    },

    /// TyArray, TySlice or TyStr.
    Array {
        /// If true, the size is exact, otherwise it's only a lower bound.
        sized: bool,
        align: Align,
        primitive_align: Align,
        element_size: Size,
        count: u64
    },

    /// TyRawPtr or TyRef with a !Sized pointee.
    FatPointer {
        metadata: Primitive,
        /// If true, the pointer cannot be null.
        non_zero: bool
    },

    // Remaining variants are all ADTs such as structs, enums or tuples.

    /// C-like enums; basically an integer.
    CEnum {
        discr: Integer,
        signed: bool,
        non_zero: bool,
        /// Inclusive discriminant range.
        /// If min > max, it represents min...u64::MAX followed by 0...max.
        // FIXME(eddyb) always use the shortest range, e.g. by finding
        // the largest space between two consecutive discriminants and
        // taking everything else as the (shortest) discriminant range.
        min: u64,
        max: u64
    },

    /// Single-case enums, and structs/tuples.
    Univariant {
        variant: Struct,
        /// If true, the structure is NonZero.
        // FIXME(eddyb) use a newtype Layout kind for this.
        non_zero: bool
    },

    /// Untagged unions.
    UntaggedUnion {
        variants: Union,
    },

    /// General-case enums: for each case there is a struct, and they
    /// all start with a field for the discriminant.
    General {
        discr: Integer,
        variants: Vec<Struct>,
        size: Size,
        align: Align,
        primitive_align: Align,
    },

    /// Two cases distinguished by a nullable pointer: the case with discriminant
    /// `nndiscr` must have single field which is known to be nonnull due to its type.
    /// The other case is known to be zero sized. Hence we represent the enum
    /// as simply a nullable pointer: if not null it indicates the `nndiscr` variant,
    /// otherwise it indicates the other case.
    ///
    /// For example, `std::option::Option` instantiated at a safe pointer type
    /// is represented such that `None` is a null pointer and `Some` is the
    /// identity function.
    RawNullablePointer {
        nndiscr: u64,
        value: Primitive
    },

    /// Two cases distinguished by a nullable pointer: the case with discriminant
    /// `nndiscr` is represented by the struct `nonnull`, where the `discrfield`th
    /// field is known to be nonnull due to its type; if that field is null, then
    /// it represents the other case, which is known to be zero sized.
    StructWrappedNullablePointer {
        nndiscr: u64,
        nonnull: Struct,
        /// N.B. There is a 0 at the start, for LLVM GEP through a pointer.
        discrfield: FieldPath,
        /// Like discrfield, but in source order. For debuginfo.
        discrfield_source: FieldPath
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LayoutError<'tcx> {
    Unknown(Ty<'tcx>),
    SizeOverflow(Ty<'tcx>)
}

impl<'tcx> fmt::Display for LayoutError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            LayoutError::Unknown(ty) => {
                write!(f, "the type `{:?}` has an unknown layout", ty)
            }
            LayoutError::SizeOverflow(ty) => {
                write!(f, "the type `{:?}` is too big for the current architecture", ty)
            }
        }
    }
}

impl<'a, 'tcx> Layout {
    pub fn compute_uncached(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            param_env: ty::ParamEnv<'tcx>,
                            ty: Ty<'tcx>)
                            -> Result<&'tcx Layout, LayoutError<'tcx>> {
        let success = |layout| Ok(tcx.intern_layout(layout));
        let dl = &tcx.data_layout;
        assert!(!ty.has_infer_types());

        let ptr_layout = |pointee: Ty<'tcx>| {
            let non_zero = !ty.is_unsafe_ptr();
            let pointee = tcx.normalize_associated_type_in_env(&pointee, param_env);
            if pointee.is_sized(tcx, param_env, DUMMY_SP) {
                Ok(Scalar { value: Pointer, non_zero: non_zero })
            } else {
                let unsized_part = tcx.struct_tail(pointee);
                let meta = match unsized_part.sty {
                    ty::TySlice(_) | ty::TyStr => {
                        Int(dl.ptr_sized_integer())
                    }
                    ty::TyDynamic(..) => Pointer,
                    _ => return Err(LayoutError::Unknown(unsized_part))
                };
                Ok(FatPointer { metadata: meta, non_zero: non_zero })
            }
        };

        let layout = match ty.sty {
            // Basic scalars.
            ty::TyBool => Scalar { value: Int(I1), non_zero: false },
            ty::TyChar => Scalar { value: Int(I32), non_zero: false },
            ty::TyInt(ity) => {
                Scalar {
                    value: Int(Integer::from_attr(dl, attr::SignedInt(ity))),
                    non_zero: false
                }
            }
            ty::TyUint(ity) => {
                Scalar {
                    value: Int(Integer::from_attr(dl, attr::UnsignedInt(ity))),
                    non_zero: false
                }
            }
            ty::TyFloat(FloatTy::F32) => Scalar { value: F32, non_zero: false },
            ty::TyFloat(FloatTy::F64) => Scalar { value: F64, non_zero: false },
            ty::TyFnPtr(_) => Scalar { value: Pointer, non_zero: true },

            // The never type.
            ty::TyNever => Univariant {
                variant: Struct::new(dl, &vec![], &ReprOptions::default(),
                  StructKind::AlwaysSizedUnivariant, ty)?,
                non_zero: false
            },

            // Potentially-fat pointers.
            ty::TyRef(_, ty::TypeAndMut { ty: pointee, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                ptr_layout(pointee)?
            }
            ty::TyAdt(def, _) if def.is_box() => {
                ptr_layout(ty.boxed_ty())?
            }

            // Arrays and slices.
            ty::TyArray(element, mut count) => {
                if count.has_projections() {
                    count = tcx.normalize_associated_type_in_env(&count, param_env);
                    if count.has_projections() {
                        return Err(LayoutError::Unknown(ty));
                    }
                }

                let element = element.layout(tcx, param_env)?;
                let element_size = element.size(dl);
                let count = count.val.to_const_int().unwrap().to_u64().unwrap();
                if element_size.checked_mul(count, dl).is_none() {
                    return Err(LayoutError::SizeOverflow(ty));
                }
                Array {
                    sized: true,
                    align: element.align(dl),
                    primitive_align: element.primitive_align(dl),
                    element_size,
                    count,
                }
            }
            ty::TySlice(element) => {
                let element = element.layout(tcx, param_env)?;
                Array {
                    sized: false,
                    align: element.align(dl),
                    primitive_align: element.primitive_align(dl),
                    element_size: element.size(dl),
                    count: 0
                }
            }
            ty::TyStr => {
                Array {
                    sized: false,
                    align: dl.i8_align,
                    primitive_align: dl.i8_align,
                    element_size: Size::from_bytes(1),
                    count: 0
                }
            }

            // Odd unit types.
            ty::TyFnDef(..) => {
                Univariant {
                    variant: Struct::new(dl, &vec![],
                      &ReprOptions::default(), StructKind::AlwaysSizedUnivariant, ty)?,
                    non_zero: false
                }
            }
            ty::TyDynamic(..) => {
                let mut unit = Struct::new(dl, &vec![], &ReprOptions::default(),
                  StructKind::AlwaysSizedUnivariant, ty)?;
                unit.sized = false;
                Univariant { variant: unit, non_zero: false }
            }

            // Tuples, generators and closures.
            ty::TyGenerator(def_id, ref substs, _) => {
                let tys = substs.field_tys(def_id, tcx);
                let st = Struct::new(dl,
                    &tys.map(|ty| ty.layout(tcx, param_env))
                      .collect::<Result<Vec<_>, _>>()?,
                    &ReprOptions::default(),
                    StructKind::AlwaysSizedUnivariant, ty)?;
                Univariant { variant: st, non_zero: false }
            }

            ty::TyClosure(def_id, ref substs) => {
                let tys = substs.upvar_tys(def_id, tcx);
                let st = Struct::new(dl,
                    &tys.map(|ty| ty.layout(tcx, param_env))
                      .collect::<Result<Vec<_>, _>>()?,
                    &ReprOptions::default(),
                    StructKind::AlwaysSizedUnivariant, ty)?;
                Univariant { variant: st, non_zero: false }
            }

            ty::TyTuple(tys, _) => {
                let kind = if tys.len() == 0 {
                    StructKind::AlwaysSizedUnivariant
                } else {
                    StructKind::MaybeUnsizedUnivariant
                };

                let st = Struct::new(dl,
                    &tys.iter().map(|ty| ty.layout(tcx, param_env))
                      .collect::<Result<Vec<_>, _>>()?,
                    &ReprOptions::default(), kind, ty)?;
                Univariant { variant: st, non_zero: false }
            }

            // SIMD vector types.
            ty::TyAdt(def, ..) if def.repr.simd() => {
                let element = ty.simd_type(tcx);
                match *element.layout(tcx, param_env)? {
                    Scalar { value, .. } => {
                        return success(Vector {
                            element: value,
                            count: ty.simd_size(tcx) as u64
                        });
                    }
                    _ => {
                        tcx.sess.fatal(&format!("monomorphising SIMD type `{}` with \
                                                a non-machine element type `{}`",
                                                ty, element));
                    }
                }
            }

            // ADTs.
            ty::TyAdt(def, substs) => {
                if def.variants.is_empty() {
                    // Uninhabitable; represent as unit
                    // (Typechecking will reject discriminant-sizing attrs.)

                    return success(Univariant {
                        variant: Struct::new(dl, &vec![],
                          &def.repr, StructKind::AlwaysSizedUnivariant, ty)?,
                        non_zero: false
                    });
                }

                if def.is_enum() && def.variants.iter().all(|v| v.fields.is_empty()) {
                    // All bodies empty -> intlike
                    let (mut min, mut max, mut non_zero) = (i64::max_value(),
                                                            i64::min_value(),
                                                            true);
                    for discr in def.discriminants(tcx) {
                        let x = discr.to_u128_unchecked() as i64;
                        if x == 0 { non_zero = false; }
                        if x < min { min = x; }
                        if x > max { max = x; }
                    }

                    // FIXME: should handle i128? signed-value based impl is weird and hard to
                    // grok.
                    let (discr, signed) = Integer::repr_discr(tcx, ty, &def.repr, min, max);
                    return success(CEnum {
                        discr,
                        signed,
                        non_zero,
                        // FIXME: should be u128?
                        min: min as u64,
                        max: max as u64
                    });
                }

                if !def.is_enum() || (def.variants.len() == 1 &&
                                      !def.repr.inhibit_enum_layout_opt()) {
                    // Struct, or union, or univariant enum equivalent to a struct.
                    // (Typechecking will reject discriminant-sizing attrs.)

                    let kind = if def.is_enum() || def.variants[0].fields.len() == 0{
                        StructKind::AlwaysSizedUnivariant
                    } else {
                        let param_env = tcx.param_env(def.did);
                        let fields = &def.variants[0].fields;
                        let last_field = &fields[fields.len()-1];
                        let always_sized = tcx.type_of(last_field.did)
                          .is_sized(tcx, param_env, DUMMY_SP);
                        if !always_sized { StructKind::MaybeUnsizedUnivariant }
                        else { StructKind::AlwaysSizedUnivariant }
                    };

                    let fields = def.variants[0].fields.iter().map(|field| {
                        field.ty(tcx, substs).layout(tcx, param_env)
                    }).collect::<Result<Vec<_>, _>>()?;
                    let layout = if def.is_union() {
                        let mut un = Union::new(dl, &def.repr);
                        un.extend(dl, fields.iter().map(|&f| Ok(f)), ty)?;
                        UntaggedUnion { variants: un }
                    } else {
                        let st = Struct::new(dl, &fields, &def.repr,
                          kind, ty)?;
                        let non_zero = Some(def.did) == tcx.lang_items().non_zero();
                        Univariant { variant: st, non_zero: non_zero }
                    };
                    return success(layout);
                }

                // Since there's at least one
                // non-empty body, explicit discriminants should have
                // been rejected by a checker before this point.
                for (i, v) in def.variants.iter().enumerate() {
                    if v.discr != ty::VariantDiscr::Relative(i) {
                        bug!("non-C-like enum {} with specified discriminants",
                            tcx.item_path_str(def.did));
                    }
                }

                // Cache the substituted and normalized variant field types.
                let variants = def.variants.iter().map(|v| {
                    v.fields.iter().map(|field| field.ty(tcx, substs)).collect::<Vec<_>>()
                }).collect::<Vec<_>>();

                if variants.len() == 2 && !def.repr.inhibit_enum_layout_opt() {
                    // Nullable pointer optimization
                    for discr in 0..2 {
                        let other_fields = variants[1 - discr].iter().map(|ty| {
                            ty.layout(tcx, param_env)
                        });
                        if !Struct::would_be_zero_sized(dl, other_fields)? {
                            continue;
                        }
                        let paths = Struct::non_zero_field_paths(tcx,
                                                                 param_env,
                                                                 variants[discr].iter().cloned(),
                                                                 None)?;
                        let (mut path, mut path_source) = if let Some(p) = paths { p }
                          else { continue };

                        // FIXME(eddyb) should take advantage of a newtype.
                        if path == &[0] && variants[discr].len() == 1 {
                            let value = match *variants[discr][0].layout(tcx, param_env)? {
                                Scalar { value, .. } => value,
                                CEnum { discr, .. } => Int(discr),
                                _ => bug!("Layout::compute: `{}`'s non-zero \
                                           `{}` field not scalar?!",
                                           ty, variants[discr][0])
                            };
                            return success(RawNullablePointer {
                                nndiscr: discr as u64,
                                value,
                            });
                        }

                        let st = Struct::new(dl,
                            &variants[discr].iter().map(|ty| ty.layout(tcx, param_env))
                              .collect::<Result<Vec<_>, _>>()?,
                            &def.repr, StructKind::AlwaysSizedUnivariant, ty)?;

                        // We have to fix the last element of path here.
                        let mut i = *path.last().unwrap();
                        i = st.memory_index[i as usize];
                        *path.last_mut().unwrap() = i;
                        path.push(0); // For GEP through a pointer.
                        path.reverse();
                        path_source.push(0);
                        path_source.reverse();

                        return success(StructWrappedNullablePointer {
                            nndiscr: discr as u64,
                            nonnull: st,
                            discrfield: path,
                            discrfield_source: path_source
                        });
                    }
                }

                // The general case.
                let discr_max = (variants.len() - 1) as i64;
                assert!(discr_max >= 0);
                let (min_ity, _) = Integer::repr_discr(tcx, ty, &def.repr, 0, discr_max);
                let mut align = dl.aggregate_align;
                let mut primitive_align = dl.aggregate_align;
                let mut size = Size::from_bytes(0);

                // We're interested in the smallest alignment, so start large.
                let mut start_align = Align::from_bytes(256, 256).unwrap();

                // Create the set of structs that represent each variant
                // Use the minimum integer type we figured out above
                let discr = Scalar { value: Int(min_ity), non_zero: false };
                let mut variants = variants.into_iter().map(|fields| {
                    let mut fields = fields.into_iter().map(|field| {
                        field.layout(tcx, param_env)
                    }).collect::<Result<Vec<_>, _>>()?;
                    fields.insert(0, &discr);
                    let st = Struct::new(dl,
                        &fields,
                        &def.repr, StructKind::EnumVariant, ty)?;
                    // Find the first field we can't move later
                    // to make room for a larger discriminant.
                    // It is important to skip the first field.
                    for i in st.field_index_by_increasing_offset().skip(1) {
                        let field = fields[i];
                        let field_align = field.align(dl);
                        if field.size(dl).bytes() != 0 || field_align.abi() != 1 {
                            start_align = start_align.min(field_align);
                            break;
                        }
                    }
                    size = cmp::max(size, st.min_size);
                    align = align.max(st.align);
                    primitive_align = primitive_align.max(st.primitive_align);
                    Ok(st)
                }).collect::<Result<Vec<_>, _>>()?;

                // Align the maximum variant size to the largest alignment.
                size = size.abi_align(align);

                if size.bytes() >= dl.obj_size_bound() {
                    return Err(LayoutError::SizeOverflow(ty));
                }

                let typeck_ity = Integer::from_attr(dl, def.repr.discr_type());
                if typeck_ity < min_ity {
                    // It is a bug if Layout decided on a greater discriminant size than typeck for
                    // some reason at this point (based on values discriminant can take on). Mostly
                    // because this discriminant will be loaded, and then stored into variable of
                    // type calculated by typeck. Consider such case (a bug): typeck decided on
                    // byte-sized discriminant, but layout thinks we need a 16-bit to store all
                    // discriminant values. That would be a bug, because then, in trans, in order
                    // to store this 16-bit discriminant into 8-bit sized temporary some of the
                    // space necessary to represent would have to be discarded (or layout is wrong
                    // on thinking it needs 16 bits)
                    bug!("layout decided on a larger discriminant type ({:?}) than typeck ({:?})",
                         min_ity, typeck_ity);
                    // However, it is fine to make discr type however large (as an optimisation)
                    // after this point – we’ll just truncate the value we load in trans.
                }

                // Check to see if we should use a different type for the
                // discriminant. We can safely use a type with the same size
                // as the alignment of the first field of each variant.
                // We increase the size of the discriminant to avoid LLVM copying
                // padding when it doesn't need to. This normally causes unaligned
                // load/stores and excessive memcpy/memset operations. By using a
                // bigger integer size, LLVM can be sure about it's contents and
                // won't be so conservative.

                // Use the initial field alignment
                let mut ity = Integer::for_abi_align(dl, start_align).unwrap_or(min_ity);

                // If the alignment is not larger than the chosen discriminant size,
                // don't use the alignment as the final size.
                if ity <= min_ity {
                    ity = min_ity;
                } else {
                    // Patch up the variants' first few fields.
                    let old_ity_size = Int(min_ity).size(dl);
                    let new_ity_size = Int(ity).size(dl);
                    for variant in &mut variants {
                        for i in variant.offsets.iter_mut() {
                            // The first field is the discrimminant, at offset 0.
                            // These aren't in order, and we need to skip it.
                            if *i <= old_ity_size && *i > Size::from_bytes(0) {
                                *i = new_ity_size;
                            }
                        }
                        // We might be making the struct larger.
                        if variant.min_size <= old_ity_size {
                            variant.min_size = new_ity_size;
                        }
                    }
                }

                General {
                    discr: ity,
                    variants,
                    size,
                    align,
                    primitive_align,
                }
            }

            // Types with no meaningful known layout.
            ty::TyProjection(_) | ty::TyAnon(..) => {
                let normalized = tcx.normalize_associated_type_in_env(&ty, param_env);
                if ty == normalized {
                    return Err(LayoutError::Unknown(ty));
                }
                return normalized.layout(tcx, param_env);
            }
            ty::TyParam(_) => {
                return Err(LayoutError::Unknown(ty));
            }
            ty::TyInfer(_) | ty::TyError => {
                bug!("Layout::compute: unexpected type `{}`", ty)
            }
        };

        success(layout)
    }

    /// Returns true if the layout corresponds to an unsized type.
    pub fn is_unsized(&self) -> bool {
        match *self {
            Scalar {..} | Vector {..} | FatPointer {..} |
            CEnum {..} | UntaggedUnion {..} | General {..} |
            RawNullablePointer {..} |
            StructWrappedNullablePointer {..} => false,

            Array { sized, .. } |
            Univariant { variant: Struct { sized, .. }, .. } => !sized
        }
    }

    pub fn size<C: HasDataLayout>(&self, cx: C) -> Size {
        let dl = cx.data_layout();

        match *self {
            Scalar { value, .. } | RawNullablePointer { value, .. } => {
                value.size(dl)
            }

            Vector { element, count } => {
                let element_size = element.size(dl);
                let vec_size = match element_size.checked_mul(count, dl) {
                    Some(size) => size,
                    None => bug!("Layout::size({:?}): {} * {} overflowed",
                                 self, element_size.bytes(), count)
                };
                vec_size.abi_align(self.align(dl))
            }

            Array { element_size, count, .. } => {
                match element_size.checked_mul(count, dl) {
                    Some(size) => size,
                    None => bug!("Layout::size({:?}): {} * {} overflowed",
                                 self, element_size.bytes(), count)
                }
            }

            FatPointer { metadata, .. } => {
                // Effectively a (ptr, meta) tuple.
                Pointer.size(dl).abi_align(metadata.align(dl))
                       .checked_add(metadata.size(dl), dl).unwrap()
                       .abi_align(self.align(dl))
            }

            CEnum { discr, .. } => Int(discr).size(dl),
            General { size, .. } => size,
            UntaggedUnion { ref variants } => variants.stride(),

            Univariant { ref variant, .. } |
            StructWrappedNullablePointer { nonnull: ref variant, .. } => {
                variant.stride()
            }
        }
    }

    pub fn align<C: HasDataLayout>(&self, cx: C) -> Align {
        let dl = cx.data_layout();

        match *self {
            Scalar { value, .. } | RawNullablePointer { value, .. } => {
                value.align(dl)
            }

            Vector { element, count } => {
                let elem_size = element.size(dl);
                let vec_size = match elem_size.checked_mul(count, dl) {
                    Some(size) => size,
                    None => bug!("Layout::align({:?}): {} * {} overflowed",
                                 self, elem_size.bytes(), count)
                };
                for &(size, align) in &dl.vector_align {
                    if size == vec_size {
                        return align;
                    }
                }
                // Default to natural alignment, which is what LLVM does.
                // That is, use the size, rounded up to a power of 2.
                let align = vec_size.bytes().next_power_of_two();
                Align::from_bytes(align, align).unwrap()
            }

            FatPointer { metadata, .. } => {
                // Effectively a (ptr, meta) tuple.
                Pointer.align(dl).max(metadata.align(dl))
            }

            CEnum { discr, .. } => Int(discr).align(dl),
            Array { align, .. } | General { align, .. } => align,
            UntaggedUnion { ref variants } => variants.align,

            Univariant { ref variant, .. } |
            StructWrappedNullablePointer { nonnull: ref variant, .. } => {
                variant.align
            }
        }
    }

    /// Returns alignment before repr alignment is applied
    pub fn primitive_align(&self, dl: &TargetDataLayout) -> Align {
        match *self {
            Array { primitive_align, .. } | General { primitive_align, .. } => primitive_align,
            Univariant { ref variant, .. } |
            StructWrappedNullablePointer { nonnull: ref variant, .. } => {
                variant.primitive_align
            },

            _ => self.align(dl)
        }
    }

    /// Returns repr alignment if it is greater than the primitive alignment.
    pub fn over_align(&self, dl: &TargetDataLayout) -> Option<u32> {
        let align = self.align(dl);
        let primitive_align = self.primitive_align(dl);
        if align.abi() > primitive_align.abi() {
            Some(align.abi() as u32)
        } else {
            None
        }
    }

    pub fn field_offset<C: HasDataLayout>(&self,
                                          cx: C,
                                          i: usize,
                                          variant_index: Option<usize>)
                                          -> Size {
        let dl = cx.data_layout();

        match *self {
            Scalar { .. } |
            CEnum { .. } |
            UntaggedUnion { .. } |
            RawNullablePointer { .. } => {
                Size::from_bytes(0)
            }

            Vector { element, count } => {
                let element_size = element.size(dl);
                let i = i as u64;
                assert!(i < count);
                Size::from_bytes(element_size.bytes() * count)
            }

            Array { element_size, count, .. } => {
                let i = i as u64;
                assert!(i < count);
                Size::from_bytes(element_size.bytes() * count)
            }

            FatPointer { metadata, .. } => {
                // Effectively a (ptr, meta) tuple.
                assert!(i < 2);
                if i == 0 {
                    Size::from_bytes(0)
                } else {
                    Pointer.size(dl).abi_align(metadata.align(dl))
                }
            }

            Univariant { ref variant, .. } => variant.offsets[i],

            General { ref variants, .. } => {
                let v = variant_index.expect("variant index required");
                variants[v].offsets[i + 1]
            }

            StructWrappedNullablePointer { nndiscr, ref nonnull, .. } => {
                if Some(nndiscr as usize) == variant_index {
                    nonnull.offsets[i]
                } else {
                    Size::from_bytes(0)
                }
            }
        }
    }

    /// This is invoked by the `layout_raw` query to record the final
    /// layout of each type.
    #[inline]
    pub fn record_layout_for_printing(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      ty: Ty<'tcx>,
                                      param_env: ty::ParamEnv<'tcx>,
                                      layout: &Layout) {
        // If we are running with `-Zprint-type-sizes`, record layouts for
        // dumping later. Ignore layouts that are done with non-empty
        // environments or non-monomorphic layouts, as the user only wants
        // to see the stuff resulting from the final trans session.
        if
            !tcx.sess.opts.debugging_opts.print_type_sizes ||
            ty.has_param_types() ||
            ty.has_self_ty() ||
            !param_env.caller_bounds.is_empty()
        {
            return;
        }

        Self::record_layout_for_printing_outlined(tcx, ty, param_env, layout)
    }

    fn record_layout_for_printing_outlined(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           ty: Ty<'tcx>,
                                           param_env: ty::ParamEnv<'tcx>,
                                           layout: &Layout) {
        // (delay format until we actually need it)
        let record = |kind, opt_discr_size, variants| {
            let type_desc = format!("{:?}", ty);
            let overall_size = layout.size(tcx);
            let align = layout.align(tcx);
            tcx.sess.code_stats.borrow_mut().record_type_size(kind,
                                                              type_desc,
                                                              align,
                                                              overall_size,
                                                              opt_discr_size,
                                                              variants);
        };

        let (adt_def, substs) = match ty.sty {
            ty::TyAdt(ref adt_def, substs) => {
                debug!("print-type-size t: `{:?}` process adt", ty);
                (adt_def, substs)
            }

            ty::TyClosure(..) => {
                debug!("print-type-size t: `{:?}` record closure", ty);
                record(DataTypeKind::Closure, None, vec![]);
                return;
            }

            _ => {
                debug!("print-type-size t: `{:?}` skip non-nominal", ty);
                return;
            }
        };

        let adt_kind = adt_def.adt_kind();

        let build_field_info = |(field_name, field_ty): (ast::Name, Ty<'tcx>), offset: &Size| {
            let layout = field_ty.layout(tcx, param_env);
            match layout {
                Err(_) => bug!("no layout found for field {} type: `{:?}`", field_name, field_ty),
                Ok(field_layout) => {
                    session::FieldInfo {
                        name: field_name.to_string(),
                        offset: offset.bytes(),
                        size: field_layout.size(tcx).bytes(),
                        align: field_layout.align(tcx).abi(),
                    }
                }
            }
        };

        let build_primitive_info = |name: ast::Name, value: &Primitive| {
            session::VariantInfo {
                name: Some(name.to_string()),
                kind: session::SizeKind::Exact,
                align: value.align(tcx).abi(),
                size: value.size(tcx).bytes(),
                fields: vec![],
            }
        };

        enum Fields<'a> {
            WithDiscrim(&'a Struct),
            NoDiscrim(&'a Struct),
        }

        let build_variant_info = |n: Option<ast::Name>,
                                  flds: &[(ast::Name, Ty<'tcx>)],
                                  layout: Fields| {
            let (s, field_offsets) = match layout {
                Fields::WithDiscrim(s) => (s, &s.offsets[1..]),
                Fields::NoDiscrim(s) => (s, &s.offsets[0..]),
            };
            let field_info: Vec<_> =
                flds.iter()
                    .zip(field_offsets.iter())
                    .map(|(&field_name_ty, offset)| build_field_info(field_name_ty, offset))
                    .collect();

            session::VariantInfo {
                name: n.map(|n|n.to_string()),
                kind: if s.sized {
                    session::SizeKind::Exact
                } else {
                    session::SizeKind::Min
                },
                align: s.align.abi(),
                size: s.min_size.bytes(),
                fields: field_info,
            }
        };

        match *layout {
            Layout::StructWrappedNullablePointer { nonnull: ref variant_layout,
                                                   nndiscr,
                                                   discrfield: _,
                                                   discrfield_source: _ } => {
                debug!("print-type-size t: `{:?}` adt struct-wrapped nullable nndiscr {} is {:?}",
                       ty, nndiscr, variant_layout);
                let variant_def = &adt_def.variants[nndiscr as usize];
                let fields: Vec<_> =
                    variant_def.fields.iter()
                                      .map(|field_def| (field_def.name, field_def.ty(tcx, substs)))
                                      .collect();
                record(adt_kind.into(),
                       None,
                       vec![build_variant_info(Some(variant_def.name),
                                               &fields,
                                               Fields::NoDiscrim(variant_layout))]);
            }
            Layout::RawNullablePointer { nndiscr, value } => {
                debug!("print-type-size t: `{:?}` adt raw nullable nndiscr {} is {:?}",
                       ty, nndiscr, value);
                let variant_def = &adt_def.variants[nndiscr as usize];
                record(adt_kind.into(), None,
                       vec![build_primitive_info(variant_def.name, &value)]);
            }
            Layout::Univariant { variant: ref variant_layout, non_zero: _ } => {
                let variant_names = || {
                    adt_def.variants.iter().map(|v|format!("{}", v.name)).collect::<Vec<_>>()
                };
                debug!("print-type-size t: `{:?}` adt univariant {:?} variants: {:?}",
                       ty, variant_layout, variant_names());
                assert!(adt_def.variants.len() <= 1,
                        "univariant with variants {:?}", variant_names());
                if adt_def.variants.len() == 1 {
                    let variant_def = &adt_def.variants[0];
                    let fields: Vec<_> =
                        variant_def.fields.iter()
                                          .map(|f| (f.name, f.ty(tcx, substs)))
                                          .collect();
                    record(adt_kind.into(),
                           None,
                           vec![build_variant_info(Some(variant_def.name),
                                                   &fields,
                                                   Fields::NoDiscrim(variant_layout))]);
                } else {
                    // (This case arises for *empty* enums; so give it
                    // zero variants.)
                    record(adt_kind.into(), None, vec![]);
                }
            }

            Layout::General { ref variants, discr, .. } => {
                debug!("print-type-size t: `{:?}` adt general variants def {} layouts {} {:?}",
                       ty, adt_def.variants.len(), variants.len(), variants);
                let variant_infos: Vec<_> =
                    adt_def.variants.iter()
                                    .zip(variants.iter())
                                    .map(|(variant_def, variant_layout)| {
                                        let fields: Vec<_> =
                                            variant_def.fields
                                                       .iter()
                                                       .map(|f| (f.name, f.ty(tcx, substs)))
                                                       .collect();
                                        build_variant_info(Some(variant_def.name),
                                                           &fields,
                                                           Fields::WithDiscrim(variant_layout))
                                    })
                                    .collect();
                record(adt_kind.into(), Some(discr.size()), variant_infos);
            }

            Layout::UntaggedUnion { ref variants } => {
                debug!("print-type-size t: `{:?}` adt union variants {:?}",
                       ty, variants);
                // layout does not currently store info about each
                // variant...
                record(adt_kind.into(), None, Vec::new());
            }

            Layout::CEnum { discr, .. } => {
                debug!("print-type-size t: `{:?}` adt c-like enum", ty);
                let variant_infos: Vec<_> =
                    adt_def.variants.iter()
                                    .map(|variant_def| {
                                        build_primitive_info(variant_def.name,
                                                             &Primitive::Int(discr))
                                    })
                                    .collect();
                record(adt_kind.into(), Some(discr.size()), variant_infos);
            }

            // other cases provide little interesting (i.e. adjustable
            // via representation tweaks) size info beyond total size.
            Layout::Scalar { .. } |
            Layout::Vector { .. } |
            Layout::Array { .. } |
            Layout::FatPointer { .. } => {
                debug!("print-type-size t: `{:?}` adt other", ty);
                record(adt_kind.into(), None, Vec::new())
            }
        }
    }
}

/// Type size "skeleton", i.e. the only information determining a type's size.
/// While this is conservative, (aside from constant sizes, only pointers,
/// newtypes thereof and null pointer optimized enums are allowed), it is
/// enough to statically check common usecases of transmute.
#[derive(Copy, Clone, Debug)]
pub enum SizeSkeleton<'tcx> {
    /// Any statically computable Layout.
    Known(Size),

    /// A potentially-fat pointer.
    Pointer {
        /// If true, this pointer is never null.
        non_zero: bool,
        /// The type which determines the unsized metadata, if any,
        /// of this pointer. Either a type parameter or a projection
        /// depending on one, with regions erased.
        tail: Ty<'tcx>
    }
}

impl<'a, 'tcx> SizeSkeleton<'tcx> {
    pub fn compute(ty: Ty<'tcx>,
                   tcx: TyCtxt<'a, 'tcx, 'tcx>,
                   param_env: ty::ParamEnv<'tcx>)
                   -> Result<SizeSkeleton<'tcx>, LayoutError<'tcx>> {
        assert!(!ty.has_infer_types());

        // First try computing a static layout.
        let err = match ty.layout(tcx, param_env) {
            Ok(layout) => {
                return Ok(SizeSkeleton::Known(layout.size(tcx)));
            }
            Err(err) => err
        };

        let ptr_skeleton = |pointee: Ty<'tcx>| {
            let non_zero = !ty.is_unsafe_ptr();
            let tail = tcx.struct_tail(pointee);
            match tail.sty {
                ty::TyParam(_) | ty::TyProjection(_) => {
                    assert!(tail.has_param_types() || tail.has_self_ty());
                    Ok(SizeSkeleton::Pointer {
                        non_zero,
                        tail: tcx.erase_regions(&tail)
                    })
                }
                _ => {
                    bug!("SizeSkeleton::compute({}): layout errored ({}), yet \
                            tail `{}` is not a type parameter or a projection",
                            ty, err, tail)
                }
            }
        };

        match ty.sty {
            ty::TyRef(_, ty::TypeAndMut { ty: pointee, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                ptr_skeleton(pointee)
            }
            ty::TyAdt(def, _) if def.is_box() => {
                ptr_skeleton(ty.boxed_ty())
            }

            ty::TyAdt(def, substs) => {
                // Only newtypes and enums w/ nullable pointer optimization.
                if def.is_union() || def.variants.is_empty() || def.variants.len() > 2 {
                    return Err(err);
                }

                // Get a zero-sized variant or a pointer newtype.
                let zero_or_ptr_variant = |i: usize| {
                    let fields = def.variants[i].fields.iter().map(|field| {
                        SizeSkeleton::compute(field.ty(tcx, substs), tcx, param_env)
                    });
                    let mut ptr = None;
                    for field in fields {
                        let field = field?;
                        match field {
                            SizeSkeleton::Known(size) => {
                                if size.bytes() > 0 {
                                    return Err(err);
                                }
                            }
                            SizeSkeleton::Pointer {..} => {
                                if ptr.is_some() {
                                    return Err(err);
                                }
                                ptr = Some(field);
                            }
                        }
                    }
                    Ok(ptr)
                };

                let v0 = zero_or_ptr_variant(0)?;
                // Newtype.
                if def.variants.len() == 1 {
                    if let Some(SizeSkeleton::Pointer { non_zero, tail }) = v0 {
                        return Ok(SizeSkeleton::Pointer {
                            non_zero: non_zero ||
                                Some(def.did) == tcx.lang_items().non_zero(),
                            tail,
                        });
                    } else {
                        return Err(err);
                    }
                }

                let v1 = zero_or_ptr_variant(1)?;
                // Nullable pointer enum optimization.
                match (v0, v1) {
                    (Some(SizeSkeleton::Pointer { non_zero: true, tail }), None) |
                    (None, Some(SizeSkeleton::Pointer { non_zero: true, tail })) => {
                        Ok(SizeSkeleton::Pointer {
                            non_zero: false,
                            tail,
                        })
                    }
                    _ => Err(err)
                }
            }

            ty::TyProjection(_) | ty::TyAnon(..) => {
                let normalized = tcx.normalize_associated_type_in_env(&ty, param_env);
                if ty == normalized {
                    Err(err)
                } else {
                    SizeSkeleton::compute(normalized, tcx, param_env)
                }
            }

            _ => Err(err)
        }
    }

    pub fn same_size(self, other: SizeSkeleton) -> bool {
        match (self, other) {
            (SizeSkeleton::Known(a), SizeSkeleton::Known(b)) => a == b,
            (SizeSkeleton::Pointer { tail: a, .. },
             SizeSkeleton::Pointer { tail: b, .. }) => a == b,
            _ => false
        }
    }
}

/// A pair of a type and its layout. Implements various
/// type traversal APIs (e.g. recursing into fields).
#[derive(Copy, Clone, Debug)]
pub struct TyLayout<'tcx> {
    pub ty: Ty<'tcx>,
    pub layout: &'tcx Layout,
    pub variant_index: Option<usize>,
}

impl<'tcx> Deref for TyLayout<'tcx> {
    type Target = Layout;
    fn deref(&self) -> &Layout {
        self.layout
    }
}

pub trait LayoutTyper<'tcx>: HasDataLayout {
    type TyLayout;

    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx>;
    fn layout_of(self, ty: Ty<'tcx>) -> Self::TyLayout;
    fn normalize_projections(self, ty: Ty<'tcx>) -> Ty<'tcx>;
}

/// Combines a tcx with the parameter environment so that you can
/// compute layout operations.
#[derive(Copy, Clone)]
pub struct LayoutCx<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'a, 'tcx> LayoutCx<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, param_env: ty::ParamEnv<'tcx>) -> Self {
        LayoutCx { tcx, param_env }
    }
}

impl<'a, 'tcx> HasDataLayout for LayoutCx<'a, 'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'a, 'tcx> LayoutTyper<'tcx> for LayoutCx<'a, 'tcx> {
    type TyLayout = Result<TyLayout<'tcx>, LayoutError<'tcx>>;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'tcx, 'tcx> {
        self.tcx
    }

    fn layout_of(self, ty: Ty<'tcx>) -> Self::TyLayout {
        let ty = self.normalize_projections(ty);

        Ok(TyLayout {
            ty,
            layout: ty.layout(self.tcx, self.param_env)?,
            variant_index: None
        })
    }

    fn normalize_projections(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.tcx.normalize_associated_type_in_env(&ty, self.param_env)
    }
}

impl<'a, 'tcx> TyLayout<'tcx> {
    pub fn for_variant(&self, variant_index: usize) -> Self {
        TyLayout {
            variant_index: Some(variant_index),
            ..*self
        }
    }

    pub fn field_offset<C: HasDataLayout>(&self, cx: C, i: usize) -> Size {
        self.layout.field_offset(cx, i, self.variant_index)
    }

    pub fn field_count(&self) -> usize {
        // Handle enum/union through the type rather than Layout.
        if let ty::TyAdt(def, _) = self.ty.sty {
            let v = self.variant_index.unwrap_or(0);
            if def.variants.is_empty() {
                assert_eq!(v, 0);
                return 0;
            } else {
                return def.variants[v].fields.len();
            }
        }

        match *self.layout {
            Scalar { .. } => {
                bug!("TyLayout::field_count({:?}): not applicable", self)
            }

            // Handled above (the TyAdt case).
            CEnum { .. } |
            General { .. } |
            UntaggedUnion { .. } |
            RawNullablePointer { .. } |
            StructWrappedNullablePointer { .. } => bug!(),

            FatPointer { .. } => 2,

            Vector { count, .. } |
            Array { count, .. } => {
                let usize_count = count as usize;
                assert_eq!(usize_count as u64, count);
                usize_count
            }

            Univariant { ref variant, .. } => variant.offsets.len(),
        }
    }

    pub fn field_type<C: LayoutTyper<'tcx>>(&self, cx: C, i: usize) -> Ty<'tcx> {
        let tcx = cx.tcx();

        let ptr_field_type = |pointee: Ty<'tcx>| {
            assert!(i < 2);
            let slice = |element: Ty<'tcx>| {
                if i == 0 {
                    tcx.mk_mut_ptr(element)
                } else {
                    tcx.types.usize
                }
            };
            match tcx.struct_tail(pointee).sty {
                ty::TySlice(element) => slice(element),
                ty::TyStr => slice(tcx.types.u8),
                ty::TyDynamic(..) => tcx.mk_mut_ptr(tcx.mk_nil()),
                _ => bug!("TyLayout::field_type({:?}): not applicable", self)
            }
        };

        match self.ty.sty {
            ty::TyBool |
            ty::TyChar |
            ty::TyInt(_) |
            ty::TyUint(_) |
            ty::TyFloat(_) |
            ty::TyFnPtr(_) |
            ty::TyNever |
            ty::TyFnDef(..) |
            ty::TyDynamic(..) => {
                bug!("TyLayout::field_type({:?}): not applicable", self)
            }

            // Potentially-fat pointers.
            ty::TyRef(_, ty::TypeAndMut { ty: pointee, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                ptr_field_type(pointee)
            }
            ty::TyAdt(def, _) if def.is_box() => {
                ptr_field_type(self.ty.boxed_ty())
            }

            // Arrays and slices.
            ty::TyArray(element, _) |
            ty::TySlice(element) => element,
            ty::TyStr => tcx.types.u8,

            // Tuples, generators and closures.
            ty::TyClosure(def_id, ref substs) => {
                substs.upvar_tys(def_id, tcx).nth(i).unwrap()
            }

            ty::TyGenerator(def_id, ref substs, _) => {
                substs.field_tys(def_id, tcx).nth(i).unwrap()
            }

            ty::TyTuple(tys, _) => tys[i],

            // SIMD vector types.
            ty::TyAdt(def, ..) if def.repr.simd() => {
                self.ty.simd_type(tcx)
            }

            // ADTs.
            ty::TyAdt(def, substs) => {
                def.variants[self.variant_index.unwrap_or(0)].fields[i].ty(tcx, substs)
            }

            ty::TyProjection(_) | ty::TyAnon(..) | ty::TyParam(_) |
            ty::TyInfer(_) | ty::TyError => {
                bug!("TyLayout::field_type: unexpected type `{}`", self.ty)
            }
        }
    }

    pub fn field<C: LayoutTyper<'tcx>>(&self,
                                       cx: C,
                                       i: usize)
                                       -> C::TyLayout {
        cx.layout_of(cx.normalize_projections(self.field_type(cx, i)))
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for Layout
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use ty::layout::Layout::*;
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Scalar { value, non_zero } => {
                value.hash_stable(hcx, hasher);
                non_zero.hash_stable(hcx, hasher);
            }
            Vector { element, count } => {
                element.hash_stable(hcx, hasher);
                count.hash_stable(hcx, hasher);
            }
            Array { sized, align, primitive_align, element_size, count } => {
                sized.hash_stable(hcx, hasher);
                align.hash_stable(hcx, hasher);
                primitive_align.hash_stable(hcx, hasher);
                element_size.hash_stable(hcx, hasher);
                count.hash_stable(hcx, hasher);
            }
            FatPointer { ref metadata, non_zero } => {
                metadata.hash_stable(hcx, hasher);
                non_zero.hash_stable(hcx, hasher);
            }
            CEnum { discr, signed, non_zero, min, max } => {
                discr.hash_stable(hcx, hasher);
                signed.hash_stable(hcx, hasher);
                non_zero.hash_stable(hcx, hasher);
                min.hash_stable(hcx, hasher);
                max.hash_stable(hcx, hasher);
            }
            Univariant { ref variant, non_zero } => {
                variant.hash_stable(hcx, hasher);
                non_zero.hash_stable(hcx, hasher);
            }
            UntaggedUnion { ref variants } => {
                variants.hash_stable(hcx, hasher);
            }
            General { discr, ref variants, size, align, primitive_align } => {
                discr.hash_stable(hcx, hasher);
                variants.hash_stable(hcx, hasher);
                size.hash_stable(hcx, hasher);
                align.hash_stable(hcx, hasher);
                primitive_align.hash_stable(hcx, hasher);
            }
            RawNullablePointer { nndiscr, ref value } => {
                nndiscr.hash_stable(hcx, hasher);
                value.hash_stable(hcx, hasher);
            }
            StructWrappedNullablePointer {
                nndiscr,
                ref nonnull,
                ref discrfield,
                ref discrfield_source
            } => {
                nndiscr.hash_stable(hcx, hasher);
                nonnull.hash_stable(hcx, hasher);
                discrfield.hash_stable(hcx, hasher);
                discrfield_source.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(enum ::ty::layout::Integer {
    I1,
    I8,
    I16,
    I32,
    I64,
    I128
});

impl_stable_hash_for!(enum ::ty::layout::Primitive {
    Int(integer),
    F32,
    F64,
    Pointer
});

impl_stable_hash_for!(struct ::ty::layout::Align {
    abi,
    pref
});

impl_stable_hash_for!(struct ::ty::layout::Size {
    raw
});

impl<'gcx> HashStable<StableHashingContext<'gcx>> for LayoutError<'gcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use ty::layout::LayoutError::*;
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Unknown(t) |
            SizeOverflow(t) => t.hash_stable(hcx, hasher)
        }
    }
}

impl_stable_hash_for!(struct ::ty::layout::Struct {
    align,
    primitive_align,
    packed,
    sized,
    offsets,
    memory_index,
    min_size
});

impl_stable_hash_for!(struct ::ty::layout::Union {
    align,
    primitive_align,
    min_size,
    packed
});
