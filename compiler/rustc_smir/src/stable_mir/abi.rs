use std::fmt::{self, Debug};
use std::num::NonZero;
use std::ops::RangeInclusive;

use serde::Serialize;
use stable_mir::compiler_interface::with;
use stable_mir::mir::FieldIdx;
use stable_mir::target::{MachineInfo, MachineSize as Size};
use stable_mir::ty::{Align, IndexedVal, Ty, VariantIdx};
use stable_mir::{Error, Opaque, error};

use crate::stable_mir;

/// A function ABI definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct FnAbi {
    /// The types of each argument.
    pub args: Vec<ArgAbi>,

    /// The expected return type.
    pub ret: ArgAbi,

    /// The count of non-variadic arguments.
    ///
    /// Should only be different from `args.len()` when a function is a C variadic function.
    pub fixed_count: u32,

    /// The ABI convention.
    pub conv: CallConvention,

    /// Whether this is a variadic C function,
    pub c_variadic: bool,
}

/// Information about the ABI of a function's argument, or return value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct ArgAbi {
    pub ty: Ty,
    pub layout: Layout,
    pub mode: PassMode,
}

/// How a function argument should be passed in to the target function.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum PassMode {
    /// Ignore the argument.
    ///
    /// The argument is either uninhabited or a ZST.
    Ignore,
    /// Pass the argument directly.
    ///
    /// The argument has a layout abi of `Scalar` or `Vector`.
    Direct(Opaque),
    /// Pass a pair's elements directly in two arguments.
    ///
    /// The argument has a layout abi of `ScalarPair`.
    Pair(Opaque, Opaque),
    /// Pass the argument after casting it.
    Cast { pad_i32: bool, cast: Opaque },
    /// Pass the argument indirectly via a hidden pointer.
    Indirect { attrs: Opaque, meta_attrs: Opaque, on_stack: bool },
}

/// The layout of a type, alongside the type itself.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct TyAndLayout {
    pub ty: Ty,
    pub layout: Layout,
}

/// The layout of a type in memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct LayoutShape {
    /// The fields location within the layout
    pub fields: FieldsShape,

    /// Encodes information about multi-variant layouts.
    /// Even with `Multiple` variants, a layout still has its own fields! Those are then
    /// shared between all variants.
    ///
    /// To access all fields of this layout, both `fields` and the fields of the active variant
    /// must be taken into account.
    pub variants: VariantsShape,

    /// The `abi` defines how this data is passed between functions.
    pub abi: ValueAbi,

    /// The ABI mandated alignment in bytes.
    pub abi_align: Align,

    /// The size of this layout in bytes.
    pub size: Size,
}

impl LayoutShape {
    /// Returns `true` if the layout corresponds to an unsized type.
    #[inline]
    pub fn is_unsized(&self) -> bool {
        self.abi.is_unsized()
    }

    #[inline]
    pub fn is_sized(&self) -> bool {
        !self.abi.is_unsized()
    }

    /// Returns `true` if the type is sized and a 1-ZST (meaning it has size 0 and alignment 1).
    pub fn is_1zst(&self) -> bool {
        self.is_sized() && self.size.bits() == 0 && self.abi_align == 1
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Layout(usize);

impl Layout {
    pub fn shape(self) -> LayoutShape {
        with(|cx| cx.layout_shape(self))
    }
}

impl IndexedVal for Layout {
    fn to_val(index: usize) -> Self {
        Layout(index)
    }
    fn to_index(&self) -> usize {
        self.0
    }
}

/// Describes how the fields of a type are shaped in memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum FieldsShape {
    /// Scalar primitives and `!`, which never have fields.
    Primitive,

    /// All fields start at no offset. The `usize` is the field count.
    Union(NonZero<usize>),

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
        /// I.e.: It follows the same order as [super::ty::VariantDef::fields()].
        /// This vector does not go in increasing order.
        offsets: Vec<Size>,
    },
}

impl FieldsShape {
    pub fn fields_by_offset_order(&self) -> Vec<FieldIdx> {
        match self {
            FieldsShape::Primitive => vec![],
            FieldsShape::Union(_) | FieldsShape::Array { .. } => (0..self.count()).collect(),
            FieldsShape::Arbitrary { offsets, .. } => {
                let mut indices = (0..offsets.len()).collect::<Vec<_>>();
                indices.sort_by_key(|idx| offsets[*idx]);
                indices
            }
        }
    }

    pub fn count(&self) -> usize {
        match self {
            FieldsShape::Primitive => 0,
            FieldsShape::Union(count) => count.get(),
            FieldsShape::Array { count, .. } => *count as usize,
            FieldsShape::Arbitrary { offsets, .. } => offsets.len(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum VariantsShape {
    /// A type with no valid variants. Must be uninhabited.
    Empty,

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
        variants: Vec<LayoutShape>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
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

/// Describes how values of the type are passed by target ABIs,
/// in terms of categories of C types there are ABI rules for.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum ValueAbi {
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

impl ValueAbi {
    /// Returns `true` if the layout corresponds to an unsized type.
    pub fn is_unsized(&self) -> bool {
        match *self {
            ValueAbi::Scalar(_) | ValueAbi::ScalarPair(..) | ValueAbi::Vector { .. } => false,
            ValueAbi::Aggregate { sized } => !sized,
        }
    }
}

/// Information about one scalar component of a Rust type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
pub enum Scalar {
    Initialized {
        /// The primitive type used to represent this value.
        value: Primitive,
        /// The range that represents valid values.
        /// The range must be valid for the `primitive` size.
        valid_range: WrappingRange,
    },
    Union {
        /// Unions never have niches, so there is no `valid_range`.
        /// Even for unions, we need to use the correct registers for the kind of
        /// values inside the union, so we keep the `Primitive` type around.
        /// It is also used to compute the size of the scalar.
        value: Primitive,
    },
}

impl Scalar {
    pub fn has_niche(&self, target: &MachineInfo) -> bool {
        match self {
            Scalar::Initialized { value, valid_range } => {
                !valid_range.is_full(value.size(target)).unwrap()
            }
            Scalar::Union { .. } => false,
        }
    }
}

/// Fundamental unit of memory access and layout.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize)]
pub enum Primitive {
    /// The `bool` is the signedness of the `Integer` type.
    ///
    /// One would think we would not care about such details this low down,
    /// but some ABIs are described in terms of C types and ISAs where the
    /// integer arithmetic is done on {sign,zero}-extended registers, e.g.
    /// a negative integer passed by zero-extension will appear positive in
    /// the callee, and most operations on it will produce the wrong values.
    Int {
        length: IntegerLength,
        signed: bool,
    },
    Float {
        length: FloatLength,
    },
    Pointer(AddressSpace),
}

impl Primitive {
    pub fn size(self, target: &MachineInfo) -> Size {
        match self {
            Primitive::Int { length, .. } => Size::from_bits(length.bits()),
            Primitive::Float { length } => Size::from_bits(length.bits()),
            Primitive::Pointer(_) => target.pointer_width,
        }
    }
}

/// Enum representing the existing integer lengths.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize)]
pub enum IntegerLength {
    I8,
    I16,
    I32,
    I64,
    I128,
}

/// Enum representing the existing float lengths.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize)]
pub enum FloatLength {
    F16,
    F32,
    F64,
    F128,
}

impl IntegerLength {
    pub fn bits(self) -> usize {
        match self {
            IntegerLength::I8 => 8,
            IntegerLength::I16 => 16,
            IntegerLength::I32 => 32,
            IntegerLength::I64 => 64,
            IntegerLength::I128 => 128,
        }
    }
}

impl FloatLength {
    pub fn bits(self) -> usize {
        match self {
            FloatLength::F16 => 16,
            FloatLength::F32 => 32,
            FloatLength::F64 => 64,
            FloatLength::F128 => 128,
        }
    }
}

/// An identifier that specifies the address space that some operation
/// should operate on. Special address spaces have an effect on code generation,
/// depending on the target and the address spaces it implements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct AddressSpace(pub u32);

impl AddressSpace {
    /// The default address space, corresponding to data space.
    pub const DATA: Self = AddressSpace(0);
}

/// Inclusive wrap-around range of valid values (bitwise representation), that is, if
/// start > end, it represents `start..=MAX`, followed by `0..=end`.
///
/// That is, for an i8 primitive, a range of `254..=2` means following
/// sequence:
///
///    254 (-2), 255 (-1), 0, 1, 2
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct WrappingRange {
    pub start: u128,
    pub end: u128,
}

impl WrappingRange {
    /// Returns `true` if `size` completely fills the range.
    #[inline]
    pub fn is_full(&self, size: Size) -> Result<bool, Error> {
        let Some(max_value) = size.unsigned_int_max() else {
            return Err(error!("Expected size <= 128 bits, but found {} instead", size.bits()));
        };
        if self.start <= max_value && self.end <= max_value {
            Ok(self.start == (self.end.wrapping_add(1) & max_value))
        } else {
            Err(error!("Range `{self:?}` out of bounds for size `{}` bits.", size.bits()))
        }
    }

    /// Returns `true` if `v` is contained in the range.
    #[inline(always)]
    pub fn contains(&self, v: u128) -> bool {
        if self.wraps_around() {
            self.start <= v || v <= self.end
        } else {
            self.start <= v && v <= self.end
        }
    }

    /// Returns `true` if the range wraps around.
    /// I.e., the range represents the union of `self.start..=MAX` and `0..=self.end`.
    /// Returns `false` if this is a non-wrapping range, i.e.: `self.start..=self.end`.
    #[inline]
    pub fn wraps_around(&self) -> bool {
        self.start > self.end
    }
}

impl Debug for WrappingRange {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start > self.end {
            write!(fmt, "(..={}) | ({}..)", self.end, self.start)?;
        } else {
            write!(fmt, "{}..={}", self.start, self.end)?;
        }
        Ok(())
    }
}

/// General language calling conventions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum CallConvention {
    C,
    Rust,

    Cold,
    PreserveMost,
    PreserveAll,

    Custom,

    // Target-specific calling conventions.
    ArmAapcs,
    CCmseNonSecureCall,
    CCmseNonSecureEntry,

    Msp430Intr,

    PtxKernel,

    GpuKernel,

    X86Fastcall,
    X86Intr,
    X86Stdcall,
    X86ThisCall,
    X86VectorCall,

    X86_64SysV,
    X86_64Win64,

    AvrInterrupt,
    AvrNonBlockingInterrupt,

    RiscvInterrupt,
}

#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize)]
pub struct ReprFlags {
    pub is_simd: bool,
    pub is_c: bool,
    pub is_transparent: bool,
    pub is_linear: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize)]
pub enum IntegerType {
    /// Pointer-sized integer type, i.e. `isize` and `usize`.
    Pointer {
        /// Signedness. e.g. `true` for `isize`
        is_signed: bool,
    },
    /// Fixed-sized integer type, e.g. `i8`, `u32`, `i128`.
    Fixed {
        /// Length of this integer type. e.g. `IntegerLength::I8` for `u8`.
        length: IntegerLength,
        /// Signedness. e.g. `false` for `u8`
        is_signed: bool,
    },
}

/// Representation options provided by the user
#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize)]
pub struct ReprOptions {
    pub int: Option<IntegerType>,
    pub align: Option<Align>,
    pub pack: Option<Align>,
    pub flags: ReprFlags,
}
