//! Definitions needed for computing data layout of types.

use std::cmp;

use la_arena::{Idx, RawIdx};
pub use rustc_abi::{
    Abi, AbiAndPrefAlign, AddressSpace, Align, Endian, FieldsShape, Integer, IntegerType,
    LayoutCalculator, Niche, Primitive, ReprFlags, ReprOptions, Scalar, Size, StructKind,
    TargetDataLayout, TargetDataLayoutErrors, WrappingRange,
};

use crate::LocalEnumVariantId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RustcEnumVariantIdx(pub LocalEnumVariantId);

impl rustc_index::vec::Idx for RustcEnumVariantIdx {
    fn new(idx: usize) -> Self {
        RustcEnumVariantIdx(Idx::from_raw(RawIdx::from(idx as u32)))
    }

    fn index(self) -> usize {
        u32::from(self.0.into_raw()) as usize
    }
}

pub type Layout = rustc_abi::LayoutS<RustcEnumVariantIdx>;
pub type TagEncoding = rustc_abi::TagEncoding<RustcEnumVariantIdx>;
pub type Variants = rustc_abi::Variants<RustcEnumVariantIdx>;

pub trait IntegerExt {
    fn repr_discr(
        dl: &TargetDataLayout,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> Result<(Integer, bool), LayoutError>;
}

impl IntegerExt for Integer {
    /// Finds the appropriate Integer type and signedness for the given
    /// signed discriminant range and `#[repr]` attribute.
    /// N.B.: `u128` values above `i128::MAX` will be treated as signed, but
    /// that shouldn't affect anything, other than maybe debuginfo.
    fn repr_discr(
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
            let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
            if discr < fit {
                return Err(LayoutError::UserError(
                    "Integer::repr_discr: `#[repr]` hint too small for \
                      discriminant range of enum "
                        .to_string(),
                ));
            }
            return Ok((discr, ity.is_signed()));
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LayoutError {
    UserError(String),
    SizeOverflow,
    TargetLayoutNotAvailable,
    HasPlaceholder,
    NotImplemented,
    Unknown,
}
