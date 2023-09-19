pub use Integer::*;
pub use Primitive::*;

use crate::json::{Json, ToJson};

use std::fmt;
use std::ops::Deref;

use rustc_macros::HashStable_Generic;

pub mod call;

pub use rustc_abi::*;

impl ToJson for Endian {
    fn to_json(&self) -> Json {
        self.as_str().to_json()
    }
}

/// The layout of a type, alongside the type itself.
/// Provides various type traversal APIs (e.g., recursing into fields).
///
/// Note that the layout is NOT guaranteed to always be identical
/// to that obtained from `layout_of(ty)`, as we need to produce
/// layouts for which Rust types do not exist, such as enum variants
/// or synthetic fields of enums (i.e., discriminants) and fat pointers.
#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable_Generic)]
pub struct TyAndLayout<'a, Ty> {
    pub ty: Ty,
    pub layout: Layout<'a>,
}

impl<'a, Ty: fmt::Display> fmt::Debug for TyAndLayout<'a, Ty> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print the type in a readable way, not its debug representation.
        f.debug_struct("TyAndLayout")
            .field("ty", &format_args!("{}", self.ty))
            .field("layout", &self.layout)
            .finish()
    }
}

impl<'a, Ty> Deref for TyAndLayout<'a, Ty> {
    type Target = &'a LayoutS;
    fn deref(&self) -> &&'a LayoutS {
        &self.layout.0.0
    }
}

/// Trait that needs to be implemented by the higher-level type representation
/// (e.g. `rustc_middle::ty::Ty`), to provide `rustc_target::abi` functionality.
pub trait TyAbiInterface<'a, C>: Sized + std::fmt::Debug {
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
    fn is_adt(this: TyAndLayout<'a, Self>) -> bool;
    fn is_never(this: TyAndLayout<'a, Self>) -> bool;
    fn is_tuple(this: TyAndLayout<'a, Self>) -> bool;
    fn is_unit(this: TyAndLayout<'a, Self>) -> bool;
    fn is_transparent(this: TyAndLayout<'a, Self>) -> bool;
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

    pub fn is_single_fp_element<C>(self, cx: &C) -> bool
    where
        Ty: TyAbiInterface<'a, C>,
        C: HasDataLayout,
    {
        match self.abi {
            Abi::Scalar(scalar) => matches!(scalar.primitive(), F32 | F64),
            Abi::Aggregate { .. } => {
                if self.fields.count() == 1 && self.fields.offset(0).bytes() == 0 {
                    self.field(cx, 0).is_single_fp_element(cx)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub fn is_adt<C>(self) -> bool
    where
        Ty: TyAbiInterface<'a, C>,
    {
        Ty::is_adt(self)
    }

    pub fn is_never<C>(self) -> bool
    where
        Ty: TyAbiInterface<'a, C>,
    {
        Ty::is_never(self)
    }

    pub fn is_tuple<C>(self) -> bool
    where
        Ty: TyAbiInterface<'a, C>,
    {
        Ty::is_tuple(self)
    }

    pub fn is_unit<C>(self) -> bool
    where
        Ty: TyAbiInterface<'a, C>,
    {
        Ty::is_unit(self)
    }

    pub fn is_transparent<C>(self) -> bool
    where
        Ty: TyAbiInterface<'a, C>,
    {
        Ty::is_transparent(self)
    }

    pub fn offset_of_subfield<C>(self, cx: &C, indices: impl Iterator<Item = usize>) -> Size
    where
        Ty: TyAbiInterface<'a, C>,
    {
        let mut layout = self;
        let mut offset = Size::ZERO;

        for index in indices {
            offset += layout.fields.offset(index);
            layout = layout.field(cx, index);
            assert!(
                layout.is_sized(),
                "offset of unsized field (type {:?}) cannot be computed statically",
                layout.ty
            );
        }

        offset
    }

    /// Finds the one field that is not a 1-ZST.
    /// Returns `None` if there are multiple non-1-ZST fields or only 1-ZST-fields.
    pub fn non_1zst_field<C>(&self, cx: &C) -> Option<(usize, Self)>
    where
        Ty: TyAbiInterface<'a, C> + Copy,
    {
        let mut found = None;
        for field_idx in 0..self.fields.count() {
            let field = self.field(cx, field_idx);
            if field.is_1zst() {
                continue;
            }
            if found.is_some() {
                // More than one non-1-ZST field.
                return None;
            }
            found = Some((field_idx, field));
        }
        found
    }
}
