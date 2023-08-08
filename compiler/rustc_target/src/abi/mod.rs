pub use Integer::*;
pub use Primitive::*;

use crate::json::{Json, ToJson};

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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable_Generic)]
pub struct TyAndLayout<'a, Ty> {
    pub ty: Ty,
    pub layout: Layout<'a>,
}

impl<'a, Ty> Deref for TyAndLayout<'a, Ty> {
    type Target = &'a LayoutS;
    fn deref(&self) -> &&'a LayoutS {
        &self.layout.0.0
    }
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
    fn is_adt(this: TyAndLayout<'a, Self>) -> bool;
    fn is_never(this: TyAndLayout<'a, Self>) -> bool;
    fn is_tuple(this: TyAndLayout<'a, Self>) -> bool;
    fn is_unit(this: TyAndLayout<'a, Self>) -> bool;
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

    pub fn offset_of_subfield<C>(self, cx: &C, indices: impl Iterator<Item = usize>) -> Size
    where
        Ty: TyAbiInterface<'a, C>,
    {
        let mut layout = self;
        let mut offset = Size::ZERO;

        for index in indices {
            offset += layout.fields.offset(index);
            layout = layout.field(cx, index);
        }

        offset
    }
}
