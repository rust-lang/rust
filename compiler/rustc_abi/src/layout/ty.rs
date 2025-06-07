use std::fmt;
use std::ops::Deref;

use rustc_data_structures::intern::Interned;
use rustc_macros::HashStable_Generic;

use crate::{
    AbiAlign, Align, BackendRepr, FieldsShape, Float, HasDataLayout, LayoutData, Niche,
    PointeeInfo, Primitive, Scalar, Size, TargetDataLayout, Variants,
};

// Explicitly import `Float` to avoid ambiguity with `Primitive::Float`.

rustc_index::newtype_index! {
    /// The *source-order* index of a field in a variant.
    ///
    /// This is how most code after type checking refers to fields, rather than
    /// using names (as names have hygiene complications and more complex lookup).
    ///
    /// Particularly for `repr(Rust)` types, this may not be the same as *layout* order.
    /// (It is for `repr(C)` `struct`s, however.)
    ///
    /// For example, in the following types,
    /// ```rust
    /// # enum Never {}
    /// # #[repr(u16)]
    /// enum Demo1 {
    ///    Variant0 { a: Never, b: i32 } = 100,
    ///    Variant1 { c: u8, d: u64 } = 10,
    /// }
    /// struct Demo2 { e: u8, f: u16, g: u8 }
    /// ```
    /// `b` is `FieldIdx(1)` in `VariantIdx(0)`,
    /// `d` is `FieldIdx(1)` in `VariantIdx(1)`, and
    /// `f` is `FieldIdx(1)` in `VariantIdx(0)`.
    #[derive(HashStable_Generic)]
    #[encodable]
    #[orderable]
    pub struct FieldIdx {}
}

impl FieldIdx {
    /// The second field, at index 1.
    ///
    /// For use alongside [`FieldIdx::ZERO`], particularly with scalar pairs.
    pub const ONE: FieldIdx = FieldIdx::from_u32(1);
}

rustc_index::newtype_index! {
    /// The *source-order* index of a variant in a type.
    ///
    /// For enums, these are always `0..variant_count`, regardless of any
    /// custom discriminants that may have been defined, and including any
    /// variants that may end up uninhabited due to field types.  (Some of the
    /// variants may not be present in a monomorphized ABI [`Variants`], but
    /// those skipped variants are always counted when determining the *index*.)
    ///
    /// `struct`s, `tuples`, and `unions`s are considered to have a single variant
    /// with variant index zero, aka [`FIRST_VARIANT`].
    #[derive(HashStable_Generic)]
    #[encodable]
    #[orderable]
    pub struct VariantIdx {
        /// Equivalent to `VariantIdx(0)`.
        const FIRST_VARIANT = 0;
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable_Generic)]
#[rustc_pass_by_value]
pub struct Layout<'a>(pub Interned<'a, LayoutData<FieldIdx, VariantIdx>>);

impl<'a> fmt::Debug for Layout<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // See comment on `<LayoutS as Debug>::fmt` above.
        self.0.0.fmt(f)
    }
}

impl<'a> Deref for Layout<'a> {
    type Target = &'a LayoutData<FieldIdx, VariantIdx>;
    fn deref(&self) -> &&'a LayoutData<FieldIdx, VariantIdx> {
        &self.0.0
    }
}

impl<'a> Layout<'a> {
    pub fn fields(self) -> &'a FieldsShape<FieldIdx> {
        &self.0.0.fields
    }

    pub fn variants(self) -> &'a Variants<FieldIdx, VariantIdx> {
        &self.0.0.variants
    }

    pub fn backend_repr(self) -> BackendRepr {
        self.0.0.backend_repr
    }

    pub fn largest_niche(self) -> Option<Niche> {
        self.0.0.largest_niche
    }

    pub fn align(self) -> AbiAlign {
        self.0.0.align
    }

    pub fn size(self) -> Size {
        self.0.0.size
    }

    pub fn max_repr_align(self) -> Option<Align> {
        self.0.0.max_repr_align
    }

    pub fn unadjusted_abi_align(self) -> Align {
        self.0.0.unadjusted_abi_align
    }

    /// Whether the layout is from a type that implements [`std::marker::PointerLike`].
    ///
    /// Currently, that means that the type is pointer-sized, pointer-aligned,
    /// and has a initialized (non-union), scalar ABI.
    pub fn is_pointer_like(self, data_layout: &TargetDataLayout) -> bool {
        self.size() == data_layout.pointer_size
            && self.align().abi == data_layout.pointer_align.abi
            && matches!(self.backend_repr(), BackendRepr::Scalar(Scalar::Initialized { .. }))
    }
}

/// The layout of a type, alongside the type itself.
/// Provides various type traversal APIs (e.g., recursing into fields).
///
/// Note that the layout is NOT guaranteed to always be identical
/// to that obtained from `layout_of(ty)`, as we need to produce
/// layouts for which Rust types do not exist, such as enum variants
/// or synthetic fields of enums (i.e., discriminants) and wide pointers.
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
    type Target = &'a LayoutData<FieldIdx, VariantIdx>;
    fn deref(&self) -> &&'a LayoutData<FieldIdx, VariantIdx> {
        &self.layout.0.0
    }
}

impl<'a, Ty> AsRef<LayoutData<FieldIdx, VariantIdx>> for TyAndLayout<'a, Ty> {
    fn as_ref(&self) -> &LayoutData<FieldIdx, VariantIdx> {
        &*self.layout.0.0
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
        match self.backend_repr {
            BackendRepr::Scalar(scalar) => {
                matches!(scalar.primitive(), Primitive::Float(Float::F32 | Float::F64))
            }
            BackendRepr::Memory { .. } => {
                if self.fields.count() == 1 && self.fields.offset(0).bytes() == 0 {
                    self.field(cx, 0).is_single_fp_element(cx)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub fn is_single_vector_element<C>(self, cx: &C, expected_size: Size) -> bool
    where
        Ty: TyAbiInterface<'a, C>,
        C: HasDataLayout,
    {
        match self.backend_repr {
            BackendRepr::SimdVector { .. } => self.size == expected_size,
            BackendRepr::Memory { .. } => {
                if self.fields.count() == 1 && self.fields.offset(0).bytes() == 0 {
                    self.field(cx, 0).is_single_vector_element(cx, expected_size)
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

    /// Finds the one field that is not a 1-ZST.
    /// Returns `None` if there are multiple non-1-ZST fields or only 1-ZST-fields.
    pub fn non_1zst_field<C>(&self, cx: &C) -> Option<(FieldIdx, Self)>
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
            found = Some((FieldIdx::from_usize(field_idx), field));
        }
        found
    }
}
