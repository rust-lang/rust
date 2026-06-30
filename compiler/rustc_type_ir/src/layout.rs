use std::fmt;
use std::ops::{Deref, Range};

use derive_where::derive_where;
use rustc_abi::{
    BackendRepr, FieldIdx, FieldsShape, Float, HasDataLayout, LayoutData, PointeeInfo, Primitive,
    Size, VariantIdx, Variants,
};
use rustc_data_structures::range_set::RangeSet;
use rustc_macros::StableHash_NoContext;

use crate::Interner;

// Explicitly import `Float` to avoid ambiguity with `Primitive::Float`.

/// The layout of a type, alongside the type itself.
/// Provides various type traversal APIs (e.g., recursing into fields).
///
/// Note that the layout is NOT guaranteed to always be identical
/// to that obtained from `layout_of(ty)`, as we need to produce
/// layouts for which Rust types do not exist, such as enum variants
/// or synthetic fields of enums (i.e., discriminants) and wide pointers.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq; I: Interner)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct TyAndLayout<I: Interner> {
    pub ty: I::Ty,
    pub layout: I::Layout,
}

impl<I: Interner> fmt::Debug for TyAndLayout<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print the type in a readable way, not its debug representation.
        f.debug_struct("TyAndLayout")
            .field("ty", &format_args!("{}", self.ty))
            .field("layout", &self.layout)
            .finish()
    }
}

impl<I: Interner> Deref for TyAndLayout<I> {
    type Target = LayoutData<FieldIdx, VariantIdx>;
    fn deref(&self) -> &LayoutData<FieldIdx, VariantIdx> {
        self.layout.deref()
    }
}

impl<I: Interner> AsRef<LayoutData<FieldIdx, VariantIdx>> for TyAndLayout<I> {
    fn as_ref(&self) -> &LayoutData<FieldIdx, VariantIdx> {
        self.layout.deref()
    }
}

#[cfg(feature = "nightly")]
impl<I: Interner, E: rustc_serialize::Encoder> rustc_serialize::Encodable<E> for TyAndLayout<I>
where
    I::Layout: rustc_serialize::Encodable<E>,
    I::Ty: rustc_serialize::Encodable<E>,
{
    fn encode(&self, s: &mut E) {
        self.layout.encode(s);
        self.ty.encode(s);
    }
}

#[cfg(feature = "nightly")]
impl<I: Interner, D: rustc_serialize::Decoder> rustc_serialize::Decodable<D> for TyAndLayout<I>
where
    I::Layout: rustc_serialize::Decodable<D>,
    I::Ty: rustc_serialize::Decodable<D>,
{
    fn decode(decoder: &mut D) -> Self {
        let layout = I::Layout::decode(decoder);
        let ty = I::Ty::decode(decoder);
        Self { layout, ty }
    }
}

/// Trait that needs to be implemented by the higher-level type representation
/// (e.g. `rustc_middle::ty::Ty`), to provide `rustc_target::abi` functionality.
pub trait TyAbiInterface<C>: Interner {
    fn ty_and_layout_for_variant(
        this: TyAndLayout<Self>,
        cx: &C,
        variant_index: VariantIdx,
    ) -> TyAndLayout<Self>;
    fn ty_and_layout_field(this: TyAndLayout<Self>, cx: &C, i: usize) -> TyAndLayout<Self>;
    fn ty_and_layout_pointee_info_at(
        this: TyAndLayout<Self>,
        cx: &C,
        offset: Size,
    ) -> Option<PointeeInfo>;
    fn is_adt(this: TyAndLayout<Self>) -> bool;
    fn is_never(this: TyAndLayout<Self>) -> bool;
    fn is_tuple(this: TyAndLayout<Self>) -> bool;
    fn is_unit(this: TyAndLayout<Self>) -> bool;
    fn is_transparent(this: TyAndLayout<Self>) -> bool;
    fn is_scalable_vector(this: TyAndLayout<Self>) -> bool;
    /// See [`TyAndLayout::pass_indirectly_in_non_rustic_abis`] for details.
    fn is_pass_indirectly_in_non_rustic_abis_flag_set(this: TyAndLayout<Self>) -> bool;
}

impl<I: Interner> TyAndLayout<I>
where
    I::Ty: fmt::Display,
{
    pub fn for_variant<C>(self, cx: &C, variant_index: VariantIdx) -> Self
    where
        I: TyAbiInterface<C>,
    {
        I::ty_and_layout_for_variant(self, cx, variant_index)
    }

    pub fn field<C>(self, cx: &C, i: usize) -> Self
    where
        I: TyAbiInterface<C>,
    {
        I::ty_and_layout_field(self, cx, i)
    }

    pub fn pointee_info_at<C>(self, cx: &C, offset: Size) -> Option<PointeeInfo>
    where
        I: TyAbiInterface<C>,
    {
        I::ty_and_layout_pointee_info_at(self, cx, offset)
    }

    pub fn is_single_fp_element<C>(self, cx: &C) -> bool
    where
        I: TyAbiInterface<C>,
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
        I: TyAbiInterface<C>,
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
        I: TyAbiInterface<C>,
    {
        I::is_adt(self)
    }

    pub fn is_never<C>(self) -> bool
    where
        I: TyAbiInterface<C>,
    {
        I::is_never(self)
    }

    pub fn is_tuple<C>(self) -> bool
    where
        I: TyAbiInterface<C>,
    {
        I::is_tuple(self)
    }

    pub fn is_unit<C>(self) -> bool
    where
        I: TyAbiInterface<C>,
    {
        I::is_unit(self)
    }

    pub fn is_transparent<C>(self) -> bool
    where
        I: TyAbiInterface<C>,
    {
        I::is_transparent(self)
    }

    pub fn is_scalable_vector<C>(self) -> bool
    where
        I: TyAbiInterface<C>,
    {
        I::is_scalable_vector(self)
    }

    /// If this method returns `true`, then this type should always have a `PassMode` of
    /// `Indirect { on_stack: false, .. }` when being used as the argument type of a function with a
    /// non-Rustic ABI (this is true for structs annotated with the
    /// `#[rustc_pass_indirectly_in_non_rustic_abis]` attribute).
    ///
    /// This is used to replicate some of the behaviour of C array-to-pointer decay; however unlike
    /// C any changes the caller makes to the passed value will not be reflected in the callee, so
    /// the attribute is only useful for types where observing the value in the caller after the
    /// function call isn't allowed (a.k.a. `va_list`).
    ///
    /// This function handles transparent types automatically.
    pub fn pass_indirectly_in_non_rustic_abis<C>(self, cx: &C) -> bool
    where
        I: TyAbiInterface<C>,
    {
        let base = self.peel_transparent_wrappers(cx);
        I::is_pass_indirectly_in_non_rustic_abis_flag_set(base)
    }

    /// Recursively peel away transparent wrappers, returning the inner value.
    ///
    /// The return value is not `repr(transparent)` and/or does
    /// not have a non-1zst field.
    pub fn peel_transparent_wrappers<C>(mut self, cx: &C) -> Self
    where
        I: TyAbiInterface<C>,
    {
        while self.is_transparent()
            && let Some((_, field)) = self.non_1zst_field(cx)
        {
            self = field;
        }

        self
    }

    /// Finds the one field that is not a 1-ZST.
    /// Returns `None` if there are multiple non-1-ZST fields or only 1-ZST-fields.
    pub fn non_1zst_field<C>(&self, cx: &C) -> Option<(FieldIdx, Self)>
    where
        I: TyAbiInterface<C>,
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

    /// The ranges of bytes that are always ignored by the representation relation of this type.
    ///
    /// In other words, for any sequence of bytes, if we reset the these padding bytes to uninit,
    /// then these two sequences of bytes represent the same value (or they are both invalid).
    /// This is the "guaranteed" padding. There may be more bytes that are padding for some
    /// but not all variants of this type; those are not included.
    /// (E.g. `Option<i8>` has no guaranteed padding so the empty range set is returned, but its `None` value still has padding).
    pub fn padding_ranges<C>(&self, cx: &C) -> Vec<Range<Size>>
    where
        I: TyAbiInterface<C>,
    {
        let mut data = RangeSet::new();
        self.add_data_ranges(cx, Size::ZERO, &mut data);

        // Find gaps between the data ranges.
        let mut uninit_ranges = Vec::new();
        let mut covered_until = Size::ZERO;
        for &(offset, size) in data.0.iter() {
            if offset > covered_until {
                uninit_ranges.push(covered_until..offset);
            }
            covered_until = Ord::max(covered_until, offset + size);
        }

        // Add trailing padding.
        if self.size > covered_until {
            uninit_ranges.push(covered_until..self.size);
        }

        uninit_ranges
    }

    /// Extend `out` with all ranges of bytes that *may* carry relevant data for values of this type.
    /// For enums and unions there are offsets that are initialized for some
    /// variants but not for others; those offset *will* get added to `out`.
    fn add_data_ranges<C>(self, cx: &C, base_offset: Size, out: &mut RangeSet<Size>)
    where
        I: TyAbiInterface<C>,
    {
        if self.is_zst() {
            return;
        }

        match &self.variants {
            Variants::Empty => { /* done */ }
            Variants::Single { index: _ } => match &self.fields {
                FieldsShape::Primitive => {
                    out.add_range(base_offset, self.size);
                }
                &FieldsShape::Union(field_count) => {
                    for field in 0..field_count.get() {
                        let field = self.field(cx, field);
                        field.add_data_ranges(cx, base_offset, out);
                    }
                }
                &FieldsShape::Array { stride, count } => {
                    let elem = self.field(cx, 0);

                    // For scalars we know there is no padding between the elements,
                    // so the entire array is a single big data range.
                    if elem.backend_repr.is_scalar() {
                        out.add_range(base_offset, elem.size * count);
                    } else {
                        // FIXME: this is really inefficient for large arrays.
                        for idx in 0..count {
                            elem.add_data_ranges(cx, base_offset + idx * stride, out);
                        }
                    }
                }
                FieldsShape::Arbitrary { offsets, in_memory_order: _ } => {
                    for (field, &offset) in offsets.iter_enumerated() {
                        let field = self.field(cx, field.as_usize());
                        field.add_data_ranges(cx, base_offset + offset, out);
                    }
                }
            },
            Variants::Multiple { variants, .. } => {
                for variant in variants.indices() {
                    let variant = self.for_variant(cx, variant);
                    variant.add_data_ranges(cx, base_offset, out);
                }
            }
        }
    }
}
