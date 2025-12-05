use std::num::NonZero;

use rustc_hashes::Hash64;
use rustc_index::{Idx, IndexVec};

use crate::{
    AbiAlign, BackendRepr, FieldsShape, HasDataLayout, LayoutData, Niche, Primitive, Scalar, Size,
    Variants,
};

/// "Simple" layout constructors that cannot fail.
impl<FieldIdx: Idx, VariantIdx: Idx> LayoutData<FieldIdx, VariantIdx> {
    pub fn unit<C: HasDataLayout>(cx: &C, sized: bool) -> Self {
        let dl = cx.data_layout();
        LayoutData {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Arbitrary {
                offsets: IndexVec::new(),
                memory_index: IndexVec::new(),
            },
            backend_repr: BackendRepr::Memory { sized },
            largest_niche: None,
            uninhabited: false,
            align: AbiAlign::new(dl.i8_align),
            size: Size::ZERO,
            max_repr_align: None,
            unadjusted_abi_align: dl.i8_align,
            randomization_seed: Hash64::new(0),
        }
    }

    pub fn never_type<C: HasDataLayout>(cx: &C) -> Self {
        let dl = cx.data_layout();
        // This is also used for uninhabited enums, so we use `Variants::Empty`.
        LayoutData {
            variants: Variants::Empty,
            fields: FieldsShape::Primitive,
            backend_repr: BackendRepr::Memory { sized: true },
            largest_niche: None,
            uninhabited: true,
            align: AbiAlign::new(dl.i8_align),
            size: Size::ZERO,
            max_repr_align: None,
            unadjusted_abi_align: dl.i8_align,
            randomization_seed: Hash64::ZERO,
        }
    }

    pub fn scalar<C: HasDataLayout>(cx: &C, scalar: Scalar) -> Self {
        let largest_niche = Niche::from_scalar(cx, Size::ZERO, scalar);
        let size = scalar.size(cx);
        let align = scalar.align(cx);

        let range = scalar.valid_range(cx);

        // All primitive types for which we don't have subtype coercions should get a distinct seed,
        // so that types wrapping them can use randomization to arrive at distinct layouts.
        //
        // Some type information is already lost at this point, so as an approximation we derive
        // the seed from what remains. For example on 64-bit targets usize and u64 can no longer
        // be distinguished.
        let randomization_seed = size
            .bytes()
            .wrapping_add(
                match scalar.primitive() {
                    Primitive::Int(_, true) => 1,
                    Primitive::Int(_, false) => 2,
                    Primitive::Float(_) => 3,
                    Primitive::Pointer(_) => 4,
                } << 32,
            )
            // distinguishes references from pointers
            .wrapping_add((range.start as u64).rotate_right(16))
            // distinguishes char from u32 and bool from u8
            .wrapping_add((range.end as u64).rotate_right(16));

        LayoutData {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Primitive,
            backend_repr: BackendRepr::Scalar(scalar),
            largest_niche,
            uninhabited: false,
            size,
            align,
            max_repr_align: None,
            unadjusted_abi_align: align.abi,
            randomization_seed: Hash64::new(randomization_seed),
        }
    }

    pub fn scalar_pair<C: HasDataLayout>(cx: &C, a: Scalar, b: Scalar) -> Self {
        let dl = cx.data_layout();
        let b_align = b.align(dl).abi;
        let align = a.align(dl).abi.max(b_align).max(dl.aggregate_align);
        let b_offset = a.size(dl).align_to(b_align);
        let size = (b_offset + b.size(dl)).align_to(align);

        // HACK(nox): We iter on `b` and then `a` because `max_by_key`
        // returns the last maximum.
        let largest_niche = Niche::from_scalar(dl, b_offset, b)
            .into_iter()
            .chain(Niche::from_scalar(dl, Size::ZERO, a))
            .max_by_key(|niche| niche.available(dl));

        let combined_seed = a.size(dl).bytes().wrapping_add(b.size(dl).bytes());

        LayoutData {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Arbitrary {
                offsets: [Size::ZERO, b_offset].into(),
                memory_index: [0, 1].into(),
            },
            backend_repr: BackendRepr::ScalarPair(a, b),
            largest_niche,
            uninhabited: false,
            align: AbiAlign::new(align),
            size,
            max_repr_align: None,
            unadjusted_abi_align: align,
            randomization_seed: Hash64::new(combined_seed),
        }
    }

    /// Returns a dummy layout for an uninhabited variant.
    ///
    /// Uninhabited variants get pruned as part of the layout calculation,
    /// so this can be used after the fact to reconstitute a layout.
    pub fn uninhabited_variant<C: HasDataLayout>(cx: &C, index: VariantIdx, fields: usize) -> Self {
        let dl = cx.data_layout();
        LayoutData {
            variants: Variants::Single { index },
            fields: match NonZero::new(fields) {
                Some(fields) => FieldsShape::Union(fields),
                None => FieldsShape::Arbitrary {
                    offsets: IndexVec::new(),
                    memory_index: IndexVec::new(),
                },
            },
            backend_repr: BackendRepr::Memory { sized: true },
            largest_niche: None,
            uninhabited: true,
            align: AbiAlign::new(dl.i8_align),
            size: Size::ZERO,
            max_repr_align: None,
            unadjusted_abi_align: dl.i8_align,
            randomization_seed: Hash64::ZERO,
        }
    }
}
