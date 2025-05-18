use std::assert_matches::assert_matches;

use rustc_abi::{BackendRepr, FieldsShape, Scalar, Size, TagEncoding, Variants};
use rustc_middle::bug;
use rustc_middle::ty::layout::{HasTyCtxt, LayoutCx, TyAndLayout};

/// Enforce some basic invariants on layouts.
pub(super) fn layout_sanity_check<'tcx>(cx: &LayoutCx<'tcx>, layout: &TyAndLayout<'tcx>) {
    let tcx = cx.tcx();

    if layout.size.bytes() % layout.align.abi.bytes() != 0 {
        bug!("size is not a multiple of align, in the following layout:\n{layout:#?}");
    }
    if layout.size.bytes() >= tcx.data_layout.obj_size_bound() {
        bug!("size is too large, in the following layout:\n{layout:#?}");
    }

    if !cfg!(debug_assertions) {
        // Stop here, the rest is kind of expensive.
        return;
    }

    // Type-level uninhabitedness should always imply ABI uninhabitedness. This can be expensive on
    // big non-exhaustive types, and is [hard to
    // fix](https://github.com/rust-lang/rust/issues/141006#issuecomment-2883415000) in general.
    // Only doing this sanity check when debug assertions are turned on avoids the issue for the
    // very specific case of #140944.
    if layout.ty.is_privately_uninhabited(tcx, cx.typing_env) {
        assert!(
            layout.is_uninhabited(),
            "{:?} is type-level uninhabited but not ABI-uninhabited?",
            layout.ty
        );
    }

    /// Yields non-ZST fields of the type
    fn non_zst_fields<'tcx, 'a>(
        cx: &'a LayoutCx<'tcx>,
        layout: &'a TyAndLayout<'tcx>,
    ) -> impl Iterator<Item = (Size, TyAndLayout<'tcx>)> {
        (0..layout.layout.fields().count()).filter_map(|i| {
            let field = layout.field(cx, i);
            // Also checking `align == 1` here leads to test failures in
            // `layout/zero-sized-array-union.rs`, where a type has a zero-size field with
            // alignment 4 that still gets ignored during layout computation (which is okay
            // since other fields already force alignment 4).
            let zst = field.is_zst();
            (!zst).then(|| (layout.fields.offset(i), field))
        })
    }

    fn skip_newtypes<'tcx>(cx: &LayoutCx<'tcx>, layout: &TyAndLayout<'tcx>) -> TyAndLayout<'tcx> {
        if matches!(layout.layout.variants(), Variants::Multiple { .. }) {
            // Definitely not a newtype of anything.
            return *layout;
        }
        let mut fields = non_zst_fields(cx, layout);
        let Some(first) = fields.next() else {
            // No fields here, so this could be a primitive or enum -- either way it's not a newtype around a thing
            return *layout;
        };
        if fields.next().is_none() {
            let (offset, first) = first;
            if offset == Size::ZERO && first.layout.size() == layout.size {
                // This is a newtype, so keep recursing.
                // FIXME(RalfJung): I don't think it would be correct to do any checks for
                // alignment here, so we don't. Is that correct?
                return skip_newtypes(cx, &first);
            }
        }
        // No more newtypes here.
        *layout
    }

    fn check_layout_abi<'tcx>(cx: &LayoutCx<'tcx>, layout: &TyAndLayout<'tcx>) {
        // Verify the ABI-mandated alignment and size for scalars.
        let align = layout.backend_repr.scalar_align(cx);
        let size = layout.backend_repr.scalar_size(cx);
        if let Some(align) = align {
            assert_eq!(
                layout.layout.align().abi,
                align,
                "alignment mismatch between ABI and layout in {layout:#?}"
            );
        }
        if let Some(size) = size {
            assert_eq!(
                layout.layout.size(),
                size,
                "size mismatch between ABI and layout in {layout:#?}"
            );
        }

        // Verify per-ABI invariants
        match layout.layout.backend_repr() {
            BackendRepr::Scalar(_) => {
                // These must always be present for `Scalar` types.
                let align = align.unwrap();
                let size = size.unwrap();
                // Check that this matches the underlying field.
                let inner = skip_newtypes(cx, layout);
                assert!(
                    matches!(inner.layout.backend_repr(), BackendRepr::Scalar(_)),
                    "`Scalar` type {} is newtype around non-`Scalar` type {}",
                    layout.ty,
                    inner.ty
                );
                match inner.layout.fields() {
                    FieldsShape::Primitive => {
                        // Fine.
                    }
                    FieldsShape::Union(..) => {
                        // FIXME: I guess we could also check something here? Like, look at all fields?
                        return;
                    }
                    FieldsShape::Arbitrary { .. } => {
                        // Should be an enum, the only field is the discriminant.
                        assert!(
                            inner.ty.is_enum(),
                            "`Scalar` layout for non-primitive non-enum type {}",
                            inner.ty
                        );
                        assert_eq!(
                            inner.layout.fields().count(),
                            1,
                            "`Scalar` layout for multiple-field type in {inner:#?}",
                        );
                        let offset = inner.layout.fields().offset(0);
                        let field = inner.field(cx, 0);
                        // The field should be at the right offset, and match the `scalar` layout.
                        assert_eq!(
                            offset,
                            Size::ZERO,
                            "`Scalar` field at non-0 offset in {inner:#?}",
                        );
                        assert_eq!(field.size, size, "`Scalar` field with bad size in {inner:#?}",);
                        assert_eq!(
                            field.align.abi, align,
                            "`Scalar` field with bad align in {inner:#?}",
                        );
                        assert!(
                            matches!(field.backend_repr, BackendRepr::Scalar(_)),
                            "`Scalar` field with bad ABI in {inner:#?}",
                        );
                    }
                    _ => {
                        panic!("`Scalar` layout for non-primitive non-enum type {}", inner.ty);
                    }
                }
            }
            BackendRepr::ScalarPair(scalar1, scalar2) => {
                // Check that the underlying pair of fields matches.
                let inner = skip_newtypes(cx, layout);
                assert!(
                    matches!(inner.layout.backend_repr(), BackendRepr::ScalarPair(..)),
                    "`ScalarPair` type {} is newtype around non-`ScalarPair` type {}",
                    layout.ty,
                    inner.ty
                );
                if matches!(inner.layout.variants(), Variants::Multiple { .. }) {
                    // FIXME: ScalarPair for enums is enormously complicated and it is very hard
                    // to check anything about them.
                    return;
                }
                match inner.layout.fields() {
                    FieldsShape::Arbitrary { .. } => {
                        // Checked below.
                    }
                    FieldsShape::Union(..) => {
                        // FIXME: I guess we could also check something here? Like, look at all fields?
                        return;
                    }
                    _ => {
                        panic!("`ScalarPair` layout with unexpected field shape in {inner:#?}");
                    }
                }
                let mut fields = non_zst_fields(cx, &inner);
                let (offset1, field1) = fields.next().unwrap_or_else(|| {
                    panic!(
                        "`ScalarPair` layout for type with not even one non-ZST field: {inner:#?}"
                    )
                });
                let (offset2, field2) = fields.next().unwrap_or_else(|| {
                    panic!(
                        "`ScalarPair` layout for type with less than two non-ZST fields: {inner:#?}"
                    )
                });
                assert_matches!(
                    fields.next(),
                    None,
                    "`ScalarPair` layout for type with at least three non-ZST fields: {inner:#?}"
                );
                // The fields might be in opposite order.
                let (offset1, field1, offset2, field2) = if offset1 <= offset2 {
                    (offset1, field1, offset2, field2)
                } else {
                    (offset2, field2, offset1, field1)
                };
                // The fields should be at the right offset, and match the `scalar` layout.
                let size1 = scalar1.size(cx);
                let align1 = scalar1.align(cx).abi;
                let size2 = scalar2.size(cx);
                let align2 = scalar2.align(cx).abi;
                assert_eq!(
                    offset1,
                    Size::ZERO,
                    "`ScalarPair` first field at non-0 offset in {inner:#?}",
                );
                assert_eq!(
                    field1.size, size1,
                    "`ScalarPair` first field with bad size in {inner:#?}",
                );
                assert_eq!(
                    field1.align.abi, align1,
                    "`ScalarPair` first field with bad align in {inner:#?}",
                );
                assert_matches!(
                    field1.backend_repr,
                    BackendRepr::Scalar(_),
                    "`ScalarPair` first field with bad ABI in {inner:#?}",
                );
                let field2_offset = size1.align_to(align2);
                assert_eq!(
                    offset2, field2_offset,
                    "`ScalarPair` second field at bad offset in {inner:#?}",
                );
                assert_eq!(
                    field2.size, size2,
                    "`ScalarPair` second field with bad size in {inner:#?}",
                );
                assert_eq!(
                    field2.align.abi, align2,
                    "`ScalarPair` second field with bad align in {inner:#?}",
                );
                assert_matches!(
                    field2.backend_repr,
                    BackendRepr::Scalar(_),
                    "`ScalarPair` second field with bad ABI in {inner:#?}",
                );
            }
            BackendRepr::SimdVector { element, count } => {
                let align = layout.align.abi;
                let size = layout.size;
                let element_align = element.align(cx).abi;
                let element_size = element.size(cx);
                // Currently, vectors must always be aligned to at least their elements:
                assert!(align >= element_align);
                // And the size has to be element * count plus alignment padding, of course
                assert!(size == (element_size * count).align_to(align));
            }
            BackendRepr::Memory { .. } => {} // Nothing to check.
        }
    }

    check_layout_abi(cx, layout);

    match &layout.variants {
        Variants::Empty => {
            assert!(layout.is_uninhabited());
        }
        Variants::Single { index } => {
            if let Some(variants) = layout.ty.variant_range(tcx) {
                assert!(variants.contains(index));
            } else {
                // Types without variants use `0` as dummy variant index.
                assert!(index.as_u32() == 0);
            }
        }
        Variants::Multiple { variants, tag, tag_encoding, .. } => {
            if let TagEncoding::Niche { niche_start, untagged_variant, niche_variants } =
                tag_encoding
            {
                let niche_size = tag.size(cx);
                assert!(*niche_start <= niche_size.unsigned_int_max());
                for (idx, variant) in variants.iter_enumerated() {
                    // Ensure all inhabited variants are accounted for.
                    if !variant.is_uninhabited() {
                        assert!(idx == *untagged_variant || niche_variants.contains(&idx));
                    }
                }
            }
            for variant in variants.iter() {
                // No nested "multiple".
                assert_matches!(variant.variants, Variants::Single { .. });
                // Variants should have the same or a smaller size as the full thing,
                // and same for alignment.
                if variant.size > layout.size {
                    bug!(
                        "Type with size {} bytes has variant with size {} bytes: {layout:#?}",
                        layout.size.bytes(),
                        variant.size.bytes(),
                    )
                }
                if variant.align.abi > layout.align.abi {
                    bug!(
                        "Type with alignment {} bytes has variant with alignment {} bytes: {layout:#?}",
                        layout.align.abi.bytes(),
                        variant.align.abi.bytes(),
                    )
                }
                // Skip empty variants.
                if variant.size == Size::ZERO
                    || variant.fields.count() == 0
                    || variant.is_uninhabited()
                {
                    // These are never actually accessed anyway, so we can skip the coherence check
                    // for them. They also fail that check, since they may have
                    // a different ABI even when the main type is
                    // `Scalar`/`ScalarPair`. (Note that sometimes, variants with fields have size
                    // 0, and sometimes, variants without fields have non-0 size.)
                    continue;
                }
                // The top-level ABI and the ABI of the variants should be coherent.
                let scalar_coherent = |s1: Scalar, s2: Scalar| {
                    s1.size(cx) == s2.size(cx) && s1.align(cx) == s2.align(cx)
                };
                let abi_coherent = match (layout.backend_repr, variant.backend_repr) {
                    (BackendRepr::Scalar(s1), BackendRepr::Scalar(s2)) => scalar_coherent(s1, s2),
                    (BackendRepr::ScalarPair(a1, b1), BackendRepr::ScalarPair(a2, b2)) => {
                        scalar_coherent(a1, a2) && scalar_coherent(b1, b2)
                    }
                    (BackendRepr::Memory { .. }, _) => true,
                    _ => false,
                };
                if !abi_coherent {
                    bug!(
                        "Variant ABI is incompatible with top-level ABI:\nvariant={:#?}\nTop-level: {layout:#?}",
                        variant
                    );
                }
            }
        }
    }
}
