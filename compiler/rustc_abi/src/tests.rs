use super::*;

#[test]
fn align_constants() {
    assert_eq!(Align::ONE, Align::from_bytes(1).unwrap());
    assert_eq!(Align::EIGHT, Align::from_bytes(8).unwrap());
}

#[test]
fn wrapping_range_contains_range() {
    let size16 = Size::from_bytes(16);

    let a = WrappingRange { start: 10, end: 20 };
    assert!(a.contains_range(a, size16));
    assert!(a.contains_range(WrappingRange { start: 11, end: 19 }, size16));
    assert!(a.contains_range(WrappingRange { start: 10, end: 10 }, size16));
    assert!(a.contains_range(WrappingRange { start: 20, end: 20 }, size16));
    assert!(!a.contains_range(WrappingRange { start: 10, end: 21 }, size16));
    assert!(!a.contains_range(WrappingRange { start: 9, end: 20 }, size16));
    assert!(!a.contains_range(WrappingRange { start: 4, end: 6 }, size16));
    assert!(!a.contains_range(WrappingRange { start: 24, end: 26 }, size16));

    assert!(!a.contains_range(WrappingRange { start: 16, end: 14 }, size16));

    let b = WrappingRange { start: 20, end: 10 };
    assert!(b.contains_range(b, size16));
    assert!(b.contains_range(WrappingRange { start: 20, end: 20 }, size16));
    assert!(b.contains_range(WrappingRange { start: 10, end: 10 }, size16));
    assert!(b.contains_range(WrappingRange { start: 0, end: 10 }, size16));
    assert!(b.contains_range(WrappingRange { start: 20, end: 30 }, size16));
    assert!(b.contains_range(WrappingRange { start: 20, end: 9 }, size16));
    assert!(b.contains_range(WrappingRange { start: 21, end: 10 }, size16));
    assert!(b.contains_range(WrappingRange { start: 999, end: 9999 }, size16));
    assert!(b.contains_range(WrappingRange { start: 999, end: 9 }, size16));
    assert!(!b.contains_range(WrappingRange { start: 19, end: 19 }, size16));
    assert!(!b.contains_range(WrappingRange { start: 11, end: 11 }, size16));
    assert!(!b.contains_range(WrappingRange { start: 19, end: 11 }, size16));
    assert!(!b.contains_range(WrappingRange { start: 11, end: 19 }, size16));

    let f = WrappingRange { start: 0, end: u128::MAX };
    assert!(f.contains_range(WrappingRange { start: 10, end: 20 }, size16));
    assert!(f.contains_range(WrappingRange { start: 20, end: 10 }, size16));

    let g = WrappingRange { start: 2, end: 1 };
    assert!(g.contains_range(WrappingRange { start: 10, end: 20 }, size16));
    assert!(g.contains_range(WrappingRange { start: 20, end: 10 }, size16));

    let size1 = Size::from_bytes(1);
    let u8r = WrappingRange { start: 0, end: 255 };
    let i8r = WrappingRange { start: 128, end: 127 };
    assert!(u8r.contains_range(i8r, size1));
    assert!(i8r.contains_range(u8r, size1));
    assert!(!u8r.contains_range(i8r, size16));
    assert!(i8r.contains_range(u8r, size16));

    let boolr = WrappingRange { start: 0, end: 1 };
    assert!(u8r.contains_range(boolr, size1));
    assert!(i8r.contains_range(boolr, size1));
    assert!(!boolr.contains_range(u8r, size1));
    assert!(!boolr.contains_range(i8r, size1));

    let cmpr = WrappingRange { start: 255, end: 1 };
    assert!(u8r.contains_range(cmpr, size1));
    assert!(i8r.contains_range(cmpr, size1));
    assert!(!cmpr.contains_range(u8r, size1));
    assert!(!cmpr.contains_range(i8r, size1));

    assert!(!boolr.contains_range(cmpr, size1));
    assert!(cmpr.contains_range(boolr, size1));
}

#[test]
fn embedded_payload_niche_layout() {
    let dl = TargetDataLayout::default();
    let cx = LayoutCalculator::new(&dl);

    let terminate = LayoutData::scalar(
        &dl,
        Scalar::Initialized {
            value: Primitive::Int(Integer::I8, false),
            valid_range: WrappingRange { start: 0, end: 1 },
        },
    );
    let cleanup = LayoutData::scalar(
        &dl,
        Scalar::Initialized {
            value: Primitive::Int(Integer::I32, false),
            valid_range: WrappingRange { start: 0, end: 0xFFFF_FF00 },
        },
    );

    #[derive(Copy, Clone)]
    struct Field<'a>(&'a LayoutData<FieldIdx, VariantIdx>);

    impl<'a> std::ops::Deref for Field<'a> {
        type Target = &'a LayoutData<FieldIdx, VariantIdx>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<'a> std::fmt::Debug for Field<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Field")
                .field("size", &self.0.size.bytes())
                .field("align", &self.0.align.abi.bytes())
                .finish()
        }
    }

    let unit: IndexVec<FieldIdx, Field<'_>> = IndexVec::new();
    let variants = IndexVec::from_raw(vec![
        unit.clone(),
        unit,
        IndexVec::from_raw(vec![Field(&terminate)]),
        IndexVec::from_raw(vec![Field(&cleanup)]),
    ]);

    let layout = match cx.layout_of_struct_or_enum(
        &ReprOptions::default(),
        &variants,
        true,
        false,
        |_, _| (Integer::I32, false),
        (0..4).map(|i| (VariantIdx::new(i), i as u128)),
        true,
    ) {
        Ok(layout) => layout,
        Err(_) => panic!("layout calculation failed"),
    };

    assert_eq!(layout.size.bytes(), 4);
    assert_eq!(layout.align, AbiAlign::new(Align::from_bytes(4).unwrap()));

    match layout.variants {
        Variants::Multiple { tag, tag_encoding, .. } => {
            match tag {
                Scalar::Initialized { value, valid_range } => {
                    assert!(matches!(value, Primitive::Int(Integer::I32, false)));
                    assert_eq!((valid_range.start, valid_range.end), (0, 0xFFFF_FF04 as u128));
                }
                _ => panic!("expected initialized tag"),
            }
            match tag_encoding {
                TagEncoding::Niche {
                    untagged_variant,
                    niche_start,
                    niche_variants,
                    embedded_payload,
                } => {
                    assert_eq!(untagged_variant, VariantIdx::new(3));
                    assert_eq!(niche_start, 0xFFFF_FF01);
                    assert_eq!(niche_variants, (VariantIdx::new(0)..=VariantIdx::new(2)).into());
                    assert_eq!(embedded_payload, Some((VariantIdx::new(2), 2)));
                }
                _ => panic!("expected niche encoding"),
            }
        }
        _ => panic!("expected multiple variants"),
    }
}
