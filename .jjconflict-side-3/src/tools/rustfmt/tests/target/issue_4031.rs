fn foo() {
    with_woff2_glyf_table("tests/fonts/woff2/SFNT-TTF-Composite.woff2", |glyf| {
        let actual = glyf
            .records
            .iter()
            .map(|glyph| match glyph {
                GlyfRecord::Parsed(
                    found @ Glyph {
                        data: GlyphData::Composite { .. },
                        ..
                    },
                ) => Some(found),
                _ => None,
            })
            .find(|candidate| candidate.is_some())
            .unwrap()
            .unwrap();

        assert_eq!(*actual, expected)
    });
}
