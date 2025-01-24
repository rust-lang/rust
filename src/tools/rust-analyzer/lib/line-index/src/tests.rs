use crate::{LineCol, LineIndex, TextSize, WideChar, WideEncoding, WideLineCol};

macro_rules! test {
    (
        case: $test_name:ident,
     text: $text:expr,
     lines: $lines:expr,
     multi_byte_chars: $multi_byte_chars:expr,
    ) => {
        #[test]
        fn $test_name() {
            let line_index = LineIndex::new($text);

            let expected_lines: Vec<TextSize> =
                $lines.into_iter().map(<TextSize as From<u32>>::from).collect();

            assert_eq!(&*line_index.newlines, &*expected_lines);

            let expected_mbcs: Vec<_> = $multi_byte_chars
                .into_iter()
                .map(|(line, (pos, end)): (u32, (u32, u32))| {
                    (line, WideChar { start: TextSize::from(pos), end: TextSize::from(end) })
                })
                .collect();

            assert_eq!(
                line_index
                    .line_wide_chars
                    .iter()
                    .flat_map(|(line, val)| std::iter::repeat(*line).zip(val.iter().copied()))
                    .collect::<Vec<_>>(),
                expected_mbcs
            );
        }
    };
}

test!(
    case: empty_text,
    text: "",
    lines: vec![],
    multi_byte_chars: vec![],
);

test!(
    case: newlines_short,
    text: "a\nc",
    lines: vec![2],
    multi_byte_chars: vec![],
);

test!(
    case: newlines_long,
    text: "012345678\nabcdef012345678\na",
    lines: vec![10, 26],
    multi_byte_chars: vec![],
);

test!(
    case: newline_and_multi_byte_char_in_same_chunk,
    text: "01234β789\nbcdef0123456789abcdef",
    lines: vec![11],
    multi_byte_chars: vec![(0, (5, 7))],
);

test!(
    case: newline_and_control_char_in_same_chunk,
    text: "01234\u{07}6789\nbcdef0123456789abcdef",
    lines: vec![11],
    multi_byte_chars: vec![],
);

test!(
    case: multi_byte_char_short,
    text: "aβc",
    lines: vec![],
    multi_byte_chars: vec![(0, (1, 3))],
);

test!(
    case: multi_byte_char_long,
    text: "0123456789abcΔf012345β",
    lines: vec![],
    multi_byte_chars: vec![(0, (13, 15)), (0, (22, 24))],
);

test!(
    case: multi_byte_char_across_chunk_boundary,
    text: "0123456789abcdeΔ123456789abcdef01234",
    lines: vec![],
    multi_byte_chars: vec![(0, (15, 17))],
);

test!(
    case: multi_byte_char_across_chunk_boundary_tail,
    text: "0123456789abcdeΔ....",
    lines: vec![],
    multi_byte_chars: vec![(0, (15, 17))],
);

test!(
    case: multi_byte_with_new_lines,
    text: "01\t345\n789abcΔf01234567\u{07}9\nbcΔf",
    lines: vec![7, 27],
    multi_byte_chars: vec![(1, (6, 8)), (2, (2, 4))],
);

test!(
    case: trailing_newline,
    text: "0123456789\n",
    lines: vec![11],
    multi_byte_chars: vec![],
);

test!(
    case: trailing_newline_chunk_boundary,
    text: "0123456789abcde\n",
    lines: vec![16],
    multi_byte_chars: vec![],
);

#[test]
fn test_try_line_col() {
    let text = "\n\n\n\n\n宽3456";
    assert_eq!(&text[5..8], "宽");
    assert_eq!(&text[11..12], "6");
    let line_index = LineIndex::new(text);
    let before_6 = TextSize::from(11);
    let line_col = line_index.try_line_col(before_6);
    assert_eq!(line_col, Some(LineCol { line: 5, col: 6 }));
}

#[test]
fn test_to_wide() {
    let text = "\n\n\n\n\n宽3456";
    assert_eq!(&text[5..8], "宽");
    assert_eq!(&text[11..12], "6");
    let line_index = LineIndex::new(text);
    let before_6 = TextSize::from(11);
    let line_col = line_index.try_line_col(before_6);
    assert_eq!(line_col, Some(LineCol { line: 5, col: 6 }));
    let wide_line_col = line_index.to_wide(WideEncoding::Utf16, line_col.unwrap());
    assert_eq!(wide_line_col, Some(WideLineCol { line: 5, col: 4 }));
}

#[test]
fn test_every_chars() {
    let text: String = {
        let mut chars: Vec<char> = ((0 as char)..char::MAX).collect(); // Neat!
        chars.extend("\n".repeat(chars.len() / 16).chars());
        let seed = std::hash::Hasher::finish(&std::hash::BuildHasher::build_hasher(
            #[allow(clippy::disallowed_types)]
            &std::collections::hash_map::RandomState::new(),
        ));
        let mut rng = oorandom::Rand32::new(seed);
        let mut rand_index = |i| rng.rand_range(0..i as u32) as usize;
        let mut remaining = chars.len() - 1;
        while remaining > 0 {
            let index = rand_index(remaining);
            chars.swap(remaining, index);
            remaining -= 1;
        }
        chars.into_iter().collect()
    };
    assert!(text.contains('💩')); // Sanity check.

    let line_index = LineIndex::new(&text);

    let mut lin_col = LineCol { line: 0, col: 0 };
    let mut col_utf16 = 0;
    let mut col_utf32 = 0;
    for (offset, c) in text.char_indices() {
        let got_offset = line_index.offset(lin_col).unwrap();
        assert_eq!(usize::from(got_offset), offset);

        let got_lin_col = line_index.line_col(got_offset);
        assert_eq!(got_lin_col, lin_col);

        for (enc, col) in [(WideEncoding::Utf16, col_utf16), (WideEncoding::Utf32, col_utf32)] {
            let wide_lin_col = line_index.to_wide(enc, lin_col).unwrap();
            let got_lin_col = line_index.to_utf8(enc, wide_lin_col).unwrap();
            assert_eq!(got_lin_col, lin_col);
            assert_eq!(wide_lin_col.col, col)
        }

        if c == '\n' {
            lin_col.line += 1;
            lin_col.col = 0;
            col_utf16 = 0;
            col_utf32 = 0;
        } else {
            lin_col.col += c.len_utf8() as u32;
            col_utf16 += c.len_utf16() as u32;
            col_utf32 += 1;
        }
    }
}

#[test]
fn test_line() {
    use text_size::TextRange;

    macro_rules! validate {
        ($text:expr, $line:expr, $expected_start:literal .. $expected_end:literal) => {
            let line_index = LineIndex::new($text);
            assert_eq!(
                line_index.line($line),
                Some(TextRange::new(
                    TextSize::from($expected_start),
                    TextSize::from($expected_end)
                ))
            );
        };
    }

    validate!("", 0, 0..0);
    validate!("\n", 1, 1..1);
    validate!("\nabc", 1, 1..4);
    validate!("\nabc\ndef", 1, 1..5);
}
