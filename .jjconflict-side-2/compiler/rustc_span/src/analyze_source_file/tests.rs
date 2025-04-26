use super::*;

macro_rules! test {
    (case: $test_name:ident,
     text: $text:expr,
     lines: $lines:expr,
     multi_byte_chars: $multi_byte_chars:expr,) => {
        #[test]
        fn $test_name() {
            let (lines, multi_byte_chars) = analyze_source_file($text);

            let expected_lines: Vec<RelativeBytePos> =
                $lines.into_iter().map(RelativeBytePos).collect();

            assert_eq!(lines, expected_lines);

            let expected_mbcs: Vec<MultiByteChar> = $multi_byte_chars
                .into_iter()
                .map(|(pos, bytes)| MultiByteChar { pos: RelativeBytePos(pos), bytes })
                .collect();

            assert_eq!(multi_byte_chars, expected_mbcs);
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
    lines: vec![0, 2],
    multi_byte_chars: vec![],
);

test!(
    case: newlines_long,
    text: "012345678\nabcdef012345678\na",
    lines: vec![0, 10, 26],
    multi_byte_chars: vec![],
);

test!(
    case: newline_and_multi_byte_char_in_same_chunk,
    text: "01234β789\nbcdef0123456789abcdef",
    lines: vec![0, 11],
    multi_byte_chars: vec![(5, 2)],
);

test!(
    case: newline_and_control_char_in_same_chunk,
    text: "01234\u{07}6789\nbcdef0123456789abcdef",
    lines: vec![0, 11],
    multi_byte_chars: vec![],
);

test!(
    case: multi_byte_char_short,
    text: "aβc",
    lines: vec![0],
    multi_byte_chars: vec![(1, 2)],
);

test!(
    case: multi_byte_char_long,
    text: "0123456789abcΔf012345β",
    lines: vec![0],
    multi_byte_chars: vec![(13, 2), (22, 2)],
);

test!(
    case: multi_byte_char_across_chunk_boundary,
    text: "0123456789abcdeΔ123456789abcdef01234",
    lines: vec![0],
    multi_byte_chars: vec![(15, 2)],
);

test!(
    case: multi_byte_char_across_chunk_boundary_tail,
    text: "0123456789abcdeΔ....",
    lines: vec![0],
    multi_byte_chars: vec![(15, 2)],
);

test!(
    case: non_narrow_short,
    text: "0\t2",
    lines: vec![0],
    multi_byte_chars: vec![],
);

test!(
    case: non_narrow_long,
    text: "01\t3456789abcdef01234567\u{07}9",
    lines: vec![0],
    multi_byte_chars: vec![],
);

test!(
    case: output_offset_all,
    text: "01\t345\n789abcΔf01234567\u{07}9\nbcΔf",
    lines: vec![0, 7, 27],
    multi_byte_chars: vec![(13, 2), (29, 2)],
);
