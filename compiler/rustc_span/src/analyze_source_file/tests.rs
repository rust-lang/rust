use super::*;

macro_rules! test {
    (case: $test_name:ident,
     text: $text:expr,
     source_file_start_pos: $source_file_start_pos:expr,
     lines: $lines:expr,
     multi_byte_chars: $multi_byte_chars:expr,
     non_narrow_chars: $non_narrow_chars:expr,) => {
        #[test]
        fn $test_name() {
            let (lines, multi_byte_chars, non_narrow_chars) =
                analyze_source_file($text, BytePos($source_file_start_pos));

            let expected_lines: Vec<BytePos> = $lines.into_iter().map(BytePos).collect();

            assert_eq!(lines, expected_lines);

            let expected_mbcs: Vec<MultiByteChar> = $multi_byte_chars
                .into_iter()
                .map(|(pos, bytes)| MultiByteChar { pos: BytePos(pos), bytes })
                .collect();

            assert_eq!(multi_byte_chars, expected_mbcs);

            let expected_nncs: Vec<NonNarrowChar> = $non_narrow_chars
                .into_iter()
                .map(|(pos, width)| NonNarrowChar::new(BytePos(pos), width))
                .collect();

            assert_eq!(non_narrow_chars, expected_nncs);
        }
    };
}

test!(
    case: empty_text,
    text: "",
    source_file_start_pos: 0,
    lines: vec![],
    multi_byte_chars: vec![],
    non_narrow_chars: vec![],
);

test!(
    case: newlines_short,
    text: "a\nc",
    source_file_start_pos: 0,
    lines: vec![0, 2],
    multi_byte_chars: vec![],
    non_narrow_chars: vec![],
);

test!(
    case: newlines_long,
    text: "012345678\nabcdef012345678\na",
    source_file_start_pos: 0,
    lines: vec![0, 10, 26],
    multi_byte_chars: vec![],
    non_narrow_chars: vec![],
);

test!(
    case: newline_and_multi_byte_char_in_same_chunk,
    text: "01234β789\nbcdef0123456789abcdef",
    source_file_start_pos: 0,
    lines: vec![0, 11],
    multi_byte_chars: vec![(5, 2)],
    non_narrow_chars: vec![],
);

test!(
    case: newline_and_control_char_in_same_chunk,
    text: "01234\u{07}6789\nbcdef0123456789abcdef",
    source_file_start_pos: 0,
    lines: vec![0, 11],
    multi_byte_chars: vec![],
    non_narrow_chars: vec![(5, 0)],
);

test!(
    case: multi_byte_char_short,
    text: "aβc",
    source_file_start_pos: 0,
    lines: vec![0],
    multi_byte_chars: vec![(1, 2)],
    non_narrow_chars: vec![],
);

test!(
    case: multi_byte_char_long,
    text: "0123456789abcΔf012345β",
    source_file_start_pos: 0,
    lines: vec![0],
    multi_byte_chars: vec![(13, 2), (22, 2)],
    non_narrow_chars: vec![],
);

test!(
    case: multi_byte_char_across_chunk_boundary,
    text: "0123456789abcdeΔ123456789abcdef01234",
    source_file_start_pos: 0,
    lines: vec![0],
    multi_byte_chars: vec![(15, 2)],
    non_narrow_chars: vec![],
);

test!(
    case: multi_byte_char_across_chunk_boundary_tail,
    text: "0123456789abcdeΔ....",
    source_file_start_pos: 0,
    lines: vec![0],
    multi_byte_chars: vec![(15, 2)],
    non_narrow_chars: vec![],
);

test!(
    case: non_narrow_short,
    text: "0\t2",
    source_file_start_pos: 0,
    lines: vec![0],
    multi_byte_chars: vec![],
    non_narrow_chars: vec![(1, 4)],
);

test!(
    case: non_narrow_long,
    text: "01\t3456789abcdef01234567\u{07}9",
    source_file_start_pos: 0,
    lines: vec![0],
    multi_byte_chars: vec![],
    non_narrow_chars: vec![(2, 4), (24, 0)],
);

test!(
    case: output_offset_all,
    text: "01\t345\n789abcΔf01234567\u{07}9\nbcΔf",
    source_file_start_pos: 1000,
    lines: vec![0 + 1000, 7 + 1000, 27 + 1000],
    multi_byte_chars: vec![(13 + 1000, 2), (29 + 1000, 2)],
    non_narrow_chars: vec![(2 + 1000, 4), (24 + 1000, 0)],
);
