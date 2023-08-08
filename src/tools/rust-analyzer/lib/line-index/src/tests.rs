use crate::{LineIndex, TextSize, WideChar};

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
    multi_byte_chars: vec![(1, (13, 15)), (2, (29, 31))],
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
