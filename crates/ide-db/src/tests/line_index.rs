use line_index::{LineCol, LineIndex, WideEncoding};
use test_utils::skip_slow_tests;

#[test]
fn test_every_chars() {
    if skip_slow_tests() {
        return;
    }

    let text: String = {
        let mut chars: Vec<char> = ((0 as char)..char::MAX).collect(); // Neat!
        chars.extend("\n".repeat(chars.len() / 16).chars());
        let mut rng = oorandom::Rand32::new(stdx::rand::seed());
        stdx::rand::shuffle(&mut chars, |i| rng.rand_range(0..i as u32) as usize);
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
