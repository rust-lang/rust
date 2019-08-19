use super::*;

use rustc_data_structures::sync::Lrc;

fn init_source_map() -> SourceMap {
    let sm = SourceMap::new(FilePathMapping::empty());
    sm.new_source_file(PathBuf::from("blork.rs").into(),
                    "first line.\nsecond line".to_string());
    sm.new_source_file(PathBuf::from("empty.rs").into(),
                    String::new());
    sm.new_source_file(PathBuf::from("blork2.rs").into(),
                    "first line.\nsecond line".to_string());
    sm
}

#[test]
fn t3() {
    // Test lookup_byte_offset
    let sm = init_source_map();

    let srcfbp1 = sm.lookup_byte_offset(BytePos(23));
    assert_eq!(srcfbp1.sf.name, PathBuf::from("blork.rs").into());
    assert_eq!(srcfbp1.pos, BytePos(23));

    let srcfbp1 = sm.lookup_byte_offset(BytePos(24));
    assert_eq!(srcfbp1.sf.name, PathBuf::from("empty.rs").into());
    assert_eq!(srcfbp1.pos, BytePos(0));

    let srcfbp2 = sm.lookup_byte_offset(BytePos(25));
    assert_eq!(srcfbp2.sf.name, PathBuf::from("blork2.rs").into());
    assert_eq!(srcfbp2.pos, BytePos(0));
}

#[test]
fn t4() {
    // Test bytepos_to_file_charpos
    let sm = init_source_map();

    let cp1 = sm.bytepos_to_file_charpos(BytePos(22));
    assert_eq!(cp1, CharPos(22));

    let cp2 = sm.bytepos_to_file_charpos(BytePos(25));
    assert_eq!(cp2, CharPos(0));
}

#[test]
fn t5() {
    // Test zero-length source_files.
    let sm = init_source_map();

    let loc1 = sm.lookup_char_pos(BytePos(22));
    assert_eq!(loc1.file.name, PathBuf::from("blork.rs").into());
    assert_eq!(loc1.line, 2);
    assert_eq!(loc1.col, CharPos(10));

    let loc2 = sm.lookup_char_pos(BytePos(25));
    assert_eq!(loc2.file.name, PathBuf::from("blork2.rs").into());
    assert_eq!(loc2.line, 1);
    assert_eq!(loc2.col, CharPos(0));
}

fn init_source_map_mbc() -> SourceMap {
    let sm = SourceMap::new(FilePathMapping::empty());
    // € is a three byte utf8 char.
    sm.new_source_file(PathBuf::from("blork.rs").into(),
                    "fir€st €€€€ line.\nsecond line".to_string());
    sm.new_source_file(PathBuf::from("blork2.rs").into(),
                    "first line€€.\n€ second line".to_string());
    sm
}

#[test]
fn t6() {
    // Test bytepos_to_file_charpos in the presence of multi-byte chars
    let sm = init_source_map_mbc();

    let cp1 = sm.bytepos_to_file_charpos(BytePos(3));
    assert_eq!(cp1, CharPos(3));

    let cp2 = sm.bytepos_to_file_charpos(BytePos(6));
    assert_eq!(cp2, CharPos(4));

    let cp3 = sm.bytepos_to_file_charpos(BytePos(56));
    assert_eq!(cp3, CharPos(12));

    let cp4 = sm.bytepos_to_file_charpos(BytePos(61));
    assert_eq!(cp4, CharPos(15));
}

#[test]
fn t7() {
    // Test span_to_lines for a span ending at the end of source_file
    let sm = init_source_map();
    let span = Span::with_root_ctxt(BytePos(12), BytePos(23));
    let file_lines = sm.span_to_lines(span).unwrap();

    assert_eq!(file_lines.file.name, PathBuf::from("blork.rs").into());
    assert_eq!(file_lines.lines.len(), 1);
    assert_eq!(file_lines.lines[0].line_index, 1);
}

/// Given a string like " ~~~~~~~~~~~~ ", produces a span
/// converting that range. The idea is that the string has the same
/// length as the input, and we uncover the byte positions. Note
/// that this can span lines and so on.
fn span_from_selection(input: &str, selection: &str) -> Span {
    assert_eq!(input.len(), selection.len());
    let left_index = selection.find('~').unwrap() as u32;
    let right_index = selection.rfind('~').map(|x|x as u32).unwrap_or(left_index);
    Span::with_root_ctxt(BytePos(left_index), BytePos(right_index + 1))
}

/// Tests span_to_snippet and span_to_lines for a span converting 3
/// lines in the middle of a file.
#[test]
fn span_to_snippet_and_lines_spanning_multiple_lines() {
    let sm = SourceMap::new(FilePathMapping::empty());
    let inputtext = "aaaaa\nbbbbBB\nCCC\nDDDDDddddd\neee\n";
    let selection = "     \n    ~~\n~~~\n~~~~~     \n   \n";
    sm.new_source_file(Path::new("blork.rs").to_owned().into(), inputtext.to_string());
    let span = span_from_selection(inputtext, selection);

    // check that we are extracting the text we thought we were extracting
    assert_eq!(&sm.span_to_snippet(span).unwrap(), "BB\nCCC\nDDDDD");

    // check that span_to_lines gives us the complete result with the lines/cols we expected
    let lines = sm.span_to_lines(span).unwrap();
    let expected = vec![
        LineInfo { line_index: 1, start_col: CharPos(4), end_col: CharPos(6) },
        LineInfo { line_index: 2, start_col: CharPos(0), end_col: CharPos(3) },
        LineInfo { line_index: 3, start_col: CharPos(0), end_col: CharPos(5) }
        ];
    assert_eq!(lines.lines, expected);
}

#[test]
fn t8() {
    // Test span_to_snippet for a span ending at the end of source_file
    let sm = init_source_map();
    let span = Span::with_root_ctxt(BytePos(12), BytePos(23));
    let snippet = sm.span_to_snippet(span);

    assert_eq!(snippet, Ok("second line".to_string()));
}

#[test]
fn t9() {
    // Test span_to_str for a span ending at the end of source_file
    let sm = init_source_map();
    let span = Span::with_root_ctxt(BytePos(12), BytePos(23));
    let sstr =  sm.span_to_string(span);

    assert_eq!(sstr, "blork.rs:2:1: 2:12");
}

/// Tests failing to merge two spans on different lines
#[test]
fn span_merging_fail() {
    let sm = SourceMap::new(FilePathMapping::empty());
    let inputtext  = "bbbb BB\ncc CCC\n";
    let selection1 = "     ~~\n      \n";
    let selection2 = "       \n   ~~~\n";
    sm.new_source_file(Path::new("blork.rs").to_owned().into(), inputtext.to_owned());
    let span1 = span_from_selection(inputtext, selection1);
    let span2 = span_from_selection(inputtext, selection2);

    assert!(sm.merge_spans(span1, span2).is_none());
}

/// Returns the span corresponding to the `n`th occurrence of
/// `substring` in `source_text`.
trait SourceMapExtension {
    fn span_substr(&self,
                file: &Lrc<SourceFile>,
                source_text: &str,
                substring: &str,
                n: usize)
                -> Span;
}

impl SourceMapExtension for SourceMap {
    fn span_substr(&self,
                file: &Lrc<SourceFile>,
                source_text: &str,
                substring: &str,
                n: usize)
                -> Span
    {
        println!("span_substr(file={:?}/{:?}, substring={:?}, n={})",
                file.name, file.start_pos, substring, n);
        let mut i = 0;
        let mut hi = 0;
        loop {
            let offset = source_text[hi..].find(substring).unwrap_or_else(|| {
                panic!("source_text `{}` does not have {} occurrences of `{}`, only {}",
                    source_text, n, substring, i);
            });
            let lo = hi + offset;
            hi = lo + substring.len();
            if i == n {
                let span = Span::with_root_ctxt(
                    BytePos(lo as u32 + file.start_pos.0),
                    BytePos(hi as u32 + file.start_pos.0),
                );
                assert_eq!(&self.span_to_snippet(span).unwrap()[..],
                        substring);
                return span;
            }
            i += 1;
        }
    }
}
