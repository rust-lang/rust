use rustc_ast as ast;
use rustc_ast::tokenstream::TokenStream;
use rustc_parse::{new_parser_from_source_str, parser::Parser, source_file_to_stream};
use rustc_session::parse::ParseSess;
use rustc_span::create_default_session_globals_then;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{BytePos, Span};

use rustc_data_structures::sync::Lrc;
use rustc_errors::emitter::HumanEmitter;
use rustc_errors::{DiagCtxt, MultiSpan, PResult};
use termcolor::WriteColor;

use std::io;
use std::io::prelude::*;
use std::iter::Peekable;
use std::path::{Path, PathBuf};
use std::str;
use std::sync::{Arc, Mutex};

pub(crate) fn psess() -> ParseSess {
    ParseSess::new(vec![crate::DEFAULT_LOCALE_RESOURCE, rustc_parse::DEFAULT_LOCALE_RESOURCE])
}

/// Map string to parser (via tts).
fn string_to_parser(psess: &ParseSess, source_str: String) -> Parser<'_> {
    new_parser_from_source_str(psess, PathBuf::from("bogofile").into(), source_str)
}

fn create_test_handler() -> (DiagCtxt, Lrc<SourceMap>, Arc<Mutex<Vec<u8>>>) {
    let output = Arc::new(Mutex::new(Vec::new()));
    let source_map = Lrc::new(SourceMap::new(FilePathMapping::empty()));
    let fallback_bundle = rustc_errors::fallback_fluent_bundle(
        vec![crate::DEFAULT_LOCALE_RESOURCE, rustc_parse::DEFAULT_LOCALE_RESOURCE],
        false,
    );
    let emitter = HumanEmitter::new(Box::new(Shared { data: output.clone() }), fallback_bundle)
        .sm(Some(source_map.clone()))
        .diagnostic_width(Some(140));
    let dcx = DiagCtxt::new(Box::new(emitter));
    (dcx, source_map, output)
}

/// Returns the result of parsing the given string via the given callback.
///
/// If there are any errors, this will panic.
pub(crate) fn with_error_checking_parse<'a, T, F>(s: String, psess: &'a ParseSess, f: F) -> T
where
    F: FnOnce(&mut Parser<'a>) -> PResult<'a, T>,
{
    let mut p = string_to_parser(&psess, s);
    let x = f(&mut p).unwrap();
    p.psess.dcx.abort_if_errors();
    x
}

/// Verifies that parsing the given string using the given callback will
/// generate an error that contains the given text.
pub(crate) fn with_expected_parse_error<T, F>(source_str: &str, expected_output: &str, f: F)
where
    F: for<'a> FnOnce(&mut Parser<'a>) -> PResult<'a, T>,
{
    let (handler, source_map, output) = create_test_handler();
    let psess = ParseSess::with_dcx(handler, source_map);
    let mut p = string_to_parser(&psess, source_str.to_string());
    let result = f(&mut p);
    assert!(result.is_ok());

    let bytes = output.lock().unwrap();
    let actual_output = str::from_utf8(&bytes).unwrap();
    println!("expected output:\n------\n{}------", expected_output);
    println!("actual output:\n------\n{}------", actual_output);

    assert!(actual_output.contains(expected_output))
}

/// Maps a string to tts, using a made-up filename.
pub(crate) fn string_to_stream(source_str: String) -> TokenStream {
    let psess = psess();
    source_file_to_stream(
        &psess,
        psess.source_map().new_source_file(PathBuf::from("bogofile").into(), source_str),
        None,
    )
}

/// Parses a string, returns a crate.
pub(crate) fn string_to_crate(source_str: String) -> ast::Crate {
    let psess = psess();
    with_error_checking_parse(source_str, &psess, |p| p.parse_crate_mod())
}

/// Does the given string match the pattern? whitespace in the first string
/// may be deleted or replaced with other whitespace to match the pattern.
/// This function is relatively Unicode-ignorant; fortunately, the careful design
/// of UTF-8 mitigates this ignorance. It doesn't do NKF-normalization(?).
pub(crate) fn matches_codepattern(a: &str, b: &str) -> bool {
    let mut a_iter = a.chars().peekable();
    let mut b_iter = b.chars().peekable();

    loop {
        let (a, b) = match (a_iter.peek(), b_iter.peek()) {
            (None, None) => return true,
            (None, _) => return false,
            (Some(&a), None) => {
                if rustc_lexer::is_whitespace(a) {
                    break; // Trailing whitespace check is out of loop for borrowck.
                } else {
                    return false;
                }
            }
            (Some(&a), Some(&b)) => (a, b),
        };

        if rustc_lexer::is_whitespace(a) && rustc_lexer::is_whitespace(b) {
            // Skip whitespace for `a` and `b`.
            scan_for_non_ws_or_end(&mut a_iter);
            scan_for_non_ws_or_end(&mut b_iter);
        } else if rustc_lexer::is_whitespace(a) {
            // Skip whitespace for `a`.
            scan_for_non_ws_or_end(&mut a_iter);
        } else if a == b {
            a_iter.next();
            b_iter.next();
        } else {
            return false;
        }
    }

    // Check if a has *only* trailing whitespace.
    a_iter.all(rustc_lexer::is_whitespace)
}

/// Advances the given peekable `Iterator` until it reaches a non-whitespace character.
fn scan_for_non_ws_or_end<I: Iterator<Item = char>>(iter: &mut Peekable<I>) {
    while iter.peek().copied().is_some_and(rustc_lexer::is_whitespace) {
        iter.next();
    }
}

/// Identifies a position in the text by the n'th occurrence of a string.
struct Position {
    string: &'static str,
    count: usize,
}

struct SpanLabel {
    start: Position,
    end: Position,
    label: &'static str,
}

pub(crate) struct Shared<T: Write> {
    pub data: Arc<Mutex<T>>,
}

impl<T: Write> WriteColor for Shared<T> {
    fn supports_color(&self) -> bool {
        false
    }

    fn set_color(&mut self, _spec: &termcolor::ColorSpec) -> io::Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<T: Write> Write for Shared<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.data.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.data.lock().unwrap().flush()
    }
}

#[allow(rustc::untranslatable_diagnostic)] // no translation needed for tests
fn test_harness(file_text: &str, span_labels: Vec<SpanLabel>, expected_output: &str) {
    create_default_session_globals_then(|| {
        let (handler, source_map, output) = create_test_handler();
        source_map.new_source_file(Path::new("test.rs").to_owned().into(), file_text.to_owned());

        let primary_span = make_span(&file_text, &span_labels[0].start, &span_labels[0].end);
        let mut msp = MultiSpan::from_span(primary_span);
        for span_label in span_labels {
            let span = make_span(&file_text, &span_label.start, &span_label.end);
            msp.push_span_label(span, span_label.label);
            println!("span: {:?} label: {:?}", span, span_label.label);
            println!("text: {:?}", source_map.span_to_snippet(span));
        }

        handler.span_err(msp, "foo");

        assert!(
            expected_output.chars().next() == Some('\n'),
            "expected output should begin with newline"
        );
        let expected_output = &expected_output[1..];

        let bytes = output.lock().unwrap();
        let actual_output = str::from_utf8(&bytes).unwrap();
        println!("expected output:\n------\n{}------", expected_output);
        println!("actual output:\n------\n{}------", actual_output);

        assert!(expected_output == actual_output)
    })
}

fn make_span(file_text: &str, start: &Position, end: &Position) -> Span {
    let start = make_pos(file_text, start);
    let end = make_pos(file_text, end) + end.string.len(); // just after matching thing ends
    assert!(start <= end);
    Span::with_root_ctxt(BytePos(start as u32), BytePos(end as u32))
}

fn make_pos(file_text: &str, pos: &Position) -> usize {
    let mut remainder = file_text;
    let mut offset = 0;
    for _ in 0..pos.count {
        if let Some(n) = remainder.find(&pos.string) {
            offset += n;
            remainder = &remainder[n + 1..];
        } else {
            panic!("failed to find {} instances of {:?} in {:?}", pos.count, pos.string, file_text);
        }
    }
    offset
}

#[test]
fn ends_on_col0() {
    test_harness(
        r#"
fn foo() {
}
"#,
        vec![SpanLabel {
            start: Position { string: "{", count: 1 },
            end: Position { string: "}", count: 1 },
            label: "test",
        }],
        r#"
error: foo
 --> test.rs:2:10
  |
2 |   fn foo() {
  |  __________^
3 | | }
  | |_^ test

"#,
    );
}

#[test]
fn ends_on_col2() {
    test_harness(
        r#"
fn foo() {


  }
"#,
        vec![SpanLabel {
            start: Position { string: "{", count: 1 },
            end: Position { string: "}", count: 1 },
            label: "test",
        }],
        r#"
error: foo
 --> test.rs:2:10
  |
2 |   fn foo() {
  |  __________^
3 | |
4 | |
5 | |   }
  | |___^ test

"#,
    );
}
#[test]
fn non_nested() {
    test_harness(
        r#"
fn foo() {
  X0 Y0
  X1 Y1
  X2 Y2
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "X0", count: 1 },
                end: Position { string: "X2", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Y0", count: 1 },
                end: Position { string: "Y2", count: 1 },
                label: "`Y` is a good letter too",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |      X0 Y0
  |   ___^__-
  |  |___|
  | ||
4 | ||   X1 Y1
5 | ||   X2 Y2
  | ||____^__- `Y` is a good letter too
  | |_____|
  |       `X` is a good letter

"#,
    );
}

#[test]
fn nested() {
    test_harness(
        r#"
fn foo() {
  X0 Y0
  Y1 X1
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "X0", count: 1 },
                end: Position { string: "X1", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Y0", count: 1 },
                end: Position { string: "Y1", count: 1 },
                label: "`Y` is a good letter too",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |      X0 Y0
  |   ___^__-
  |  |___|
  | ||
4 | ||   Y1 X1
  | ||____-__^ `X` is a good letter
  |  |____|
  |       `Y` is a good letter too

"#,
    );
}

#[test]
fn different_overlap() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "Y0", count: 1 },
                end: Position { string: "X2", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Z1", count: 1 },
                end: Position { string: "X3", count: 1 },
                label: "`Y` is a good letter too",
            },
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |      X0 Y0 Z0
  |  _______^
4 | |    X1 Y1 Z1
  | | _________-
5 | ||   X2 Y2 Z2
  | ||____^ `X` is a good letter
6 |  |   X3 Y3 Z3
  |  |____- `Y` is a good letter too

"#,
    );
}

#[test]
fn triple_overlap() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "X0", count: 1 },
                end: Position { string: "X2", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Y0", count: 1 },
                end: Position { string: "Y2", count: 1 },
                label: "`Y` is a good letter too",
            },
            SpanLabel {
                start: Position { string: "Z0", count: 1 },
                end: Position { string: "Z2", count: 1 },
                label: "`Z` label",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |       X0 Y0 Z0
  |    ___^__-__-
  |   |___|__|
  |  ||___|
  | |||
4 | |||   X1 Y1 Z1
5 | |||   X2 Y2 Z2
  | |||____^__-__- `Z` label
  | ||_____|__|
  | |______|  `Y` is a good letter too
  |        `X` is a good letter

"#,
    );
}

#[test]
fn triple_exact_overlap() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "X0", count: 1 },
                end: Position { string: "X2", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "X0", count: 1 },
                end: Position { string: "X2", count: 1 },
                label: "`Y` is a good letter too",
            },
            SpanLabel {
                start: Position { string: "X0", count: 1 },
                end: Position { string: "X2", count: 1 },
                label: "`Z` label",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 | /   X0 Y0 Z0
4 | |   X1 Y1 Z1
5 | |   X2 Y2 Z2
  | |    ^
  | |    |
  | |    `X` is a good letter
  | |____`Y` is a good letter too
  |      `Z` label

"#,
    );
}

#[test]
fn minimum_depth() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "Y0", count: 1 },
                end: Position { string: "X1", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Y1", count: 1 },
                end: Position { string: "Z2", count: 1 },
                label: "`Y` is a good letter too",
            },
            SpanLabel {
                start: Position { string: "X2", count: 1 },
                end: Position { string: "Y3", count: 1 },
                label: "`Z`",
            },
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |      X0 Y0 Z0
  |  _______^
4 | |    X1 Y1 Z1
  | | ____^_-
  | ||____|
  |  |    `X` is a good letter
5 |  |   X2 Y2 Z2
  |  |___-______- `Y` is a good letter too
  |   ___|
  |  |
6 |  |   X3 Y3 Z3
  |  |_______- `Z`

"#,
    );
}

#[test]
fn non_overlapping() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "X0", count: 1 },
                end: Position { string: "X1", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Y2", count: 1 },
                end: Position { string: "Z3", count: 1 },
                label: "`Y` is a good letter too",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 | /   X0 Y0 Z0
4 | |   X1 Y1 Z1
  | |____^ `X` is a good letter
5 |     X2 Y2 Z2
  |  ______-
6 | |   X3 Y3 Z3
  | |__________- `Y` is a good letter too

"#,
    );
}

#[test]
fn overlapping_start_and_end() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "Y0", count: 1 },
                end: Position { string: "X1", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Z1", count: 1 },
                end: Position { string: "Z3", count: 1 },
                label: "`Y` is a good letter too",
            },
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |      X0 Y0 Z0
  |  _______^
4 | |    X1 Y1 Z1
  | | ____^____-
  | ||____|
  |  |    `X` is a good letter
5 |  |   X2 Y2 Z2
6 |  |   X3 Y3 Z3
  |  |__________- `Y` is a good letter too

"#,
    );
}

#[test]
fn multiple_labels_primary_without_message() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "b", count: 1 },
                end: Position { string: "}", count: 1 },
                label: "",
            },
            SpanLabel {
                start: Position { string: "a", count: 1 },
                end: Position { string: "d", count: 1 },
                label: "`a` is a good letter",
            },
            SpanLabel {
                start: Position { string: "c", count: 1 },
                end: Position { string: "c", count: 1 },
                label: "",
            },
        ],
        r#"
error: foo
 --> test.rs:3:7
  |
3 |   a { b { c } d }
  |   ----^^^^-^^-- `a` is a good letter

"#,
    );
}

#[test]
fn multiple_labels_secondary_without_message() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "a", count: 1 },
                end: Position { string: "d", count: 1 },
                label: "`a` is a good letter",
            },
            SpanLabel {
                start: Position { string: "b", count: 1 },
                end: Position { string: "}", count: 1 },
                label: "",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^ `a` is a good letter

"#,
    );
}

#[test]
fn multiple_labels_primary_without_message_2() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "b", count: 1 },
                end: Position { string: "}", count: 1 },
                label: "`b` is a good letter",
            },
            SpanLabel {
                start: Position { string: "a", count: 1 },
                end: Position { string: "d", count: 1 },
                label: "",
            },
            SpanLabel {
                start: Position { string: "c", count: 1 },
                end: Position { string: "c", count: 1 },
                label: "",
            },
        ],
        r#"
error: foo
 --> test.rs:3:7
  |
3 |   a { b { c } d }
  |   ----^^^^-^^--
  |       |
  |       `b` is a good letter

"#,
    );
}

#[test]
fn multiple_labels_secondary_without_message_2() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "a", count: 1 },
                end: Position { string: "d", count: 1 },
                label: "",
            },
            SpanLabel {
                start: Position { string: "b", count: 1 },
                end: Position { string: "}", count: 1 },
                label: "`b` is a good letter",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^
  |       |
  |       `b` is a good letter

"#,
    );
}

#[test]
fn multiple_labels_secondary_without_message_3() {
    test_harness(
        r#"
fn foo() {
  a  bc  d
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "a", count: 1 },
                end: Position { string: "b", count: 1 },
                label: "`a` is a good letter",
            },
            SpanLabel {
                start: Position { string: "c", count: 1 },
                end: Position { string: "d", count: 1 },
                label: "",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a  bc  d
  |   ^^^^----
  |   |
  |   `a` is a good letter

"#,
    );
}

#[test]
fn multiple_labels_without_message() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "a", count: 1 },
                end: Position { string: "d", count: 1 },
                label: "",
            },
            SpanLabel {
                start: Position { string: "b", count: 1 },
                end: Position { string: "}", count: 1 },
                label: "",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^

"#,
    );
}

#[test]
fn multiple_labels_without_message_2() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "b", count: 1 },
                end: Position { string: "}", count: 1 },
                label: "",
            },
            SpanLabel {
                start: Position { string: "a", count: 1 },
                end: Position { string: "d", count: 1 },
                label: "",
            },
            SpanLabel {
                start: Position { string: "c", count: 1 },
                end: Position { string: "c", count: 1 },
                label: "",
            },
        ],
        r#"
error: foo
 --> test.rs:3:7
  |
3 |   a { b { c } d }
  |   ----^^^^-^^--

"#,
    );
}

#[test]
fn multiple_labels_with_message() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "a", count: 1 },
                end: Position { string: "d", count: 1 },
                label: "`a` is a good letter",
            },
            SpanLabel {
                start: Position { string: "b", count: 1 },
                end: Position { string: "}", count: 1 },
                label: "`b` is a good letter",
            },
        ],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^
  |   |   |
  |   |   `b` is a good letter
  |   `a` is a good letter

"#,
    );
}

#[test]
fn single_label_with_message() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![SpanLabel {
            start: Position { string: "a", count: 1 },
            end: Position { string: "d", count: 1 },
            label: "`a` is a good letter",
        }],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^^^^^^^^^^ `a` is a good letter

"#,
    );
}

#[test]
fn single_label_without_message() {
    test_harness(
        r#"
fn foo() {
  a { b { c } d }
}
"#,
        vec![SpanLabel {
            start: Position { string: "a", count: 1 },
            end: Position { string: "d", count: 1 },
            label: "",
        }],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^^^^^^^^^^

"#,
    );
}

#[test]
fn long_snippet() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
1
2
3
4
5
6
7
8
9
10
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "Y0", count: 1 },
                end: Position { string: "X1", count: 1 },
                label: "`X` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Z1", count: 1 },
                end: Position { string: "Z3", count: 1 },
                label: "`Y` is a good letter too",
            },
        ],
        r#"
error: foo
  --> test.rs:3:6
   |
3  |      X0 Y0 Z0
   |  _______^
4  | |    X1 Y1 Z1
   | | ____^____-
   | ||____|
   |  |    `X` is a good letter
5  |  | 1
6  |  | 2
7  |  | 3
...   |
15 |  |   X2 Y2 Z2
16 |  |   X3 Y3 Z3
   |  |__________- `Y` is a good letter too

"#,
    );
}

#[test]
fn long_snippet_multiple_spans() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
1
2
3
  X1 Y1 Z1
4
5
6
  X2 Y2 Z2
7
8
9
10
  X3 Y3 Z3
}
"#,
        vec![
            SpanLabel {
                start: Position { string: "Y0", count: 1 },
                end: Position { string: "Y3", count: 1 },
                label: "`Y` is a good letter",
            },
            SpanLabel {
                start: Position { string: "Z1", count: 1 },
                end: Position { string: "Z2", count: 1 },
                label: "`Z` is a good letter too",
            },
        ],
        r#"
error: foo
  --> test.rs:3:6
   |
3  |      X0 Y0 Z0
   |  _______^
4  | |  1
5  | |  2
6  | |  3
7  | |    X1 Y1 Z1
   | | _________-
8  | || 4
9  | || 5
10 | || 6
11 | ||   X2 Y2 Z2
   | ||__________- `Z` is a good letter too
...  |
15 | |  10
16 | |    X3 Y3 Z3
   | |________^ `Y` is a good letter

"#,
    );
}
