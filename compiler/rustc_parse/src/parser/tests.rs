#![allow(rustc::symbol_intern_string_literal)]

use std::assert_matches::assert_matches;
use std::io::prelude::*;
use std::iter::Peekable;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{io, str};

use ast::token::IdentIsRaw;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast::{self as ast, PatKind, visit};
use rustc_ast_pretty::pprust::item_to_string;
use rustc_errors::emitter::{HumanEmitter, OutputTheme};
use rustc_errors::{DiagCtxt, MultiSpan, PResult};
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{
    BytePos, FileName, Pos, Span, Symbol, create_default_session_globals_then, kw, sym,
};
use termcolor::WriteColor;

use crate::parser::{ForceCollect, Parser};
use crate::{new_parser_from_source_str, source_str_to_stream, unwrap_or_emit_fatal};

fn psess() -> ParseSess {
    ParseSess::new(vec![crate::DEFAULT_LOCALE_RESOURCE])
}

/// Map string to parser (via tts).
fn string_to_parser(psess: &ParseSess, source_str: String) -> Parser<'_> {
    unwrap_or_emit_fatal(new_parser_from_source_str(
        psess,
        PathBuf::from("bogofile").into(),
        source_str,
    ))
}

fn create_test_handler(theme: OutputTheme) -> (DiagCtxt, Arc<SourceMap>, Arc<Mutex<Vec<u8>>>) {
    let output = Arc::new(Mutex::new(Vec::new()));
    let source_map = Arc::new(SourceMap::new(FilePathMapping::empty()));
    let fallback_bundle =
        rustc_errors::fallback_fluent_bundle(vec![crate::DEFAULT_LOCALE_RESOURCE], false);
    let mut emitter = HumanEmitter::new(Box::new(Shared { data: output.clone() }), fallback_bundle)
        .sm(Some(source_map.clone()))
        .diagnostic_width(Some(140));
    emitter = emitter.theme(theme);
    let dcx = DiagCtxt::new(Box::new(emitter));
    (dcx, source_map, output)
}

/// Returns the result of parsing the given string via the given callback.
///
/// If there are any errors, this will panic.
fn with_error_checking_parse<'a, T, F>(s: String, psess: &'a ParseSess, f: F) -> T
where
    F: FnOnce(&mut Parser<'a>) -> PResult<'a, T>,
{
    let mut p = string_to_parser(&psess, s);
    let x = f(&mut p).unwrap();
    p.dcx().abort_if_errors();
    x
}

/// Verifies that parsing the given string using the given callback will
/// generate an error that contains the given text.
fn with_expected_parse_error<T, F>(source_str: &str, expected_output: &str, f: F)
where
    F: for<'a> FnOnce(&mut Parser<'a>) -> PResult<'a, T>,
{
    let (handler, source_map, output) = create_test_handler(OutputTheme::Ascii);
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
    unwrap_or_emit_fatal(source_str_to_stream(
        &psess,
        PathBuf::from("bogofile").into(),
        source_str,
        None,
    ))
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

struct Shared<T: Write> {
    data: Arc<Mutex<T>>,
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
fn test_harness(
    file_text: &str,
    span_labels: Vec<SpanLabel>,
    notes: Vec<(Option<(Position, Position)>, &'static str)>,
    expected_output_ascii: &str,
    expected_output_unicode: &str,
) {
    create_default_session_globals_then(|| {
        for (theme, expected_output) in [
            (OutputTheme::Ascii, expected_output_ascii),
            (OutputTheme::Unicode, expected_output_unicode),
        ] {
            let (dcx, source_map, output) = create_test_handler(theme);
            source_map
                .new_source_file(Path::new("test.rs").to_owned().into(), file_text.to_owned());

            let primary_span = make_span(&file_text, &span_labels[0].start, &span_labels[0].end);
            let mut msp = MultiSpan::from_span(primary_span);
            for span_label in &span_labels {
                let span = make_span(&file_text, &span_label.start, &span_label.end);
                msp.push_span_label(span, span_label.label);
                println!("span: {:?} label: {:?}", span, span_label.label);
                println!("text: {:?}", source_map.span_to_snippet(span));
            }

            let mut err = dcx.handle().struct_span_err(msp, "foo");
            for (position, note) in &notes {
                if let Some((start, end)) = position {
                    let span = make_span(&file_text, &start, &end);
                    err.span_note(span, *note);
                } else {
                    err.note(*note);
                }
            }
            err.emit();

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
        }
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
        vec![],
        r#"
error: foo
 --> test.rs:2:10
  |
2 |   fn foo() {
  |  __________^
3 | | }
  | |_^ test

"#,
        r#"
error: foo
  ╭▸ test.rs:2:10
  │
2 │   fn foo() {
  │ ┏━━━━━━━━━━┛
3 │ ┃ }
  ╰╴┗━┛ test

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
        vec![],
        r#"
error: foo
 --> test.rs:2:10
  |
2 |   fn foo() {
  |  __________^
... |
5 | |   }
  | |___^ test

"#,
        r#"
error: foo
  ╭▸ test.rs:2:10
  │
2 │   fn foo() {
  │ ┏━━━━━━━━━━┛
  ‡ ┃
5 │ ┃   }
  ╰╴┗━━━┛ test

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |      X0 Y0
  |  ____^  -
  | | ______|
4 | ||   X1 Y1
5 | ||   X2 Y2
  | ||____^__- `Y` is a good letter too
  | |_____|
  |       `X` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │      X0 Y0
  │ ┏━━━━┛  │
  │ ┃┌──────┘
4 │ ┃│   X1 Y1
5 │ ┃│   X2 Y2
  │ ┃└────╿──┘ `Y` is a good letter too
  │ ┗━━━━━┥
  ╰╴      `X` is a good letter

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |      X0 Y0
  |  ____^  -
  | | ______|
4 | ||   Y1 X1
  | ||____-__^ `X` is a good letter
  |  |____|
  |       `Y` is a good letter too

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │      X0 Y0
  │ ┏━━━━┛  │
  │ ┃┌──────┘
4 │ ┃│   Y1 X1
  │ ┗│━━━━│━━┛ `X` is a good letter
  │  └────┤
  ╰╴      `Y` is a good letter too

"#,
    );
}

#[test]
fn multiline_and_normal_overlap() {
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
                start: Position { string: "X0", count: 1 },
                end: Position { string: "Y0", count: 1 },
                label: "`Y` is a good letter too",
            },
        ],
        vec![],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |     X0 Y0 Z0
  |  ___---^-
  | |   |
  | |   `Y` is a good letter too
4 | |   X1 Y1 Z1
5 | |   X2 Y2 Z2
  | |____^ `X` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │     X0 Y0 Z0
  │ ┏━━━┬──┛─
  │ ┃   │
  │ ┃   `Y` is a good letter too
4 │ ┃   X1 Y1 Z1
5 │ ┃   X2 Y2 Z2
  ╰╴┗━━━━┛ `X` is a good letter

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
        vec![],
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
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │      X0 Y0 Z0
  │ ┏━━━━━━━┛
4 │ ┃    X1 Y1 Z1
  │ ┃┌─────────┘
5 │ ┃│   X2 Y2 Z2
  │ ┗│━━━━┛ `X` is a good letter
6 │  │   X3 Y3 Z3
  ╰╴ └────┘ `Y` is a good letter too

"#,
    );
}

#[test]
fn different_note_1() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![(None, "bar")],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
  = note: bar

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  │
  ╰ note: bar

"#,
    );
}

#[test]
fn different_note_2() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![(None, "bar"), (None, "qux")],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
  = note: bar
  = note: qux

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  │
  ├ note: bar
  ╰ note: qux

"#,
    );
}

#[test]
fn different_note_3() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![(None, "bar"), (None, "baz"), (None, "qux")],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
  = note: bar
  = note: baz
  = note: qux

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  │
  ├ note: bar
  ├ note: baz
  ╰ note: qux

"#,
    );
}

#[test]
fn different_note_spanned_1() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![(
            Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
            "bar",
        )],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
note: bar
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  ╰╴
note: bar
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━

"#,
    );
}

#[test]
fn different_note_spanned_2() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "bar",
            ),
            (
                Some((Position { string: "X2", count: 1 }, Position { string: "Y2", count: 1 })),
                "qux",
            ),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
note: bar
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^
note: qux
 --> test.rs:5:3
  |
5 |   X2 Y2 Z2
  |   ^^^^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  ╰╴
note: bar
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━
note: qux
  ╭▸ test.rs:5:3
  │
5 │   X2 Y2 Z2
  ╰╴  ━━━━━

"#,
    );
}

#[test]
fn different_note_spanned_3() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "bar",
            ),
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "baz",
            ),
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "qux",
            ),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
note: bar
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^
note: baz
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^
note: qux
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  ╰╴
note: bar
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━
note: baz
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━
note: qux
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━

"#,
    );
}

#[test]
fn different_note_spanned_4() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "bar",
            ),
            (None, "qux"),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
note: bar
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^
  = note: qux

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  ╰╴
note: bar
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  │   ━━━━━━━━
  ╰ note: qux

"#,
    );
}

#[test]
fn different_note_spanned_5() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (None, "bar"),
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "qux",
            ),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
  = note: bar
note: qux
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  │
  ╰ note: bar
note: qux
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━

"#,
    );
}

#[test]
fn different_note_spanned_6() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (None, "bar"),
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "baz",
            ),
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "qux",
            ),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
  = note: bar
note: baz
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^
note: qux
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  │
  ╰ note: bar
note: baz
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━
note: qux
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━

"#,
    );
}

#[test]
fn different_note_spanned_7() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z3", count: 1 })),
                "bar",
            ),
            (None, "baz"),
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "qux",
            ),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
note: bar
 --> test.rs:4:3
  |
4 | /   X1 Y1 Z1
5 | |   X2 Y2 Z2
6 | |   X3 Y3 Z3
  | |__________^
  = note: baz
note: qux
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  ╰╴
note: bar
  ╭▸ test.rs:4:3
  │
4 │ ┏   X1 Y1 Z1
5 │ ┃   X2 Y2 Z2
6 │ ┃   X3 Y3 Z3
  │ ┗━━━━━━━━━━┛
  ╰ note: baz
note: qux
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━

"#,
    );
}

#[test]
fn different_note_spanned_8() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "bar",
            ),
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "baz",
            ),
            (None, "qux"),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
note: bar
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^
note: baz
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^
  = note: qux

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  ╰╴
note: bar
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━
note: baz
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  │   ━━━━━━━━
  ╰ note: qux

"#,
    );
}

#[test]
fn different_note_spanned_9() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (None, "bar"),
            (None, "baz"),
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "qux",
            ),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
  = note: bar
  = note: baz
note: qux
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  │
  ├ note: bar
  ╰ note: baz
note: qux
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  ╰╴  ━━━━━━━━

"#,
    );
}

#[test]
fn different_note_spanned_10() {
    test_harness(
        r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
        vec![SpanLabel {
            start: Position { string: "Y0", count: 1 },
            end: Position { string: "Z0", count: 1 },
            label: "`X` is a good letter",
        }],
        vec![
            (
                Some((Position { string: "X1", count: 1 }, Position { string: "Z1", count: 1 })),
                "bar",
            ),
            (None, "baz"),
            (None, "qux"),
        ],
        r#"
error: foo
 --> test.rs:3:6
  |
3 |   X0 Y0 Z0
  |      ^^^^^ `X` is a good letter
  |
note: bar
 --> test.rs:4:3
  |
4 |   X1 Y1 Z1
  |   ^^^^^^^^
  = note: baz
  = note: qux

"#,
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │   X0 Y0 Z0
  │      ━━━━━ `X` is a good letter
  ╰╴
note: bar
  ╭▸ test.rs:4:3
  │
4 │   X1 Y1 Z1
  │   ━━━━━━━━
  ├ note: baz
  ╰ note: qux

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |       X0 Y0 Z0
  |  _____^  -  -
  | | _______|  |
  | || _________|
4 | |||   X1 Y1 Z1
5 | |||   X2 Y2 Z2
  | |||____^__-__- `Z` label
  | ||_____|__|
  | |______|  `Y` is a good letter too
  |        `X` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │       X0 Y0 Z0
  │ ┏━━━━━┛  │  │
  │ ┃┌───────┘  │
  │ ┃│┌─────────┘
4 │ ┃││   X1 Y1 Z1
5 │ ┃││   X2 Y2 Z2
  │ ┃│└────╿──│──┘ `Z` label
  │ ┃└─────│──┤
  │ ┗━━━━━━┥  `Y` is a good letter too
  ╰╴       `X` is a good letter

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
        vec![],
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
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │ ┏   X0 Y0 Z0
4 │ ┃   X1 Y1 Z1
5 │ ┃   X2 Y2 Z2
  │ ┃    ╿
  │ ┃    │
  │ ┃    `X` is a good letter
  │ ┗━━━━`Y` is a good letter too
  ╰╴     `Z` label

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
        vec![],
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
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │      X0 Y0 Z0
  │ ┏━━━━━━━┛
4 │ ┃    X1 Y1 Z1
  │ ┃┌────╿─┘
  │ ┗│━━━━┥
  │  │    `X` is a good letter
5 │  │   X2 Y2 Z2
  │  └───│──────┘ `Y` is a good letter too
  │  ┌───┘
  │  │
6 │  │   X3 Y3 Z3
  ╰╴ └───────┘ `Z`

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
        vec![],
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
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │ ┏   X0 Y0 Z0
4 │ ┃   X1 Y1 Z1
  │ ┗━━━━┛ `X` is a good letter
5 │     X2 Y2 Z2
  │ ┌──────┘
6 │ │   X3 Y3 Z3
  ╰╴└──────────┘ `Y` is a good letter too

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
        vec![],
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
        r#"
error: foo
  ╭▸ test.rs:3:6
  │
3 │      X0 Y0 Z0
  │ ┏━━━━━━━┛
4 │ ┃    X1 Y1 Z1
  │ ┃┌────╿────┘
  │ ┗│━━━━┥
  │  │    `X` is a good letter
5 │  │   X2 Y2 Z2
6 │  │   X3 Y3 Z3
  ╰╴ └──────────┘ `Y` is a good letter too

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
        vec![],
        r#"
error: foo
 --> test.rs:3:7
  |
3 |   a { b { c } d }
  |   ----^^^^-^^-- `a` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:7
  │
3 │   a { b { c } d }
  ╰╴  ────━━━━─━━── `a` is a good letter

"#,
    );
}

#[test]
fn multiline_notes() {
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
        vec![(None, "foo\nbar"), (None, "foo\nbar")],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^^^^^^^^^^ `a` is a good letter
  |
  = note: foo
          bar
  = note: foo
          bar

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │   a { b { c } d }
  │   ━━━━━━━━━━━━━ `a` is a good letter
  │
  ├ note: foo
  │       bar
  ╰ note: foo
          bar

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^ `a` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │   a { b { c } d }
  ╰╴  ━━━━───────━━ `a` is a good letter

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
        vec![],
        r#"
error: foo
 --> test.rs:3:7
  |
3 |   a { b { c } d }
  |   ----^^^^-^^--
  |       |
  |       `b` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:7
  │
3 │   a { b { c } d }
  │   ────┯━━━─━━──
  │       │
  ╰╴      `b` is a good letter

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^
  |       |
  |       `b` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │   a { b { c } d }
  │   ━━━━┬──────━━
  │       │
  ╰╴      `b` is a good letter

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a  bc  d
  |   ^^^^----
  |   |
  |   `a` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │   a  bc  d
  │   ┯━━━────
  │   │
  ╰╴  `a` is a good letter

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │   a { b { c } d }
  ╰╴  ━━━━───────━━

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
        vec![],
        r#"
error: foo
 --> test.rs:3:7
  |
3 |   a { b { c } d }
  |   ----^^^^-^^--

"#,
        r#"
error: foo
  ╭▸ test.rs:3:7
  │
3 │   a { b { c } d }
  ╰╴  ────━━━━─━━──

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
        vec![],
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
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │   a { b { c } d }
  │   ┯━━━┬──────━━
  │   │   │
  │   │   `b` is a good letter
  ╰╴  `a` is a good letter

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^^^^^^^^^^ `a` is a good letter

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │   a { b { c } d }
  ╰╴  ━━━━━━━━━━━━━ `a` is a good letter

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
        vec![],
        r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^^^^^^^^^^

"#,
        r#"
error: foo
  ╭▸ test.rs:3:3
  │
3 │   a { b { c } d }
  ╰╴  ━━━━━━━━━━━━━

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
        vec![],
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
        r#"
error: foo
   ╭▸ test.rs:3:6
   │
3  │      X0 Y0 Z0
   │ ┏━━━━━━━┛
4  │ ┃    X1 Y1 Z1
   │ ┃┌────╿────┘
   │ ┗│━━━━┥
   │  │    `X` is a good letter
5  │  │ 1
6  │  │ 2
7  │  │ 3
   ‡  │
15 │  │   X2 Y2 Z2
16 │  │   X3 Y3 Z3
   ╰╴ └──────────┘ `Y` is a good letter too

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
        vec![],
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
        r#"
error: foo
   ╭▸ test.rs:3:6
   │
3  │      X0 Y0 Z0
   │ ┏━━━━━━━┛
4  │ ┃  1
5  │ ┃  2
6  │ ┃  3
7  │ ┃    X1 Y1 Z1
   │ ┃┌─────────┘
8  │ ┃│ 4
9  │ ┃│ 5
10 │ ┃│ 6
11 │ ┃│   X2 Y2 Z2
   │ ┃└──────────┘ `Z` is a good letter too
   ‡ ┃
15 │ ┃  10
16 │ ┃    X3 Y3 Z3
   ╰╴┗━━━━━━━━┛ `Y` is a good letter

"#,
    );
}

/// Parses an item.
///
/// Returns `Ok(Some(item))` when successful, `Ok(None)` when no item was found, and `Err`
/// when a syntax error occurred.
fn parse_item_from_source_str(
    name: FileName,
    source: String,
    psess: &ParseSess,
) -> PResult<'_, Option<P<ast::Item>>> {
    unwrap_or_emit_fatal(new_parser_from_source_str(psess, name, source))
        .parse_item(ForceCollect::No)
}

// Produces a `rustc_span::span`.
fn sp(a: u32, b: u32) -> Span {
    Span::with_root_ctxt(BytePos(a), BytePos(b))
}

/// Parses a string, return an expression.
fn string_to_expr(source_str: String) -> P<ast::Expr> {
    with_error_checking_parse(source_str, &psess(), |p| p.parse_expr())
}

/// Parses a string, returns an item.
fn string_to_item(source_str: String) -> Option<P<ast::Item>> {
    with_error_checking_parse(source_str, &psess(), |p| p.parse_item(ForceCollect::No))
}

#[test]
fn bad_path_expr_1() {
    // This should trigger error: expected identifier, found keyword `return`
    create_default_session_globals_then(|| {
        with_expected_parse_error(
            "::abc::def::return",
            "expected identifier, found keyword `return`",
            |p| p.parse_expr(),
        );
    })
}

// Checks the token-tree-ization of macros.
#[test]
fn string_to_tts_macro() {
    create_default_session_globals_then(|| {
        let stream = string_to_stream("macro_rules! zip (($a)=>($a))".to_string());
        let tts = &stream.iter().collect::<Vec<_>>()[..];

        match tts {
            [
                TokenTree::Token(
                    Token { kind: token::Ident(name_macro_rules, IdentIsRaw::No), .. },
                    _,
                ),
                TokenTree::Token(Token { kind: token::Bang, .. }, _),
                TokenTree::Token(Token { kind: token::Ident(name_zip, IdentIsRaw::No), .. }, _),
                TokenTree::Delimited(.., macro_delim, macro_tts),
            ] if name_macro_rules == &kw::MacroRules && name_zip.as_str() == "zip" => {
                let tts = &macro_tts.iter().collect::<Vec<_>>();
                match &tts[..] {
                    [
                        TokenTree::Delimited(.., first_delim, first_tts),
                        TokenTree::Token(Token { kind: token::FatArrow, .. }, _),
                        TokenTree::Delimited(.., second_delim, second_tts),
                    ] if macro_delim == &Delimiter::Parenthesis => {
                        let tts = &first_tts.iter().collect::<Vec<_>>();
                        match &tts[..] {
                            [
                                TokenTree::Token(Token { kind: token::Dollar, .. }, _),
                                TokenTree::Token(
                                    Token { kind: token::Ident(name, IdentIsRaw::No), .. },
                                    _,
                                ),
                            ] if first_delim == &Delimiter::Parenthesis && name.as_str() == "a" => {
                            }
                            _ => panic!("value 3: {:?} {:?}", first_delim, first_tts),
                        }
                        let tts = &second_tts.iter().collect::<Vec<_>>();
                        match &tts[..] {
                            [
                                TokenTree::Token(Token { kind: token::Dollar, .. }, _),
                                TokenTree::Token(
                                    Token { kind: token::Ident(name, IdentIsRaw::No), .. },
                                    _,
                                ),
                            ] if second_delim == &Delimiter::Parenthesis
                                && name.as_str() == "a" => {}
                            _ => panic!("value 4: {:?} {:?}", second_delim, second_tts),
                        }
                    }
                    _ => panic!("value 2: {:?} {:?}", macro_delim, macro_tts),
                }
            }
            _ => panic!("value: {:?}", tts),
        }
    })
}

#[test]
fn string_to_tts_1() {
    create_default_session_globals_then(|| {
        let tts = string_to_stream("fn a(b: i32) { b; }".to_string());

        let expected = TokenStream::new(vec![
            TokenTree::token_alone(token::Ident(kw::Fn, IdentIsRaw::No), sp(0, 2)),
            TokenTree::token_joint_hidden(
                token::Ident(Symbol::intern("a"), IdentIsRaw::No),
                sp(3, 4),
            ),
            TokenTree::Delimited(
                DelimSpan::from_pair(sp(4, 5), sp(11, 12)),
                // `JointHidden` because the `(` is followed immediately by
                // `b`, `Alone` because the `)` is followed by whitespace.
                DelimSpacing::new(Spacing::JointHidden, Spacing::Alone),
                Delimiter::Parenthesis,
                TokenStream::new(vec![
                    TokenTree::token_joint(
                        token::Ident(Symbol::intern("b"), IdentIsRaw::No),
                        sp(5, 6),
                    ),
                    TokenTree::token_alone(token::Colon, sp(6, 7)),
                    // `JointHidden` because the `i32` is immediately followed by the `)`.
                    TokenTree::token_joint_hidden(
                        token::Ident(sym::i32, IdentIsRaw::No),
                        sp(8, 11),
                    ),
                ]),
            ),
            TokenTree::Delimited(
                DelimSpan::from_pair(sp(13, 14), sp(18, 19)),
                // First `Alone` because the `{` is followed by whitespace,
                // second `Alone` because the `}` is followed immediately by
                // EOF.
                DelimSpacing::new(Spacing::Alone, Spacing::Alone),
                Delimiter::Brace,
                TokenStream::new(vec![
                    TokenTree::token_joint(
                        token::Ident(Symbol::intern("b"), IdentIsRaw::No),
                        sp(15, 16),
                    ),
                    // `Alone` because the `;` is followed by whitespace.
                    TokenTree::token_alone(token::Semi, sp(16, 17)),
                ]),
            ),
        ]);

        assert_eq!(tts, expected);
    })
}

#[test]
fn parse_use() {
    create_default_session_globals_then(|| {
        let use_s = "use foo::bar::baz;";
        let vitem = string_to_item(use_s.to_string()).unwrap();
        let vitem_s = item_to_string(&vitem);
        assert_eq!(&vitem_s[..], use_s);

        let use_s = "use foo::bar as baz;";
        let vitem = string_to_item(use_s.to_string()).unwrap();
        let vitem_s = item_to_string(&vitem);
        assert_eq!(&vitem_s[..], use_s);
    })
}

#[test]
fn parse_extern_crate() {
    create_default_session_globals_then(|| {
        let ex_s = "extern crate foo;";
        let vitem = string_to_item(ex_s.to_string()).unwrap();
        let vitem_s = item_to_string(&vitem);
        assert_eq!(&vitem_s[..], ex_s);

        let ex_s = "extern crate foo as bar;";
        let vitem = string_to_item(ex_s.to_string()).unwrap();
        let vitem_s = item_to_string(&vitem);
        assert_eq!(&vitem_s[..], ex_s);
    })
}

fn get_spans_of_pat_idents(src: &str) -> Vec<Span> {
    let item = string_to_item(src.to_string()).unwrap();

    struct PatIdentVisitor {
        spans: Vec<Span>,
    }
    impl<'a> visit::Visitor<'a> for PatIdentVisitor {
        fn visit_pat(&mut self, p: &'a ast::Pat) {
            match &p.kind {
                PatKind::Ident(_, ident, _) => {
                    self.spans.push(ident.span);
                }
                _ => {
                    visit::walk_pat(self, p);
                }
            }
        }
    }
    let mut v = PatIdentVisitor { spans: Vec::new() };
    visit::walk_item(&mut v, &item);
    return v.spans;
}

#[test]
fn span_of_self_arg_pat_idents_are_correct() {
    create_default_session_globals_then(|| {
        let srcs = [
            "impl z { fn a (&self, &myarg: i32) {} }",
            "impl z { fn a (&mut self, &myarg: i32) {} }",
            "impl z { fn a (&'a self, &myarg: i32) {} }",
            "impl z { fn a (self, &myarg: i32) {} }",
            "impl z { fn a (self: Foo, &myarg: i32) {} }",
        ];

        for src in srcs {
            let spans = get_spans_of_pat_idents(src);
            let (lo, hi) = (spans[0].lo(), spans[0].hi());
            assert!(
                "self" == &src[lo.to_usize()..hi.to_usize()],
                "\"{}\" != \"self\". src=\"{}\"",
                &src[lo.to_usize()..hi.to_usize()],
                src
            )
        }
    })
}

#[test]
fn parse_exprs() {
    create_default_session_globals_then(|| {
        // just make sure that they parse....
        string_to_expr("3 + 4".to_string());
        string_to_expr("a::z.froob(b,&(987+3))".to_string());
    })
}

#[test]
fn attrs_fix_bug() {
    create_default_session_globals_then(|| {
        string_to_item(
            "pub fn mk_file_writer(path: &Path, flags: &[FileFlag])
                -> Result<Box<Writer>, String> {
#[cfg(windows)]
fn wb() -> c_int {
    (O_WRONLY | libc::consts::os::extra::O_BINARY) as c_int
}

#[cfg(unix)]
fn wb() -> c_int { O_WRONLY as c_int }

let mut fflags: c_int = wb();
}"
            .to_string(),
        );
    })
}

#[test]
fn crlf_doc_comments() {
    create_default_session_globals_then(|| {
        let psess = psess();

        let name_1 = FileName::Custom("crlf_source_1".to_string());
        let source = "/// doc comment\r\nfn foo() {}".to_string();
        let item = parse_item_from_source_str(name_1, source, &psess).unwrap().unwrap();
        let doc = item.attrs.iter().filter_map(|at| at.doc_str()).next().unwrap();
        assert_eq!(doc.as_str(), " doc comment");

        let name_2 = FileName::Custom("crlf_source_2".to_string());
        let source = "/// doc comment\r\n/// line 2\r\nfn foo() {}".to_string();
        let item = parse_item_from_source_str(name_2, source, &psess).unwrap().unwrap();
        let docs = item.attrs.iter().filter_map(|at| at.doc_str()).collect::<Vec<_>>();
        let b: &[_] = &[Symbol::intern(" doc comment"), Symbol::intern(" line 2")];
        assert_eq!(&docs[..], b);

        let name_3 = FileName::Custom("clrf_source_3".to_string());
        let source = "/** doc comment\r\n *  with CRLF */\r\nfn foo() {}".to_string();
        let item = parse_item_from_source_str(name_3, source, &psess).unwrap().unwrap();
        let doc = item.attrs.iter().filter_map(|at| at.doc_str()).next().unwrap();
        assert_eq!(doc.as_str(), " doc comment\n *  with CRLF ");
    });
}

#[test]
fn ttdelim_span() {
    fn parse_expr_from_source_str(
        name: FileName,
        source: String,
        psess: &ParseSess,
    ) -> PResult<'_, P<ast::Expr>> {
        unwrap_or_emit_fatal(new_parser_from_source_str(psess, name, source)).parse_expr()
    }

    create_default_session_globals_then(|| {
        let psess = psess();
        let expr = parse_expr_from_source_str(
            PathBuf::from("foo").into(),
            "foo!( fn main() { body } )".to_string(),
            &psess,
        )
        .unwrap();

        let ast::ExprKind::MacCall(mac) = &expr.kind else { panic!("not a macro") };
        let span = mac.args.tokens.iter().last().unwrap().span();

        match psess.source_map().span_to_snippet(span) {
            Ok(s) => assert_eq!(&s[..], "{ body }"),
            Err(_) => panic!("could not get snippet"),
        }
    });
}

#[track_caller]
fn look(p: &Parser<'_>, dist: usize, kind: rustc_ast::token::TokenKind) {
    // Do the `assert_eq` outside the closure so that `track_caller` works.
    // (`#![feature(closure_track_caller)]` + `#[track_caller]` on the closure
    // doesn't give the line number in the test below if the assertion fails.)
    let tok = p.look_ahead(dist, |tok| *tok);
    assert_eq!(kind, tok.kind);
}

#[test]
fn look_ahead() {
    create_default_session_globals_then(|| {
        let sym_f = Symbol::intern("f");
        let sym_x = Symbol::intern("x");
        #[allow(non_snake_case)]
        let sym_S = Symbol::intern("S");
        let raw_no = IdentIsRaw::No;

        let psess = psess();
        let mut p = string_to_parser(&psess, "fn f(x: u32) { x } struct S;".to_string());

        // Current position is the `fn`.
        look(&p, 0, token::Ident(kw::Fn, raw_no));
        look(&p, 1, token::Ident(sym_f, raw_no));
        look(&p, 2, token::OpenDelim(Delimiter::Parenthesis));
        look(&p, 3, token::Ident(sym_x, raw_no));
        look(&p, 4, token::Colon);
        look(&p, 5, token::Ident(sym::u32, raw_no));
        look(&p, 6, token::CloseDelim(Delimiter::Parenthesis));
        look(&p, 7, token::OpenDelim(Delimiter::Brace));
        look(&p, 8, token::Ident(sym_x, raw_no));
        look(&p, 9, token::CloseDelim(Delimiter::Brace));
        look(&p, 10, token::Ident(kw::Struct, raw_no));
        look(&p, 11, token::Ident(sym_S, raw_no));
        look(&p, 12, token::Semi);
        // Any lookahead past the end of the token stream returns `Eof`.
        look(&p, 13, token::Eof);
        look(&p, 14, token::Eof);
        look(&p, 15, token::Eof);
        look(&p, 100, token::Eof);

        // Move forward to the first `x`.
        for _ in 0..3 {
            p.bump();
        }
        look(&p, 0, token::Ident(sym_x, raw_no));
        look(&p, 1, token::Colon);
        look(&p, 2, token::Ident(sym::u32, raw_no));
        look(&p, 3, token::CloseDelim(Delimiter::Parenthesis));
        look(&p, 4, token::OpenDelim(Delimiter::Brace));
        look(&p, 5, token::Ident(sym_x, raw_no));
        look(&p, 6, token::CloseDelim(Delimiter::Brace));
        look(&p, 7, token::Ident(kw::Struct, raw_no));
        look(&p, 8, token::Ident(sym_S, raw_no));
        look(&p, 9, token::Semi);
        look(&p, 10, token::Eof);
        look(&p, 11, token::Eof);
        look(&p, 100, token::Eof);

        // Move forward to the `;`.
        for _ in 0..9 {
            p.bump();
        }
        look(&p, 0, token::Semi);
        // Any lookahead past the end of the token stream returns `Eof`.
        look(&p, 1, token::Eof);
        look(&p, 100, token::Eof);

        // Move one past the `;`, i.e. past the end of the token stream.
        p.bump();
        look(&p, 0, token::Eof);
        look(&p, 1, token::Eof);
        look(&p, 100, token::Eof);

        // Bumping after Eof is idempotent.
        p.bump();
        look(&p, 0, token::Eof);
        look(&p, 1, token::Eof);
        look(&p, 100, token::Eof);
    });
}

/// There used to be some buggy behaviour when using `look_ahead` not within
/// the outermost token stream, which this test covers.
#[test]
fn look_ahead_non_outermost_stream() {
    create_default_session_globals_then(|| {
        let sym_f = Symbol::intern("f");
        let sym_x = Symbol::intern("x");
        #[allow(non_snake_case)]
        let sym_S = Symbol::intern("S");
        let raw_no = IdentIsRaw::No;

        let psess = psess();
        let mut p = string_to_parser(&psess, "mod m { fn f(x: u32) { x } struct S; }".to_string());

        // Move forward to the `fn`, which is not within the outermost token
        // stream (because it's inside the `mod { ... }`).
        for _ in 0..3 {
            p.bump();
        }
        look(&p, 0, token::Ident(kw::Fn, raw_no));
        look(&p, 1, token::Ident(sym_f, raw_no));
        look(&p, 2, token::OpenDelim(Delimiter::Parenthesis));
        look(&p, 3, token::Ident(sym_x, raw_no));
        look(&p, 4, token::Colon);
        look(&p, 5, token::Ident(sym::u32, raw_no));
        look(&p, 6, token::CloseDelim(Delimiter::Parenthesis));
        look(&p, 7, token::OpenDelim(Delimiter::Brace));
        look(&p, 8, token::Ident(sym_x, raw_no));
        look(&p, 9, token::CloseDelim(Delimiter::Brace));
        look(&p, 10, token::Ident(kw::Struct, raw_no));
        look(&p, 11, token::Ident(sym_S, raw_no));
        look(&p, 12, token::Semi);
        look(&p, 13, token::CloseDelim(Delimiter::Brace));
        // Any lookahead past the end of the token stream returns `Eof`.
        look(&p, 14, token::Eof);
        look(&p, 15, token::Eof);
        look(&p, 100, token::Eof);
    });
}

// FIXME(nnethercote) All the output is currently wrong.
#[test]
fn debug_lookahead() {
    create_default_session_globals_then(|| {
        let psess = psess();
        let mut p = string_to_parser(&psess, "fn f(x: u32) { x } struct S;".to_string());

        // Current position is the `fn`.
        assert_eq!(
            &format!("{:#?}", p.debug_lookahead(0)),
            "Parser {
    prev_token: Token {
        kind: Question,
        span: Span {
            lo: BytePos(
                0,
            ),
            hi: BytePos(
                0,
            ),
            ctxt: #0,
        },
    },
    tokens: [],
    approx_token_stream_pos: 0,
    ..
}"
        );
        assert_eq!(
            &format!("{:#?}", p.debug_lookahead(7)),
            "Parser {
    prev_token: Token {
        kind: Question,
        span: Span {
            lo: BytePos(
                0,
            ),
            hi: BytePos(
                0,
            ),
            ctxt: #0,
        },
    },
    tokens: [
        Ident(
            \"fn\",
            No,
        ),
        Ident(
            \"f\",
            No,
        ),
        OpenDelim(
            Parenthesis,
        ),
        Ident(
            \"x\",
            No,
        ),
        Colon,
        Ident(
            \"u32\",
            No,
        ),
        CloseDelim(
            Parenthesis,
        ),
    ],
    approx_token_stream_pos: 0,
    ..
}"
        );
        // There are 13 tokens. We request 15, get 14; the last one is `Eof`.
        assert_eq!(
            &format!("{:#?}", p.debug_lookahead(15)),
            "Parser {
    prev_token: Token {
        kind: Question,
        span: Span {
            lo: BytePos(
                0,
            ),
            hi: BytePos(
                0,
            ),
            ctxt: #0,
        },
    },
    tokens: [
        Ident(
            \"fn\",
            No,
        ),
        Ident(
            \"f\",
            No,
        ),
        OpenDelim(
            Parenthesis,
        ),
        Ident(
            \"x\",
            No,
        ),
        Colon,
        Ident(
            \"u32\",
            No,
        ),
        CloseDelim(
            Parenthesis,
        ),
        OpenDelim(
            Brace,
        ),
        Ident(
            \"x\",
            No,
        ),
        CloseDelim(
            Brace,
        ),
        Ident(
            \"struct\",
            No,
        ),
        Ident(
            \"S\",
            No,
        ),
        Semi,
        Eof,
    ],
    approx_token_stream_pos: 0,
    ..
}"
        );

        // Move forward to the second `x`.
        for _ in 0..8 {
            p.bump();
        }
        assert_eq!(
            &format!("{:#?}", p.debug_lookahead(1)),
            "Parser {
    prev_token: Token {
        kind: OpenDelim(
            Brace,
        ),
        span: Span {
            lo: BytePos(
                13,
            ),
            hi: BytePos(
                14,
            ),
            ctxt: #0,
        },
    },
    tokens: [
        Ident(
            \"x\",
            No,
        ),
    ],
    approx_token_stream_pos: 8,
    ..
}"
        );
        assert_eq!(
            &format!("{:#?}", p.debug_lookahead(4)),
            "Parser {
    prev_token: Token {
        kind: OpenDelim(
            Brace,
        ),
        span: Span {
            lo: BytePos(
                13,
            ),
            hi: BytePos(
                14,
            ),
            ctxt: #0,
        },
    },
    tokens: [
        Ident(
            \"x\",
            No,
        ),
        CloseDelim(
            Brace,
        ),
        Ident(
            \"struct\",
            No,
        ),
        Ident(
            \"S\",
            No,
        ),
    ],
    approx_token_stream_pos: 8,
    ..
}"
        );

        // Move two past the final token (the `;`).
        for _ in 0..6 {
            p.bump();
        }
        assert_eq!(
            &format!("{:#?}", p.debug_lookahead(3)),
            "Parser {
    prev_token: Token {
        kind: Eof,
        span: Span {
            lo: BytePos(
                27,
            ),
            hi: BytePos(
                28,
            ),
            ctxt: #0,
        },
    },
    tokens: [
        Eof,
    ],
    approx_token_stream_pos: 14,
    ..
}"
        );
    });
}

// This tests that when parsing a string (rather than a file) we don't try
// and read in a file for a module declaration and just parse a stub.
// See `recurse_into_file_modules` in the parser.
#[test]
fn out_of_line_mod() {
    create_default_session_globals_then(|| {
        let item = parse_item_from_source_str(
            PathBuf::from("foo").into(),
            "mod foo { struct S; mod this_does_not_exist; }".to_owned(),
            &psess(),
        )
        .unwrap()
        .unwrap();

        let ast::ItemKind::Mod(_, _, mod_kind) = &item.kind else { panic!() };
        assert_matches!(mod_kind, ast::ModKind::Loaded(items, ..) if items.len() == 2);
    });
}

#[test]
fn eqmodws() {
    assert_eq!(matches_codepattern("", ""), true);
    assert_eq!(matches_codepattern("", "a"), false);
    assert_eq!(matches_codepattern("a", ""), false);
    assert_eq!(matches_codepattern("a", "a"), true);
    assert_eq!(matches_codepattern("a b", "a   \n\t\r  b"), true);
    assert_eq!(matches_codepattern("a b ", "a   \n\t\r  b"), true);
    assert_eq!(matches_codepattern("a b", "a   \n\t\r  b "), false);
    assert_eq!(matches_codepattern("a   b", "a b"), true);
    assert_eq!(matches_codepattern("ab", "a b"), false);
    assert_eq!(matches_codepattern("a   b", "ab"), true);
    assert_eq!(matches_codepattern(" a   b", "ab"), true);
}

#[test]
fn pattern_whitespace() {
    assert_eq!(matches_codepattern("", "\x0C"), false);
    assert_eq!(matches_codepattern("a b ", "a   \u{0085}\n\t\r  b"), true);
    assert_eq!(matches_codepattern("a b", "a   \u{0085}\n\t\r  b "), false);
}

#[test]
fn non_pattern_whitespace() {
    // These have the property 'White_Space' but not 'Pattern_White_Space'
    assert_eq!(matches_codepattern("a b", "a\u{2002}b"), false);
    assert_eq!(matches_codepattern("a   b", "a\u{2002}b"), false);
    assert_eq!(matches_codepattern("\u{205F}a   b", "ab"), false);
    assert_eq!(matches_codepattern("a  \u{3000}b", "ab"), false);
}
