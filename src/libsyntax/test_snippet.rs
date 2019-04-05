use crate::source_map::{SourceMap, FilePathMapping};
use crate::with_default_globals;

use errors::Handler;
use errors::emitter::EmitterWriter;

use std::io;
use std::io::prelude::*;
use rustc_data_structures::sync::Lrc;
use std::str;
use std::sync::{Arc, Mutex};
use std::path::Path;
use syntax_pos::{BytePos, NO_EXPANSION, Span, MultiSpan};

/// Identify a position in the text by the Nth occurrence of a string.
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

impl<T: Write> Write for Shared<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.data.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.data.lock().unwrap().flush()
    }
}

fn test_harness(file_text: &str, span_labels: Vec<SpanLabel>, expected_output: &str) {
    with_default_globals(|| {
        let output = Arc::new(Mutex::new(Vec::new()));

        let source_map = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        source_map.new_source_file(Path::new("test.rs").to_owned().into(), file_text.to_owned());

        let primary_span = make_span(&file_text, &span_labels[0].start, &span_labels[0].end);
        let mut msp = MultiSpan::from_span(primary_span);
        for span_label in span_labels {
            let span = make_span(&file_text, &span_label.start, &span_label.end);
            msp.push_span_label(span, span_label.label.to_string());
            println!("span: {:?} label: {:?}", span, span_label.label);
            println!("text: {:?}", source_map.span_to_snippet(span));
        }

        let emitter = EmitterWriter::new(Box::new(Shared { data: output.clone() }),
                                        Some(source_map.clone()),
                                        false,
                                        false,
                                        false);
        let handler = Handler::with_emitter(true, None, Box::new(emitter));
        handler.span_err(msp, "foo");

        assert!(expected_output.chars().next() == Some('\n'),
                "expected output should begin with newline");
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
    Span::new(BytePos(start as u32), BytePos(end as u32), NO_EXPANSION)
}

fn make_pos(file_text: &str, pos: &Position) -> usize {
    let mut remainder = file_text;
    let mut offset = 0;
    for _ in 0..pos.count {
        if let Some(n) = remainder.find(&pos.string) {
            offset += n;
            remainder = &remainder[n + 1..];
        } else {
            panic!("failed to find {} instances of {:?} in {:?}",
                   pos.count,
                   pos.string,
                   file_text);
        }
    }
    offset
}

#[test]
fn ends_on_col0() {
    test_harness(r#"
fn foo() {
}
"#,
    vec![
        SpanLabel {
           start: Position {
               string: "{",
               count: 1,
           },
           end: Position {
               string: "}",
               count: 1,
           },
           label: "test",
       },
    ],
    r#"
error: foo
 --> test.rs:2:10
  |
2 |   fn foo() {
  |  __________^
3 | | }
  | |_^ test

"#);
}

#[test]
fn ends_on_col2() {
    test_harness(r#"
fn foo() {


  }
"#,
     vec![
        SpanLabel {
            start: Position {
                string: "{",
                count: 1,
            },
            end: Position {
                string: "}",
                count: 1,
            },
            label: "test",
        },
     ],
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

"#);
}
#[test]
fn non_nested() {
    test_harness(r#"
fn foo() {
  X0 Y0
  X1 Y1
  X2 Y2
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "X0",
                count: 1,
            },
            end: Position {
                string: "X2",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Y0",
                count: 1,
            },
            end: Position {
                string: "Y2",
                count: 1,
            },
            label: "`Y` is a good letter too",
        },
    ],
    r#"
error: foo
 --> test.rs:3:3
  |
3 |      X0 Y0
  |  ____^__-
  | | ___|
  | ||
4 | ||   X1 Y1
5 | ||   X2 Y2
  | ||____^__- `Y` is a good letter too
  |  |____|
  |       `X` is a good letter

"#);
}

#[test]
fn nested() {
    test_harness(r#"
fn foo() {
  X0 Y0
  Y1 X1
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "X0",
                count: 1,
            },
            end: Position {
                string: "X1",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Y0",
                count: 1,
            },
            end: Position {
                string: "Y1",
                count: 1,
            },
            label: "`Y` is a good letter too",
        },
    ],
r#"
error: foo
 --> test.rs:3:3
  |
3 |      X0 Y0
  |  ____^__-
  | | ___|
  | ||
4 | ||   Y1 X1
  | ||____-__^ `X` is a good letter
  | |_____|
  |       `Y` is a good letter too

"#);
}

#[test]
fn different_overlap() {
    test_harness(r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "Y0",
                count: 1,
            },
            end: Position {
                string: "X2",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Z1",
                count: 1,
            },
            end: Position {
                string: "X3",
                count: 1,
            },
            label: "`Y` is a good letter too",
        },
    ],
    r#"
error: foo
 --> test.rs:3:6
  |
3 |      X0 Y0 Z0
  |   ______^
4 |  |   X1 Y1 Z1
  |  |_________-
5 | ||   X2 Y2 Z2
  | ||____^ `X` is a good letter
6 | |    X3 Y3 Z3
  | |_____- `Y` is a good letter too

"#);
}

#[test]
fn triple_overlap() {
    test_harness(r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "X0",
                count: 1,
            },
            end: Position {
                string: "X2",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Y0",
                count: 1,
            },
            end: Position {
                string: "Y2",
                count: 1,
            },
            label: "`Y` is a good letter too",
        },
        SpanLabel {
            start: Position {
                string: "Z0",
                count: 1,
            },
            end: Position {
                string: "Z2",
                count: 1,
            },
            label: "`Z` label",
        },
    ],
    r#"
error: foo
 --> test.rs:3:3
  |
3 |       X0 Y0 Z0
  |  _____^__-__-
  | | ____|__|
  | || ___|
  | |||
4 | |||   X1 Y1 Z1
5 | |||   X2 Y2 Z2
  | |||____^__-__- `Z` label
  |  ||____|__|
  |   |____|  `Y` is a good letter too
  |        `X` is a good letter

"#);
}

#[test]
fn triple_exact_overlap() {
    test_harness(r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "X0",
                count: 1,
            },
            end: Position {
                string: "X2",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "X0",
                count: 1,
            },
            end: Position {
                string: "X2",
                count: 1,
            },
            label: "`Y` is a good letter too",
        },
        SpanLabel {
            start: Position {
                string: "X0",
                count: 1,
            },
            end: Position {
                string: "X2",
                count: 1,
            },
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

"#);
}

#[test]
fn minimum_depth() {
    test_harness(r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "Y0",
                count: 1,
            },
            end: Position {
                string: "X1",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Y1",
                count: 1,
            },
            end: Position {
                string: "Z2",
                count: 1,
            },
            label: "`Y` is a good letter too",
        },
        SpanLabel {
            start: Position {
                string: "X2",
                count: 1,
            },
            end: Position {
                string: "Y3",
                count: 1,
            },
            label: "`Z`",
        },
    ],
    r#"
error: foo
 --> test.rs:3:6
  |
3 |      X0 Y0 Z0
  |   ______^
4 |  |   X1 Y1 Z1
  |  |____^_-
  | ||____|
  | |     `X` is a good letter
5 | |    X2 Y2 Z2
  | |____-______- `Y` is a good letter too
  |  ____|
  | |
6 | |    X3 Y3 Z3
  | |________- `Z`

"#);
}

#[test]
fn non_overlaping() {
    test_harness(r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "X0",
                count: 1,
            },
            end: Position {
                string: "X1",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Y2",
                count: 1,
            },
            end: Position {
                string: "Z3",
                count: 1,
            },
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

"#);
}

#[test]
fn overlaping_start_and_end() {
    test_harness(r#"
fn foo() {
  X0 Y0 Z0
  X1 Y1 Z1
  X2 Y2 Z2
  X3 Y3 Z3
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "Y0",
                count: 1,
            },
            end: Position {
                string: "X1",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Z1",
                count: 1,
            },
            end: Position {
                string: "Z3",
                count: 1,
            },
            label: "`Y` is a good letter too",
        },
    ],
    r#"
error: foo
 --> test.rs:3:6
  |
3 |      X0 Y0 Z0
  |   ______^
4 |  |   X1 Y1 Z1
  |  |____^____-
  | ||____|
  | |     `X` is a good letter
5 | |    X2 Y2 Z2
6 | |    X3 Y3 Z3
  | |___________- `Y` is a good letter too

"#);
}

#[test]
fn multiple_labels_primary_without_message() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "b",
                count: 1,
            },
            end: Position {
                string: "}",
                count: 1,
            },
            label: "",
        },
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "`a` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "c",
                count: 1,
            },
            end: Position {
                string: "c",
                count: 1,
            },
            label: "",
        },
    ],
    r#"
error: foo
 --> test.rs:3:7
  |
3 |   a { b { c } d }
  |   ----^^^^-^^-- `a` is a good letter

"#);
}

#[test]
fn multiple_labels_secondary_without_message() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "`a` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "b",
                count: 1,
            },
            end: Position {
                string: "}",
                count: 1,
            },
            label: "",
        },
    ],
    r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^ `a` is a good letter

"#);
}

#[test]
fn multiple_labels_primary_without_message_2() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "b",
                count: 1,
            },
            end: Position {
                string: "}",
                count: 1,
            },
            label: "`b` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "",
        },
        SpanLabel {
            start: Position {
                string: "c",
                count: 1,
            },
            end: Position {
                string: "c",
                count: 1,
            },
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

"#);
}

#[test]
fn multiple_labels_secondary_without_message_2() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "",
        },
        SpanLabel {
            start: Position {
                string: "b",
                count: 1,
            },
            end: Position {
                string: "}",
                count: 1,
            },
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

"#);
}

#[test]
fn multiple_labels_secondary_without_message_3() {
    test_harness(r#"
fn foo() {
  a  bc  d
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "b",
                count: 1,
            },
            label: "`a` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "c",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
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

"#);
}

#[test]
fn multiple_labels_without_message() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "",
        },
        SpanLabel {
            start: Position {
                string: "b",
                count: 1,
            },
            end: Position {
                string: "}",
                count: 1,
            },
            label: "",
        },
    ],
    r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^-------^^

"#);
}

#[test]
fn multiple_labels_without_message_2() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "b",
                count: 1,
            },
            end: Position {
                string: "}",
                count: 1,
            },
            label: "",
        },
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "",
        },
        SpanLabel {
            start: Position {
                string: "c",
                count: 1,
            },
            end: Position {
                string: "c",
                count: 1,
            },
            label: "",
        },
    ],
    r#"
error: foo
 --> test.rs:3:7
  |
3 |   a { b { c } d }
  |   ----^^^^-^^--

"#);
}

#[test]
fn multiple_labels_with_message() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "`a` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "b",
                count: 1,
            },
            end: Position {
                string: "}",
                count: 1,
            },
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

"#);
}

#[test]
fn single_label_with_message() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "`a` is a good letter",
        },
    ],
    r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^^^^^^^^^^ `a` is a good letter

"#);
}

#[test]
fn single_label_without_message() {
    test_harness(r#"
fn foo() {
  a { b { c } d }
}
"#,
    vec![
        SpanLabel {
            start: Position {
                string: "a",
                count: 1,
            },
            end: Position {
                string: "d",
                count: 1,
            },
            label: "",
        },
    ],
    r#"
error: foo
 --> test.rs:3:3
  |
3 |   a { b { c } d }
  |   ^^^^^^^^^^^^^

"#);
}

#[test]
fn long_snippet() {
    test_harness(r#"
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
            start: Position {
                string: "Y0",
                count: 1,
            },
            end: Position {
                string: "X1",
                count: 1,
            },
            label: "`X` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Z1",
                count: 1,
            },
            end: Position {
                string: "Z3",
                count: 1,
            },
            label: "`Y` is a good letter too",
        },
    ],
    r#"
error: foo
  --> test.rs:3:6
   |
3  |      X0 Y0 Z0
   |   ______^
4  |  |   X1 Y1 Z1
   |  |____^____-
   | ||____|
   | |     `X` is a good letter
5  | |  1
6  | |  2
7  | |  3
...  |
15 | |    X2 Y2 Z2
16 | |    X3 Y3 Z3
   | |___________- `Y` is a good letter too

"#);
}

#[test]
fn long_snippet_multiple_spans() {
    test_harness(r#"
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
            start: Position {
                string: "Y0",
                count: 1,
            },
            end: Position {
                string: "Y3",
                count: 1,
            },
            label: "`Y` is a good letter",
        },
        SpanLabel {
            start: Position {
                string: "Z1",
                count: 1,
            },
            end: Position {
                string: "Z2",
                count: 1,
            },
            label: "`Z` is a good letter too",
        },
    ],
    r#"
error: foo
  --> test.rs:3:6
   |
3  |      X0 Y0 Z0
   |   ______^
4  |  | 1
5  |  | 2
6  |  | 3
7  |  |   X1 Y1 Z1
   |  |_________-
8  | || 4
9  | || 5
10 | || 6
11 | ||   X2 Y2 Z2
   | ||__________- `Z` is a good letter too
...   |
15 |  | 10
16 |  |   X3 Y3 Z3
   |  |_______^ `Y` is a good letter

"#);
}
