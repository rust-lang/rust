// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use codemap::CodeMap;
use errors::Handler;
use errors::emitter::EmitterWriter;
use std::io;
use std::io::prelude::*;
use std::rc::Rc;
use std::str;
use std::sync::{Arc, Mutex};
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
    let output = Arc::new(Mutex::new(Vec::new()));

    let code_map = Rc::new(CodeMap::new());
    code_map.new_filemap_and_lines("test.rs", None, &file_text);

    let primary_span = make_span(&file_text, &span_labels[0].start, &span_labels[0].end);
    let mut msp = MultiSpan::from_span(primary_span);
    for span_label in span_labels {
        let span = make_span(&file_text, &span_label.start, &span_label.end);
        msp.push_span_label(span, span_label.label.to_string());
        println!("span: {:?} label: {:?}", span, span_label.label);
        println!("text: {:?}", code_map.span_to_snippet(span));
    }

    let emitter = EmitterWriter::new(Box::new(Shared { data: output.clone() }),
                                     Some(code_map.clone()));
    let handler = Handler::with_emitter(true, false, Box::new(emitter));
    handler.span_err(msp, "foo");

    assert!(expected_output.chars().next() == Some('\n'),
            "expected output should begin with newline");
    let expected_output = &expected_output[1..];

    let bytes = output.lock().unwrap();
    let actual_output = str::from_utf8(&bytes).unwrap();
    println!("expected output:\n------\n{}------", expected_output);
    println!("actual output:\n------\n{}------", actual_output);

    assert!(expected_output == actual_output)
}

fn make_span(file_text: &str, start: &Position, end: &Position) -> Span {
    let start = make_pos(file_text, start);
    let end = make_pos(file_text, end) + end.string.len(); // just after matching thing ends
    assert!(start <= end);
    Span {
        lo: BytePos(start as u32),
        hi: BytePos(end as u32),
        expn_id: NO_EXPANSION,
    }
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
  |  __________^ starting here...
3 | | }
  | |_^ ...ending here: test

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
  |  __________^ starting here...
3 | |
4 | |
5 | |   }
  | |___^ ...ending here: test

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
  |  ____^__- starting here...
  | | ___|
  | ||   starting here...
4 | ||   X1 Y1
5 | ||   X2 Y2
  | ||____^__- ...ending here: `Y` is a good letter too
  |  |____|
  |       ...ending here: `X` is a good letter

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
  |  ____^__- starting here...
  | | ___|
  | ||   starting here...
4 | ||   Y1 X1
  | ||____-__^ ...ending here: `X` is a good letter
  | |_____|
  |       ...ending here: `Y` is a good letter too

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
  |   ______^ starting here...
4 |  |   X1 Y1 Z1
  |  |_________- starting here...
5 | ||   X2 Y2 Z2
  | ||____^ ...ending here: `X` is a good letter
6 | |    X3 Y3 Z3
  | |_____- ...ending here: `Y` is a good letter too

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
  |  _____^__-__- starting here...
  | | ____|__|
  | || ___|  starting here...
  | |||   starting here...
4 | |||   X1 Y1 Z1
5 | |||   X2 Y2 Z2
  | |||____^__-__- ...ending here: `Z` label
  |  ||____|__|
  |   |____|  ...ending here: `Y` is a good letter too
  |        ...ending here: `X` is a good letter

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
  |   ______^ starting here...
4 |  |   X1 Y1 Z1
  |  |____^_- starting here...
  | ||____|
  | |     ...ending here: `X` is a good letter
5 | |    X2 Y2 Z2
  | |____-______- ...ending here: `Y` is a good letter too
  |  ____|
  | |    starting here...
6 | |    X3 Y3 Z3
  | |________- ...ending here: `Z`

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
 --> test.rs:3:6
  |
3 |     X0 Y0 Z0
  |  ______^ starting here...
4 | |   X1 Y1 Z1
  | |____^ ...ending here: `X` is a good letter
5 |     X2 Y2 Z2
  |  ______- starting here...
6 | |   X3 Y3 Z3
  | |__________- ...ending here: `Y` is a good letter too

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
  |   ______^ starting here...
4 |  |   X1 Y1 Z1
  |  |____^____- starting here...
  | ||____|
  | |     ...ending here: `X` is a good letter
5 | |    X2 Y2 Z2
6 | |    X3 Y3 Z3
  | |___________- ...ending here: `Y` is a good letter too

"#);
}
