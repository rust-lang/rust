use super::*;

use crate::json::JsonEmitter;
use rustc_span::source_map::{FilePathMapping, SourceMap};

use crate::emitter::{ColorConfig, HumanReadableErrorType};
use crate::Handler;
use rustc_serialize::json::decode;
use rustc_span::{BytePos, Span};

use std::str;

#[derive(Decodable, Debug, PartialEq, Eq)]
struct TestData {
    spans: Vec<SpanTestData>,
}

#[derive(Decodable, Debug, PartialEq, Eq)]
struct SpanTestData {
    pub byte_start: u32,
    pub byte_end: u32,
    pub line_start: u32,
    pub column_start: u32,
    pub line_end: u32,
    pub column_end: u32,
}

struct Shared<T> {
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

/// Test the span yields correct positions in JSON.
fn test_positions(code: &str, span: (u32, u32), expected_output: SpanTestData) {
    let expected_output = TestData { spans: vec![expected_output] };

    rustc_span::create_default_session_globals_then(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        sm.new_source_file(Path::new("test.rs").to_owned().into(), code.to_owned());

        let output = Arc::new(Mutex::new(Vec::new()));
        let je = JsonEmitter::new(
            Box::new(Shared { data: output.clone() }),
            None,
            sm,
            true,
            HumanReadableErrorType::Short(ColorConfig::Never),
            None,
            false,
        );

        let span = Span::with_root_ctxt(BytePos(span.0), BytePos(span.1));
        let handler = Handler::with_emitter(true, None, Box::new(je));
        handler.span_err(span, "foo");

        let bytes = output.lock().unwrap();
        let actual_output = str::from_utf8(&bytes).unwrap();
        let actual_output: TestData = decode(actual_output);

        assert_eq!(expected_output, actual_output)
    })
}

#[test]
fn empty() {
    test_positions(
        " ",
        (0, 1),
        SpanTestData {
            byte_start: 0,
            byte_end: 1,
            line_start: 1,
            column_start: 1,
            line_end: 1,
            column_end: 2,
        },
    )
}

#[test]
fn bom() {
    test_positions(
        "\u{feff} ",
        (0, 1),
        SpanTestData {
            byte_start: 3,
            byte_end: 4,
            line_start: 1,
            column_start: 1,
            line_end: 1,
            column_end: 2,
        },
    )
}

#[test]
fn lf_newlines() {
    test_positions(
        "\nmod foo;\nmod bar;\n",
        (5, 12),
        SpanTestData {
            byte_start: 5,
            byte_end: 12,
            line_start: 2,
            column_start: 5,
            line_end: 3,
            column_end: 3,
        },
    )
}

#[test]
fn crlf_newlines() {
    test_positions(
        "\r\nmod foo;\r\nmod bar;\r\n",
        (5, 12),
        SpanTestData {
            byte_start: 6,
            byte_end: 14,
            line_start: 2,
            column_start: 5,
            line_end: 3,
            column_end: 3,
        },
    )
}

#[test]
fn crlf_newlines_with_bom() {
    test_positions(
        "\u{feff}\r\nmod foo;\r\nmod bar;\r\n",
        (5, 12),
        SpanTestData {
            byte_start: 9,
            byte_end: 17,
            line_start: 2,
            column_start: 5,
            line_end: 3,
            column_end: 3,
        },
    )
}

#[test]
fn span_before_crlf() {
    test_positions(
        "foo\r\nbar",
        (2, 3),
        SpanTestData {
            byte_start: 2,
            byte_end: 3,
            line_start: 1,
            column_start: 3,
            line_end: 1,
            column_end: 4,
        },
    )
}

#[test]
fn span_on_crlf() {
    test_positions(
        "foo\r\nbar",
        (3, 4),
        SpanTestData {
            byte_start: 3,
            byte_end: 5,
            line_start: 1,
            column_start: 4,
            line_end: 2,
            column_end: 1,
        },
    )
}

#[test]
fn span_after_crlf() {
    test_positions(
        "foo\r\nbar",
        (4, 5),
        SpanTestData {
            byte_start: 5,
            byte_end: 6,
            line_start: 2,
            column_start: 1,
            line_end: 2,
            column_end: 2,
        },
    )
}
