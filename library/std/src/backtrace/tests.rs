use super::*;

#[test]
fn test_debug() {
    let backtrace = Backtrace {
        inner: Inner::Captured(LazilyResolvedCapture::new(Capture {
            actual_start: 1,
            resolved: true,
            frames: vec![
                BacktraceFrame {
                    frame: RawFrame::Fake,
                    symbols: vec![BacktraceSymbol {
                        name: Some(b"std::backtrace::Backtrace::create".to_vec()),
                        filename: Some(BytesOrWide::Bytes(b"rust/backtrace.rs".to_vec())),
                        lineno: Some(100),
                        colno: None,
                    }],
                },
                BacktraceFrame {
                    frame: RawFrame::Fake,
                    symbols: vec![BacktraceSymbol {
                        name: Some(b"__rust_maybe_catch_panic".to_vec()),
                        filename: None,
                        lineno: None,
                        colno: None,
                    }],
                },
                BacktraceFrame {
                    frame: RawFrame::Fake,
                    symbols: vec![
                        BacktraceSymbol {
                            name: Some(b"std::rt::lang_start_internal".to_vec()),
                            filename: Some(BytesOrWide::Bytes(b"rust/rt.rs".to_vec())),
                            lineno: Some(300),
                            colno: Some(5),
                        },
                        BacktraceSymbol {
                            name: Some(b"std::rt::lang_start".to_vec()),
                            filename: Some(BytesOrWide::Bytes(b"rust/rt.rs".to_vec())),
                            lineno: Some(400),
                            colno: None,
                        },
                    ],
                },
            ],
        })),
    };

    #[rustfmt::skip]
    let expected = "Backtrace [\
    \n    { fn: \"__rust_maybe_catch_panic\" },\
    \n    { fn: \"std::rt::lang_start_internal\", file: \"rust/rt.rs\", line: 300 },\
    \n    { fn: \"std::rt::lang_start\", file: \"rust/rt.rs\", line: 400 },\
    \n]";

    assert_eq!(format!("{:#?}", backtrace), expected);

    // Format the backtrace a second time, just to make sure lazily resolved state is stable
    assert_eq!(format!("{:#?}", backtrace), expected);
}
