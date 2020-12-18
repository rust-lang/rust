use super::*;

fn generate_fake_backtrace() -> Backtrace {
    Backtrace {
        inner: Inner::Captured(Mutex::new(Capture {
            actual_start: 1,
            resolved: true,
            frames: vec![
                BacktraceFrame {
                    frame: RawFrame::Fake,
                    symbols: vec![BacktraceSymbol {
                        name: Some(b"std::backtrace::Backtrace::create".to_vec()),
                        filename: Some(BytesOrWide::Bytes(b"rust/backtrace.rs".to_vec())),
                        lineno: Some(100),
                    }],
                },
                BacktraceFrame {
                    frame: RawFrame::Fake,
                    symbols: vec![BacktraceSymbol {
                        name: Some(b"__rust_maybe_catch_panic".to_vec()),
                        filename: None,
                        lineno: None,
                    }],
                },
                BacktraceFrame {
                    frame: RawFrame::Fake,
                    symbols: vec![
                        BacktraceSymbol {
                            name: Some(b"std::rt::lang_start_internal".to_vec()),
                            filename: Some(BytesOrWide::Bytes(b"rust/rt.rs".to_vec())),
                            lineno: Some(300),
                        },
                        BacktraceSymbol {
                            name: Some(b"std::rt::lang_start".to_vec()),
                            filename: Some(BytesOrWide::Bytes(b"rust/rt.rs".to_vec())),
                            lineno: Some(400),
                        },
                    ],
                },
            ],
        })),
    }
}

#[test]
fn test_debug() {
    let backtrace = generate_fake_backtrace();

    #[rustfmt::skip]
    let expected = "Backtrace [\
    \n    { fn: \"__rust_maybe_catch_panic\" },\
    \n    { fn: \"std::rt::lang_start_internal\", file: \"rust/rt.rs\", line: 300 },\
    \n    { fn: \"std::rt::lang_start\", file: \"rust/rt.rs\", line: 400 },\
    \n]";

    assert_eq!(format!("{:#?}", backtrace), expected);
}

#[test]
fn test_empty_frames_iterator() {
    let empty_backtrace = Backtrace {
        inner: Inner::Captured(Mutex::new(Capture {
            actual_start: 1,
            resolved: true,
            frames: vec![],
        }))
    };

    let iter = empty_backtrace.frames(); 

    assert_eq!(iter.count(), 0);
}

#[test]
fn test_frames_iterator() {
    let backtrace = generate_fake_backtrace();
    let iter = backtrace.frames();

    assert_eq!(iter.count(), 3);
}
