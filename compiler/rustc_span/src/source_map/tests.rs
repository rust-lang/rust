use super::*;

fn init_source_map() -> SourceMap {
    let sm = SourceMap::new(FilePathMapping::empty());
    sm.new_source_file(PathBuf::from("blork.rs").into(), "first line.\nsecond line".to_string());
    sm.new_source_file(PathBuf::from("empty.rs").into(), String::new());
    sm.new_source_file(PathBuf::from("blork2.rs").into(), "first line.\nsecond line".to_string());
    sm
}

impl SourceMap {
    /// Returns `Some(span)`, a union of the LHS and RHS span. The LHS must precede the RHS. If
    /// there are gaps between LHS and RHS, the resulting union will cross these gaps.
    /// For this to work,
    ///
    ///    * the syntax contexts of both spans much match,
    ///    * the LHS span needs to end on the same line the RHS span begins,
    ///    * the LHS span must start at or before the RHS span.
    fn merge_spans(&self, sp_lhs: Span, sp_rhs: Span) -> Option<Span> {
        // Ensure we're at the same expansion ID.
        if !sp_lhs.eq_ctxt(sp_rhs) {
            return None;
        }

        let lhs_end = match self.lookup_line(sp_lhs.hi()) {
            Ok(x) => x,
            Err(_) => return None,
        };
        let rhs_begin = match self.lookup_line(sp_rhs.lo()) {
            Ok(x) => x,
            Err(_) => return None,
        };

        // If we must cross lines to merge, don't merge.
        if lhs_end.line != rhs_begin.line {
            return None;
        }

        // Ensure these follow the expected order and that we don't overlap.
        if (sp_lhs.lo() <= sp_rhs.lo()) && (sp_lhs.hi() <= sp_rhs.lo()) {
            Some(sp_lhs.to(sp_rhs))
        } else {
            None
        }
    }

    /// Converts an absolute `BytePos` to a `CharPos` relative to the `SourceFile`.
    fn bytepos_to_file_charpos(&self, bpos: BytePos) -> CharPos {
        let idx = self.lookup_source_file_idx(bpos);
        let sf = &(*self.files.borrow().source_files)[idx];
        let bpos = sf.relative_position(bpos);
        sf.bytepos_to_file_charpos(bpos)
    }
}

/// Tests `lookup_byte_offset`.
#[test]
fn t3() {
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

/// Tests `bytepos_to_file_charpos`.
#[test]
fn t4() {
    let sm = init_source_map();

    let cp1 = sm.bytepos_to_file_charpos(BytePos(22));
    assert_eq!(cp1, CharPos(22));

    let cp2 = sm.bytepos_to_file_charpos(BytePos(25));
    assert_eq!(cp2, CharPos(0));
}

/// Tests zero-length `SourceFile`s.
#[test]
fn t5() {
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
    // "€" is a three-byte UTF8 char.
    sm.new_source_file(
        PathBuf::from("blork.rs").into(),
        "fir€st €€€€ line.\nsecond line".to_string(),
    );
    sm.new_source_file(
        PathBuf::from("blork2.rs").into(),
        "first line€€.\n€ second line".to_string(),
    );
    sm
}

/// Tests `bytepos_to_file_charpos` in the presence of multi-byte chars.
#[test]
fn t6() {
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

/// Test `span_to_lines` for a span ending at the end of a `SourceFile`.
#[test]
fn t7() {
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
    let right_index = selection.rfind('~').map_or(left_index, |x| x as u32);
    Span::with_root_ctxt(BytePos(left_index), BytePos(right_index + 1))
}

/// Tests `span_to_snippet` and `span_to_lines` for a span converting 3
/// lines in the middle of a file.
#[test]
fn span_to_snippet_and_lines_spanning_multiple_lines() {
    let sm = SourceMap::new(FilePathMapping::empty());
    let inputtext = "aaaaa\nbbbbBB\nCCC\nDDDDDddddd\neee\n";
    let selection = "     \n    ~~\n~~~\n~~~~~     \n   \n";
    sm.new_source_file(Path::new("blork.rs").to_owned().into(), inputtext.to_string());
    let span = span_from_selection(inputtext, selection);

    // Check that we are extracting the text we thought we were extracting.
    assert_eq!(&sm.span_to_snippet(span).unwrap(), "BB\nCCC\nDDDDD");

    // Check that span_to_lines gives us the complete result with the lines/cols we expected.
    let lines = sm.span_to_lines(span).unwrap();
    let expected = vec![
        LineInfo { line_index: 1, start_col: CharPos(4), end_col: CharPos(6) },
        LineInfo { line_index: 2, start_col: CharPos(0), end_col: CharPos(3) },
        LineInfo { line_index: 3, start_col: CharPos(0), end_col: CharPos(5) },
    ];
    assert_eq!(lines.lines, expected);
}

/// Test span_to_snippet for a span ending at the end of a `SourceFile`.
#[test]
fn t8() {
    let sm = init_source_map();
    let span = Span::with_root_ctxt(BytePos(12), BytePos(23));
    let snippet = sm.span_to_snippet(span);

    assert_eq!(snippet, Ok("second line".to_string()));
}

/// Test `span_to_str` for a span ending at the end of a `SourceFile`.
#[test]
fn t9() {
    let sm = init_source_map();
    let span = Span::with_root_ctxt(BytePos(12), BytePos(23));
    let sstr = sm.span_to_diagnostic_string(span);

    assert_eq!(sstr, "blork.rs:2:1: 2:12");
}

/// Tests failing to merge two spans on different lines.
#[test]
fn span_merging_fail() {
    let sm = SourceMap::new(FilePathMapping::empty());
    let inputtext = "bbbb BB\ncc CCC\n";
    let selection1 = "     ~~\n      \n";
    let selection2 = "       \n   ~~~\n";
    sm.new_source_file(Path::new("blork.rs").to_owned().into(), inputtext.to_owned());
    let span1 = span_from_selection(inputtext, selection1);
    let span2 = span_from_selection(inputtext, selection2);

    assert!(sm.merge_spans(span1, span2).is_none());
}

/// Tests loading an external source file that requires normalization.
#[test]
fn t10() {
    let sm = SourceMap::new(FilePathMapping::empty());
    let unnormalized = "first line.\r\nsecond line";
    let normalized = "first line.\nsecond line";

    let src_file = sm.new_source_file(PathBuf::from("blork.rs").into(), unnormalized.to_string());

    assert_eq!(src_file.src.as_ref().unwrap().as_ref(), normalized);
    assert!(
        src_file.src_hash.matches(unnormalized),
        "src_hash should use the source before normalization"
    );

    let SourceFile {
        name,
        src_hash,
        source_len,
        lines,
        multibyte_chars,
        normalized_pos,
        stable_id,
        ..
    } = (*src_file).clone();

    let imported_src_file = sm.new_imported_source_file(
        name,
        src_hash,
        stable_id,
        source_len.to_u32(),
        CrateNum::ZERO,
        FreezeLock::new(lines.read().clone()),
        multibyte_chars,
        normalized_pos,
        0,
    );

    assert!(
        imported_src_file.external_src.borrow().get_source().is_none(),
        "imported source file should not have source yet"
    );
    imported_src_file.add_external_src(|| Some(unnormalized.to_string()));
    assert_eq!(
        imported_src_file.external_src.borrow().get_source().unwrap().as_ref(),
        normalized,
        "imported source file should be normalized"
    );
}

// Takes a unix-style path and returns a platform specific path.
fn path(p: &str) -> PathBuf {
    path_str(p).into()
}

// Takes a unix-style path and returns a platform specific path.
fn path_str(p: &str) -> String {
    #[cfg(not(windows))]
    {
        return p.into();
    }

    #[cfg(windows)]
    {
        let mut path = p.replace('/', "\\");
        if let Some(rest) = path.strip_prefix('\\') {
            path = ["X:\\", rest].concat();
        }

        path
    }
}

fn map_path_prefix(mapping: &FilePathMapping, p: &str) -> String {
    // It's important that we convert to a string here because that's what
    // later stages do too (e.g. in the backend), and comparing `Path` values
    // won't catch some differences at the string level, e.g. "abc" and "abc/"
    // compare as equal.
    mapping.map_prefix(path(p)).0.to_string_lossy().to_string()
}

fn reverse_map_prefix(mapping: &FilePathMapping, p: &str) -> Option<String> {
    mapping.reverse_map_prefix_heuristically(&path(p)).map(|q| q.to_string_lossy().to_string())
}

#[test]
fn path_prefix_remapping() {
    // Relative to relative
    {
        let mapping = &FilePathMapping::new(
            vec![(path("abc/def"), path("foo"))],
            FileNameDisplayPreference::Remapped,
        );

        assert_eq!(map_path_prefix(mapping, "abc/def/src/main.rs"), path_str("foo/src/main.rs"));
        assert_eq!(map_path_prefix(mapping, "abc/def"), path_str("foo"));
    }

    // Relative to absolute
    {
        let mapping = &FilePathMapping::new(
            vec![(path("abc/def"), path("/foo"))],
            FileNameDisplayPreference::Remapped,
        );

        assert_eq!(map_path_prefix(mapping, "abc/def/src/main.rs"), path_str("/foo/src/main.rs"));
        assert_eq!(map_path_prefix(mapping, "abc/def"), path_str("/foo"));
    }

    // Absolute to relative
    {
        let mapping = &FilePathMapping::new(
            vec![(path("/abc/def"), path("foo"))],
            FileNameDisplayPreference::Remapped,
        );

        assert_eq!(map_path_prefix(mapping, "/abc/def/src/main.rs"), path_str("foo/src/main.rs"));
        assert_eq!(map_path_prefix(mapping, "/abc/def"), path_str("foo"));
    }

    // Absolute to absolute
    {
        let mapping = &FilePathMapping::new(
            vec![(path("/abc/def"), path("/foo"))],
            FileNameDisplayPreference::Remapped,
        );

        assert_eq!(map_path_prefix(mapping, "/abc/def/src/main.rs"), path_str("/foo/src/main.rs"));
        assert_eq!(map_path_prefix(mapping, "/abc/def"), path_str("/foo"));
    }
}

#[test]
fn path_prefix_remapping_expand_to_absolute() {
    // "virtual" working directory is relative path
    let mapping = &FilePathMapping::new(
        vec![(path("/foo"), path("FOO")), (path("/bar"), path("BAR"))],
        FileNameDisplayPreference::Remapped,
    );
    let working_directory = path("/foo");
    let working_directory = RealFileName::Remapped {
        local_path: Some(working_directory.clone()),
        virtual_name: mapping.map_prefix(working_directory).0.into_owned(),
    };

    assert_eq!(working_directory.remapped_path_if_available(), path("FOO"));

    // Unmapped absolute path
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::LocalPath(path("/foo/src/main.rs")),
            &working_directory
        ),
        RealFileName::Remapped { local_path: None, virtual_name: path("FOO/src/main.rs") }
    );

    // Unmapped absolute path with unrelated working directory
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::LocalPath(path("/bar/src/main.rs")),
            &working_directory
        ),
        RealFileName::Remapped { local_path: None, virtual_name: path("BAR/src/main.rs") }
    );

    // Unmapped absolute path that does not match any prefix
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::LocalPath(path("/quux/src/main.rs")),
            &working_directory
        ),
        RealFileName::LocalPath(path("/quux/src/main.rs")),
    );

    // Unmapped relative path
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::LocalPath(path("src/main.rs")),
            &working_directory
        ),
        RealFileName::Remapped { local_path: None, virtual_name: path("FOO/src/main.rs") }
    );

    // Unmapped relative path with `./`
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::LocalPath(path("./src/main.rs")),
            &working_directory
        ),
        RealFileName::Remapped { local_path: None, virtual_name: path("FOO/src/main.rs") }
    );

    // Unmapped relative path that does not match any prefix
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::LocalPath(path("quux/src/main.rs")),
            &RealFileName::LocalPath(path("/abc")),
        ),
        RealFileName::LocalPath(path("/abc/quux/src/main.rs")),
    );

    // Already remapped absolute path
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::Remapped {
                local_path: Some(path("/foo/src/main.rs")),
                virtual_name: path("FOO/src/main.rs"),
            },
            &working_directory
        ),
        RealFileName::Remapped { local_path: None, virtual_name: path("FOO/src/main.rs") }
    );

    // Already remapped absolute path, with unrelated working directory
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::Remapped {
                local_path: Some(path("/bar/src/main.rs")),
                virtual_name: path("BAR/src/main.rs"),
            },
            &working_directory
        ),
        RealFileName::Remapped { local_path: None, virtual_name: path("BAR/src/main.rs") }
    );

    // Already remapped relative path
    assert_eq!(
        mapping.to_embeddable_absolute_path(
            RealFileName::Remapped { local_path: None, virtual_name: path("XYZ/src/main.rs") },
            &working_directory
        ),
        RealFileName::Remapped { local_path: None, virtual_name: path("XYZ/src/main.rs") }
    );
}

#[test]
fn path_prefix_remapping_reverse() {
    // Ignores options without alphanumeric chars.
    {
        let mapping = &FilePathMapping::new(
            vec![(path("abc"), path("/")), (path("def"), path("."))],
            FileNameDisplayPreference::Remapped,
        );

        assert_eq!(reverse_map_prefix(mapping, "/hello.rs"), None);
        assert_eq!(reverse_map_prefix(mapping, "./hello.rs"), None);
    }

    // Returns `None` if multiple options match.
    {
        let mapping = &FilePathMapping::new(
            vec![(path("abc"), path("/redacted")), (path("def"), path("/redacted"))],
            FileNameDisplayPreference::Remapped,
        );

        assert_eq!(reverse_map_prefix(mapping, "/redacted/hello.rs"), None);
    }

    // Distinct reverse mappings.
    {
        let mapping = &FilePathMapping::new(
            vec![(path("abc"), path("/redacted")), (path("def/ghi"), path("/fake/dir"))],
            FileNameDisplayPreference::Remapped,
        );

        assert_eq!(
            reverse_map_prefix(mapping, "/redacted/path/hello.rs"),
            Some(path_str("abc/path/hello.rs"))
        );
        assert_eq!(
            reverse_map_prefix(mapping, "/fake/dir/hello.rs"),
            Some(path_str("def/ghi/hello.rs"))
        );
    }
}

#[test]
fn test_next_point() {
    let sm = SourceMap::new(FilePathMapping::empty());
    sm.new_source_file(PathBuf::from("example.rs").into(), "a…b".to_string());

    // Dummy spans don't advance.
    let span = DUMMY_SP;
    let span = sm.next_point(span);
    assert_eq!(span.lo().0, 0);
    assert_eq!(span.hi().0, 0);

    // Span advance respect multi-byte character
    let span = Span::with_root_ctxt(BytePos(0), BytePos(1));
    assert_eq!(sm.span_to_snippet(span), Ok("a".to_string()));
    let span = sm.next_point(span);
    assert_eq!(sm.span_to_snippet(span), Ok("…".to_string()));
    assert_eq!(span.lo().0, 1);
    assert_eq!(span.hi().0, 4);

    // An empty span pointing just before a multi-byte character should
    // advance to contain the multi-byte character.
    let span = Span::with_root_ctxt(BytePos(1), BytePos(1));
    let span = sm.next_point(span);
    assert_eq!(span.lo().0, 1);
    assert_eq!(span.hi().0, 4);

    let span = Span::with_root_ctxt(BytePos(1), BytePos(4));
    let span = sm.next_point(span);
    assert_eq!(span.lo().0, 4);
    assert_eq!(span.hi().0, 5);

    // Reaching to the end of file, return a span that will get error with `span_to_snippet`
    let span = Span::with_root_ctxt(BytePos(4), BytePos(5));
    let span = sm.next_point(span);
    assert_eq!(span.lo().0, 5);
    assert_eq!(span.hi().0, 6);
    assert!(sm.span_to_snippet(span).is_err());

    // Reaching to the end of file, return a span that will get error with `span_to_snippet`
    let span = Span::with_root_ctxt(BytePos(5), BytePos(5));
    let span = sm.next_point(span);
    assert_eq!(span.lo().0, 5);
    assert_eq!(span.hi().0, 6);
    assert!(sm.span_to_snippet(span).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn read_binary_file_handles_lying_stat() {
    // read_binary_file tries to read the contents of a file into an Lrc<[u8]> while
    // never having two copies of the data in memory at once. This is an optimization
    // to support include_bytes! with large files. But since Rust allocators are
    // sensitive to alignment, our implementation can't be bootstrapped off calling
    // std::fs::read. So we test that we have the same behavior even on files where
    // fs::metadata lies.

    // stat always says that /proc/self/cmdline is length 0, but it isn't.
    let cmdline = Path::new("/proc/self/cmdline");
    let len = std::fs::metadata(cmdline).unwrap().len() as usize;
    let real = std::fs::read(cmdline).unwrap();
    assert!(len < real.len());
    let bin = RealFileLoader.read_binary_file(cmdline).unwrap();
    assert_eq!(&real[..], &bin[..]);

    // stat always says that /sys/devices/system/cpu/kernel_max is the size of a block.
    let kernel_max = Path::new("/sys/devices/system/cpu/kernel_max");
    let len = std::fs::metadata(kernel_max).unwrap().len() as usize;
    let real = std::fs::read(kernel_max).unwrap();
    assert!(len > real.len());
    let bin = RealFileLoader.read_binary_file(kernel_max).unwrap();
    assert_eq!(&real[..], &bin[..]);
}
