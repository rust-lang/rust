use std::io::{BufRead, BufReader, Read};
use std::path::Path;

pub mod directives;

#[derive(Debug, Clone, Copy)]
/// Represents a single line comment in a test header.
pub struct TestComment<'line> {
    revision: Option<&'line str>,
    comment: CommentKind<'line>,
    line_num: usize,
    full_line: &'line str,
}

impl<'line> TestComment<'line> {
    pub const fn new(
        line: &'line str,
        revision: Option<&'line str>,
        comment: CommentKind<'line>,
        line_num: usize,
    ) -> Self {
        Self { revision, comment, line_num, full_line: line }
    }

    pub const fn revision(&self) -> Option<&str> {
        self.revision
    }

    pub const fn comment(&self) -> CommentKind<'_> {
        self.comment
    }

    pub const fn line_num(&self) -> usize {
        self.line_num
    }

    pub const fn comment_str(&self) -> &str {
        self.comment.line()
    }

    /// The full line that contains the comment. You almost never want this.
    pub const fn full_line(&self) -> &str {
        self.full_line
    }
}

#[derive(Debug, Clone, Copy)]
/// What sort of comment the header is, and the full contents after the comment start.
pub enum CommentKind<'line> {
    /// Comments understood by compiletest. These are comments in rust files that start with // or
    /// comments in non-rust files.
    Compiletest(&'line str),
    /// Comments understood by ui_test. These are only comments in rust files that start with //@.
    UiTest(&'line str),
}

impl CommentKind<'_> {
    pub const fn line(&self) -> &str {
        match self {
            CommentKind::Compiletest(line) | CommentKind::UiTest(line) => line,
        }
    }
}

/// Iterates over the test header for the given `testfile`, and calls a closure
/// with each parsed test header comment.
pub fn iter_header<R: Read>(testfile: &Path, rdr: R, it: &mut dyn FnMut(TestComment<'_>)) {
    if testfile.is_dir() {
        return;
    }

    let is_rust_file = testfile.extension().is_some_and(|e| e == "rs");

    let mut rdr = BufReader::new(rdr);
    let mut full_ln = String::new();
    let mut line_num = 0;

    loop {
        line_num += 1;
        full_ln.clear();
        if rdr.read_line(&mut full_ln).unwrap() == 0 {
            break;
        }

        // the different types of comments recognized
        const NON_RUST_COMMENT: &str = "#";
        const RUST_UI_TEST_COMMENT: &str = "//@";
        const RUST_COMPILETEST_COMMENT: &str = "//";

        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        let ln = full_ln.trim();
        if ln.starts_with("fn") || ln.starts_with("mod") {
            return;
        } else if is_rust_file {
            // first try to parse as a ui test comment, then as a compiletest comment
            if let Some((lncfg, ln)) = line_directive(RUST_UI_TEST_COMMENT, ln) {
                let directive =
                    TestComment::new(full_ln.as_str(), lncfg, CommentKind::UiTest(ln), line_num);
                it(directive);
            } else if let Some((lncfg, ln)) = line_directive(RUST_COMPILETEST_COMMENT, ln) {
                let directive = TestComment::new(
                    full_ln.as_str(),
                    lncfg,
                    CommentKind::Compiletest(ln),
                    line_num,
                );
                it(directive);
            }
        } else {
            // parse a non-rust file that compiletest knows about
            if let Some((lncfg, ln)) = line_directive(NON_RUST_COMMENT, ln) {
                let directive = TestComment::new(
                    full_ln.as_str(),
                    lncfg,
                    CommentKind::Compiletest(ln),
                    line_num,
                );
                it(directive);
            }
        }
    }
}

pub fn line_directive<'line>(
    comment_start: &str,
    ln: &'line str,
) -> Option<(Option<&'line str>, &'line str)> {
    if ln.starts_with(comment_start) {
        let ln = ln[comment_start.len()..].trim_start();
        if ln.starts_with('[') {
            // A comment like `//[foo]` is specific to revision `foo`
            if let Some(close_brace) = ln.find(']') {
                let lncfg = &ln[1..close_brace];

                Some((Some(lncfg), ln[(close_brace + 1)..].trim_start()))
            } else {
                panic!(
                    "malformed condition directive: expected `{}[foo]`, found `{}`",
                    comment_start, ln
                )
            }
        } else {
            Some((None, ln))
        }
    } else {
        None
    }
}
