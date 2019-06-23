//! Tidy check to enforce various stylistic guidelines on the Rust codebase.
//!
//! Example checks are:
//!
//! * No lines over 100 characters.
//! * No files with over 3000 lines.
//! * No tabs.
//! * No trailing whitespace.
//! * No CR characters.
//! * No `TODO` or `XXX` directives.
//! * No unexplained ` ```ignore ` or ` ```rust,ignore ` doc tests.
//!
//! A number of these checks can be opted-out of with various directives of the form:
//! `// ignore-tidy-CHECK-NAME`.

use std::path::Path;

const COLS: usize = 100;

const LINES: usize = 3000;

const UNEXPLAINED_IGNORE_DOCTEST_INFO: &str = r#"unexplained "```ignore" doctest; try one:

* make the test actually pass, by adding necessary imports and declarations, or
* use "```text", if the code is not Rust code, or
* use "```compile_fail,Ennnn", if the code is expected to fail at compile time, or
* use "```should_panic", if the code is expected to fail at run time, or
* use "```no_run", if the code should type-check but not necessary linkable/runnable, or
* explain it like "```ignore (cannot-test-this-because-xxxx)", if the annotation cannot be avoided.

"#;

const LLVM_UNREACHABLE_INFO: &str = r"\
C++ code used llvm_unreachable, which triggers undefined behavior
when executed when assertions are disabled.
Use llvm::report_fatal_error for increased robustness.";

/// Parser states for `line_is_url`.
#[derive(Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
enum LIUState {
    EXP_COMMENT_START,
    EXP_LINK_LABEL_OR_URL,
    EXP_URL,
    EXP_END,
}

/// Returns `true` if `line` appears to be a line comment containing an URL,
/// possibly with a Markdown link label in front, and nothing else.
/// The Markdown link label, if present, may not contain whitespace.
/// Lines of this form are allowed to be overlength, because Markdown
/// offers no way to split a line in the middle of a URL, and the lengths
/// of URLs to external references are beyond our control.
fn line_is_url(line: &str) -> bool {
    use self::LIUState::*;
    let mut state: LIUState = EXP_COMMENT_START;
    let is_url = |w: &str| w.starts_with("http://") || w.starts_with("https://");

    for tok in line.split_whitespace() {
        match (state, tok) {
            (EXP_COMMENT_START, "//") |
            (EXP_COMMENT_START, "///") |
            (EXP_COMMENT_START, "//!") => state = EXP_LINK_LABEL_OR_URL,

            (EXP_LINK_LABEL_OR_URL, w)
                if w.len() >= 4 && w.starts_with('[') && w.ends_with("]:")
                => state = EXP_URL,

            (EXP_LINK_LABEL_OR_URL, w)
                if is_url(w)
                => state = EXP_END,

            (EXP_URL, w)
                if is_url(w) || w.starts_with("../")
                => state = EXP_END,

            (_, w)
                if w.len() > COLS && is_url(w)
                => state = EXP_END,

            (_, _) => {}
        }
    }

    state == EXP_END
}

/// Returns `true` if `line` is allowed to be longer than the normal limit.
/// Currently there is only one exception, for long URLs, but more
/// may be added in the future.
fn long_line_is_ok(line: &str) -> bool {
    if line_is_url(line) {
        return true;
    }

    false
}

enum Directive {
    /// By default, tidy always warns against style issues.
    Deny,

    /// `Ignore(false)` means that an `ignore-tidy-*` directive
    /// has been provided, but is unnecessary. `Ignore(true)`
    /// means that it is necessary (i.e. a warning would be
    /// produced if `ignore-tidy-*` was not present).
    Ignore(bool),
}

fn contains_ignore_directive(can_contain: bool, contents: &str, check: &str) -> Directive {
    if !can_contain {
        return Directive::Deny;
    }
    // Update `can_contain` when changing this
    if contents.contains(&format!("// ignore-tidy-{}", check)) ||
        contents.contains(&format!("# ignore-tidy-{}", check)) {
        Directive::Ignore(false)
    } else {
        Directive::Deny
    }
}

macro_rules! suppressible_tidy_err {
    ($err:ident, $skip:ident, $msg:expr) => {
        if let Directive::Deny = $skip {
            $err($msg);
        } else {
            $skip = Directive::Ignore(true);
        }
    };
}

pub fn check(path: &Path, bad: &mut bool) {
    super::walk(path, &mut super::filter_dirs, &mut |entry, contents| {
        let file = entry.path();
        let filename = file.file_name().unwrap().to_string_lossy();
        let extensions = [".rs", ".py", ".js", ".sh", ".c", ".cpp", ".h"];
        if extensions.iter().all(|e| !filename.ends_with(e)) ||
           filename.starts_with(".#") {
            return
        }

        if contents.is_empty() {
            tidy_error!(bad, "{}: empty file", file.display());
        }

        let can_contain = contents.contains("// ignore-tidy-") ||
            contents.contains("# ignore-tidy-");
        let mut skip_cr = contains_ignore_directive(can_contain, &contents, "cr");
        let mut skip_tab = contains_ignore_directive(can_contain, &contents, "tab");
        let mut skip_line_length = contains_ignore_directive(can_contain, &contents, "linelength");
        let mut skip_file_length = contains_ignore_directive(can_contain, &contents, "filelength");
        let mut skip_end_whitespace =
            contains_ignore_directive(can_contain, &contents, "end-whitespace");
        let mut skip_copyright = contains_ignore_directive(can_contain, &contents, "copyright");
        let mut leading_new_lines = false;
        let mut trailing_new_lines = 0;
        let mut lines = 0;
        for (i, line) in contents.split('\n').enumerate() {
            let mut err = |msg: &str| {
                tidy_error!(bad, "{}:{}: {}", file.display(), i + 1, msg);
            };
            if line.chars().count() > COLS && !long_line_is_ok(line) {
                suppressible_tidy_err!(
                    err,
                    skip_line_length,
                    &format!("line longer than {} chars", COLS)
                );
            }
            if line.contains('\t') {
                suppressible_tidy_err!(err, skip_tab, "tab character");
            }
            if line.ends_with(' ') || line.ends_with('\t') {
                suppressible_tidy_err!(err, skip_end_whitespace, "trailing whitespace");
            }
            if line.contains('\r') {
                suppressible_tidy_err!(err, skip_cr, "CR character");
            }
            if filename != "style.rs" {
                if line.contains("TODO") {
                    err("TODO is deprecated; use FIXME")
                }
                if line.contains("//") && line.contains(" XXX") {
                    err("XXX is deprecated; use FIXME")
                }
            }
            if (line.starts_with("// Copyright") ||
                line.starts_with("# Copyright") ||
                line.starts_with("Copyright"))
                && (line.contains("Rust Developers") ||
                    line.contains("Rust Project Developers")) {
                suppressible_tidy_err!(
                    err,
                    skip_copyright,
                    "copyright notices attributed to the Rust Project Developers are deprecated"
                );
            }
            if line.ends_with("```ignore") || line.ends_with("```rust,ignore") {
                err(UNEXPLAINED_IGNORE_DOCTEST_INFO);
            }
            if filename.ends_with(".cpp") && line.contains("llvm_unreachable") {
                err(LLVM_UNREACHABLE_INFO);
            }
            if line.is_empty() {
                if i == 0 {
                    leading_new_lines = true;
                }
                trailing_new_lines += 1;
            } else {
                trailing_new_lines = 0;
            }
            lines = i;
        }
        if leading_new_lines {
            tidy_error!(bad, "{}: leading newline", file.display());
        }
        match trailing_new_lines {
            0 => tidy_error!(bad, "{}: missing trailing newline", file.display()),
            1 => {}
            n => tidy_error!(bad, "{}: too many trailing newlines ({})", file.display(), n),
        };
        if lines > LINES {
            let mut err = |_| {
                tidy_error!(
                    bad,
                    "{}: too many lines ({}) (add `// \
                     ignore-tidy-filelength` to the file to suppress this error)",
                    file.display(),
                    lines
                );
            };
            suppressible_tidy_err!(err, skip_file_length, "");
        }

        if let Directive::Ignore(false) = skip_cr {
            tidy_error!(bad, "{}: ignoring CR characters unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_tab {
            tidy_error!(bad, "{}: ignoring tab characters unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_line_length {
            tidy_error!(bad, "{}: ignoring line length unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_file_length {
            tidy_error!(bad, "{}: ignoring file length unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_end_whitespace {
            tidy_error!(bad, "{}: ignoring trailing whitespace unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_copyright {
            tidy_error!(bad, "{}: ignoring copyright unnecessarily", file.display());
        }
    })
}
