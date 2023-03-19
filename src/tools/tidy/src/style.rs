//! Tidy check to enforce various stylistic guidelines on the Rust codebase.
//!
//! Example checks are:
//!
//! * No lines over 100 characters (in non-Rust files).
//! * No files with over 3000 lines (in non-Rust files).
//! * No tabs.
//! * No trailing whitespace.
//! * No CR characters.
//! * No `TODO` or `XXX` directives.
//! * No unexplained ` ```ignore ` or ` ```rust,ignore ` doc tests.
//!
//! Note that some of these rules are excluded from Rust files because we enforce rustfmt. It is
//! preferable to be formatted rather than tidy-clean.
//!
//! A number of these checks can be opted-out of with various directives of the form:
//! `// ignore-tidy-CHECK-NAME`.
// ignore-tidy-dbg

use crate::walk::{filter_dirs, walk};
use regex::{Regex, RegexSet};
use std::{ffi::OsStr, path::Path};

/// Error code markdown is restricted to 80 columns because they can be
/// displayed on the console with --example.
const ERROR_CODE_COLS: usize = 80;
const COLS: usize = 100;
const GOML_COLS: usize = 120;

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

const DOUBLE_SPACE_AFTER_DOT: &str = r"\
Use a single space after dots in comments.";

const ANNOTATIONS_TO_IGNORE: &[&str] = &[
    "// @!has",
    "// @has",
    "// @matches",
    "// CHECK",
    "// EMIT_MIR",
    "// compile-flags",
    "// error-pattern",
    "// gdb",
    "// lldb",
    "// cdb",
    "// normalize-stderr-test",
];

// Intentionally written in decimal rather than hex
const PROBLEMATIC_CONSTS: &[u32] = &[
    184594741, 2880289470, 2881141438, 2965027518, 2976579765, 3203381950, 3405691582, 3405697037,
    3735927486, 3735932941, 4027431614, 4276992702,
];

const INTERNAL_COMPILER_DOCS_LINE: &str = "#### This error code is internal to the compiler and will not be emitted with normal Rust code.";

/// Parser states for `line_is_url`.
#[derive(Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
enum LIUState {
    EXP_COMMENT_START,
    EXP_LINK_LABEL_OR_URL,
    EXP_URL,
    EXP_END,
}

/// Returns `true` if `line` appears to be a line comment containing a URL,
/// possibly with a Markdown link label in front, and nothing else.
/// The Markdown link label, if present, may not contain whitespace.
/// Lines of this form are allowed to be overlength, because Markdown
/// offers no way to split a line in the middle of a URL, and the lengths
/// of URLs to external references are beyond our control.
fn line_is_url(is_error_code: bool, columns: usize, line: &str) -> bool {
    // more basic check for markdown, to avoid complexity in implementing two state machines
    if is_error_code {
        return line.starts_with('[') && line.contains("]:") && line.contains("http");
    }

    use self::LIUState::*;
    let mut state: LIUState = EXP_COMMENT_START;
    let is_url = |w: &str| w.starts_with("http://") || w.starts_with("https://");

    for tok in line.split_whitespace() {
        match (state, tok) {
            (EXP_COMMENT_START, "//") | (EXP_COMMENT_START, "///") | (EXP_COMMENT_START, "//!") => {
                state = EXP_LINK_LABEL_OR_URL
            }

            (EXP_LINK_LABEL_OR_URL, w)
                if w.len() >= 4 && w.starts_with('[') && w.ends_with("]:") =>
            {
                state = EXP_URL
            }

            (EXP_LINK_LABEL_OR_URL, w) if is_url(w) => state = EXP_END,

            (EXP_URL, w) if is_url(w) || w.starts_with("../") => state = EXP_END,

            (_, w) if w.len() > columns && is_url(w) => state = EXP_END,

            (_, _) => {}
        }
    }

    state == EXP_END
}

/// Returns `true` if `line` can be ignored. This is the case when it contains
/// an annotation that is explicitly ignored.
fn should_ignore(line: &str) -> bool {
    // Matches test annotations like `//~ ERROR text`.
    // This mirrors the regex in src/tools/compiletest/src/runtest.rs, please
    // update both if either are changed.
    let re = Regex::new("\\s*//(\\[.*\\])?~.*").unwrap();
    re.is_match(line) || ANNOTATIONS_TO_IGNORE.iter().any(|a| line.contains(a))
}

/// Returns `true` if `line` is allowed to be longer than the normal limit.
fn long_line_is_ok(extension: &str, is_error_code: bool, max_columns: usize, line: &str) -> bool {
    match extension {
        // fluent files are allowed to be any length
        "ftl" => true,
        // non-error code markdown is allowed to be any length
        "md" if !is_error_code => true,
        // HACK(Ezrashaw): there is no way to split a markdown header over multiple lines
        "md" if line == INTERNAL_COMPILER_DOCS_LINE => true,
        _ => line_is_url(is_error_code, max_columns, line) || should_ignore(line),
    }
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
    if contents.contains(&format!("// ignore-tidy-{check}"))
        || contents.contains(&format!("# ignore-tidy-{check}"))
        || contents.contains(&format!("/* ignore-tidy-{check} */"))
    {
        Directive::Ignore(false)
    } else {
        Directive::Deny
    }
}

macro_rules! suppressible_tidy_err {
    ($err:ident, $skip:ident, $msg:literal) => {
        if let Directive::Deny = $skip {
            $err(&format!($msg));
        } else {
            $skip = Directive::Ignore(true);
        }
    };
}

pub fn is_in(full_path: &Path, parent_folder_to_find: &str, folder_to_find: &str) -> bool {
    if let Some(parent) = full_path.parent() {
        if parent.file_name().map_or_else(
            || false,
            |f| {
                f.to_string_lossy() == folder_to_find
                    && parent
                        .parent()
                        .and_then(|f| f.file_name())
                        .map_or_else(|| false, |f| f == parent_folder_to_find)
            },
        ) {
            true
        } else {
            is_in(parent, parent_folder_to_find, folder_to_find)
        }
    } else {
        false
    }
}

fn skip_markdown_path(path: &Path) -> bool {
    // These aren't ready for tidy.
    const SKIP_MD: &[&str] = &[
        "src/doc/edition-guide",
        "src/doc/embedded-book",
        "src/doc/nomicon",
        "src/doc/reference",
        "src/doc/rust-by-example",
        "src/doc/rustc-dev-guide",
    ];
    SKIP_MD.iter().any(|p| path.ends_with(p))
}

fn is_unexplained_ignore(extension: &str, line: &str) -> bool {
    if !line.ends_with("```ignore") && !line.ends_with("```rust,ignore") {
        return false;
    }
    if extension == "md" && line.trim().starts_with("//") {
        // Markdown examples may include doc comments with ignore inside a
        // code block.
        return false;
    }
    true
}

pub fn check(path: &Path, bad: &mut bool) {
    fn skip(path: &Path) -> bool {
        if path.file_name().map_or(false, |name| name.to_string_lossy().starts_with(".#")) {
            // vim or emacs temporary file
            return true;
        }

        if filter_dirs(path) || skip_markdown_path(path) {
            return true;
        }

        let extensions = ["rs", "py", "js", "sh", "c", "cpp", "h", "md", "css", "ftl", "goml"];
        if extensions.iter().all(|e| path.extension() != Some(OsStr::new(e))) {
            return true;
        }

        // We only check CSS files in rustdoc.
        path.extension().map_or(false, |e| e == "css") && !is_in(path, "src", "librustdoc")
    }

    let problematic_consts_strings: Vec<String> = (PROBLEMATIC_CONSTS.iter().map(u32::to_string))
        .chain(PROBLEMATIC_CONSTS.iter().map(|v| format!("{:x}", v)))
        .chain(PROBLEMATIC_CONSTS.iter().map(|v| format!("{:X}", v)))
        .collect();
    let problematic_regex = RegexSet::new(problematic_consts_strings.as_slice()).unwrap();

    walk(path, skip, &mut |entry, contents| {
        let file = entry.path();
        let filename = file.file_name().unwrap().to_string_lossy();

        let is_style_file = filename.ends_with(".css");
        let under_rustfmt = filename.ends_with(".rs") &&
            // This list should ideally be sourced from rustfmt.toml but we don't want to add a toml
            // parser to tidy.
            !file.ancestors().any(|a| {
                (a.ends_with("tests") && a.join("COMPILER_TESTS.md").exists()) ||
                    a.ends_with("src/doc/book")
            });

        if contents.is_empty() {
            tidy_error!(bad, "{}: empty file", file.display());
        }

        let extension = file.extension().unwrap().to_string_lossy();
        let is_error_code = extension == "md" && is_in(file, "src", "error_codes");
        let is_goml_code = extension == "goml";

        let max_columns = if is_error_code {
            ERROR_CODE_COLS
        } else if is_goml_code {
            GOML_COLS
        } else {
            COLS
        };

        let can_contain = contents.contains("// ignore-tidy-")
            || contents.contains("# ignore-tidy-")
            || contents.contains("/* ignore-tidy-");
        // Enable testing ICE's that require specific (untidy)
        // file formats easily eg. `issue-1234-ignore-tidy.rs`
        if filename.contains("ignore-tidy") {
            return;
        }
        // apfloat shouldn't be changed because of license problems
        if is_in(file, "compiler", "rustc_apfloat") {
            return;
        }
        let mut skip_cr = contains_ignore_directive(can_contain, &contents, "cr");
        let mut skip_undocumented_unsafe =
            contains_ignore_directive(can_contain, &contents, "undocumented-unsafe");
        let mut skip_tab = contains_ignore_directive(can_contain, &contents, "tab");
        let mut skip_line_length = contains_ignore_directive(can_contain, &contents, "linelength");
        let mut skip_file_length = contains_ignore_directive(can_contain, &contents, "filelength");
        let mut skip_end_whitespace =
            contains_ignore_directive(can_contain, &contents, "end-whitespace");
        let mut skip_trailing_newlines =
            contains_ignore_directive(can_contain, &contents, "trailing-newlines");
        let mut skip_leading_newlines =
            contains_ignore_directive(can_contain, &contents, "leading-newlines");
        let mut skip_copyright = contains_ignore_directive(can_contain, &contents, "copyright");
        let mut skip_dbg = contains_ignore_directive(can_contain, &contents, "dbg");
        let mut skip_odd_backticks =
            contains_ignore_directive(can_contain, &contents, "odd-backticks");
        let mut leading_new_lines = false;
        let mut trailing_new_lines = 0;
        let mut lines = 0;
        let mut last_safety_comment = false;
        let mut comment_block: Option<(usize, usize)> = None;
        let is_test = file.components().any(|c| c.as_os_str() == "tests");
        // scanning the whole file for multiple needles at once is more efficient than
        // executing lines times needles separate searches.
        let any_problematic_line = problematic_regex.is_match(contents);
        for (i, line) in contents.split('\n').enumerate() {
            if line.is_empty() {
                if i == 0 {
                    leading_new_lines = true;
                }
                trailing_new_lines += 1;
                continue;
            } else {
                trailing_new_lines = 0;
            }

            let trimmed = line.trim();

            if !trimmed.starts_with("//") {
                lines += 1;
            }

            let mut err = |msg: &str| {
                tidy_error!(bad, "{}:{}: {}", file.display(), i + 1, msg);
            };

            if trimmed.contains("dbg!")
                && !trimmed.starts_with("//")
                && !file.ancestors().any(|a| {
                    (a.ends_with("tests") && a.join("COMPILER_TESTS.md").exists())
                        || a.ends_with("library/alloc/tests")
                })
                && filename != "tests.rs"
            {
                suppressible_tidy_err!(
                    err,
                    skip_dbg,
                    "`dbg!` macro is intended as a debugging tool. It should not be in version control."
                )
            }

            if !under_rustfmt
                && line.chars().count() > max_columns
                && !long_line_is_ok(&extension, is_error_code, max_columns, line)
            {
                suppressible_tidy_err!(
                    err,
                    skip_line_length,
                    "line longer than {max_columns} chars"
                );
            }
            if !is_style_file && line.contains('\t') {
                suppressible_tidy_err!(err, skip_tab, "tab character");
            }
            if line.ends_with(' ') || line.ends_with('\t') {
                suppressible_tidy_err!(err, skip_end_whitespace, "trailing whitespace");
            }
            if is_style_file && line.starts_with(' ') {
                err("CSS files use tabs for indent");
            }
            if line.contains('\r') {
                suppressible_tidy_err!(err, skip_cr, "CR character");
            }
            if filename != "style.rs" {
                if trimmed.contains("TODO") {
                    err("TODO is deprecated; use FIXME")
                }
                if trimmed.contains("//") && trimmed.contains(" XXX") {
                    err("XXX is deprecated; use FIXME")
                }
                if any_problematic_line {
                    for s in problematic_consts_strings.iter() {
                        if trimmed.contains(s) {
                            err("Don't use magic numbers that spell things (consider 0x12345678)");
                        }
                    }
                }
            }
            // for now we just check libcore
            if trimmed.contains("unsafe {") && !trimmed.starts_with("//") && !last_safety_comment {
                if file.components().any(|c| c.as_os_str() == "core") && !is_test {
                    suppressible_tidy_err!(err, skip_undocumented_unsafe, "undocumented unsafe");
                }
            }
            if trimmed.contains("// SAFETY:") {
                last_safety_comment = true;
            } else if trimmed.starts_with("//") || trimmed.is_empty() {
                // keep previous value
            } else {
                last_safety_comment = false;
            }
            if (line.starts_with("// Copyright")
                || line.starts_with("# Copyright")
                || line.starts_with("Copyright"))
                && (trimmed.contains("Rust Developers")
                    || trimmed.contains("Rust Project Developers"))
            {
                suppressible_tidy_err!(
                    err,
                    skip_copyright,
                    "copyright notices attributed to the Rust Project Developers are deprecated"
                );
            }
            if is_unexplained_ignore(&extension, line) {
                err(UNEXPLAINED_IGNORE_DOCTEST_INFO);
            }
            if filename.ends_with(".cpp") && line.contains("llvm_unreachable") {
                err(LLVM_UNREACHABLE_INFO);
            }

            // For now only enforce in compiler
            let is_compiler = || file.components().any(|c| c.as_os_str() == "compiler");

            if is_compiler() {
                if line.contains("//")
                    && line
                        .chars()
                        .collect::<Vec<_>>()
                        .windows(4)
                        .any(|cs| matches!(cs, ['.', ' ', ' ', last] if last.is_alphabetic()))
                {
                    err(DOUBLE_SPACE_AFTER_DOT)
                }

                if filename.ends_with(".ftl") {
                    let line_backticks = trimmed.chars().filter(|ch| *ch == '`').count();
                    if line_backticks % 2 == 1 {
                        suppressible_tidy_err!(err, skip_odd_backticks, "odd number of backticks");
                    }
                } else if trimmed.contains("//") {
                    let (start_line, mut backtick_count) = comment_block.unwrap_or((i + 1, 0));
                    let line_backticks = trimmed.chars().filter(|ch| *ch == '`').count();
                    let comment_text = trimmed.split("//").nth(1).unwrap();
                    // This check ensures that we don't lint for code that has `//` in a string literal
                    if line_backticks % 2 == 1 {
                        backtick_count += comment_text.chars().filter(|ch| *ch == '`').count();
                    }
                    comment_block = Some((start_line, backtick_count));
                } else {
                    if let Some((start_line, backtick_count)) = comment_block.take() {
                        if backtick_count % 2 == 1 {
                            let mut err = |msg: &str| {
                                tidy_error!(bad, "{}:{start_line}: {msg}", file.display());
                            };
                            let block_len = (i + 1) - start_line;
                            if block_len == 1 {
                                suppressible_tidy_err!(
                                    err,
                                    skip_odd_backticks,
                                    "comment with odd number of backticks"
                                );
                            } else {
                                suppressible_tidy_err!(
                                    err,
                                    skip_odd_backticks,
                                    "{block_len}-line comment block with odd number of backticks"
                                );
                            }
                        }
                    }
                }
            }
        }
        if leading_new_lines {
            let mut err = |_| {
                tidy_error!(bad, "{}: leading newline", file.display());
            };
            suppressible_tidy_err!(err, skip_leading_newlines, "mising leading newline");
        }
        let mut err = |msg: &str| {
            tidy_error!(bad, "{}: {}", file.display(), msg);
        };
        match trailing_new_lines {
            0 => suppressible_tidy_err!(err, skip_trailing_newlines, "missing trailing newline"),
            1 => {}
            n => suppressible_tidy_err!(
                err,
                skip_trailing_newlines,
                "too many trailing newlines ({n})"
            ),
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
        if let Directive::Ignore(false) = skip_end_whitespace {
            tidy_error!(bad, "{}: ignoring trailing whitespace unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_trailing_newlines {
            tidy_error!(bad, "{}: ignoring trailing newlines unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_leading_newlines {
            tidy_error!(bad, "{}: ignoring leading newlines unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_copyright {
            tidy_error!(bad, "{}: ignoring copyright unnecessarily", file.display());
        }
        // We deliberately do not warn about these being unnecessary,
        // that would just lead to annoying churn.
        let _unused = skip_line_length;
        let _unused = skip_file_length;
    })
}
