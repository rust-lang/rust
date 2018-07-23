// High level formatting functions.

use std::collections::HashMap;
use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::time::Duration;

use syntax::ast;
use syntax::codemap::{CodeMap, FilePathMapping, Span};
use syntax::errors::emitter::{ColorConfig, EmitterWriter};
use syntax::errors::{DiagnosticBuilder, Handler};
use syntax::parse::{self, ParseSess};

use comment::{CharClasses, FullCodeCharKind};
use issues::BadIssueSeeker;
use visitor::{FmtVisitor, SnippetProvider};
use {filemap, modules, ErrorKind, FormatReport, Input};

use config::summary::Summary;
use config::{Config, FileName, NewlineStyle, Verbosity};

// A map of the files of a crate, with their new content
pub(crate) type FileMap = Vec<FileRecord>;

pub(crate) type FileRecord = (FileName, String);

pub(crate) struct FormattingError {
    pub(crate) line: usize,
    pub(crate) kind: ErrorKind,
    is_comment: bool,
    is_string: bool,
    pub(crate) line_buffer: String,
}

impl FormattingError {
    pub(crate) fn from_span(span: &Span, codemap: &CodeMap, kind: ErrorKind) -> FormattingError {
        FormattingError {
            line: codemap.lookup_char_pos(span.lo()).line,
            is_comment: kind.is_comment(),
            kind,
            is_string: false,
            line_buffer: codemap
                .span_to_lines(*span)
                .ok()
                .and_then(|fl| {
                    fl.file
                        .get_line(fl.lines[0].line_index)
                        .map(|l| l.into_owned())
                })
                .unwrap_or_else(|| String::new()),
        }
    }

    pub(crate) fn msg_prefix(&self) -> &str {
        match self.kind {
            ErrorKind::LineOverflow(..)
            | ErrorKind::TrailingWhitespace
            | ErrorKind::IoError(_)
            | ErrorKind::ParseError
            | ErrorKind::LostComment => "internal error:",
            ErrorKind::LicenseCheck | ErrorKind::BadAttr | ErrorKind::VersionMismatch => "error:",
            ErrorKind::BadIssue(_) | ErrorKind::DeprecatedAttr => "warning:",
        }
    }

    pub(crate) fn msg_suffix(&self) -> &str {
        if self.is_comment || self.is_string {
            "set `error_on_unformatted = false` to suppress \
             the warning against comments or string literals\n"
        } else {
            ""
        }
    }

    // (space, target)
    pub(crate) fn format_len(&self) -> (usize, usize) {
        match self.kind {
            ErrorKind::LineOverflow(found, max) => (max, found - max),
            ErrorKind::TrailingWhitespace
            | ErrorKind::DeprecatedAttr
            | ErrorKind::BadAttr
            | ErrorKind::LostComment => {
                let trailing_ws_start = self
                    .line_buffer
                    .rfind(|c: char| !c.is_whitespace())
                    .map(|pos| pos + 1)
                    .unwrap_or(0);
                (
                    trailing_ws_start,
                    self.line_buffer.len() - trailing_ws_start,
                )
            }
            _ => unreachable!(),
        }
    }
}

pub(crate) type FormatErrorMap = HashMap<FileName, Vec<FormattingError>>;

#[derive(Default, Debug)]
pub(crate) struct ReportedErrors {
    pub(crate) has_operational_errors: bool,
    pub(crate) has_check_errors: bool,
}

fn should_emit_verbose<F>(path: &FileName, config: &Config, f: F)
where
    F: Fn(),
{
    if config.verbose() == Verbosity::Verbose && path != &FileName::Stdin {
        f();
    }
}

// Formatting which depends on the AST.
fn format_ast<F>(
    krate: &ast::Crate,
    parse_session: &mut ParseSess,
    main_file: &FileName,
    config: &Config,
    report: FormatReport,
    mut after_file: F,
) -> Result<(FileMap, bool, bool), io::Error>
where
    F: FnMut(&FileName, &mut String, &[(usize, usize)], &FormatReport) -> Result<bool, io::Error>,
{
    let mut result = FileMap::new();
    // diff mode: check if any files are differing
    let mut has_diff = false;
    let mut has_macro_rewrite_failure = false;

    let skip_children = config.skip_children();
    for (path, module) in modules::list_files(krate, parse_session.codemap())? {
        if (skip_children && path != *main_file) || config.ignore().skip_file(&path) {
            continue;
        }
        should_emit_verbose(&path, config, || println!("Formatting {}", path));
        let filemap = parse_session
            .codemap()
            .lookup_char_pos(module.inner.lo())
            .file;
        let big_snippet = filemap.src.as_ref().unwrap();
        let snippet_provider = SnippetProvider::new(filemap.start_pos, big_snippet);
        let mut visitor =
            FmtVisitor::from_codemap(parse_session, config, &snippet_provider, report.clone());
        // Format inner attributes if available.
        if !krate.attrs.is_empty() && path == *main_file {
            visitor.skip_empty_lines(filemap.end_pos);
            if visitor.visit_attrs(&krate.attrs, ast::AttrStyle::Inner) {
                visitor.push_rewrite(module.inner, None);
            } else {
                visitor.format_separate_mod(module, &*filemap);
            }
        } else {
            visitor.last_pos = filemap.start_pos;
            visitor.skip_empty_lines(filemap.end_pos);
            visitor.format_separate_mod(module, &*filemap);
        };

        debug_assert_eq!(
            visitor.line_number,
            ::utils::count_newlines(&visitor.buffer)
        );

        has_diff |= match after_file(&path, &mut visitor.buffer, &visitor.skipped_range, &report) {
            Ok(result) => result,
            Err(e) => {
                // Create a new error with path_str to help users see which files failed
                let err_msg = format!("{}: {}", path, e);
                return Err(io::Error::new(e.kind(), err_msg));
            }
        };

        has_macro_rewrite_failure |= visitor.macro_rewrite_failure;

        result.push((path.clone(), visitor.buffer));
    }

    Ok((result, has_diff, has_macro_rewrite_failure))
}

/// Returns true if the line with the given line number was skipped by `#[rustfmt::skip]`.
fn is_skipped_line(line_number: usize, skipped_range: &[(usize, usize)]) -> bool {
    skipped_range
        .iter()
        .any(|&(lo, hi)| lo <= line_number && line_number <= hi)
}

fn should_report_error(
    config: &Config,
    char_kind: FullCodeCharKind,
    is_string: bool,
    error_kind: &ErrorKind,
) -> bool {
    let allow_error_report = if char_kind.is_comment() || is_string || error_kind.is_comment() {
        config.error_on_unformatted()
    } else {
        true
    };

    match error_kind {
        ErrorKind::LineOverflow(..) => config.error_on_line_overflow() && allow_error_report,
        ErrorKind::TrailingWhitespace | ErrorKind::LostComment => allow_error_report,
        _ => true,
    }
}

// Formatting done on a char by char or line by line basis.
// FIXME(#20) other stuff for parity with make tidy
fn format_lines(
    text: &mut String,
    name: &FileName,
    skipped_range: &[(usize, usize)],
    config: &Config,
    report: &FormatReport,
) {
    let mut last_was_space = false;
    let mut line_len = 0;
    let mut cur_line = 1;
    let mut newline_count = 0;
    let mut errors = vec![];
    let mut issue_seeker = BadIssueSeeker::new(config.report_todo(), config.report_fixme());
    let mut line_buffer = String::with_capacity(config.max_width() * 2);
    let mut is_string = false; // true if the current line contains a string literal.
    let mut format_line = config.file_lines().contains_line(name, cur_line);
    let allow_issue_seek = !issue_seeker.is_disabled();

    // Check license.
    if let Some(ref license_template) = config.license_template {
        if !license_template.is_match(text) {
            errors.push(FormattingError {
                line: cur_line,
                kind: ErrorKind::LicenseCheck,
                is_comment: false,
                is_string: false,
                line_buffer: String::new(),
            });
        }
    }

    // Iterate over the chars in the file map.
    for (kind, c) in CharClasses::new(text.chars()) {
        if c == '\r' {
            continue;
        }

        if allow_issue_seek && format_line {
            // Add warnings for bad todos/ fixmes
            if let Some(issue) = issue_seeker.inspect(c) {
                errors.push(FormattingError {
                    line: cur_line,
                    kind: ErrorKind::BadIssue(issue),
                    is_comment: false,
                    is_string: false,
                    line_buffer: String::new(),
                });
            }
        }

        if c == '\n' {
            if format_line {
                // Check for (and record) trailing whitespace.
                if last_was_space {
                    if should_report_error(config, kind, is_string, &ErrorKind::TrailingWhitespace)
                        && !is_skipped_line(cur_line, skipped_range)
                    {
                        errors.push(FormattingError {
                            line: cur_line,
                            kind: ErrorKind::TrailingWhitespace,
                            is_comment: kind.is_comment(),
                            is_string: kind.is_string(),
                            line_buffer: line_buffer.clone(),
                        });
                    }
                    line_len -= 1;
                }

                // Check for any line width errors we couldn't correct.
                let error_kind = ErrorKind::LineOverflow(line_len, config.max_width());
                if line_len > config.max_width()
                    && !is_skipped_line(cur_line, skipped_range)
                    && should_report_error(config, kind, is_string, &error_kind)
                {
                    errors.push(FormattingError {
                        line: cur_line,
                        kind: error_kind,
                        is_comment: kind.is_comment(),
                        is_string,
                        line_buffer: line_buffer.clone(),
                    });
                }
            }

            line_len = 0;
            cur_line += 1;
            format_line = config.file_lines().contains_line(name, cur_line);
            newline_count += 1;
            last_was_space = false;
            line_buffer.clear();
            is_string = false;
        } else {
            newline_count = 0;
            line_len += if c == '\t' { config.tab_spaces() } else { 1 };
            last_was_space = c.is_whitespace();
            line_buffer.push(c);
            if kind.is_string() {
                is_string = true;
            }
        }
    }

    if newline_count > 1 {
        debug!("track truncate: {} {}", text.len(), newline_count);
        let line = text.len() - newline_count + 1;
        text.truncate(line);
    }

    report.append(name.clone(), errors);
}

fn parse_input<'sess>(
    input: Input,
    parse_session: &'sess ParseSess,
    config: &Config,
) -> Result<ast::Crate, ParseError<'sess>> {
    let mut parser = match input {
        Input::File(file) => parse::new_parser_from_file(parse_session, &file),
        Input::Text(text) => parse::new_parser_from_source_str(
            parse_session,
            syntax::codemap::FileName::Custom("stdin".to_owned()),
            text,
        ),
    };

    parser.cfg_mods = false;
    if config.skip_children() {
        parser.recurse_into_file_modules = false;
    }

    let mut parser = AssertUnwindSafe(parser);
    let result = catch_unwind(move || parser.0.parse_crate_mod());

    match result {
        Ok(Ok(c)) => {
            if parse_session.span_diagnostic.has_errors() {
                // Bail out if the parser recovered from an error.
                Err(ParseError::Recovered)
            } else {
                Ok(c)
            }
        }
        Ok(Err(e)) => Err(ParseError::Error(e)),
        Err(_) => Err(ParseError::Panic),
    }
}

/// All the ways that parsing can fail.
enum ParseError<'sess> {
    /// There was an error, but the parser recovered.
    Recovered,
    /// There was an error (supplied) and parsing failed.
    Error(DiagnosticBuilder<'sess>),
    /// The parser panicked.
    Panic,
}

pub(crate) fn format_input_inner<T: Write>(
    input: Input,
    config: &Config,
    mut out: Option<&mut T>,
) -> Result<(Summary, FileMap, FormatReport), (ErrorKind, Summary)> {
    syntax_pos::hygiene::set_default_edition(config.edition().to_libsyntax_pos_edition());
    let mut summary = Summary::default();
    if config.disable_all_formatting() {
        // When the input is from stdin, echo back the input.
        if let Input::Text(ref buf) = input {
            if let Err(e) = io::stdout().write_all(buf.as_bytes()) {
                return Err((From::from(e), summary));
            }
        }
        return Ok((summary, FileMap::new(), FormatReport::new()));
    }
    let codemap = Rc::new(CodeMap::new(FilePathMapping::empty()));

    let tty_handler = if config.hide_parse_errors() {
        let silent_emitter = Box::new(EmitterWriter::new(
            Box::new(Vec::new()),
            Some(codemap.clone()),
            false,
            false,
        ));
        Handler::with_emitter(true, false, silent_emitter)
    } else {
        let supports_color = term::stderr().map_or(false, |term| term.supports_color());
        let color_cfg = if supports_color {
            ColorConfig::Auto
        } else {
            ColorConfig::Never
        };
        Handler::with_tty_emitter(color_cfg, true, false, Some(codemap.clone()))
    };
    let mut parse_session = ParseSess::with_span_handler(tty_handler, codemap.clone());

    let main_file = match input {
        Input::File(ref file) => FileName::Real(file.clone()),
        Input::Text(..) => FileName::Stdin,
    };

    let krate = match parse_input(input, &parse_session, config) {
        Ok(krate) => krate,
        Err(err) => {
            match err {
                ParseError::Error(mut diagnostic) => diagnostic.emit(),
                ParseError::Panic => {
                    // Note that if you see this message and want more information,
                    // then go to `parse_input` and run the parse function without
                    // `catch_unwind` so rustfmt panics and you can get a backtrace.
                    should_emit_verbose(&main_file, config, || {
                        println!("The Rust parser panicked")
                    });
                }
                ParseError::Recovered => {}
            }
            summary.add_parsing_error();
            return Err((ErrorKind::ParseError, summary));
        }
    };

    summary.mark_parse_time();

    // Suppress error output after parsing.
    let silent_emitter = Box::new(EmitterWriter::new(
        Box::new(Vec::new()),
        Some(codemap.clone()),
        false,
        false,
    ));
    parse_session.span_diagnostic = Handler::with_emitter(true, false, silent_emitter);

    let report = FormatReport::new();

    let format_result = format_ast(
        &krate,
        &mut parse_session,
        &main_file,
        config,
        report.clone(),
        |file_name, file, skipped_range, report| {
            // For some reason, the codemap does not include terminating
            // newlines so we must add one on for each file. This is sad.
            filemap::append_newline(file);

            format_lines(file, file_name, skipped_range, config, report);
            replace_with_system_newlines(file, config);

            if let Some(ref mut out) = out {
                return filemap::write_file(file, file_name, out, config);
            }
            Ok(false)
        },
    );

    summary.mark_format_time();

    should_emit_verbose(&main_file, config, || {
        fn duration_to_f32(d: Duration) -> f32 {
            d.as_secs() as f32 + d.subsec_nanos() as f32 / 1_000_000_000f32
        }

        println!(
            "Spent {0:.3} secs in the parsing phase, and {1:.3} secs in the formatting phase",
            duration_to_f32(summary.get_parse_time().unwrap()),
            duration_to_f32(summary.get_format_time().unwrap()),
        )
    });

    {
        let report_errs = &report.internal.borrow().1;
        if report_errs.has_check_errors {
            summary.add_check_error();
        }
        if report_errs.has_operational_errors {
            summary.add_operational_error();
        }
    }

    match format_result {
        Ok((file_map, has_diff, has_macro_rewrite_failure)) => {
            if report.has_warnings() {
                summary.add_formatting_error();
            }

            if has_diff {
                summary.add_diff();
            }

            if has_macro_rewrite_failure {
                summary.add_macro_foramt_failure();
            }

            Ok((summary, file_map, report))
        }
        Err(e) => Err((From::from(e), summary)),
    }
}

fn replace_with_system_newlines(text: &mut String, config: &Config) -> () {
    let style = if config.newline_style() == NewlineStyle::Native {
        if cfg!(windows) {
            NewlineStyle::Windows
        } else {
            NewlineStyle::Unix
        }
    } else {
        config.newline_style()
    };

    match style {
        NewlineStyle::Unix => return,
        NewlineStyle::Windows => {
            let mut transformed = String::with_capacity(text.capacity());
            for c in text.chars() {
                match c {
                    '\n' => transformed.push_str("\r\n"),
                    '\r' => continue,
                    c => transformed.push(c),
                }
            }
            *text = transformed;
        }
        NewlineStyle::Native => unreachable!(),
    }
}

/// A single span of changed lines, with 0 or more removed lines
/// and a vector of 0 or more inserted lines.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ModifiedChunk {
    /// The first to be removed from the original text
    pub line_number_orig: u32,
    /// The number of lines which have been replaced
    pub lines_removed: u32,
    /// The new lines
    pub lines: Vec<String>,
}

/// Set of changed sections of a file.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ModifiedLines {
    /// The set of changed chunks.
    pub chunks: Vec<ModifiedChunk>,
}

/// Format a file and return a `ModifiedLines` data structure describing
/// the changed ranges of lines.
#[cfg(test)]
pub(crate) fn get_modified_lines(
    input: Input,
    config: &Config,
) -> Result<ModifiedLines, (ErrorKind, Summary)> {
    use std::io::BufRead;

    let mut data = Vec::new();

    let mut config = config.clone();
    config.set().emit_mode(::config::EmitMode::ModifiedLines);
    ::format_input(input, &config, Some(&mut data))?;

    let mut lines = data.lines();
    let mut chunks = Vec::new();
    while let Some(Ok(header)) = lines.next() {
        // Parse the header line
        let values: Vec<_> = header
            .split(' ')
            .map(|s| s.parse::<u32>().unwrap())
            .collect();
        assert_eq!(values.len(), 3);
        let line_number_orig = values[0];
        let lines_removed = values[1];
        let num_added = values[2];
        let mut added_lines = Vec::new();
        for _ in 0..num_added {
            added_lines.push(lines.next().unwrap().unwrap());
        }
        chunks.push(ModifiedChunk {
            line_number_orig,
            lines_removed,
            lines: added_lines,
        });
    }
    Ok(ModifiedLines { chunks })
}
