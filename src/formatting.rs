// High level formatting functions.

use std::collections::HashMap;
use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::time::{Duration, Instant};

use syntax::ast;
use syntax::codemap::{CodeMap, FilePathMapping, Span};
use syntax::errors::emitter::{ColorConfig, EmitterWriter};
use syntax::errors::{DiagnosticBuilder, Handler};
use syntax::parse::{self, ParseSess};

use comment::{CharClasses, FullCodeCharKind};
use config::{Config, FileName, NewlineStyle, Verbosity};
use issues::BadIssueSeeker;
use visitor::{FmtVisitor, SnippetProvider};
use {filemap, modules, ErrorKind, FormatReport, Input, Session};

// A map of the files of a crate, with their new content
pub(crate) type FileMap = Vec<FileRecord>;
pub(crate) type FileRecord = (FileName, String);

impl<'b, T: Write + 'b> Session<'b, T> {
    pub(crate) fn format_input_inner(&mut self, input: Input) -> Result<FormatReport, ErrorKind> {
        if !self.config.version_meets_requirement() {
            return Err(ErrorKind::VersionMismatch);
        }

        syntax::with_globals(|| {
            syntax_pos::hygiene::set_default_edition(
                self.config.edition().to_libsyntax_pos_edition(),
            );

            if self.config.disable_all_formatting() {
                // When the input is from stdin, echo back the input.
                if let Input::Text(ref buf) = input {
                    if let Err(e) = io::stdout().write_all(buf.as_bytes()) {
                        return Err(From::from(e));
                    }
                }
                return Ok(FormatReport::new());
            }

            let config = &self.config.clone();
            let format_result = format_project(input, config, self);

            format_result.map(|(report, summary)| {
                self.summary.add(summary);
                report
            })
        })
    }
}

// Format an entire crate (or subset of the module tree).
fn format_project<T: FormatHandler>(
    input: Input,
    config: &Config,
    handler: &mut T,
) -> Result<(FormatReport, Summary), ErrorKind> {
    let mut summary = Summary::default();
    let mut timer = Timer::Initialized(Instant::now());

    let input_is_stdin = input.is_text();
    let main_file = match input {
        Input::File(ref file) => FileName::Real(file.clone()),
        Input::Text(..) => FileName::Stdin,
    };

    // Parse the crate.
    let codemap = Rc::new(CodeMap::new(FilePathMapping::empty()));
    let mut parse_session = make_parse_sess(codemap.clone(), config);
    let krate = match parse_input(input, &parse_session, config) {
        Ok(krate) => krate,
        Err(err) => {
            match err {
                ParseError::Error(mut diagnostic) => diagnostic.emit(),
                ParseError::Panic => {
                    // Note that if you see this message and want more information,
                    // then go to `parse_input` and run the parse function without
                    // `catch_unwind` so rustfmt panics and you can get a backtrace.
                    should_emit_verbose(!input_is_stdin, config, || {
                        println!("The Rust parser panicked")
                    });
                }
                ParseError::Recovered => {}
            }
            summary.add_parsing_error();
            return Err(ErrorKind::ParseError);
        }
    };
    timer = timer.done_parsing();

    // Suppress error output if we have to do any further parsing.
    let silent_emitter = Box::new(EmitterWriter::new(
        Box::new(Vec::new()),
        Some(codemap.clone()),
        false,
        false,
    ));
    parse_session.span_diagnostic = Handler::with_emitter(true, false, silent_emitter);

    let mut context = FormatContext::new(
        &krate,
        FormatReport::new(),
        summary,
        parse_session,
        config,
        handler,
    );

    let files = modules::list_files(&krate, context.parse_session.codemap())?;
    for (path, module) in files {
        if (config.skip_children() && path != main_file) || config.ignore().skip_file(&path) {
            continue;
        }
        should_emit_verbose(!input_is_stdin, config, || println!("Formatting {}", path));
        let is_root = path == main_file;
        context.format_file(path, module, is_root)?;
    }
    timer = timer.done_formatting();

    should_emit_verbose(!input_is_stdin, config, || {
        println!(
            "Spent {0:.3} secs in the parsing phase, and {1:.3} secs in the formatting phase",
            timer.get_parse_time(),
            timer.get_format_time(),
        )
    });

    if context.report.has_warnings() {
        context.summary.add_formatting_error();
    }
    {
        let report_errs = &context.report.internal.borrow().1;
        if report_errs.has_check_errors {
            context.summary.add_check_error();
        }
        if report_errs.has_operational_errors {
            context.summary.add_operational_error();
        }
    }

    Ok((context.report, context.summary))
}

// Used for formatting files.
#[derive(new)]
struct FormatContext<'a, T: FormatHandler + 'a> {
    krate: &'a ast::Crate,
    report: FormatReport,
    summary: Summary,
    parse_session: ParseSess,
    config: &'a Config,
    handler: &'a mut T,
}

impl<'a, T: FormatHandler + 'a> FormatContext<'a, T> {
    // Formats a single file/module.
    fn format_file(
        &mut self,
        path: FileName,
        module: &ast::Mod,
        is_root: bool,
    ) -> Result<(), ErrorKind> {
        let filemap = self
            .parse_session
            .codemap()
            .lookup_char_pos(module.inner.lo())
            .file;
        let big_snippet = filemap.src.as_ref().unwrap();
        let snippet_provider = SnippetProvider::new(filemap.start_pos, big_snippet);
        let mut visitor = FmtVisitor::from_codemap(
            &self.parse_session,
            &self.config,
            &snippet_provider,
            self.report.clone(),
        );
        // Format inner attributes if available.
        if !self.krate.attrs.is_empty() && is_root {
            visitor.skip_empty_lines(filemap.end_pos);
            if visitor.visit_attrs(&self.krate.attrs, ast::AttrStyle::Inner) {
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

        // For some reason, the codemap does not include terminating
        // newlines so we must add one on for each file. This is sad.
        filemap::append_newline(&mut visitor.buffer);

        format_lines(
            &mut visitor.buffer,
            &path,
            &visitor.skipped_range,
            &self.config,
            &self.report,
        );
        replace_with_system_newlines(&mut visitor.buffer, &self.config);

        if visitor.macro_rewrite_failure {
            self.summary.add_macro_format_failure();
        }

        self.handler.handle_formatted_file(path, visitor.buffer)
    }
}

// Handle the results of formatting.
trait FormatHandler {
    fn handle_formatted_file(&mut self, path: FileName, result: String) -> Result<(), ErrorKind>;
}

impl<'b, T: Write + 'b> FormatHandler for Session<'b, T> {
    // Called for each formatted file.
    fn handle_formatted_file(
        &mut self,
        path: FileName,
        mut result: String,
    ) -> Result<(), ErrorKind> {
        if let Some(ref mut out) = self.out {
            match filemap::write_file(&mut result, &path, out, &self.config) {
                Ok(b) if b => self.summary.add_diff(),
                Err(e) => {
                    // Create a new error with path_str to help users see which files failed
                    let err_msg = format!("{}: {}", path, e);
                    return Err(io::Error::new(e.kind(), err_msg).into());
                }
                _ => {}
            }
        }

        self.filemap.push((path, result));
        Ok(())
    }
}

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

/// A summary of a Rustfmt run.
#[derive(Debug, Default, Clone, Copy)]
pub struct Summary {
    // Encountered e.g. an IO error.
    has_operational_errors: bool,

    // Failed to reformat code because of parsing errors.
    has_parsing_errors: bool,

    // Code is valid, but it is impossible to format it properly.
    has_formatting_errors: bool,

    // Code contains macro call that was unable to format.
    pub(crate) has_macro_format_failure: bool,

    // Failed a check, such as the license check or other opt-in checking.
    has_check_errors: bool,

    /// Formatted code differs from existing code (--check only).
    pub has_diff: bool,
}

impl Summary {
    pub fn has_operational_errors(&self) -> bool {
        self.has_operational_errors
    }

    pub fn has_parsing_errors(&self) -> bool {
        self.has_parsing_errors
    }

    pub fn has_formatting_errors(&self) -> bool {
        self.has_formatting_errors
    }

    pub fn has_check_errors(&self) -> bool {
        self.has_check_errors
    }

    pub(crate) fn has_macro_formatting_failure(&self) -> bool {
        self.has_macro_format_failure
    }

    pub fn add_operational_error(&mut self) {
        self.has_operational_errors = true;
    }

    pub(crate) fn add_parsing_error(&mut self) {
        self.has_parsing_errors = true;
    }

    pub(crate) fn add_formatting_error(&mut self) {
        self.has_formatting_errors = true;
    }

    pub(crate) fn add_check_error(&mut self) {
        self.has_check_errors = true;
    }

    pub(crate) fn add_diff(&mut self) {
        self.has_diff = true;
    }

    pub(crate) fn add_macro_format_failure(&mut self) {
        self.has_macro_format_failure = true;
    }

    pub fn has_no_errors(&self) -> bool {
        !(self.has_operational_errors
            || self.has_parsing_errors
            || self.has_formatting_errors
            || self.has_diff)
    }

    /// Combine two summaries together.
    pub fn add(&mut self, other: Summary) {
        self.has_operational_errors |= other.has_operational_errors;
        self.has_formatting_errors |= other.has_formatting_errors;
        self.has_macro_format_failure |= other.has_macro_format_failure;
        self.has_parsing_errors |= other.has_parsing_errors;
        self.has_check_errors |= other.has_check_errors;
        self.has_diff |= other.has_diff;
    }
}

#[derive(Clone, Copy, Debug)]
enum Timer {
    Initialized(Instant),
    DoneParsing(Instant, Instant),
    DoneFormatting(Instant, Instant, Instant),
}

impl Timer {
    fn done_parsing(self) -> Self {
        match self {
            Timer::Initialized(init_time) => Timer::DoneParsing(init_time, Instant::now()),
            _ => panic!("Timer can only transition to DoneParsing from Initialized state"),
        }
    }

    fn done_formatting(self) -> Self {
        match self {
            Timer::DoneParsing(init_time, parse_time) => {
                Timer::DoneFormatting(init_time, parse_time, Instant::now())
            }
            _ => panic!("Timer can only transition to DoneFormatting from DoneParsing state"),
        }
    }

    /// Returns the time it took to parse the source files in seconds.
    fn get_parse_time(&self) -> f32 {
        match *self {
            Timer::DoneParsing(init, parse_time) | Timer::DoneFormatting(init, parse_time, _) => {
                // This should never underflow since `Instant::now()` guarantees monotonicity.
                Self::duration_to_f32(parse_time.duration_since(init))
            }
            Timer::Initialized(..) => unreachable!(),
        }
    }

    /// Returns the time it took to go from the parsed AST to the formatted output. Parsing time is
    /// not included.
    fn get_format_time(&self) -> f32 {
        match *self {
            Timer::DoneFormatting(_init, parse_time, format_time) => {
                Self::duration_to_f32(format_time.duration_since(parse_time))
            }
            Timer::DoneParsing(..) | Timer::Initialized(..) => unreachable!(),
        }
    }

    fn duration_to_f32(d: Duration) -> f32 {
        d.as_secs() as f32 + d.subsec_nanos() as f32 / 1_000_000_000f32
    }
}

fn should_emit_verbose<F>(is_stdin: bool, config: &Config, f: F)
where
    F: Fn(),
{
    if config.verbose() == Verbosity::Verbose && !is_stdin {
        f();
    }
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

fn make_parse_sess(codemap: Rc<CodeMap>, config: &Config) -> ParseSess {
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

    ParseSess::with_span_handler(tty_handler, codemap)
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
