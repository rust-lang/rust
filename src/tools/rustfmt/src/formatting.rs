// High level formatting functions.

use std::collections::HashMap;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use rustc_ast::ast;
use rustc_span::Span;
use tracing::debug;

use self::newline_style::apply_newline_style;
use crate::comment::{CharClasses, FullCodeCharKind};
use crate::config::{Config, FileName, Verbosity};
use crate::formatting::generated::is_generated_file;
use crate::modules::Module;
use crate::parse::parser::{DirectoryOwnership, Parser, ParserError};
use crate::parse::session::ParseSess;
use crate::utils::{contains_skip, count_newlines};
use crate::visitor::FmtVisitor;
use crate::{ErrorKind, FormatReport, Input, Session, modules, source_file};

mod generated;
mod newline_style;

// A map of the files of a crate, with their new content
pub(crate) type SourceFile = Vec<FileRecord>;
pub(crate) type FileRecord = (FileName, String);

impl<'b, T: Write + 'b> Session<'b, T> {
    pub(crate) fn format_input_inner(
        &mut self,
        input: Input,
        is_macro_def: bool,
    ) -> Result<FormatReport, ErrorKind> {
        if !self.config.version_meets_requirement() {
            return Err(ErrorKind::VersionMismatch);
        }

        rustc_span::create_session_if_not_set_then(self.config.edition().into(), |_| {
            if self.config.disable_all_formatting() {
                // When the input is from stdin, echo back the input.
                return match input {
                    Input::Text(ref buf) => echo_back_stdin(buf),
                    _ => Ok(FormatReport::new()),
                };
            }

            let config = &self.config.clone();
            let format_result = format_project(input, config, self, is_macro_def);

            format_result.map(|report| {
                self.errors.add(&report.internal.borrow().1);
                report
            })
        })
    }
}

/// Determine if a module should be skipped. True if the module should be skipped, false otherwise.
fn should_skip_module<T: FormatHandler>(
    config: &Config,
    context: &FormatContext<'_, T>,
    input_is_stdin: bool,
    main_file: &FileName,
    path: &FileName,
    module: &Module<'_>,
) -> bool {
    if contains_skip(module.attrs()) {
        return true;
    }

    if config.skip_children() && path != main_file {
        return true;
    }

    if !input_is_stdin && context.ignore_file(path) {
        return true;
    }

    // FIXME(calebcartwright) - we need to determine how we'll handle the
    // `format_generated_files` option with stdin based input.
    if !input_is_stdin && !config.format_generated_files() {
        let source_file = context.psess.span_to_file_contents(module.span);
        let src = source_file.src.as_ref().expect("SourceFile without src");

        if is_generated_file(src, config) {
            return true;
        }
    }

    false
}

fn echo_back_stdin(input: &str) -> Result<FormatReport, ErrorKind> {
    if let Err(e) = io::stdout().write_all(input.as_bytes()) {
        return Err(From::from(e));
    }
    Ok(FormatReport::new())
}

// Format an entire crate (or subset of the module tree).
fn format_project<T: FormatHandler>(
    input: Input,
    config: &Config,
    handler: &mut T,
    is_macro_def: bool,
) -> Result<FormatReport, ErrorKind> {
    let mut timer = Timer::start();

    let main_file = input.file_name();
    let input_is_stdin = main_file == FileName::Stdin;

    let psess = ParseSess::new(config)?;
    if config.skip_children() && psess.ignore_file(&main_file) {
        return Ok(FormatReport::new());
    }

    // Parse the crate.
    let mut report = FormatReport::new();
    let directory_ownership = input.to_directory_ownership();

    let krate = match Parser::parse_crate(input, &psess) {
        Ok(krate) => krate,
        // Surface parse error via Session (errors are merged there from report)
        Err(e) => {
            let forbid_verbose = input_is_stdin || e != ParserError::ParsePanicError;
            should_emit_verbose(forbid_verbose, config, || {
                eprintln!("The Rust parser panicked");
            });
            report.add_parsing_error();
            return Ok(report);
        }
    };

    let mut context = FormatContext::new(&krate, report, psess, config, handler);
    let files = modules::ModResolver::new(
        &context.psess,
        directory_ownership.unwrap_or(DirectoryOwnership::UnownedViaBlock),
        !input_is_stdin && !config.skip_children(),
    )
    .visit_crate(&krate)?
    .into_iter()
    .filter(|(path, module)| {
        input_is_stdin
            || !should_skip_module(config, &context, input_is_stdin, &main_file, path, module)
    })
    .collect::<Vec<_>>();

    timer = timer.done_parsing();

    // Suppress error output if we have to do any further parsing.
    context.psess.set_silent_emitter();

    for (path, module) in files {
        if input_is_stdin && contains_skip(module.attrs()) {
            return echo_back_stdin(context.psess.snippet_provider(module.span).entire_snippet());
        }
        should_emit_verbose(input_is_stdin, config, || println!("Formatting {}", path));
        context.format_file(path, &module, is_macro_def)?;
    }
    timer = timer.done_formatting();

    should_emit_verbose(input_is_stdin, config, || {
        println!(
            "Spent {0:.3} secs in the parsing phase, and {1:.3} secs in the formatting phase",
            timer.get_parse_time(),
            timer.get_format_time(),
        )
    });

    Ok(context.report)
}

// Used for formatting files.
struct FormatContext<'a, T: FormatHandler> {
    krate: &'a ast::Crate,
    report: FormatReport,
    psess: ParseSess,
    config: &'a Config,
    handler: &'a mut T,
}

impl<'a, T: FormatHandler + 'a> FormatContext<'a, T> {
    fn new(
        krate: &'a ast::Crate,
        report: FormatReport,
        psess: ParseSess,
        config: &'a Config,
        handler: &'a mut T,
    ) -> Self {
        FormatContext {
            krate,
            report,
            psess,
            config,
            handler,
        }
    }

    fn ignore_file(&self, path: &FileName) -> bool {
        self.psess.ignore_file(path)
    }

    // Formats a single file/module.
    fn format_file(
        &mut self,
        path: FileName,
        module: &Module<'_>,
        is_macro_def: bool,
    ) -> Result<(), ErrorKind> {
        let snippet_provider = self.psess.snippet_provider(module.span);
        let mut visitor = FmtVisitor::from_psess(
            &self.psess,
            self.config,
            &snippet_provider,
            self.report.clone(),
        );
        visitor.skip_context.update_with_attrs(&self.krate.attrs);
        visitor.is_macro_def = is_macro_def;
        visitor.last_pos = snippet_provider.start_pos();
        visitor.skip_empty_lines(snippet_provider.end_pos());
        visitor.format_separate_mod(module, snippet_provider.end_pos());

        debug_assert_eq!(
            visitor.line_number,
            count_newlines(&visitor.buffer),
            "failed in format_file visitor.buffer:\n {:?}",
            &visitor.buffer
        );

        // For some reason, the source_map does not include terminating
        // newlines so we must add one on for each file. This is sad.
        source_file::append_newline(&mut visitor.buffer);

        format_lines(
            &mut visitor.buffer,
            &path,
            &visitor.skipped_range.borrow(),
            self.config,
            &self.report,
        );

        apply_newline_style(
            self.config.newline_style(),
            &mut visitor.buffer,
            snippet_provider.entire_snippet(),
        );

        if visitor.macro_rewrite_failure {
            self.report.add_macro_format_failure();
        }
        self.report
            .add_non_formatted_ranges(visitor.skipped_range.borrow().clone());

        self.handler.handle_formatted_file(
            &self.psess,
            path,
            visitor.buffer.to_owned(),
            &mut self.report,
        )
    }
}

// Handle the results of formatting.
trait FormatHandler {
    fn handle_formatted_file(
        &mut self,
        psess: &ParseSess,
        path: FileName,
        result: String,
        report: &mut FormatReport,
    ) -> Result<(), ErrorKind>;
}

impl<'b, T: Write + 'b> FormatHandler for Session<'b, T> {
    // Called for each formatted file.
    fn handle_formatted_file(
        &mut self,
        psess: &ParseSess,
        path: FileName,
        result: String,
        report: &mut FormatReport,
    ) -> Result<(), ErrorKind> {
        if let Some(ref mut out) = self.out {
            match source_file::write_file(
                Some(psess),
                &path,
                &result,
                out,
                &mut *self.emitter,
                self.config.newline_style(),
            ) {
                Ok(ref result) if result.has_diff => report.add_diff(),
                Err(e) => {
                    // Create a new error with path_str to help users see which files failed
                    let err_msg = format!("{path}: {e}");
                    return Err(io::Error::new(e.kind(), err_msg).into());
                }
                _ => {}
            }
        }

        self.source_file.push((path, result));
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
    pub(crate) fn from_span(span: Span, psess: &ParseSess, kind: ErrorKind) -> FormattingError {
        FormattingError {
            line: psess.line_of_byte_pos(span.lo()),
            is_comment: kind.is_comment(),
            kind,
            is_string: false,
            line_buffer: psess.span_to_first_line_string(span),
        }
    }

    pub(crate) fn is_internal(&self) -> bool {
        match self.kind {
            ErrorKind::LineOverflow(..)
            | ErrorKind::TrailingWhitespace
            | ErrorKind::IoError(_)
            | ErrorKind::ParseError
            | ErrorKind::LostComment => true,
            _ => false,
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

#[derive(Default, Debug, PartialEq)]
pub(crate) struct ReportedErrors {
    // Encountered e.g., an IO error.
    pub(crate) has_operational_errors: bool,

    // Failed to reformat code because of parsing errors.
    pub(crate) has_parsing_errors: bool,

    // Code is valid, but it is impossible to format it properly.
    pub(crate) has_formatting_errors: bool,

    // Code contains macro call that was unable to format.
    pub(crate) has_macro_format_failure: bool,

    // Failed an opt-in checking.
    pub(crate) has_check_errors: bool,

    /// Formatted code differs from existing code (--check only).
    pub(crate) has_diff: bool,

    /// Formatted code missed something, like lost comments or extra trailing space
    pub(crate) has_unformatted_code_errors: bool,
}

impl ReportedErrors {
    /// Combine two summaries together.
    pub(crate) fn add(&mut self, other: &ReportedErrors) {
        self.has_operational_errors |= other.has_operational_errors;
        self.has_parsing_errors |= other.has_parsing_errors;
        self.has_formatting_errors |= other.has_formatting_errors;
        self.has_macro_format_failure |= other.has_macro_format_failure;
        self.has_check_errors |= other.has_check_errors;
        self.has_diff |= other.has_diff;
        self.has_unformatted_code_errors |= other.has_unformatted_code_errors;
    }
}

#[derive(Clone, Copy, Debug)]
enum Timer {
    Disabled,
    Initialized(Instant),
    DoneParsing(Instant, Instant),
    DoneFormatting(Instant, Instant, Instant),
}

impl Timer {
    fn start() -> Timer {
        if cfg!(target_arch = "wasm32") {
            Timer::Disabled
        } else {
            Timer::Initialized(Instant::now())
        }
    }
    fn done_parsing(self) -> Self {
        match self {
            Timer::Disabled => Timer::Disabled,
            Timer::Initialized(init_time) => Timer::DoneParsing(init_time, Instant::now()),
            _ => panic!("Timer can only transition to DoneParsing from Initialized state"),
        }
    }

    fn done_formatting(self) -> Self {
        match self {
            Timer::Disabled => Timer::Disabled,
            Timer::DoneParsing(init_time, parse_time) => {
                Timer::DoneFormatting(init_time, parse_time, Instant::now())
            }
            _ => panic!("Timer can only transition to DoneFormatting from DoneParsing state"),
        }
    }

    /// Returns the time it took to parse the source files in seconds.
    fn get_parse_time(&self) -> f32 {
        match *self {
            Timer::Disabled => panic!("this platform cannot time execution"),
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
            Timer::Disabled => panic!("this platform cannot time execution"),
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

// Formatting done on a char by char or line by line basis.
// FIXME(#20): other stuff for parity with make tidy.
fn format_lines(
    text: &mut String,
    name: &FileName,
    skipped_range: &[(usize, usize)],
    config: &Config,
    report: &FormatReport,
) {
    let mut formatter = FormatLines::new(name, skipped_range, config);
    formatter.iterate(text);

    if formatter.newline_count > 1 {
        debug!("track truncate: {} {}", text.len(), formatter.newline_count);
        let line = text.len() - formatter.newline_count + 1;
        text.truncate(line);
    }

    report.append(name.clone(), formatter.errors);
}

struct FormatLines<'a> {
    name: &'a FileName,
    skipped_range: &'a [(usize, usize)],
    last_was_space: bool,
    line_len: usize,
    cur_line: usize,
    newline_count: usize,
    errors: Vec<FormattingError>,
    line_buffer: String,
    current_line_contains_string_literal: bool,
    format_line: bool,
    config: &'a Config,
}

impl<'a> FormatLines<'a> {
    fn new(
        name: &'a FileName,
        skipped_range: &'a [(usize, usize)],
        config: &'a Config,
    ) -> FormatLines<'a> {
        FormatLines {
            name,
            skipped_range,
            last_was_space: false,
            line_len: 0,
            cur_line: 1,
            newline_count: 0,
            errors: vec![],
            line_buffer: String::with_capacity(config.max_width() * 2),
            current_line_contains_string_literal: false,
            format_line: config.file_lines().contains_line(name, 1),
            config,
        }
    }

    // Iterate over the chars in the file map.
    fn iterate(&mut self, text: &mut String) {
        for (kind, c) in CharClasses::new(text.chars()) {
            if c == '\r' {
                continue;
            }

            if c == '\n' {
                self.new_line(kind);
            } else {
                self.char(c, kind);
            }
        }
    }

    fn new_line(&mut self, kind: FullCodeCharKind) {
        if self.format_line {
            // Check for (and record) trailing whitespace.
            if self.last_was_space {
                if self.should_report_error(kind, &ErrorKind::TrailingWhitespace)
                    && !self.is_skipped_line()
                {
                    self.push_err(
                        ErrorKind::TrailingWhitespace,
                        kind.is_comment(),
                        kind.is_string(),
                    );
                }
                self.line_len -= 1;
            }

            // Check for any line width errors we couldn't correct.
            let error_kind = ErrorKind::LineOverflow(self.line_len, self.config.max_width());
            if self.line_len > self.config.max_width()
                && !self.is_skipped_line()
                && self.should_report_error(kind, &error_kind)
            {
                let is_string = self.current_line_contains_string_literal;
                self.push_err(error_kind, kind.is_comment(), is_string);
            }
        }

        self.line_len = 0;
        self.cur_line += 1;
        self.format_line = self
            .config
            .file_lines()
            .contains_line(self.name, self.cur_line);
        self.newline_count += 1;
        self.last_was_space = false;
        self.line_buffer.clear();
        self.current_line_contains_string_literal = false;
    }

    fn char(&mut self, c: char, kind: FullCodeCharKind) {
        self.newline_count = 0;
        self.line_len += if c == '\t' {
            self.config.tab_spaces()
        } else {
            1
        };
        self.last_was_space = c.is_whitespace();
        self.line_buffer.push(c);
        if kind.is_string() {
            self.current_line_contains_string_literal = true;
        }
    }

    fn push_err(&mut self, kind: ErrorKind, is_comment: bool, is_string: bool) {
        self.errors.push(FormattingError {
            line: self.cur_line,
            kind,
            is_comment,
            is_string,
            line_buffer: self.line_buffer.clone(),
        });
    }

    fn should_report_error(&self, char_kind: FullCodeCharKind, error_kind: &ErrorKind) -> bool {
        let allow_error_report = if char_kind.is_comment()
            || self.current_line_contains_string_literal
            || error_kind.is_comment()
        {
            self.config.error_on_unformatted()
        } else {
            true
        };

        match error_kind {
            ErrorKind::LineOverflow(..) => {
                self.config.error_on_line_overflow() && allow_error_report
            }
            ErrorKind::TrailingWhitespace | ErrorKind::LostComment => allow_error_report,
            _ => true,
        }
    }

    /// Returns `true` if the line with the given line number was skipped by `#[rustfmt::skip]`.
    fn is_skipped_line(&self) -> bool {
        self.skipped_range
            .iter()
            .any(|&(lo, hi)| lo <= self.cur_line && self.cur_line <= hi)
    }
}

fn should_emit_verbose<F>(forbid_verbose_output: bool, config: &Config, f: F)
where
    F: Fn(),
{
    if config.verbose() == Verbosity::Verbose && !forbid_verbose_output {
        f();
    }
}
