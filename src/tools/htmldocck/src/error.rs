use std::ops::Range;

use unicode_width::UnicodeWidthStr;

pub(crate) struct DiagCtxt {
    count: usize,
}

impl DiagCtxt {
    pub(crate) fn scope(run: impl FnOnce(&mut Self) -> Result<(), ()>) -> Result<(), ()> {
        let mut dcx = Self::new();
        let result = run(&mut dcx);
        dcx.summarize();
        match result {
            Ok(()) if dcx.is_empty() => Ok(()),
            _ => Err(()),
        }
    }

    fn new() -> Self {
        Self { count: 0 }
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    // FIXME: Support for multiple subdiagnostics.
    pub(crate) fn emit<'a>(
        &mut self,
        message: &str,
        source: impl Into<Option<Source<'a>>>,
        help: impl Into<Option<&'a str>>,
    ) {
        self.count += 1;
        self.print(message, source.into(), help.into());
    }

    fn print(&mut self, message: &str, source: Option<Source<'_>>, help: Option<&str>) {
        // FIXME: use proper coloring library
        eprintln!("\x1b[31merror\x1b[0m: {message}");

        let Some(source) = source else { return };

        eprintln!("\x1b[1;36m{} | \x1b[0m{}", source.lineno, source.line);

        let underline_offset = source.line[..source.range.start].width();
        let underline_length = source.line[source.range].width();
        eprintln!(
            "\x1b[1;36m{}   \x1b[0m\x1b[31m{}{}{}\x1b[0m",
            " ".repeat(source.lineno.ilog10() as usize + 1),
            " ".repeat(underline_offset),
            "^".repeat(underline_length),
            // FIXME: get rid of format here
            help.map(|help| format!(" help: {help}")).unwrap_or_default(),
        );
    }

    fn summarize(&self) {
        if self.is_empty() {
            return;
        }

        eprintln!();
        eprintln!("encountered {} error{}", self.count, if self.count == 1 { "" } else { "s" });
    }
}

#[derive(Clone)] // FIXME: derive `Copy` once we can use `new_range`.
pub(crate) struct Source<'src> {
    pub(crate) line: &'src str,
    /// The one-based line number.
    pub(crate) lineno: usize,
    pub(crate) range: Range<usize>,
}
