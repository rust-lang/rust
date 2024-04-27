// Code for creating styled buffers

use crate::snippet::{Style, StyledString};

#[derive(Debug)]
pub struct StyledBuffer {
    lines: Vec<Vec<StyledChar>>,
}

#[derive(Debug, Clone)]
struct StyledChar {
    chr: char,
    style: Style,
}

impl StyledChar {
    const SPACE: Self = StyledChar::new(' ', Style::NoStyle);

    const fn new(chr: char, style: Style) -> Self {
        StyledChar { chr, style }
    }
}

impl StyledBuffer {
    pub fn new() -> StyledBuffer {
        StyledBuffer { lines: vec![] }
    }

    /// Returns content of `StyledBuffer` split by lines and line styles
    pub fn render(&self) -> Vec<Vec<StyledString>> {
        // Tabs are assumed to have been replaced by spaces in calling code.
        debug_assert!(self.lines.iter().all(|r| !r.iter().any(|sc| sc.chr == '\t')));

        let mut output: Vec<Vec<StyledString>> = vec![];
        let mut styled_vec: Vec<StyledString> = vec![];

        for styled_line in &self.lines {
            let mut current_style = Style::NoStyle;
            let mut current_text = String::new();

            for sc in styled_line {
                if sc.style != current_style {
                    if !current_text.is_empty() {
                        styled_vec.push(StyledString { text: current_text, style: current_style });
                    }
                    current_style = sc.style;
                    current_text = String::new();
                }
                current_text.push(sc.chr);
            }
            if !current_text.is_empty() {
                styled_vec.push(StyledString { text: current_text, style: current_style });
            }

            // We're done with the row, push and keep going
            output.push(styled_vec);

            styled_vec = vec![];
        }

        output
    }

    fn ensure_lines(&mut self, line: usize) {
        if line >= self.lines.len() {
            self.lines.resize(line + 1, Vec::new());
        }
    }

    /// Sets `chr` with `style` for given `line`, `col`.
    /// If `line` does not exist in our buffer, adds empty lines up to the given
    /// and fills the last line with unstyled whitespace.
    pub fn putc(&mut self, line: usize, col: usize, chr: char, style: Style) {
        self.ensure_lines(line);
        if col >= self.lines[line].len() {
            self.lines[line].resize(col + 1, StyledChar::SPACE);
        }
        self.lines[line][col] = StyledChar::new(chr, style);
    }

    /// Sets `string` with `style` for given `line`, starting from `col`.
    /// If `line` does not exist in our buffer, adds empty lines up to the given
    /// and fills the last line with unstyled whitespace.
    pub fn puts(&mut self, line: usize, col: usize, string: &str, style: Style) {
        let mut n = col;
        for c in string.chars() {
            self.putc(line, n, c, style);
            n += 1;
        }
    }

    /// For given `line` inserts `string` with `style` before old content of that line,
    /// adding lines if needed
    pub fn prepend(&mut self, line: usize, string: &str, style: Style) {
        self.ensure_lines(line);
        let string_len = string.chars().count();

        if !self.lines[line].is_empty() {
            // Push the old content over to make room for new content
            for _ in 0..string_len {
                self.lines[line].insert(0, StyledChar::SPACE);
            }
        }

        self.puts(line, 0, string, style);
    }

    /// For given `line` inserts `string` with `style` after old content of that line,
    /// adding lines if needed
    pub fn append(&mut self, line: usize, string: &str, style: Style) {
        if line >= self.lines.len() {
            self.puts(line, 0, string, style);
        } else {
            let col = self.lines[line].len();
            self.puts(line, col, string, style);
        }
    }

    pub fn num_lines(&self) -> usize {
        self.lines.len()
    }

    /// Set `style` for `line`, `col_start..col_end` range if:
    /// 1. That line and column range exist in `StyledBuffer`
    /// 2. `overwrite` is `true` or existing style is `Style::NoStyle` or `Style::Quotation`
    pub fn set_style_range(
        &mut self,
        line: usize,
        col_start: usize,
        col_end: usize,
        style: Style,
        overwrite: bool,
    ) {
        for col in col_start..col_end {
            self.set_style(line, col, style, overwrite);
        }
    }

    /// Set `style` for `line`, `col` if:
    /// 1. That line and column exist in `StyledBuffer`
    /// 2. `overwrite` is `true` or existing style is `Style::NoStyle` or `Style::Quotation`
    pub fn set_style(&mut self, line: usize, col: usize, style: Style, overwrite: bool) {
        if let Some(ref mut line) = self.lines.get_mut(line) {
            if let Some(StyledChar { style: s, .. }) = line.get_mut(col) {
                if overwrite || matches!(s, Style::NoStyle | Style::Quotation) {
                    *s = style;
                }
            }
        }
    }
}
