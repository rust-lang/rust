// Code for creating styled buffers

use crate::snippet::{Style, StyledString};

#[derive(Debug)]
pub struct StyledBuffer {
    text: Vec<Vec<StyledChar>>,
}

#[derive(Debug)]
struct StyledChar {
    chr: char,
    style: Style,
}

impl StyledChar {
    fn new(chr: char, style: Style) -> Self {
        StyledChar { chr, style }
    }
}

impl Default for StyledChar {
    fn default() -> Self {
        StyledChar::new(' ', Style::NoStyle)
    }
}

impl StyledBuffer {
    pub fn new() -> StyledBuffer {
        StyledBuffer { text: vec![] }
    }

    /// Returns content of `StyledBuffer` splitted by lines and line styles
    pub fn render(&self) -> Vec<Vec<StyledString>> {
        // Tabs are assumed to have been replaced by spaces in calling code.
        debug_assert!(self.text.iter().all(|r| !r.iter().any(|sc| sc.chr == '\t')));

        let mut output: Vec<Vec<StyledString>> = vec![];
        let mut styled_vec: Vec<StyledString> = vec![];

        for styled_row in &self.text {
            let mut current_style = Style::NoStyle;
            let mut current_text = String::new();

            for sc in styled_row {
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
        while line >= self.text.len() {
            self.text.push(vec![]);
        }
    }

    /// Sets `chr` with `style` for given `line`, `col`.
    /// If line not exist in `StyledBuffer`, adds lines up to given
    /// and fills last line with spaces and `Style::NoStyle` style
    pub fn putc(&mut self, line: usize, col: usize, chr: char, style: Style) {
        self.ensure_lines(line);
        if col < self.text[line].len() {
            self.text[line][col] = StyledChar::new(chr, style);
        } else {
            let mut i = self.text[line].len();
            while i < col {
                self.text[line].push(StyledChar::default());
                i += 1;
            }
            self.text[line].push(StyledChar::new(chr, style));
        }
    }

    /// Sets `string` with `style` for given `line`, starting from `col`.
    /// If line not exist in `StyledBuffer`, adds lines up to given
    /// and fills last line with spaces and `Style::NoStyle` style
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

        // Push the old content over to make room for new content
        for _ in 0..string_len {
            self.text[line].insert(0, StyledChar::default());
        }

        self.puts(line, 0, string, style);
    }

    /// For given `line` inserts `string` with `style` after old content of that line,
    /// adding lines if needed
    pub fn append(&mut self, line: usize, string: &str, style: Style) {
        if line >= self.text.len() {
            self.puts(line, 0, string, style);
        } else {
            let col = self.text[line].len();
            self.puts(line, col, string, style);
        }
    }

    pub fn num_lines(&self) -> usize {
        self.text.len()
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
    /// 2. Existing style is `Style::NoStyle` or `Style::Quotation` or `overwrite` is `true`
    pub fn set_style(&mut self, line: usize, col: usize, style: Style, overwrite: bool) {
        if let Some(ref mut line) = self.text.get_mut(line) {
            if let Some(StyledChar { style: s, .. }) = line.get_mut(col) {
                if *s == Style::NoStyle || *s == Style::Quotation || overwrite {
                    *s = style;
                }
            }
        }
    }
}
