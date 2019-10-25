// Code for creating styled buffers

use crate::snippet::{Style, StyledString};

#[derive(Debug)]
pub struct StyledBuffer {
    text: Vec<Vec<char>>,
    styles: Vec<Vec<Style>>,
}

impl StyledBuffer {
    pub fn new() -> StyledBuffer {
        StyledBuffer {
            text: vec![],
            styles: vec![],
        }
    }

    fn replace_tabs(&mut self) {
        for (line_pos, line) in self.text.iter_mut().enumerate() {
            let mut tab_pos = vec![];
            for (pos, c) in line.iter().enumerate() {
                if *c == '\t' {
                    tab_pos.push(pos);
                }
            }
            // start with the tabs at the end of the line to replace them with 4 space chars
            for pos in tab_pos.iter().rev() {
                assert_eq!(line.remove(*pos), '\t');
                // fix the position of the style to match up after replacing the tabs
                let s = self.styles[line_pos].remove(*pos);
                for _ in 0..4 {
                    line.insert(*pos, ' ');
                    self.styles[line_pos].insert(*pos, s);
                }
            }
        }
    }

    pub fn render(&mut self) -> Vec<Vec<StyledString>> {
        let mut output: Vec<Vec<StyledString>> = vec![];
        let mut styled_vec: Vec<StyledString> = vec![];

        // before we render, replace tabs with spaces
        self.replace_tabs();

        for (row, row_style) in self.text.iter().zip(&self.styles) {
            let mut current_style = Style::NoStyle;
            let mut current_text = String::new();

            for (&c, &s) in row.iter().zip(row_style) {
                if s != current_style {
                    if !current_text.is_empty() {
                        styled_vec.push(StyledString {
                            text: current_text,
                            style: current_style,
                        });
                    }
                    current_style = s;
                    current_text = String::new();
                }
                current_text.push(c);
            }
            if !current_text.is_empty() {
                styled_vec.push(StyledString {
                    text: current_text,
                    style: current_style,
                });
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
            self.styles.push(vec![]);
        }
    }

    pub fn putc(&mut self, line: usize, col: usize, chr: char, style: Style) {
        self.ensure_lines(line);
        if col < self.text[line].len() {
            self.text[line][col] = chr;
            self.styles[line][col] = style;
        } else {
            let mut i = self.text[line].len();
            while i < col {
                self.text[line].push(' ');
                self.styles[line].push(Style::NoStyle);
                i += 1;
            }
            self.text[line].push(chr);
            self.styles[line].push(style);
        }
    }

    pub fn puts(&mut self, line: usize, col: usize, string: &str, style: Style) {
        let mut n = col;
        for c in string.chars() {
            self.putc(line, n, c, style);
            n += 1;
        }
    }

    pub fn prepend(&mut self, line: usize, string: &str, style: Style) {
        self.ensure_lines(line);
        let string_len = string.len();

        // Push the old content over to make room for new content
        for _ in 0..string_len {
            self.styles[line].insert(0, Style::NoStyle);
            self.text[line].insert(0, ' ');
        }

        self.puts(line, 0, string, style);
    }

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

    pub fn set_style_range(&mut self,
                           line: usize,
                           col_start: usize,
                           col_end: usize,
                           style: Style,
                           overwrite: bool) {
        for col in col_start..col_end {
            self.set_style(line, col, style, overwrite);
        }
    }

    pub fn set_style(&mut self, line: usize, col: usize, style: Style, overwrite: bool) {
        if let Some(ref mut line) = self.styles.get_mut(line) {
            if let Some(s) = line.get_mut(col) {
                if *s == Style::NoStyle || *s == Style::Quotation || overwrite {
                    *s = style;
                }
            }
        }
    }
}
