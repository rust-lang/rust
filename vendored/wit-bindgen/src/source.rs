use std::fmt::{self, Write};
use std::ops::Deref;

/// Helper structure to maintain indentation automatically when printing.
#[derive(Default)]
pub struct Source {
    s: String,
    indent: usize,
}

impl Source {
    pub fn push_str(&mut self, src: &str) {
        let lines = src.lines().collect::<Vec<_>>();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with('}') && self.s.ends_with("  ") {
                self.s.pop();
                self.s.pop();
            }
            self.s.push_str(if lines.len() == 1 {
                line
            } else {
                line.trim_start()
            });
            if trimmed.ends_with('{') {
                self.indent += 1;
            }
            if trimmed.starts_with('}') {
                // Note that a `saturating_sub` is used here to prevent a panic
                // here in the case of invalid code being generated in debug
                // mode. It's typically easier to debug those issues through
                // looking at the source code rather than getting a panic.
                self.indent = self.indent.saturating_sub(1);
            }
            if i != lines.len() - 1 || src.ends_with('\n') {
                self.newline();
            }
        }
    }

    pub fn indent(&mut self, amt: usize) {
        self.indent += amt;
    }

    pub fn deindent(&mut self, amt: usize) {
        self.indent -= amt;
    }

    fn newline(&mut self) {
        self.s.push('\n');
        for _ in 0..self.indent {
            self.s.push_str("  ");
        }
    }

    pub fn as_mut_string(&mut self) -> &mut String {
        &mut self.s
    }
}

impl Write for Source {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

impl Deref for Source {
    type Target = str;
    fn deref(&self) -> &str {
        &self.s
    }
}

impl From<Source> for String {
    fn from(s: Source) -> String {
        s.s
    }
}

#[cfg(test)]
mod tests {
    use super::Source;

    #[test]
    fn simple_append() {
        let mut s = Source::default();
        s.push_str("x");
        assert_eq!(s.s, "x");
        s.push_str("y");
        assert_eq!(s.s, "xy");
        s.push_str("z ");
        assert_eq!(s.s, "xyz ");
        s.push_str(" a ");
        assert_eq!(s.s, "xyz  a ");
        s.push_str("\na");
        assert_eq!(s.s, "xyz  a \na");
    }

    #[test]
    fn newline_remap() {
        let mut s = Source::default();
        s.push_str("function() {\n");
        s.push_str("y\n");
        s.push_str("}\n");
        assert_eq!(s.s, "function() {\n  y\n}\n");
    }

    #[test]
    fn if_else() {
        let mut s = Source::default();
        s.push_str("if() {\n");
        s.push_str("y\n");
        s.push_str("} else if () {\n");
        s.push_str("z\n");
        s.push_str("}\n");
        assert_eq!(s.s, "if() {\n  y\n} else if () {\n  z\n}\n");
    }

    #[test]
    fn trim_ws() {
        let mut s = Source::default();
        s.push_str(
            "function() {
                x
        }",
        );
        assert_eq!(s.s, "function() {\n  x\n}");
    }
}
