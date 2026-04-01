use std::borrow::Cow;

use crate::pp::Printer;

impl Printer {
    pub fn word_space<W: Into<Cow<'static, str>>>(&mut self, w: W) {
        self.word(w);
        self.space();
    }

    pub fn popen(&mut self) {
        self.word("(");
    }

    pub fn pclose(&mut self) {
        self.word(")");
    }

    pub fn hardbreak_if_not_bol(&mut self) {
        if !self.is_beginning_of_line() {
            self.hardbreak()
        }
    }

    pub fn space_if_not_bol(&mut self) {
        if !self.is_beginning_of_line() {
            self.space();
        }
    }

    pub fn nbsp(&mut self) {
        self.word(" ")
    }

    pub fn word_nbsp<S: Into<Cow<'static, str>>>(&mut self, w: S) {
        self.word(w);
        self.nbsp()
    }

    /// Synthesizes a comment that was not textually present in the original
    /// source file.
    pub fn synth_comment(&mut self, text: impl Into<Cow<'static, str>>) {
        self.word("/*");
        self.space();
        self.word(text);
        self.space();
        self.word("*/")
    }
}
