//! Markdown formatting.
//!
//! Sometimes, we want to display a "rich text" in the UI. At the moment, we use
//! markdown for this purpose. It doesn't feel like a right option, but that's
//! what is used by LSP, so let's keep it simple.
use std::fmt;

#[derive(Default, Debug)]
pub struct Markup {
    text: String,
}

impl From<Markup> for String {
    fn from(markup: Markup) -> Self {
        markup.text
    }
}

impl fmt::Display for Markup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl Markup {
    pub fn as_str(&self) -> &str {
        self.text.as_str()
    }
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
    pub fn push_section(&mut self, section: &str) {
        if !self.text.is_empty() {
            self.text.push_str("\n\n___\n");
        }
        self.text.push_str(section);
    }
}
