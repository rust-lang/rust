//! Removes markdown from strings.

use pulldown_cmark::{Event, Parser};

/// Removes all markdown, keeping the text and code blocks
pub fn remove_markdown(markdown: &str) -> String {
    let mut out = String::new();
    let parser = Parser::new(markdown);

    for event in parser {
        match event {
            Event::Text(text) | Event::Code(text) => out.push_str(&text),
            Event::SoftBreak | Event::HardBreak | Event::Rule => out.push('\n'),
            _ => {}
        }
    }

    out
}
