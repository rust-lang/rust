//! Removes markdown from strings.
use pulldown_cmark::{Event, Parser, Tag};

/// Removes all markdown, keeping the text and code blocks
///
/// Currently limited in styling, i.e. no ascii tables or lists
pub(crate) fn remove_markdown(markdown: &str) -> String {
    let mut out = String::new();
    let parser = Parser::new(markdown);

    for event in parser {
        match event {
            Event::Text(text) | Event::Code(text) => out.push_str(&text),
            Event::SoftBreak | Event::HardBreak | Event::Rule | Event::End(Tag::CodeBlock(_)) => {
                out.push('\n')
            }
            _ => {}
        }
    }

    out
}
