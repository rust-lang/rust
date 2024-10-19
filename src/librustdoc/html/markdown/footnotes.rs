//! Markdown footnote handling.
use std::fmt::Write as _;

use pulldown_cmark::{Event, Tag, TagEnd, html};
use rustc_data_structures::fx::FxIndexMap;

use super::SpannedEvent;

/// Moves all footnote definitions to the end and add back links to the
/// references.
pub(super) struct Footnotes<'a, I> {
    inner: I,
    footnotes: FxIndexMap<String, FootnoteDef<'a>>,
}

/// The definition of a single footnote.
struct FootnoteDef<'a> {
    content: Vec<Event<'a>>,
    /// The number that appears in the footnote reference and list.
    id: u16,
}

impl<'a, I> Footnotes<'a, I> {
    pub(super) fn new(iter: I) -> Self {
        Footnotes { inner: iter, footnotes: FxIndexMap::default() }
    }

    fn get_entry(&mut self, key: &str) -> (&mut Vec<Event<'a>>, u16) {
        let new_id = self.footnotes.len() + 1;
        let key = key.to_owned();
        let FootnoteDef { content, id } = self
            .footnotes
            .entry(key)
            .or_insert(FootnoteDef { content: Vec::new(), id: new_id as u16 });
        // Don't allow changing the ID of existing entrys, but allow changing the contents.
        (content, *id)
    }
}

impl<'a, I: Iterator<Item = SpannedEvent<'a>>> Iterator for Footnotes<'a, I> {
    type Item = SpannedEvent<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some((Event::FootnoteReference(ref reference), range)) => {
                    // When we see a reference (to a footnote we may not know) the definition of,
                    // reserve a number for it, and emit a link to that number.
                    let (_, id) = self.get_entry(reference);
                    let reference =
                        format!("<sup id=\"fnref{0}\"><a href=\"#fn{0}\">{0}</a></sup>", id);
                    return Some((Event::Html(reference.into()), range));
                }
                Some((Event::Start(Tag::FootnoteDefinition(def)), _)) => {
                    // When we see a footnote definition, collect the assocated content, and store
                    // that for rendering later.
                    let content = collect_footnote_def(&mut self.inner);
                    let (entry_content, _) = self.get_entry(&def);
                    *entry_content = content;
                }
                Some(e) => return Some(e),
                None => {
                    if !self.footnotes.is_empty() {
                        // After all the markdown is emmited, emit an <hr> then all the footnotes
                        // in a list.
                        let defs: Vec<_> = self.footnotes.drain(..).map(|(_, x)| x).collect();
                        let defs_html = render_footnotes_defs(defs);
                        return Some((Event::Html(defs_html.into()), 0..0));
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

fn collect_footnote_def<'a>(events: impl Iterator<Item = SpannedEvent<'a>>) -> Vec<Event<'a>> {
    let mut content = Vec::new();
    for (event, _) in events {
        if let Event::End(TagEnd::FootnoteDefinition) = event {
            break;
        }
        content.push(event);
    }
    content
}

fn render_footnotes_defs(mut footnotes: Vec<FootnoteDef<'_>>) -> String {
    let mut ret = String::from("<div class=\"footnotes\"><hr><ol>");

    // Footnotes must listed in order of id, so the numbers the
    // browser generated for <li> are right.
    footnotes.sort_by_key(|x| x.id);

    for FootnoteDef { mut content, id } in footnotes {
        write!(ret, "<li id=\"fn{id}\">").unwrap();
        let mut is_paragraph = false;
        if let Some(&Event::End(TagEnd::Paragraph)) = content.last() {
            content.pop();
            is_paragraph = true;
        }
        html::push_html(&mut ret, content.into_iter());
        write!(ret, "&nbsp;<a href=\"#fnref{id}\">↩</a>").unwrap();
        if is_paragraph {
            ret.push_str("</p>");
        }
        ret.push_str("</li>");
    }
    ret.push_str("</ol></div>");

    ret
}
