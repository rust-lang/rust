//! Markdown footnote handling.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use pulldown_cmark::{CowStr, Event, Tag, TagEnd, html};
use rustc_data_structures::fx::FxIndexMap;

use super::SpannedEvent;

/// Moves all footnote definitions to the end and add back links to the
/// references.
pub(super) struct Footnotes<'a, I> {
    inner: I,
    footnotes: FxIndexMap<String, FootnoteDef<'a>>,
    existing_footnotes: Arc<AtomicUsize>,
    start_id: usize,
}

/// The definition of a single footnote.
struct FootnoteDef<'a> {
    content: Vec<Event<'a>>,
    /// The number that appears in the footnote reference and list.
    id: usize,
    /// The number of footnote references.
    num_refs: usize,
}

impl<'a, I: Iterator<Item = SpannedEvent<'a>>> Footnotes<'a, I> {
    pub(super) fn new(iter: I, existing_footnotes: Weak<AtomicUsize>) -> Self {
        let existing_footnotes =
            existing_footnotes.upgrade().expect("`existing_footnotes` was dropped");
        let start_id = existing_footnotes.load(Ordering::Relaxed);
        Footnotes { inner: iter, footnotes: FxIndexMap::default(), existing_footnotes, start_id }
    }

    fn get_entry(&mut self, key: &str) -> (&mut Vec<Event<'a>>, usize, &mut usize) {
        let new_id = self.footnotes.len() + 1 + self.start_id;
        let key = key.to_owned();
        let FootnoteDef { content, id, num_refs } = self
            .footnotes
            .entry(key)
            .or_insert(FootnoteDef { content: Vec::new(), id: new_id, num_refs: 0 });
        // Don't allow changing the ID of existing entries, but allow changing the contents.
        (content, *id, num_refs)
    }

    fn handle_footnote_reference(&mut self, reference: &CowStr<'a>) -> Event<'a> {
        // When we see a reference (to a footnote we may not know) the definition of,
        // reserve a number for it, and emit a link to that number.
        let (_, id, num_refs) = self.get_entry(reference);
        *num_refs += 1;
        let fnref_suffix = if *num_refs <= 1 { "".to_owned() } else { format!("-{num_refs}") };
        let reference = format!(
            "<sup id=\"fnref{0}{fnref_suffix}\"><a href=\"#fn{0}\">{1}</a></sup>",
            id,
            // Although the ID count is for the whole page, the footnote reference
            // are local to the item so we make this ID "local" when displayed.
            id - self.start_id
        );
        Event::Html(reference.into())
    }

    fn collect_footnote_def(&mut self) -> Vec<Event<'a>> {
        let mut content = Vec::new();
        while let Some((event, _)) = self.inner.next() {
            match event {
                Event::End(TagEnd::FootnoteDefinition) => break,
                Event::FootnoteReference(ref reference) => {
                    content.push(self.handle_footnote_reference(reference));
                }
                event => content.push(event),
            }
        }
        content
    }
}

impl<'a, I: Iterator<Item = SpannedEvent<'a>>> Iterator for Footnotes<'a, I> {
    type Item = SpannedEvent<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = self.inner.next();
            match next {
                Some((Event::FootnoteReference(ref reference), range)) => {
                    return Some((self.handle_footnote_reference(reference), range));
                }
                Some((Event::Start(Tag::FootnoteDefinition(def)), _)) => {
                    // When we see a footnote definition, collect the associated content, and store
                    // that for rendering later.
                    let content = self.collect_footnote_def();
                    let (entry_content, _, _) = self.get_entry(&def);
                    *entry_content = content;
                }
                Some(e) => return Some(e),
                None => {
                    if !self.footnotes.is_empty() {
                        // After all the markdown is emitted, emit an <hr> then all the footnotes
                        // in a list.
                        let defs: Vec<_> = self.footnotes.drain(..).map(|(_, x)| x).collect();
                        self.existing_footnotes.fetch_add(defs.len(), Ordering::Relaxed);
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

fn render_footnotes_defs(mut footnotes: Vec<FootnoteDef<'_>>) -> String {
    let mut ret = String::from("<div class=\"footnotes\"><hr><ol>");

    // Footnotes must listed in order of id, so the numbers the
    // browser generated for <li> are right.
    footnotes.sort_by_key(|x| x.id);

    for FootnoteDef { mut content, id, num_refs } in footnotes {
        write!(ret, "<li id=\"fn{id}\">").unwrap();
        let mut is_paragraph = false;
        if let Some(&Event::End(TagEnd::Paragraph)) = content.last() {
            content.pop();
            is_paragraph = true;
        }
        html::push_html(&mut ret, content.into_iter());
        if num_refs <= 1 {
            write!(ret, "&nbsp;<a href=\"#fnref{id}\">↩</a>").unwrap();
        } else {
            // There are multiple references to single footnote. Make the first
            // back link a single "a" element to make touch region larger.
            write!(ret, "&nbsp;<a href=\"#fnref{id}\">↩&nbsp;<sup>1</sup></a>").unwrap();
            for refid in 2..=num_refs {
                write!(ret, "&nbsp;<sup><a href=\"#fnref{id}-{refid}\">{refid}</a></sup>").unwrap();
            }
        }
        if is_paragraph {
            ret.push_str("</p>");
        }
        ret.push_str("</li>");
    }
    ret.push_str("</ol></div>");

    ret
}
