//! Markdown footnote handling.
use std::fmt::Write as _;

use pulldown_cmark::{Event, Tag, TagEnd, html};
use rustc_data_structures::fx::FxIndexMap;

use super::SpannedEvent;

/// Moves all footnote definitions to the end and add back links to the
/// references.
pub(super) struct Footnotes<'a, I> {
    inner: I,
    footnotes: FxIndexMap<String, (Vec<Event<'a>>, u16)>,
}

impl<'a, I> Footnotes<'a, I> {
    pub(super) fn new(iter: I) -> Self {
        Footnotes { inner: iter, footnotes: FxIndexMap::default() }
    }

    fn get_entry(&mut self, key: &str) -> &mut (Vec<Event<'a>>, u16) {
        let new_id = self.footnotes.len() + 1;
        let key = key.to_owned();
        self.footnotes.entry(key).or_insert((Vec::new(), new_id as u16))
    }
}

impl<'a, I: Iterator<Item = SpannedEvent<'a>>> Iterator for Footnotes<'a, I> {
    type Item = SpannedEvent<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some((Event::FootnoteReference(ref reference), range)) => {
                    let entry = self.get_entry(reference);
                    let reference = format!(
                        "<sup id=\"fnref{0}\"><a href=\"#fn{0}\">{0}</a></sup>",
                        (*entry).1
                    );
                    return Some((Event::Html(reference.into()), range));
                }
                Some((Event::Start(Tag::FootnoteDefinition(def)), _)) => {
                    let mut content = Vec::new();
                    for (event, _) in &mut self.inner {
                        if let Event::End(TagEnd::FootnoteDefinition) = event {
                            break;
                        }
                        content.push(event);
                    }
                    let entry = self.get_entry(&def);
                    (*entry).0 = content;
                }
                Some(e) => return Some(e),
                None => {
                    if !self.footnotes.is_empty() {
                        let mut v: Vec<_> = self.footnotes.drain(..).map(|(_, x)| x).collect();
                        v.sort_by(|a, b| a.1.cmp(&b.1));
                        let mut ret = String::from("<div class=\"footnotes\"><hr><ol>");
                        for (mut content, id) in v {
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
                        return Some((Event::Html(ret.into()), 0..0));
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}
