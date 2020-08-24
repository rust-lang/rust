//! Resolves and rewrites links in markdown documentation.
//!
//! Most of the implementation can be found in [`hir::doc_links`].

use pulldown_cmark::{CowStr, Event, Options, Parser, Tag};
use pulldown_cmark_to_cmark::{cmark_with_options, Options as CmarkOptions};

use hir::resolve_doc_link;
use ide_db::{defs::Definition, RootDatabase};

/// Rewrite documentation links in markdown to point to an online host (e.g. docs.rs)
pub fn rewrite_links(db: &RootDatabase, markdown: &str, definition: &Definition) -> String {
    let doc = Parser::new_with_broken_link_callback(
        markdown,
        Options::empty(),
        Some(&|label, _| Some((/*url*/ label.to_string(), /*title*/ label.to_string()))),
    );

    let doc = map_links(doc, |target, title: &str| {
        // This check is imperfect, there's some overlap between valid intra-doc links
        // and valid URLs so we choose to be too eager to try to resolve what might be
        // a URL.
        if target.contains("://") {
            (target.to_string(), title.to_string())
        } else {
            // Two posibilities:
            // * path-based links: `../../module/struct.MyStruct.html`
            // * module-based links (AKA intra-doc links): `super::super::module::MyStruct`
            let resolved = match definition {
                Definition::ModuleDef(t) => resolve_doc_link(db, t, title, target),
                Definition::Macro(t) => resolve_doc_link(db, t, title, target),
                Definition::Field(t) => resolve_doc_link(db, t, title, target),
                Definition::SelfType(t) => resolve_doc_link(db, t, title, target),
                Definition::Local(t) => resolve_doc_link(db, t, title, target),
                Definition::TypeParam(t) => resolve_doc_link(db, t, title, target),
            };

            match resolved {
                Some((target, title)) => (target, title),
                None => (target.to_string(), title.to_string()),
            }
        }
    });
    let mut out = String::new();
    let mut options = CmarkOptions::default();
    options.code_block_backticks = 3;
    cmark_with_options(doc, &mut out, None, options).ok();
    out
}

// Rewrites a markdown document, resolving links using `callback` and additionally striping prefixes/suffixes on link titles.
fn map_links<'e>(
    events: impl Iterator<Item = Event<'e>>,
    callback: impl Fn(&str, &str) -> (String, String),
) -> impl Iterator<Item = Event<'e>> {
    let mut in_link = false;
    let mut link_target: Option<CowStr> = None;

    events.map(move |evt| match evt {
        Event::Start(Tag::Link(_link_type, ref target, _)) => {
            in_link = true;
            link_target = Some(target.clone());
            evt
        }
        Event::End(Tag::Link(link_type, _target, _)) => {
            in_link = false;
            Event::End(Tag::Link(link_type, link_target.take().unwrap(), CowStr::Borrowed("")))
        }
        Event::Text(s) if in_link => {
            let (link_target_s, link_name) = callback(&link_target.take().unwrap(), &s);
            link_target = Some(CowStr::Boxed(link_target_s.into()));
            Event::Text(CowStr::Boxed(link_name.into()))
        }
        Event::Code(s) if in_link => {
            let (link_target_s, link_name) = callback(&link_target.take().unwrap(), &s);
            link_target = Some(CowStr::Boxed(link_target_s.into()));
            Event::Code(CowStr::Boxed(link_name.into()))
        }
        _ => evt,
    })
}
