use super::{span_of_attrs, Pass};
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::opts;
use pulldown_cmark::{Event, Parser, Tag};
use rustc_feature::UnstableFeatures;
use rustc_session::lint;

pub const CHECK_AUTOMATIC_LINKS: Pass = Pass {
    name: "check-automatic-links",
    run: check_automatic_links,
    description: "detects URLS/email addresses that could be written using brackets",
};

struct AutomaticLinksLinter<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
}

impl<'a, 'tcx> AutomaticLinksLinter<'a, 'tcx> {
    fn new(cx: &'a DocContext<'tcx>) -> Self {
        AutomaticLinksLinter { cx }
    }
}

pub fn check_automatic_links(krate: Crate, cx: &DocContext<'_>) -> Crate {
    if !UnstableFeatures::from_environment().is_nightly_build() {
        krate
    } else {
        let mut coll = AutomaticLinksLinter::new(cx);

        coll.fold_crate(krate)
    }
}

impl<'a, 'tcx> DocFolder for AutomaticLinksLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let hir_id = match self.cx.as_local_hir_id(item.def_id) {
            Some(hir_id) => hir_id,
            None => {
                // If non-local, no need to check anything.
                return self.fold_item_recur(item);
            }
        };
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();
        if !dox.is_empty() {
            let cx = &self.cx;

            let p = Parser::new_ext(&dox, opts()).into_offset_iter();

            let mut title = String::new();
            let mut in_link = false;

            for (event, range) in p {
                match event {
                    Event::Start(Tag::Link(..)) => in_link = true,
                    Event::End(Tag::Link(_, url, _)) => {
                        in_link = false;
                        if url.as_ref() != title {
                            continue;
                        }
                        let sp = match super::source_span_for_markdown_range(
                            cx,
                            &dox,
                            &range,
                            &item.attrs,
                        ) {
                            Some(sp) => sp,
                            None => span_of_attrs(&item.attrs).unwrap_or(item.source.span()),
                        };
                        cx.tcx.struct_span_lint_hir(
                            lint::builtin::AUTOMATIC_LINKS,
                            hir_id,
                            sp,
                            |lint| {
                                lint.build("Unneeded long form for URL")
                                    .help(&format!("Try with `<{}>` instead", url))
                                    .emit()
                            },
                        );
                        title.clear();
                    }
                    Event::Text(s) if in_link => {
                        title.push_str(&s);
                    }
                    _ => {}
                }
            }
        }

        self.fold_item_recur(item)
    }
}
