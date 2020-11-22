use crate::clean::{self, DocFragment, DocFragmentKind, Item};
use crate::core::DocContext;
use crate::fold;
use crate::fold::DocFolder;
use crate::passes::Pass;

use std::mem::take;

crate const COLLAPSE_DOCS: Pass = Pass {
    name: "collapse-docs",
    run: collapse_docs,
    description: "concatenates all document attributes into one document attribute",
};

crate fn collapse_docs(krate: clean::Crate, _: &DocContext<'_>) -> clean::Crate {
    let mut krate = Collapser.fold_crate(krate);
    krate.collapsed = true;
    krate
}

struct Collapser;

impl fold::DocFolder for Collapser {
    fn fold_item(&mut self, mut i: Item) -> Option<Item> {
        i.attrs.collapse_doc_comments();
        Some(self.fold_item_recur(i))
    }
}

fn collapse(doc_strings: &mut Vec<DocFragment>) {
    let mut docs = vec![];
    let mut last_frag: Option<DocFragment> = None;

    for frag in take(doc_strings) {
        if let Some(mut curr_frag) = last_frag.take() {
            let curr_kind = &curr_frag.kind;
            let new_kind = &frag.kind;

            if matches!(*curr_kind, DocFragmentKind::Include { .. })
                || curr_kind != new_kind
                || curr_frag.parent_module != frag.parent_module
            {
                if *curr_kind == DocFragmentKind::SugaredDoc
                    || *curr_kind == DocFragmentKind::RawDoc
                {
                    // add a newline for extra padding between segments
                    curr_frag.doc.push('\n');
                }
                docs.push(curr_frag);
                last_frag = Some(frag);
            } else {
                curr_frag.doc.push('\n');
                curr_frag.doc.push_str(&frag.doc);
                curr_frag.span = curr_frag.span.to(frag.span);
                last_frag = Some(curr_frag);
            }
        } else {
            last_frag = Some(frag);
        }
    }

    if let Some(frag) = last_frag.take() {
        docs.push(frag);
    }
    *doc_strings = docs;
}

impl clean::Attributes {
    crate fn collapse_doc_comments(&mut self) {
        collapse(&mut self.doc_strings);
    }
}
