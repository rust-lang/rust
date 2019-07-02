use crate::clean::{self, DocFragment, Item};
use crate::core::DocContext;
use crate::fold;
use crate::fold::{DocFolder};
use crate::passes::Pass;

use std::mem::take;

pub const COLLAPSE_DOCS: Pass = Pass {
    name: "collapse-docs",
    pass: collapse_docs,
    description: "concatenates all document attributes into one document attribute",
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DocFragmentKind {
    Sugared,
    Raw,
    Include,
}

impl DocFragment {
    fn kind(&self) -> DocFragmentKind {
        match *self {
            DocFragment::SugaredDoc(..) => DocFragmentKind::Sugared,
            DocFragment::RawDoc(..) => DocFragmentKind::Raw,
            DocFragment::Include(..) => DocFragmentKind::Include,
        }
    }
}

pub fn collapse_docs(krate: clean::Crate, _: &DocContext<'_>) -> clean::Crate {
    Collapser.fold_crate(krate)
}

struct Collapser;

impl fold::DocFolder for Collapser {
    fn fold_item(&mut self, mut i: Item) -> Option<Item> {
        i.attrs.collapse_doc_comments();
        self.fold_item_recur(i)
    }
}

fn collapse(doc_strings: &mut Vec<DocFragment>) {
    let mut docs = vec![];
    let mut last_frag: Option<DocFragment> = None;

    for frag in take(doc_strings) {
        if let Some(mut curr_frag) = last_frag.take() {
            let curr_kind = curr_frag.kind();
            let new_kind = frag.kind();

            if curr_kind == DocFragmentKind::Include || curr_kind != new_kind {
                match curr_frag {
                    DocFragment::SugaredDoc(_, _, ref mut doc_string)
                        | DocFragment::RawDoc(_, _, ref mut doc_string) => {
                            // add a newline for extra padding between segments
                            doc_string.push('\n');
                        }
                    _ => {}
                }
                docs.push(curr_frag);
                last_frag = Some(frag);
            } else {
                match curr_frag {
                    DocFragment::SugaredDoc(_, ref mut span, ref mut doc_string)
                        | DocFragment::RawDoc(_, ref mut span, ref mut doc_string) => {
                            doc_string.push('\n');
                            doc_string.push_str(frag.as_str());
                            *span = span.to(frag.span());
                        }
                    _ => unreachable!(),
                }
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
    pub fn collapse_doc_comments(&mut self) {
        collapse(&mut self.doc_strings);
    }
}
