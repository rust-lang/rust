use std::cmp;

use crate::clean::{self, DocFragment, DocFragmentKind, Item};
use crate::core::DocContext;
use crate::fold::{self, DocFolder};
use crate::passes::Pass;

#[cfg(test)]
mod tests;

pub const UNINDENT_COMMENTS: Pass = Pass {
    name: "unindent-comments",
    run: unindent_comments,
    description: "removes excess indentation on comments in order for markdown to like it",
};

pub fn unindent_comments(krate: clean::Crate, _: &DocContext<'_>) -> clean::Crate {
    CommentCleaner.fold_crate(krate)
}

struct CommentCleaner;

impl fold::DocFolder for CommentCleaner {
    fn fold_item(&mut self, mut i: Item) -> Option<Item> {
        i.attrs.unindent_doc_comments();
        self.fold_item_recur(i)
    }
}

impl clean::Attributes {
    pub fn unindent_doc_comments(&mut self) {
        unindent_fragments(&mut self.doc_strings);
    }
}

fn unindent_fragments(docs: &mut Vec<DocFragment>) {
    let mut saw_first_line = false;
    let mut saw_second_line = false;

    let add = if !docs.windows(2).all(|arr| arr[0].kind == arr[1].kind)
        && docs.iter().any(|d| d.kind == DocFragmentKind::SugaredDoc)
    {
        // In case we have a mix of sugared doc comments and "raw" ones, we want the sugared one to
        // "decide" how much the minimum indent will be.
        1
    } else {
        0
    };

    let min_indent = match docs
        .iter()
        .map(|fragment| {
            fragment.doc.lines().fold(usize::MAX, |min_indent, line| {
                // After we see the first non-whitespace line, look at
                // the line we have. If it is not whitespace, and therefore
                // part of the first paragraph, then ignore the indentation
                // level of the first line
                let ignore_previous_indents =
                    saw_first_line && !saw_second_line && !line.chars().all(|c| c.is_whitespace());

                let min_indent = if ignore_previous_indents { usize::MAX } else { min_indent };

                if saw_first_line {
                    saw_second_line = true;
                }

                if line.chars().all(|c| c.is_whitespace()) {
                    min_indent
                } else {
                    saw_first_line = true;
                    // Compare against either space or tab, ignoring whether they are
                    // mixed or not.
                    let whitespace = line.chars().take_while(|c| *c == ' ' || *c == '\t').count();
                    cmp::min(min_indent, whitespace)
                        + if fragment.kind == DocFragmentKind::SugaredDoc { 0 } else { add }
                }
            })
        })
        .min()
    {
        Some(x) => x,
        None => return,
    };

    let mut first_ignored = false;
    for fragment in docs {
        let lines: Vec<_> = fragment.doc.lines().collect();

        if !lines.is_empty() {
            let min_indent = if fragment.kind != DocFragmentKind::SugaredDoc && min_indent > 0 {
                min_indent - add
            } else {
                min_indent
            };

            let mut iter = lines.iter();
            let mut result = if !first_ignored {
                first_ignored = true;
                vec![iter.next().unwrap().trim_start().to_string()]
            } else {
                Vec::new()
            };
            result.extend_from_slice(
                &iter
                    .map(|&line| {
                        if line.chars().all(|c| c.is_whitespace()) {
                            line.to_string()
                        } else {
                            assert!(line.len() >= min_indent);
                            line[min_indent..].to_string()
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            fragment.doc = result.join("\n");
        }
    }
}
