use std::cmp;

use crate::clean::{self, DocFragment, DocFragmentKind, Item};
use crate::core::DocContext;
use crate::fold::{self, DocFolder};
use crate::passes::Pass;

#[cfg(test)]
mod tests;

crate const UNINDENT_COMMENTS: Pass = Pass {
    name: "unindent-comments",
    run: unindent_comments,
    description: "removes excess indentation on comments in order for markdown to like it",
};

crate fn unindent_comments(krate: clean::Crate, _: &DocContext<'_>) -> clean::Crate {
    CommentCleaner.fold_crate(krate)
}

struct CommentCleaner;

impl fold::DocFolder for CommentCleaner {
    fn fold_item(&mut self, mut i: Item) -> Option<Item> {
        i.attrs.unindent_doc_comments();
        Some(self.fold_item_recur(i))
    }
}

impl clean::Attributes {
    crate fn unindent_doc_comments(&mut self) {
        unindent_fragments(&mut self.doc_strings);
    }
}

fn unindent_fragments(docs: &mut Vec<DocFragment>) {
    // `add` is used in case the most common sugared doc syntax is used ("/// "). The other
    // fragments kind's lines are never starting with a whitespace unless they are using some
    // markdown formatting requiring it. Therefore, if the doc block have a mix between the two,
    // we need to take into account the fact that the minimum indent minus one (to take this
    // whitespace into account).
    //
    // For example:
    //
    // /// hello!
    // #[doc = "another"]
    //
    // In this case, you want "hello! another" and not "hello!  another".
    let add = if docs.windows(2).any(|arr| arr[0].kind != arr[1].kind)
        && docs.iter().any(|d| d.kind == DocFragmentKind::SugaredDoc)
    {
        // In case we have a mix of sugared doc comments and "raw" ones, we want the sugared one to
        // "decide" how much the minimum indent will be.
        1
    } else {
        0
    };

    // `min_indent` is used to know how much whitespaces from the start of each lines must be
    // removed. Example:
    //
    // ///     hello!
    // #[doc = "another"]
    //
    // In here, the `min_indent` is 1 (because non-sugared fragment are always counted with minimum
    // 1 whitespace), meaning that "hello!" will be considered a codeblock because it starts with 4
    // (5 - 1) whitespaces.
    let min_indent = match docs
        .iter()
        .map(|fragment| {
            fragment.doc.lines().fold(usize::MAX, |min_indent, line| {
                if line.chars().all(|c| c.is_whitespace()) {
                    min_indent
                } else {
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

    for fragment in docs {
        if fragment.doc.lines().count() == 0 {
            continue;
        }

        let min_indent = if fragment.kind != DocFragmentKind::SugaredDoc && min_indent > 0 {
            min_indent - add
        } else {
            min_indent
        };

        fragment.doc = fragment
            .doc
            .lines()
            .map(|line| {
                if line.chars().all(|c| c.is_whitespace()) {
                    line.to_string()
                } else {
                    assert!(line.len() >= min_indent);
                    line[min_indent..].to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
}
