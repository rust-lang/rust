//! Table-of-contents creation.
use crate::html::escape::Escape;

/// A (recursive) table of contents
#[derive(Debug, PartialEq)]
pub(crate) struct Toc {
    /// The levels are strictly decreasing, i.e.
    ///
    /// `entries[0].level >= entries[1].level >= ...`
    ///
    /// Normally they are equal, but can differ in cases like A and B,
    /// both of which end up in the same `Toc` as they have the same
    /// parent (Main).
    ///
    /// ```text
    /// # Main
    /// ### A
    /// ## B
    /// ```
    pub(crate) entries: Vec<TocEntry>,
}

impl Toc {
    fn count_entries_with_level(&self, level: u32) -> usize {
        self.entries.iter().filter(|e| e.level == level).count()
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct TocEntry {
    pub(crate) level: u32,
    pub(crate) sec_number: String,
    // name is a plain text header that works in a `title` tag
    // html includes `<code>` tags
    // the tooltip is used so that, when a toc is truncated,
    // you can mouse over it to see the whole thing
    pub(crate) name: String,
    pub(crate) html: String,
    pub(crate) id: String,
    pub(crate) children: Toc,
}

/// Progressive construction of a table of contents.
#[derive(PartialEq)]
pub(crate) struct TocBuilder {
    top_level: Toc,
    /// The current hierarchy of parent headings, the levels are
    /// strictly increasing (i.e., `chain[0].level < chain[1].level <
    /// ...`) with each entry being the most recent occurrence of a
    /// heading with that level (it doesn't include the most recent
    /// occurrences of every level, just, if it *is* in `chain` then
    /// it is the most recent one).
    ///
    /// We also have `chain[0].level <= top_level.entries[last]`.
    chain: Vec<TocEntry>,
}

impl TocBuilder {
    pub(crate) fn new() -> TocBuilder {
        TocBuilder { top_level: Toc { entries: Vec::new() }, chain: Vec::new() }
    }

    /// Converts into a true `Toc` struct.
    pub(crate) fn into_toc(mut self) -> Toc {
        // we know all levels are >= 1.
        self.fold_until(0);
        self.top_level
    }

    /// Collapse the chain until the first heading more important than
    /// `level` (i.e., lower level)
    ///
    /// Example:
    ///
    /// ```text
    /// ## A
    /// # B
    /// # C
    /// ## D
    /// ## E
    /// ### F
    /// #### G
    /// ### H
    /// ```
    ///
    /// If we are considering H (i.e., level 3), then A and B are in
    /// self.top_level, D is in C.children, and C, E, F, G are in
    /// self.chain.
    ///
    /// When we attempt to push H, we realize that first G is not the
    /// parent (level is too high) so it is popped from chain and put
    /// into F.children, then F isn't the parent (level is equal, aka
    /// sibling), so it's also popped and put into E.children.
    ///
    /// This leaves us looking at E, which does have a smaller level,
    /// and, by construction, it's the most recent thing with smaller
    /// level, i.e., it's the immediate parent of H.
    fn fold_until(&mut self, level: u32) {
        let mut this = None;
        loop {
            match self.chain.pop() {
                Some(mut next) => {
                    next.children.entries.extend(this);
                    if next.level < level {
                        // this is the parent we want, so return it to
                        // its rightful place.
                        self.chain.push(next);
                        return;
                    } else {
                        this = Some(next);
                    }
                }
                None => {
                    self.top_level.entries.extend(this);
                    return;
                }
            }
        }
    }

    /// Push a level `level` heading into the appropriate place in the
    /// hierarchy, returning a string containing the section number in
    /// `<num>.<num>.<num>` format.
    pub(crate) fn push(&mut self, level: u32, name: String, html: String, id: String) -> &str {
        assert!(level >= 1);

        // collapse all previous sections into their parents until we
        // get to relevant heading (i.e., the first one with a smaller
        // level than us)
        self.fold_until(level);

        let mut sec_number;
        {
            let (toc_level, toc) = match self.chain.last() {
                None => {
                    sec_number = String::new();
                    (0, &self.top_level)
                }
                Some(entry) => {
                    sec_number = entry.sec_number.clone();
                    sec_number.push('.');
                    (entry.level, &entry.children)
                }
            };
            // fill in any missing zeros, e.g., for
            // # Foo (1)
            // ### Bar (1.0.1)
            for _ in toc_level..level - 1 {
                sec_number.push_str("0.");
            }
            let number = toc.count_entries_with_level(level);
            sec_number.push_str(&(number + 1).to_string())
        }

        self.chain.push(TocEntry {
            level,
            name,
            html,
            sec_number,
            id,
            children: Toc { entries: Vec::new() },
        });

        // get the thing we just pushed, so we can borrow the string
        // out of it with the right lifetime
        let just_inserted = self.chain.last_mut().unwrap();
        &just_inserted.sec_number
    }
}

impl Toc {
    fn print_inner(&self, v: &mut String) {
        use std::fmt::Write as _;

        v.push_str("<ul>");
        for entry in &self.entries {
            // recursively format this table of contents
            let _ = write!(
                v,
                "\n<li><a href=\"#{id}\" title=\"{name}\">{num} {html}</a>",
                id = entry.id,
                num = entry.sec_number,
                name = Escape(&entry.name),
                html = &entry.html,
            );
            entry.children.print_inner(&mut *v);
            v.push_str("</li>");
        }
        v.push_str("</ul>");
    }
    pub(crate) fn print(&self) -> String {
        let mut v = String::new();
        self.print_inner(&mut v);
        v
    }
}

#[cfg(test)]
mod tests;
