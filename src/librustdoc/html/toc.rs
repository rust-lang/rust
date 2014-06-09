// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Table-of-contents creation.

use std::fmt;
use std::string::String;

/// A (recursive) table of contents
#[deriving(PartialEq)]
pub struct Toc {
    /// The levels are strictly decreasing, i.e.
    ///
    /// entries[0].level >= entries[1].level >= ...
    ///
    /// Normally they are equal, but can differ in cases like A and B,
    /// both of which end up in the same `Toc` as they have the same
    /// parent (Main).
    ///
    /// # Main
    /// ### A
    /// ## B
    entries: Vec<TocEntry>
}

impl Toc {
    fn count_entries_with_level(&self, level: u32) -> uint {
        self.entries.iter().filter(|e| e.level == level).count()
    }
}

#[deriving(PartialEq)]
pub struct TocEntry {
    level: u32,
    sec_number: String,
    name: String,
    id: String,
    children: Toc,
}

/// Progressive construction of a table of contents.
#[deriving(PartialEq)]
pub struct TocBuilder {
    top_level: Toc,
    /// The current hierarchy of parent headings, the levels are
    /// strictly increasing (i.e. chain[0].level < chain[1].level <
    /// ...) with each entry being the most recent occurrence of a
    /// heading with that level (it doesn't include the most recent
    /// occurrences of every level, just, if *is* in `chain` then is is
    /// the most recent one).
    ///
    /// We also have `chain[0].level <= top_level.entries[last]`.
    chain: Vec<TocEntry>
}

impl TocBuilder {
    pub fn new() -> TocBuilder {
        TocBuilder { top_level: Toc { entries: Vec::new() }, chain: Vec::new() }
    }


    /// Convert into a true `Toc` struct.
    pub fn into_toc(mut self) -> Toc {
        // we know all levels are >= 1.
        self.fold_until(0);
        self.top_level
    }

    /// Collapse the chain until the first heading more important than
    /// `level` (i.e. lower level)
    ///
    /// Example:
    ///
    /// ## A
    /// # B
    /// # C
    /// ## D
    /// ## E
    /// ### F
    /// #### G
    /// ### H
    ///
    /// If we are considering H (i.e. level 3), then A and B are in
    /// self.top_level, D is in C.children, and C, E, F, G are in
    /// self.chain.
    ///
    /// When we attempt to push H, we realise that first G is not the
    /// parent (level is too high) so it is popped from chain and put
    /// into F.children, then F isn't the parent (level is equal, aka
    /// sibling), so it's also popped and put into E.children.
    ///
    /// This leaves us looking at E, which does have a smaller level,
    /// and, by construction, it's the most recent thing with smaller
    /// level, i.e. it's the immediate parent of H.
    fn fold_until(&mut self, level: u32) {
        let mut this = None;
        loop {
            match self.chain.pop() {
                Some(mut next) => {
                    this.map(|e| next.children.entries.push(e));
                    if next.level < level {
                        // this is the parent we want, so return it to
                        // its rightful place.
                        self.chain.push(next);
                        return
                    } else {
                        this = Some(next);
                    }
                }
                None => {
                    this.map(|e| self.top_level.entries.push(e));
                    return
                }
            }
        }
    }

    /// Push a level `level` heading into the appropriate place in the
    /// hierarchy, returning a string containing the section number in
    /// `<num>.<num>.<num>` format.
    pub fn push<'a>(&'a mut self, level: u32, name: String, id: String) -> &'a str {
        assert!(level >= 1);

        // collapse all previous sections into their parents until we
        // get to relevant heading (i.e. the first one with a smaller
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
                    sec_number = String::from_str(entry.sec_number
                                                       .as_slice());
                    sec_number.push_str(".");
                    (entry.level, &entry.children)
                }
            };
            // fill in any missing zeros, e.g. for
            // # Foo (1)
            // ### Bar (1.0.1)
            for _ in range(toc_level, level - 1) {
                sec_number.push_str("0.");
            }
            let number = toc.count_entries_with_level(level);
            sec_number.push_str(format!("{}", number + 1).as_slice())
        }

        self.chain.push(TocEntry {
            level: level,
            name: name,
            sec_number: sec_number,
            id: id,
            children: Toc { entries: Vec::new() }
        });

        // get the thing we just pushed, so we can borrow the string
        // out of it with the right lifetime
        let just_inserted = self.chain.mut_last().unwrap();
        just_inserted.sec_number.as_slice()
    }
}

impl fmt::Show for Toc {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "<ul>"));
        for entry in self.entries.iter() {
            // recursively format this table of contents (the
            // `{children}` is the key).
            try!(write!(fmt,
                        "\n<li><a href=\"\\#{id}\">{num} {name}</a>{children}</li>",
                        id = entry.id,
                        num = entry.sec_number, name = entry.name,
                        children = entry.children))
        }
        write!(fmt, "</ul>")
    }
}

#[cfg(test)]
mod test {
    use super::{TocBuilder, Toc, TocEntry};

    #[test]
    fn builder_smoke() {
        let mut builder = TocBuilder::new();

        // this is purposely not using a fancy macro like below so
        // that we're sure that this is doing the correct thing, and
        // there's been no macro mistake.
        macro_rules! push {
            ($level: expr, $name: expr) => {
                assert_eq!(builder.push($level,
                                        $name.to_string(),
                                        "".to_string()),
                           $name);
            }
        }
        push!(2, "0.1");
        push!(1, "1");
        {
            push!(2, "1.1");
            {
                push!(3, "1.1.1");
                push!(3, "1.1.2");
            }
            push!(2, "1.2");
            {
                push!(3, "1.2.1");
                push!(3, "1.2.2");
            }
        }
        push!(1, "2");
        push!(1, "3");
        {
            push!(4, "3.0.0.1");
            {
                push!(6, "3.0.0.1.0.1");
            }
            push!(4, "3.0.0.2");
            push!(2, "3.1");
            {
                push!(4, "3.1.0.1");
            }
        }

        macro_rules! toc {
            ($(($level: expr, $name: expr, $(($sub: tt))* )),*) => {
                Toc {
                    entries: vec!(
                        $(
                            TocEntry {
                                level: $level,
                                name: $name.to_string(),
                                sec_number: $name.to_string(),
                                id: "".to_string(),
                                children: toc!($($sub),*)
                            }
                            ),*
                        )
                }
            }
        }
        let expected = toc!(
            (2, "0.1", ),

            (1, "1",
             ((2, "1.1", ((3, "1.1.1", )) ((3, "1.1.2", ))))
             ((2, "1.2", ((3, "1.2.1", )) ((3, "1.2.2", ))))
             ),

            (1, "2", ),

            (1, "3",
             ((4, "3.0.0.1", ((6, "3.0.0.1.0.1", ))))
             ((4, "3.0.0.2", ))
             ((2, "3.1", ((4, "3.1.0.1", ))))
             )
            );
        assert_eq!(expected, builder.into_toc());
    }
}
