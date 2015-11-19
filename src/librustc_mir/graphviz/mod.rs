// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dot;
use rustc::mir::repr::*;
use std::borrow::IntoCow;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct EdgeIndex {
    source: BasicBlock,
    target: BasicBlock,
    index: usize,
}

impl<'a,'tcx> dot::Labeller<'a, BasicBlock, EdgeIndex> for Mir<'tcx> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("Mir").unwrap()
    }

    fn node_id(&'a self, n: &BasicBlock) -> dot::Id<'a> {
        dot::Id::new(format!("BB{}", n.index())).unwrap()
    }

    fn node_shape(&'a self, _: &BasicBlock) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::label("none"))
    }

    fn node_label(&'a self, &n: &BasicBlock) -> dot::LabelText<'a> {
        let mut buffer = String::new();
        buffer.push_str("<TABLE ALIGN=\"LEFT\">");

        buffer.push_str("<TR><TD>");
        buffer.push_str(&format!("{:?}", n));
        buffer.push_str("</TD></TR>");

        let data = self.basic_block_data(n);
        for statement in &data.statements {
            buffer.push_str("<TR><TD>");
            buffer.push_str(&escape(format!("{:?}", statement)));
            buffer.push_str("</TD></TR>");
        }

        buffer.push_str("<TR><TD>");
        buffer.push_str(&escape(format!("{:?}", &data.terminator)));
        buffer.push_str("</TD></TR>");

        buffer.push_str("</TABLE>");

        dot::LabelText::html(buffer)
    }

    fn edge_label(&'a self, edge: &EdgeIndex) -> dot::LabelText<'a> {
        dot::LabelText::label(format!("{}", edge.index))
    }
}

impl<'a,'tcx> dot::GraphWalk<'a, BasicBlock, EdgeIndex> for Mir<'tcx> {
    fn nodes(&'a self) -> dot::Nodes<'a, BasicBlock> {
        self.all_basic_blocks().into_cow()
    }

    fn edges(&'a self) -> dot::Edges<'a, EdgeIndex> {
        self.all_basic_blocks()
            .into_iter()
            .flat_map(|source| {
                self.basic_block_data(source)
                    .terminator
                    .successors()
                    .iter()
                    .enumerate()
                    .map(move |(index, &target)| {
                        EdgeIndex {
                            source: source,
                            target: target,
                            index: index,
                        }
                    })
            })
            .collect::<Vec<_>>()
            .into_cow()
    }

    fn source(&'a self, edge: &EdgeIndex) -> BasicBlock {
        edge.source
    }

    fn target(&'a self, edge: &EdgeIndex) -> BasicBlock {
        edge.target
    }
}

fn escape(text: String) -> String {
    let text = dot::escape_html(&text);
    let text = all_to_subscript("Temp", text);
    let text = all_to_subscript("Var", text);
    let text = all_to_subscript("Arg", text);
    let text = all_to_subscript("BB", text);
    text
}

/// A call like `all_to_subscript("Temp", "Temp(123)")` will convert
/// to `Temp₁₂₃`.
fn all_to_subscript(header: &str, mut text: String) -> String {
    let mut offset = 0;
    while offset < text.len() {
        if let Some(text1) = to_subscript1(header, &text, &mut offset) {
            text = text1;
        }
    }
    return text;

    /// Looks for `Foo(\d*)` where `header=="Foo"` and replaces the `\d` with subscripts.
    /// Updates `offset` to point to the next location where we might want to search.
    /// Returns an updated string if changes were made, else None.
    fn to_subscript1(header: &str, text: &str, offset: &mut usize) -> Option<String> {
        let a = match text[*offset..].find(header) {
            None => {
                *offset = text.len();
                return None;
            }
            Some(a) => a + *offset,
        };

        // Example:
        //
        // header: "Foo"
        // text:   ....Foo(123)...
        //             ^  ^
        //             a  b

        let b = a + header.len();
        *offset = b;

        let mut chars = text[b..].chars();
        if Some('(') != chars.next() {
            return None;
        }

        let mut result = String::new();
        result.push_str(&text[..b]);

        while let Some(c) = chars.next() {
            if c == ')' {
                break;
            }
            if !c.is_digit(10) {
                return None;
            }

            // 0x208 is _0 in unicode, 0x209 is _1, etc
            const SUBSCRIPTS: &'static str = "₀₁₂₃₄₅₆₇₈₉";
            let n = (c as usize) - ('0' as usize);
            result.extend(SUBSCRIPTS.chars().skip(n).take(1));
        }

        result.extend(chars);
        return Some(result);
    }
}
