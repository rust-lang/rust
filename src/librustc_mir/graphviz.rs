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
use rustc::middle::ty;
use std::io::{self, Write};

pub fn write_mir_graphviz<W: Write>(mir: &Mir, w: &mut W) -> io::Result<()> {
    try!(writeln!(w, "digraph Mir {{"));

    // Global graph properties
    try!(writeln!(w, r#"graph [fontname="monospace"];"#));
    try!(writeln!(w, r#"node [fontname="monospace"];"#));
    try!(writeln!(w, r#"edge [fontname="monospace"];"#));

    // Graph label
    try!(write_graph_label(mir, w));

    // Nodes
    for block in mir.all_basic_blocks() {
        try!(write_node(block, mir, w));
    }

    // Edges
    for source in mir.all_basic_blocks() {
        try!(write_edges(source, mir, w));
    }

    writeln!(w, "}}")
}

fn write_node<W: Write>(block: BasicBlock, mir: &Mir, w: &mut W) -> io::Result<()> {
    let data = mir.basic_block_data(block);

    try!(write!(w, r#"bb{} [shape="none", label=<"#, block.index()));
    try!(write!(w, r#"<table border="0" cellborder="1" cellspacing="0">"#));

    try!(write!(w, r#"<tr><td bgcolor="gray" align="center">"#));
    try!(write!(w, "{}", block.index()));
    try!(write!(w, "</td></tr>"));

    if !data.statements.is_empty() {
        try!(write!(w, r#"<tr><td align="left" balign="left">"#));
        for statement in &data.statements {
            try!(write!(w, "{}", dot::escape_html(&format!("{:?}", statement))));
            try!(write!(w, "<br/>"));
        }
        try!(write!(w, "</td></tr>"));
    }

    try!(write!(w, r#"<tr><td align="left">"#));

    let mut terminator_head = String::new();
    data.terminator.fmt_head(&mut terminator_head).unwrap();
    try!(write!(w, "{}", dot::escape_html(&terminator_head)));
    try!(write!(w, "</td></tr>"));

    try!(write!(w, "</table>"));
    writeln!(w, ">];")
}

fn write_edges<W: Write>(source: BasicBlock, mir: &Mir, w: &mut W) -> io::Result<()> {
    let terminator = &mir.basic_block_data(source).terminator;
    let labels = terminator.fmt_successor_labels();

    for (i, target) in terminator.successors().into_iter().enumerate() {
        try!(write!(w, "bb{} -> bb{}", source.index(), target.index()));
        try!(writeln!(w, r#" [label="{}"];"#, labels[i]));
    }

    Ok(())
}

fn write_graph_label<W: Write>(mir: &Mir, w: &mut W) -> io::Result<()> {
    try!(write!(w, "label=<"));
    try!(write!(w, "fn("));

    for (i, arg) in mir.arg_decls.iter().enumerate() {
        if i > 0 {
            try!(write!(w, ", "));
        }
        try!(write!(w, "{}", dot::escape_html(&format!("a{}: {:?}", i, arg.ty))));
    }

    try!(write!(w, "{}", dot::escape_html(") -> ")));

    match mir.return_ty {
        ty::FnOutput::FnConverging(ty) =>
            try!(write!(w, "{}", dot::escape_html(&format!("{:?}", ty)))),
        ty::FnOutput::FnDiverging =>
            try!(write!(w, "{}", dot::escape_html("!"))),
    }

    try!(write!(w, r#"<br align="left"/>"#));

    for (i, var) in mir.var_decls.iter().enumerate() {
        try!(write!(w, "let "));
        if var.mutability == Mutability::Mut {
            try!(write!(w, "mut "));
        }
        let text = format!("v{}: {:?}; // {}", i, var.ty, var.name);
        try!(write!(w, "{}", dot::escape_html(&text)));
        try!(write!(w, r#"<br align="left"/>"#));
    }

    for (i, temp) in mir.temp_decls.iter().enumerate() {
        try!(write!(w, "{}", dot::escape_html(&format!("let t{}: {:?};", i, temp.ty))));
        try!(write!(w, r#"<br align="left"/>"#));
    }

    writeln!(w, ">;")
}
