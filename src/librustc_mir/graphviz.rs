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
use rustc::hir::def_id::DefId;
use rustc::mir::*;
use rustc::ty::TyCtxt;
use std::fmt::Debug;
use std::io::{self, Write};
use syntax::ast::NodeId;

use rustc_data_structures::indexed_vec::Idx;

/// Write a graphviz DOT graph of a list of MIRs.
pub fn write_mir_graphviz<'a, 'b, 'tcx, W, I>(tcx: TyCtxt<'b, 'tcx, 'tcx>,
                                              iter: I,
                                              w: &mut W)
                                              -> io::Result<()>
    where W: Write, I: Iterator<Item=DefId>
{
    for def_id in iter {
        let nodeid = tcx.map.as_local_node_id(def_id).unwrap();
        let mir = &tcx.item_mir(def_id);

        writeln!(w, "digraph Mir_{} {{", nodeid)?;

        // Global graph properties
        writeln!(w, r#"    graph [fontname="monospace"];"#)?;
        writeln!(w, r#"    node [fontname="monospace"];"#)?;
        writeln!(w, r#"    edge [fontname="monospace"];"#)?;

        // Graph label
        write_graph_label(tcx, nodeid, mir, w)?;

        // Nodes
        for (block, _) in mir.basic_blocks().iter_enumerated() {
            write_node(block, mir, w)?;
        }

        // Edges
        for (source, _) in mir.basic_blocks().iter_enumerated() {
            write_edges(source, mir, w)?;
        }
        writeln!(w, "}}")?
    }
    Ok(())
}

/// Write a graphviz HTML-styled label for the given basic block, with
/// all necessary escaping already performed. (This is suitable for
/// emitting directly, as is done in this module, or for use with the
/// LabelText::HtmlStr from libgraphviz.)
///
/// `init` and `fini` are callbacks for emitting additional rows of
/// data (using HTML enclosed with `<tr>` in the emitted text).
pub fn write_node_label<W: Write, INIT, FINI>(block: BasicBlock,
                                              mir: &Mir,
                                              w: &mut W,
                                              num_cols: u32,
                                              init: INIT,
                                              fini: FINI) -> io::Result<()>
    where INIT: Fn(&mut W) -> io::Result<()>,
          FINI: Fn(&mut W) -> io::Result<()>
{
    let data = &mir[block];

    write!(w, r#"<table border="0" cellborder="1" cellspacing="0">"#)?;

    // Basic block number at the top.
    write!(w, r#"<tr><td {attrs} colspan="{colspan}">{blk}</td></tr>"#,
           attrs=r#"bgcolor="gray" align="center""#,
           colspan=num_cols,
           blk=block.index())?;

    init(w)?;

    // List of statements in the middle.
    if !data.statements.is_empty() {
        write!(w, r#"<tr><td align="left" balign="left">"#)?;
        for statement in &data.statements {
            write!(w, "{}<br/>", escape(statement))?;
        }
        write!(w, "</td></tr>")?;
    }

    // Terminator head at the bottom, not including the list of successor blocks. Those will be
    // displayed as labels on the edges between blocks.
    let mut terminator_head = String::new();
    data.terminator().kind.fmt_head(&mut terminator_head).unwrap();
    write!(w, r#"<tr><td align="left">{}</td></tr>"#, dot::escape_html(&terminator_head))?;

    fini(w)?;

    // Close the table
    writeln!(w, "</table>")
}

/// Write a graphviz DOT node for the given basic block.
fn write_node<W: Write>(block: BasicBlock, mir: &Mir, w: &mut W) -> io::Result<()> {
    // Start a new node with the label to follow, in one of DOT's pseudo-HTML tables.
    write!(w, r#"    {} [shape="none", label=<"#, node(block))?;
    write_node_label(block, mir, w, 1, |_| Ok(()), |_| Ok(()))?;
    // Close the node label and the node itself.
    writeln!(w, ">];")
}

/// Write graphviz DOT edges with labels between the given basic block and all of its successors.
fn write_edges<W: Write>(source: BasicBlock, mir: &Mir, w: &mut W) -> io::Result<()> {
    let terminator = mir[source].terminator();
    let labels = terminator.kind.fmt_successor_labels();

    for (&target, label) in terminator.successors().iter().zip(labels) {
        writeln!(w, r#"    {} -> {} [label="{}"];"#, node(source), node(target), label)?;
    }

    Ok(())
}

/// Write the graphviz DOT label for the overall graph. This is essentially a block of text that
/// will appear below the graph, showing the type of the `fn` this MIR represents and the types of
/// all the variables and temporaries.
fn write_graph_label<'a, 'tcx, W: Write>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                         nid: NodeId,
                                         mir: &Mir,
                                         w: &mut W)
                                         -> io::Result<()> {
    write!(w, "    label=<fn {}(", dot::escape_html(&tcx.node_path_str(nid)))?;

    // fn argument types.
    for (i, arg) in mir.args_iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "{:?}: {}", Lvalue::Local(arg), escape(&mir.local_decls[arg].ty))?;
    }

    write!(w, ") -&gt; {}", escape(mir.return_ty))?;
    write!(w, r#"<br align="left"/>"#)?;

    for local in mir.vars_and_temps_iter() {
        let decl = &mir.local_decls[local];

        write!(w, "let ")?;
        if decl.mutability == Mutability::Mut {
            write!(w, "mut ")?;
        }

        if let Some(name) = decl.name {
            write!(w, r#"{:?}: {}; // {}<br align="left"/>"#,
                   Lvalue::Local(local), escape(&decl.ty), name)?;
        } else {
            write!(w, r#"let mut {:?}: {};<br align="left"/>"#,
                   Lvalue::Local(local), escape(&decl.ty))?;
        }
    }

    writeln!(w, ">;")
}

fn node(block: BasicBlock) -> String {
    format!("bb{}", block.index())
}

fn escape<T: Debug>(t: &T) -> String {
    dot::escape_html(&format!("{:?}", t))
}
