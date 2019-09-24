use rustc::hir::def_id::DefId;
use rustc::mir::*;
use rustc::ty::TyCtxt;
use rustc_data_structures::indexed_vec::Idx;
use std::fmt::Debug;
use std::io::{self, Write};

use super::pretty::dump_mir_def_ids;

/// Write a graphviz DOT graph of a list of MIRs.
pub fn write_mir_graphviz<W>(
    tcx: TyCtxt<'_>,
    single: Option<DefId>,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    for def_id in dump_mir_def_ids(tcx, single) {
        let body = &tcx.optimized_mir(def_id);
        write_mir_fn_graphviz(tcx, def_id, body, w)?;
    }
    Ok(())
}

// Must match `[0-9A-Za-z_]*`. This does not appear in the rendered graph, so
// it does not have to be user friendly.
pub fn graphviz_safe_def_name(def_id: DefId) -> String {
    format!(
        "{}_{}",
        def_id.krate.index(),
        def_id.index.index(),
    )
}

/// Write a graphviz DOT graph of the MIR.
pub fn write_mir_fn_graphviz<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    body: &Body<'_>,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    writeln!(w, "digraph Mir_{} {{", graphviz_safe_def_name(def_id))?;

    // Global graph properties
    writeln!(w, r#"    graph [fontname="monospace"];"#)?;
    writeln!(w, r#"    node [fontname="monospace"];"#)?;
    writeln!(w, r#"    edge [fontname="monospace"];"#)?;

    // Graph label
    write_graph_label(tcx, def_id, body, w)?;

    // Nodes
    for (block, _) in body.basic_blocks().iter_enumerated() {
        write_node(block, body, w)?;
    }

    // Edges
    for (source, _) in body.basic_blocks().iter_enumerated() {
        write_edges(source, body, w)?;
    }
    writeln!(w, "}}")
}

/// Write a graphviz HTML-styled label for the given basic block, with
/// all necessary escaping already performed. (This is suitable for
/// emitting directly, as is done in this module, or for use with the
/// LabelText::HtmlStr from libgraphviz.)
///
/// `init` and `fini` are callbacks for emitting additional rows of
/// data (using HTML enclosed with `<tr>` in the emitted text).
pub fn write_node_label<W: Write, INIT, FINI>(block: BasicBlock,
                                              body: &Body<'_>,
                                              w: &mut W,
                                              num_cols: u32,
                                              init: INIT,
                                              fini: FINI) -> io::Result<()>
    where INIT: Fn(&mut W) -> io::Result<()>,
          FINI: Fn(&mut W) -> io::Result<()>
{
    let data = &body[block];

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
fn write_node<W: Write>(block: BasicBlock, body: &Body<'_>, w: &mut W) -> io::Result<()> {
    // Start a new node with the label to follow, in one of DOT's pseudo-HTML tables.
    write!(w, r#"    {} [shape="none", label=<"#, node(block))?;
    write_node_label(block, body, w, 1, |_| Ok(()), |_| Ok(()))?;
    // Close the node label and the node itself.
    writeln!(w, ">];")
}

/// Write graphviz DOT edges with labels between the given basic block and all of its successors.
fn write_edges<W: Write>(source: BasicBlock, body: &Body<'_>, w: &mut W) -> io::Result<()> {
    let terminator = body[source].terminator();
    let labels = terminator.kind.fmt_successor_labels();

    for (&target, label) in terminator.successors().zip(labels) {
        writeln!(w, r#"    {} -> {} [label="{}"];"#, node(source), node(target), label)?;
    }

    Ok(())
}

/// Write the graphviz DOT label for the overall graph. This is essentially a block of text that
/// will appear below the graph, showing the type of the `fn` this MIR represents and the types of
/// all the variables and temporaries.
fn write_graph_label<'tcx, W: Write>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    body: &Body<'_>,
    w: &mut W,
) -> io::Result<()> {
    write!(w, "    label=<fn {}(", dot::escape_html(&tcx.def_path_str(def_id)))?;

    // fn argument types.
    for (i, arg) in body.args_iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w,
               "{:?}: {}",
               Place::from(arg),
               escape(&body.local_decls[arg].ty)
        )?;
    }

    write!(w, ") -&gt; {}", escape(&body.return_ty()))?;
    write!(w, r#"<br align="left"/>"#)?;

    for local in body.vars_and_temps_iter() {
        let decl = &body.local_decls[local];

        write!(w, "let ")?;
        if decl.mutability == Mutability::Mut {
            write!(w, "mut ")?;
        }

        if let Some(name) = decl.name {
            write!(w, r#"{:?}: {}; // {}<br align="left"/>"#,
                   Place::from(local), escape(&decl.ty), name)?;
        } else {
            write!(w, r#"{:?}: {};<br align="left"/>"#,
                   Place::from(local), escape(&decl.ty))?;
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
