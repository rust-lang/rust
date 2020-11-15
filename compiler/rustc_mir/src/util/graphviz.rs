use rustc_graphviz as dot;
use rustc_hir::def_id::DefId;
use rustc_index::vec::Idx;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use std::fmt::Debug;
use std::io::{self, Write};

use super::pretty::dump_mir_def_ids;

/// Write a graphviz DOT graph of a list of MIRs.
pub fn write_mir_graphviz<W>(tcx: TyCtxt<'_>, single: Option<DefId>, w: &mut W) -> io::Result<()>
where
    W: Write,
{
    let def_ids = dump_mir_def_ids(tcx, single);

    let use_subgraphs = def_ids.len() > 1;
    if use_subgraphs {
        writeln!(w, "digraph __crate__ {{")?;
    }

    for def_id in def_ids {
        let body = &tcx.optimized_mir(def_id);
        write_mir_fn_graphviz(tcx, body, use_subgraphs, w)?;
    }

    if use_subgraphs {
        writeln!(w, "}}")?;
    }

    Ok(())
}

// Must match `[0-9A-Za-z_]*`. This does not appear in the rendered graph, so
// it does not have to be user friendly.
pub fn graphviz_safe_def_name(def_id: DefId) -> String {
    format!("{}_{}", def_id.krate.index(), def_id.index.index(),)
}

/// Write a graphviz DOT graph of the MIR.
pub fn write_mir_fn_graphviz<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'_>,
    subgraph: bool,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    let def_id = body.source.def_id();
    let kind = if subgraph { "subgraph" } else { "digraph" };
    let cluster = if subgraph { "cluster_" } else { "" }; // Prints a border around MIR
    let def_name = graphviz_safe_def_name(def_id);
    writeln!(w, "{} {}Mir_{} {{", kind, cluster, def_name)?;

    // Global graph properties
    let font = format!(r#"fontname="{}""#, tcx.sess.opts.debugging_opts.graphviz_font);
    let mut graph_attrs = vec![&font[..]];
    let mut content_attrs = vec![&font[..]];

    let dark_mode = tcx.sess.opts.debugging_opts.graphviz_dark_mode;
    if dark_mode {
        graph_attrs.push(r#"bgcolor="black""#);
        graph_attrs.push(r#"fontcolor="white""#);
        content_attrs.push(r#"color="white""#);
        content_attrs.push(r#"fontcolor="white""#);
    }

    writeln!(w, r#"    graph [{}];"#, graph_attrs.join(" "))?;
    let content_attrs_str = content_attrs.join(" ");
    writeln!(w, r#"    node [{}];"#, content_attrs_str)?;
    writeln!(w, r#"    edge [{}];"#, content_attrs_str)?;

    // Graph label
    write_graph_label(tcx, body, w)?;

    // Nodes
    for (block, _) in body.basic_blocks().iter_enumerated() {
        write_node(block, body, dark_mode, w)?;
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
/// LabelText::HtmlStr from librustc_graphviz.)
///
/// `init` and `fini` are callbacks for emitting additional rows of
/// data (using HTML enclosed with `<tr>` in the emitted text).
pub fn write_node_label<W: Write, INIT, FINI>(
    block: BasicBlock,
    body: &Body<'_>,
    dark_mode: bool,
    w: &mut W,
    num_cols: u32,
    init: INIT,
    fini: FINI,
) -> io::Result<()>
where
    INIT: Fn(&mut W) -> io::Result<()>,
    FINI: Fn(&mut W) -> io::Result<()>,
{
    let data = &body[block];

    write!(w, r#"<table border="0" cellborder="1" cellspacing="0">"#)?;

    // Basic block number at the top.
    let (blk, bgcolor) = if data.is_cleanup {
        let color = if dark_mode { "royalblue" } else { "lightblue" };
        (format!("{} (cleanup)", block.index()), color)
    } else {
        let color = if dark_mode { "dimgray" } else { "gray" };
        (format!("{}", block.index()), color)
    };
    write!(
        w,
        r#"<tr><td bgcolor="{bgcolor}" {attrs} colspan="{colspan}">{blk}</td></tr>"#,
        attrs = r#"align="center""#,
        colspan = num_cols,
        blk = blk,
        bgcolor = bgcolor
    )?;

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
    write!(w, "</table>")
}

/// Write a graphviz DOT node for the given basic block.
fn write_node<W: Write>(
    block: BasicBlock,
    body: &Body<'_>,
    dark_mode: bool,
    w: &mut W,
) -> io::Result<()> {
    let def_id = body.source.def_id();
    // Start a new node with the label to follow, in one of DOT's pseudo-HTML tables.
    write!(w, r#"    {} [shape="none", label=<"#, node(def_id, block))?;
    write_node_label(block, body, dark_mode, w, 1, |_| Ok(()), |_| Ok(()))?;
    // Close the node label and the node itself.
    writeln!(w, ">];")
}

/// Write graphviz DOT edges with labels between the given basic block and all of its successors.
fn write_edges<W: Write>(source: BasicBlock, body: &Body<'_>, w: &mut W) -> io::Result<()> {
    let def_id = body.source.def_id();
    let terminator = body[source].terminator();
    let labels = terminator.kind.fmt_successor_labels();

    for (&target, label) in terminator.successors().zip(labels) {
        let src = node(def_id, source);
        let trg = node(def_id, target);
        writeln!(w, r#"    {} -> {} [label="{}"];"#, src, trg, label)?;
    }

    Ok(())
}

/// Write the graphviz DOT label for the overall graph. This is essentially a block of text that
/// will appear below the graph, showing the type of the `fn` this MIR represents and the types of
/// all the variables and temporaries.
fn write_graph_label<'tcx, W: Write>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'_>,
    w: &mut W,
) -> io::Result<()> {
    let def_id = body.source.def_id();

    write!(w, "    label=<fn {}(", dot::escape_html(&tcx.def_path_str(def_id)))?;

    // fn argument types.
    for (i, arg) in body.args_iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "{:?}: {}", Place::from(arg), escape(&body.local_decls[arg].ty))?;
    }

    write!(w, ") -&gt; {}", escape(&body.return_ty()))?;
    write!(w, r#"<br align="left"/>"#)?;

    for local in body.vars_and_temps_iter() {
        let decl = &body.local_decls[local];

        write!(w, "let ")?;
        if decl.mutability == Mutability::Mut {
            write!(w, "mut ")?;
        }

        write!(w, r#"{:?}: {};<br align="left"/>"#, Place::from(local), escape(&decl.ty))?;
    }

    for var_debug_info in &body.var_debug_info {
        write!(
            w,
            r#"debug {} =&gt; {};<br align="left"/>"#,
            var_debug_info.name,
            escape(&var_debug_info.place)
        )?;
    }

    writeln!(w, ">;")
}

fn node(def_id: DefId, block: BasicBlock) -> String {
    format!("bb{}__{}", block.index(), graphviz_safe_def_name(def_id))
}

fn escape<T: Debug>(t: &T) -> String {
    dot::escape_html(&format!("{:?}", t))
}
