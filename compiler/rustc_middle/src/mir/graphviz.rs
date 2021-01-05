use gsgdt::GraphvizSettings;
use rustc_graphviz as dot;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use std::fmt::Debug;
use std::io::{self, Write};

use super::generic_graph::mir_fn_to_generic_graph;
use super::pretty::dump_mir_def_ids;

/// Write a graphviz DOT graph of a list of MIRs.
pub fn write_mir_graphviz<W>(tcx: TyCtxt<'_>, single: Option<DefId>, w: &mut W) -> io::Result<()>
where
    W: Write,
{
    let def_ids = dump_mir_def_ids(tcx, single);

    let mirs =
        def_ids
            .iter()
            .flat_map(|def_id| {
                if tcx.is_const_fn_raw(*def_id) {
                    vec![tcx.optimized_mir(*def_id), tcx.mir_for_ctfe(*def_id)]
                } else {
                    vec![tcx.instance_mir(ty::InstanceDef::Item(ty::WithOptConstParam::unknown(
                        *def_id,
                    )))]
                }
            })
            .collect::<Vec<_>>();

    let use_subgraphs = mirs.len() > 1;
    if use_subgraphs {
        writeln!(w, "digraph __crate__ {{")?;
    }

    for mir in mirs {
        write_mir_fn_graphviz(tcx, mir, use_subgraphs, w)?;
    }

    if use_subgraphs {
        writeln!(w, "}}")?;
    }

    Ok(())
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

    // Graph label
    let mut label = String::from("");
    // FIXME: remove this unwrap
    write_graph_label(tcx, body, &mut label).unwrap();
    let g = mir_fn_to_generic_graph(tcx, body);
    let settings = GraphvizSettings {
        graph_attrs: Some(graph_attrs.join(" ")),
        node_attrs: Some(content_attrs.join(" ")),
        edge_attrs: Some(content_attrs.join(" ")),
        graph_label: Some(label),
    };
    g.to_dot(w, &settings, subgraph)
}

/// Write the graphviz DOT label for the overall graph. This is essentially a block of text that
/// will appear below the graph, showing the type of the `fn` this MIR represents and the types of
/// all the variables and temporaries.
fn write_graph_label<'tcx, W: std::fmt::Write>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'_>,
    w: &mut W,
) -> std::fmt::Result {
    let def_id = body.source.def_id();

    write!(w, "fn {}(", dot::escape_html(&tcx.def_path_str(def_id)))?;

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
            escape(&var_debug_info.value),
        )?;
    }

    Ok(())
}

fn escape<T: Debug>(t: &T) -> String {
    dot::escape_html(&format!("{:?}", t))
}
