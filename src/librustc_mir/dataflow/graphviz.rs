//! Hook into libgraphviz for rendering dataflow graphs for MIR.

use rustc::hir::def_id::DefId;
use rustc::mir::{BasicBlock, Body};

use std::fs;
use std::io;
use std::marker::PhantomData;
use std::path::Path;

use crate::util::graphviz_safe_def_name;

use super::{BitDenotation, DataflowState};
use super::DataflowBuilder;
use super::DebugFormatted;

pub trait MirWithFlowState<'tcx> {
    type BD: BitDenotation<'tcx>;
    fn def_id(&self) -> DefId;
    fn body(&self) -> &Body<'tcx>;
    fn flow_state(&self) -> &DataflowState<'tcx, Self::BD>;
}

impl<'a, 'tcx, BD> MirWithFlowState<'tcx> for DataflowBuilder<'a, 'tcx, BD>
    where BD: BitDenotation<'tcx>
{
    type BD = BD;
    fn def_id(&self) -> DefId { self.def_id }
    fn body(&self) -> &Body<'tcx> { self.flow_state.body() }
    fn flow_state(&self) -> &DataflowState<'tcx, Self::BD> { &self.flow_state.flow_state }
}

struct Graph<'a, 'tcx, MWF, P> where
    MWF: MirWithFlowState<'tcx>
{
    mbcx: &'a MWF,
    phantom: PhantomData<&'tcx ()>,
    render_idx: P,
}

pub(crate) fn print_borrowck_graph_to<'a, 'tcx, BD, P>(
    mbcx: &DataflowBuilder<'a, 'tcx, BD>,
    path: &Path,
    render_idx: P)
    -> io::Result<()>
    where BD: BitDenotation<'tcx>,
          P: Fn(&BD, BD::Idx) -> DebugFormatted,
{
    let g = Graph { mbcx, phantom: PhantomData, render_idx };
    let mut v = Vec::new();
    dot::render(&g, &mut v)?;
    debug!("print_borrowck_graph_to path: {} def_id: {:?}",
           path.display(), mbcx.def_id);
    fs::write(path, v)
}

pub type Node = BasicBlock;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Edge { source: BasicBlock, index: usize }

fn outgoing(body: &Body<'_>, bb: BasicBlock) -> Vec<Edge> {
    (0..body[bb].terminator().successors().count())
        .map(|index| Edge { source: bb, index: index}).collect()
}

impl<'a, 'tcx, MWF, P> dot::Labeller<'a> for Graph<'a, 'tcx, MWF, P>
    where MWF: MirWithFlowState<'tcx>,
          P: Fn(&MWF::BD, <MWF::BD as BitDenotation<'tcx>>::Idx) -> DebugFormatted,
{
    type Node = Node;
    type Edge = Edge;
    fn graph_id(&self) -> dot::Id<'_> {
        let name = graphviz_safe_def_name(self.mbcx.def_id());
        dot::Id::new(format!("graph_for_def_id_{}", name)).unwrap()
    }

    fn node_id(&self, n: &Node) -> dot::Id<'_> {
        dot::Id::new(format!("bb_{}", n.index()))
            .unwrap()
    }

    fn node_label(&self, n: &Node) -> dot::LabelText<'_> {
        // Node label is something like this:
        // +---------+----------------------------------+------------------+------------------+
        // | ENTRY   | MIR                              | GEN              | KILL             |
        // +---------+----------------------------------+------------------+------------------+
        // |         |  0: StorageLive(_7)              | bb3[2]: reserved | bb2[0]: reserved |
        // |         |  1: StorageLive(_8)              | bb3[2]: active   | bb2[0]: active   |
        // |         |  2: _8 = &mut _1                 |                  | bb4[2]: reserved |
        // |         |                                  |                  | bb4[2]: active   |
        // |         |                                  |                  | bb9[0]: reserved |
        // |         |                                  |                  | bb9[0]: active   |
        // |         |                                  |                  | bb10[0]: reserved|
        // |         |                                  |                  | bb10[0]: active  |
        // |         |                                  |                  | bb11[0]: reserved|
        // |         |                                  |                  | bb11[0]: active  |
        // +---------+----------------------------------+------------------+------------------+
        // | [00-00] | _7 = const Foo::twiddle(move _8) | [0c-00]          | [f3-0f]          |
        // +---------+----------------------------------+------------------+------------------+
        let mut v = Vec::new();
        self.node_label_internal(n, &mut v, *n, self.mbcx.body()).unwrap();
        dot::LabelText::html(String::from_utf8(v).unwrap())
    }


    fn node_shape(&self, _n: &Node) -> Option<dot::LabelText<'_>> {
        Some(dot::LabelText::label("none"))
    }

    fn edge_label(&'a self, e: &Edge) -> dot::LabelText<'a> {
        let term = self.mbcx.body()[e.source].terminator();
        let label = &term.kind.fmt_successor_labels()[e.index];
        dot::LabelText::label(label.clone())
    }
}

impl<'a, 'tcx, MWF, P> Graph<'a, 'tcx, MWF, P>
where MWF: MirWithFlowState<'tcx>,
      P: Fn(&MWF::BD, <MWF::BD as BitDenotation<'tcx>>::Idx) -> DebugFormatted,
{
    /// Generate the node label
    fn node_label_internal<W: io::Write>(&self,
                                         n: &Node,
                                         w: &mut W,
                                         block: BasicBlock,
                                         body: &Body<'_>) -> io::Result<()> {
        // Header rows
        const HDRS: [&str; 4] = ["ENTRY", "MIR", "BLOCK GENS", "BLOCK KILLS"];
        const HDR_FMT: &str = "bgcolor=\"grey\"";
        write!(w, "<table><tr><td rowspan=\"{}\">", HDRS.len())?;
        write!(w, "{:?}", block.index())?;
        write!(w, "</td></tr><tr>")?;
        for hdr in &HDRS {
            write!(w, "<td {}>{}</td>", HDR_FMT, hdr)?;
        }
        write!(w, "</tr>")?;

        // Data row
        self.node_label_verbose_row(n, w, block, body)?;
        self.node_label_final_row(n, w, block, body)?;
        write!(w, "</table>")?;

        Ok(())
    }

    /// Builds the verbose row: full MIR data, and detailed gen/kill/entry sets.
    fn node_label_verbose_row<W: io::Write>(&self,
                                            n: &Node,
                                            w: &mut W,
                                            block: BasicBlock,
                                            body: &Body<'_>)
                                            -> io::Result<()> {
        let i = n.index();

        macro_rules! dump_set_for {
            ($set:ident, $interpret:ident) => {
                write!(w, "<td>")?;

                let flow = self.mbcx.flow_state();
                let entry_interp = flow.$interpret(&flow.operator,
                                                   flow.sets.$set(i),
                                                   &self.render_idx);
                for e in &entry_interp {
                    write!(w, "{:?}<br/>", e)?;
                }
                write!(w, "</td>")?;
            }
        }

        write!(w, "<tr>")?;
        // Entry
        dump_set_for!(entry_set_for, interpret_set);

        // MIR statements
        write!(w, "<td>")?;
        {
            let data = &body[block];
            for (i, statement) in data.statements.iter().enumerate() {
                write!(w, "{}<br align=\"left\"/>",
                       dot::escape_html(&format!("{:3}: {:?}", i, statement)))?;
            }
        }
        write!(w, "</td>")?;

        // Gen
        dump_set_for!(gen_set_for, interpret_hybrid_set);

        // Kill
        dump_set_for!(kill_set_for, interpret_hybrid_set);

        write!(w, "</tr>")?;

        Ok(())
    }

    /// Builds the summary row: terminator, gen/kill/entry bit sets.
    fn node_label_final_row<W: io::Write>(&self,
                                          n: &Node,
                                          w: &mut W,
                                          block: BasicBlock,
                                          body: &Body<'_>)
                                          -> io::Result<()> {
        let i = n.index();

        let flow = self.mbcx.flow_state();

        write!(w, "<tr>")?;

        // Entry
        let set = flow.sets.entry_set_for(i);
        write!(w, "<td>{:?}</td>", dot::escape_html(&set.to_string()))?;

        // Terminator
        write!(w, "<td>")?;
        {
            let data = &body[block];
            let mut terminator_head = String::new();
            data.terminator().kind.fmt_head(&mut terminator_head).unwrap();
            write!(w, "{}", dot::escape_html(&terminator_head))?;
        }
        write!(w, "</td>")?;

        // Gen/Kill
        let trans = flow.sets.trans_for(i);
        write!(w, "<td>{:?}</td>", dot::escape_html(&format!("{:?}", trans.gen_set)))?;
        write!(w, "<td>{:?}</td>", dot::escape_html(&format!("{:?}", trans.kill_set)))?;

        write!(w, "</tr>")?;

        Ok(())
    }
}

impl<'a, 'tcx, MWF, P> dot::GraphWalk<'a> for Graph<'a, 'tcx, MWF, P>
    where MWF: MirWithFlowState<'tcx>
{
    type Node = Node;
    type Edge = Edge;
    fn nodes(&self) -> dot::Nodes<'_, Node> {
        self.mbcx.body()
            .basic_blocks()
            .indices()
            .collect::<Vec<_>>()
            .into()
    }

    fn edges(&self) -> dot::Edges<'_, Edge> {
        let body = self.mbcx.body();

        body.basic_blocks()
           .indices()
           .flat_map(|bb| outgoing(body, bb))
           .collect::<Vec<_>>()
           .into()
    }

    fn source(&self, edge: &Edge) -> Node {
        edge.source
    }

    fn target(&self, edge: &Edge) -> Node {
        let body = self.mbcx.body();
        *body[edge.source].terminator().successors().nth(edge.index).unwrap()
    }
}
