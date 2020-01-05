use std::cell::RefCell;
use std::io::{self, Write};
use std::{ops, str};

use rustc::mir::{self, BasicBlock, Body, Location};
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::{BitSet, HybridBitSet};
use rustc_index::vec::Idx;

use super::{Analysis, Results, ResultsRefCursor};
use crate::util::graphviz_safe_def_name;

pub struct Formatter<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    body: &'a Body<'tcx>,
    def_id: DefId,

    // This must be behind a `RefCell` because `dot::Labeller` takes `&self`.
    block_formatter: RefCell<BlockFormatter<'a, 'tcx, A>>,
}

impl<A> Formatter<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    pub fn new(body: &'a Body<'tcx>, def_id: DefId, results: &'a Results<'tcx, A>) -> Self {
        let block_formatter = BlockFormatter {
            bg: Background::Light,
            prev_state: BitSet::new_empty(results.analysis.bits_per_block(body)),
            results: ResultsRefCursor::new(body, results),
        };

        Formatter { body, def_id, block_formatter: RefCell::new(block_formatter) }
    }
}

/// A pair of a basic block and an index into that basic blocks `successors`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CfgEdge {
    source: BasicBlock,
    index: usize,
}

fn outgoing_edges(body: &Body<'_>, bb: BasicBlock) -> Vec<CfgEdge> {
    body[bb]
        .terminator()
        .successors()
        .enumerate()
        .map(|(index, _)| CfgEdge { source: bb, index })
        .collect()
}

impl<A> dot::Labeller<'_> for Formatter<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    type Node = BasicBlock;
    type Edge = CfgEdge;

    fn graph_id(&self) -> dot::Id<'_> {
        let name = graphviz_safe_def_name(self.def_id);
        dot::Id::new(format!("graph_for_def_id_{}", name)).unwrap()
    }

    fn node_id(&self, n: &Self::Node) -> dot::Id<'_> {
        dot::Id::new(format!("bb_{}", n.index())).unwrap()
    }

    fn node_label(&self, block: &Self::Node) -> dot::LabelText<'_> {
        let mut label = Vec::new();
        self.block_formatter.borrow_mut().write_node_label(&mut label, self.body, *block).unwrap();
        dot::LabelText::html(String::from_utf8(label).unwrap())
    }

    fn node_shape(&self, _n: &Self::Node) -> Option<dot::LabelText<'_>> {
        Some(dot::LabelText::label("none"))
    }

    fn edge_label(&self, e: &Self::Edge) -> dot::LabelText<'_> {
        let label = &self.body[e.source].terminator().kind.fmt_successor_labels()[e.index];
        dot::LabelText::label(label.clone())
    }
}

impl<A> dot::GraphWalk<'a> for Formatter<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    type Node = BasicBlock;
    type Edge = CfgEdge;

    fn nodes(&self) -> dot::Nodes<'_, Self::Node> {
        self.body.basic_blocks().indices().collect::<Vec<_>>().into()
    }

    fn edges(&self) -> dot::Edges<'_, Self::Edge> {
        self.body
            .basic_blocks()
            .indices()
            .flat_map(|bb| outgoing_edges(self.body, bb))
            .collect::<Vec<_>>()
            .into()
    }

    fn source(&self, edge: &Self::Edge) -> Self::Node {
        edge.source
    }

    fn target(&self, edge: &Self::Edge) -> Self::Node {
        self.body[edge.source].terminator().successors().nth(edge.index).copied().unwrap()
    }
}

struct BlockFormatter<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    prev_state: BitSet<A::Idx>,
    results: ResultsRefCursor<'a, 'a, 'tcx, A>,
    bg: Background,
}

impl<A> BlockFormatter<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    fn toggle_background(&mut self) -> Background {
        let bg = self.bg;
        self.bg = !bg;
        bg
    }

    fn write_node_label(
        &mut self,
        w: &mut impl io::Write,
        body: &'a Body<'tcx>,
        block: BasicBlock,
    ) -> io::Result<()> {
        //   Sample output:
        //   +-+-----------------------------------------------+
        // A |                      bb4                        |
        //   +-+----------------------------------+------------+
        // B |                MIR                 |   STATE    |
        //   +-+----------------------------------+------------+
        // C | | (on entry)                       | {_0,_2,_3} |
        //   +-+----------------------------------+------------+
        // D |0| StorageLive(_7)                  |            |
        //   +-+----------------------------------+------------+
        //   |1| StorageLive(_8)                  |            |
        //   +-+----------------------------------+------------+
        //   |2| _8 = &mut _1                     | +_8        |
        //   +-+----------------------------------+------------+
        // E |T| _4 = const Foo::twiddle(move _2) | -_2        |
        //   +-+----------------------------------+------------+
        // F | | (on unwind)                      | {_0,_3,_8} |
        //   +-+----------------------------------+------------+
        //   | | (on successful return)           | +_4        |
        //   +-+----------------------------------+------------+

        write!(
            w,
            r#"<table border="1" cellborder="1" cellspacing="0" cellpadding="3" sides="rb">"#,
        )?;

        // A: Block info
        write!(
            w,
            r#"<tr>
                 <td colspan="{num_headers}" sides="tl">bb{block_id}</td>
               </tr>"#,
            num_headers = 3,
            block_id = block.index(),
        )?;

        // B: Column headings
        write!(
            w,
            r#"<tr>
                 <td colspan="2" {fmt}>MIR</td>
                 <td {fmt}>STATE</td>
               </tr>"#,
            fmt = r##"bgcolor="#a0a0a0" sides="tl""##,
        )?;

        // C: Entry state
        self.bg = Background::Light;
        self.results.seek_to_block_start(block);
        self.write_row_with_curr_state(w, "", "(on entry)")?;
        self.prev_state.overwrite(self.results.get());

        // D: Statement transfer functions
        for (i, statement) in body[block].statements.iter().enumerate() {
            let location = Location { block, statement_index: i };

            let mir_col = format!("{:?}", statement);
            let i_col = i.to_string();

            self.results.seek_after(location);
            self.write_row_with_curr_diff(w, &i_col, &mir_col)?;
            self.prev_state.overwrite(self.results.get());
        }

        // E: Terminator transfer function
        let terminator = body[block].terminator();
        let location = body.terminator_loc(block);

        let mut mir_col = String::new();
        terminator.kind.fmt_head(&mut mir_col).unwrap();

        self.results.seek_after(location);
        self.write_row_with_curr_diff(w, "T", &mir_col)?;
        self.prev_state.overwrite(self.results.get());

        // F: Exit state
        if let mir::TerminatorKind::Call { destination: Some(_), .. } = &terminator.kind {
            self.write_row_with_curr_state(w, "", "(on unwind)")?;

            self.results.seek_after_assume_call_returns(location);
            self.write_row_with_curr_diff(w, "", "(on successful return)")?;
        } else {
            self.write_row_with_curr_state(w, "", "(on exit)")?;
        }

        write!(w, "</table>")
    }

    fn write_row_with_curr_state(
        &mut self,
        w: &mut impl io::Write,
        i: &str,
        mir: &str,
    ) -> io::Result<()> {
        let bg = self.toggle_background();

        let mut out = Vec::new();
        write!(&mut out, "{{")?;
        pretty_print_state_elems(&mut out, self.results.analysis(), self.results.get().iter())?;
        write!(&mut out, "}}")?;

        write!(
            w,
            r#"<tr>
                 <td {fmt} align="right">{i}</td>
                 <td {fmt} align="left">{mir}</td>
                 <td {fmt} align="left">{state}</td>
               </tr>"#,
            fmt = &["sides=\"tl\"", bg.attr()].join(" "),
            i = i,
            mir = dot::escape_html(mir),
            state = dot::escape_html(str::from_utf8(&out).unwrap()),
        )
    }

    fn write_row_with_curr_diff(
        &mut self,
        w: &mut impl io::Write,
        i: &str,
        mir: &str,
    ) -> io::Result<()> {
        let bg = self.toggle_background();
        let analysis = self.results.analysis();

        let diff = BitSetDiff::compute(&self.prev_state, self.results.get());

        let mut set = Vec::new();
        pretty_print_state_elems(&mut set, analysis, diff.set.iter())?;

        let mut clear = Vec::new();
        pretty_print_state_elems(&mut clear, analysis, diff.clear.iter())?;

        write!(
            w,
            r#"<tr>
                 <td {fmt} align="right">{i}</td>
                 <td {fmt} align="left">{mir}</td>
                 <td {fmt} align="left">"#,
            i = i,
            fmt = &["sides=\"tl\"", bg.attr()].join(" "),
            mir = dot::escape_html(mir),
        )?;

        if !set.is_empty() {
            write!(
                w,
                r#"<font color="darkgreen">+{}</font>"#,
                dot::escape_html(str::from_utf8(&set).unwrap()),
            )?;
        }

        if !set.is_empty() && !clear.is_empty() {
            write!(w, "  ")?;
        }

        if !clear.is_empty() {
            write!(
                w,
                r#"<font color="red">-{}</font>"#,
                dot::escape_html(str::from_utf8(&clear).unwrap()),
            )?;
        }

        write!(w, "</td></tr>")
    }
}

/// The operations required to transform one `BitSet` into another.
struct BitSetDiff<T: Idx> {
    set: HybridBitSet<T>,
    clear: HybridBitSet<T>,
}

impl<T: Idx> BitSetDiff<T> {
    fn compute(from: &BitSet<T>, to: &BitSet<T>) -> Self {
        assert_eq!(from.domain_size(), to.domain_size());
        let len = from.domain_size();

        let mut set = HybridBitSet::new_empty(len);
        let mut clear = HybridBitSet::new_empty(len);

        // FIXME: This could be made faster if `BitSet::xor` were implemented.
        for i in (0..len).map(|i| T::new(i)) {
            match (from.contains(i), to.contains(i)) {
                (false, true) => set.insert(i),
                (true, false) => clear.insert(i),
                _ => continue,
            };
        }

        BitSetDiff { set, clear }
    }
}

/// Formats each `elem` using the pretty printer provided by `analysis` into a comma-separated
/// list.
fn pretty_print_state_elems<A>(
    w: &mut impl io::Write,
    analysis: &A,
    elems: impl Iterator<Item = A::Idx>,
) -> io::Result<()>
where
    A: Analysis<'tcx>,
{
    let mut first = true;
    for idx in elems {
        if first {
            first = false;
        } else {
            write!(w, ",")?;
        }

        analysis.pretty_print_idx(w, idx)?;
    }

    Ok(())
}

/// The background color used for zebra-striping the table.
#[derive(Clone, Copy)]
enum Background {
    Light,
    Dark,
}

impl Background {
    fn attr(self) -> &'static str {
        match self {
            Self::Dark => "bgcolor=\"#f0f0f0\"",
            Self::Light => "",
        }
    }
}

impl ops::Not for Background {
    type Output = Self;

    fn not(self) -> Self {
        match self {
            Self::Light => Self::Dark,
            Self::Dark => Self::Light,
        }
    }
}
