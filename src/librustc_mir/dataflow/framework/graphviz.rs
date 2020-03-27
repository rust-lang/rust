//! A helpful diagram for debugging dataflow problems.

use std::cell::RefCell;
use std::{io, ops, str};

use rustc::mir::{self, BasicBlock, Body, Location};
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::{BitSet, HybridBitSet};
use rustc_index::vec::{Idx, IndexVec};

use super::{Analysis, GenKillSet, Results, ResultsRefCursor};
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
    pub fn new(
        body: &'a Body<'tcx>,
        def_id: DefId,
        results: &'a Results<'tcx, A>,
        state_formatter: &'a mut dyn StateFormatter<'tcx, A>,
    ) -> Self {
        let block_formatter = BlockFormatter {
            bg: Background::Light,
            results: ResultsRefCursor::new(body, results),
            state_formatter,
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
    results: ResultsRefCursor<'a, 'a, 'tcx, A>,
    bg: Background,
    state_formatter: &'a mut dyn StateFormatter<'tcx, A>,
}

impl<A> BlockFormatter<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    const HEADER_COLOR: &'static str = "#a0a0a0";

    fn num_state_columns(&self) -> usize {
        std::cmp::max(1, self.state_formatter.column_names().len())
    }

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

        // N.B., Some attributes (`align`, `balign`) are repeated on parent elements and their
        // children. This is because `xdot` seemed to have a hard time correctly propagating
        // attributes. Make sure to test the output before trying to remove the redundancy.
        // Notably, `align` was found to have no effect when applied only to <table>.

        let table_fmt = concat!(
            " border=\"1\"",
            " cellborder=\"1\"",
            " cellspacing=\"0\"",
            " cellpadding=\"3\"",
            " sides=\"rb\"",
        );
        write!(w, r#"<table{fmt}>"#, fmt = table_fmt)?;

        // A + B: Block header
        if self.state_formatter.column_names().is_empty() {
            self.write_block_header_simple(w, block)?;
        } else {
            self.write_block_header_with_state_columns(w, block)?;
        }

        // C: Entry state
        self.bg = Background::Light;
        self.results.seek_to_block_start(block);
        let block_entry_state = self.results.get().clone();

        self.write_row_with_full_state(w, "", "(on entry)")?;

        // D: Statement transfer functions
        for (i, statement) in body[block].statements.iter().enumerate() {
            let location = Location { block, statement_index: i };
            let statement_str = format!("{:?}", statement);
            self.write_row_for_location(w, &i.to_string(), &statement_str, location)?;
        }

        // E: Terminator transfer function
        let terminator = body[block].terminator();
        let terminator_loc = body.terminator_loc(block);
        let mut terminator_str = String::new();
        terminator.kind.fmt_head(&mut terminator_str).unwrap();

        self.write_row_for_location(w, "T", &terminator_str, terminator_loc)?;

        // F: Exit state

        // Write the full dataflow state immediately after the terminator if it differs from the
        // state at block entry.
        self.results.seek_after(terminator_loc);
        if self.results.get() != &block_entry_state {
            let after_terminator_name = match terminator.kind {
                mir::TerminatorKind::Call { destination: Some(_), .. } => "(on unwind)",
                _ => "(on exit)",
            };

            self.write_row_with_full_state(w, "", after_terminator_name)?;
        }

        // Write any changes caused by terminator-specific effects
        match terminator.kind {
            mir::TerminatorKind::Call { destination: Some(_), .. } => {
                let num_state_columns = self.num_state_columns();
                self.write_row(w, "", "(on successful return)", |this, w, fmt| {
                    write!(
                        w,
                        r#"<td balign="left" colspan="{colspan}" {fmt} align="left">"#,
                        colspan = num_state_columns,
                        fmt = fmt,
                    )?;

                    let state_on_unwind = this.results.get().clone();
                    this.results.seek_after_assume_success(terminator_loc);
                    write_diff(w, this.results.analysis(), &state_on_unwind, this.results.get())?;

                    write!(w, "</td>")
                })?;
            }

            _ => {}
        };

        write!(w, "</table>")
    }

    fn write_block_header_simple(
        &mut self,
        w: &mut impl io::Write,
        block: BasicBlock,
    ) -> io::Result<()> {
        //   +-------------------------------------------------+
        // A |                      bb4                        |
        //   +-----------------------------------+-------------+
        // B |                MIR                |    STATE    |
        //   +-+---------------------------------+-------------+
        //   | |              ...                |             |

        // A
        write!(
            w,
            concat!("<tr>", r#"<td colspan="3" sides="tl">bb{block_id}</td>"#, "</tr>",),
            block_id = block.index(),
        )?;

        // B
        write!(
            w,
            concat!(
                "<tr>",
                r#"<td colspan="2" {fmt}>MIR</td>"#,
                r#"<td {fmt}>STATE</td>"#,
                "</tr>",
            ),
            fmt = format!("bgcolor=\"{}\" sides=\"tl\"", Self::HEADER_COLOR),
        )
    }

    fn write_block_header_with_state_columns(
        &mut self,
        w: &mut impl io::Write,
        block: BasicBlock,
    ) -> io::Result<()> {
        //   +------------------------------------+-------------+
        // A |                bb4                 |    STATE    |
        //   +------------------------------------+------+------+
        // B |                MIR                 |  GEN | KILL |
        //   +-+----------------------------------+------+------+
        //   | |              ...                 |      |      |

        let state_column_names = self.state_formatter.column_names();

        // A
        write!(
            w,
            concat!(
                "<tr>",
                r#"<td {fmt} colspan="2">bb{block_id}</td>"#,
                r#"<td {fmt} colspan="{num_state_cols}">STATE</td>"#,
                "</tr>",
            ),
            fmt = "sides=\"tl\"",
            num_state_cols = state_column_names.len(),
            block_id = block.index(),
        )?;

        // B
        let fmt = format!("bgcolor=\"{}\" sides=\"tl\"", Self::HEADER_COLOR);
        write!(w, concat!("<tr>", r#"<td colspan="2" {fmt}>MIR</td>"#,), fmt = fmt,)?;

        for name in state_column_names {
            write!(w, "<td {fmt}>{name}</td>", fmt = fmt, name = name)?;
        }

        write!(w, "</tr>")
    }

    /// Write a row with the given index and MIR, using the function argument to fill in the
    /// "STATE" column(s).
    fn write_row<W: io::Write>(
        &mut self,
        w: &mut W,
        i: &str,
        mir: &str,
        f: impl FnOnce(&mut Self, &mut W, &str) -> io::Result<()>,
    ) -> io::Result<()> {
        let bg = self.toggle_background();
        let valign = if mir.starts_with("(on ") && mir != "(on entry)" { "bottom" } else { "top" };

        let fmt = format!("valign=\"{}\" sides=\"tl\" {}", valign, bg.attr());

        write!(
            w,
            concat!(
                "<tr>",
                r#"<td {fmt} align="right">{i}</td>"#,
                r#"<td {fmt} align="left">{mir}</td>"#,
            ),
            i = i,
            fmt = fmt,
            mir = dot::escape_html(mir),
        )?;

        f(self, w, &fmt)?;
        write!(w, "</tr>")
    }

    fn write_row_with_full_state(
        &mut self,
        w: &mut impl io::Write,
        i: &str,
        mir: &str,
    ) -> io::Result<()> {
        self.write_row(w, i, mir, |this, w, fmt| {
            let state = this.results.get();
            let analysis = this.results.analysis();

            write!(
                w,
                r#"<td colspan="{colspan}" {fmt} align="left">{{"#,
                colspan = this.num_state_columns(),
                fmt = fmt,
            )?;
            pretty_print_state_elems(w, analysis, state.iter(), ", ", LIMIT_30_ALIGN_1)?;
            write!(w, "}}</td>")
        })
    }

    fn write_row_for_location(
        &mut self,
        w: &mut impl io::Write,
        i: &str,
        mir: &str,
        location: Location,
    ) -> io::Result<()> {
        self.write_row(w, i, mir, |this, w, fmt| {
            this.state_formatter.write_state_for_location(w, fmt, &mut this.results, location)
        })
    }
}

/// Controls what gets printed under the `STATE` header.
pub trait StateFormatter<'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// The columns that will get printed under `STATE`.
    fn column_names(&self) -> &[&str];

    fn write_state_for_location(
        &mut self,
        w: &mut dyn io::Write,
        fmt: &str,
        results: &mut ResultsRefCursor<'_, '_, 'tcx, A>,
        location: Location,
    ) -> io::Result<()>;
}

/// Prints a single column containing the state vector immediately *after* each statement.
pub struct SimpleDiff<T: Idx> {
    prev_state: BitSet<T>,
    prev_loc: Location,
}

impl<T: Idx> SimpleDiff<T> {
    pub fn new(bits_per_block: usize) -> Self {
        SimpleDiff { prev_state: BitSet::new_empty(bits_per_block), prev_loc: Location::START }
    }
}

impl<A> StateFormatter<'tcx, A> for SimpleDiff<A::Idx>
where
    A: Analysis<'tcx>,
{
    fn column_names(&self) -> &[&str] {
        &[]
    }

    fn write_state_for_location(
        &mut self,
        mut w: &mut dyn io::Write,
        fmt: &str,
        results: &mut ResultsRefCursor<'_, '_, 'tcx, A>,
        location: Location,
    ) -> io::Result<()> {
        if location.statement_index == 0 {
            results.seek_to_block_start(location.block);
            self.prev_state.overwrite(results.get());
        } else {
            // Ensure that we are visiting statements in order, so `prev_state` is correct.
            assert_eq!(self.prev_loc.successor_within_block(), location);
        }

        self.prev_loc = location;
        write!(w, r#"<td {fmt} balign="left" align="left">"#, fmt = fmt)?;
        results.seek_after(location);
        let curr_state = results.get();
        write_diff(&mut w, results.analysis(), &self.prev_state, curr_state)?;
        self.prev_state.overwrite(curr_state);
        write!(w, "</td>")
    }
}

/// Prints two state columns, one containing only the "before" effect of each statement and one
/// containing the full effect.
pub struct TwoPhaseDiff<T: Idx> {
    prev_state: BitSet<T>,
    prev_loc: Location,
}

impl<T: Idx> TwoPhaseDiff<T> {
    pub fn new(bits_per_block: usize) -> Self {
        TwoPhaseDiff { prev_state: BitSet::new_empty(bits_per_block), prev_loc: Location::START }
    }
}

impl<A> StateFormatter<'tcx, A> for TwoPhaseDiff<A::Idx>
where
    A: Analysis<'tcx>,
{
    fn column_names(&self) -> &[&str] {
        &["BEFORE", " AFTER"]
    }

    fn write_state_for_location(
        &mut self,
        mut w: &mut dyn io::Write,
        fmt: &str,
        results: &mut ResultsRefCursor<'_, '_, 'tcx, A>,
        location: Location,
    ) -> io::Result<()> {
        if location.statement_index == 0 {
            results.seek_to_block_start(location.block);
            self.prev_state.overwrite(results.get());
        } else {
            // Ensure that we are visiting statements in order, so `prev_state` is correct.
            assert_eq!(self.prev_loc.successor_within_block(), location);
        }

        self.prev_loc = location;

        // Before

        write!(w, r#"<td {fmt} align="left">"#, fmt = fmt)?;
        results.seek_before(location);
        let curr_state = results.get();
        write_diff(&mut w, results.analysis(), &self.prev_state, curr_state)?;
        self.prev_state.overwrite(curr_state);
        write!(w, "</td>")?;

        // After

        write!(w, r#"<td {fmt} align="left">"#, fmt = fmt)?;
        results.seek_after(location);
        let curr_state = results.get();
        write_diff(&mut w, results.analysis(), &self.prev_state, curr_state)?;
        self.prev_state.overwrite(curr_state);
        write!(w, "</td>")
    }
}

/// Prints the gen/kill set for the entire block.
pub struct BlockTransferFunc<'a, 'tcx, T: Idx> {
    body: &'a mir::Body<'tcx>,
    trans_for_block: IndexVec<BasicBlock, GenKillSet<T>>,
}

impl<T: Idx> BlockTransferFunc<'mir, 'tcx, T> {
    pub fn new(
        body: &'mir mir::Body<'tcx>,
        trans_for_block: IndexVec<BasicBlock, GenKillSet<T>>,
    ) -> Self {
        BlockTransferFunc { body, trans_for_block }
    }
}

impl<A> StateFormatter<'tcx, A> for BlockTransferFunc<'mir, 'tcx, A::Idx>
where
    A: Analysis<'tcx>,
{
    fn column_names(&self) -> &[&str] {
        &["GEN", "KILL"]
    }

    fn write_state_for_location(
        &mut self,
        mut w: &mut dyn io::Write,
        fmt: &str,
        results: &mut ResultsRefCursor<'_, '_, 'tcx, A>,
        location: Location,
    ) -> io::Result<()> {
        // Only print a single row.
        if location.statement_index != 0 {
            return Ok(());
        }

        let block_trans = &self.trans_for_block[location.block];
        let rowspan = self.body.basic_blocks()[location.block].statements.len();

        for set in &[&block_trans.gen, &block_trans.kill] {
            write!(
                w,
                r#"<td {fmt} rowspan="{rowspan}" balign="left" align="left">"#,
                fmt = fmt,
                rowspan = rowspan
            )?;

            pretty_print_state_elems(&mut w, results.analysis(), set.iter(), BR_LEFT, None)?;
            write!(w, "</td>")?;
        }

        Ok(())
    }
}

/// Writes two lines, one containing the added bits and one the removed bits.
fn write_diff<A: Analysis<'tcx>>(
    w: &mut impl io::Write,
    analysis: &A,
    from: &BitSet<A::Idx>,
    to: &BitSet<A::Idx>,
) -> io::Result<()> {
    assert_eq!(from.domain_size(), to.domain_size());
    let len = from.domain_size();

    let mut set = HybridBitSet::new_empty(len);
    let mut clear = HybridBitSet::new_empty(len);

    // FIXME: Implement a lazy iterator over the symmetric difference of two bitsets.
    for i in (0..len).map(A::Idx::new) {
        match (from.contains(i), to.contains(i)) {
            (false, true) => set.insert(i),
            (true, false) => clear.insert(i),
            _ => continue,
        };
    }

    if !set.is_empty() {
        write!(w, r#"<font color="darkgreen">+"#)?;
        pretty_print_state_elems(w, analysis, set.iter(), ", ", LIMIT_30_ALIGN_1)?;
        write!(w, r#"</font>"#)?;
    }

    if !set.is_empty() && !clear.is_empty() {
        write!(w, "{}", BR_LEFT)?;
    }

    if !clear.is_empty() {
        write!(w, r#"<font color="red">-"#)?;
        pretty_print_state_elems(w, analysis, clear.iter(), ", ", LIMIT_30_ALIGN_1)?;
        write!(w, r#"</font>"#)?;
    }

    Ok(())
}

const BR_LEFT: &str = r#"<br align="left"/>"#;
const BR_LEFT_SPACE: &str = r#"<br align="left"/> "#;

/// Line break policy that breaks at 40 characters and starts the next line with a single space.
const LIMIT_30_ALIGN_1: Option<LineBreak> = Some(LineBreak { sequence: BR_LEFT_SPACE, limit: 30 });

struct LineBreak {
    sequence: &'static str,
    limit: usize,
}

/// Formats each `elem` using the pretty printer provided by `analysis` into a list with the given
/// separator (`sep`).
///
/// Optionally, it will break lines using the given character sequence (usually `<br/>`) and
/// character limit.
fn pretty_print_state_elems<A>(
    w: &mut impl io::Write,
    analysis: &A,
    elems: impl Iterator<Item = A::Idx>,
    sep: &str,
    line_break: Option<LineBreak>,
) -> io::Result<bool>
where
    A: Analysis<'tcx>,
{
    let sep_width = sep.chars().count();

    let mut buf = Vec::new();

    let mut first = true;
    let mut curr_line_width = 0;
    let mut line_break_inserted = false;

    for idx in elems {
        buf.clear();
        analysis.pretty_print_idx(&mut buf, idx)?;
        let idx_str =
            str::from_utf8(&buf).expect("Output of `pretty_print_idx` must be valid UTF-8");
        let escaped = dot::escape_html(idx_str);
        let escaped_width = escaped.chars().count();

        if first {
            first = false;
        } else {
            write!(w, "{}", sep)?;
            curr_line_width += sep_width;

            if let Some(line_break) = &line_break {
                if curr_line_width + sep_width + escaped_width > line_break.limit {
                    write!(w, "{}", line_break.sequence)?;
                    line_break_inserted = true;
                    curr_line_width = 0;
                }
            }
        }

        write!(w, "{}", escaped)?;
        curr_line_width += escaped_width;
    }

    Ok(line_break_inserted)
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
