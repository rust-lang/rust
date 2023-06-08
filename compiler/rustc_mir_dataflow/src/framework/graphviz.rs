//! A helpful diagram for debugging dataflow problems.

use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::OnceLock;
use std::{io, ops, str};

use regex::Regex;
use rustc_graphviz as dot;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::graphviz_safe_def_name;
use rustc_middle::mir::{self, BasicBlock, Body, Location};

use super::fmt::{DebugDiffWithAdapter, DebugWithAdapter, DebugWithContext};
use super::{Analysis, CallReturnPlaces, Direction, Results, ResultsRefCursor, ResultsVisitor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputStyle {
    AfterOnly,
    BeforeAndAfter,
}

impl OutputStyle {
    fn num_state_columns(&self) -> usize {
        match self {
            Self::AfterOnly => 1,
            Self::BeforeAndAfter => 2,
        }
    }
}

pub struct Formatter<'res, 'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    body: &'mir Body<'tcx>,
    results: RefCell<&'res mut Results<'tcx, A>>,
    style: OutputStyle,
    reachable: BitSet<BasicBlock>,
}

impl<'res, 'mir, 'tcx, A> Formatter<'res, 'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    pub fn new(
        body: &'mir Body<'tcx>,
        results: &'res mut Results<'tcx, A>,
        style: OutputStyle,
    ) -> Self {
        let reachable = mir::traversal::reachable_as_bitset(body);
        Formatter { body, results: results.into(), style, reachable }
    }
}

/// A pair of a basic block and an index into that basic blocks `successors`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CfgEdge {
    source: BasicBlock,
    index: usize,
}

fn dataflow_successors(body: &Body<'_>, bb: BasicBlock) -> Vec<CfgEdge> {
    body[bb]
        .terminator()
        .successors()
        .enumerate()
        .map(|(index, _)| CfgEdge { source: bb, index })
        .collect()
}

impl<'tcx, A> dot::Labeller<'_> for Formatter<'_, '_, 'tcx, A>
where
    A: Analysis<'tcx>,
    A::Domain: DebugWithContext<A>,
{
    type Node = BasicBlock;
    type Edge = CfgEdge;

    fn graph_id(&self) -> dot::Id<'_> {
        let name = graphviz_safe_def_name(self.body.source.def_id());
        dot::Id::new(format!("graph_for_def_id_{name}")).unwrap()
    }

    fn node_id(&self, n: &Self::Node) -> dot::Id<'_> {
        dot::Id::new(format!("bb_{}", n.index())).unwrap()
    }

    fn node_label(&self, block: &Self::Node) -> dot::LabelText<'_> {
        let mut label = Vec::new();
        let mut results = self.results.borrow_mut();
        let mut fmt = BlockFormatter {
            results: results.as_results_cursor(self.body),
            style: self.style,
            bg: Background::Light,
        };

        fmt.write_node_label(&mut label, *block).unwrap();
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

impl<'mir, 'tcx, A> dot::GraphWalk<'mir> for Formatter<'_, 'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    type Node = BasicBlock;
    type Edge = CfgEdge;

    fn nodes(&self) -> dot::Nodes<'_, Self::Node> {
        self.body
            .basic_blocks
            .indices()
            .filter(|&idx| self.reachable.contains(idx))
            .collect::<Vec<_>>()
            .into()
    }

    fn edges(&self) -> dot::Edges<'_, Self::Edge> {
        self.body
            .basic_blocks
            .indices()
            .flat_map(|bb| dataflow_successors(self.body, bb))
            .collect::<Vec<_>>()
            .into()
    }

    fn source(&self, edge: &Self::Edge) -> Self::Node {
        edge.source
    }

    fn target(&self, edge: &Self::Edge) -> Self::Node {
        self.body[edge.source].terminator().successors().nth(edge.index).unwrap()
    }
}

struct BlockFormatter<'res, 'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    results: ResultsRefCursor<'res, 'mir, 'tcx, A>,
    bg: Background,
    style: OutputStyle,
}

impl<'res, 'mir, 'tcx, A> BlockFormatter<'res, 'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
    A::Domain: DebugWithContext<A>,
{
    const HEADER_COLOR: &'static str = "#a0a0a0";

    fn toggle_background(&mut self) -> Background {
        let bg = self.bg;
        self.bg = !bg;
        bg
    }

    fn write_node_label(&mut self, w: &mut impl io::Write, block: BasicBlock) -> io::Result<()> {
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
        write!(w, r#"<table{table_fmt}>"#)?;

        // A + B: Block header
        match self.style {
            OutputStyle::AfterOnly => self.write_block_header_simple(w, block)?,
            OutputStyle::BeforeAndAfter => {
                self.write_block_header_with_state_columns(w, block, &["BEFORE", "AFTER"])?
            }
        }

        // C: State at start of block
        self.bg = Background::Light;
        self.results.seek_to_block_start(block);
        let block_start_state = self.results.get().clone();
        self.write_row_with_full_state(w, "", "(on start)")?;

        // D + E: Statement and terminator transfer functions
        self.write_statements_and_terminator(w, block)?;

        // F: State at end of block

        let terminator = self.results.body()[block].terminator();

        // Write the full dataflow state immediately after the terminator if it differs from the
        // state at block entry.
        self.results.seek_to_block_end(block);
        if self.results.get() != &block_start_state || A::Direction::IS_BACKWARD {
            let after_terminator_name = match terminator.kind {
                mir::TerminatorKind::Call { target: Some(_), .. } => "(on unwind)",
                _ => "(on end)",
            };

            self.write_row_with_full_state(w, "", after_terminator_name)?;
        }

        // Write any changes caused by terminator-specific effects.
        //
        // FIXME: These should really be printed as part of each outgoing edge rather than the node
        // for the basic block itself. That way, we could display terminator-specific effects for
        // backward dataflow analyses as well as effects for `SwitchInt` terminators.
        match terminator.kind {
            mir::TerminatorKind::Call { destination, .. } => {
                self.write_row(w, "", "(on successful return)", |this, w, fmt| {
                    let state_on_unwind = this.results.get().clone();
                    this.results.apply_custom_effect(|analysis, state| {
                        analysis.apply_call_return_effect(
                            state,
                            block,
                            CallReturnPlaces::Call(destination),
                        );
                    });

                    write!(
                        w,
                        r#"<td balign="left" colspan="{colspan}" {fmt} align="left">{diff}</td>"#,
                        colspan = this.style.num_state_columns(),
                        fmt = fmt,
                        diff = diff_pretty(
                            this.results.get(),
                            &state_on_unwind,
                            this.results.analysis()
                        ),
                    )
                })?;
            }

            mir::TerminatorKind::Yield { resume, resume_arg, .. } => {
                self.write_row(w, "", "(on yield resume)", |this, w, fmt| {
                    let state_on_generator_drop = this.results.get().clone();
                    this.results.apply_custom_effect(|analysis, state| {
                        analysis.apply_yield_resume_effect(state, resume, resume_arg);
                    });

                    write!(
                        w,
                        r#"<td balign="left" colspan="{colspan}" {fmt} align="left">{diff}</td>"#,
                        colspan = this.style.num_state_columns(),
                        fmt = fmt,
                        diff = diff_pretty(
                            this.results.get(),
                            &state_on_generator_drop,
                            this.results.analysis()
                        ),
                    )
                })?;
            }

            mir::TerminatorKind::InlineAsm { destination: Some(_), ref operands, .. } => {
                self.write_row(w, "", "(on successful return)", |this, w, fmt| {
                    let state_on_unwind = this.results.get().clone();
                    this.results.apply_custom_effect(|analysis, state| {
                        analysis.apply_call_return_effect(
                            state,
                            block,
                            CallReturnPlaces::InlineAsm(operands),
                        );
                    });

                    write!(
                        w,
                        r#"<td balign="left" colspan="{colspan}" {fmt} align="left">{diff}</td>"#,
                        colspan = this.style.num_state_columns(),
                        fmt = fmt,
                        diff = diff_pretty(
                            this.results.get(),
                            &state_on_unwind,
                            this.results.analysis()
                        ),
                    )
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
        state_column_names: &[&str],
    ) -> io::Result<()> {
        //   +------------------------------------+-------------+
        // A |                bb4                 |    STATE    |
        //   +------------------------------------+------+------+
        // B |                MIR                 |  GEN | KILL |
        //   +-+----------------------------------+------+------+
        //   | |              ...                 |      |      |

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
            write!(w, "<td {fmt}>{name}</td>")?;
        }

        write!(w, "</tr>")
    }

    fn write_statements_and_terminator(
        &mut self,
        w: &mut impl io::Write,
        block: BasicBlock,
    ) -> io::Result<()> {
        let diffs = StateDiffCollector::run(
            self.results.body(),
            block,
            self.results.mut_results(),
            self.style,
        );

        let mut diffs_before = diffs.before.map(|v| v.into_iter());
        let mut diffs_after = diffs.after.into_iter();

        let next_in_dataflow_order = |it: &mut std::vec::IntoIter<_>| {
            if A::Direction::IS_FORWARD { it.next().unwrap() } else { it.next_back().unwrap() }
        };

        for (i, statement) in self.results.body()[block].statements.iter().enumerate() {
            let statement_str = format!("{statement:?}");
            let index_str = format!("{i}");

            let after = next_in_dataflow_order(&mut diffs_after);
            let before = diffs_before.as_mut().map(next_in_dataflow_order);

            self.write_row(w, &index_str, &statement_str, |_this, w, fmt| {
                if let Some(before) = before {
                    write!(w, r#"<td {fmt} align="left">{before}</td>"#)?;
                }

                write!(w, r#"<td {fmt} align="left">{after}</td>"#)
            })?;
        }

        let after = next_in_dataflow_order(&mut diffs_after);
        let before = diffs_before.as_mut().map(next_in_dataflow_order);

        assert!(diffs_after.is_empty());
        assert!(diffs_before.as_ref().map_or(true, ExactSizeIterator::is_empty));

        let terminator = self.results.body()[block].terminator();
        let mut terminator_str = String::new();
        terminator.kind.fmt_head(&mut terminator_str).unwrap();

        self.write_row(w, "T", &terminator_str, |_this, w, fmt| {
            if let Some(before) = before {
                write!(w, r#"<td {fmt} align="left">{before}</td>"#)?;
            }

            write!(w, r#"<td {fmt} align="left">{after}</td>"#)
        })
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

            // FIXME: The full state vector can be quite long. It would be nice to split on commas
            // and use some text wrapping algorithm.
            write!(
                w,
                r#"<td colspan="{colspan}" {fmt} align="left">{state}</td>"#,
                colspan = this.style.num_state_columns(),
                fmt = fmt,
                state = dot::escape_html(&format!(
                    "{:?}",
                    DebugWithAdapter { this: state, ctxt: analysis }
                )),
            )
        })
    }
}

struct StateDiffCollector<D> {
    prev_state: D,
    before: Option<Vec<String>>,
    after: Vec<String>,
}

impl<D> StateDiffCollector<D> {
    fn run<'tcx, A>(
        body: &mir::Body<'tcx>,
        block: BasicBlock,
        results: &mut Results<'tcx, A>,
        style: OutputStyle,
    ) -> Self
    where
        A: Analysis<'tcx, Domain = D>,
        D: DebugWithContext<A>,
    {
        let mut collector = StateDiffCollector {
            prev_state: results.analysis.bottom_value(body),
            after: vec![],
            before: (style == OutputStyle::BeforeAndAfter).then_some(vec![]),
        };

        results.visit_with(body, std::iter::once(block), &mut collector);
        collector
    }
}

impl<'tcx, A> ResultsVisitor<'_, 'tcx, Results<'tcx, A>> for StateDiffCollector<A::Domain>
where
    A: Analysis<'tcx>,
    A::Domain: DebugWithContext<A>,
{
    type FlowState = A::Domain;

    fn visit_block_start(
        &mut self,
        _results: &Results<'tcx, A>,
        state: &Self::FlowState,
        _block_data: &mir::BasicBlockData<'tcx>,
        _block: BasicBlock,
    ) {
        if A::Direction::IS_FORWARD {
            self.prev_state.clone_from(state);
        }
    }

    fn visit_block_end(
        &mut self,
        _results: &Results<'tcx, A>,
        state: &Self::FlowState,
        _block_data: &mir::BasicBlockData<'tcx>,
        _block: BasicBlock,
    ) {
        if A::Direction::IS_BACKWARD {
            self.prev_state.clone_from(state);
        }
    }

    fn visit_statement_before_primary_effect(
        &mut self,
        results: &Results<'tcx, A>,
        state: &Self::FlowState,
        _statement: &mir::Statement<'tcx>,
        _location: Location,
    ) {
        if let Some(before) = self.before.as_mut() {
            before.push(diff_pretty(state, &self.prev_state, &results.analysis));
            self.prev_state.clone_from(state)
        }
    }

    fn visit_statement_after_primary_effect(
        &mut self,
        results: &Results<'tcx, A>,
        state: &Self::FlowState,
        _statement: &mir::Statement<'tcx>,
        _location: Location,
    ) {
        self.after.push(diff_pretty(state, &self.prev_state, &results.analysis));
        self.prev_state.clone_from(state)
    }

    fn visit_terminator_before_primary_effect(
        &mut self,
        results: &Results<'tcx, A>,
        state: &Self::FlowState,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
        if let Some(before) = self.before.as_mut() {
            before.push(diff_pretty(state, &self.prev_state, &results.analysis));
            self.prev_state.clone_from(state)
        }
    }

    fn visit_terminator_after_primary_effect(
        &mut self,
        results: &Results<'tcx, A>,
        state: &Self::FlowState,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
        self.after.push(diff_pretty(state, &self.prev_state, &results.analysis));
        self.prev_state.clone_from(state)
    }
}

macro_rules! regex {
    ($re:literal $(,)?) => {{
        static RE: OnceLock<regex::Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new($re).unwrap())
    }};
}

fn diff_pretty<T, C>(new: T, old: T, ctxt: &C) -> String
where
    T: DebugWithContext<C>,
{
    if new == old {
        return String::new();
    }

    let re = regex!("\t?\u{001f}([+-])");

    let raw_diff = format!("{:#?}", DebugDiffWithAdapter { new, old, ctxt });

    // Replace newlines in the `Debug` output with `<br/>`
    let raw_diff = raw_diff.replace('\n', r#"<br align="left"/>"#);

    let mut inside_font_tag = false;
    let html_diff = re.replace_all(&raw_diff, |captures: &regex::Captures<'_>| {
        let mut ret = String::new();
        if inside_font_tag {
            ret.push_str(r#"</font>"#);
        }

        let tag = match &captures[1] {
            "+" => r#"<font color="darkgreen">+"#,
            "-" => r#"<font color="red">-"#,
            _ => unreachable!(),
        };

        inside_font_tag = true;
        ret.push_str(tag);
        ret
    });

    let Cow::Owned(mut html_diff) = html_diff else {
        return raw_diff;
    };

    if inside_font_tag {
        html_diff.push_str("</font>");
    }

    html_diff
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
