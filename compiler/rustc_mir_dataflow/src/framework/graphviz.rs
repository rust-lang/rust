//! A helpful diagram for debugging dataflow problems.

use std::borrow::Cow;
use std::cell::RefCell;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::{io, ops, str};

use regex::Regex;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{
    self, BasicBlock, Body, Location, MirDumper, graphviz_safe_def_name, traversal,
};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_span::def_id::DefId;
use rustc_span::{Symbol, sym};
use tracing::debug;
use {rustc_ast as ast, rustc_graphviz as dot};

use super::fmt::{DebugDiffWithAdapter, DebugWithAdapter, DebugWithContext};
use super::{
    Analysis, CallReturnPlaces, Direction, Results, ResultsCursor, ResultsVisitor, visit_results,
};
use crate::errors::{
    DuplicateValuesFor, PathMustEndInFilename, RequiresAnArgument, UnknownFormatter,
};

/// Writes a DOT file containing the results of a dataflow analysis if the user requested it via
/// `rustc_mir` attributes and `-Z dump-mir-dataflow`. The `Result` in and the `Results` out are
/// the same.
pub(super) fn write_graphviz_results<'tcx, A>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    analysis: &mut A,
    results: &Results<A::Domain>,
    pass_name: Option<&'static str>,
) -> std::io::Result<()>
where
    A: Analysis<'tcx>,
    A::Domain: DebugWithContext<A>,
{
    use std::fs;
    use std::io::Write;

    let def_id = body.source.def_id();
    let Ok(attrs) = RustcMirAttrs::parse(tcx, def_id) else {
        // Invalid `rustc_mir` attrs are reported in `RustcMirAttrs::parse`
        return Ok(());
    };

    let file = try {
        match attrs.output_path(A::NAME) {
            Some(path) => {
                debug!("printing dataflow results for {:?} to {}", def_id, path.display());
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::File::create_buffered(&path)?
            }

            None => {
                let Some(dumper) = MirDumper::new(tcx, A::NAME, body) else {
                    return Ok(());
                };
                let disambiguator = &pass_name.unwrap_or("-----");
                dumper.set_disambiguator(disambiguator).create_dump_file("dot", body)?
            }
        }
    };
    let mut file = match file {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    let style = match attrs.formatter {
        Some(sym::two_phase) => OutputStyle::BeforeAndAfter,
        _ => OutputStyle::AfterOnly,
    };

    let mut buf = Vec::new();

    let graphviz = Formatter::new(body, analysis, results, style);
    let mut render_opts =
        vec![dot::RenderOption::Fontname(tcx.sess.opts.unstable_opts.graphviz_font.clone())];
    if tcx.sess.opts.unstable_opts.graphviz_dark_mode {
        render_opts.push(dot::RenderOption::DarkTheme);
    }
    let r = with_no_trimmed_paths!(dot::render_opts(&graphviz, &mut buf, &render_opts));

    let lhs = try {
        r?;
        file.write_all(&buf)?;
    };

    lhs
}

#[derive(Default)]
struct RustcMirAttrs {
    basename_and_suffix: Option<PathBuf>,
    formatter: Option<Symbol>,
}

impl RustcMirAttrs {
    fn parse(tcx: TyCtxt<'_>, def_id: DefId) -> Result<Self, ()> {
        let mut result = Ok(());
        let mut ret = RustcMirAttrs::default();

        let rustc_mir_attrs = tcx
            .get_attrs(def_id, sym::rustc_mir)
            .flat_map(|attr| attr.meta_item_list().into_iter().flat_map(|v| v.into_iter()));

        for attr in rustc_mir_attrs {
            let attr_result = match attr.name() {
                Some(name @ sym::borrowck_graphviz_postflow) => {
                    Self::set_field(&mut ret.basename_and_suffix, tcx, name, &attr, |s| {
                        let path = PathBuf::from(s.to_string());
                        match path.file_name() {
                            Some(_) => Ok(path),
                            None => {
                                tcx.dcx().emit_err(PathMustEndInFilename { span: attr.span() });
                                Err(())
                            }
                        }
                    })
                }
                Some(name @ sym::borrowck_graphviz_format) => {
                    Self::set_field(&mut ret.formatter, tcx, name, &attr, |s| match s {
                        sym::two_phase => Ok(s),
                        _ => {
                            tcx.dcx().emit_err(UnknownFormatter { span: attr.span() });
                            Err(())
                        }
                    })
                }
                _ => Ok(()),
            };

            result = result.and(attr_result);
        }

        result.map(|()| ret)
    }

    fn set_field<T>(
        field: &mut Option<T>,
        tcx: TyCtxt<'_>,
        name: Symbol,
        attr: &ast::MetaItemInner,
        mapper: impl FnOnce(Symbol) -> Result<T, ()>,
    ) -> Result<(), ()> {
        if field.is_some() {
            tcx.dcx().emit_err(DuplicateValuesFor { span: attr.span(), name });

            return Err(());
        }

        if let Some(s) = attr.value_str() {
            *field = Some(mapper(s)?);
            Ok(())
        } else {
            tcx.dcx()
                .emit_err(RequiresAnArgument { span: attr.span(), name: attr.name().unwrap() });
            Err(())
        }
    }

    /// Returns the path where dataflow results should be written, or `None`
    /// `borrowck_graphviz_postflow` was not specified.
    ///
    /// This performs the following transformation to the argument of `borrowck_graphviz_postflow`:
    ///
    /// "path/suffix.dot" -> "path/analysis_name_suffix.dot"
    fn output_path(&self, analysis_name: &str) -> Option<PathBuf> {
        let mut ret = self.basename_and_suffix.as_ref().cloned()?;
        let suffix = ret.file_name().unwrap(); // Checked when parsing attrs

        let mut file_name: OsString = analysis_name.into();
        file_name.push("_");
        file_name.push(suffix);
        ret.set_file_name(file_name);

        Some(ret)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputStyle {
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

struct Formatter<'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    body: &'mir Body<'tcx>,
    // The `RefCell` is used because `<Formatter as Labeller>::node_label`
    // takes `&self`, but it needs to modify the analysis. This is also the
    // reason for the `Formatter`/`BlockFormatter` split; `BlockFormatter` has
    // the operations that involve the mutation, i.e. within the `borrow_mut`.
    analysis: RefCell<&'mir mut A>,
    results: &'mir Results<A::Domain>,
    style: OutputStyle,
    reachable: DenseBitSet<BasicBlock>,
}

impl<'mir, 'tcx, A> Formatter<'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    fn new(
        body: &'mir Body<'tcx>,
        analysis: &'mir mut A,
        results: &'mir Results<A::Domain>,
        style: OutputStyle,
    ) -> Self {
        let reachable = traversal::reachable_as_bitset(body);
        Formatter { body, analysis: analysis.into(), results, style, reachable }
    }
}

/// A pair of a basic block and an index into that basic blocks `successors`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct CfgEdge {
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

impl<'tcx, A> dot::Labeller<'_> for Formatter<'_, 'tcx, A>
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
        let analysis = &mut **self.analysis.borrow_mut();

        let diffs = StateDiffCollector::run(self.body, *block, analysis, self.results, self.style);

        let mut fmt = BlockFormatter {
            cursor: ResultsCursor::new_borrowing(self.body, analysis, self.results),
            style: self.style,
            bg: Background::Light,
        };
        let label = fmt.write_node_label(*block, diffs).unwrap();

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

impl<'tcx, A> dot::GraphWalk<'_> for Formatter<'_, 'tcx, A>
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

struct BlockFormatter<'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    cursor: ResultsCursor<'mir, 'tcx, A>,
    bg: Background,
    style: OutputStyle,
}

impl<'tcx, A> BlockFormatter<'_, 'tcx, A>
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

    fn write_node_label(
        &mut self,
        block: BasicBlock,
        diffs: StateDiffCollector<A::Domain>,
    ) -> io::Result<Vec<u8>> {
        use std::io::Write;

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

        let mut v = vec![];
        let w = &mut v;

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
        self.cursor.seek_to_block_start(block);
        let block_start_state = self.cursor.get().clone();
        self.write_row_with_full_state(w, "", "(on start)")?;

        // D + E: Statement and terminator transfer functions
        self.write_statements_and_terminator(w, block, diffs)?;

        // F: State at end of block

        let terminator = self.cursor.body()[block].terminator();

        // Write the full dataflow state immediately after the terminator if it differs from the
        // state at block entry.
        self.cursor.seek_to_block_end(block);
        if self.cursor.get() != &block_start_state || A::Direction::IS_BACKWARD {
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
                    let state_on_unwind = this.cursor.get().clone();
                    this.cursor.apply_custom_effect(|analysis, state| {
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
                            this.cursor.get(),
                            &state_on_unwind,
                            this.cursor.analysis()
                        ),
                    )
                })?;
            }

            mir::TerminatorKind::Yield { resume, resume_arg, .. } => {
                self.write_row(w, "", "(on yield resume)", |this, w, fmt| {
                    let state_on_coroutine_drop = this.cursor.get().clone();
                    this.cursor.apply_custom_effect(|analysis, state| {
                        analysis.apply_call_return_effect(
                            state,
                            resume,
                            CallReturnPlaces::Yield(resume_arg),
                        );
                    });

                    write!(
                        w,
                        r#"<td balign="left" colspan="{colspan}" {fmt} align="left">{diff}</td>"#,
                        colspan = this.style.num_state_columns(),
                        fmt = fmt,
                        diff = diff_pretty(
                            this.cursor.get(),
                            &state_on_coroutine_drop,
                            this.cursor.analysis()
                        ),
                    )
                })?;
            }

            mir::TerminatorKind::InlineAsm { ref targets, ref operands, .. }
                if !targets.is_empty() =>
            {
                self.write_row(w, "", "(on successful return)", |this, w, fmt| {
                    let state_on_unwind = this.cursor.get().clone();
                    this.cursor.apply_custom_effect(|analysis, state| {
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
                            this.cursor.get(),
                            &state_on_unwind,
                            this.cursor.analysis()
                        ),
                    )
                })?;
            }

            _ => {}
        };

        write!(w, "</table>")?;

        Ok(v)
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
        diffs: StateDiffCollector<A::Domain>,
    ) -> io::Result<()> {
        let mut diffs_before = diffs.before.map(|v| v.into_iter());
        let mut diffs_after = diffs.after.into_iter();

        let next_in_dataflow_order = |it: &mut std::vec::IntoIter<_>| {
            if A::Direction::IS_FORWARD { it.next().unwrap() } else { it.next_back().unwrap() }
        };

        for (i, statement) in self.cursor.body()[block].statements.iter().enumerate() {
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
        assert!(diffs_before.as_ref().is_none_or(ExactSizeIterator::is_empty));

        let terminator = self.cursor.body()[block].terminator();
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
            let state = this.cursor.get();
            let analysis = this.cursor.analysis();

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
        body: &Body<'tcx>,
        block: BasicBlock,
        analysis: &mut A,
        results: &Results<A::Domain>,
        style: OutputStyle,
    ) -> Self
    where
        A: Analysis<'tcx, Domain = D>,
        D: DebugWithContext<A>,
    {
        let mut collector = StateDiffCollector {
            prev_state: analysis.bottom_value(body),
            after: vec![],
            before: (style == OutputStyle::BeforeAndAfter).then_some(vec![]),
        };

        visit_results(body, std::iter::once(block), analysis, results, &mut collector);
        collector
    }
}

impl<'tcx, A> ResultsVisitor<'tcx, A> for StateDiffCollector<A::Domain>
where
    A: Analysis<'tcx>,
    A::Domain: DebugWithContext<A>,
{
    fn visit_block_start(&mut self, state: &A::Domain) {
        if A::Direction::IS_FORWARD {
            self.prev_state.clone_from(state);
        }
    }

    fn visit_block_end(&mut self, state: &A::Domain) {
        if A::Direction::IS_BACKWARD {
            self.prev_state.clone_from(state);
        }
    }

    fn visit_after_early_statement_effect(
        &mut self,
        analysis: &mut A,
        state: &A::Domain,
        _statement: &mir::Statement<'tcx>,
        _location: Location,
    ) {
        if let Some(before) = self.before.as_mut() {
            before.push(diff_pretty(state, &self.prev_state, analysis));
            self.prev_state.clone_from(state)
        }
    }

    fn visit_after_primary_statement_effect(
        &mut self,
        analysis: &mut A,
        state: &A::Domain,
        _statement: &mir::Statement<'tcx>,
        _location: Location,
    ) {
        self.after.push(diff_pretty(state, &self.prev_state, analysis));
        self.prev_state.clone_from(state)
    }

    fn visit_after_early_terminator_effect(
        &mut self,
        analysis: &mut A,
        state: &A::Domain,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
        if let Some(before) = self.before.as_mut() {
            before.push(diff_pretty(state, &self.prev_state, analysis));
            self.prev_state.clone_from(state)
        }
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        analysis: &mut A,
        state: &A::Domain,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
        self.after.push(diff_pretty(state, &self.prev_state, analysis));
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
    let raw_diff = dot::escape_html(&raw_diff);

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
