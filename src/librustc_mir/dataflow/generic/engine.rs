//! A solver for dataflow problems.

use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;

use rustc::mir::{self, traversal, BasicBlock, Location};
use rustc::ty::{self, TyCtxt};
use rustc_data_structures::work_queue::WorkQueue;
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_span::symbol::{sym, Symbol};
use syntax::ast;

use super::graphviz;
use super::{Analysis, GenKillAnalysis, GenKillSet, Results};

/// A solver for dataflow problems.
pub struct Engine<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    bits_per_block: usize,
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    def_id: DefId,
    dead_unwinds: Option<&'a BitSet<BasicBlock>>,
    entry_sets: IndexVec<BasicBlock, BitSet<A::Idx>>,
    analysis: A,

    /// Cached, cumulative transfer functions for each block.
    trans_for_block: Option<IndexVec<BasicBlock, GenKillSet<A::Idx>>>,
}

impl<A> Engine<'a, 'tcx, A>
where
    A: GenKillAnalysis<'tcx>,
{
    /// Creates a new `Engine` to solve a gen-kill dataflow problem.
    pub fn new_gen_kill(
        tcx: TyCtxt<'tcx>,
        body: &'a mir::Body<'tcx>,
        def_id: DefId,
        analysis: A,
    ) -> Self {
        // If there are no back-edges in the control-flow graph, we only ever need to apply the
        // transfer function for each block exactly once (assuming that we process blocks in RPO).
        //
        // In this case, there's no need to compute the block transfer functions ahead of time.
        if !body.is_cfg_cyclic() {
            return Self::new(tcx, body, def_id, analysis, None);
        }

        // Otherwise, compute and store the cumulative transfer function for each block.

        let bits_per_block = analysis.bits_per_block(body);
        let mut trans_for_block =
            IndexVec::from_elem(GenKillSet::identity(bits_per_block), body.basic_blocks());

        for (block, block_data) in body.basic_blocks().iter_enumerated() {
            let trans = &mut trans_for_block[block];

            for (i, statement) in block_data.statements.iter().enumerate() {
                let loc = Location { block, statement_index: i };
                analysis.before_statement_effect(trans, statement, loc);
                analysis.statement_effect(trans, statement, loc);
            }

            let terminator = block_data.terminator();
            let loc = Location { block, statement_index: block_data.statements.len() };
            analysis.before_terminator_effect(trans, terminator, loc);
            analysis.terminator_effect(trans, terminator, loc);
        }

        Self::new(tcx, body, def_id, analysis, Some(trans_for_block))
    }
}

impl<A> Engine<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// Creates a new `Engine` to solve a dataflow problem with an arbitrary transfer
    /// function.
    ///
    /// Gen-kill problems should use `new_gen_kill`, which will coalesce transfer functions for
    /// better performance.
    pub fn new_generic(
        tcx: TyCtxt<'tcx>,
        body: &'a mir::Body<'tcx>,
        def_id: DefId,
        analysis: A,
    ) -> Self {
        Self::new(tcx, body, def_id, analysis, None)
    }

    fn new(
        tcx: TyCtxt<'tcx>,
        body: &'a mir::Body<'tcx>,
        def_id: DefId,
        analysis: A,
        trans_for_block: Option<IndexVec<BasicBlock, GenKillSet<A::Idx>>>,
    ) -> Self {
        let bits_per_block = analysis.bits_per_block(body);

        let bottom_value_set = if A::BOTTOM_VALUE {
            BitSet::new_filled(bits_per_block)
        } else {
            BitSet::new_empty(bits_per_block)
        };

        let mut entry_sets = IndexVec::from_elem(bottom_value_set, body.basic_blocks());
        analysis.initialize_start_block(body, &mut entry_sets[mir::START_BLOCK]);

        Engine {
            analysis,
            bits_per_block,
            tcx,
            body,
            def_id,
            dead_unwinds: None,
            entry_sets,
            trans_for_block,
        }
    }

    /// Signals that we do not want dataflow state to propagate across unwind edges for these
    /// `BasicBlock`s.
    ///
    /// You must take care that `dead_unwinds` does not contain a `BasicBlock` that *can* actually
    /// unwind during execution. Otherwise, your dataflow results will not be correct.
    pub fn dead_unwinds(mut self, dead_unwinds: &'a BitSet<BasicBlock>) -> Self {
        self.dead_unwinds = Some(dead_unwinds);
        self
    }

    /// Computes the fixpoint for this dataflow problem and returns it.
    pub fn iterate_to_fixpoint(mut self) -> Results<'tcx, A> {
        let mut temp_state = BitSet::new_empty(self.bits_per_block);

        let mut dirty_queue: WorkQueue<BasicBlock> =
            WorkQueue::with_none(self.body.basic_blocks().len());

        for (bb, _) in traversal::reverse_postorder(self.body) {
            dirty_queue.insert(bb);
        }

        // Add blocks that are not reachable from START_BLOCK to the work queue. These blocks will
        // be processed after the ones added above.
        for bb in self.body.basic_blocks().indices() {
            dirty_queue.insert(bb);
        }

        while let Some(bb) = dirty_queue.pop() {
            let bb_data = &self.body[bb];
            let on_entry = &self.entry_sets[bb];

            temp_state.overwrite(on_entry);
            self.apply_whole_block_effect(&mut temp_state, bb, bb_data);

            self.propagate_bits_into_graph_successors_of(
                &mut temp_state,
                (bb, bb_data),
                &mut dirty_queue,
            );
        }

        let Engine { tcx, body, def_id, trans_for_block, entry_sets, analysis, .. } = self;
        let results = Results { analysis, entry_sets };

        let res = write_graphviz_results(tcx, def_id, body, &results, trans_for_block);
        if let Err(e) = res {
            warn!("Failed to write graphviz dataflow results: {}", e);
        }

        results
    }

    /// Applies the cumulative effect of an entire block, excluding the call return effect if one
    /// exists.
    fn apply_whole_block_effect(
        &self,
        state: &mut BitSet<A::Idx>,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) {
        // Use the cached block transfer function if available.
        if let Some(trans_for_block) = &self.trans_for_block {
            trans_for_block[block].apply(state);
            return;
        }

        // Otherwise apply effects one-by-one.

        for (statement_index, statement) in block_data.statements.iter().enumerate() {
            let location = Location { block, statement_index };
            self.analysis.apply_before_statement_effect(state, statement, location);
            self.analysis.apply_statement_effect(state, statement, location);
        }

        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        self.analysis.apply_before_terminator_effect(state, terminator, location);
        self.analysis.apply_terminator_effect(state, terminator, location);
    }

    fn propagate_bits_into_graph_successors_of(
        &mut self,
        in_out: &mut BitSet<A::Idx>,
        (bb, bb_data): (BasicBlock, &'a mir::BasicBlockData<'tcx>),
        dirty_list: &mut WorkQueue<BasicBlock>,
    ) {
        use mir::TerminatorKind::*;

        match bb_data.terminator().kind {
            Return | Resume | Abort | GeneratorDrop | Unreachable => {}

            Goto { target }
            | Assert { target, cleanup: None, .. }
            | Yield { resume: target, drop: None, .. }
            | Drop { target, location: _, unwind: None }
            | DropAndReplace { target, value: _, location: _, unwind: None } => {
                self.propagate_bits_into_entry_set_for(in_out, target, dirty_list)
            }

            Yield { resume: target, drop: Some(drop), .. } => {
                self.propagate_bits_into_entry_set_for(in_out, target, dirty_list);
                self.propagate_bits_into_entry_set_for(in_out, drop, dirty_list);
            }

            Assert { target, cleanup: Some(unwind), .. }
            | Drop { target, location: _, unwind: Some(unwind) }
            | DropAndReplace { target, value: _, location: _, unwind: Some(unwind) } => {
                self.propagate_bits_into_entry_set_for(in_out, target, dirty_list);
                if self.dead_unwinds.map_or(true, |bbs| !bbs.contains(bb)) {
                    self.propagate_bits_into_entry_set_for(in_out, unwind, dirty_list);
                }
            }

            SwitchInt { ref targets, ref values, ref discr, .. } => {
                self.propagate_bits_into_switch_int_successors(
                    in_out,
                    (bb, bb_data),
                    dirty_list,
                    discr,
                    &*values,
                    &*targets,
                );
            }

            Call { cleanup, ref destination, ref func, ref args, .. } => {
                if let Some(unwind) = cleanup {
                    if self.dead_unwinds.map_or(true, |bbs| !bbs.contains(bb)) {
                        self.propagate_bits_into_entry_set_for(in_out, unwind, dirty_list);
                    }
                }

                if let Some((ref dest_place, dest_bb)) = *destination {
                    // N.B.: This must be done *last*, otherwise the unwind path will see the call
                    // return effect.
                    self.analysis.apply_call_return_effect(in_out, bb, func, args, dest_place);
                    self.propagate_bits_into_entry_set_for(in_out, dest_bb, dirty_list);
                }
            }

            FalseEdges { real_target, imaginary_target } => {
                self.propagate_bits_into_entry_set_for(in_out, real_target, dirty_list);
                self.propagate_bits_into_entry_set_for(in_out, imaginary_target, dirty_list);
            }

            FalseUnwind { real_target, unwind } => {
                self.propagate_bits_into_entry_set_for(in_out, real_target, dirty_list);
                if let Some(unwind) = unwind {
                    if self.dead_unwinds.map_or(true, |bbs| !bbs.contains(bb)) {
                        self.propagate_bits_into_entry_set_for(in_out, unwind, dirty_list);
                    }
                }
            }
        }
    }

    fn propagate_bits_into_entry_set_for(
        &mut self,
        in_out: &BitSet<A::Idx>,
        bb: BasicBlock,
        dirty_queue: &mut WorkQueue<BasicBlock>,
    ) {
        let entry_set = &mut self.entry_sets[bb];
        let set_changed = self.analysis.join(entry_set, &in_out);
        if set_changed {
            dirty_queue.insert(bb);
        }
    }

    fn propagate_bits_into_switch_int_successors(
        &mut self,
        in_out: &mut BitSet<A::Idx>,
        (bb, bb_data): (BasicBlock, &mir::BasicBlockData<'tcx>),
        dirty_list: &mut WorkQueue<BasicBlock>,
        switch_on: &mir::Operand<'tcx>,
        values: &[u128],
        targets: &[BasicBlock],
    ) {
        match bb_data.statements.last().map(|stmt| &stmt.kind) {
            // Look at the last statement to see if it is an assignment of an enum discriminant to
            // the local that determines the target of a `SwitchInt` like so:
            //   _42 = discriminant(..)
            //   SwitchInt(_42, ..)
            Some(mir::StatementKind::Assign(box (lhs, mir::Rvalue::Discriminant(enum_))))
                if Some(lhs) == switch_on.place() =>
            {
                let adt = match enum_.ty(self.body, self.tcx).ty.kind {
                    ty::Adt(def, _) => def,
                    _ => bug!("Switch on discriminant of non-ADT"),
                };

                // MIR building adds discriminants to the `values` array in the same order as they
                // are yielded by `AdtDef::discriminants`. We rely on this to match each
                // discriminant in `values` to its corresponding variant in linear time.
                let mut tmp = BitSet::new_empty(in_out.domain_size());
                let mut discriminants = adt.discriminants(self.tcx);
                for (value, target) in values.iter().zip(targets.iter().copied()) {
                    let (variant_idx, _) =
                        discriminants.find(|&(_, discr)| discr.val == *value).expect(
                            "Order of `AdtDef::discriminants` differed \
                                 from that of `SwitchInt::values`",
                        );

                    tmp.overwrite(in_out);
                    self.analysis.apply_discriminant_switch_effect(
                        &mut tmp,
                        bb,
                        enum_,
                        adt,
                        variant_idx,
                    );
                    self.propagate_bits_into_entry_set_for(&tmp, target, dirty_list);
                }

                std::mem::drop(tmp);

                // Propagate dataflow state along the "otherwise" edge.
                let otherwise = targets.last().copied().unwrap();
                self.propagate_bits_into_entry_set_for(&in_out, otherwise, dirty_list);
            }

            _ => {
                for target in targets.iter().copied() {
                    self.propagate_bits_into_entry_set_for(&in_out, target, dirty_list);
                }
            }
        }
    }
}

// Graphviz

/// Writes a DOT file containing the results of a dataflow analysis if the user requested it via
/// `rustc_mir` attributes.
fn write_graphviz_results<A>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    body: &mir::Body<'tcx>,
    results: &Results<'tcx, A>,
    block_transfer_functions: Option<IndexVec<BasicBlock, GenKillSet<A::Idx>>>,
) -> std::io::Result<()>
where
    A: Analysis<'tcx>,
{
    let attrs = match RustcMirAttrs::parse(tcx, def_id) {
        Ok(attrs) => attrs,

        // Invalid `rustc_mir` attrs will be reported using `span_err`.
        Err(()) => return Ok(()),
    };

    let path = match attrs.output_path(A::NAME) {
        Some(path) => path,
        None => return Ok(()),
    };

    let bits_per_block = results.analysis.bits_per_block(body);

    let mut formatter: Box<dyn graphviz::StateFormatter<'tcx, _>> = match attrs.formatter {
        Some(sym::two_phase) => Box::new(graphviz::TwoPhaseDiff::new(bits_per_block)),
        Some(sym::gen_kill) => {
            if let Some(trans_for_block) = block_transfer_functions {
                Box::new(graphviz::BlockTransferFunc::new(body, trans_for_block))
            } else {
                Box::new(graphviz::SimpleDiff::new(bits_per_block))
            }
        }

        // Default to the `SimpleDiff` output style.
        _ => Box::new(graphviz::SimpleDiff::new(bits_per_block)),
    };

    debug!("printing dataflow results for {:?} to {}", def_id, path.display());
    let mut buf = Vec::new();

    let graphviz = graphviz::Formatter::new(body, def_id, results, &mut *formatter);
    dot::render_opts(&graphviz, &mut buf, &[dot::RenderOption::Monospace])?;
    fs::write(&path, buf)?;
    Ok(())
}

#[derive(Default)]
struct RustcMirAttrs {
    basename_and_suffix: Option<PathBuf>,
    formatter: Option<Symbol>,
}

impl RustcMirAttrs {
    fn parse(tcx: TyCtxt<'tcx>, def_id: DefId) -> Result<Self, ()> {
        let attrs = tcx.get_attrs(def_id);

        let mut result = Ok(());
        let mut ret = RustcMirAttrs::default();

        let rustc_mir_attrs = attrs
            .iter()
            .filter(|attr| attr.check_name(sym::rustc_mir))
            .flat_map(|attr| attr.meta_item_list().into_iter().flat_map(|v| v.into_iter()));

        for attr in rustc_mir_attrs {
            let attr_result = if attr.check_name(sym::borrowck_graphviz_postflow) {
                Self::set_field(&mut ret.basename_and_suffix, tcx, &attr, |s| {
                    let path = PathBuf::from(s.to_string());
                    match path.file_name() {
                        Some(_) => Ok(path),
                        None => {
                            tcx.sess.span_err(attr.span(), "path must end in a filename");
                            Err(())
                        }
                    }
                })
            } else if attr.check_name(sym::borrowck_graphviz_format) {
                Self::set_field(&mut ret.formatter, tcx, &attr, |s| match s {
                    sym::gen_kill | sym::two_phase => Ok(s),
                    _ => {
                        tcx.sess.span_err(attr.span(), "unknown formatter");
                        Err(())
                    }
                })
            } else {
                Ok(())
            };

            result = result.and(attr_result);
        }

        result.map(|()| ret)
    }

    fn set_field<T>(
        field: &mut Option<T>,
        tcx: TyCtxt<'tcx>,
        attr: &ast::NestedMetaItem,
        mapper: impl FnOnce(Symbol) -> Result<T, ()>,
    ) -> Result<(), ()> {
        if field.is_some() {
            tcx.sess
                .span_err(attr.span(), &format!("duplicate values for `{}`", attr.name_or_empty()));

            return Err(());
        }

        if let Some(s) = attr.value_str() {
            *field = Some(mapper(s)?);
            Ok(())
        } else {
            tcx.sess
                .span_err(attr.span(), &format!("`{}` requires an argument", attr.name_or_empty()));
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
