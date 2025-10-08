use rustc_data_structures::fx::FxHashMap;
use rustc_index::Idx;
use rustc_middle::mir::*;
use rustc_middle::ty::Ty;
use rustc_span::Span;
use tracing::debug;

/// This struct lets you "patch" a MIR body, i.e. modify it. You can queue up
/// various changes, such as the addition of new statements and basic blocks
/// and replacement of terminators, and then apply the queued changes all at
/// once with `apply`. This is useful for MIR transformation passes.
pub(crate) struct MirPatch<'tcx> {
    term_patch_map: FxHashMap<BasicBlock, TerminatorKind<'tcx>>,
    /// Set of statements that should be replaced by `Nop`.
    nop_statements: Vec<Location>,
    new_blocks: Vec<BasicBlockData<'tcx>>,
    new_statements: Vec<(Location, StatementKind<'tcx>)>,
    new_locals: Vec<LocalDecl<'tcx>>,
    resume_block: Option<BasicBlock>,
    // Only for unreachable in cleanup path.
    unreachable_cleanup_block: Option<BasicBlock>,
    // Only for unreachable not in cleanup path.
    unreachable_no_cleanup_block: Option<BasicBlock>,
    // Cached block for UnwindTerminate (with reason)
    terminate_block: Option<(BasicBlock, UnwindTerminateReason)>,
    body_span: Span,
    next_local: usize,
    /// The number of blocks at the start of the transformation. New blocks
    /// get appended at the end.
    next_block: usize,
}

impl<'tcx> MirPatch<'tcx> {
    /// Creates a new, empty patch.
    pub(crate) fn new(body: &Body<'tcx>) -> Self {
        let mut result = MirPatch {
            term_patch_map: Default::default(),
            nop_statements: vec![],
            new_blocks: vec![],
            new_statements: vec![],
            new_locals: vec![],
            next_local: body.local_decls.len(),
            next_block: body.basic_blocks.len(),
            resume_block: None,
            unreachable_cleanup_block: None,
            unreachable_no_cleanup_block: None,
            terminate_block: None,
            body_span: body.span,
        };

        for (bb, block) in body.basic_blocks.iter_enumerated() {
            // Check if we already have a resume block
            if matches!(block.terminator().kind, TerminatorKind::UnwindResume)
                && block.statements.is_empty()
            {
                result.resume_block = Some(bb);
                continue;
            }

            // Check if we already have an unreachable block
            if matches!(block.terminator().kind, TerminatorKind::Unreachable)
                && block.statements.is_empty()
            {
                if block.is_cleanup {
                    result.unreachable_cleanup_block = Some(bb);
                } else {
                    result.unreachable_no_cleanup_block = Some(bb);
                }
                continue;
            }

            // Check if we already have a terminate block
            if let TerminatorKind::UnwindTerminate(reason) = block.terminator().kind
                && block.statements.is_empty()
            {
                result.terminate_block = Some((bb, reason));
                continue;
            }
        }

        result
    }

    pub(crate) fn resume_block(&mut self) -> BasicBlock {
        if let Some(bb) = self.resume_block {
            return bb;
        }

        let bb = self.new_block(BasicBlockData::new(
            Some(Terminator {
                source_info: SourceInfo::outermost(self.body_span),
                kind: TerminatorKind::UnwindResume,
            }),
            true,
        ));
        self.resume_block = Some(bb);
        bb
    }

    pub(crate) fn unreachable_cleanup_block(&mut self) -> BasicBlock {
        if let Some(bb) = self.unreachable_cleanup_block {
            return bb;
        }

        let bb = self.new_block(BasicBlockData::new(
            Some(Terminator {
                source_info: SourceInfo::outermost(self.body_span),
                kind: TerminatorKind::Unreachable,
            }),
            true,
        ));
        self.unreachable_cleanup_block = Some(bb);
        bb
    }

    pub(crate) fn unreachable_no_cleanup_block(&mut self) -> BasicBlock {
        if let Some(bb) = self.unreachable_no_cleanup_block {
            return bb;
        }

        let bb = self.new_block(BasicBlockData::new(
            Some(Terminator {
                source_info: SourceInfo::outermost(self.body_span),
                kind: TerminatorKind::Unreachable,
            }),
            false,
        ));
        self.unreachable_no_cleanup_block = Some(bb);
        bb
    }

    pub(crate) fn terminate_block(&mut self, reason: UnwindTerminateReason) -> BasicBlock {
        if let Some((cached_bb, cached_reason)) = self.terminate_block
            && reason == cached_reason
        {
            return cached_bb;
        }

        let bb = self.new_block(BasicBlockData::new(
            Some(Terminator {
                source_info: SourceInfo::outermost(self.body_span),
                kind: TerminatorKind::UnwindTerminate(reason),
            }),
            true,
        ));
        self.terminate_block = Some((bb, reason));
        bb
    }

    /// Has a replacement of this block's terminator been queued in this patch?
    pub(crate) fn is_term_patched(&self, bb: BasicBlock) -> bool {
        self.term_patch_map.contains_key(&bb)
    }

    /// Universal getter for block data, either it is in 'old' blocks or in patched ones
    pub(crate) fn block<'a>(
        &'a self,
        body: &'a Body<'tcx>,
        bb: BasicBlock,
    ) -> &'a BasicBlockData<'tcx> {
        match bb.index().checked_sub(body.basic_blocks.len()) {
            Some(new) => &self.new_blocks[new],
            None => &body[bb],
        }
    }

    pub(crate) fn terminator_loc(&self, body: &Body<'tcx>, bb: BasicBlock) -> Location {
        let offset = self.block(body, bb).statements.len();
        Location { block: bb, statement_index: offset }
    }

    /// Queues the addition of a new temporary with additional local info.
    pub(crate) fn new_local_with_info(
        &mut self,
        ty: Ty<'tcx>,
        span: Span,
        local_info: LocalInfo<'tcx>,
    ) -> Local {
        let index = self.next_local;
        self.next_local += 1;
        let mut new_decl = LocalDecl::new(ty, span);
        **new_decl.local_info.as_mut().unwrap_crate_local() = local_info;
        self.new_locals.push(new_decl);
        Local::new(index)
    }

    /// Queues the addition of a new temporary.
    pub(crate) fn new_temp(&mut self, ty: Ty<'tcx>, span: Span) -> Local {
        let index = self.next_local;
        self.next_local += 1;
        self.new_locals.push(LocalDecl::new(ty, span));
        Local::new(index)
    }

    /// Returns the type of a local that's newly-added in the patch.
    pub(crate) fn local_ty(&self, local: Local) -> Ty<'tcx> {
        let local = local.as_usize();
        assert!(local < self.next_local);
        let new_local_idx = self.new_locals.len() - (self.next_local - local);
        self.new_locals[new_local_idx].ty
    }

    /// Queues the addition of a new basic block.
    pub(crate) fn new_block(&mut self, data: BasicBlockData<'tcx>) -> BasicBlock {
        let block = BasicBlock::from_usize(self.next_block + self.new_blocks.len());
        debug!("MirPatch: new_block: {:?}: {:?}", block, data);
        self.new_blocks.push(data);
        block
    }

    /// Queues the replacement of a block's terminator.
    pub(crate) fn patch_terminator(&mut self, block: BasicBlock, new: TerminatorKind<'tcx>) {
        assert!(!self.term_patch_map.contains_key(&block));
        debug!("MirPatch: patch_terminator({:?}, {:?})", block, new);
        self.term_patch_map.insert(block, new);
    }

    /// Mark given statement to be replaced by a `Nop`.
    ///
    /// This method only works on statements from the initial body, and cannot be used to remove
    /// statements from `add_statement` or `add_assign`.
    #[tracing::instrument(level = "debug", skip(self))]
    pub(crate) fn nop_statement(&mut self, loc: Location) {
        self.nop_statements.push(loc);
    }

    /// Queues the insertion of a statement at a given location. The statement
    /// currently at that location, and all statements that follow, are shifted
    /// down. If multiple statements are queued for addition at the same
    /// location, the final statement order after calling `apply` will match
    /// the queue insertion order.
    ///
    /// E.g. if we have `s0` at location `loc` and do these calls:
    ///
    ///   p.add_statement(loc, s1);
    ///   p.add_statement(loc, s2);
    ///   p.apply(body);
    ///
    /// then the final order will be `s1, s2, s0`, with `s1` at `loc`.
    pub(crate) fn add_statement(&mut self, loc: Location, stmt: StatementKind<'tcx>) {
        debug!("MirPatch: add_statement({:?}, {:?})", loc, stmt);
        self.new_statements.push((loc, stmt));
    }

    /// Like `add_statement`, but specialized for assignments.
    pub(crate) fn add_assign(&mut self, loc: Location, place: Place<'tcx>, rv: Rvalue<'tcx>) {
        self.add_statement(loc, StatementKind::Assign(Box::new((place, rv))));
    }

    /// Applies the queued changes.
    pub(crate) fn apply(self, body: &mut Body<'tcx>) {
        debug!(
            "MirPatch: {:?} new temps, starting from index {}: {:?}",
            self.new_locals.len(),
            body.local_decls.len(),
            self.new_locals
        );
        debug!(
            "MirPatch: {} new blocks, starting from index {}",
            self.new_blocks.len(),
            body.basic_blocks.len()
        );
        debug_assert_eq!(self.next_block, body.basic_blocks.len());
        let bbs = if self.term_patch_map.is_empty() && self.new_blocks.is_empty() {
            body.basic_blocks.as_mut_preserves_cfg()
        } else {
            body.basic_blocks.as_mut()
        };
        bbs.extend(self.new_blocks);
        body.local_decls.extend(self.new_locals);

        for loc in self.nop_statements {
            bbs[loc.block].statements[loc.statement_index].make_nop(true);
        }

        let mut new_statements = self.new_statements;

        // This must be a stable sort to provide the ordering described in the
        // comment for `add_statement`.
        new_statements.sort_by_key(|s| s.0);

        let mut delta = 0;
        let mut last_bb = START_BLOCK;
        for (mut loc, stmt) in new_statements {
            if loc.block != last_bb {
                delta = 0;
                last_bb = loc.block;
            }
            debug!("MirPatch: adding statement {:?} at loc {:?}+{}", stmt, loc, delta);
            loc.statement_index += delta;
            let source_info = Self::source_info_for_index(&bbs[loc.block], loc);
            bbs[loc.block]
                .statements
                .insert(loc.statement_index, Statement::new(source_info, stmt));
            delta += 1;
        }

        // The order in which we patch terminators does not change the result.
        #[allow(rustc::potential_query_instability)]
        for (src, patch) in self.term_patch_map {
            debug!("MirPatch: patching block {:?}", src);
            let bb = &mut bbs[src];
            if let TerminatorKind::Unreachable = patch {
                bb.statements.clear();
            }
            bb.terminator_mut().kind = patch;
        }
    }

    fn source_info_for_index(data: &BasicBlockData<'_>, loc: Location) -> SourceInfo {
        match data.statements.get(loc.statement_index) {
            Some(stmt) => stmt.source_info,
            None => data.terminator().source_info,
        }
    }

    pub(crate) fn source_info_for_location(&self, body: &Body<'tcx>, loc: Location) -> SourceInfo {
        let data = self.block(body, loc.block);
        Self::source_info_for_index(data, loc)
    }
}
