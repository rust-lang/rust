use rustc_middle::mir::*;
use tracing::debug;

/// This struct represents a patch to MIR, which can add
/// new statements and basic blocks and patch over block
/// terminators.
pub struct MirPatch<'tcx> {
    patch_map: IndexVec<BasicBlock, Option<TerminatorKind<'tcx>>>,
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
}

impl<'tcx> MirPatch<'tcx> {
    pub fn new(body: &Body<'tcx>) -> Self {
        let mut result = MirPatch {
            patch_map: IndexVec::from_elem(None, &body.basic_blocks),
            new_blocks: vec![],
            new_statements: vec![],
            new_locals: vec![],
            next_local: body.local_decls.len(),
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

    pub fn resume_block(&mut self) -> BasicBlock {
        if let Some(bb) = self.resume_block {
            return bb;
        }

        let bb = self.new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: SourceInfo::outermost(self.body_span),
                kind: TerminatorKind::UnwindResume,
            }),
            is_cleanup: true,
        });
        self.resume_block = Some(bb);
        bb
    }

    pub fn unreachable_cleanup_block(&mut self) -> BasicBlock {
        if let Some(bb) = self.unreachable_cleanup_block {
            return bb;
        }

        let bb = self.new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: SourceInfo::outermost(self.body_span),
                kind: TerminatorKind::Unreachable,
            }),
            is_cleanup: true,
        });
        self.unreachable_cleanup_block = Some(bb);
        bb
    }

    pub fn unreachable_no_cleanup_block(&mut self) -> BasicBlock {
        if let Some(bb) = self.unreachable_no_cleanup_block {
            return bb;
        }

        let bb = self.new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: SourceInfo::outermost(self.body_span),
                kind: TerminatorKind::Unreachable,
            }),
            is_cleanup: false,
        });
        self.unreachable_no_cleanup_block = Some(bb);
        bb
    }

    pub fn terminate_block(&mut self, reason: UnwindTerminateReason) -> BasicBlock {
        if let Some((cached_bb, cached_reason)) = self.terminate_block
            && reason == cached_reason
        {
            return cached_bb;
        }

        let bb = self.new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: SourceInfo::outermost(self.body_span),
                kind: TerminatorKind::UnwindTerminate(reason),
            }),
            is_cleanup: true,
        });
        self.terminate_block = Some((bb, reason));
        bb
    }

    pub fn is_patched(&self, bb: BasicBlock) -> bool {
        self.patch_map[bb].is_some()
    }

    pub fn terminator_loc(&self, body: &Body<'tcx>, bb: BasicBlock) -> Location {
        let offset = match bb.index().checked_sub(body.basic_blocks.len()) {
            Some(index) => self.new_blocks[index].statements.len(),
            None => body[bb].statements.len(),
        };
        Location { block: bb, statement_index: offset }
    }

    pub fn new_local_with_info(
        &mut self,
        ty: Ty<'tcx>,
        span: Span,
        local_info: LocalInfo<'tcx>,
    ) -> Local {
        let index = self.next_local;
        self.next_local += 1;
        let mut new_decl = LocalDecl::new(ty, span);
        **new_decl.local_info.as_mut().assert_crate_local() = local_info;
        self.new_locals.push(new_decl);
        Local::new(index)
    }

    pub fn new_temp(&mut self, ty: Ty<'tcx>, span: Span) -> Local {
        let index = self.next_local;
        self.next_local += 1;
        self.new_locals.push(LocalDecl::new(ty, span));
        Local::new(index)
    }

    pub fn new_block(&mut self, data: BasicBlockData<'tcx>) -> BasicBlock {
        let block = BasicBlock::new(self.patch_map.len());
        debug!("MirPatch: new_block: {:?}: {:?}", block, data);
        self.new_blocks.push(data);
        self.patch_map.push(None);
        block
    }

    pub fn patch_terminator(&mut self, block: BasicBlock, new: TerminatorKind<'tcx>) {
        assert!(self.patch_map[block].is_none());
        debug!("MirPatch: patch_terminator({:?}, {:?})", block, new);
        self.patch_map[block] = Some(new);
    }

    pub fn add_statement(&mut self, loc: Location, stmt: StatementKind<'tcx>) {
        debug!("MirPatch: add_statement({:?}, {:?})", loc, stmt);
        self.new_statements.push((loc, stmt));
    }

    pub fn add_assign(&mut self, loc: Location, place: Place<'tcx>, rv: Rvalue<'tcx>) {
        self.add_statement(loc, StatementKind::Assign(Box::new((place, rv))));
    }

    pub fn apply(self, body: &mut Body<'tcx>) {
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
        let bbs = if self.patch_map.is_empty() && self.new_blocks.is_empty() {
            body.basic_blocks.as_mut_preserves_cfg()
        } else {
            body.basic_blocks.as_mut()
        };
        bbs.extend(self.new_blocks);
        body.local_decls.extend(self.new_locals);
        for (src, patch) in self.patch_map.into_iter_enumerated() {
            if let Some(patch) = patch {
                debug!("MirPatch: patching block {:?}", src);
                bbs[src].terminator_mut().kind = patch;
            }
        }

        let mut new_statements = self.new_statements;
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
            let source_info = Self::source_info_for_index(&body[loc.block], loc);
            body[loc.block]
                .statements
                .insert(loc.statement_index, Statement { source_info, kind: stmt });
            delta += 1;
        }
    }

    pub fn source_info_for_index(data: &BasicBlockData<'_>, loc: Location) -> SourceInfo {
        match data.statements.get(loc.statement_index) {
            Some(stmt) => stmt.source_info,
            None => data.terminator().source_info,
        }
    }

    pub fn source_info_for_location(&self, body: &Body<'tcx>, loc: Location) -> SourceInfo {
        let data = match loc.block.index().checked_sub(body.basic_blocks.len()) {
            Some(new) => &self.new_blocks[new],
            None => &body[loc.block],
        };
        Self::source_info_for_index(data, loc)
    }
}
