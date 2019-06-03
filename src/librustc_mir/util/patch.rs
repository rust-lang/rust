use rustc::ty::Ty;
use rustc::mir::*;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use syntax_pos::Span;

/// This struct represents a patch to MIR, which can add
/// new statements and basic blocks and patch over block
/// terminators.
pub struct MirPatch<'tcx> {
    patch_map: IndexVec<BasicBlock, Option<TerminatorKind<'tcx>>>,
    new_blocks: Vec<BasicBlockData<'tcx>>,
    new_statements: Vec<(Location, StatementKind<'tcx>)>,
    new_locals: Vec<LocalDecl<'tcx>>,
    resume_block: BasicBlock,
    next_local: usize,
    make_nop: Vec<Location>,
}

impl<'tcx> MirPatch<'tcx> {
    pub fn new(body: &Body<'tcx>) -> Self {
        let mut result = MirPatch {
            patch_map: IndexVec::from_elem(None, body.basic_blocks()),
            new_blocks: vec![],
            new_statements: vec![],
            new_locals: vec![],
            next_local: body.local_decls.len(),
            resume_block: START_BLOCK,
            make_nop: vec![]
        };

        // make sure the MIR we create has a resume block. It is
        // completely legal to convert jumps to the resume block
        // to jumps to None, but we occasionally have to add
        // instructions just before that.

        let mut resume_block = None;
        let mut resume_stmt_block = None;
        for (bb, block) in body.basic_blocks().iter_enumerated() {
            if let TerminatorKind::Resume = block.terminator().kind {
                if block.statements.len() > 0 {
                    assert!(resume_stmt_block.is_none());
                    resume_stmt_block = Some(bb);
                } else {
                    resume_block = Some(bb);
                }
                break
            }
        }
        let resume_block = resume_block.unwrap_or_else(|| {
            result.new_block(BasicBlockData {
                statements: vec![],
                terminator: Some(Terminator {
                    source_info: SourceInfo {
                        span: body.span,
                        scope: OUTERMOST_SOURCE_SCOPE
                    },
                    kind: TerminatorKind::Resume
                }),
                is_cleanup: true
            })});
        result.resume_block = resume_block;
        if let Some(resume_stmt_block) = resume_stmt_block {
            result.patch_terminator(resume_stmt_block, TerminatorKind::Goto {
                target: resume_block
            });
        }
        result
    }

    pub fn resume_block(&self) -> BasicBlock {
        self.resume_block
    }

    pub fn is_patched(&self, bb: BasicBlock) -> bool {
        self.patch_map[bb].is_some()
    }

    pub fn terminator_loc(&self, body: &Body<'tcx>, bb: BasicBlock) -> Location {
        let offset = match bb.index().checked_sub(body.basic_blocks().len()) {
            Some(index) => self.new_blocks[index].statements.len(),
            None => body[bb].statements.len()
        };
        Location {
            block: bb,
            statement_index: offset
        }
    }

    pub fn new_temp(&mut self, ty: Ty<'tcx>, span: Span) -> Local {
        let index = self.next_local;
        self.next_local += 1;
        self.new_locals.push(LocalDecl::new_temp(ty, span));
        Local::new(index as usize)
    }

    pub fn new_internal(&mut self, ty: Ty<'tcx>, span: Span) -> Local {
        let index = self.next_local;
        self.next_local += 1;
        self.new_locals.push(LocalDecl::new_internal(ty, span));
        Local::new(index as usize)
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
        self.add_statement(loc, StatementKind::Assign(place, box rv));
    }

    pub fn make_nop(&mut self, loc: Location) {
        self.make_nop.push(loc);
    }

    pub fn apply(self, body: &mut Body<'tcx>) {
        debug!("MirPatch: make nops at: {:?}", self.make_nop);
        for loc in self.make_nop {
            body.make_statement_nop(loc);
        }
        debug!("MirPatch: {:?} new temps, starting from index {}: {:?}",
               self.new_locals.len(), body.local_decls.len(), self.new_locals);
        debug!("MirPatch: {} new blocks, starting from index {}",
               self.new_blocks.len(), body.basic_blocks().len());
        body.basic_blocks_mut().extend(self.new_blocks);
        body.local_decls.extend(self.new_locals);
        for (src, patch) in self.patch_map.into_iter_enumerated() {
            if let Some(patch) = patch {
                debug!("MirPatch: patching block {:?}", src);
                body[src].terminator_mut().kind = patch;
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
            debug!("MirPatch: adding statement {:?} at loc {:?}+{}",
                   stmt, loc, delta);
            loc.statement_index += delta;
            let source_info = Self::source_info_for_index(
                &body[loc.block], loc
            );
            body[loc.block].statements.insert(
                loc.statement_index, Statement {
                    source_info,
                    kind: stmt
                });
            delta += 1;
        }
    }

    pub fn source_info_for_index(data: &BasicBlockData<'_>, loc: Location) -> SourceInfo {
        match data.statements.get(loc.statement_index) {
            Some(stmt) => stmt.source_info,
            None => data.terminator().source_info
        }
    }

    pub fn source_info_for_location(&self, body: &Body<'_>, loc: Location) -> SourceInfo {
        let data = match loc.block.index().checked_sub(body.basic_blocks().len()) {
            Some(new) => &self.new_blocks[new],
            None => &body[loc.block]
        };
        Self::source_info_for_index(data, loc)
    }
}
