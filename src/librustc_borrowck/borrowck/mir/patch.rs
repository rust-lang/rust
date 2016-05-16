// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::gather_moves::Location;
use rustc::ty::Ty;
use rustc::mir::repr::*;
use syntax::codemap::Span;

use std::iter;
use std::u32;

/// This struct represents a patch to MIR, which can add
/// new statements and basic blocks and patch over block
/// terminators.
pub struct MirPatch<'tcx> {
    patch_map: Vec<Option<TerminatorKind<'tcx>>>,
    new_blocks: Vec<BasicBlockData<'tcx>>,
    new_statements: Vec<(Location, StatementKind<'tcx>)>,
    new_temps: Vec<TempDecl<'tcx>>,
    resume_block: BasicBlock,
    next_temp: u32,
}

impl<'tcx> MirPatch<'tcx> {
    pub fn new(mir: &Mir<'tcx>) -> Self {
        let mut result = MirPatch {
            patch_map: iter::repeat(None)
                .take(mir.basic_blocks.len()).collect(),
            new_blocks: vec![],
            new_temps: vec![],
            new_statements: vec![],
            next_temp: mir.temp_decls.len() as u32,
            resume_block: START_BLOCK
        };

        // make sure the MIR we create has a resume block. It is
        // completely legal to convert jumps to the resume block
        // to jumps to None, but we occasionally have to add
        // instructions just before that.

        let mut resume_block = None;
        let mut resume_stmt_block = None;
        for block in mir.all_basic_blocks() {
            let data = mir.basic_block_data(block);
            if let TerminatorKind::Resume = data.terminator().kind {
                if data.statements.len() > 0 {
                    resume_stmt_block = Some(block);
                } else {
                    resume_block = Some(block);
                }
                break
            }
        }
        let resume_block = resume_block.unwrap_or_else(|| {
            result.new_block(BasicBlockData {
                statements: vec![],
                terminator: Some(Terminator {
                    span: mir.span,
                    scope: ScopeId::new(0),
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
        self.patch_map[bb.index()].is_some()
    }

    pub fn terminator_loc(&self, mir: &Mir<'tcx>, bb: BasicBlock) -> Location {
        let offset = match bb.index().checked_sub(mir.basic_blocks.len()) {
            Some(index) => self.new_blocks[index].statements.len(),
            None => mir.basic_block_data(bb).statements.len()
        };
        Location {
            block: bb,
            index: offset
        }
    }

    pub fn new_temp(&mut self, ty: Ty<'tcx>) -> u32 {
        let index = self.next_temp;
        assert!(self.next_temp < u32::MAX);
        self.next_temp += 1;
        self.new_temps.push(TempDecl { ty: ty });
        index
    }

    pub fn new_block(&mut self, data: BasicBlockData<'tcx>) -> BasicBlock {
        let block = BasicBlock::new(self.patch_map.len());
        debug!("MirPatch: new_block: {:?}: {:?}", block, data);
        self.new_blocks.push(data);
        self.patch_map.push(None);
        block
    }

    pub fn patch_terminator(&mut self, block: BasicBlock, new: TerminatorKind<'tcx>) {
        assert!(self.patch_map[block.index()].is_none());
        debug!("MirPatch: patch_terminator({:?}, {:?})", block, new);
        self.patch_map[block.index()] = Some(new);
    }

    pub fn add_statement(&mut self, loc: Location, stmt: StatementKind<'tcx>) {
        debug!("MirPatch: add_statement({:?}, {:?})", loc, stmt);
        self.new_statements.push((loc, stmt));
    }

    pub fn add_assign(&mut self, loc: Location, lv: Lvalue<'tcx>, rv: Rvalue<'tcx>) {
        self.add_statement(loc, StatementKind::Assign(lv, rv));
    }

    pub fn apply(self, mir: &mut Mir<'tcx>) {
        debug!("MirPatch: {:?} new temps, starting from index {}: {:?}",
               self.new_temps.len(), mir.temp_decls.len(), self.new_temps);
        debug!("MirPatch: {} new blocks, starting from index {}",
               self.new_blocks.len(), mir.basic_blocks.len());
        mir.basic_blocks.extend(self.new_blocks);
        mir.temp_decls.extend(self.new_temps);
        for (src, patch) in self.patch_map.into_iter().enumerate() {
            if let Some(patch) = patch {
                debug!("MirPatch: patching block {:?}", src);
                mir.basic_blocks[src].terminator_mut().kind = patch;
            }
        }

        let mut new_statements = self.new_statements;
        new_statements.sort_by(|u,v| u.0.cmp(&v.0));

        let mut delta = 0;
        let mut last_bb = START_BLOCK;
        for (mut loc, stmt) in new_statements {
            if loc.block != last_bb {
                delta = 0;
                last_bb = loc.block;
            }
            debug!("MirPatch: adding statement {:?} at loc {:?}+{}",
                   stmt, loc, delta);
            loc.index += delta;
            let (span, scope) = Self::context_for_index(
                mir.basic_block_data(loc.block), loc
            );
            mir.basic_block_data_mut(loc.block).statements.insert(
                loc.index, Statement {
                    span: span,
                    scope: scope,
                    kind: stmt
                });
            delta += 1;
        }
    }

    pub fn context_for_index(data: &BasicBlockData, loc: Location) -> (Span, ScopeId) {
        match data.statements.get(loc.index) {
            Some(stmt) => (stmt.span, stmt.scope),
            None => (data.terminator().span, data.terminator().scope)
        }
    }

    pub fn context_for_location(&self, mir: &Mir, loc: Location) -> (Span, ScopeId) {
        let data = match loc.block.index().checked_sub(mir.basic_blocks.len()) {
            Some(new) => &self.new_blocks[new],
            None => mir.basic_block_data(loc.block)
        };
        Self::context_for_index(data, loc)
    }
}
