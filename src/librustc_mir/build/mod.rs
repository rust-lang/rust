// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hair::cx::Cx;
use rustc::middle::region::CodeExtent;
use rustc::middle::ty::{FnOutput, Ty};
use rustc::mir::repr::*;
use rustc_data_structures::fnv::FnvHashMap;
use rustc_front::hir;

use syntax::ast;
use syntax::codemap::Span;

pub struct Builder<'a, 'tcx: 'a> {
    hir: Cx<'a, 'tcx>,
    cfg: CFG<'tcx>,

    // the current set of scopes, updated as we traverse;
    // see the `scope` module for more details
    scopes: Vec<scope::Scope<'tcx>>,

    // for each scope, a span of blocks that defines it;
    // we track these for use in region and borrow checking,
    // but these are liable to get out of date once optimization
    // begins. They are also hopefully temporary, and will be
    // no longer needed when we adopt graph-based regions.
    scope_auxiliary: Vec<ScopeAuxiliary>,

    // the current set of loops; see the `scope` module for more
    // details
    loop_scopes: Vec<scope::LoopScope>,

    // the vector of all scopes that we have created thus far;
    // we track this for debuginfo later
    scope_data_vec: ScopeDataVec,

    var_decls: Vec<VarDecl<'tcx>>,
    var_indices: FnvHashMap<ast::NodeId, u32>,
    temp_decls: Vec<TempDecl<'tcx>>,
    unit_temp: Option<Lvalue<'tcx>>,
}

struct CFG<'tcx> {
    basic_blocks: Vec<BasicBlockData<'tcx>>,
}

/// For each scope, we track the extent (from the HIR) and a
/// single-entry-multiple-exit subgraph that contains all the
/// statements/terminators within it.
///
/// This information is separated out from the main `ScopeData`
/// because it is short-lived. First, the extent contains node-ids,
/// so it cannot be saved and re-loaded. Second, any optimization will mess up
/// the dominator/postdominator information.
///
/// The intention is basically to use this information to do
/// regionck/borrowck and then throw it away once we are done.
pub struct ScopeAuxiliary {
    /// extent of this scope from the MIR.
    pub extent: CodeExtent,

    /// "entry point": dominator of all nodes in the scope
    pub dom: Location,

    /// "exit points": mutual postdominators of all nodes in the scope
    pub postdoms: Vec<Location>,
}

pub struct Location {
    /// the location is within this block
    pub block: BasicBlock,

    /// the location is the start of the this statement; or, if `statement_index`
    /// == num-statements, then the start of the terminator.
    pub statement_index: usize,
}

pub struct MirPlusPlus<'tcx> {
    pub mir: Mir<'tcx>,
    pub scope_auxiliary: Vec<ScopeAuxiliary>,
}

///////////////////////////////////////////////////////////////////////////
/// The `BlockAnd` "monad" packages up the new basic block along with a
/// produced value (sometimes just unit, of course). The `unpack!`
/// macro (and methods below) makes working with `BlockAnd` much more
/// convenient.

#[must_use] // if you don't use one of these results, you're leaving a dangling edge
pub struct BlockAnd<T>(BasicBlock, T);

trait BlockAndExtension {
    fn and<T>(self, v: T) -> BlockAnd<T>;
    fn unit(self) -> BlockAnd<()>;
}

impl BlockAndExtension for BasicBlock {
    fn and<T>(self, v: T) -> BlockAnd<T> {
        BlockAnd(self, v)
    }

    fn unit(self) -> BlockAnd<()> {
        BlockAnd(self, ())
    }
}

/// Update a block pointer and return the value.
/// Use it like `let x = unpack!(block = self.foo(block, foo))`.
macro_rules! unpack {
    ($x:ident = $c:expr) => {
        {
            let BlockAnd(b, v) = $c;
            $x = b;
            v
        }
    };

    ($c:expr) => {
        {
            let BlockAnd(b, ()) = $c;
            b
        }
    };
}

///////////////////////////////////////////////////////////////////////////
/// the main entry point for building MIR for a function

pub fn construct<'a,'tcx>(hir: Cx<'a,'tcx>,
                          span: Span,
                          implicit_arguments: Vec<Ty<'tcx>>,
                          explicit_arguments: Vec<(Ty<'tcx>, &'tcx hir::Pat)>,
                          argument_extent: CodeExtent,
                          return_ty: FnOutput<'tcx>,
                          ast_block: &'tcx hir::Block)
                          -> MirPlusPlus<'tcx> {
    let cfg = CFG { basic_blocks: vec![] };

    let mut builder = Builder {
        hir: hir,
        cfg: cfg,
        scopes: vec![],
        scope_data_vec: ScopeDataVec::new(),
        scope_auxiliary: vec![],
        loop_scopes: vec![],
        temp_decls: vec![],
        var_decls: vec![],
        var_indices: FnvHashMap(),
        unit_temp: None,
    };

    assert_eq!(builder.cfg.start_new_block(), START_BLOCK);
    assert_eq!(builder.cfg.start_new_block(), END_BLOCK);

    let mut block = START_BLOCK;
    let arg_decls = unpack!(block = builder.args_and_body(block,
                                                          implicit_arguments,
                                                          explicit_arguments,
                                                          argument_extent,
                                                          ast_block));

    builder.cfg.terminate(block, Terminator::Goto { target: END_BLOCK });
    builder.cfg.terminate(END_BLOCK, Terminator::Return);

    MirPlusPlus {
        mir: Mir {
            basic_blocks: builder.cfg.basic_blocks,
            scopes: builder.scope_data_vec,
            var_decls: builder.var_decls,
            arg_decls: arg_decls,
            temp_decls: builder.temp_decls,
            return_ty: return_ty,
            span: span
        },
        scope_auxiliary: builder.scope_auxiliary,
    }
}

impl<'a,'tcx> Builder<'a,'tcx> {
    fn args_and_body(&mut self,
                     mut block: BasicBlock,
                     implicit_arguments: Vec<Ty<'tcx>>,
                     explicit_arguments: Vec<(Ty<'tcx>, &'tcx hir::Pat)>,
                     argument_extent: CodeExtent,
                     ast_block: &'tcx hir::Block)
                     -> BlockAnd<Vec<ArgDecl<'tcx>>>
    {
        self.in_scope(argument_extent, block, |this| {
            // to start, translate the argument patterns and collect the argument types.
            let implicits = implicit_arguments.into_iter().map(|ty| (ty, None));
            let explicits = explicit_arguments.into_iter().map(|(ty, pat)| (ty, Some(pat)));
            let arg_decls =
                implicits
                .chain(explicits)
                .enumerate()
                .map(|(index, (ty, pattern))| {
                    let lvalue = Lvalue::Arg(index as u32);
                    if let Some(pattern) = pattern {
                        let pattern = this.hir.irrefutable_pat(pattern);
                        unpack!(block = this.lvalue_into_pattern(block,
                                                                 argument_extent,
                                                                 pattern,
                                                                 &lvalue));
                    }
                    // Make sure we drop (parts of) the argument even when not matched on.
                    this.schedule_drop(pattern.as_ref().map_or(ast_block.span, |pat| pat.span),
                                       argument_extent, &lvalue, ty);
                    ArgDecl { ty: ty, spread: false }
                })
                .collect();

            // start the first basic block and translate the body
            unpack!(block = this.ast_block(&Lvalue::ReturnPointer, block, ast_block));

            block.and(arg_decls)
        })
    }

    fn get_unit_temp(&mut self) -> Lvalue<'tcx> {
        match self.unit_temp {
            Some(ref tmp) => tmp.clone(),
            None => {
                let ty = self.hir.unit_ty();
                let tmp = self.temp(ty);
                self.unit_temp = Some(tmp.clone());
                tmp
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Builder methods are broken up into modules, depending on what kind
// of thing is being translated. Note that they use the `unpack` macro
// above extensively.

mod block;
mod cfg;
mod expr;
mod into;
mod matches;
mod misc;
mod scope;
