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
    scopes: Vec<scope::Scope<'tcx>>,
    loop_scopes: Vec<scope::LoopScope>,
    unit_temp: Lvalue<'tcx>,
    var_decls: Vec<VarDecl<'tcx>>,
    var_indices: FnvHashMap<ast::NodeId, u32>,
    temp_decls: Vec<TempDecl<'tcx>>,
}

struct CFG<'tcx> {
    basic_blocks: Vec<BasicBlockData<'tcx>>,
}

///////////////////////////////////////////////////////////////////////////
// The `BlockAnd` "monad" packages up the new basic block along with a
// produced value (sometimes just unit, of course). The `unpack!`
// macro (and methods below) makes working with `BlockAnd` much more
// convenient.

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
// construct() -- the main entry point for building MIR for a function

pub fn construct<'a,'tcx>(mut hir: Cx<'a,'tcx>,
                          _span: Span,
                          implicit_arguments: Vec<Ty<'tcx>>,
                          explicit_arguments: Vec<(Ty<'tcx>, &'tcx hir::Pat)>,
                          argument_extent: CodeExtent,
                          return_ty: FnOutput<'tcx>,
                          ast_block: &'tcx hir::Block)
                          -> Mir<'tcx> {
    let cfg = CFG { basic_blocks: vec![] };

    // it's handy to have a temporary of type `()` sometimes, so make
    // one from the start and keep it available
    let temp_decls = vec![TempDecl::<'tcx> { ty: hir.unit_ty() }];
    let unit_temp = Lvalue::Temp(0);

    let mut builder = Builder {
        hir: hir,
        cfg: cfg,
        scopes: vec![],
        loop_scopes: vec![],
        temp_decls: temp_decls,
        var_decls: vec![],
        var_indices: FnvHashMap(),
        unit_temp: unit_temp,
    };

    assert_eq!(builder.cfg.start_new_block(), START_BLOCK);
    assert_eq!(builder.cfg.start_new_block(), END_BLOCK);
    assert_eq!(builder.cfg.start_new_block(), DIVERGE_BLOCK);

    let mut block = START_BLOCK;
    let arg_decls = unpack!(block = builder.args_and_body(block,
                                                          implicit_arguments,
                                                          explicit_arguments,
                                                          argument_extent,
                                                          ast_block));

    builder.cfg.terminate(block, Terminator::Goto { target: END_BLOCK });
    builder.cfg.terminate(END_BLOCK, Terminator::Return);

    Mir {
        basic_blocks: builder.cfg.basic_blocks,
        var_decls: builder.var_decls,
        arg_decls: arg_decls,
        temp_decls: builder.temp_decls,
        return_ty: return_ty,
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
                    if let Some(pattern) = pattern {
                        let lvalue = Lvalue::Arg(index as u32);
                        let pattern = this.hir.irrefutable_pat(pattern);
                        unpack!(block = this.lvalue_into_pattern(block,
                                                                 argument_extent,
                                                                 pattern,
                                                                 &lvalue));
                    }
                    ArgDecl { ty: ty }
                })
                .collect();

            // start the first basic block and translate the body
            unpack!(block = this.ast_block(&Lvalue::ReturnPointer, block, ast_block));

            block.and(arg_decls)
        })
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
mod stmt;
