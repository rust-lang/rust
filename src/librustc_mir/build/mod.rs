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
use rustc::middle::region::{CodeExtent, CodeExtentData, ROOT_CODE_EXTENT};
use rustc::ty::{self, Ty};
use rustc::mir::repr::*;
use rustc_data_structures::fnv::FnvHashMap;
use rustc::hir;
use std::ops::{Index, IndexMut};
use syntax::abi::Abi;
use syntax::ast;
use syntax::codemap::Span;
use syntax::parse::token::keywords;

pub struct Builder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    hir: Cx<'a, 'gcx, 'tcx>,
    cfg: CFG<'tcx>,

    fn_span: Span,

    /// the current set of scopes, updated as we traverse;
    /// see the `scope` module for more details
    scopes: Vec<scope::Scope<'tcx>>,

    ///  for each scope, a span of blocks that defines it;
    ///  we track these for use in region and borrow checking,
    ///  but these are liable to get out of date once optimization
    ///  begins. They are also hopefully temporary, and will be
    ///  no longer needed when we adopt graph-based regions.
    scope_auxiliary: ScopeAuxiliaryVec,

    /// the current set of loops; see the `scope` module for more
    /// details
    loop_scopes: Vec<scope::LoopScope>,

    /// the vector of all scopes that we have created thus far;
    /// we track this for debuginfo later
    scope_datas: Vec<ScopeData>,

    var_decls: Vec<VarDecl<'tcx>>,
    var_indices: FnvHashMap<ast::NodeId, u32>,
    temp_decls: Vec<TempDecl<'tcx>>,
    unit_temp: Option<Lvalue<'tcx>>,

    /// cached block with the RESUME terminator; this is created
    /// when first set of cleanups are built.
    cached_resume_block: Option<BasicBlock>,
    /// cached block with the RETURN terminator
    cached_return_block: Option<BasicBlock>,
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

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Location {
    /// the location is within this block
    pub block: BasicBlock,

    /// the location is the start of the this statement; or, if `statement_index`
    /// == num-statements, then the start of the terminator.
    pub statement_index: usize,
}

pub struct ScopeAuxiliaryVec {
    pub vec: Vec<ScopeAuxiliary>
}

impl Index<ScopeId> for ScopeAuxiliaryVec {
    type Output = ScopeAuxiliary;

    #[inline]
    fn index(&self, index: ScopeId) -> &ScopeAuxiliary {
        &self.vec[index.index()]
    }
}

impl IndexMut<ScopeId> for ScopeAuxiliaryVec {
    #[inline]
    fn index_mut(&mut self, index: ScopeId) -> &mut ScopeAuxiliary {
        &mut self.vec[index.index()]
    }
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

pub fn construct_fn<'a, 'gcx, 'tcx, A>(hir: Cx<'a, 'gcx, 'tcx>,
                                       fn_id: ast::NodeId,
                                       arguments: A,
                                       return_ty: ty::FnOutput<'gcx>,
                                       ast_block: &'gcx hir::Block)
                                       -> (Mir<'tcx>, ScopeAuxiliaryVec)
    where A: Iterator<Item=(Ty<'gcx>, Option<&'gcx hir::Pat>)>
{
    let tcx = hir.tcx();
    let span = tcx.map.span(fn_id);
    let mut builder = Builder::new(hir, span);

    let body_id = ast_block.id;
    let call_site_extent =
        tcx.region_maps.lookup_code_extent(
            CodeExtentData::CallSiteScope { fn_id: fn_id, body_id: body_id });
    let arg_extent =
        tcx.region_maps.lookup_code_extent(
            CodeExtentData::ParameterScope { fn_id: fn_id, body_id: body_id });
    let mut block = START_BLOCK;
    let mut arg_decls = unpack!(block = builder.in_scope(call_site_extent, block,
                                                         |builder, call_site_scope_id| {
        let arg_decls = unpack!(block = builder.in_scope(arg_extent, block,
                                                         |builder, arg_scope_id| {
            builder.args_and_body(block, return_ty, arguments, arg_scope_id, ast_block)
        }));

        let return_block = builder.return_block();
        builder.cfg.terminate(block, call_site_scope_id, span,
                              TerminatorKind::Goto { target: return_block });
        builder.cfg.terminate(return_block, call_site_scope_id, span,
                              TerminatorKind::Return);
        return_block.and(arg_decls)
    }));
    assert_eq!(block, builder.return_block());

    match tcx.node_id_to_type(fn_id).sty {
        ty::TyFnDef(_, _, f) if f.abi == Abi::RustCall => {
            // RustCall pseudo-ABI untuples the last argument.
            if let Some(arg_decl) = arg_decls.last_mut() {
                arg_decl.spread = true;
            }
        }
        _ => {}
    }

    // Gather the upvars of a closure, if any.
    let upvar_decls: Vec<_> = tcx.with_freevars(fn_id, |freevars| {
        freevars.iter().map(|fv| {
            let by_ref = tcx.upvar_capture(ty::UpvarId {
                var_id: fv.def.var_id(),
                closure_expr_id: fn_id
            }).map_or(false, |capture| match capture {
                ty::UpvarCapture::ByValue => false,
                ty::UpvarCapture::ByRef(..) => true
            });
            let mut decl = UpvarDecl {
                debug_name: keywords::Invalid.name(),
                by_ref: by_ref
            };
            if let Some(hir::map::NodeLocal(pat)) = tcx.map.find(fv.def.var_id()) {
                if let hir::PatKind::Binding(_, ref ident, _) = pat.node {
                    decl.debug_name = ident.node;
                }
            }
            decl
        }).collect()
    });

    builder.finish(upvar_decls, arg_decls, return_ty)
}

pub fn construct_const<'a, 'gcx, 'tcx>(hir: Cx<'a, 'gcx, 'tcx>,
                                       item_id: ast::NodeId,
                                       ast_expr: &'tcx hir::Expr)
                                       -> (Mir<'tcx>, ScopeAuxiliaryVec) {
    let tcx = hir.tcx();
    let span = tcx.map.span(item_id);
    let mut builder = Builder::new(hir, span);

    let extent = ROOT_CODE_EXTENT;
    let mut block = START_BLOCK;
    let _ = builder.in_scope(extent, block, |builder, call_site_scope_id| {
        let expr = builder.hir.mirror(ast_expr);
        unpack!(block = builder.into(&Lvalue::ReturnPointer, block, expr));

        let return_block = builder.return_block();
        builder.cfg.terminate(block, call_site_scope_id, span,
                              TerminatorKind::Goto { target: return_block });
        builder.cfg.terminate(return_block, call_site_scope_id, span,
                              TerminatorKind::Return);

        return_block.unit()
    });

    let ty = tcx.expr_ty_adjusted(ast_expr);
    builder.finish(vec![], vec![], ty::FnConverging(ty))
}

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    fn new(hir: Cx<'a, 'gcx, 'tcx>, span: Span) -> Builder<'a, 'gcx, 'tcx> {
        let mut builder = Builder {
            hir: hir,
            cfg: CFG { basic_blocks: vec![] },
            fn_span: span,
            scopes: vec![],
            scope_datas: vec![],
            scope_auxiliary: ScopeAuxiliaryVec { vec: vec![] },
            loop_scopes: vec![],
            temp_decls: vec![],
            var_decls: vec![],
            var_indices: FnvHashMap(),
            unit_temp: None,
            cached_resume_block: None,
            cached_return_block: None
        };

        assert_eq!(builder.cfg.start_new_block(), START_BLOCK);

        builder
    }

    fn finish(self,
              upvar_decls: Vec<UpvarDecl>,
              arg_decls: Vec<ArgDecl<'tcx>>,
              return_ty: ty::FnOutput<'tcx>)
              -> (Mir<'tcx>, ScopeAuxiliaryVec) {
        for (index, block) in self.cfg.basic_blocks.iter().enumerate() {
            if block.terminator.is_none() {
                span_bug!(self.fn_span, "no terminator on block {:?}", index);
            }
        }

        (Mir {
            basic_blocks: self.cfg.basic_blocks,
            scopes: self.scope_datas,
            promoted: vec![],
            var_decls: self.var_decls,
            arg_decls: arg_decls,
            temp_decls: self.temp_decls,
            upvar_decls: upvar_decls,
            return_ty: return_ty,
            span: self.fn_span
        }, self.scope_auxiliary)
    }

    fn args_and_body<A>(&mut self,
                        mut block: BasicBlock,
                        return_ty: ty::FnOutput<'tcx>,
                        arguments: A,
                        argument_scope_id: ScopeId,
                        ast_block: &'gcx hir::Block)
                        -> BlockAnd<Vec<ArgDecl<'tcx>>>
        where A: Iterator<Item=(Ty<'gcx>, Option<&'gcx hir::Pat>)>
    {
        // to start, translate the argument patterns and collect the argument types.
        let arg_decls = arguments.enumerate().map(|(index, (ty, pattern))| {
            let lvalue = Lvalue::Arg(index as u32);
            if let Some(pattern) = pattern {
                let pattern = self.hir.irrefutable_pat(pattern);
                unpack!(block = self.lvalue_into_pattern(block,
                                                         argument_scope_id,
                                                         pattern,
                                                         &lvalue));
            }

            // Make sure we drop (parts of) the argument even when not matched on.
            let argument_extent = self.scope_auxiliary[argument_scope_id].extent;
            self.schedule_drop(pattern.as_ref().map_or(ast_block.span, |pat| pat.span),
                               argument_extent, &lvalue, ty);

            let mut name = keywords::Invalid.name();
            if let Some(pat) = pattern {
                if let hir::PatKind::Binding(_, ref ident, _) = pat.node {
                    name = ident.node;
                }
            }

            ArgDecl {
                ty: ty,
                spread: false,
                debug_name: name
            }
        }).collect();

        // FIXME(#32959): temporary hack for the issue at hand
        let return_is_unit = if let ty::FnConverging(t) = return_ty {
            t.is_nil()
        } else {
            false
        };
        // start the first basic block and translate the body
        unpack!(block = self.ast_block(&Lvalue::ReturnPointer, return_is_unit, block, ast_block));

        block.and(arg_decls)
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

    fn return_block(&mut self) -> BasicBlock {
        match self.cached_return_block {
            Some(rb) => rb,
            None => {
                let rb = self.cfg.start_new_block();
                self.cached_return_block = Some(rb);
                rb
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
