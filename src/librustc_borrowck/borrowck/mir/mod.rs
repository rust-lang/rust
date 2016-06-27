// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrowck::BorrowckCtxt;

use syntax::ast::{self, MetaItem};
use syntax::attr::AttrMetaMethods;
use syntax::ptr::P;
use syntax_pos::{Span, DUMMY_SP};

use rustc::hir;
use rustc::hir::intravisit::{FnKind};

use rustc::mir::repr;
use rustc::mir::repr::{BasicBlock, BasicBlockData, Mir, Statement, Terminator};
use rustc::session::Session;
use rustc::ty::{self, TyCtxt};

mod abs_domain;
pub mod elaborate_drops;
mod dataflow;
mod gather_moves;
mod patch;
// mod graphviz;

use self::dataflow::{BitDenotation};
use self::dataflow::{DataflowOperator};
use self::dataflow::{Dataflow, DataflowAnalysis, DataflowResults};
use self::dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use self::dataflow::{DefinitelyInitializedLvals};
use self::gather_moves::{MoveData, MovePathIndex, Location};
use self::gather_moves::{MovePathContent, MovePathData};

fn has_rustc_mir_with(attrs: &[ast::Attribute], name: &str) -> Option<P<MetaItem>> {
    for attr in attrs {
        if attr.check_name("rustc_mir") {
            let items = attr.meta_item_list();
            for item in items.iter().flat_map(|l| l.iter()) {
                if item.check_name(name) {
                    return Some(item.clone())
                }
            }
        }
    }
    return None;
}

pub struct MoveDataParamEnv<'tcx> {
    move_data: MoveData<'tcx>,
    param_env: ty::ParameterEnvironment<'tcx>,
}

pub fn borrowck_mir<'a, 'tcx: 'a>(
    bcx: &mut BorrowckCtxt<'a, 'tcx>,
    fk: FnKind,
    _decl: &hir::FnDecl,
    mir: &'a Mir<'tcx>,
    body: &hir::Block,
    _sp: Span,
    id: ast::NodeId,
    attributes: &[ast::Attribute]) {
    match fk {
        FnKind::ItemFn(name, _, _, _, _, _, _) |
        FnKind::Method(name, _, _, _) => {
            debug!("borrowck_mir({}) UNIMPLEMENTED", name);
        }
        FnKind::Closure(_) => {
            debug!("borrowck_mir closure (body.id={}) UNIMPLEMENTED", body.id);
        }
    }

    let tcx = bcx.tcx;

    let move_data = MoveData::gather_moves(mir, tcx);
    let param_env = ty::ParameterEnvironment::for_item(tcx, id);
    let mdpe = MoveDataParamEnv { move_data: move_data, param_env: param_env };
    let flow_inits =
        do_dataflow(tcx, mir, id, attributes, &mdpe, MaybeInitializedLvals::new(tcx, mir));
    let flow_uninits =
        do_dataflow(tcx, mir, id, attributes, &mdpe, MaybeUninitializedLvals::new(tcx, mir));
    let flow_def_inits =
        do_dataflow(tcx, mir, id, attributes, &mdpe, DefinitelyInitializedLvals::new(tcx, mir));

    if has_rustc_mir_with(attributes, "rustc_peek_maybe_init").is_some() {
        dataflow::sanity_check_via_rustc_peek(bcx.tcx, mir, id, attributes, &mdpe, &flow_inits);
    }
    if has_rustc_mir_with(attributes, "rustc_peek_maybe_uninit").is_some() {
        dataflow::sanity_check_via_rustc_peek(bcx.tcx, mir, id, attributes, &mdpe, &flow_uninits);
    }
    if has_rustc_mir_with(attributes, "rustc_peek_definite_init").is_some() {
        dataflow::sanity_check_via_rustc_peek(bcx.tcx, mir, id, attributes, &mdpe, &flow_def_inits);
    }

    if has_rustc_mir_with(attributes, "stop_after_dataflow").is_some() {
        bcx.tcx.sess.fatal("stop_after_dataflow ended compilation");
    }

    let mut mbcx = MirBorrowckCtxt {
        bcx: bcx,
        mir: mir,
        node_id: id,
        move_data: mdpe.move_data,
        flow_inits: flow_inits,
        flow_uninits: flow_uninits,
    };

    for bb in mir.basic_blocks().indices() {
        mbcx.process_basic_block(bb);
    }

    debug!("borrowck_mir done");
}

fn do_dataflow<'a, 'tcx, BD>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             mir: &Mir<'tcx>,
                             node_id: ast::NodeId,
                             attributes: &[ast::Attribute],
                             ctxt: &BD::Ctxt,
                             bd: BD) -> DataflowResults<BD>
    where BD: BitDenotation<Idx=MovePathIndex, Ctxt=MoveDataParamEnv<'tcx>> + DataflowOperator
{
    use syntax::attr::AttrMetaMethods;

    let name_found = |sess: &Session, attrs: &[ast::Attribute], name| -> Option<String> {
        if let Some(item) = has_rustc_mir_with(attrs, name) {
            if let Some(s) = item.value_str() {
                return Some(s.to_string())
            } else {
                sess.span_err(
                    item.span,
                    &format!("{} attribute requires a path", item.name()));
                return None;
            }
        }
        return None;
    };

    let print_preflow_to =
        name_found(tcx.sess, attributes, "borrowck_graphviz_preflow");
    let print_postflow_to =
        name_found(tcx.sess, attributes, "borrowck_graphviz_postflow");

    let mut mbcx = MirBorrowckCtxtPreDataflow {
        node_id: node_id,
        print_preflow_to: print_preflow_to,
        print_postflow_to: print_postflow_to,
        flow_state: DataflowAnalysis::new(tcx, mir, ctxt, bd),
    };

    mbcx.dataflow(|ctxt, i| &ctxt.move_data.move_paths[i]);
    mbcx.flow_state.results()
}


pub struct MirBorrowckCtxtPreDataflow<'a, 'tcx: 'a, BD>
    where BD: BitDenotation, BD::Ctxt: 'a
{
    node_id: ast::NodeId,
    flow_state: DataflowAnalysis<'a, 'tcx, BD>,
    print_preflow_to: Option<String>,
    print_postflow_to: Option<String>,
}

#[allow(dead_code)]
pub struct MirBorrowckCtxt<'b, 'a: 'b, 'tcx: 'a> {
    bcx: &'b mut BorrowckCtxt<'a, 'tcx>,
    mir: &'b Mir<'tcx>,
    node_id: ast::NodeId,
    move_data: MoveData<'tcx>,
    flow_inits: DataflowResults<MaybeInitializedLvals<'a, 'tcx>>,
    flow_uninits: DataflowResults<MaybeUninitializedLvals<'a, 'tcx>>
}

impl<'b, 'a: 'b, 'tcx: 'a> MirBorrowckCtxt<'b, 'a, 'tcx> {
    fn process_basic_block(&mut self, bb: BasicBlock) {
        let BasicBlockData { ref statements, ref terminator, is_cleanup: _ } =
            self.mir[bb];
        for stmt in statements {
            self.process_statement(bb, stmt);
        }

        self.process_terminator(bb, terminator);
    }

    fn process_statement(&mut self, bb: BasicBlock, stmt: &Statement<'tcx>) {
        debug!("MirBorrowckCtxt::process_statement({:?}, {:?}", bb, stmt);
    }

    fn process_terminator(&mut self, bb: BasicBlock, term: &Option<Terminator<'tcx>>) {
        debug!("MirBorrowckCtxt::process_terminator({:?}, {:?})", bb, term);
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum DropFlagState {
    Present, // i.e. initialized
    Absent, // i.e. deinitialized or "moved"
}

impl DropFlagState {
    fn value(self) -> bool {
        match self {
            DropFlagState::Present => true,
            DropFlagState::Absent => false
        }
    }
}

fn move_path_children_matching<'tcx, F>(move_paths: &MovePathData<'tcx>,
                                        path: MovePathIndex,
                                        mut cond: F)
                                        -> Option<MovePathIndex>
    where F: FnMut(&repr::LvalueProjection<'tcx>) -> bool
{
    let mut next_child = move_paths[path].first_child;
    while let Some(child_index) = next_child {
        match move_paths[child_index].content {
            MovePathContent::Lvalue(repr::Lvalue::Projection(ref proj)) => {
                if cond(proj) {
                    return Some(child_index)
                }
            }
            _ => {}
        }
        next_child = move_paths[child_index].next_sibling;
    }

    None
}

/// When enumerating the child fragments of a path, don't recurse into
/// paths (1.) past arrays, slices, and pointers, nor (2.) into a type
/// that implements `Drop`.
///
/// Lvalues behind references or arrays are not tracked by elaboration
/// and are always assumed to be initialized when accessible. As
/// references and indexes can be reseated, trying to track them can
/// only lead to trouble.
///
/// Lvalues behind ADT's with a Drop impl are not tracked by
/// elaboration since they can never have a drop-flag state that
/// differs from that of the parent with the Drop impl.
///
/// In both cases, the contents can only be accessed if and only if
/// their parents are initialized. This implies for example that there
/// is no need to maintain separate drop flags to track such state.
///
/// FIXME: we have to do something for moving slice patterns.
fn lvalue_contents_drop_state_cannot_differ<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                      mir: &Mir<'tcx>,
                                                      lv: &repr::Lvalue<'tcx>) -> bool {
    let ty = mir.lvalue_ty(tcx, lv).to_ty(tcx);
    match ty.sty {
        ty::TyArray(..) | ty::TySlice(..) | ty::TyRef(..) | ty::TyRawPtr(..) => {
            debug!("lvalue_contents_drop_state_cannot_differ lv: {:?} ty: {:?} refd => false",
                   lv, ty);
            true
        }
        ty::TyStruct(def, _) | ty::TyEnum(def, _) if def.has_dtor() => {
            debug!("lvalue_contents_drop_state_cannot_differ lv: {:?} ty: {:?} Drop => false",
                   lv, ty);
            true
        }
        _ => {
            false
        }
    }
}

fn on_all_children_bits<'a, 'tcx, F>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &Mir<'tcx>,
    move_data: &MoveData<'tcx>,
    move_path_index: MovePathIndex,
    mut each_child: F)
    where F: FnMut(MovePathIndex)
{
    fn is_terminal_path<'a, 'tcx>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        mir: &Mir<'tcx>,
        move_data: &MoveData<'tcx>,
        path: MovePathIndex) -> bool
    {
        match move_data.move_paths[path].content {
            MovePathContent::Lvalue(ref lvalue) => {
                lvalue_contents_drop_state_cannot_differ(tcx, mir, lvalue)
            }
            _ => true
        }
    }

    fn on_all_children_bits<'a, 'tcx, F>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        mir: &Mir<'tcx>,
        move_data: &MoveData<'tcx>,
        move_path_index: MovePathIndex,
        each_child: &mut F)
        where F: FnMut(MovePathIndex)
    {
        each_child(move_path_index);

        if is_terminal_path(tcx, mir, move_data, move_path_index) {
            return
        }

        let mut next_child_index = move_data.move_paths[move_path_index].first_child;
        while let Some(child_index) = next_child_index {
            on_all_children_bits(tcx, mir, move_data, child_index, each_child);
            next_child_index = move_data.move_paths[child_index].next_sibling;
        }
    }
    on_all_children_bits(tcx, mir, move_data, move_path_index, &mut each_child);
}

fn drop_flag_effects_for_function_entry<'a, 'tcx, F>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &Mir<'tcx>,
    ctxt: &MoveDataParamEnv<'tcx>,
    mut callback: F)
    where F: FnMut(MovePathIndex, DropFlagState)
{
    let move_data = &ctxt.move_data;
    for (arg, _) in mir.arg_decls.iter_enumerated() {
        let lvalue = repr::Lvalue::Arg(arg);
        let move_path_index = move_data.rev_lookup.find(&lvalue);
        on_all_children_bits(tcx, mir, move_data,
                             move_path_index,
                             |moi| callback(moi, DropFlagState::Present));
    }
}

fn drop_flag_effects_for_location<'a, 'tcx, F>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &Mir<'tcx>,
    ctxt: &MoveDataParamEnv<'tcx>,
    loc: Location,
    mut callback: F)
    where F: FnMut(MovePathIndex, DropFlagState)
{
    let move_data = &ctxt.move_data;
    let param_env = &ctxt.param_env;
    debug!("drop_flag_effects_for_location({:?})", loc);

    // first, move out of the RHS
    for mi in &move_data.loc_map[loc] {
        let path = mi.move_path_index(move_data);
        debug!("moving out of path {:?}", move_data.move_paths[path]);

        // don't move out of non-Copy things
        if let MovePathContent::Lvalue(ref lvalue) = move_data.move_paths[path].content {
            let ty = mir.lvalue_ty(tcx, lvalue).to_ty(tcx);
            if !ty.moves_by_default(tcx, param_env, DUMMY_SP) {
                continue;
            }
        }

        on_all_children_bits(tcx, mir, move_data,
                             path,
                             |moi| callback(moi, DropFlagState::Absent))
    }

    let block = &mir[loc.block];
    match block.statements.get(loc.index) {
        Some(stmt) => match stmt.kind {
            repr::StatementKind::Assign(ref lvalue, _) => {
                debug!("drop_flag_effects: assignment {:?}", stmt);
                 on_all_children_bits(tcx, mir, move_data,
                                     move_data.rev_lookup.find(lvalue),
                                     |moi| callback(moi, DropFlagState::Present))
            }
        },
        None => {
            debug!("drop_flag_effects: replace {:?}", block.terminator());
            match block.terminator().kind {
                repr::TerminatorKind::DropAndReplace { ref location, .. } => {
                    on_all_children_bits(tcx, mir, move_data,
                                         move_data.rev_lookup.find(location),
                                         |moi| callback(moi, DropFlagState::Present))
                }
                _ => {
                    // other terminators do not contain move-ins
                }
            }
        }
    }
}
