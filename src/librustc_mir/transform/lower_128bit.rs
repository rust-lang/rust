// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Replaces 128-bit operators with lang item calls

use rustc::hir::def_id::DefId;
use rustc::middle::lang_items::LangItem;
use rustc::mir::*;
use rustc::ty::{Slice, Ty, TyCtxt, TypeVariants};
use rustc_data_structures::indexed_vec::{Idx};
use transform::{MirPass, MirSource};
use syntax;

pub struct Lower128Bit;

impl MirPass for Lower128Bit {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _src: MirSource,
                          mir: &mut Mir<'tcx>) {
        if !tcx.sess.opts.debugging_opts.lower_128bit_ops {
            return
        }

        self.lower_128bit_ops(tcx, mir);
    }
}

impl Lower128Bit {
    fn lower_128bit_ops<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, mir: &mut Mir<'tcx>) {
        let mut new_blocks = Vec::new();
        let cur_len = mir.basic_blocks().len();

        let (basic_blocks, local_decls) = mir.basic_blocks_and_local_decls_mut();
        for block in basic_blocks.iter_mut() {
            for i in (0..block.statements.len()).rev() {
                let lang_item =
                    if let Some(lang_item) = lower_to(&block.statements[i], local_decls, tcx) {
                        lang_item
                    } else {
                        continue;
                    };

                let after_call = BasicBlockData {
                    statements: block.statements.drain((i+1)..).collect(),
                    is_cleanup: block.is_cleanup,
                    terminator: block.terminator.take(),
                };

                let bin_statement = block.statements.pop().unwrap();
                let (source_info, lvalue, lhs, rhs) = match bin_statement {
                    Statement {
                        source_info,
                        kind: StatementKind::Assign(
                            lvalue,
                            Rvalue::BinaryOp(_, lhs, rhs))
                    } => (source_info, lvalue, lhs, rhs),
                    Statement {
                        source_info,
                        kind: StatementKind::Assign(
                            lvalue,
                            Rvalue::CheckedBinaryOp(_, lhs, rhs))
                    } => (source_info, lvalue, lhs, rhs),
                    _ => bug!("Statement doesn't match pattern any more?"),
                };

                let call_did = check_lang_item_type(
                    lang_item, &lvalue, &lhs, &rhs, local_decls, tcx);

                let bb = BasicBlock::new(cur_len + new_blocks.len());
                new_blocks.push(after_call);

                block.terminator =
                    Some(Terminator {
                        source_info,
                        kind: TerminatorKind::Call {
                            func: Operand::function_handle(tcx, call_did,
                                Slice::empty(), source_info.span),
                            args: vec![lhs, rhs],
                            destination: Some((lvalue, bb)),
                            cleanup: None,
                        },
                    });
            }
        }

        basic_blocks.extend(new_blocks);
    }
}

fn check_lang_item_type<'a, 'tcx, D>(
    lang_item: LangItem,
    lvalue: &Lvalue<'tcx>,
    lhs: &Operand<'tcx>,
    rhs: &Operand<'tcx>,
    local_decls: &D,
    tcx: TyCtxt<'a, 'tcx, 'tcx>)
-> DefId
    where D: HasLocalDecls<'tcx>
{
    let did = tcx.require_lang_item(lang_item);
    let poly_sig = tcx.fn_sig(did);
    let sig = tcx.no_late_bound_regions(&poly_sig).unwrap();
    let lhs_ty = lhs.ty(local_decls, tcx);
    let rhs_ty = rhs.ty(local_decls, tcx);
    let lvalue_ty = lvalue.ty(local_decls, tcx).to_ty(tcx);
    let expected = [lhs_ty, rhs_ty, lvalue_ty];
    assert_eq!(sig.inputs_and_output[..], expected,
        "lang item {}", tcx.def_symbol_name(did));
    did
}

fn lower_to<'a, 'tcx, D>(statement: &Statement<'tcx>, local_decls: &D, tcx: TyCtxt<'a, 'tcx, 'tcx>)
    -> Option<LangItem>
    where D: HasLocalDecls<'tcx>
{
    match statement.kind {
        StatementKind::Assign(_, Rvalue::BinaryOp(bin_op, ref lhs, _)) => {
            let ty = lhs.ty(local_decls, tcx);
            if let Some(is_signed) = sign_of_128bit(ty) {
                return item_for_op(bin_op, is_signed);
            }
        },
        StatementKind::Assign(_, Rvalue::CheckedBinaryOp(bin_op, ref lhs, _)) => {
            let ty = lhs.ty(local_decls, tcx);
            if let Some(is_signed) = sign_of_128bit(ty) {
                return item_for_checked_op(bin_op, is_signed);
            }
        },
        _ => {},
    }
    None
}

fn sign_of_128bit(ty: Ty) -> Option<bool> {
    match ty.sty {
        TypeVariants::TyInt(syntax::ast::IntTy::I128) => Some(true),
        TypeVariants::TyUint(syntax::ast::UintTy::U128) => Some(false),
        _ => None,
    }
}

fn item_for_op(bin_op: BinOp, is_signed: bool) -> Option<LangItem> {
    let i = match (bin_op, is_signed) {
        (BinOp::Add, true) => LangItem::I128AddFnLangItem,
        (BinOp::Add, false) => LangItem::U128AddFnLangItem,
        (BinOp::Sub, true) => LangItem::I128SubFnLangItem,
        (BinOp::Sub, false) => LangItem::U128SubFnLangItem,
        (BinOp::Mul, true) => LangItem::I128MulFnLangItem,
        (BinOp::Mul, false) => LangItem::U128MulFnLangItem,
        (BinOp::Div, true) => LangItem::I128DivFnLangItem,
        (BinOp::Div, false) => LangItem::U128DivFnLangItem,
        (BinOp::Rem, true) => LangItem::I128RemFnLangItem,
        (BinOp::Rem, false) => LangItem::U128RemFnLangItem,
        (BinOp::Shl, true) => LangItem::I128ShlFnLangItem,
        (BinOp::Shl, false) => LangItem::U128ShlFnLangItem,
        (BinOp::Shr, true) => LangItem::I128ShrFnLangItem,
        (BinOp::Shr, false) => LangItem::U128ShrFnLangItem,
        _ => return None,
    };
    Some(i)
}

fn item_for_checked_op(bin_op: BinOp, is_signed: bool) -> Option<LangItem> {
    let i = match (bin_op, is_signed) {
        (BinOp::Add, true) => LangItem::I128AddoFnLangItem,
        (BinOp::Add, false) => LangItem::U128AddoFnLangItem,
        (BinOp::Sub, true) => LangItem::I128SuboFnLangItem,
        (BinOp::Sub, false) => LangItem::U128SuboFnLangItem,
        (BinOp::Mul, true) => LangItem::I128MuloFnLangItem,
        (BinOp::Mul, false) => LangItem::U128MuloFnLangItem,
        (BinOp::Shl, true) => LangItem::I128ShloFnLangItem,
        (BinOp::Shl, false) => LangItem::U128ShloFnLangItem,
        (BinOp::Shr, true) => LangItem::I128ShroFnLangItem,
        (BinOp::Shr, false) => LangItem::U128ShroFnLangItem,
        _ => bug!("That should be all the checked ones?"),
    };
    Some(i)
}