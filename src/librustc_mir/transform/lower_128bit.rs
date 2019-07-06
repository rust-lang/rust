//! Replaces 128-bit operators with lang item calls

use rustc::hir::def_id::DefId;
use rustc::middle::lang_items::LangItem;
use rustc::mir::*;
use rustc::ty::{self, List, Ty, TyCtxt};
use rustc_data_structures::indexed_vec::{Idx};
use crate::transform::{MirPass, MirSource};

pub struct Lower128Bit;

impl MirPass for Lower128Bit {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, _src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let debugging_override = tcx.sess.opts.debugging_opts.lower_128bit_ops;
        let target_default = tcx.sess.host.options.i128_lowering;
        if !debugging_override.unwrap_or(target_default) {
            return
        }

        self.lower_128bit_ops(tcx, body);
}
}

impl Lower128Bit {
    fn lower_128bit_ops<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut new_blocks = Vec::new();
        let cur_len = body.basic_blocks().len();

        let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();
        for block in basic_blocks.iter_mut() {
            for i in (0..block.statements.len()).rev() {
                let (lang_item, rhs_kind) =
                    if let Some((lang_item, rhs_kind)) =
                        lower_to(&block.statements[i], local_decls, tcx)
                    {
                        (lang_item, rhs_kind)
                    } else {
                        continue;
                    };

                let rhs_override_ty = rhs_kind.ty(tcx);
                let cast_local =
                    match rhs_override_ty {
                        None => None,
                        Some(ty) => {
                            let local_decl = LocalDecl::new_internal(
                                ty, block.statements[i].source_info.span);
                            Some(local_decls.push(local_decl))
                        },
                    };

                let storage_dead = cast_local.map(|local| {
                    Statement {
                        source_info: block.statements[i].source_info,
                        kind: StatementKind::StorageDead(local),
                    }
                });
                let after_call = BasicBlockData {
                    statements: storage_dead.into_iter()
                        .chain(block.statements.drain((i+1)..)).collect(),
                    is_cleanup: block.is_cleanup,
                    terminator: block.terminator.take(),
                };

                let bin_statement = block.statements.pop().unwrap();
                let source_info = bin_statement.source_info;
                let (place, lhs, mut rhs) = match bin_statement.kind {
                    StatementKind::Assign(place, box rvalue) => {
                        match rvalue {
                            Rvalue::BinaryOp(_, lhs, rhs)
                            | Rvalue::CheckedBinaryOp(_, lhs, rhs) => (place, lhs, rhs),
                            _ => bug!(),
                        }
                    }
                    _ => bug!()
                };

                if let Some(local) = cast_local {
                    block.statements.push(Statement {
                        source_info: source_info,
                        kind: StatementKind::StorageLive(local),
                    });
                    block.statements.push(Statement {
                        source_info: source_info,
                        kind: StatementKind::Assign(
                            Place::from(local),
                            box Rvalue::Cast(
                                CastKind::Misc,
                                rhs,
                                rhs_override_ty.unwrap())),
                    });
                    rhs = Operand::Move(Place::from(local));
                }

                let call_did = check_lang_item_type(
                    lang_item, &place, &lhs, &rhs, local_decls, tcx);

                let bb = BasicBlock::new(cur_len + new_blocks.len());
                new_blocks.push(after_call);

                block.terminator =
                    Some(Terminator {
                        source_info,
                        kind: TerminatorKind::Call {
                            func: Operand::function_handle(tcx, call_did,
                                List::empty(), source_info.span),
                            args: vec![lhs, rhs],
                            destination: Some((place, bb)),
                            cleanup: None,
                            from_hir_call: false,
                        },
                    });
            }
        }

        basic_blocks.extend(new_blocks);
    }
}

fn check_lang_item_type<'tcx, D>(
    lang_item: LangItem,
    place: &Place<'tcx>,
    lhs: &Operand<'tcx>,
    rhs: &Operand<'tcx>,
    local_decls: &D,
    tcx: TyCtxt<'tcx>,
) -> DefId
where
    D: HasLocalDecls<'tcx>,
{
    let did = tcx.require_lang_item(lang_item);
    let poly_sig = tcx.fn_sig(did);
    let sig = poly_sig.no_bound_vars().unwrap();
    let lhs_ty = lhs.ty(local_decls, tcx);
    let rhs_ty = rhs.ty(local_decls, tcx);
    let place_ty = place.ty(local_decls, tcx).ty;
    let expected = [lhs_ty, rhs_ty, place_ty];
    assert_eq!(sig.inputs_and_output[..], expected,
        "lang item `{}`", tcx.def_path_str(did));
    did
}

fn lower_to<'tcx, D>(
    statement: &Statement<'tcx>,
    local_decls: &D,
    tcx: TyCtxt<'tcx>,
) -> Option<(LangItem, RhsKind)>
where
    D: HasLocalDecls<'tcx>,
{
    match statement.kind {
        StatementKind::Assign(_, box Rvalue::BinaryOp(bin_op, ref lhs, _)) => {
            let ty = lhs.ty(local_decls, tcx);
            if let Some(is_signed) = sign_of_128bit(ty) {
                return item_for_op(bin_op, is_signed);
            }
        },
        StatementKind::Assign(_, box Rvalue::CheckedBinaryOp(bin_op, ref lhs, _)) => {
            let ty = lhs.ty(local_decls, tcx);
            if let Some(is_signed) = sign_of_128bit(ty) {
                return item_for_checked_op(bin_op, is_signed);
            }
        },
        _ => {},
    }
    None
}

#[derive(Copy, Clone)]
enum RhsKind {
    Unchanged,
    ForceU128,
    ForceU32,
}

impl RhsKind {
    fn ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Option<Ty<'tcx>> {
        match *self {
            RhsKind::Unchanged => None,
            RhsKind::ForceU128 => Some(tcx.types.u128),
            RhsKind::ForceU32 => Some(tcx.types.u32),
        }
    }
}

fn sign_of_128bit(ty: Ty<'_>) -> Option<bool> {
    match ty.sty {
        ty::Int(syntax::ast::IntTy::I128) => Some(true),
        ty::Uint(syntax::ast::UintTy::U128) => Some(false),
        _ => None,
    }
}

fn item_for_op(bin_op: BinOp, is_signed: bool) -> Option<(LangItem, RhsKind)> {
    let i = match (bin_op, is_signed) {
        (BinOp::Add, true) => (LangItem::I128AddFnLangItem, RhsKind::Unchanged),
        (BinOp::Add, false) => (LangItem::U128AddFnLangItem, RhsKind::Unchanged),
        (BinOp::Sub, true) => (LangItem::I128SubFnLangItem, RhsKind::Unchanged),
        (BinOp::Sub, false) => (LangItem::U128SubFnLangItem, RhsKind::Unchanged),
        (BinOp::Mul, true) => (LangItem::I128MulFnLangItem, RhsKind::Unchanged),
        (BinOp::Mul, false) => (LangItem::U128MulFnLangItem, RhsKind::Unchanged),
        (BinOp::Div, true) => (LangItem::I128DivFnLangItem, RhsKind::Unchanged),
        (BinOp::Div, false) => (LangItem::U128DivFnLangItem, RhsKind::Unchanged),
        (BinOp::Rem, true) => (LangItem::I128RemFnLangItem, RhsKind::Unchanged),
        (BinOp::Rem, false) => (LangItem::U128RemFnLangItem, RhsKind::Unchanged),
        (BinOp::Shl, true) => (LangItem::I128ShlFnLangItem, RhsKind::ForceU32),
        (BinOp::Shl, false) => (LangItem::U128ShlFnLangItem, RhsKind::ForceU32),
        (BinOp::Shr, true) => (LangItem::I128ShrFnLangItem, RhsKind::ForceU32),
        (BinOp::Shr, false) => (LangItem::U128ShrFnLangItem, RhsKind::ForceU32),
        _ => return None,
    };
    Some(i)
}

fn item_for_checked_op(bin_op: BinOp, is_signed: bool) -> Option<(LangItem, RhsKind)> {
    let i = match (bin_op, is_signed) {
        (BinOp::Add, true) => (LangItem::I128AddoFnLangItem, RhsKind::Unchanged),
        (BinOp::Add, false) => (LangItem::U128AddoFnLangItem, RhsKind::Unchanged),
        (BinOp::Sub, true) => (LangItem::I128SuboFnLangItem, RhsKind::Unchanged),
        (BinOp::Sub, false) => (LangItem::U128SuboFnLangItem, RhsKind::Unchanged),
        (BinOp::Mul, true) => (LangItem::I128MuloFnLangItem, RhsKind::Unchanged),
        (BinOp::Mul, false) => (LangItem::U128MuloFnLangItem, RhsKind::Unchanged),
        (BinOp::Shl, true) => (LangItem::I128ShloFnLangItem, RhsKind::ForceU128),
        (BinOp::Shl, false) => (LangItem::U128ShloFnLangItem, RhsKind::ForceU128),
        (BinOp::Shr, true) => (LangItem::I128ShroFnLangItem, RhsKind::ForceU128),
        (BinOp::Shr, false) => (LangItem::U128ShroFnLangItem, RhsKind::ForceU128),
        _ => bug!("That should be all the checked ones?"),
    };
    Some(i)
}
