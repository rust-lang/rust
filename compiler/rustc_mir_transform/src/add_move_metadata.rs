//! This module provides a pass that modifies spans for statements with move operands,
//! adding information about moved types and their sizes to the filename.

use rustc_middle::mir::{Body, Operand, Rvalue, StatementKind};
use rustc_middle::ty::{Ty, TyCtxt, TypingEnv};
use rustc_span::{FileName, Span};
use tracing::debug;

pub(super) struct AddMoveMetadata;

impl<'tcx> crate::MirPass<'tcx> for AddMoveMetadata {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for basic_block in body.basic_blocks.as_mut() {
            for statement in basic_block.statements.iter_mut() {
                if let StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    if rvalue_has_move(rvalue) {
                        let moved_types =
                            get_moved_types_from_rvalue(rvalue, &body.local_decls, tcx);
                        let mut moved_types_with_sizes: Vec<_> = moved_types
                            .iter()
                            .map(|ty| {
                                let size = tcx
                                    .layout_of(TypingEnv::fully_monomorphized().as_query_input(*ty))
                                    .map(|layout| layout.size.bytes())
                                    .unwrap_or(0);
                                (*ty, size)
                            })
                            .collect();
                        moved_types_with_sizes.sort_by(|a, b| b.1.cmp(&a.1));
                        let old_span = statement.source_info.span;
                        statement.source_info.span = modify_span_filename(
                            statement.source_info.span,
                            &moved_types_with_sizes,
                            tcx,
                        );
                        for (ty, size) in &moved_types_with_sizes {
                            debug!("Moved type: {} (size: {} bytes)", ty, size);
                        }
                        debug!(
                            "Changed span from {:?} to {:?}, place: {:?}, rvalue: {:?}",
                            old_span, statement.source_info.span, place, rvalue
                        );
                    }
                }
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

fn rvalue_has_move(rvalue: &Rvalue<'_>) -> bool {
    match rvalue {
        Rvalue::Use(operand)
        | Rvalue::Repeat(operand, _)
        | Rvalue::Cast(_, operand, _)
        | Rvalue::UnaryOp(_, operand) => matches!(operand, Operand::Move(_)),
        Rvalue::BinaryOp(_, box (left, right)) => {
            matches!(left, Operand::Move(_)) || matches!(right, Operand::Move(_))
        }
        Rvalue::Aggregate(_, operands) => operands.iter().any(|op| matches!(op, Operand::Move(_))),
        _ => false,
    }
}

fn get_moved_types_from_rvalue<'tcx>(
    rvalue: &Rvalue<'tcx>,
    local_decls: &rustc_middle::mir::LocalDecls<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> Vec<Ty<'tcx>> {
    let mut types = Vec::new();
    match rvalue {
        Rvalue::Use(Operand::Move(place))
        | Rvalue::Repeat(Operand::Move(place), _)
        | Rvalue::Cast(_, Operand::Move(place), _)
        | Rvalue::UnaryOp(_, Operand::Move(place)) => {
            types.push(place.ty(local_decls, tcx).ty);
        }
        Rvalue::BinaryOp(_, box (left, right)) => {
            if let Operand::Move(place) = left {
                types.push(place.ty(local_decls, tcx).ty);
            }
            if let Operand::Move(place) = right {
                types.push(place.ty(local_decls, tcx).ty);
            }
        }
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                if let Operand::Move(place) = operand {
                    types.push(place.ty(local_decls, tcx).ty);
                }
            }
        }
        _ => {}
    }
    types
}

fn modify_span_filename<'tcx>(
    span: Span,
    moved_types_with_sizes: &[(Ty<'tcx>, u64)],
    tcx: TyCtxt<'tcx>,
) -> Span {
    let source_map = tcx.sess.source_map();
    let source_file = source_map.lookup_source_file(span.lo());
    let filename = source_file.name.prefer_local().to_string();
    let types_str = moved_types_with_sizes
        .iter()
        .map(|(ty, size)| format!("{}({} bytes)", ty, size))
        .collect::<Vec<_>>()
        .join(", ");
    let new_filename = format!("{} Moved:[{}] Line:", filename, types_str);

    let src = source_file.src.as_ref().map(|s| s.as_str()).unwrap_or("").to_string();
    let new_source_file = source_map.new_source_file(
        FileName::Real(rustc_span::RealFileName::LocalPath(std::path::PathBuf::from(new_filename))),
        src,
    );

    let offset = span.lo() - source_file.start_pos;
    Span::new(
        new_source_file.start_pos + offset,
        new_source_file.start_pos + offset + (span.hi() - span.lo()),
        span.ctxt(),
        span.parent(),
    )
}
