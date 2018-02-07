// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc_data_structures::indexed_vec::Idx;
use transform::{MirPass, MirSource};

pub struct Deaggregator;

impl MirPass for Deaggregator {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>) {
        // Don't run on constant MIR, because trans might not be able to
        // evaluate the modified MIR.
        // FIXME(eddyb) Remove check after miri is merged.
        let id = tcx.hir.as_local_node_id(source.def_id).unwrap();
        match (tcx.hir.body_owner_kind(id), source.promoted) {
            (_, Some(_)) |
            (hir::BodyOwnerKind::Const, _) |
            (hir::BodyOwnerKind::Static(_), _) => return,

            (hir::BodyOwnerKind::Fn, _) => {
                if tcx.is_const_fn(source.def_id) {
                    // Don't run on const functions, as, again, trans might not be able to evaluate
                    // the optimized IR.
                    return
                }
            }
        }

        let can_deaggregate = |statement: &Statement| {
            if let StatementKind::Assign(_, ref rhs) = statement.kind {
                if let Rvalue::Aggregate(..) = *rhs {
                    return true;
                }
            }

            false
        };

        let (basic_blocks, local_decls) = mir.basic_blocks_and_local_decls_mut();
        for bb in basic_blocks {
            let mut start = 0;
            while let Some(i) = bb.statements[start..].iter().position(&can_deaggregate) {
                let i = start + i;

                // FIXME(eddyb) this is probably more expensive than it should be.
                // Ideally we'd move the block's statements all at once.
                let suffix_stmts = bb.statements.split_off(i + 1);
                let orig_stmt = bb.statements.pop().unwrap();
                let source_info = orig_stmt.source_info;
                let (mut lhs, kind, operands) = match orig_stmt.kind {
                    StatementKind::Assign(lhs, Rvalue::Aggregate(kind, operands))
                        => (lhs, kind, operands),
                    _ => bug!()
                };

                let mut set_discriminant = None;
                let active_field_index = match *kind {
                    AggregateKind::Adt(adt_def, variant_index, _, active_field_index) => {
                        if adt_def.is_enum() {
                            set_discriminant = Some(Statement {
                                kind: StatementKind::SetDiscriminant {
                                    place: lhs.clone(),
                                    variant_index,
                                },
                                source_info,
                            });
                            lhs = lhs.downcast(adt_def, variant_index);
                        }
                        active_field_index
                    }
                    _ => None
                };

                let new_total_count = bb.statements.len() +
                    operands.len() +
                    (set_discriminant.is_some() as usize) +
                    suffix_stmts.len();
                bb.statements.reserve(new_total_count);

                for (j, op) in operands.into_iter().enumerate() {
                    let lhs_field = if let AggregateKind::Array(_) = *kind {
                        // FIXME(eddyb) `offset` should be u64.
                        let offset = j as u32;
                        assert_eq!(offset as usize, j);
                        lhs.clone().elem(ProjectionElem::ConstantIndex {
                            offset,
                            // FIXME(eddyb) `min_length` doesn't appear to be used.
                            min_length: offset + 1,
                            from_end: false
                        })
                    } else {
                        let ty = op.ty(local_decls, tcx);
                        let field = Field::new(active_field_index.unwrap_or(j));
                        lhs.clone().field(field, ty)
                    };
                    bb.statements.push(Statement {
                        source_info,
                        kind: StatementKind::Assign(lhs_field, Rvalue::Use(op)),
                    });
                }

                // If the aggregate was an enum, we need to set the discriminant.
                bb.statements.extend(set_discriminant);

                start = bb.statements.len();
                bb.statements.extend(suffix_stmts);
            }
        }
    }
}
