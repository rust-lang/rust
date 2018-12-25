use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc_data_structures::indexed_vec::Idx;
use transform::{MirPass, MirSource};

pub struct Deaggregator;

impl MirPass for Deaggregator {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _source: MirSource,
                          mir: &mut Mir<'tcx>) {
        let (basic_blocks, local_decls) = mir.basic_blocks_and_local_decls_mut();
        let local_decls = &*local_decls;
        for bb in basic_blocks {
            bb.expand_statements(|stmt| {
                // FIXME(eddyb) don't match twice on `stmt.kind` (post-NLL).
                if let StatementKind::Assign(_, ref rhs) = stmt.kind {
                    if let Rvalue::Aggregate(ref kind, _) = **rhs {
                        // FIXME(#48193) Deaggregate arrays when it's cheaper to do so.
                        if let AggregateKind::Array(_) = **kind {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }

                let stmt = stmt.replace_nop();
                let source_info = stmt.source_info;
                let (mut lhs, kind, operands) = match stmt.kind {
                    StatementKind::Assign(lhs, box rvalue) => {
                        match rvalue {
                            Rvalue::Aggregate(kind, operands) => (lhs, kind, operands),
                            _ => bug!()
                        }
                    }
                    _ => bug!()
                };

                let mut set_discriminant = None;
                let active_field_index = match *kind {
                    AggregateKind::Adt(adt_def, variant_index, _, _, active_field_index) => {
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

                Some(operands.into_iter().enumerate().map(move |(i, op)| {
                    let lhs_field = if let AggregateKind::Array(_) = *kind {
                        // FIXME(eddyb) `offset` should be u64.
                        let offset = i as u32;
                        assert_eq!(offset as usize, i);
                        lhs.clone().elem(ProjectionElem::ConstantIndex {
                            offset,
                            // FIXME(eddyb) `min_length` doesn't appear to be used.
                            min_length: offset + 1,
                            from_end: false
                        })
                    } else {
                        let ty = op.ty(local_decls, tcx);
                        let field = Field::new(active_field_index.unwrap_or(i));
                        lhs.clone().field(field, ty)
                    };
                    Statement {
                        source_info,
                        kind: StatementKind::Assign(lhs_field, box Rvalue::Use(op)),
                    }
                }).chain(set_discriminant))
            });
        }
    }
}
