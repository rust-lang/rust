use rustc_index::vec::Idx;
use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_target::abi::VariantIdx;

use std::convert::TryFrom;
use std::iter::TrustedLen;

/// Expand `lhs = Rvalue::Aggregate(kind, operands)` into assignments to the fields.
///
/// Produces something like
///
/// (lhs as Variant).field0 = arg0;     // We only have a downcast if this is an enum
/// (lhs as Variant).field1 = arg1;
/// discriminant(lhs) = variant_index;  // If lhs is an enum or generator.
pub fn expand_aggregate<'tcx>(
    mut lhs: Place<'tcx>,
    operands: impl Iterator<Item = (Operand<'tcx>, Ty<'tcx>)> + TrustedLen,
    kind: AggregateKind<'tcx>,
    source_info: SourceInfo,
    tcx: TyCtxt<'tcx>,
) -> impl Iterator<Item = Statement<'tcx>> + TrustedLen {
    let mut set_discriminant = None;
    let active_field_index = match kind {
        AggregateKind::Adt(adt_did, variant_index, _, _, active_field_index) => {
            let adt_def = tcx.adt_def(adt_did);
            if adt_def.is_enum() {
                set_discriminant = Some(Statement {
                    kind: StatementKind::SetDiscriminant { place: Box::new(lhs), variant_index },
                    source_info,
                });
                lhs = tcx.mk_place_downcast(lhs, adt_def, variant_index);
            }
            active_field_index
        }
        AggregateKind::Generator(..) => {
            // Right now we only support initializing generators to
            // variant 0 (Unresumed).
            let variant_index = VariantIdx::new(0);
            set_discriminant = Some(Statement {
                kind: StatementKind::SetDiscriminant { place: Box::new(lhs), variant_index },
                source_info,
            });

            // Operands are upvars stored on the base place, so no
            // downcast is necessary.

            None
        }
        _ => None,
    };

    operands
        .enumerate()
        .map(move |(i, (op, ty))| {
            let lhs_field = if let AggregateKind::Array(_) = kind {
                let offset = u64::try_from(i).unwrap();
                tcx.mk_place_elem(
                    lhs,
                    ProjectionElem::ConstantIndex {
                        offset,
                        min_length: offset + 1,
                        from_end: false,
                    },
                )
            } else {
                let field = Field::new(active_field_index.unwrap_or(i));
                tcx.mk_place_field(lhs, field, ty)
            };
            Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((lhs_field, Rvalue::Use(op)))),
            }
        })
        .chain(set_discriminant)
}
