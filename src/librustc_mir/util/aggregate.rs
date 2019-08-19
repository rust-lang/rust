use rustc::mir::*;
use rustc::ty::Ty;
use rustc::ty::layout::VariantIdx;
use rustc_data_structures::indexed_vec::Idx;

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
    operands: impl Iterator<Item=(Operand<'tcx>, Ty<'tcx>)> + TrustedLen,
    kind: AggregateKind<'tcx>,
    source_info: SourceInfo,
) -> impl Iterator<Item=Statement<'tcx>> + TrustedLen {
    let mut set_discriminant = None;
    let active_field_index = match kind {
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
        AggregateKind::Generator(..) => {
            // Right now we only support initializing generators to
            // variant 0 (Unresumed).
            let variant_index = VariantIdx::new(0);
            set_discriminant = Some(Statement {
                kind: StatementKind::SetDiscriminant {
                    place: lhs.clone(),
                    variant_index,
                },
                source_info,
            });

            // Operands are upvars stored on the base place, so no
            // downcast is necessary.

            None
        }
        _ => None
    };

    operands.into_iter().enumerate().map(move |(i, (op, ty))| {
        let lhs_field = if let AggregateKind::Array(_) = kind {
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
            let field = Field::new(active_field_index.unwrap_or(i));
            lhs.clone().field(field, ty)
        };
        Statement {
            source_info,
            kind: StatementKind::Assign(lhs_field, box Rvalue::Use(op)),
        }
    }).chain(set_discriminant)
}
