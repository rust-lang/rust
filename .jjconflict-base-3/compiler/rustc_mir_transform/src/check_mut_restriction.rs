use rustc_hir::def::{CtorOf, DefKind};
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Span;

use crate::diagnostics;

pub(super) struct CheckMutRestriction;

impl<'tcx> crate::MirLint<'tcx> for CheckMutRestriction {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        if body.tainted_by_errors.is_some() {
            return;
        }
        let mut checker = MutRestrictionChecker { body, tcx, mutating_span: body.span };
        checker.visit_body(body);
    }
}

struct MutRestrictionChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    mutating_span: Span,
}

impl<'tcx> Visitor<'tcx> for MutRestrictionChecker<'_, 'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.mutating_span = terminator.source_info.span;
        self.super_terminator(terminator, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        self.mutating_span = statement.source_info.span;
        self.super_statement(statement, location);
    }

    // Tuple constructors used as values can bypass field mut restrictions if not checked here.
    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, location: Location) {
        if let ty::FnDef(def_id, _) = *constant.const_.ty().kind()
            && let DefKind::Ctor(ctor_of, _) = self.tcx.def_kind(def_id)
        {
            let body_did = self.body.source.instance.def_id();
            let adt_did = match ctor_of {
                CtorOf::Struct => self.tcx.parent(def_id),
                CtorOf::Variant => self.tcx.parent(self.tcx.parent(def_id)),
            };
            let adt = self.tcx.adt_def(adt_did);
            let variant = match ctor_of {
                CtorOf::Struct => adt.non_enum_variant(),
                CtorOf::Variant => adt.variant_with_ctor_id(def_id),
            };

            let mut_restriction =
                variant.fields.iter().fold(ty::RestrictionKind::Unrestricted, |acc, field| {
                    acc.stricter_of(field.mut_restriction, self.tcx)
                });
            if !mut_restriction.is_allowed_in(body_did, self.tcx) {
                self.tcx.dcx().emit_err(diagnostics::ConstructionOfTyWithMutRestrictedField {
                    construction_span: constant.span,
                    restriction_span: mut_restriction.expect_span(),
                    name: variant.name,
                    descr: adt.variant_descr(),
                    restriction_path: mut_restriction.restriction_path(self.tcx),
                });
            }
        }

        self.super_const_operand(constant, location);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if context.is_mutating_use() {
            let body_did = self.body.source.instance.def_id();

            for (place_base, elem) in place.iter_projections() {
                // Even when the field is an array or slice and is accessed by index,
                // as in `foo.array[0]`, the projection chain still contains a field
                // projection. Therefore, it is sufficient to check for field projections.
                let ProjectionElem::Field(field_idx, _field_ty) = elem else {
                    continue;
                };

                let base_ty = place_base.ty(self.body, self.tcx);

                // Field projections are also used for tuples, closures, and coroutines,
                // but mutability restrictions only apply to ADT fields.
                // Mutating an ADT field through a captured value still produces a
                // separate field projection whose base type is that ADT.
                // Therefore, it is sufficient to check for ADT base types.
                // Generic arguments do not affect the field's restriction, so we ignore them.
                let ty::Adt(adt_def, _args) = *base_ty.ty.kind() else {
                    continue;
                };

                let variant_def = if let Some(idx) = base_ty.variant_index {
                    assert!(adt_def.is_enum());
                    adt_def.variant(idx)
                } else {
                    adt_def.non_enum_variant()
                };

                let field_def: &ty::FieldDef = &variant_def.fields[field_idx];
                let mut_restriction = field_def.mut_restriction;

                if !mut_restriction.is_allowed_in(body_did, self.tcx) {
                    self.tcx.dcx().emit_err(diagnostics::MutOfRestrictedField {
                        mut_span: self.mutating_span,
                        restriction_span: mut_restriction.expect_span(),
                        name: field_def.name,
                        restriction_path: mut_restriction.restriction_path(self.tcx),
                    });
                }
            }
        }

        self.super_place(place, context, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::Aggregate(aggr, _) = rvalue
            && let AggregateKind::Adt(adt_did, variant_idx, _args, _user_ty, active_field) = &**aggr
        {
            let body_did = self.body.source.instance.def_id();
            let adt = self.tcx.adt_def(*adt_did);
            let variant = &adt.variants()[*variant_idx];

            if let Some(field_idx) = active_field {
                // union
                let field_def = &variant.fields[*field_idx];
                let mut_restriction = field_def.mut_restriction;
                if !mut_restriction.is_allowed_in(body_did, self.tcx) {
                    self.tcx.dcx().emit_err(diagnostics::ConstructionOfTyWithMutRestrictedField {
                        construction_span: self.mutating_span,
                        restriction_span: mut_restriction.expect_span(),
                        name: variant.name,
                        descr: adt.variant_descr(),
                        restriction_path: mut_restriction.restriction_path(self.tcx),
                    });
                }
            } else {
                // struct / enum variant
                let mut_restriction =
                    variant.fields.iter().fold(ty::RestrictionKind::Unrestricted, |acc, field| {
                        acc.stricter_of(field.mut_restriction, self.tcx)
                    });
                if !mut_restriction.is_allowed_in(body_did, self.tcx) {
                    self.tcx.dcx().emit_err(diagnostics::ConstructionOfTyWithMutRestrictedField {
                        construction_span: self.mutating_span,
                        restriction_span: mut_restriction.expect_span(),
                        name: variant.name,
                        descr: adt.variant_descr(),
                        restriction_path: mut_restriction.restriction_path(self.tcx),
                    });
                }
            }
        }

        self.super_rvalue(rvalue, location);
    }
}
