use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt};
use rustc_target::abi::VariantIdx;

use crate::util;

const INSTR_COST: usize = 5;
const CALL_PENALTY: usize = 25;
const LANDINGPAD_PENALTY: usize = 50;
const RESUME_PENALTY: usize = 45;

const UNKNOWN_SIZE_COST: usize = 10;

/// Verify that the callee body is compatible with the caller.
///
/// This visitor mostly computes the inlining cost,
/// but also needs to verify that types match because of normalization failure.
#[derive(Clone)]
pub(crate) struct CostChecker<'b, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    cost: usize,
    callee_body: &'b Body<'tcx>,
    instance: Option<ty::Instance<'tcx>>,
    validation: Result<(), &'static str>,
}

impl<'b, 'tcx> CostChecker<'b, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        instance: Option<ty::Instance<'tcx>>,
        callee_body: &'b Body<'tcx>,
    ) -> CostChecker<'b, 'tcx> {
        CostChecker { tcx, param_env, callee_body, instance, cost: 0, validation: Ok(()) }
    }

    pub fn cost(&self) -> usize {
        self.cost
    }

    pub fn validation(&self) -> Result<(), &'static str> {
        self.validation
    }

    fn subst_ty(&self, v: Ty<'tcx>) -> Ty<'tcx> {
        if let Some(instance) = self.instance { instance.subst_mir(self.tcx, &v) } else { v }
    }
}

impl<'tcx> Visitor<'tcx> for CostChecker<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // Don't count StorageLive/StorageDead in the inlining cost.
        match statement.kind {
            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Deinit(_)
            | StatementKind::Nop => {}
            _ => self.cost += INSTR_COST,
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        let tcx = self.tcx;
        match terminator.kind {
            TerminatorKind::Drop { ref place, unwind, .. }
            | TerminatorKind::DropAndReplace { ref place, unwind, .. } => {
                // If the place doesn't actually need dropping, treat it like a regular goto.
                let ty = self.subst_ty(place.ty(self.callee_body, tcx).ty);
                if ty.needs_drop(tcx, self.param_env) {
                    self.cost += CALL_PENALTY;
                    if unwind.is_some() {
                        self.cost += LANDINGPAD_PENALTY;
                    }
                } else {
                    self.cost += INSTR_COST;
                }
            }
            TerminatorKind::Call { func: Operand::Constant(ref f), cleanup, .. } => {
                let fn_ty = self.subst_ty(f.literal.ty());
                self.cost += if let ty::FnDef(def_id, _) = *fn_ty.kind() && tcx.is_intrinsic(def_id) {
                    // Don't give intrinsics the extra penalty for calls
                    INSTR_COST
                } else {
                    CALL_PENALTY
                };
                if cleanup.is_some() {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Assert { cleanup, .. } => {
                self.cost += CALL_PENALTY;
                if cleanup.is_some() {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Resume => self.cost += RESUME_PENALTY,
            TerminatorKind::InlineAsm { cleanup, .. } => {
                self.cost += INSTR_COST;
                if cleanup.is_some() {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            _ => self.cost += INSTR_COST,
        }

        self.super_terminator(terminator, location);
    }

    /// Count up the cost of local variables and temps, if we know the size
    /// use that, otherwise we use a moderately-large dummy cost.
    fn visit_local_decl(&mut self, local: Local, local_decl: &LocalDecl<'tcx>) {
        let tcx = self.tcx;
        let ptr_size = tcx.data_layout.pointer_size.bytes();

        let ty = self.subst_ty(local_decl.ty);
        // Cost of the var is the size in machine-words, if we know
        // it.
        if let Some(size) = type_size_of(tcx, self.param_env, ty) {
            self.cost += ((size + ptr_size - 1) / ptr_size) as usize;
        } else {
            self.cost += UNKNOWN_SIZE_COST;
        }

        self.super_local_decl(local, local_decl)
    }

    /// This method duplicates code from MIR validation in an attempt to detect type mismatches due
    /// to normalization failure.
    fn visit_projection_elem(
        &mut self,
        local: Local,
        proj_base: &[PlaceElem<'tcx>],
        elem: PlaceElem<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        if let ProjectionElem::Field(f, ty) = elem {
            let parent = Place { local, projection: self.tcx.intern_place_elems(proj_base) };
            let parent_ty = parent.ty(&self.callee_body.local_decls, self.tcx);
            let check_equal = |this: &mut Self, f_ty| {
                if !util::is_equal_up_to_subtyping(this.tcx, this.param_env, ty, f_ty) {
                    trace!(?ty, ?f_ty);
                    this.validation = Err("failed to normalize projection type");
                    return;
                }
            };

            let kind = match parent_ty.ty.kind() {
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
                    self.tcx.bound_type_of(def_id).subst(self.tcx, substs).kind()
                }
                kind => kind,
            };

            match kind {
                ty::Tuple(fields) => {
                    let Some(f_ty) = fields.get(f.as_usize()) else {
                        self.validation = Err("malformed MIR");
                        return;
                    };
                    check_equal(self, *f_ty);
                }
                ty::Adt(adt_def, substs) => {
                    let var = parent_ty.variant_index.unwrap_or(VariantIdx::from_u32(0));
                    let Some(field) = adt_def.variant(var).fields.get(f.as_usize()) else {
                        self.validation = Err("malformed MIR");
                        return;
                    };
                    check_equal(self, field.ty(self.tcx, substs));
                }
                ty::Closure(_, substs) => {
                    let substs = substs.as_closure();
                    let Some(f_ty) = substs.upvar_tys().nth(f.as_usize()) else {
                        self.validation = Err("malformed MIR");
                        return;
                    };
                    check_equal(self, f_ty);
                }
                &ty::Generator(def_id, substs, _) => {
                    let f_ty = if let Some(var) = parent_ty.variant_index {
                        let gen_body = if def_id == self.callee_body.source.def_id() {
                            self.callee_body
                        } else {
                            self.tcx.optimized_mir(def_id)
                        };

                        let Some(layout) = gen_body.generator_layout() else {
                            self.validation = Err("malformed MIR");
                            return;
                        };

                        let Some(&local) = layout.variant_fields[var].get(f) else {
                            self.validation = Err("malformed MIR");
                            return;
                        };

                        let Some(f_ty) = layout.field_tys.get(local) else {
                            self.validation = Err("malformed MIR");
                            return;
                        };

                        f_ty.ty
                    } else {
                        let Some(f_ty) = substs.as_generator().prefix_tys().nth(f.index()) else {
                            self.validation = Err("malformed MIR");
                            return;
                        };

                        f_ty
                    };

                    check_equal(self, f_ty);
                }
                _ => self.validation = Err("malformed MIR"),
            }
        }

        self.super_projection_elem(local, proj_base, elem, context, location);
    }
}

fn type_size_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Option<u64> {
    tcx.layout_of(param_env.and(ty)).ok().map(|layout| layout.size.bytes())
}
