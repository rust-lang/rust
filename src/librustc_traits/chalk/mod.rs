//! Calls `chalk-solve` to solve a `ty::Predicate`
//!
//! In order to call `chalk-solve`, this file must convert a
//! `ChalkCanonicalGoal` into a Chalk ucanonical goal. It then calls Chalk, and
//! converts the answer back into rustc solution.

crate mod db;
crate mod lowering;

use rustc_data_structures::fx::FxHashMap;

use rustc_index::vec::IndexVec;

use rustc_middle::infer::canonical::{CanonicalTyVarKind, CanonicalVarKind};
use rustc_middle::traits::ChalkRustInterner;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::subst::GenericArg;
use rustc_middle::ty::{
    self, Bound, BoundVar, ParamTy, Region, RegionKind, Ty, TyCtxt, TypeFoldable,
};

use rustc_infer::infer::canonical::{
    Canonical, CanonicalVarValues, Certainty, QueryRegionConstraints, QueryResponse,
};
use rustc_infer::traits::{self, ChalkCanonicalGoal};

use crate::chalk::db::RustIrDatabase as ChalkRustIrDatabase;
use crate::chalk::lowering::{LowerInto, ParamsSubstitutor};

use chalk_solve::Solution;

crate fn provide(p: &mut Providers<'_>) {
    *p = Providers { evaluate_goal, ..*p };
}

crate fn evaluate_goal<'tcx>(
    tcx: TyCtxt<'tcx>,
    obligation: ChalkCanonicalGoal<'tcx>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, traits::query::NoSolution> {
    let interner = ChalkRustInterner { tcx };

    // Chalk doesn't have a notion of `Params`, so instead we use placeholders.
    let mut params_substitutor = ParamsSubstitutor::new(tcx);
    let obligation = obligation.fold_with(&mut params_substitutor);
    let _params: FxHashMap<usize, ParamTy> = params_substitutor.params;
    let max_universe = obligation.max_universe.index();

    let _lowered_goal: chalk_ir::UCanonical<
        chalk_ir::InEnvironment<chalk_ir::Goal<ChalkRustInterner<'tcx>>>,
    > = chalk_ir::UCanonical {
        canonical: chalk_ir::Canonical {
            binders: chalk_ir::CanonicalVarKinds::from(
                &interner,
                obligation.variables.iter().map(|v| match v.kind {
                    CanonicalVarKind::PlaceholderTy(_ty) => unimplemented!(),
                    CanonicalVarKind::PlaceholderRegion(_ui) => unimplemented!(),
                    CanonicalVarKind::Ty(ty) => match ty {
                        CanonicalTyVarKind::General(ui) => chalk_ir::WithKind::new(
                            chalk_ir::VariableKind::Ty(chalk_ir::TyKind::General),
                            chalk_ir::UniverseIndex { counter: ui.index() },
                        ),
                        CanonicalTyVarKind::Int => chalk_ir::WithKind::new(
                            chalk_ir::VariableKind::Ty(chalk_ir::TyKind::Integer),
                            chalk_ir::UniverseIndex::root(),
                        ),
                        CanonicalTyVarKind::Float => chalk_ir::WithKind::new(
                            chalk_ir::VariableKind::Ty(chalk_ir::TyKind::Float),
                            chalk_ir::UniverseIndex::root(),
                        ),
                    },
                    CanonicalVarKind::Region(ui) => chalk_ir::WithKind::new(
                        chalk_ir::VariableKind::Lifetime,
                        chalk_ir::UniverseIndex { counter: ui.index() },
                    ),
                    CanonicalVarKind::Const(_ui) => unimplemented!(),
                    CanonicalVarKind::PlaceholderConst(_pc) => unimplemented!(),
                }),
            ),
            value: obligation.value.lower_into(&interner),
        },
        universes: max_universe + 1,
    };

    let solver_choice = chalk_solve::SolverChoice::SLG { max_size: 32, expected_answers: None };
    let mut solver = solver_choice.into_solver::<ChalkRustInterner<'tcx>>();

    let db = ChalkRustIrDatabase { tcx, interner };
    let solution = solver.solve(&db, &_lowered_goal);

    // Ideally, the code to convert *back* to rustc types would live close to
    // the code to convert *from* rustc types. Right now though, we don't
    // really need this and so it's really minimal.
    // Right now, we also treat a `Unique` solution the same as
    // `Ambig(Definite)`. This really isn't right.
    let make_solution = |_subst: chalk_ir::Substitution<_>| {
        let mut var_values: IndexVec<BoundVar, GenericArg<'tcx>> = IndexVec::new();
        _subst.parameters(&interner).iter().for_each(|p| {
            // FIXME(chalk): we should move this elsewhere, since this is
            // essentially inverse of lowering a `GenericArg`.
            let _data = p.data(&interner);
            match _data {
                chalk_ir::GenericArgData::Ty(_t) => {
                    use chalk_ir::TyData;
                    use rustc_ast::ast;

                    let _data = _t.data(&interner);
                    let kind = match _data {
                        TyData::Apply(_application_ty) => match _application_ty.name {
                            chalk_ir::TypeName::Adt(_struct_id) => unimplemented!(),
                            chalk_ir::TypeName::Scalar(scalar) => match scalar {
                                chalk_ir::Scalar::Bool => ty::Bool,
                                chalk_ir::Scalar::Char => ty::Char,
                                chalk_ir::Scalar::Int(int_ty) => match int_ty {
                                    chalk_ir::IntTy::Isize => ty::Int(ast::IntTy::Isize),
                                    chalk_ir::IntTy::I8 => ty::Int(ast::IntTy::I8),
                                    chalk_ir::IntTy::I16 => ty::Int(ast::IntTy::I16),
                                    chalk_ir::IntTy::I32 => ty::Int(ast::IntTy::I32),
                                    chalk_ir::IntTy::I64 => ty::Int(ast::IntTy::I64),
                                    chalk_ir::IntTy::I128 => ty::Int(ast::IntTy::I128),
                                },
                                chalk_ir::Scalar::Uint(int_ty) => match int_ty {
                                    chalk_ir::UintTy::Usize => ty::Uint(ast::UintTy::Usize),
                                    chalk_ir::UintTy::U8 => ty::Uint(ast::UintTy::U8),
                                    chalk_ir::UintTy::U16 => ty::Uint(ast::UintTy::U16),
                                    chalk_ir::UintTy::U32 => ty::Uint(ast::UintTy::U32),
                                    chalk_ir::UintTy::U64 => ty::Uint(ast::UintTy::U64),
                                    chalk_ir::UintTy::U128 => ty::Uint(ast::UintTy::U128),
                                },
                                chalk_ir::Scalar::Float(float_ty) => match float_ty {
                                    chalk_ir::FloatTy::F32 => ty::Float(ast::FloatTy::F32),
                                    chalk_ir::FloatTy::F64 => ty::Float(ast::FloatTy::F64),
                                },
                            },
                            chalk_ir::TypeName::Array => unimplemented!(),
                            chalk_ir::TypeName::FnDef(_) => unimplemented!(),
                            chalk_ir::TypeName::Never => unimplemented!(),
                            chalk_ir::TypeName::Tuple(_size) => unimplemented!(),
                            chalk_ir::TypeName::Slice => unimplemented!(),
                            chalk_ir::TypeName::Raw(_) => unimplemented!(),
                            chalk_ir::TypeName::Ref(_) => unimplemented!(),
                            chalk_ir::TypeName::Str => unimplemented!(),
                            chalk_ir::TypeName::OpaqueType(_ty) => unimplemented!(),
                            chalk_ir::TypeName::AssociatedType(_assoc_ty) => unimplemented!(),
                            chalk_ir::TypeName::Error => unimplemented!(),
                        },
                        TyData::Placeholder(_placeholder) => {
                            unimplemented!();
                        }
                        TyData::Alias(_alias_ty) => unimplemented!(),
                        TyData::Function(_quantified_ty) => unimplemented!(),
                        TyData::BoundVar(_bound) => Bound(
                            ty::DebruijnIndex::from_usize(_bound.debruijn.depth() as usize),
                            ty::BoundTy {
                                var: ty::BoundVar::from_usize(_bound.index),
                                kind: ty::BoundTyKind::Anon,
                            },
                        ),
                        TyData::InferenceVar(_, _) => unimplemented!(),
                        TyData::Dyn(_) => unimplemented!(),
                    };
                    let _ty: Ty<'_> = tcx.mk_ty(kind);
                    let _arg: GenericArg<'_> = _ty.into();
                    var_values.push(_arg);
                }
                chalk_ir::GenericArgData::Lifetime(_l) => {
                    let _data = _l.data(&interner);
                    let _lifetime: Region<'_> = match _data {
                        chalk_ir::LifetimeData::BoundVar(_var) => {
                            tcx.mk_region(RegionKind::ReLateBound(
                                rustc_middle::ty::DebruijnIndex::from_usize(
                                    _var.debruijn.depth() as usize
                                ),
                                rustc_middle::ty::BoundRegion::BrAnon(_var.index as u32),
                            ))
                        }
                        chalk_ir::LifetimeData::InferenceVar(_var) => unimplemented!(),
                        chalk_ir::LifetimeData::Placeholder(_index) => unimplemented!(),
                        chalk_ir::LifetimeData::Phantom(_, _) => unimplemented!(),
                    };
                    let _arg: GenericArg<'_> = _lifetime.into();
                    var_values.push(_arg);
                }
                chalk_ir::GenericArgData::Const(_) => unimplemented!(),
            }
        });
        let sol = Canonical {
            max_universe: ty::UniverseIndex::from_usize(0),
            variables: obligation.variables.clone(),
            value: QueryResponse {
                var_values: CanonicalVarValues { var_values },
                region_constraints: QueryRegionConstraints::default(),
                certainty: Certainty::Proven,
                value: (),
            },
        };
        &*tcx.arena.alloc(sol)
    };
    solution
        .map(|s| match s {
            Solution::Unique(_subst) => {
                // FIXME(chalk): handle constraints
                make_solution(_subst.value.subst)
            }
            Solution::Ambig(_guidance) => {
                match _guidance {
                    chalk_solve::Guidance::Definite(_subst) => make_solution(_subst.value),
                    chalk_solve::Guidance::Suggested(_) => unimplemented!(),
                    chalk_solve::Guidance::Unknown => {
                        // chalk_fulfill doesn't use the var_values here, so
                        // let's just ignore that
                        let sol = Canonical {
                            max_universe: ty::UniverseIndex::from_usize(0),
                            variables: obligation.variables.clone(),
                            value: QueryResponse {
                                var_values: CanonicalVarValues { var_values: IndexVec::new() }
                                    .make_identity(tcx),
                                region_constraints: QueryRegionConstraints::default(),
                                certainty: Certainty::Ambiguous,
                                value: (),
                            },
                        };
                        &*tcx.arena.alloc(sol)
                    }
                }
            }
        })
        .ok_or(traits::query::NoSolution)
}
