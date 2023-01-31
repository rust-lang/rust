//! Contains the logic to lower rustc types into Chalk types
//!
//! In many cases there is a 1:1 relationship between a rustc type and a Chalk type.
//! For example, a `SubstsRef` maps almost directly to a `Substitution`. In some
//! other cases, such as `Param`s, there is no Chalk type, so we have to handle
//! accordingly.
//!
//! ## `Ty` lowering
//! Much of the `Ty` lowering is 1:1 with Chalk. (Or will be eventually). A
//! helpful table for what types lower to what can be found in the
//! [Chalk book](https://rust-lang.github.io/chalk/book/types/rust_types.html).
//! The most notable difference lies with `Param`s. To convert from rustc to
//! Chalk, we eagerly and deeply convert `Param`s to placeholders (in goals) or
//! bound variables (for clause generation through functions in `db`).
//!
//! ## `Region` lowering
//! Regions are handled in rustc and Chalk is quite differently. In rustc, there
//! is a difference between "early bound" and "late bound" regions, where only
//! the late bound regions have a `DebruijnIndex`. Moreover, in Chalk all
//! regions (Lifetimes) have an associated index. In rustc, only `BrAnon`s have
//! an index, whereas `BrNamed` don't. In order to lower regions to Chalk, we
//! convert all regions into `BrAnon` late-bound regions.
//!
//! ## `Const` lowering
//! Chalk doesn't handle consts currently, so consts are currently lowered to
//! an empty tuple.
//!
//! ## Bound variable collection
//! Another difference between rustc and Chalk lies in the handling of binders.
//! Chalk requires that we store the bound parameter kinds, whereas rustc does
//! not. To lower anything wrapped in a `Binder`, we first deeply find any bound
//! variables from the current `Binder`.

use rustc_ast::ast;
use rustc_middle::traits::{ChalkEnvironmentAndGoal, ChalkRustInterner as RustInterner};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, SubstsRef};
use rustc_middle::ty::{
    self, Binder, Region, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeSuperVisitable, TypeVisitable, TypeVisitor,
};
use rustc_span::def_id::DefId;

use chalk_ir::{FnSig, ForeignDefId};
use rustc_hir::Unsafety;
use std::collections::btree_map::{BTreeMap, Entry};
use std::ops::ControlFlow;

/// Essentially an `Into` with a `&RustInterner` parameter
pub(crate) trait LowerInto<'tcx, T> {
    /// Lower a rustc construct (e.g., `ty::TraitPredicate`) to a chalk type, consuming `self`.
    fn lower_into(self, interner: RustInterner<'tcx>) -> T;
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Substitution<RustInterner<'tcx>>> for SubstsRef<'tcx> {
    fn lower_into(
        self,
        interner: RustInterner<'tcx>,
    ) -> chalk_ir::Substitution<RustInterner<'tcx>> {
        chalk_ir::Substitution::from_iter(interner, self.iter().map(|s| s.lower_into(interner)))
    }
}

impl<'tcx> LowerInto<'tcx, SubstsRef<'tcx>> for &chalk_ir::Substitution<RustInterner<'tcx>> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> SubstsRef<'tcx> {
        interner.tcx.mk_substs(self.iter(interner).map(|subst| subst.lower_into(interner)))
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::InEnvironment<chalk_ir::Goal<RustInterner<'tcx>>>>
    for ChalkEnvironmentAndGoal<'tcx>
{
    fn lower_into(
        self,
        interner: RustInterner<'tcx>,
    ) -> chalk_ir::InEnvironment<chalk_ir::Goal<RustInterner<'tcx>>> {
        let clauses = self.environment.into_iter().map(|predicate| {
            let (predicate, binders, _named_regions) =
                collect_bound_vars(interner, interner.tcx, predicate.kind());
            let consequence = match predicate {
                ty::PredicateKind::TypeWellFormedFromEnv(ty) => {
                    chalk_ir::DomainGoal::FromEnv(chalk_ir::FromEnv::Ty(ty.lower_into(interner)))
                }
                ty::PredicateKind::Clause(ty::Clause::Trait(predicate)) => {
                    chalk_ir::DomainGoal::FromEnv(chalk_ir::FromEnv::Trait(
                        predicate.trait_ref.lower_into(interner),
                    ))
                }
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(predicate)) => {
                    chalk_ir::DomainGoal::Holds(chalk_ir::WhereClause::LifetimeOutlives(
                        chalk_ir::LifetimeOutlives {
                            a: predicate.0.lower_into(interner),
                            b: predicate.1.lower_into(interner),
                        },
                    ))
                }
                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(predicate)) => {
                    chalk_ir::DomainGoal::Holds(chalk_ir::WhereClause::TypeOutlives(
                        chalk_ir::TypeOutlives {
                            ty: predicate.0.lower_into(interner),
                            lifetime: predicate.1.lower_into(interner),
                        },
                    ))
                }
                ty::PredicateKind::Clause(ty::Clause::Projection(predicate)) => {
                    chalk_ir::DomainGoal::Holds(chalk_ir::WhereClause::AliasEq(
                        predicate.lower_into(interner),
                    ))
                }
                ty::PredicateKind::WellFormed(arg) => match arg.unpack() {
                    ty::GenericArgKind::Type(ty) => chalk_ir::DomainGoal::WellFormed(
                        chalk_ir::WellFormed::Ty(ty.lower_into(interner)),
                    ),
                    // FIXME(chalk): we need to change `WellFormed` in Chalk to take a `GenericArg`
                    _ => chalk_ir::DomainGoal::WellFormed(chalk_ir::WellFormed::Ty(
                        interner.tcx.types.unit.lower_into(interner),
                    )),
                },
                ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::Ambiguous
                | ty::PredicateKind::ConstEquate(..) => bug!("unexpected predicate {}", predicate),
            };
            let value = chalk_ir::ProgramClauseImplication {
                consequence,
                conditions: chalk_ir::Goals::empty(interner),
                priority: chalk_ir::ClausePriority::High,
                constraints: chalk_ir::Constraints::empty(interner),
            };
            chalk_ir::ProgramClauseData(chalk_ir::Binders::new(binders, value)).intern(interner)
        });

        let goal: chalk_ir::GoalData<RustInterner<'tcx>> = self.goal.lower_into(interner);
        chalk_ir::InEnvironment {
            environment: chalk_ir::Environment {
                clauses: chalk_ir::ProgramClauses::from_iter(interner, clauses),
            },
            goal: goal.intern(interner),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::GoalData<RustInterner<'tcx>>> for ty::Predicate<'tcx> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> chalk_ir::GoalData<RustInterner<'tcx>> {
        let (predicate, binders, _named_regions) =
            collect_bound_vars(interner, interner.tcx, self.kind());

        let value = match predicate {
            ty::PredicateKind::Clause(ty::Clause::Trait(predicate)) => {
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(
                    chalk_ir::WhereClause::Implemented(predicate.trait_ref.lower_into(interner)),
                ))
            }
            ty::PredicateKind::Clause(ty::Clause::RegionOutlives(predicate)) => {
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(
                    chalk_ir::WhereClause::LifetimeOutlives(chalk_ir::LifetimeOutlives {
                        a: predicate.0.lower_into(interner),
                        b: predicate.1.lower_into(interner),
                    }),
                ))
            }
            ty::PredicateKind::Clause(ty::Clause::TypeOutlives(predicate)) => {
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(
                    chalk_ir::WhereClause::TypeOutlives(chalk_ir::TypeOutlives {
                        ty: predicate.0.lower_into(interner),
                        lifetime: predicate.1.lower_into(interner),
                    }),
                ))
            }
            ty::PredicateKind::Clause(ty::Clause::Projection(predicate)) => {
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(
                    chalk_ir::WhereClause::AliasEq(predicate.lower_into(interner)),
                ))
            }
            ty::PredicateKind::WellFormed(arg) => match arg.unpack() {
                GenericArgKind::Type(ty) => match ty.kind() {
                    // FIXME(chalk): In Chalk, a placeholder is WellFormed if it
                    // `FromEnv`. However, when we "lower" Params, we don't update
                    // the environment.
                    ty::Placeholder(..) => {
                        chalk_ir::GoalData::All(chalk_ir::Goals::empty(interner))
                    }

                    _ => chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::WellFormed(
                        chalk_ir::WellFormed::Ty(ty.lower_into(interner)),
                    )),
                },
                // FIXME(chalk): handle well formed consts
                GenericArgKind::Const(..) => {
                    chalk_ir::GoalData::All(chalk_ir::Goals::empty(interner))
                }
                GenericArgKind::Lifetime(lt) => bug!("unexpected well formed predicate: {:?}", lt),
            },

            ty::PredicateKind::ObjectSafe(t) => chalk_ir::GoalData::DomainGoal(
                chalk_ir::DomainGoal::ObjectSafe(chalk_ir::TraitId(t)),
            ),

            ty::PredicateKind::Subtype(ty::SubtypePredicate { a, b, a_is_expected: _ }) => {
                chalk_ir::GoalData::SubtypeGoal(chalk_ir::SubtypeGoal {
                    a: a.lower_into(interner),
                    b: b.lower_into(interner),
                })
            }

            // FIXME(chalk): other predicates
            //
            // We can defer this, but ultimately we'll want to express
            // some of these in terms of chalk operations.
            ty::PredicateKind::ClosureKind(..)
            | ty::PredicateKind::Coerce(..)
            | ty::PredicateKind::ConstEvaluatable(..)
            | ty::PredicateKind::Ambiguous
            | ty::PredicateKind::ConstEquate(..) => {
                chalk_ir::GoalData::All(chalk_ir::Goals::empty(interner))
            }
            ty::PredicateKind::TypeWellFormedFromEnv(ty) => chalk_ir::GoalData::DomainGoal(
                chalk_ir::DomainGoal::FromEnv(chalk_ir::FromEnv::Ty(ty.lower_into(interner))),
            ),
        };

        chalk_ir::GoalData::Quantified(
            chalk_ir::QuantifierKind::ForAll,
            chalk_ir::Binders::new(binders, value.intern(interner)),
        )
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::TraitRef<RustInterner<'tcx>>>
    for rustc_middle::ty::TraitRef<'tcx>
{
    fn lower_into(self, interner: RustInterner<'tcx>) -> chalk_ir::TraitRef<RustInterner<'tcx>> {
        chalk_ir::TraitRef {
            trait_id: chalk_ir::TraitId(self.def_id),
            substitution: self.substs.lower_into(interner),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::AliasEq<RustInterner<'tcx>>>
    for rustc_middle::ty::ProjectionPredicate<'tcx>
{
    fn lower_into(self, interner: RustInterner<'tcx>) -> chalk_ir::AliasEq<RustInterner<'tcx>> {
        // FIXME(associated_const_equality): teach chalk about terms for alias eq.
        chalk_ir::AliasEq {
            ty: self.term.ty().unwrap().lower_into(interner),
            alias: chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
                associated_ty_id: chalk_ir::AssocTypeId(self.projection_ty.def_id),
                substitution: self.projection_ty.substs.lower_into(interner),
            }),
        }
    }
}

/*
// FIXME(...): Where do I add this to Chalk? I can't find it in the rustc repo anywhere.
impl<'tcx> LowerInto<'tcx, chalk_ir::Term<RustInterner<'tcx>>> for rustc_middle::ty::Term<'tcx> {
  fn lower_into(self, interner: RustInterner<'tcx>) -> chalk_ir::Term<RustInterner<'tcx>> {
    match self {
      ty::Term::Ty(ty) => ty.lower_into(interner).into(),
      ty::Term::Const(c) => c.lower_into(interner).into(),
    }
  }
}
*/

impl<'tcx> LowerInto<'tcx, chalk_ir::Ty<RustInterner<'tcx>>> for Ty<'tcx> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> chalk_ir::Ty<RustInterner<'tcx>> {
        let int = |i| chalk_ir::TyKind::Scalar(chalk_ir::Scalar::Int(i));
        let uint = |i| chalk_ir::TyKind::Scalar(chalk_ir::Scalar::Uint(i));
        let float = |f| chalk_ir::TyKind::Scalar(chalk_ir::Scalar::Float(f));

        match *self.kind() {
            ty::Bool => chalk_ir::TyKind::Scalar(chalk_ir::Scalar::Bool),
            ty::Char => chalk_ir::TyKind::Scalar(chalk_ir::Scalar::Char),
            ty::Int(ty) => match ty {
                ty::IntTy::Isize => int(chalk_ir::IntTy::Isize),
                ty::IntTy::I8 => int(chalk_ir::IntTy::I8),
                ty::IntTy::I16 => int(chalk_ir::IntTy::I16),
                ty::IntTy::I32 => int(chalk_ir::IntTy::I32),
                ty::IntTy::I64 => int(chalk_ir::IntTy::I64),
                ty::IntTy::I128 => int(chalk_ir::IntTy::I128),
            },
            ty::Uint(ty) => match ty {
                ty::UintTy::Usize => uint(chalk_ir::UintTy::Usize),
                ty::UintTy::U8 => uint(chalk_ir::UintTy::U8),
                ty::UintTy::U16 => uint(chalk_ir::UintTy::U16),
                ty::UintTy::U32 => uint(chalk_ir::UintTy::U32),
                ty::UintTy::U64 => uint(chalk_ir::UintTy::U64),
                ty::UintTy::U128 => uint(chalk_ir::UintTy::U128),
            },
            ty::Float(ty) => match ty {
                ty::FloatTy::F32 => float(chalk_ir::FloatTy::F32),
                ty::FloatTy::F64 => float(chalk_ir::FloatTy::F64),
            },
            ty::Adt(def, substs) => {
                chalk_ir::TyKind::Adt(chalk_ir::AdtId(def), substs.lower_into(interner))
            }
            ty::Foreign(def_id) => chalk_ir::TyKind::Foreign(ForeignDefId(def_id)),
            ty::Str => chalk_ir::TyKind::Str,
            ty::Array(ty, len) => {
                chalk_ir::TyKind::Array(ty.lower_into(interner), len.lower_into(interner))
            }
            ty::Slice(ty) => chalk_ir::TyKind::Slice(ty.lower_into(interner)),

            ty::RawPtr(ptr) => {
                chalk_ir::TyKind::Raw(ptr.mutbl.lower_into(interner), ptr.ty.lower_into(interner))
            }
            ty::Ref(region, ty, mutability) => chalk_ir::TyKind::Ref(
                mutability.lower_into(interner),
                region.lower_into(interner),
                ty.lower_into(interner),
            ),
            ty::FnDef(def_id, substs) => {
                chalk_ir::TyKind::FnDef(chalk_ir::FnDefId(def_id), substs.lower_into(interner))
            }
            ty::FnPtr(sig) => {
                let (inputs_and_outputs, binders, _named_regions) =
                    collect_bound_vars(interner, interner.tcx, sig.inputs_and_output());
                chalk_ir::TyKind::Function(chalk_ir::FnPointer {
                    num_binders: binders.len(interner),
                    sig: sig.lower_into(interner),
                    substitution: chalk_ir::FnSubst(chalk_ir::Substitution::from_iter(
                        interner,
                        inputs_and_outputs.iter().map(|ty| {
                            chalk_ir::GenericArgData::Ty(ty.lower_into(interner)).intern(interner)
                        }),
                    )),
                })
            }
            // FIXME(dyn-star): handle the dynamic kind (dyn or dyn*)
            ty::Dynamic(predicates, region, _kind) => chalk_ir::TyKind::Dyn(chalk_ir::DynTy {
                bounds: predicates.lower_into(interner),
                lifetime: region.lower_into(interner),
            }),
            ty::Closure(def_id, substs) => {
                chalk_ir::TyKind::Closure(chalk_ir::ClosureId(def_id), substs.lower_into(interner))
            }
            ty::Generator(def_id, substs, _) => chalk_ir::TyKind::Generator(
                chalk_ir::GeneratorId(def_id),
                substs.lower_into(interner),
            ),
            ty::GeneratorWitness(_) => unimplemented!(),
            ty::GeneratorWitnessMIR(..) => unimplemented!(),
            ty::Never => chalk_ir::TyKind::Never,
            ty::Tuple(types) => {
                chalk_ir::TyKind::Tuple(types.len(), types.as_substs().lower_into(interner))
            }
            ty::Alias(ty::Projection, ty::AliasTy { def_id, substs, .. }) => {
                chalk_ir::TyKind::Alias(chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
                    associated_ty_id: chalk_ir::AssocTypeId(def_id),
                    substitution: substs.lower_into(interner),
                }))
            }
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
                chalk_ir::TyKind::Alias(chalk_ir::AliasTy::Opaque(chalk_ir::OpaqueTy {
                    opaque_ty_id: chalk_ir::OpaqueTyId(def_id),
                    substitution: substs.lower_into(interner),
                }))
            }
            // This should have been done eagerly prior to this, and all Params
            // should have been substituted to placeholders
            ty::Param(_) => panic!("Lowering Param when not expected."),
            ty::Bound(db, bound) => chalk_ir::TyKind::BoundVar(chalk_ir::BoundVar::new(
                chalk_ir::DebruijnIndex::new(db.as_u32()),
                bound.var.index(),
            )),
            ty::Placeholder(_placeholder) => {
                chalk_ir::TyKind::Placeholder(chalk_ir::PlaceholderIndex {
                    ui: chalk_ir::UniverseIndex { counter: _placeholder.universe.as_usize() },
                    idx: _placeholder.name.expect_anon() as usize,
                })
            }
            ty::Infer(_infer) => unimplemented!(),
            ty::Error(_) => chalk_ir::TyKind::Error,
        }
        .intern(interner)
    }
}

impl<'tcx> LowerInto<'tcx, Ty<'tcx>> for &chalk_ir::Ty<RustInterner<'tcx>> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> Ty<'tcx> {
        use chalk_ir::TyKind;

        let kind = match self.kind(interner) {
            TyKind::Adt(struct_id, substitution) => {
                ty::Adt(struct_id.0, substitution.lower_into(interner))
            }
            TyKind::Scalar(scalar) => match scalar {
                chalk_ir::Scalar::Bool => ty::Bool,
                chalk_ir::Scalar::Char => ty::Char,
                chalk_ir::Scalar::Int(int_ty) => match int_ty {
                    chalk_ir::IntTy::Isize => ty::Int(ty::IntTy::Isize),
                    chalk_ir::IntTy::I8 => ty::Int(ty::IntTy::I8),
                    chalk_ir::IntTy::I16 => ty::Int(ty::IntTy::I16),
                    chalk_ir::IntTy::I32 => ty::Int(ty::IntTy::I32),
                    chalk_ir::IntTy::I64 => ty::Int(ty::IntTy::I64),
                    chalk_ir::IntTy::I128 => ty::Int(ty::IntTy::I128),
                },
                chalk_ir::Scalar::Uint(int_ty) => match int_ty {
                    chalk_ir::UintTy::Usize => ty::Uint(ty::UintTy::Usize),
                    chalk_ir::UintTy::U8 => ty::Uint(ty::UintTy::U8),
                    chalk_ir::UintTy::U16 => ty::Uint(ty::UintTy::U16),
                    chalk_ir::UintTy::U32 => ty::Uint(ty::UintTy::U32),
                    chalk_ir::UintTy::U64 => ty::Uint(ty::UintTy::U64),
                    chalk_ir::UintTy::U128 => ty::Uint(ty::UintTy::U128),
                },
                chalk_ir::Scalar::Float(float_ty) => match float_ty {
                    chalk_ir::FloatTy::F32 => ty::Float(ty::FloatTy::F32),
                    chalk_ir::FloatTy::F64 => ty::Float(ty::FloatTy::F64),
                },
            },
            TyKind::Array(ty, c) => {
                let ty = ty.lower_into(interner);
                let c = c.lower_into(interner);
                ty::Array(ty, c)
            }
            TyKind::FnDef(id, substitution) => ty::FnDef(id.0, substitution.lower_into(interner)),
            TyKind::Closure(closure, substitution) => {
                ty::Closure(closure.0, substitution.lower_into(interner))
            }
            TyKind::Generator(generator, substitution) => ty::Generator(
                generator.0,
                substitution.lower_into(interner),
                ast::Movability::Static,
            ),
            TyKind::GeneratorWitness(..) => unimplemented!(),
            TyKind::Never => ty::Never,
            TyKind::Tuple(_len, substitution) => {
                ty::Tuple(substitution.lower_into(interner).try_as_type_list().unwrap())
            }
            TyKind::Slice(ty) => ty::Slice(ty.lower_into(interner)),
            TyKind::Raw(mutbl, ty) => ty::RawPtr(ty::TypeAndMut {
                ty: ty.lower_into(interner),
                mutbl: mutbl.lower_into(interner),
            }),
            TyKind::Ref(mutbl, lifetime, ty) => ty::Ref(
                lifetime.lower_into(interner),
                ty.lower_into(interner),
                mutbl.lower_into(interner),
            ),
            TyKind::Str => ty::Str,
            TyKind::OpaqueType(opaque_ty, substitution) => ty::Alias(
                ty::Opaque,
                interner.tcx.mk_alias_ty(opaque_ty.0, substitution.lower_into(interner)),
            ),
            TyKind::AssociatedType(assoc_ty, substitution) => ty::Alias(
                ty::Projection,
                interner.tcx.mk_alias_ty(assoc_ty.0, substitution.lower_into(interner)),
            ),
            TyKind::Foreign(def_id) => ty::Foreign(def_id.0),
            TyKind::Error => return interner.tcx.ty_error(),
            TyKind::Alias(alias_ty) => match alias_ty {
                chalk_ir::AliasTy::Projection(projection) => ty::Alias(
                    ty::Projection,
                    interner.tcx.mk_alias_ty(
                        projection.associated_ty_id.0,
                        projection.substitution.lower_into(interner),
                    ),
                ),
                chalk_ir::AliasTy::Opaque(opaque) => ty::Alias(
                    ty::Opaque,
                    interner.tcx.mk_alias_ty(
                        opaque.opaque_ty_id.0,
                        opaque.substitution.lower_into(interner),
                    ),
                ),
            },
            TyKind::Function(_quantified_ty) => unimplemented!(),
            TyKind::BoundVar(bound) => ty::Bound(
                ty::DebruijnIndex::from_usize(bound.debruijn.depth() as usize),
                ty::BoundTy {
                    var: ty::BoundVar::from_usize(bound.index),
                    kind: ty::BoundTyKind::Anon(bound.index as u32),
                },
            ),
            TyKind::Placeholder(placeholder) => ty::Placeholder(ty::Placeholder {
                universe: ty::UniverseIndex::from_usize(placeholder.ui.counter),
                name: ty::BoundTyKind::Anon(placeholder.idx as u32),
            }),
            TyKind::InferenceVar(_, _) => unimplemented!(),
            TyKind::Dyn(_) => unimplemented!(),
        };
        interner.tcx.mk_ty(kind)
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Lifetime<RustInterner<'tcx>>> for Region<'tcx> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> chalk_ir::Lifetime<RustInterner<'tcx>> {
        match *self {
            ty::ReEarlyBound(_) => {
                panic!("Should have already been substituted.");
            }
            ty::ReLateBound(db, br) => chalk_ir::LifetimeData::BoundVar(chalk_ir::BoundVar::new(
                chalk_ir::DebruijnIndex::new(db.as_u32()),
                br.var.as_usize(),
            ))
            .intern(interner),
            ty::ReFree(_) => unimplemented!(),
            ty::ReStatic => chalk_ir::LifetimeData::Static.intern(interner),
            ty::ReVar(_) => unimplemented!(),
            ty::RePlaceholder(placeholder_region) => {
                chalk_ir::LifetimeData::Placeholder(chalk_ir::PlaceholderIndex {
                    ui: chalk_ir::UniverseIndex { counter: placeholder_region.universe.index() },
                    idx: 0, // FIXME: This `idx: 0` is sus.
                })
                .intern(interner)
            }
            ty::ReErased => chalk_ir::LifetimeData::Erased.intern(interner),
        }
    }
}

impl<'tcx> LowerInto<'tcx, Region<'tcx>> for &chalk_ir::Lifetime<RustInterner<'tcx>> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> Region<'tcx> {
        let kind = match self.data(interner) {
            chalk_ir::LifetimeData::BoundVar(var) => ty::ReLateBound(
                ty::DebruijnIndex::from_u32(var.debruijn.depth()),
                ty::BoundRegion {
                    var: ty::BoundVar::from_usize(var.index),
                    kind: ty::BrAnon(var.index as u32, None),
                },
            ),
            chalk_ir::LifetimeData::InferenceVar(_var) => unimplemented!(),
            chalk_ir::LifetimeData::Placeholder(p) => ty::RePlaceholder(ty::Placeholder {
                universe: ty::UniverseIndex::from_usize(p.ui.counter),
                name: ty::BoundRegionKind::BrAnon(p.idx as u32, None),
            }),
            chalk_ir::LifetimeData::Static => return interner.tcx.lifetimes.re_static,
            chalk_ir::LifetimeData::Erased => return interner.tcx.lifetimes.re_erased,
            chalk_ir::LifetimeData::Phantom(void, _) => match *void {},
        };
        interner.tcx.mk_region(kind)
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Const<RustInterner<'tcx>>> for ty::Const<'tcx> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> chalk_ir::Const<RustInterner<'tcx>> {
        let ty = self.ty().lower_into(interner);
        let value = match self.kind() {
            ty::ConstKind::Value(val) => {
                chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst { interned: val })
            }
            ty::ConstKind::Bound(db, bound) => chalk_ir::ConstValue::BoundVar(
                chalk_ir::BoundVar::new(chalk_ir::DebruijnIndex::new(db.as_u32()), bound.index()),
            ),
            _ => unimplemented!("Const not implemented. {:?}", self),
        };
        chalk_ir::ConstData { ty, value }.intern(interner)
    }
}

impl<'tcx> LowerInto<'tcx, ty::Const<'tcx>> for &chalk_ir::Const<RustInterner<'tcx>> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> ty::Const<'tcx> {
        let data = self.data(interner);
        let ty = data.ty.lower_into(interner);
        let kind = match data.value {
            chalk_ir::ConstValue::BoundVar(var) => ty::ConstKind::Bound(
                ty::DebruijnIndex::from_u32(var.debruijn.depth()),
                ty::BoundVar::from_u32(var.index as u32),
            ),
            chalk_ir::ConstValue::InferenceVar(_var) => unimplemented!(),
            chalk_ir::ConstValue::Placeholder(_p) => unimplemented!(),
            chalk_ir::ConstValue::Concrete(c) => ty::ConstKind::Value(c.interned),
        };
        interner.tcx.mk_const(kind, ty)
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::GenericArg<RustInterner<'tcx>>> for GenericArg<'tcx> {
    fn lower_into(self, interner: RustInterner<'tcx>) -> chalk_ir::GenericArg<RustInterner<'tcx>> {
        match self.unpack() {
            ty::subst::GenericArgKind::Type(ty) => {
                chalk_ir::GenericArgData::Ty(ty.lower_into(interner))
            }
            ty::subst::GenericArgKind::Lifetime(lifetime) => {
                chalk_ir::GenericArgData::Lifetime(lifetime.lower_into(interner))
            }
            ty::subst::GenericArgKind::Const(c) => {
                chalk_ir::GenericArgData::Const(c.lower_into(interner))
            }
        }
        .intern(interner)
    }
}

impl<'tcx> LowerInto<'tcx, ty::subst::GenericArg<'tcx>>
    for &chalk_ir::GenericArg<RustInterner<'tcx>>
{
    fn lower_into(self, interner: RustInterner<'tcx>) -> ty::subst::GenericArg<'tcx> {
        match self.data(interner) {
            chalk_ir::GenericArgData::Ty(ty) => {
                let t: Ty<'tcx> = ty.lower_into(interner);
                t.into()
            }
            chalk_ir::GenericArgData::Lifetime(lifetime) => {
                let r: Region<'tcx> = lifetime.lower_into(interner);
                r.into()
            }
            chalk_ir::GenericArgData::Const(c) => {
                let c: ty::Const<'tcx> = c.lower_into(interner);
                c.into()
            }
        }
    }
}

// We lower into an Option here since there are some predicates which Chalk
// doesn't have a representation for yet (as a `WhereClause`), but are so common
// that we just are accepting the unsoundness for now. The `Option` will
// eventually be removed.
impl<'tcx> LowerInto<'tcx, Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>
    for ty::Predicate<'tcx>
{
    fn lower_into(
        self,
        interner: RustInterner<'tcx>,
    ) -> Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>> {
        let (predicate, binders, _named_regions) =
            collect_bound_vars(interner, interner.tcx, self.kind());
        let value = match predicate {
            ty::PredicateKind::Clause(ty::Clause::Trait(predicate)) => {
                Some(chalk_ir::WhereClause::Implemented(predicate.trait_ref.lower_into(interner)))
            }
            ty::PredicateKind::Clause(ty::Clause::RegionOutlives(predicate)) => {
                Some(chalk_ir::WhereClause::LifetimeOutlives(chalk_ir::LifetimeOutlives {
                    a: predicate.0.lower_into(interner),
                    b: predicate.1.lower_into(interner),
                }))
            }
            ty::PredicateKind::Clause(ty::Clause::TypeOutlives(predicate)) => {
                Some(chalk_ir::WhereClause::TypeOutlives(chalk_ir::TypeOutlives {
                    ty: predicate.0.lower_into(interner),
                    lifetime: predicate.1.lower_into(interner),
                }))
            }
            ty::PredicateKind::Clause(ty::Clause::Projection(predicate)) => {
                Some(chalk_ir::WhereClause::AliasEq(predicate.lower_into(interner)))
            }
            ty::PredicateKind::WellFormed(_ty) => None,

            ty::PredicateKind::ObjectSafe(..)
            | ty::PredicateKind::ClosureKind(..)
            | ty::PredicateKind::Subtype(..)
            | ty::PredicateKind::Coerce(..)
            | ty::PredicateKind::ConstEvaluatable(..)
            | ty::PredicateKind::ConstEquate(..)
            | ty::PredicateKind::Ambiguous
            | ty::PredicateKind::TypeWellFormedFromEnv(..) => {
                bug!("unexpected predicate {}", &self)
            }
        };
        value.map(|value| chalk_ir::Binders::new(binders, value))
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Binders<chalk_ir::QuantifiedWhereClauses<RustInterner<'tcx>>>>
    for &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>
{
    fn lower_into(
        self,
        interner: RustInterner<'tcx>,
    ) -> chalk_ir::Binders<chalk_ir::QuantifiedWhereClauses<RustInterner<'tcx>>> {
        // `Self` has one binder:
        // Binder<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>
        // The return type has two:
        // Binders<&[Binders<WhereClause<I>>]>
        // This means that any variables that are escaping `self` need to be
        // shifted in by one so that they are still escaping.
        let predicates = ty::fold::shift_vars(interner.tcx, self, 1);

        let self_ty = interner.tcx.mk_ty(ty::Bound(
            // This is going to be wrapped in a binder
            ty::DebruijnIndex::from_usize(1),
            ty::BoundTy { var: ty::BoundVar::from_usize(0), kind: ty::BoundTyKind::Anon(0) },
        ));
        let where_clauses = predicates.into_iter().map(|predicate| {
            let (predicate, binders, _named_regions) =
                collect_bound_vars(interner, interner.tcx, predicate);
            match predicate {
                ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef { def_id, substs }) => {
                    chalk_ir::Binders::new(
                        binders.clone(),
                        chalk_ir::WhereClause::Implemented(chalk_ir::TraitRef {
                            trait_id: chalk_ir::TraitId(def_id),
                            substitution: interner
                                .tcx
                                .mk_substs_trait(self_ty, substs)
                                .lower_into(interner),
                        }),
                    )
                }
                ty::ExistentialPredicate::Projection(predicate) => chalk_ir::Binders::new(
                    binders.clone(),
                    chalk_ir::WhereClause::AliasEq(chalk_ir::AliasEq {
                        alias: chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
                            associated_ty_id: chalk_ir::AssocTypeId(predicate.def_id),
                            substitution: interner
                                .tcx
                                .mk_substs_trait(self_ty, predicate.substs)
                                .lower_into(interner),
                        }),
                        // FIXME(associated_const_equality): teach chalk about terms for alias eq.
                        ty: predicate.term.ty().unwrap().lower_into(interner),
                    }),
                ),
                ty::ExistentialPredicate::AutoTrait(def_id) => chalk_ir::Binders::new(
                    binders.clone(),
                    chalk_ir::WhereClause::Implemented(chalk_ir::TraitRef {
                        trait_id: chalk_ir::TraitId(def_id),
                        substitution: interner
                            .tcx
                            .mk_substs_trait(self_ty, [])
                            .lower_into(interner),
                    }),
                ),
            }
        });

        // Binder for the bound variable representing the concrete underlying type.
        let existential_binder = chalk_ir::VariableKinds::from1(
            interner,
            chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General),
        );
        let value = chalk_ir::QuantifiedWhereClauses::from_iter(interner, where_clauses);
        chalk_ir::Binders::new(existential_binder, value)
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::FnSig<RustInterner<'tcx>>>
    for ty::Binder<'tcx, ty::FnSig<'tcx>>
{
    fn lower_into(self, _interner: RustInterner<'_>) -> FnSig<RustInterner<'tcx>> {
        chalk_ir::FnSig {
            abi: self.abi(),
            safety: match self.unsafety() {
                Unsafety::Normal => chalk_ir::Safety::Safe,
                Unsafety::Unsafe => chalk_ir::Safety::Unsafe,
            },
            variadic: self.c_variadic(),
        }
    }
}

// We lower into an Option here since there are some predicates which Chalk
// doesn't have a representation for yet (as an `InlineBound`). The `Option` will
// eventually be removed.
impl<'tcx> LowerInto<'tcx, Option<chalk_solve::rust_ir::QuantifiedInlineBound<RustInterner<'tcx>>>>
    for ty::Predicate<'tcx>
{
    fn lower_into(
        self,
        interner: RustInterner<'tcx>,
    ) -> Option<chalk_solve::rust_ir::QuantifiedInlineBound<RustInterner<'tcx>>> {
        let (predicate, binders, _named_regions) =
            collect_bound_vars(interner, interner.tcx, self.kind());
        match predicate {
            ty::PredicateKind::Clause(ty::Clause::Trait(predicate)) => {
                Some(chalk_ir::Binders::new(
                    binders,
                    chalk_solve::rust_ir::InlineBound::TraitBound(
                        predicate.trait_ref.lower_into(interner),
                    ),
                ))
            }
            ty::PredicateKind::Clause(ty::Clause::Projection(predicate)) => {
                Some(chalk_ir::Binders::new(
                    binders,
                    chalk_solve::rust_ir::InlineBound::AliasEqBound(predicate.lower_into(interner)),
                ))
            }
            ty::PredicateKind::Clause(ty::Clause::TypeOutlives(_predicate)) => None,
            ty::PredicateKind::WellFormed(_ty) => None,

            ty::PredicateKind::Clause(ty::Clause::RegionOutlives(..))
            | ty::PredicateKind::ObjectSafe(..)
            | ty::PredicateKind::ClosureKind(..)
            | ty::PredicateKind::Subtype(..)
            | ty::PredicateKind::Coerce(..)
            | ty::PredicateKind::ConstEvaluatable(..)
            | ty::PredicateKind::ConstEquate(..)
            | ty::PredicateKind::Ambiguous
            | ty::PredicateKind::TypeWellFormedFromEnv(..) => {
                bug!("unexpected predicate {}", &self)
            }
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_solve::rust_ir::TraitBound<RustInterner<'tcx>>>
    for ty::TraitRef<'tcx>
{
    fn lower_into(
        self,
        interner: RustInterner<'tcx>,
    ) -> chalk_solve::rust_ir::TraitBound<RustInterner<'tcx>> {
        chalk_solve::rust_ir::TraitBound {
            trait_id: chalk_ir::TraitId(self.def_id),
            args_no_self: self.substs[1..].iter().map(|arg| arg.lower_into(interner)).collect(),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Mutability> for ast::Mutability {
    fn lower_into(self, _interner: RustInterner<'tcx>) -> chalk_ir::Mutability {
        match self {
            rustc_ast::Mutability::Mut => chalk_ir::Mutability::Mut,
            rustc_ast::Mutability::Not => chalk_ir::Mutability::Not,
        }
    }
}

impl<'tcx> LowerInto<'tcx, ast::Mutability> for chalk_ir::Mutability {
    fn lower_into(self, _interner: RustInterner<'tcx>) -> ast::Mutability {
        match self {
            chalk_ir::Mutability::Mut => ast::Mutability::Mut,
            chalk_ir::Mutability::Not => ast::Mutability::Not,
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_solve::rust_ir::Polarity> for ty::ImplPolarity {
    fn lower_into(self, _interner: RustInterner<'tcx>) -> chalk_solve::rust_ir::Polarity {
        match self {
            ty::ImplPolarity::Positive => chalk_solve::rust_ir::Polarity::Positive,
            ty::ImplPolarity::Negative => chalk_solve::rust_ir::Polarity::Negative,
            // FIXME(chalk) reservation impls
            ty::ImplPolarity::Reservation => chalk_solve::rust_ir::Polarity::Negative,
        }
    }
}
impl<'tcx> LowerInto<'tcx, chalk_ir::Variance> for ty::Variance {
    fn lower_into(self, _interner: RustInterner<'tcx>) -> chalk_ir::Variance {
        match self {
            ty::Variance::Covariant => chalk_ir::Variance::Covariant,
            ty::Variance::Invariant => chalk_ir::Variance::Invariant,
            ty::Variance::Contravariant => chalk_ir::Variance::Contravariant,
            ty::Variance::Bivariant => unimplemented!(),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_solve::rust_ir::AliasEqBound<RustInterner<'tcx>>>
    for ty::ProjectionPredicate<'tcx>
{
    fn lower_into(
        self,
        interner: RustInterner<'tcx>,
    ) -> chalk_solve::rust_ir::AliasEqBound<RustInterner<'tcx>> {
        let (trait_ref, own_substs) = self.projection_ty.trait_ref_and_own_substs(interner.tcx);
        chalk_solve::rust_ir::AliasEqBound {
            trait_bound: trait_ref.lower_into(interner),
            associated_ty_id: chalk_ir::AssocTypeId(self.projection_ty.def_id),
            parameters: own_substs.iter().map(|arg| arg.lower_into(interner)).collect(),
            value: self.term.ty().unwrap().lower_into(interner),
        }
    }
}

/// To collect bound vars, we have to do two passes. In the first pass, we
/// collect all `BoundRegionKind`s and `ty::Bound`s. In the second pass, we then
/// replace `BrNamed` into `BrAnon`. The two separate passes are important,
/// since we can only replace `BrNamed` with `BrAnon`s with indices *after* all
/// "real" `BrAnon`s.
///
/// It's important to note that because of prior substitution, we may have
/// late-bound regions, even outside of fn contexts, since this is the best way
/// to prep types for chalk lowering.
pub(crate) fn collect_bound_vars<'tcx, T: TypeFoldable<'tcx>>(
    interner: RustInterner<'tcx>,
    tcx: TyCtxt<'tcx>,
    ty: Binder<'tcx, T>,
) -> (T, chalk_ir::VariableKinds<RustInterner<'tcx>>, BTreeMap<DefId, u32>) {
    let mut bound_vars_collector = BoundVarsCollector::new();
    ty.as_ref().skip_binder().visit_with(&mut bound_vars_collector);
    let mut parameters = bound_vars_collector.parameters;
    let named_parameters: BTreeMap<DefId, u32> = bound_vars_collector
        .named_parameters
        .into_iter()
        .enumerate()
        .map(|(i, def_id)| (def_id, (i + parameters.len()) as u32))
        .collect();

    let mut bound_var_substitutor = NamedBoundVarSubstitutor::new(tcx, &named_parameters);
    let new_ty = ty.skip_binder().fold_with(&mut bound_var_substitutor);

    for var in named_parameters.values() {
        parameters.insert(*var, chalk_ir::VariableKind::Lifetime);
    }

    (0..parameters.len()).for_each(|i| {
        parameters
            .get(&(i as u32))
            .or_else(|| bug!("Skipped bound var index: parameters={:?}", parameters));
    });

    let binders =
        chalk_ir::VariableKinds::from_iter(interner, parameters.into_iter().map(|(_, v)| v));

    (new_ty, binders, named_parameters)
}

pub(crate) struct BoundVarsCollector<'tcx> {
    binder_index: ty::DebruijnIndex,
    pub(crate) parameters: BTreeMap<u32, chalk_ir::VariableKind<RustInterner<'tcx>>>,
    pub(crate) named_parameters: Vec<DefId>,
}

impl<'tcx> BoundVarsCollector<'tcx> {
    pub(crate) fn new() -> Self {
        BoundVarsCollector {
            binder_index: ty::INNERMOST,
            parameters: BTreeMap::new(),
            named_parameters: vec![],
        }
    }
}

impl<'tcx> TypeVisitor<'tcx> for BoundVarsCollector<'tcx> {
    fn visit_binder<T: TypeVisitable<'tcx>>(
        &mut self,
        t: &Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        self.binder_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *t.kind() {
            ty::Bound(debruijn, bound_ty) if debruijn == self.binder_index => {
                match self.parameters.entry(bound_ty.var.as_u32()) {
                    Entry::Vacant(entry) => {
                        entry.insert(chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General));
                    }
                    Entry::Occupied(entry) => match entry.get() {
                        chalk_ir::VariableKind::Ty(_) => {}
                        _ => panic!(),
                    },
                }
            }

            _ => (),
        };

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *r {
            ty::ReLateBound(index, br) if index == self.binder_index => match br.kind {
                ty::BoundRegionKind::BrNamed(def_id, _name) => {
                    if !self.named_parameters.iter().any(|d| *d == def_id) {
                        self.named_parameters.push(def_id);
                    }
                }

                ty::BoundRegionKind::BrAnon(var, _) => match self.parameters.entry(var) {
                    Entry::Vacant(entry) => {
                        entry.insert(chalk_ir::VariableKind::Lifetime);
                    }
                    Entry::Occupied(entry) => match entry.get() {
                        chalk_ir::VariableKind::Lifetime => {}
                        _ => panic!(),
                    },
                },

                ty::BoundRegionKind::BrEnv => unimplemented!(),
            },

            ty::ReEarlyBound(_re) => {
                // FIXME(chalk): jackh726 - I think we should always have already
                // substituted away `ReEarlyBound`s for `ReLateBound`s, but need to confirm.
                unimplemented!();
            }

            _ => (),
        };

        r.super_visit_with(self)
    }
}

/// This is used to replace `BoundRegionKind::BrNamed` with `BoundRegionKind::BrAnon`.
/// Note: we assume that we will always have room for more bound vars. (i.e. we
/// won't ever hit the `u32` limit in `BrAnon`s).
struct NamedBoundVarSubstitutor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    binder_index: ty::DebruijnIndex,
    named_parameters: &'a BTreeMap<DefId, u32>,
}

impl<'a, 'tcx> NamedBoundVarSubstitutor<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, named_parameters: &'a BTreeMap<DefId, u32>) -> Self {
        NamedBoundVarSubstitutor { tcx, binder_index: ty::INNERMOST, named_parameters }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for NamedBoundVarSubstitutor<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: Binder<'tcx, T>) -> Binder<'tcx, T> {
        self.binder_index.shift_in(1);
        let result = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn fold_region(&mut self, r: Region<'tcx>) -> Region<'tcx> {
        match *r {
            ty::ReLateBound(index, br) if index == self.binder_index => match br.kind {
                ty::BrNamed(def_id, _name) => match self.named_parameters.get(&def_id) {
                    Some(idx) => {
                        let new_br = ty::BoundRegion { var: br.var, kind: ty::BrAnon(*idx, None) };
                        return self.tcx.mk_region(ty::ReLateBound(index, new_br));
                    }
                    None => panic!("Missing `BrNamed`."),
                },
                ty::BrEnv => unimplemented!(),
                ty::BrAnon(..) => {}
            },
            _ => (),
        };

        r.super_fold_with(self)
    }
}

/// Used to substitute `Param`s with placeholders. We do this since Chalk
/// have a notion of `Param`s.
pub(crate) struct ParamsSubstitutor<'tcx> {
    tcx: TyCtxt<'tcx>,
    binder_index: ty::DebruijnIndex,
    list: Vec<rustc_middle::ty::ParamTy>,
    next_ty_placeholder: usize,
    pub(crate) params: rustc_data_structures::fx::FxHashMap<u32, rustc_middle::ty::ParamTy>,
    pub(crate) named_regions: BTreeMap<DefId, u32>,
}

impl<'tcx> ParamsSubstitutor<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, next_ty_placeholder: usize) -> Self {
        ParamsSubstitutor {
            tcx,
            binder_index: ty::INNERMOST,
            list: vec![],
            next_ty_placeholder,
            params: rustc_data_structures::fx::FxHashMap::default(),
            named_regions: BTreeMap::default(),
        }
    }
}

impl<'tcx> TypeFolder<'tcx> for ParamsSubstitutor<'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: Binder<'tcx, T>) -> Binder<'tcx, T> {
        self.binder_index.shift_in(1);
        let result = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Param(param) => match self.list.iter().position(|r| r == &param) {
                Some(idx) => self.tcx.mk_ty(ty::Placeholder(ty::PlaceholderType {
                    universe: ty::UniverseIndex::from_usize(0),
                    name: ty::BoundTyKind::Anon(idx as u32),
                })),
                None => {
                    self.list.push(param);
                    let idx = self.list.len() - 1 + self.next_ty_placeholder;
                    self.params.insert(idx as u32, param);
                    self.tcx.mk_ty(ty::Placeholder(ty::PlaceholderType {
                        universe: ty::UniverseIndex::from_usize(0),
                        name: ty::BoundTyKind::Anon(idx as u32),
                    }))
                }
            },
            _ => t.super_fold_with(self),
        }
    }

    fn fold_region(&mut self, r: Region<'tcx>) -> Region<'tcx> {
        match *r {
            // FIXME(chalk) - jackh726 - this currently isn't hit in any tests,
            // since canonicalization will already change these to canonical
            // variables (ty::ReLateBound).
            ty::ReEarlyBound(_re) => match self.named_regions.get(&_re.def_id) {
                Some(idx) => {
                    let br = ty::BoundRegion {
                        var: ty::BoundVar::from_u32(*idx),
                        kind: ty::BrAnon(*idx, None),
                    };
                    self.tcx.mk_region(ty::ReLateBound(self.binder_index, br))
                }
                None => {
                    let idx = self.named_regions.len() as u32;
                    let br = ty::BoundRegion {
                        var: ty::BoundVar::from_u32(idx),
                        kind: ty::BrAnon(idx, None),
                    };
                    self.named_regions.insert(_re.def_id, idx);
                    self.tcx.mk_region(ty::ReLateBound(self.binder_index, br))
                }
            },

            _ => r.super_fold_with(self),
        }
    }
}

pub(crate) struct ReverseParamsSubstitutor<'tcx> {
    tcx: TyCtxt<'tcx>,
    params: rustc_data_structures::fx::FxHashMap<u32, rustc_middle::ty::ParamTy>,
}

impl<'tcx> ReverseParamsSubstitutor<'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        params: rustc_data_structures::fx::FxHashMap<u32, rustc_middle::ty::ParamTy>,
    ) -> Self {
        Self { tcx, params }
    }
}

impl<'tcx> TypeFolder<'tcx> for ReverseParamsSubstitutor<'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Placeholder(ty::PlaceholderType { universe: ty::UniverseIndex::ROOT, name }) => {
                match self.params.get(&name.expect_anon()) {
                    Some(param) => self.tcx.mk_ty(ty::Param(*param)),
                    None => t,
                }
            }

            _ => t.super_fold_with(self),
        }
    }
}

/// Used to collect `Placeholder`s.
pub(crate) struct PlaceholdersCollector {
    universe_index: ty::UniverseIndex,
    pub(crate) next_ty_placeholder: usize,
    pub(crate) next_anon_region_placeholder: u32,
}

impl PlaceholdersCollector {
    pub(crate) fn new() -> Self {
        PlaceholdersCollector {
            universe_index: ty::UniverseIndex::ROOT,
            next_ty_placeholder: 0,
            next_anon_region_placeholder: 0,
        }
    }
}

impl<'tcx> TypeVisitor<'tcx> for PlaceholdersCollector {
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        match t.kind() {
            ty::Placeholder(p) if p.universe == self.universe_index => {
                self.next_ty_placeholder =
                    self.next_ty_placeholder.max(p.name.expect_anon() as usize + 1);
            }

            _ => (),
        };

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *r {
            ty::RePlaceholder(p) if p.universe == self.universe_index => {
                if let ty::BoundRegionKind::BrAnon(anon, _) = p.name {
                    self.next_anon_region_placeholder = self.next_anon_region_placeholder.max(anon);
                }
                // FIXME: This doesn't seem to handle BrNamed at all?
            }

            _ => (),
        };

        r.super_visit_with(self)
    }
}
