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
//! [Chalk book](http://rust-lang.github.io/chalk/book/types/rust_types.html).
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

use rustc_middle::traits::{
    ChalkEnvironmentAndGoal, ChalkEnvironmentClause, ChalkRustDefId as RustDefId,
    ChalkRustInterner as RustInterner,
};
use rustc_middle::ty::fold::TypeFolder;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, SubstsRef};
use rustc_middle::ty::{
    self, Binder, BoundRegion, Region, RegionKind, Ty, TyCtxt, TyKind, TypeFoldable, TypeVisitor,
};
use rustc_span::def_id::DefId;

use std::collections::btree_map::{BTreeMap, Entry};

/// Essentially an `Into` with a `&RustInterner` parameter
crate trait LowerInto<'tcx, T> {
    /// Lower a rustc construct (e.g., `ty::TraitPredicate`) to a chalk type, consuming `self`.
    fn lower_into(self, interner: &RustInterner<'tcx>) -> T;
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Substitution<RustInterner<'tcx>>> for SubstsRef<'tcx> {
    fn lower_into(
        self,
        interner: &RustInterner<'tcx>,
    ) -> chalk_ir::Substitution<RustInterner<'tcx>> {
        chalk_ir::Substitution::from(interner, self.iter().map(|s| s.lower_into(interner)))
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::AliasTy<RustInterner<'tcx>>> for ty::ProjectionTy<'tcx> {
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::AliasTy<RustInterner<'tcx>> {
        chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
            associated_ty_id: chalk_ir::AssocTypeId(RustDefId::AssocTy(self.item_def_id)),
            substitution: self.substs.lower_into(interner),
        })
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::InEnvironment<chalk_ir::Goal<RustInterner<'tcx>>>>
    for ChalkEnvironmentAndGoal<'tcx>
{
    fn lower_into(
        self,
        interner: &RustInterner<'tcx>,
    ) -> chalk_ir::InEnvironment<chalk_ir::Goal<RustInterner<'tcx>>> {
        let clauses = self.environment.into_iter().filter_map(|clause| match clause {
            ChalkEnvironmentClause::Predicate(predicate) => {
                match predicate.kind() {
                    ty::PredicateKind::Trait(predicate, _) => {
                        let (predicate, binders, _named_regions) =
                            collect_bound_vars(interner, interner.tcx, predicate);

                        Some(
                            chalk_ir::ProgramClauseData::ForAll(chalk_ir::Binders::new(
                                binders,
                                chalk_ir::ProgramClauseImplication {
                                    consequence: chalk_ir::DomainGoal::FromEnv(
                                        chalk_ir::FromEnv::Trait(
                                            predicate.trait_ref.lower_into(interner),
                                        ),
                                    ),
                                    conditions: chalk_ir::Goals::new(interner),
                                    priority: chalk_ir::ClausePriority::High,
                                },
                            ))
                            .intern(interner),
                        )
                    }
                    // FIXME(chalk): need to add RegionOutlives/TypeOutlives
                    ty::PredicateKind::RegionOutlives(_) => None,
                    ty::PredicateKind::TypeOutlives(_) => None,
                    ty::PredicateKind::Projection(predicate) => {
                        let (predicate, binders, _named_regions) =
                            collect_bound_vars(interner, interner.tcx, predicate);

                        Some(
                            chalk_ir::ProgramClauseData::ForAll(chalk_ir::Binders::new(
                                binders,
                                chalk_ir::ProgramClauseImplication {
                                    consequence: chalk_ir::DomainGoal::Holds(
                                        chalk_ir::WhereClause::AliasEq(
                                            predicate.lower_into(interner),
                                        ),
                                    ),
                                    conditions: chalk_ir::Goals::new(interner),
                                    priority: chalk_ir::ClausePriority::High,
                                },
                            ))
                            .intern(interner),
                        )
                    }
                    ty::PredicateKind::WellFormed(..)
                    | ty::PredicateKind::ObjectSafe(..)
                    | ty::PredicateKind::ClosureKind(..)
                    | ty::PredicateKind::Subtype(..)
                    | ty::PredicateKind::ConstEvaluatable(..)
                    | ty::PredicateKind::ConstEquate(..) => {
                        bug!("unexpected predicate {}", predicate)
                    }
                }
            }
            ChalkEnvironmentClause::TypeFromEnv(ty) => Some(
                chalk_ir::ProgramClauseData::Implies(chalk_ir::ProgramClauseImplication {
                    consequence: chalk_ir::DomainGoal::FromEnv(chalk_ir::FromEnv::Ty(
                        ty.lower_into(interner),
                    )),
                    conditions: chalk_ir::Goals::new(interner),
                    priority: chalk_ir::ClausePriority::High,
                })
                .intern(interner),
            ),
        });

        let goal: chalk_ir::GoalData<RustInterner<'tcx>> = self.goal.lower_into(&interner);
        chalk_ir::InEnvironment {
            environment: chalk_ir::Environment {
                clauses: chalk_ir::ProgramClauses::from(&interner, clauses),
            },
            goal: goal.intern(&interner),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::GoalData<RustInterner<'tcx>>> for ty::Predicate<'tcx> {
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::GoalData<RustInterner<'tcx>> {
        match self.kind() {
            ty::PredicateKind::Trait(predicate, _) => predicate.lower_into(interner),
            // FIXME(chalk): we need to register constraints.
            ty::PredicateKind::RegionOutlives(_predicate) => {
                chalk_ir::GoalData::All(chalk_ir::Goals::new(interner))
            }
            ty::PredicateKind::TypeOutlives(_predicate) => {
                chalk_ir::GoalData::All(chalk_ir::Goals::new(interner))
            }
            ty::PredicateKind::Projection(predicate) => predicate.lower_into(interner),
            ty::PredicateKind::WellFormed(arg) => match arg.unpack() {
                GenericArgKind::Type(ty) => match ty.kind {
                    // These types are always WF.
                    ty::Str | ty::Placeholder(..) | ty::Error(_) | ty::Never => {
                        chalk_ir::GoalData::All(chalk_ir::Goals::new(interner))
                    }

                    // FIXME(chalk): Well-formed only if ref lifetime outlives type
                    ty::Ref(..) => chalk_ir::GoalData::All(chalk_ir::Goals::new(interner)),

                    ty::Param(..) => panic!("No Params expected."),

                    // FIXME(chalk) -- ultimately I think this is what we
                    // want to do, and we just have rules for how to prove
                    // `WellFormed` for everything above, instead of
                    // inlining a bit the rules of the proof here.
                    _ => chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::WellFormed(
                        chalk_ir::WellFormed::Ty(ty.lower_into(interner)),
                    )),
                },
                // FIXME(chalk): handle well formed consts
                GenericArgKind::Const(..) => {
                    chalk_ir::GoalData::All(chalk_ir::Goals::new(interner))
                }
                GenericArgKind::Lifetime(lt) => bug!("unexpect well formed predicate: {:?}", lt),
            },

            // FIXME(chalk): other predicates
            //
            // We can defer this, but ultimately we'll want to express
            // some of these in terms of chalk operations.
            ty::PredicateKind::ObjectSafe(..)
            | ty::PredicateKind::ClosureKind(..)
            | ty::PredicateKind::Subtype(..)
            | ty::PredicateKind::ConstEvaluatable(..)
            | ty::PredicateKind::ConstEquate(..) => {
                chalk_ir::GoalData::All(chalk_ir::Goals::new(interner))
            }
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::TraitRef<RustInterner<'tcx>>>
    for rustc_middle::ty::TraitRef<'tcx>
{
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::TraitRef<RustInterner<'tcx>> {
        chalk_ir::TraitRef {
            trait_id: chalk_ir::TraitId(RustDefId::Trait(self.def_id)),
            substitution: self.substs.lower_into(interner),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::GoalData<RustInterner<'tcx>>>
    for ty::PolyTraitPredicate<'tcx>
{
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::GoalData<RustInterner<'tcx>> {
        let (ty, binders, _named_regions) = collect_bound_vars(interner, interner.tcx, &self);

        chalk_ir::GoalData::Quantified(
            chalk_ir::QuantifierKind::ForAll,
            chalk_ir::Binders::new(
                binders,
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(
                    chalk_ir::WhereClause::Implemented(ty.trait_ref.lower_into(interner)),
                ))
                .intern(interner),
            ),
        )
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::AliasEq<RustInterner<'tcx>>>
    for rustc_middle::ty::ProjectionPredicate<'tcx>
{
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::AliasEq<RustInterner<'tcx>> {
        chalk_ir::AliasEq {
            ty: self.ty.lower_into(interner),
            alias: self.projection_ty.lower_into(interner),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::GoalData<RustInterner<'tcx>>>
    for ty::PolyProjectionPredicate<'tcx>
{
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::GoalData<RustInterner<'tcx>> {
        let (ty, binders, _named_regions) = collect_bound_vars(interner, interner.tcx, &self);

        chalk_ir::GoalData::Quantified(
            chalk_ir::QuantifierKind::ForAll,
            chalk_ir::Binders::new(
                binders,
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(
                    chalk_ir::WhereClause::AliasEq(ty.lower_into(interner)),
                ))
                .intern(interner),
            ),
        )
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Ty<RustInterner<'tcx>>> for Ty<'tcx> {
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::Ty<RustInterner<'tcx>> {
        use chalk_ir::TyData;
        use rustc_ast::ast;
        use TyKind::*;

        let empty = || chalk_ir::Substitution::empty(interner);
        let struct_ty = |def_id| chalk_ir::TypeName::Struct(chalk_ir::StructId(def_id));
        let apply = |name, substitution| {
            TyData::Apply(chalk_ir::ApplicationTy { name, substitution }).intern(interner)
        };
        let int = |i| apply(chalk_ir::TypeName::Scalar(chalk_ir::Scalar::Int(i)), empty());
        let uint = |i| apply(chalk_ir::TypeName::Scalar(chalk_ir::Scalar::Uint(i)), empty());
        let float = |f| apply(chalk_ir::TypeName::Scalar(chalk_ir::Scalar::Float(f)), empty());

        match self.kind {
            Bool => apply(chalk_ir::TypeName::Scalar(chalk_ir::Scalar::Bool), empty()),
            Char => apply(chalk_ir::TypeName::Scalar(chalk_ir::Scalar::Char), empty()),
            Int(ty) => match ty {
                ast::IntTy::Isize => int(chalk_ir::IntTy::Isize),
                ast::IntTy::I8 => int(chalk_ir::IntTy::I8),
                ast::IntTy::I16 => int(chalk_ir::IntTy::I16),
                ast::IntTy::I32 => int(chalk_ir::IntTy::I32),
                ast::IntTy::I64 => int(chalk_ir::IntTy::I64),
                ast::IntTy::I128 => int(chalk_ir::IntTy::I128),
            },
            Uint(ty) => match ty {
                ast::UintTy::Usize => uint(chalk_ir::UintTy::Usize),
                ast::UintTy::U8 => uint(chalk_ir::UintTy::U8),
                ast::UintTy::U16 => uint(chalk_ir::UintTy::U16),
                ast::UintTy::U32 => uint(chalk_ir::UintTy::U32),
                ast::UintTy::U64 => uint(chalk_ir::UintTy::U64),
                ast::UintTy::U128 => uint(chalk_ir::UintTy::U128),
            },
            Float(ty) => match ty {
                ast::FloatTy::F32 => float(chalk_ir::FloatTy::F32),
                ast::FloatTy::F64 => float(chalk_ir::FloatTy::F64),
            },
            Adt(def, substs) => {
                apply(struct_ty(RustDefId::Adt(def.did)), substs.lower_into(interner))
            }
            Foreign(_def_id) => unimplemented!(),
            Str => apply(struct_ty(RustDefId::Str), empty()),
            Array(ty, _) => apply(
                struct_ty(RustDefId::Array),
                chalk_ir::Substitution::from1(
                    interner,
                    chalk_ir::ParameterKind::Ty(ty.lower_into(interner)).intern(interner),
                ),
            ),
            Slice(ty) => apply(
                struct_ty(RustDefId::Slice),
                chalk_ir::Substitution::from1(
                    interner,
                    chalk_ir::ParameterKind::Ty(ty.lower_into(interner)).intern(interner),
                ),
            ),
            RawPtr(_) => apply(struct_ty(RustDefId::RawPtr), empty()),
            Ref(region, ty, mutability) => apply(
                struct_ty(RustDefId::Ref(mutability)),
                chalk_ir::Substitution::from(
                    interner,
                    [
                        chalk_ir::ParameterKind::Lifetime(region.lower_into(interner))
                            .intern(interner),
                        chalk_ir::ParameterKind::Ty(ty.lower_into(interner)).intern(interner),
                    ]
                    .iter(),
                ),
            ),
            FnDef(def_id, _) => apply(struct_ty(RustDefId::FnDef(def_id)), empty()),
            FnPtr(sig) => {
                let (inputs_and_outputs, binders, _named_regions) =
                    collect_bound_vars(interner, interner.tcx, &sig.inputs_and_output());
                TyData::Function(chalk_ir::Fn {
                    num_binders: binders.len(interner),
                    substitution: chalk_ir::Substitution::from(
                        interner,
                        inputs_and_outputs.iter().map(|ty| {
                            chalk_ir::ParameterKind::Ty(ty.lower_into(interner)).intern(interner)
                        }),
                    ),
                })
                .intern(interner)
            }
            Dynamic(_, _) => unimplemented!(),
            Closure(_def_id, _) => unimplemented!(),
            Generator(_def_id, _substs, _) => unimplemented!(),
            GeneratorWitness(_) => unimplemented!(),
            Never => apply(struct_ty(RustDefId::Never), empty()),
            Tuple(substs) => {
                apply(chalk_ir::TypeName::Tuple(substs.len()), substs.lower_into(interner))
            }
            Projection(proj) => TyData::Alias(proj.lower_into(interner)).intern(interner),
            Opaque(_def_id, _substs) => unimplemented!(),
            // This should have been done eagerly prior to this, and all Params
            // should have been substituted to placeholders
            Param(_) => panic!("Lowering Param when not expected."),
            Bound(db, bound) => TyData::BoundVar(chalk_ir::BoundVar::new(
                chalk_ir::DebruijnIndex::new(db.as_u32()),
                bound.var.index(),
            ))
            .intern(interner),
            Placeholder(_placeholder) => TyData::Placeholder(chalk_ir::PlaceholderIndex {
                ui: chalk_ir::UniverseIndex { counter: _placeholder.universe.as_usize() },
                idx: _placeholder.name.as_usize(),
            })
            .intern(interner),
            Infer(_infer) => unimplemented!(),
            Error(_) => unimplemented!(),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Lifetime<RustInterner<'tcx>>> for Region<'tcx> {
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::Lifetime<RustInterner<'tcx>> {
        use rustc_middle::ty::RegionKind::*;

        match self {
            ReEarlyBound(_) => {
                panic!("Should have already been substituted.");
            }
            ReLateBound(db, br) => match br {
                ty::BoundRegion::BrAnon(var) => {
                    chalk_ir::LifetimeData::BoundVar(chalk_ir::BoundVar::new(
                        chalk_ir::DebruijnIndex::new(db.as_u32()),
                        *var as usize,
                    ))
                    .intern(interner)
                }
                ty::BoundRegion::BrNamed(_def_id, _name) => unimplemented!(),
                ty::BrEnv => unimplemented!(),
            },
            ReFree(_) => unimplemented!(),
            ReStatic => unimplemented!(),
            ReVar(_) => unimplemented!(),
            RePlaceholder(placeholder_region) => {
                chalk_ir::LifetimeData::Placeholder(chalk_ir::PlaceholderIndex {
                    ui: chalk_ir::UniverseIndex { counter: placeholder_region.universe.index() },
                    idx: 0,
                })
                .intern(interner)
            }
            ReEmpty(_) => unimplemented!(),
            ReErased => unimplemented!(),
        }
    }
}

impl<'tcx> LowerInto<'tcx, chalk_ir::Parameter<RustInterner<'tcx>>> for GenericArg<'tcx> {
    fn lower_into(self, interner: &RustInterner<'tcx>) -> chalk_ir::Parameter<RustInterner<'tcx>> {
        match self.unpack() {
            ty::subst::GenericArgKind::Type(ty) => {
                chalk_ir::ParameterKind::Ty(ty.lower_into(interner))
            }
            ty::subst::GenericArgKind::Lifetime(lifetime) => {
                chalk_ir::ParameterKind::Lifetime(lifetime.lower_into(interner))
            }
            ty::subst::GenericArgKind::Const(_) => chalk_ir::ParameterKind::Ty(
                chalk_ir::TyData::Apply(chalk_ir::ApplicationTy {
                    name: chalk_ir::TypeName::Tuple(0),
                    substitution: chalk_ir::Substitution::empty(interner),
                })
                .intern(interner),
            ),
        }
        .intern(interner)
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
        interner: &RustInterner<'tcx>,
    ) -> Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>> {
        match &self.kind() {
            ty::PredicateKind::Trait(predicate, _) => {
                let (predicate, binders, _named_regions) =
                    collect_bound_vars(interner, interner.tcx, predicate);

                Some(chalk_ir::Binders::new(
                    binders,
                    chalk_ir::WhereClause::Implemented(predicate.trait_ref.lower_into(interner)),
                ))
            }
            ty::PredicateKind::RegionOutlives(_predicate) => None,
            ty::PredicateKind::TypeOutlives(_predicate) => None,
            ty::PredicateKind::Projection(_predicate) => None,
            ty::PredicateKind::WellFormed(_ty) => None,

            ty::PredicateKind::ObjectSafe(..)
            | ty::PredicateKind::ClosureKind(..)
            | ty::PredicateKind::Subtype(..)
            | ty::PredicateKind::ConstEvaluatable(..)
            | ty::PredicateKind::ConstEquate(..) => bug!("unexpected predicate {}", &self),
        }
    }
}

/// To collect bound vars, we have to do two passes. In the first pass, we
/// collect all `BoundRegion`s and `ty::Bound`s. In the second pass, we then
/// replace `BrNamed` into `BrAnon`. The two separate passes are important,
/// since we can only replace `BrNamed` with `BrAnon`s with indices *after* all
/// "real" `BrAnon`s.
///
/// It's important to note that because of prior substitution, we may have
/// late-bound regions, even outside of fn contexts, since this is the best way
/// to prep types for chalk lowering.
crate fn collect_bound_vars<'a, 'tcx, T: TypeFoldable<'tcx>>(
    interner: &RustInterner<'tcx>,
    tcx: TyCtxt<'tcx>,
    ty: &'a Binder<T>,
) -> (T, chalk_ir::ParameterKinds<RustInterner<'tcx>>, BTreeMap<DefId, u32>) {
    let mut bound_vars_collector = BoundVarsCollector::new();
    ty.skip_binder().visit_with(&mut bound_vars_collector);
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
        parameters.insert(*var, chalk_ir::ParameterKind::Lifetime(()));
    }

    (0..parameters.len()).for_each(|i| {
        parameters.get(&(i as u32)).expect("Skipped bound var index.");
    });

    let binders = chalk_ir::ParameterKinds::from(interner, parameters.into_iter().map(|(_, v)| v));

    (new_ty, binders, named_parameters)
}

crate struct BoundVarsCollector {
    binder_index: ty::DebruijnIndex,
    crate parameters: BTreeMap<u32, chalk_ir::ParameterKind<()>>,
    crate named_parameters: Vec<DefId>,
}

impl BoundVarsCollector {
    crate fn new() -> Self {
        BoundVarsCollector {
            binder_index: ty::INNERMOST,
            parameters: BTreeMap::new(),
            named_parameters: vec![],
        }
    }
}

impl<'tcx> TypeVisitor<'tcx> for BoundVarsCollector {
    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> bool {
        self.binder_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
        match t.kind {
            ty::Bound(debruijn, bound_ty) if debruijn == self.binder_index => {
                match self.parameters.entry(bound_ty.var.as_u32()) {
                    Entry::Vacant(entry) => {
                        entry.insert(chalk_ir::ParameterKind::Ty(()));
                    }
                    Entry::Occupied(entry) => {
                        entry.get().assert_ty_ref();
                    }
                }
            }

            _ => (),
        };

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: Region<'tcx>) -> bool {
        match r {
            ty::ReLateBound(index, br) if *index == self.binder_index => match br {
                ty::BoundRegion::BrNamed(def_id, _name) => {
                    if self.named_parameters.iter().find(|d| *d == def_id).is_none() {
                        self.named_parameters.push(*def_id);
                    }
                }

                ty::BoundRegion::BrAnon(var) => match self.parameters.entry(*var) {
                    Entry::Vacant(entry) => {
                        entry.insert(chalk_ir::ParameterKind::Lifetime(()));
                    }
                    Entry::Occupied(entry) => {
                        entry.get().assert_lifetime_ref();
                    }
                },

                ty::BrEnv => unimplemented!(),
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

/// This is used to replace `BoundRegion::BrNamed` with `BoundRegion::BrAnon`.
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

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> Binder<T> {
        self.binder_index.shift_in(1);
        let result = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        t.super_fold_with(self)
    }

    fn fold_region(&mut self, r: Region<'tcx>) -> Region<'tcx> {
        match r {
            ty::ReLateBound(index, br) if *index == self.binder_index => match br {
                ty::BoundRegion::BrNamed(def_id, _name) => {
                    match self.named_parameters.get(def_id) {
                        Some(idx) => {
                            return self.tcx.mk_region(RegionKind::ReLateBound(
                                *index,
                                BoundRegion::BrAnon(*idx),
                            ));
                        }
                        None => panic!("Missing `BrNamed`."),
                    }
                }
                ty::BrEnv => unimplemented!(),
                ty::BoundRegion::BrAnon(_) => {}
            },
            _ => (),
        };

        r.super_fold_with(self)
    }
}

/// Used to substitute `Param`s with placeholders. We do this since Chalk
/// have a notion of `Param`s.
crate struct ParamsSubstitutor<'tcx> {
    tcx: TyCtxt<'tcx>,
    binder_index: ty::DebruijnIndex,
    list: Vec<rustc_middle::ty::ParamTy>,
    crate params: rustc_data_structures::fx::FxHashMap<usize, rustc_middle::ty::ParamTy>,
    crate named_regions: BTreeMap<DefId, u32>,
}

impl<'tcx> ParamsSubstitutor<'tcx> {
    crate fn new(tcx: TyCtxt<'tcx>) -> Self {
        ParamsSubstitutor {
            tcx,
            binder_index: ty::INNERMOST,
            list: vec![],
            params: rustc_data_structures::fx::FxHashMap::default(),
            named_regions: BTreeMap::default(),
        }
    }
}

impl<'tcx> TypeFolder<'tcx> for ParamsSubstitutor<'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> Binder<T> {
        self.binder_index.shift_in(1);
        let result = t.super_fold_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.kind {
            // FIXME(chalk): currently we convert params to placeholders starting at
            // index `0`. To support placeholders, we'll actually need to do a
            // first pass to collect placeholders. Then we can insert params after.
            ty::Placeholder(_) => unimplemented!(),
            ty::Param(param) => match self.list.iter().position(|r| r == &param) {
                Some(_idx) => self.tcx.mk_ty(ty::Placeholder(ty::PlaceholderType {
                    universe: ty::UniverseIndex::from_usize(0),
                    name: ty::BoundVar::from_usize(_idx),
                })),
                None => {
                    self.list.push(param);
                    let idx = self.list.len() - 1;
                    self.params.insert(idx, param);
                    self.tcx.mk_ty(ty::Placeholder(ty::PlaceholderType {
                        universe: ty::UniverseIndex::from_usize(0),
                        name: ty::BoundVar::from_usize(idx),
                    }))
                }
            },

            _ => t.super_fold_with(self),
        }
    }

    fn fold_region(&mut self, r: Region<'tcx>) -> Region<'tcx> {
        match r {
            // FIXME(chalk) - jackh726 - this currently isn't hit in any tests.
            // This covers any region variables in a goal, right?
            ty::ReEarlyBound(_re) => match self.named_regions.get(&_re.def_id) {
                Some(idx) => self.tcx.mk_region(RegionKind::ReLateBound(
                    self.binder_index,
                    BoundRegion::BrAnon(*idx),
                )),
                None => {
                    let idx = self.named_regions.len() as u32;
                    self.named_regions.insert(_re.def_id, idx);
                    self.tcx.mk_region(RegionKind::ReLateBound(
                        self.binder_index,
                        BoundRegion::BrAnon(idx),
                    ))
                }
            },

            _ => r.super_fold_with(self),
        }
    }
}
