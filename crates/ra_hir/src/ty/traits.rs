//! Chalk integration.
use std::sync::{Arc, Mutex};

use chalk_ir::{TypeId, TraitId, StructId, ImplId, TypeKindId, ProjectionTy, Parameter, Identifier, cast::Cast};
use chalk_rust_ir::{AssociatedTyDatum, TraitDatum, StructDatum, ImplDatum};

use crate::{Crate, Trait, db::HirDatabase, HasGenericParams, ImplBlock};
use super::{TraitRef, Ty, ApplicationTy, TypeCtor, Substs, infer::Canonical};

#[derive(Debug, Copy, Clone)]
struct ChalkContext<'a, DB> {
    db: &'a DB,
    krate: Crate,
}

pub(crate) trait ToChalk {
    type Chalk;
    fn to_chalk(self, db: &impl HirDatabase) -> Self::Chalk;
    fn from_chalk(db: &impl HirDatabase, chalk: Self::Chalk) -> Self;
}

pub(crate) fn from_chalk<T, ChalkT>(db: &impl HirDatabase, chalk: ChalkT) -> T
where
    T: ToChalk<Chalk = ChalkT>,
{
    T::from_chalk(db, chalk)
}

impl ToChalk for Ty {
    type Chalk = chalk_ir::Ty;
    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Ty {
        match self {
            Ty::Apply(apply_ty) => chalk_ir::Ty::Apply(apply_ty.to_chalk(db)),
            Ty::Param { idx, .. } => {
                chalk_ir::PlaceholderIndex { ui: chalk_ir::UniverseIndex::ROOT, idx: idx as usize }
                    .to_ty()
            }
            Ty::Bound(idx) => chalk_ir::Ty::BoundVar(idx as usize),
            Ty::Infer(_infer_ty) => panic!("uncanonicalized infer ty"),
            Ty::Unknown => unimplemented!(), // TODO turn into placeholder?
        }
    }
    fn from_chalk(db: &impl HirDatabase, chalk: chalk_ir::Ty) -> Self {
        match chalk {
            chalk_ir::Ty::Apply(apply_ty) => {
                match apply_ty.name {
                    // FIXME handle TypeKindId::Trait/Type here
                    chalk_ir::TypeName::TypeKindId(_) => Ty::Apply(from_chalk(db, apply_ty)),
                    chalk_ir::TypeName::AssociatedType(_) => unimplemented!(),
                    chalk_ir::TypeName::Placeholder(idx) => {
                        assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
                        Ty::Param { idx: idx.idx as u32, name: crate::Name::missing() }
                    }
                }
            }
            chalk_ir::Ty::Projection(_) => unimplemented!(),
            chalk_ir::Ty::UnselectedProjection(_) => unimplemented!(),
            chalk_ir::Ty::ForAll(_) => unimplemented!(),
            chalk_ir::Ty::BoundVar(idx) => Ty::Bound(idx as u32),
            chalk_ir::Ty::InferenceVar(_iv) => panic!("unexpected chalk infer ty"),
        }
    }
}

impl ToChalk for ApplicationTy {
    type Chalk = chalk_ir::ApplicationTy;

    fn to_chalk(self: ApplicationTy, db: &impl HirDatabase) -> chalk_ir::ApplicationTy {
        let struct_id = self.ctor.to_chalk(db);
        let name = chalk_ir::TypeName::TypeKindId(struct_id.into());
        let parameters = self.parameters.to_chalk(db);
        chalk_ir::ApplicationTy { name, parameters }
    }

    fn from_chalk(db: &impl HirDatabase, apply_ty: chalk_ir::ApplicationTy) -> ApplicationTy {
        let ctor = match apply_ty.name {
            chalk_ir::TypeName::TypeKindId(chalk_ir::TypeKindId::StructId(struct_id)) => {
                from_chalk(db, struct_id)
            }
            chalk_ir::TypeName::TypeKindId(_) => unimplemented!(),
            chalk_ir::TypeName::Placeholder(_) => unimplemented!(),
            chalk_ir::TypeName::AssociatedType(_) => unimplemented!(),
        };
        let parameters = from_chalk(db, apply_ty.parameters);
        ApplicationTy { ctor, parameters }
    }
}

impl ToChalk for Substs {
    type Chalk = Vec<chalk_ir::Parameter>;

    fn to_chalk(self, db: &impl HirDatabase) -> Vec<chalk_ir::Parameter> {
        self.iter().map(|ty| ty.clone().to_chalk(db).cast()).collect()
    }

    fn from_chalk(db: &impl HirDatabase, parameters: Vec<chalk_ir::Parameter>) -> Substs {
        parameters
            .into_iter()
            .map(|p| match p {
                chalk_ir::Parameter(chalk_ir::ParameterKind::Ty(ty)) => from_chalk(db, ty),
                chalk_ir::Parameter(chalk_ir::ParameterKind::Lifetime(_)) => unimplemented!(),
            })
            .collect::<Vec<_>>()
            .into()
    }
}

impl ToChalk for TraitRef {
    type Chalk = chalk_ir::TraitRef;

    fn to_chalk(self: TraitRef, db: &impl HirDatabase) -> chalk_ir::TraitRef {
        let trait_id = self.trait_.to_chalk(db);
        let parameters = self.substs.to_chalk(db);
        chalk_ir::TraitRef { trait_id, parameters }
    }

    fn from_chalk(db: &impl HirDatabase, trait_ref: chalk_ir::TraitRef) -> Self {
        let trait_ = from_chalk(db, trait_ref.trait_id);
        let substs = from_chalk(db, trait_ref.parameters);
        TraitRef { trait_, substs }
    }
}

impl ToChalk for Trait {
    type Chalk = TraitId;

    fn to_chalk(self, _db: &impl HirDatabase) -> TraitId {
        self.id.into()
    }

    fn from_chalk(_db: &impl HirDatabase, trait_id: TraitId) -> Trait {
        Trait { id: trait_id.into() }
    }
}

impl ToChalk for TypeCtor {
    type Chalk = chalk_ir::StructId;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::StructId {
        db.intern_type_ctor(self).into()
    }

    fn from_chalk(db: &impl HirDatabase, struct_id: chalk_ir::StructId) -> TypeCtor {
        db.lookup_intern_type_ctor(struct_id.into())
    }
}

impl ToChalk for ImplBlock {
    type Chalk = chalk_ir::ImplId;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::ImplId {
        db.intern_impl_block(self).into()
    }

    fn from_chalk(db: &impl HirDatabase, impl_id: chalk_ir::ImplId) -> ImplBlock {
        db.lookup_intern_impl_block(impl_id.into())
    }
}

fn make_binders<T>(value: T, num_vars: usize) -> chalk_ir::Binders<T> {
    chalk_ir::Binders {
        value,
        binders: std::iter::repeat(chalk_ir::ParameterKind::Ty(())).take(num_vars).collect(),
    }
}

impl<'a, DB> chalk_solve::RustIrDatabase for ChalkContext<'a, DB>
where
    DB: HirDatabase,
{
    fn associated_ty_data(&self, _ty: TypeId) -> Arc<AssociatedTyDatum> {
        unimplemented!()
    }
    fn trait_datum(&self, trait_id: TraitId) -> Arc<TraitDatum> {
        eprintln!("trait_datum {:?}", trait_id);
        let trait_: Trait = from_chalk(self.db, trait_id);
        let generic_params = trait_.generic_params(self.db);
        let bound_vars = Substs::bound_vars(&generic_params);
        let trait_ref = trait_.trait_ref(self.db).subst(&bound_vars).to_chalk(self.db);
        let flags = chalk_rust_ir::TraitFlags {
            // FIXME set these flags correctly
            auto: false,
            marker: false,
            upstream: trait_.module(self.db).krate(self.db) != Some(self.krate),
            fundamental: false,
        };
        let where_clauses = Vec::new(); // FIXME add where clauses
        let trait_datum_bound = chalk_rust_ir::TraitDatumBound { trait_ref, where_clauses, flags };
        let trait_datum = TraitDatum { binders: make_binders(trait_datum_bound, bound_vars.len()) };
        Arc::new(trait_datum)
    }
    fn struct_datum(&self, struct_id: StructId) -> Arc<StructDatum> {
        eprintln!("struct_datum {:?}", struct_id);
        let type_ctor = from_chalk(self.db, struct_id);
        // TODO might be nicer if we can create a fake GenericParams for the TypeCtor
        let (num_params, upstream) = match type_ctor {
            TypeCtor::Bool
            | TypeCtor::Char
            | TypeCtor::Int(_)
            | TypeCtor::Float(_)
            | TypeCtor::Never
            | TypeCtor::Str => (0, true),
            TypeCtor::Slice | TypeCtor::Array | TypeCtor::RawPtr(_) | TypeCtor::Ref(_) => (1, true),
            TypeCtor::FnPtr | TypeCtor::Tuple => unimplemented!(), // FIXME tuples and FnPtr are currently variadic... we need to make the parameter number explicit
            TypeCtor::FnDef(_) => unimplemented!(),
            TypeCtor::Adt(adt) => {
                let generic_params = adt.generic_params(self.db);
                (
                    generic_params.count_params_including_parent(),
                    adt.krate(self.db) != Some(self.krate),
                )
            }
        };
        let flags = chalk_rust_ir::StructFlags {
            upstream,
            // FIXME set fundamental flag correctly
            fundamental: false,
        };
        let where_clauses = Vec::new(); // FIXME add where clauses
        let ty = ApplicationTy {
            ctor: type_ctor,
            parameters: (0..num_params).map(|i| Ty::Bound(i as u32)).collect::<Vec<_>>().into(),
        };
        let struct_datum_bound = chalk_rust_ir::StructDatumBound {
            self_ty: ty.to_chalk(self.db),
            fields: Vec::new(), // FIXME add fields (only relevant for auto traits)
            where_clauses,
            flags,
        };
        let struct_datum = StructDatum { binders: make_binders(struct_datum_bound, num_params) };
        Arc::new(struct_datum)
    }
    fn impl_datum(&self, impl_id: ImplId) -> Arc<ImplDatum> {
        eprintln!("impl_datum {:?}", impl_id);
        let impl_block: ImplBlock = from_chalk(self.db, impl_id);
        let generic_params = impl_block.generic_params(self.db);
        let bound_vars = Substs::bound_vars(&generic_params);
        let trait_ref = impl_block
            .target_trait_ref(self.db)
            .expect("FIXME handle unresolved impl block trait ref")
            .subst(&bound_vars);
        let impl_type = if impl_block.module().krate(self.db) == Some(self.krate) {
            chalk_rust_ir::ImplType::Local
        } else {
            chalk_rust_ir::ImplType::External
        };
        let impl_datum_bound = chalk_rust_ir::ImplDatumBound {
            // FIXME handle negative impls (impl !Sync for Foo)
            trait_ref: chalk_rust_ir::PolarizedTraitRef::Positive(trait_ref.to_chalk(self.db)),
            where_clauses: Vec::new(),        // FIXME add where clauses
            associated_ty_values: Vec::new(), // FIXME add associated type values
            impl_type,
        };
        let impl_datum = ImplDatum { binders: make_binders(impl_datum_bound, bound_vars.len()) };
        Arc::new(impl_datum)
    }
    fn impls_for_trait(&self, trait_id: TraitId) -> Vec<ImplId> {
        eprintln!("impls_for_trait {:?}", trait_id);
        let trait_ = from_chalk(self.db, trait_id);
        self.db
            .impls_for_trait(self.krate, trait_)
            .iter()
            // FIXME temporary hack -- as long as we're not lowering where clauses
            // correctly, ignore impls with them completely so as to not treat
            // impl<T> Trait for T where T: ... as a blanket impl on all types
            .filter(|impl_block| impl_block.generic_params(self.db).where_predicates.is_empty())
            .map(|impl_block| impl_block.to_chalk(self.db))
            .collect()
    }
    fn impl_provided_for(&self, auto_trait_id: TraitId, struct_id: StructId) -> bool {
        eprintln!("impl_provided_for {:?}, {:?}", auto_trait_id, struct_id);
        false // FIXME
    }
    fn type_name(&self, _id: TypeKindId) -> Identifier {
        unimplemented!()
    }
    fn split_projection<'p>(
        &self,
        projection: &'p ProjectionTy,
    ) -> (Arc<AssociatedTyDatum>, &'p [Parameter], &'p [Parameter]) {
        eprintln!("split_projection {:?}", projection);
        unimplemented!()
    }
}

pub(crate) fn solver(_db: &impl HirDatabase, _krate: Crate) -> Arc<Mutex<chalk_solve::Solver>> {
    // krate parameter is just so we cache a unique solver per crate
    let solver_choice = chalk_solve::SolverChoice::SLG { max_size: 10 };
    Arc::new(Mutex::new(solver_choice.into_solver()))
}

/// Collects impls for the given trait in the whole dependency tree of `krate`.
pub(crate) fn impls_for_trait(
    db: &impl HirDatabase,
    krate: Crate,
    trait_: Trait,
) -> Arc<[ImplBlock]> {
    let mut impls = Vec::new();
    // We call the query recursively here. On the one hand, this means we can
    // reuse results from queries for different crates; on the other hand, this
    // will only ever get called for a few crates near the root of the tree (the
    // ones the user is editing), so this may actually be a waste of memory. I'm
    // doing it like this mainly for simplicity for now.
    for dep in krate.dependencies(db) {
        impls.extend(db.impls_for_trait(dep.krate, trait_).iter());
    }
    let crate_impl_blocks = db.impls_in_crate(krate);
    impls.extend(crate_impl_blocks.lookup_impl_blocks_for_trait(&trait_));
    impls.into()
}

fn solve(
    db: &impl HirDatabase,
    krate: Crate,
    goal: &chalk_ir::UCanonical<chalk_ir::InEnvironment<chalk_ir::Goal>>,
) -> Option<chalk_solve::Solution> {
    let context = ChalkContext { db, krate };
    let solver = db.chalk_solver(krate);
    let solution = solver.lock().unwrap().solve(&context, goal);
    eprintln!("solve({:?}) => {:?}", goal, solution);
    solution
}

/// Something that needs to be proven (by Chalk) during type checking, e.g. that
/// a certain type implements a certain trait. Proving the Obligation might
/// result in additional information about inference variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Obligation {
    /// Prove that a certain type implements a trait (the type is the `Self` type
    /// parameter to the `TraitRef`).
    Trait(TraitRef),
}

/// Check using Chalk whether trait is implemented for given parameters including `Self` type.
pub(crate) fn implements(
    db: &impl HirDatabase,
    krate: Crate,
    trait_ref: Canonical<TraitRef>,
) -> Option<Solution> {
    let goal: chalk_ir::Goal = trait_ref.value.to_chalk(db).cast();
    eprintln!("goal: {:?}", goal);
    let env = chalk_ir::Environment::new();
    let in_env = chalk_ir::InEnvironment::new(&env, goal);
    let parameter = chalk_ir::ParameterKind::Ty(chalk_ir::UniverseIndex::ROOT);
    let canonical =
        chalk_ir::Canonical { value: in_env, binders: vec![parameter; trait_ref.num_vars] };
    // We currently don't deal with universes (I think / hope they're not yet
    // relevant for our use cases?)
    let u_canonical = chalk_ir::UCanonical { canonical, universes: 1 };
    let solution = solve(db, krate, &u_canonical);
    solution.map(|solution| solution_from_chalk(db, solution))
}

fn solution_from_chalk(
    db: &impl HirDatabase,
    solution: chalk_solve::Solution,
) -> Solution {
    let convert_subst = |subst: chalk_ir::Canonical<chalk_ir::Substitution>| {
        let value = subst
            .value
            .parameters
            .into_iter()
            .map(|p| {
                let ty = match p {
                    chalk_ir::Parameter(chalk_ir::ParameterKind::Ty(ty)) => from_chalk(db, ty),
                    chalk_ir::Parameter(chalk_ir::ParameterKind::Lifetime(_)) => unimplemented!(),
                };
                ty
            })
            .collect();
        let result = Canonical { value, num_vars: subst.binders.len() };
        SolutionVariables(result)
    };
    match solution {
        chalk_solve::Solution::Unique(constr_subst) => {
            let subst = chalk_ir::Canonical {
                value: constr_subst.value.subst,
                binders: constr_subst.binders,
            };
            Solution::Unique(convert_subst(subst))
        }
        chalk_solve::Solution::Ambig(chalk_solve::Guidance::Definite(subst)) => {
            Solution::Ambig(Guidance::Definite(convert_subst(subst)))
        }
        chalk_solve::Solution::Ambig(chalk_solve::Guidance::Suggested(subst)) => {
            Solution::Ambig(Guidance::Suggested(convert_subst(subst)))
        }
        chalk_solve::Solution::Ambig(chalk_solve::Guidance::Unknown) => {
            Solution::Ambig(Guidance::Unknown)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SolutionVariables(pub Canonical<Vec<Ty>>);

#[derive(Clone, Debug, PartialEq, Eq)]
/// A (possible) solution for a proposed goal.
pub(crate) enum Solution {
    /// The goal indeed holds, and there is a unique value for all existential
    /// variables.
    Unique(SolutionVariables),

    /// The goal may be provable in multiple ways, but regardless we may have some guidance
    /// for type inference. In this case, we don't return any lifetime
    /// constraints, since we have not "committed" to any particular solution
    /// yet.
    Ambig(Guidance),
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// When a goal holds ambiguously (e.g., because there are multiple possible
/// solutions), we issue a set of *guidance* back to type inference.
pub(crate) enum Guidance {
    /// The existential variables *must* have the given values if the goal is
    /// ever to hold, but that alone isn't enough to guarantee the goal will
    /// actually hold.
    Definite(SolutionVariables),

    /// There are multiple plausible values for the existentials, but the ones
    /// here are suggested as the preferred choice heuristically. These should
    /// be used for inference fallback only.
    Suggested(SolutionVariables),

    /// There's no useful information to feed back to type inference
    Unknown,
}
