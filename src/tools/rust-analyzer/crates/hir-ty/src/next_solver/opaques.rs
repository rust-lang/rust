//! Things related to opaques in the next-trait-solver.

use intern::Interned;
use rustc_ast_ir::try_visit;

use crate::next_solver::SolverDefId;

use super::{CanonicalVarKind, DbInterner, interned_vec_nolifetime_salsa};

pub type OpaqueTypeKey<'db> = rustc_type_ir::OpaqueTypeKey<DbInterner<'db>>;
pub type PredefinedOpaquesData<'db> = rustc_type_ir::solve::PredefinedOpaquesData<DbInterner<'db>>;
pub type ExternalConstraintsData<'db> =
    rustc_type_ir::solve::ExternalConstraintsData<DbInterner<'db>>;

#[salsa::interned(constructor = new_, debug)]
pub struct PredefinedOpaques<'db> {
    #[returns(ref)]
    kind_: rustc_type_ir::solve::PredefinedOpaquesData<DbInterner<'db>>,
}

impl<'db> PredefinedOpaques<'db> {
    pub fn new(interner: DbInterner<'db>, data: PredefinedOpaquesData<'db>) -> Self {
        PredefinedOpaques::new_(interner.db(), data)
    }

    pub fn inner(&self) -> &PredefinedOpaquesData<'db> {
        crate::with_attached_db(|db| {
            let inner = self.kind_(db);
            // SAFETY: ¯\_(ツ)_/¯
            unsafe { std::mem::transmute(inner) }
        })
    }
}

impl<'db> rustc_type_ir::TypeVisitable<DbInterner<'db>> for PredefinedOpaques<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        self.opaque_types.visit_with(visitor)
    }
}

impl<'db> rustc_type_ir::TypeFoldable<DbInterner<'db>> for PredefinedOpaques<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(PredefinedOpaques::new(
            folder.cx(),
            PredefinedOpaquesData {
                opaque_types: self
                    .opaque_types
                    .iter()
                    .cloned()
                    .map(|opaque| opaque.try_fold_with(folder))
                    .collect::<Result<_, F::Error>>()?,
            },
        ))
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        PredefinedOpaques::new(
            folder.cx(),
            PredefinedOpaquesData {
                opaque_types: self
                    .opaque_types
                    .iter()
                    .cloned()
                    .map(|opaque| opaque.fold_with(folder))
                    .collect(),
            },
        )
    }
}

impl<'db> std::ops::Deref for PredefinedOpaques<'db> {
    type Target = PredefinedOpaquesData<'db>;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

interned_vec_nolifetime_salsa!(SolverDefIds, SolverDefId);

#[salsa::interned(constructor = new_, debug)]
pub struct ExternalConstraints<'db> {
    #[returns(ref)]
    kind_: rustc_type_ir::solve::ExternalConstraintsData<DbInterner<'db>>,
}

impl<'db> ExternalConstraints<'db> {
    pub fn new(interner: DbInterner<'db>, data: ExternalConstraintsData<'db>) -> Self {
        ExternalConstraints::new_(interner.db(), data)
    }

    pub fn inner(&self) -> &ExternalConstraintsData<'db> {
        crate::with_attached_db(|db| {
            let inner = self.kind_(db);
            // SAFETY: ¯\_(ツ)_/¯
            unsafe { std::mem::transmute(inner) }
        })
    }
}

impl<'db> std::ops::Deref for ExternalConstraints<'db> {
    type Target = ExternalConstraintsData<'db>;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

impl<'db> rustc_type_ir::TypeVisitable<DbInterner<'db>> for ExternalConstraints<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        try_visit!(self.region_constraints.visit_with(visitor));
        try_visit!(self.opaque_types.visit_with(visitor));
        self.normalization_nested_goals.visit_with(visitor)
    }
}

impl<'db> rustc_type_ir::TypeFoldable<DbInterner<'db>> for ExternalConstraints<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(ExternalConstraints::new(
            folder.cx(),
            ExternalConstraintsData {
                region_constraints: self.region_constraints.clone().try_fold_with(folder)?,
                opaque_types: self
                    .opaque_types
                    .iter()
                    .cloned()
                    .map(|opaque| opaque.try_fold_with(folder))
                    .collect::<Result<_, F::Error>>()?,
                normalization_nested_goals: self
                    .normalization_nested_goals
                    .clone()
                    .try_fold_with(folder)?,
            },
        ))
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        ExternalConstraints::new(
            folder.cx(),
            ExternalConstraintsData {
                region_constraints: self.region_constraints.clone().fold_with(folder),
                opaque_types: self
                    .opaque_types
                    .iter()
                    .cloned()
                    .map(|opaque| opaque.fold_with(folder))
                    .collect(),
                normalization_nested_goals: self
                    .normalization_nested_goals
                    .clone()
                    .fold_with(folder),
            },
        )
    }
}
