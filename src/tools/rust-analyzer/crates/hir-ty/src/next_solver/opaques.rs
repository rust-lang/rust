//! Things related to opaques in the next-trait-solver.

use intern::{Interned, InternedRef, impl_internable};
use macros::GenericTypeVisitable;
use rustc_ast_ir::try_visit;
use rustc_type_ir::inherent::SliceLike;

use crate::next_solver::{impl_foldable_for_interned_slice, interned_slice};

use super::{DbInterner, SolverDefId, Ty};

pub type OpaqueTypeKey<'db> = rustc_type_ir::OpaqueTypeKey<DbInterner<'db>>;

type PredefinedOpaque<'db> = (OpaqueTypeKey<'db>, Ty<'db>);
interned_slice!(
    PredefinedOpaquesStorage,
    PredefinedOpaques,
    StoredPredefinedOpaques,
    predefined_opaques,
    PredefinedOpaque<'db>,
    PredefinedOpaque<'static>,
);
impl_foldable_for_interned_slice!(PredefinedOpaques);

pub type ExternalConstraintsData<'db> =
    rustc_type_ir::solve::ExternalConstraintsData<DbInterner<'db>>;

interned_slice!(
    SolverDefIdsStorage,
    SolverDefIds,
    StoredSolverDefIds,
    def_ids,
    SolverDefId,
    SolverDefId,
);
impl_foldable_for_interned_slice!(SolverDefIds);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternalConstraints<'db> {
    interned: InternedRef<'db, ExternalConstraintsInterned>,
}

#[derive(PartialEq, Eq, Hash, GenericTypeVisitable)]
pub(super) struct ExternalConstraintsInterned(ExternalConstraintsData<'static>);

impl_internable!(gc; ExternalConstraintsInterned);

const _: () = {
    const fn is_copy<T: Copy>() {}
    is_copy::<ExternalConstraints<'static>>();
};

impl<'db> ExternalConstraints<'db> {
    #[inline]
    pub fn new(_interner: DbInterner<'db>, data: ExternalConstraintsData<'db>) -> Self {
        let data = unsafe {
            std::mem::transmute::<ExternalConstraintsData<'db>, ExternalConstraintsData<'static>>(
                data,
            )
        };
        Self { interned: Interned::new_gc(ExternalConstraintsInterned(data)) }
    }

    #[inline]
    pub fn inner(&self) -> &ExternalConstraintsData<'db> {
        let inner = &self.interned.0;
        unsafe {
            std::mem::transmute::<&ExternalConstraintsData<'static>, &ExternalConstraintsData<'db>>(
                inner,
            )
        }
    }
}

impl<'db> std::ops::Deref for ExternalConstraints<'db> {
    type Target = ExternalConstraintsData<'db>;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

impl std::fmt::Debug for ExternalConstraints<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner().fmt(f)
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
