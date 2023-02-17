use std::ops::ControlFlow;

use rustc_data_structures::intern::Interned;

use crate::ty::{
    FallibleTypeFolder, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor,
};

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub struct ExternalConstraints<'tcx>(pub(crate) Interned<'tcx, ExternalConstraintsData<'tcx>>);

impl<'tcx> std::ops::Deref for ExternalConstraints<'tcx> {
    type Target = ExternalConstraintsData<'tcx>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Additional constraints returned on success.
#[derive(Debug, PartialEq, Eq, Clone, Hash, Default)]
pub struct ExternalConstraintsData<'tcx> {
    // FIXME: implement this.
    pub regions: (),
    pub opaque_types: Vec<(Ty<'tcx>, Ty<'tcx>)>,
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(FallibleTypeFolder::interner(folder).mk_external_constraints(ExternalConstraintsData {
            regions: (),
            opaque_types: self
                .opaque_types
                .iter()
                .map(|opaque| opaque.try_fold_with(folder))
                .collect::<Result<_, F::Error>>()?,
        }))
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        TypeFolder::interner(folder).mk_external_constraints(ExternalConstraintsData {
            regions: (),
            opaque_types: self.opaque_types.iter().map(|opaque| opaque.fold_with(folder)).collect(),
        })
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        visitor: &mut V,
    ) -> std::ops::ControlFlow<V::BreakTy> {
        self.regions.visit_with(visitor)?;
        self.opaque_types.visit_with(visitor)?;
        ControlFlow::Continue(())
    }
}
