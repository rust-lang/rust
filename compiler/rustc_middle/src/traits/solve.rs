use std::ops::ControlFlow;

use rustc_data_structures::intern::Interned;

use crate::ty::{FallibleTypeFolder, Ty, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor};

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

impl<'tcx> TypeFoldable<'tcx> for ExternalConstraints<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(FallibleTypeFolder::tcx(folder).intern_external_constraints(ExternalConstraintsData {
            regions: (),
            opaque_types: self
                .opaque_types
                .iter()
                .map(|opaque| opaque.try_fold_with(folder))
                .collect::<Result<_, F::Error>>()?,
        }))
    }

    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        TypeFolder::tcx(folder).intern_external_constraints(ExternalConstraintsData {
            regions: (),
            opaque_types: self.opaque_types.iter().map(|opaque| opaque.fold_with(folder)).collect(),
        })
    }
}

impl<'tcx> TypeVisitable<'tcx> for ExternalConstraints<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(
        &self,
        visitor: &mut V,
    ) -> std::ops::ControlFlow<V::BreakTy> {
        self.regions.visit_with(visitor)?;
        self.opaque_types.visit_with(visitor)?;
        ControlFlow::Continue(())
    }
}
