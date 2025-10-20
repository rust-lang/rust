//! Things useful for mapping to/from Chalk and next-trait-solver types.

use crate::next_solver::interner::DbInterner;

pub(crate) trait ChalkToNextSolver<'db, Out> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Out;
}

impl<'db> ChalkToNextSolver<'db, crate::lower::ImplTraitIdx<'db>> for crate::ImplTraitIdx {
    fn to_nextsolver(&self, _interner: DbInterner<'db>) -> crate::lower::ImplTraitIdx<'db> {
        crate::lower::ImplTraitIdx::from_raw(self.into_raw())
    }
}
