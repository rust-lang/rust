//! Things related to the infer context of the next-trait-solver.

pub(crate) mod table;

pub(crate) use table::{OpaqueTypeStorage, OpaqueTypeTable};

use macros::{TypeFoldable, TypeVisitable};

use crate::next_solver::{OpaqueTypeKey, Ty, infer::InferCtxt};

#[derive(Copy, Clone, Debug, TypeVisitable, TypeFoldable)]
pub struct OpaqueHiddenType<'db> {
    pub ty: Ty<'db>,
}

impl<'db> InferCtxt<'db> {
    /// Insert a hidden type into the opaque type storage, making sure
    /// it hasn't previously been defined. This does not emit any
    /// constraints and it's the responsibility of the caller to make
    /// sure that the item bounds of the opaque are checked.
    pub fn register_hidden_type_in_storage(
        &self,
        opaque_type_key: OpaqueTypeKey<'db>,
        hidden_ty: OpaqueHiddenType<'db>,
    ) -> Option<Ty<'db>> {
        self.inner.borrow_mut().opaque_types().register(opaque_type_key, hidden_ty)
    }
}
