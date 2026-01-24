use rustc_abi::FieldIdx;

use crate::interpret::{InterpCx, InterpResult, Machine, Projectable};

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    fn va_list_key_index(&self) -> FieldIdx {
        // FIXME: this is target-dependent.
        FieldIdx::from_usize(2)
    }

    /// Get the MPlace of the key from the place storing the VaList.
    pub(super) fn va_list_key_mplace<P: Projectable<'tcx, M::Provenance>>(
        &self,
        va_list: &P,
    ) -> InterpResult<'tcx, P> {
        let va_list_inner = self.project_field(va_list, FieldIdx::ZERO)?;
        let field_idx = self.va_list_key_index();
        self.project_field(&va_list_inner, field_idx)
    }
}
