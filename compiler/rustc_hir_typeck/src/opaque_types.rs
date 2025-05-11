use super::FnCtxt;
impl<'tcx> FnCtxt<'_, 'tcx> {
    /// We may in theory add further uses of an opaque after cloning the opaque
    /// types storage during writeback when computing the defining uses.
    ///
    /// Silently ignoring them is dangerous and could result in ICE or even in
    /// unsoundness, so we make sure we catch such cases here. There's currently
    /// no known code where this actually happens, even with the new solver which
    /// does normalize types in writeback after cloning the opaque type storage.
    ///
    /// FIXME(@lcnr): I believe this should be possible in theory and would like
    /// an actual test here. After playing around with this for an hour, I wasn't
    /// able to do anything which didn't already try to normalize the opaque before
    /// then, either allowing compilation to succeed or causing an ambiguity error.
    pub(super) fn detect_opaque_types_added_during_writeback(&self) {
        let num_entries = self.checked_opaque_types_storage_entries.take().unwrap();
        for (key, hidden_type) in
            self.inner.borrow_mut().opaque_types().opaque_types_added_since(num_entries)
        {
            let opaque_type_string = self.tcx.def_path_str(key.def_id);
            let msg = format!("unexpected cyclic definition of `{opaque_type_string}`");
            self.dcx().span_delayed_bug(hidden_type.span, msg);
        }
        let _ = self.take_opaque_types();
    }
}
