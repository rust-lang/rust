//! Strips all private imports (`use`, `extern crate`) from a crate.

use crate::clean;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::ImportStripper;

pub(crate) fn strip_priv_imports(krate: clean::Crate, cx: &mut DocContext<'_>) -> clean::Crate {
    if !cx.document_private() {
        // We don't need to do anything since it'll be handled by the `strip_private` pass.
        return krate;
    }

    let is_json_output = cx.is_json_output();
    ImportStripper { tcx: cx.tcx, is_json_output, document_hidden: cx.document_hidden() }
        .fold_crate(krate)
}
