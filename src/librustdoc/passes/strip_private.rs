//! Strip all private items from the output.
//!
//! Implies `strip_priv_imports`.

use crate::clean::{self, ItemIdSet};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::{ImplStripper, ImportStripper, Stripper};

pub(crate) fn strip_private(mut krate: clean::Crate, cx: &mut DocContext<'_>) -> clean::Crate {
    if cx.document_private() {
        return krate;
    }

    // This stripper collects all *retained* nodes.
    let mut retained = ItemIdSet::default();
    let is_json_output = cx.is_json_output();

    // strip all private items
    {
        let mut stripper = Stripper {
            retained: &mut retained,
            effective_visibilities: &cx.cache.effective_visibilities,
            update_retained: true,
            is_json_output,
            tcx: cx.tcx,
        };
        krate =
            ImportStripper { tcx: cx.tcx, is_json_output, document_hidden: cx.document_hidden() }
                .fold_crate(stripper.fold_crate(krate));
    }

    // strip all impls referencing private items
    let mut stripper = ImplStripper {
        tcx: cx.tcx,
        retained: &retained,
        cache: &cx.cache,
        is_json_output,
        document_private: cx.document_private(),
        document_hidden: cx.document_hidden(),
    };
    stripper.fold_crate(krate)
}
