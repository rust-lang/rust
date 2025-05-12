//! Strip all private items from the output. Additionally implies strip_priv_imports.
//! Basically, the goal is to remove items that are not relevant for public documentation.

use crate::clean::{self, ItemIdSet};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::{ImplStripper, ImportStripper, Pass, Stripper};

pub(crate) const STRIP_PRIVATE: Pass = Pass {
    name: "strip-private",
    run: Some(strip_private),
    description: "strips all private items from a crate which cannot be seen externally, \
                  implies strip-priv-imports",
};

/// Strip private items from the point of view of a crate or externally from a
/// crate, specified by the `xcrate` flag.
pub(crate) fn strip_private(mut krate: clean::Crate, cx: &mut DocContext<'_>) -> clean::Crate {
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
        krate = ImportStripper {
            tcx: cx.tcx,
            is_json_output,
            document_hidden: cx.render_options.document_hidden,
        }
        .fold_crate(stripper.fold_crate(krate));
    }

    // strip all impls referencing private items
    let mut stripper = ImplStripper {
        tcx: cx.tcx,
        retained: &retained,
        cache: &cx.cache,
        is_json_output,
        document_private: cx.render_options.document_private,
        document_hidden: cx.render_options.document_hidden,
    };
    stripper.fold_crate(krate)
}
