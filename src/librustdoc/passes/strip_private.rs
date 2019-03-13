use rustc::util::nodemap::DefIdSet;

use crate::clean;
use crate::fold::{DocFolder};
use crate::core::DocContext;
use crate::passes::{ImplStripper, ImportStripper, Stripper, Pass};

pub const STRIP_PRIVATE: Pass = Pass {
    name: "strip-private",
    pass: strip_private,
    description: "strips all private items from a crate which cannot be seen externally, \
        implies strip-priv-imports",
};

/// Strip private items from the point of view of a crate or externally from a
/// crate, specified by the `xcrate` flag.
pub fn strip_private(mut krate: clean::Crate, cx: &DocContext<'_>) -> clean::Crate {
    // This stripper collects all *retained* nodes.
    let mut retained = DefIdSet::default();
    let access_levels = cx.renderinfo.borrow().access_levels.clone();

    // strip all private items
    {
        let mut stripper = Stripper {
            retained: &mut retained,
            access_levels: &access_levels,
            update_retained: true,
        };
        krate = ImportStripper.fold_crate(stripper.fold_crate(krate));
    }

    // strip all impls referencing private items
    let mut stripper = ImplStripper { retained: &retained };
    stripper.fold_crate(krate)
}
