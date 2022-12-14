use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;

use crate::{
    clean::{self, Import, ImportSource, Item},
    fold::DocFolder,
};

/// Get the id's of all items that are `pub use`d in the crate.
///
/// We need this to know if a stripped module is `pub use mod::*`, to decide
/// if it needs to be kept in the index, despite being stripped.
///
/// See [#100973](https://github.com/rust-lang/rust/issues/100973) and
/// [#101103](https://github.com/rust-lang/rust/issues/101103) for times when
/// this information is needed.
pub(crate) fn get_imports(krate: clean::Crate) -> (clean::Crate, FxHashSet<DefId>) {
    let mut finder = ImportFinder { imported: FxHashSet::default() };
    let krate = finder.fold_crate(krate);
    (krate, finder.imported)
}

struct ImportFinder {
    imported: FxHashSet<DefId>,
}

impl DocFolder for ImportFinder {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match *i.kind {
            clean::ImportItem(Import { source: ImportSource { did: Some(did), .. }, .. }) => {
                self.imported.insert(did);
                Some(i)
            }

            _ => Some(self.fold_item_recur(i)),
        }
    }
}
