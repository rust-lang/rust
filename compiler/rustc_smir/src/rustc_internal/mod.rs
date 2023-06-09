//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use crate::{
    rustc_smir::Tables,
    stable_mir::{self, with},
};
use rustc_middle::ty::TyCtxt;
pub use rustc_span::def_id::{CrateNum, DefId};

fn with_tables<R>(mut f: impl FnMut(&mut Tables<'_>) -> R) -> R {
    let mut ret = None;
    with(|tables| tables.rustc_tables(&mut |t| ret = Some(f(t))));
    ret.unwrap()
}

pub fn item_def_id(item: &stable_mir::CrateItem) -> DefId {
    with_tables(|t| t.item_def_id(item))
}

pub fn crate_item(did: DefId) -> stable_mir::CrateItem {
    with_tables(|t| t.crate_item(did))
}

impl<'tcx> Tables<'tcx> {
    pub fn item_def_id(&self, item: &stable_mir::CrateItem) -> DefId {
        self.def_ids[item.0]
    }

    pub fn crate_item(&mut self, did: DefId) -> stable_mir::CrateItem {
        // FIXME: this becomes inefficient when we have too many ids
        for (i, &d) in self.def_ids.iter().enumerate() {
            if d == did {
                return stable_mir::CrateItem(i);
            }
        }
        let id = self.def_ids.len();
        self.def_ids.push(did);
        stable_mir::CrateItem(id)
    }
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}

pub fn run(tcx: TyCtxt<'_>, f: impl FnOnce()) {
    crate::stable_mir::run(Tables { tcx, def_ids: vec![], types: vec![] }, f);
}
