//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use std::fmt::Debug;
use std::string::ToString;

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

pub fn adt_def(did: DefId) -> stable_mir::ty::AdtDef {
    with_tables(|t| t.adt_def(did))
}

impl<'tcx> Tables<'tcx> {
    pub fn item_def_id(&self, item: &stable_mir::CrateItem) -> DefId {
        self.def_ids[item.0]
    }

    pub fn crate_item(&mut self, did: DefId) -> stable_mir::CrateItem {
        stable_mir::CrateItem(self.create_def_id(did))
    }

    pub fn adt_def(&mut self, did: DefId) -> stable_mir::ty::AdtDef {
        stable_mir::ty::AdtDef(self.create_def_id(did))
    }

    fn create_def_id(&mut self, did: DefId) -> stable_mir::DefId {
        // FIXME: this becomes inefficient when we have too many ids
        for (i, &d) in self.def_ids.iter().enumerate() {
            if d == did {
                return i;
            }
        }
        let id = self.def_ids.len();
        self.def_ids.push(did);
        id
    }
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}

pub fn run(tcx: TyCtxt<'_>, f: impl FnOnce()) {
    crate::stable_mir::run(Tables { tcx, def_ids: vec![], types: vec![] }, f);
}

/// A type that provides internal information but that can still be used for debug purpose.
pub type Opaque = impl Debug + ToString + Clone;

pub(crate) fn opaque<T: Debug>(value: &T) -> Opaque {
    format!("{value:?}")
}
