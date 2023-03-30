//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use std::sync::RwLock;

use crate::stable_mir;
pub use rustc_span::def_id::{CrateNum, DefId};

static DEF_ID_MAP: RwLock<Vec<DefId>> = RwLock::new(Vec::new());

pub fn item_def_id(item: &stable_mir::CrateItem) -> DefId {
    DEF_ID_MAP.read().unwrap()[item.0]
}

pub fn crate_item(did: DefId) -> stable_mir::CrateItem {
    // FIXME: this becomes inefficient when we have too many ids
    let mut map = DEF_ID_MAP.write().unwrap();
    for (i, &d) in map.iter().enumerate() {
        if d == did {
            return stable_mir::CrateItem(i);
        }
    }
    let id = map.len();
    map.push(did);
    stable_mir::CrateItem(id)
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}
