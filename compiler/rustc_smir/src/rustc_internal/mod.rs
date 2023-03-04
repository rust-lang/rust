//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use crate::stable_mir::CrateItem;

pub type DefId = rustc_span::def_id::DefId;

pub fn item_def_id(item: &CrateItem) -> DefId {
    item.0
}
