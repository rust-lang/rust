//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use crate::stable_mir;
pub use rustc_span::def_id::{CrateNum, DefId};

pub fn item_def_id(item: &stable_mir::CrateItem) -> DefId {
    item.0
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}
