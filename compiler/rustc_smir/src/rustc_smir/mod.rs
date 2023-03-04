//! Module that implements what will become the rustc side of Stable MIR.
//!
//! This module is responsible for building Stable MIR components from internal components.
//!
//! This module is not intended to be invoked directly by users. It will eventually
//! become the public API of rustc that will be invoked by the `stable_mir` crate.
//!
//! For now, we are developing everything inside `rustc`, thus, we keep this module private.

use crate::stable_mir::{self};
use rustc_middle::ty::{tls::with, TyCtxt};
use rustc_span::def_id::{CrateNum, DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use tracing::debug;

/// Get information about the local crate.
pub fn local_crate() -> stable_mir::Crate {
    with(|tcx| smir_crate(tcx, LOCAL_CRATE))
}

/// Find a crate with the given name.
pub fn find_crate(name: &str) -> Option<stable_mir::Crate> {
    with(|tcx| {
        [LOCAL_CRATE].iter().chain(tcx.crates(()).iter()).find_map(|crate_num| {
            let crate_name = tcx.crate_name(*crate_num).to_string();
            (name == crate_name).then(|| smir_crate(tcx, *crate_num))
        })
    })
}

/// Build a stable mir crate from a given crate number.
fn smir_crate(tcx: TyCtxt<'_>, crate_num: CrateNum) -> stable_mir::Crate {
    let crate_name = tcx.crate_name(crate_num).to_string();
    let is_local = crate_num == LOCAL_CRATE;
    let mod_id = DefId { index: CRATE_DEF_INDEX, krate: crate_num };
    let items = if is_local {
        tcx.hir_module_items(mod_id.expect_local())
            .items()
            .map(|item| {
                let def_id = item.owner_id.def_id.to_def_id();
                stable_mir::CrateItem(def_id)
            })
            .collect()
    } else {
        tcx.module_children(mod_id)
            .iter()
            .filter_map(|item| item.res.opt_def_id())
            .map(stable_mir::CrateItem)
            .collect::<Vec<_>>()
    };
    debug!(?crate_name, ?crate_num, "smir_crate");
    stable_mir::Crate { id: crate_num.into(), name: crate_name, is_local, root_items: items }
}
