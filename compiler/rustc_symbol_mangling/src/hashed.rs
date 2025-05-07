use std::fmt::Write;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hashes::Hash64;
use rustc_hir::def_id::CrateNum;
use rustc_middle::ty::{Instance, TyCtxt};

use crate::v0;

pub(super) fn mangle<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    instantiating_crate: Option<CrateNum>,
    full_mangling_name: impl FnOnce() -> String,
) -> String {
    // The symbol of a generic function may be scattered in multiple downstream dylibs.
    // If the symbol of a generic function still contains `crate name`, hash conflicts between the
    // generic function and other symbols of the same `crate` cannot be detected in time during
    // construction. This symbol conflict is left over until it occurs during run time.
    // In this case, `instantiating-crate name` is used to replace `crate name` can completely
    // eliminate the risk of the preceding potential hash conflict.
    let crate_num =
        if let Some(krate) = instantiating_crate { krate } else { instance.def_id().krate };

    let mut symbol = "_RNxC".to_string();
    v0::push_ident(tcx.crate_name(crate_num).as_str(), &mut symbol);

    let hash = tcx.with_stable_hashing_context(|mut hcx| {
        let mut hasher = StableHasher::new();
        full_mangling_name().hash_stable(&mut hcx, &mut hasher);
        hasher.finish::<Hash64>().as_u64()
    });

    push_hash64(hash, &mut symbol);

    symbol
}

// The hash is encoded based on `base-62` and the final terminator `_` is removed because it does
// not help prevent hash collisions
fn push_hash64(hash: u64, output: &mut String) {
    let hash = v0::encode_integer_62(hash);
    let hash_len = hash.len();
    let _ = write!(output, "{hash_len}H{}", &hash[..hash_len - 1]);
}
