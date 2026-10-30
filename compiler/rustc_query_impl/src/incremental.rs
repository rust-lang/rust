use rustc_middle::query::on_disk_cache::CacheEncoder;
use rustc_middle::query::q_impl::incremental::{
    encode_query_values_inner, verify_query_key_hashes_inner,
};
use rustc_middle::ty::TyCtxt;

use crate::query_vtables::for_each_query_vtable;

pub(crate) fn encode_query_values<'tcx>(tcx: TyCtxt<'tcx>, encoder: &mut CacheEncoder<'tcx>) {
    for_each_query_vtable!(CACHE_ON_DISK, tcx, |query| {
        encode_query_values_inner(tcx, query, encoder)
    });
}

pub(crate) fn verify_query_key_hashes<'tcx>(tcx: TyCtxt<'tcx>) {
    if tcx.sess.opts.unstable_opts.incremental_verify_ich || cfg!(debug_assertions) {
        tcx.sess.time("verify_query_key_hashes", || {
            for_each_query_vtable!(ALL, tcx, |query| {
                verify_query_key_hashes_inner(query, tcx);
            });
        });
    }
}
