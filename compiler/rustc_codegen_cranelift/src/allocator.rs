//! Allocator shim
// Adapted from rustc

use rustc_ast::expand::allocator::NO_ALLOC_SHIM_IS_UNSTABLE;
use rustc_codegen_ssa::base::needs_allocator_shim;
use rustc_session::config::OomStrategy;
use rustc_symbol_mangling::mangle_internal_symbol;

use crate::prelude::*;

/// Returns whether an allocator shim was created
pub(crate) fn codegen(tcx: TyCtxt<'_>, module: &mut dyn Module) -> bool {
    if needs_allocator_shim(tcx) {
        codegen_inner(tcx, module, tcx.sess.opts.unstable_opts.oom);
        true
    } else {
        false
    }
}

fn codegen_inner(tcx: TyCtxt<'_>, module: &mut dyn Module, oom_strategy: OomStrategy) {
    let data_id = module.declare_data(OomStrategy::SYMBOL, Linkage::Export, false, false).unwrap();
    let mut data = DataDescription::new();
    data.set_align(1);
    let val = oom_strategy.should_panic();
    data.define(Box::new([val]));
    module.define_data(data_id, &data).unwrap();

    let data_id = module
        .declare_data(
            &mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE),
            Linkage::Export,
            false,
            false,
        )
        .unwrap();
    let mut data = DataDescription::new();
    data.set_align(1);
    data.define(Box::new([0]));
    module.define_data(data_id, &data).unwrap();
}
