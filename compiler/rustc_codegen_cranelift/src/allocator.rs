//! Allocator shim
// Adapted from rustc

use rustc_ast::expand::allocator::{
    AllocatorKind, NO_ALLOC_SHIM_IS_UNSTABLE, alloc_error_handler_name,
};
use rustc_codegen_ssa::base::needs_allocator_shim;
use rustc_session::config::OomStrategy;
use rustc_symbol_mangling::mangle_internal_symbol;

use crate::prelude::*;

/// Returns whether an allocator shim was created
pub(crate) fn codegen(tcx: TyCtxt<'_>, module: &mut dyn Module) -> bool {
    if needs_allocator_shim(tcx) {
        codegen_inner(
            tcx,
            module,
            tcx.alloc_error_handler_kind(()).unwrap(),
            tcx.sess.opts.unstable_opts.oom,
        );
        true
    } else {
        false
    }
}

fn codegen_inner(
    tcx: TyCtxt<'_>,
    module: &mut dyn Module,
    alloc_error_handler_kind: AllocatorKind,
    oom_strategy: OomStrategy,
) {
    let usize_ty = module.target_config().pointer_type();

    let sig = Signature {
        call_conv: module.target_config().default_call_conv,
        params: vec![AbiParam::new(usize_ty), AbiParam::new(usize_ty)],
        returns: vec![],
    };
    crate::common::create_wrapper_function(
        module,
        sig,
        &mangle_internal_symbol(tcx, "__rust_alloc_error_handler"),
        &mangle_internal_symbol(tcx, alloc_error_handler_name(alloc_error_handler_kind)),
    );

    let data_id = module
        .declare_data(
            &mangle_internal_symbol(tcx, OomStrategy::SYMBOL),
            Linkage::Export,
            false,
            false,
        )
        .unwrap();
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
