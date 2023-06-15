//! Allocator shim
// Adapted from rustc

use crate::prelude::*;

use rustc_ast::expand::allocator::{
    alloc_error_handler_name, default_fn_name, global_fn_name, AllocatorKind, AllocatorTy,
    ALLOCATOR_METHODS, NO_ALLOC_SHIM_IS_UNSTABLE,
};
use rustc_codegen_ssa::base::allocator_kind_for_codegen;
use rustc_session::config::OomStrategy;

/// Returns whether an allocator shim was created
pub(crate) fn codegen(
    tcx: TyCtxt<'_>,
    module: &mut impl Module,
    unwind_context: &mut UnwindContext,
) -> bool {
    let Some(kind) = allocator_kind_for_codegen(tcx) else { return false };
    codegen_inner(
        module,
        unwind_context,
        kind,
        tcx.alloc_error_handler_kind(()).unwrap(),
        tcx.sess.opts.unstable_opts.oom,
    );
    true
}

fn codegen_inner(
    module: &mut impl Module,
    unwind_context: &mut UnwindContext,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
    oom_strategy: OomStrategy,
) {
    let usize_ty = module.target_config().pointer_type();

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut arg_tys = Vec::with_capacity(method.inputs.len());
            for ty in method.inputs.iter() {
                match *ty {
                    AllocatorTy::Layout => {
                        arg_tys.push(usize_ty); // size
                        arg_tys.push(usize_ty); // align
                    }
                    AllocatorTy::Ptr => arg_tys.push(usize_ty),
                    AllocatorTy::Usize => arg_tys.push(usize_ty),

                    AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
                }
            }
            let output = match method.output {
                AllocatorTy::ResultPtr => Some(usize_ty),
                AllocatorTy::Unit => None,

                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                    panic!("invalid allocator output")
                }
            };

            let sig = Signature {
                call_conv: module.target_config().default_call_conv,
                params: arg_tys.iter().cloned().map(AbiParam::new).collect(),
                returns: output.into_iter().map(AbiParam::new).collect(),
            };
            crate::common::create_wrapper_function(
                module,
                unwind_context,
                sig,
                &global_fn_name(method.name),
                &default_fn_name(method.name),
            );
        }
    }

    let sig = Signature {
        call_conv: module.target_config().default_call_conv,
        params: vec![AbiParam::new(usize_ty), AbiParam::new(usize_ty)],
        returns: vec![],
    };
    crate::common::create_wrapper_function(
        module,
        unwind_context,
        sig,
        "__rust_alloc_error_handler",
        &alloc_error_handler_name(alloc_error_handler_kind),
    );

    let data_id = module.declare_data(OomStrategy::SYMBOL, Linkage::Export, false, false).unwrap();
    let mut data = DataDescription::new();
    data.set_align(1);
    let val = oom_strategy.should_panic();
    data.define(Box::new([val]));
    module.define_data(data_id, &data).unwrap();

    let data_id =
        module.declare_data(NO_ALLOC_SHIM_IS_UNSTABLE, Linkage::Export, false, false).unwrap();
    let mut data = DataDescription::new();
    data.set_align(1);
    data.define(Box::new([0]));
    module.define_data(data_id, &data).unwrap();
}
