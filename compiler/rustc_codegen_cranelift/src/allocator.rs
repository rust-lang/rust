//! Allocator shim
// Adapted from rustc

use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorKind, AllocatorTy, NO_ALLOC_SHIM_IS_UNSTABLE,
    alloc_error_handler_name, default_fn_name, global_fn_name,
};
use rustc_codegen_ssa::base::allocator_kind_for_codegen;
use rustc_session::config::OomStrategy;
use rustc_symbol_mangling::mangle_internal_symbol;

use crate::prelude::*;

/// Returns whether an allocator shim was created
pub(crate) fn codegen(tcx: TyCtxt<'_>, module: &mut dyn Module) -> bool {
    let Some(kind) = allocator_kind_for_codegen(tcx) else { return false };
    codegen_inner(
        tcx,
        module,
        kind,
        tcx.alloc_error_handler_kind(()).unwrap(),
        tcx.sess.opts.unstable_opts.oom,
    );
    true
}

fn codegen_inner(
    tcx: TyCtxt<'_>,
    module: &mut dyn Module,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
    oom_strategy: OomStrategy,
) {
    let usize_ty = module.target_config().pointer_type();

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut arg_tys = Vec::with_capacity(method.inputs.len());
            for input in method.inputs.iter() {
                match input.ty {
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
                sig,
                &mangle_internal_symbol(tcx, &global_fn_name(method.name)),
                &mangle_internal_symbol(tcx, &default_fn_name(method.name)),
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

    {
        let sig = Signature {
            call_conv: module.target_config().default_call_conv,
            params: vec![],
            returns: vec![],
        };
        let func_id = module
            .declare_function(
                &mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE),
                Linkage::Export,
                &sig,
            )
            .unwrap();

        let mut ctx = Context::new();
        ctx.func.signature = sig;
        let mut func_ctx = FunctionBuilderContext::new();
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

        let block = bcx.create_block();
        bcx.switch_to_block(block);
        bcx.ins().return_(&[]);
        bcx.seal_all_blocks();
        bcx.finalize();

        module.define_function(func_id, &mut ctx).unwrap();
    }
}
