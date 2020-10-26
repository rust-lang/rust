//! Allocator shim
// Adapted from rustc

use crate::prelude::*;

use rustc_ast::expand::allocator::{AllocatorKind, AllocatorTy, ALLOCATOR_METHODS};
use rustc_span::symbol::sym;

/// Returns whether an allocator shim was created
pub(crate) fn codegen(
    tcx: TyCtxt<'_>,
    module: &mut impl Module,
    unwind_context: &mut UnwindContext<'_>,
) -> bool {
    let any_dynamic_crate = tcx.dependency_formats(LOCAL_CRATE).iter().any(|(_, list)| {
        use rustc_middle::middle::dependency_format::Linkage;
        list.iter().any(|&linkage| linkage == Linkage::Dynamic)
    });
    if any_dynamic_crate {
        false
    } else if let Some(kind) = tcx.allocator_kind() {
        codegen_inner(module, unwind_context, kind);
        true
    } else {
        false
    }
}

fn codegen_inner(
    module: &mut impl Module,
    unwind_context: &mut UnwindContext<'_>,
    kind: AllocatorKind,
) {
    let usize_ty = module.target_config().pointer_type();

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
            call_conv: CallConv::triple_default(module.isa().triple()),
            params: arg_tys.iter().cloned().map(AbiParam::new).collect(),
            returns: output.into_iter().map(AbiParam::new).collect(),
        };

        let caller_name = format!("__rust_{}", method.name);
        let callee_name = kind.fn_name(method.name);
        //eprintln!("Codegen allocator shim {} -> {} ({:?} -> {:?})", caller_name, callee_name, sig.params, sig.returns);

        let func_id = module
            .declare_function(&caller_name, Linkage::Export, &sig)
            .unwrap();

        let callee_func_id = module
            .declare_function(&callee_name, Linkage::Import, &sig)
            .unwrap();

        let mut ctx = Context::new();
        ctx.func = Function::with_name_signature(ExternalName::user(0, 0), sig.clone());
        {
            let mut func_ctx = FunctionBuilderContext::new();
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

            let block = bcx.create_block();
            bcx.switch_to_block(block);
            let args = arg_tys
                .into_iter()
                .map(|ty| bcx.append_block_param(block, ty))
                .collect::<Vec<Value>>();

            let callee_func_ref = module.declare_func_in_func(callee_func_id, &mut bcx.func);
            let call_inst = bcx.ins().call(callee_func_ref, &args);
            let results = bcx.inst_results(call_inst).to_vec(); // Clone to prevent borrow error

            bcx.ins().return_(&results);
            bcx.seal_all_blocks();
            bcx.finalize();
        }
        module
            .define_function(
                func_id,
                &mut ctx,
                &mut cranelift_codegen::binemit::NullTrapSink {},
            )
            .unwrap();
        unwind_context.add_function(func_id, &ctx, module.isa());
    }

    let sig = Signature {
        call_conv: CallConv::triple_default(module.isa().triple()),
        params: vec![AbiParam::new(usize_ty), AbiParam::new(usize_ty)],
        returns: vec![],
    };

    let callee_name = kind.fn_name(sym::oom);
    //eprintln!("Codegen allocator shim {} -> {} ({:?} -> {:?})", caller_name, callee_name, sig.params, sig.returns);

    let func_id = module
        .declare_function("__rust_alloc_error_handler", Linkage::Export, &sig)
        .unwrap();

    let callee_func_id = module
        .declare_function(&callee_name, Linkage::Import, &sig)
        .unwrap();

    let mut ctx = Context::new();
    ctx.func = Function::with_name_signature(ExternalName::user(0, 0), sig.clone());
    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

        let block = bcx.create_block();
        bcx.switch_to_block(block);
        let args = (&[usize_ty, usize_ty])
            .into_iter()
            .map(|&ty| bcx.append_block_param(block, ty))
            .collect::<Vec<Value>>();

        let callee_func_ref = module.declare_func_in_func(callee_func_id, &mut bcx.func);
        bcx.ins().call(callee_func_ref, &args);

        bcx.ins().trap(TrapCode::UnreachableCodeReached);
        bcx.seal_all_blocks();
        bcx.finalize();
    }
    module
        .define_function(
            func_id,
            &mut ctx,
            &mut cranelift_codegen::binemit::NullTrapSink {},
        )
        .unwrap();
    unwind_context.add_function(func_id, &ctx, module.isa());
}
