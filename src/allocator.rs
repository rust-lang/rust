// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::prelude::*;

use syntax::ext::allocator::{AllocatorKind, AllocatorTy, ALLOCATOR_METHODS};

/// Returns whether an allocator shim was created
pub fn codegen(sess: &Session, module: &mut Module<impl Backend + 'static>) -> bool {
    let any_dynamic_crate = sess.dependency_formats.borrow().iter().any(|(_, list)| {
        use rustc::middle::dependency_format::Linkage;
        list.iter().any(|&linkage| linkage == Linkage::Dynamic)
    });
    if any_dynamic_crate {
        false
    } else if let Some(kind) = *sess.allocator_kind.get() {
        codegen_inner(module, kind);
        true
    } else {
        false
    }
}

pub fn codegen_inner(module: &mut Module<impl Backend + 'static>, kind: AllocatorKind) {
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
            call_conv: CallConv::SystemV,
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

            let ebb = bcx.create_ebb();
            bcx.switch_to_block(ebb);
            let args = arg_tys
                .into_iter()
                .map(|ty| bcx.append_ebb_param(ebb, ty))
                .collect::<Vec<Value>>();

            let callee_func_ref = module.declare_func_in_func(callee_func_id, &mut bcx.func);

            let call_inst = bcx.ins().call(callee_func_ref, &args);

            let results = bcx.inst_results(call_inst).to_vec(); // Clone to prevent borrow error
            bcx.ins().return_(&results);
            bcx.seal_all_blocks();
            bcx.finalize();
        }
        module.define_function(func_id, &mut ctx).unwrap();
    }
}
