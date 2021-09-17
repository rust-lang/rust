use gccjit::{FunctionType, ToRValue};
use rustc_ast::expand::allocator::{AllocatorKind, AllocatorTy, ALLOCATOR_METHODS};
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

use crate::GccContext;

pub(crate) unsafe fn codegen(tcx: TyCtxt<'_>, mods: &mut GccContext, _module_name: &str, kind: AllocatorKind, has_alloc_error_handler: bool) {
    let context = &mods.context;
    let usize =
        match tcx.sess.target.pointer_width {
            16 => context.new_type::<u16>(),
            32 => context.new_type::<u32>(),
            64 => context.new_type::<u64>(),
            tws => bug!("Unsupported target word size for int: {}", tws),
        };
    let i8 = context.new_type::<i8>();
    let i8p = i8.make_pointer();
    let void = context.new_type::<()>();

    for method in ALLOCATOR_METHODS {
        let mut types = Vec::with_capacity(method.inputs.len());
        for ty in method.inputs.iter() {
            match *ty {
                AllocatorTy::Layout => {
                    types.push(usize);
                    types.push(usize);
                }
                AllocatorTy::Ptr => types.push(i8p),
                AllocatorTy::Usize => types.push(usize),

                AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
            }
        }
        let output = match method.output {
            AllocatorTy::ResultPtr => Some(i8p),
            AllocatorTy::Unit => None,

            AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                panic!("invalid allocator output")
            }
        };
        let name = format!("__rust_{}", method.name);

        let args: Vec<_> = types.iter().enumerate()
            .map(|(index, typ)| context.new_parameter(None, *typ, &format!("param{}", index)))
            .collect();
        let func = context.new_function(None, FunctionType::Exported, output.unwrap_or(void), &args, name, false);

        if tcx.sess.target.options.default_hidden_visibility {
            // TODO(antoyo): set visibility.
        }
        if tcx.sess.must_emit_unwind_tables() {
            // TODO(antoyo): emit unwind tables.
        }

        let callee = kind.fn_name(method.name);
        let args: Vec<_> = types.iter().enumerate()
            .map(|(index, typ)| context.new_parameter(None, *typ, &format!("param{}", index)))
            .collect();
        let callee = context.new_function(None, FunctionType::Extern, output.unwrap_or(void), &args, callee, false);
        // TODO(antoyo): set visibility.

        let block = func.new_block("entry");

        let args = args
            .iter()
            .enumerate()
            .map(|(i, _)| func.get_param(i as i32).to_rvalue())
            .collect::<Vec<_>>();
        let ret = context.new_call(None, callee, &args);
        //llvm::LLVMSetTailCall(ret, True);
        if output.is_some() {
            block.end_with_return(None, ret);
        }
        else {
            block.end_with_void_return(None);
        }

        // TODO(@Commeownist): Check if we need to emit some extra debugging info in certain circumstances
        // as described in https://github.com/rust-lang/rust/commit/77a96ed5646f7c3ee8897693decc4626fe380643
    }

    let types = [usize, usize];
    let name = "__rust_alloc_error_handler".to_string();
    let args: Vec<_> = types.iter().enumerate()
        .map(|(index, typ)| context.new_parameter(None, *typ, &format!("param{}", index)))
        .collect();
    let func = context.new_function(None, FunctionType::Exported, void, &args, name, false);

    let kind =
        if has_alloc_error_handler {
            AllocatorKind::Global
        }
        else {
            AllocatorKind::Default
        };
    let callee = kind.fn_name(sym::oom);
    let args: Vec<_> = types.iter().enumerate()
        .map(|(index, typ)| context.new_parameter(None, *typ, &format!("param{}", index)))
        .collect();
    let callee = context.new_function(None, FunctionType::Extern, void, &args, callee, false);
    //llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden);

    let block = func.new_block("entry");

    let args = args
        .iter()
        .enumerate()
        .map(|(i, _)| func.get_param(i as i32).to_rvalue())
        .collect::<Vec<_>>();
    let _ret = context.new_call(None, callee, &args);
    //llvm::LLVMSetTailCall(ret, True);
    block.end_with_void_return(None);
}
