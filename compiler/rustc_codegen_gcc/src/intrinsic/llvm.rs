use gccjit::Function;

use crate::context::CodegenCx;

pub fn intrinsic<'gcc, 'tcx>(name: &str, cx: &CodegenCx<'gcc, 'tcx>) -> Function<'gcc> {
    let _gcc_name =
        match name {
            "llvm.x86.xgetbv" => {
                let gcc_name = "__builtin_trap";
                let func = cx.context.get_builtin_function(gcc_name);
                cx.functions.borrow_mut().insert(gcc_name.to_string(), func);
                return func;
            },
            // NOTE: this doc specifies the equivalent GCC builtins: http://huonw.github.io/llvmint/llvmint/x86/index.html
            "llvm.x86.sse2.cmp.pd" => "__builtin_ia32_cmppd",
            "llvm.x86.sse2.movmsk.pd" => "__builtin_ia32_movmskpd",
            "llvm.x86.sse2.pmovmskb.128" => "__builtin_ia32_pmovmskb128",
            _ => unimplemented!("unsupported LLVM intrinsic {}", name)
        };

    unimplemented!();
}
