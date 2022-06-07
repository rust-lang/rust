use std::borrow::Cow;

use gccjit::{Function, FunctionPtrType, RValue, ToRValue};

use crate::{context::CodegenCx, builder::Builder};

pub fn adjust_intrinsic_arguments<'a, 'b, 'gcc, 'tcx>(builder: &Builder<'a, 'gcc, 'tcx>, gcc_func: FunctionPtrType<'gcc>, mut args: Cow<'b, [RValue<'gcc>]>, func_name: &str) -> Cow<'b, [RValue<'gcc>]> {
    // Some LLVM intrinsics do not map 1-to-1 to GCC intrinsics, so we add the missing
    // arguments here.
    if gcc_func.get_param_count() != args.len() {
        match &*func_name {
            "__builtin_ia32_pmuldq512_mask" | "__builtin_ia32_pmuludq512_mask"
                // FIXME(antoyo): the following intrinsics has 4 (or 5) arguments according to the doc, but is defined with 2 (or 3) arguments in library/stdarch/crates/core_arch/src/x86/avx512f.rs.
                | "__builtin_ia32_pmaxsd512_mask" | "__builtin_ia32_pmaxsq512_mask" | "__builtin_ia32_pmaxsq256_mask"
                | "__builtin_ia32_pmaxsq128_mask" | "__builtin_ia32_maxps512_mask" | "__builtin_ia32_maxpd512_mask"
                | "__builtin_ia32_pmaxud512_mask" | "__builtin_ia32_pmaxuq512_mask" | "__builtin_ia32_pmaxuq256_mask"
                | "__builtin_ia32_pmaxuq128_mask"
                | "__builtin_ia32_pminsd512_mask" | "__builtin_ia32_pminsq512_mask" | "__builtin_ia32_pminsq256_mask"
                | "__builtin_ia32_pminsq128_mask" | "__builtin_ia32_minps512_mask" | "__builtin_ia32_minpd512_mask"
                | "__builtin_ia32_pminud512_mask" | "__builtin_ia32_pminuq512_mask" | "__builtin_ia32_pminuq256_mask"
                | "__builtin_ia32_pminuq128_mask" | "__builtin_ia32_sqrtps512_mask" | "__builtin_ia32_sqrtpd512_mask"
                => {
                    // TODO: refactor by separating those intrinsics outside of this branch.
                    let add_before_last_arg =
                        match &*func_name {
                            "__builtin_ia32_maxps512_mask" | "__builtin_ia32_maxpd512_mask"
                                | "__builtin_ia32_minps512_mask" | "__builtin_ia32_minpd512_mask"
                                | "__builtin_ia32_sqrtps512_mask" | "__builtin_ia32_sqrtpd512_mask" => true,
                            _ => false,
                        };
                    let new_first_arg_is_zero =
                        match &*func_name {
                            "__builtin_ia32_pmaxuq256_mask" | "__builtin_ia32_pmaxuq128_mask"
                                | "__builtin_ia32_pminuq256_mask" | "__builtin_ia32_pminuq128_mask" => true,
                            _ => false
                        };
                    let arg3_index =
                        match &*func_name {
                            "__builtin_ia32_sqrtps512_mask" | "__builtin_ia32_sqrtpd512_mask" => 1,
                            _ => 2,
                        };
                    let mut new_args = args.to_vec();
                    let arg3_type = gcc_func.get_param_type(arg3_index);
                    let first_arg =
                        if new_first_arg_is_zero {
                            let vector_type = arg3_type.dyncast_vector().expect("vector type");
                            let zero = builder.context.new_rvalue_zero(vector_type.get_element_type());
                            let num_units = vector_type.get_num_units();
                            builder.context.new_rvalue_from_vector(None, arg3_type, &vec![zero; num_units])
                        }
                        else {
                            builder.current_func().new_local(None, arg3_type, "undefined_for_intrinsic").to_rvalue()
                        };
                    if add_before_last_arg {
                        new_args.insert(new_args.len() - 1, first_arg);
                    }
                    else {
                        new_args.push(first_arg);
                    }
                    let arg4_index =
                        match &*func_name {
                            "__builtin_ia32_sqrtps512_mask" | "__builtin_ia32_sqrtpd512_mask" => 2,
                            _ => 3,
                        };
                    let arg4_type = gcc_func.get_param_type(arg4_index);
                    let minus_one = builder.context.new_rvalue_from_int(arg4_type, -1);
                    if add_before_last_arg {
                        new_args.insert(new_args.len() - 1, minus_one);
                    }
                    else {
                        new_args.push(minus_one);
                    }
                    args = new_args.into();
                },
                "__builtin_ia32_pternlogd512_mask" | "__builtin_ia32_pternlogd256_mask"
                    | "__builtin_ia32_pternlogd128_mask" | "__builtin_ia32_pternlogq512_mask"
                    | "__builtin_ia32_pternlogq256_mask" | "__builtin_ia32_pternlogq128_mask" => {
                        let mut new_args = args.to_vec();
                        let arg5_type = gcc_func.get_param_type(4);
                        let minus_one = builder.context.new_rvalue_from_int(arg5_type, -1);
                        new_args.push(minus_one);
                        args = new_args.into();
                    },
                    "__builtin_ia32_vfmaddps512_mask" | "__builtin_ia32_vfmaddpd512_mask" => {
                        let mut new_args = args.to_vec();

                        let mut last_arg = None;
                        if args.len() == 4 {
                            last_arg = new_args.pop();
                        }

                        let arg4_type = gcc_func.get_param_type(3);
                        let minus_one = builder.context.new_rvalue_from_int(arg4_type, -1);
                        new_args.push(minus_one);

                        if args.len() == 3 {
                            // Both llvm.fma.v16f32 and llvm.x86.avx512.vfmadd.ps.512 maps to
                            // the same GCC intrinsic, but the former has 3 parameters and the
                            // latter has 4 so it doesn't require this additional argument.
                            let arg5_type = gcc_func.get_param_type(4);
                            new_args.push(builder.context.new_rvalue_from_int(arg5_type, 4));
                        }

                        if let Some(last_arg) = last_arg {
                            new_args.push(last_arg);
                        }

                        args = new_args.into();
                    },
                    "__builtin_ia32_addps512_mask" | "__builtin_ia32_addpd512_mask"
                        | "__builtin_ia32_subps512_mask" | "__builtin_ia32_subpd512_mask"
                        | "__builtin_ia32_mulps512_mask" | "__builtin_ia32_mulpd512_mask"
                        | "__builtin_ia32_divps512_mask" | "__builtin_ia32_divpd512_mask" => {
                        let mut new_args = args.to_vec();
                        let last_arg = new_args.pop().expect("last arg");
                        let arg3_type = gcc_func.get_param_type(2);
                        let undefined = builder.current_func().new_local(None, arg3_type, "undefined_for_intrinsic").to_rvalue();
                        new_args.push(undefined);
                        let arg4_type = gcc_func.get_param_type(3);
                        let minus_one = builder.context.new_rvalue_from_int(arg4_type, -1);
                        new_args.push(minus_one);
                        new_args.push(last_arg);
                        args = new_args.into();
                    },
                    "__builtin_ia32_vfmaddsubps512_mask" | "__builtin_ia32_vfmaddsubpd512_mask" => {
                        let mut new_args = args.to_vec();
                        let last_arg = new_args.pop().expect("last arg");
                        let arg4_type = gcc_func.get_param_type(3);
                        let minus_one = builder.context.new_rvalue_from_int(arg4_type, -1);
                        new_args.push(minus_one);
                        new_args.push(last_arg);
                        args = new_args.into();
                    },
                    _ => (),
        }
    }

    args
}

pub fn ignore_arg_cast(func_name: &str, index: usize, args_len: usize) -> bool {
    // NOTE: these intrinsics have missing parameters before the last one, so ignore the
    // last argument type check.
    // FIXME(antoyo): find a way to refactor in order to avoid this hack.
    match func_name {
        "__builtin_ia32_maxps512_mask" | "__builtin_ia32_maxpd512_mask"
            | "__builtin_ia32_minps512_mask" | "__builtin_ia32_minpd512_mask" | "__builtin_ia32_sqrtps512_mask"
            | "__builtin_ia32_sqrtpd512_mask" | "__builtin_ia32_addps512_mask" | "__builtin_ia32_addpd512_mask"
            | "__builtin_ia32_subps512_mask" | "__builtin_ia32_subpd512_mask"
            | "__builtin_ia32_mulps512_mask" | "__builtin_ia32_mulpd512_mask"
            | "__builtin_ia32_divps512_mask" | "__builtin_ia32_divpd512_mask"
            | "__builtin_ia32_vfmaddsubps512_mask" | "__builtin_ia32_vfmaddsubpd512_mask" => {
                if index == args_len - 1 {
                    return true;
                }
            },
        "__builtin_ia32_vfmaddps512_mask" | "__builtin_ia32_vfmaddpd512_mask" => {
            // Since there are two LLVM intrinsics that map to each of these GCC builtins and only
            // one of them has a missing parameter before the last one, we check the number of
            // arguments to distinguish those cases.
            if args_len == 4 && index == args_len - 1 {
                return true;
            }
        },
        _ => (),
    }

    false
}

#[cfg(not(feature="master"))]
pub fn intrinsic<'gcc, 'tcx>(name: &str, cx: &CodegenCx<'gcc, 'tcx>) -> Function<'gcc> {
    match name {
        "llvm.x86.xgetbv" => {
            let gcc_name = "__builtin_trap";
            let func = cx.context.get_builtin_function(gcc_name);
            cx.functions.borrow_mut().insert(gcc_name.to_string(), func);
            return func;
        },
        _ => unimplemented!("unsupported LLVM intrinsic {}", name),
    }
}

#[cfg(feature="master")]
pub fn intrinsic<'gcc, 'tcx>(name: &str, cx: &CodegenCx<'gcc, 'tcx>) -> Function<'gcc> {
    let gcc_name = match name {
        "llvm.x86.xgetbv" => "__builtin_ia32_xgetbv",
        // NOTE: this doc specifies the equivalent GCC builtins: http://huonw.github.io/llvmint/llvmint/x86/index.html
        "llvm.sqrt.v2f64" => "__builtin_ia32_sqrtpd",
        "llvm.x86.avx512.pmul.dq.512" => "__builtin_ia32_pmuldq512_mask",
        "llvm.x86.avx512.pmulu.dq.512" => "__builtin_ia32_pmuludq512_mask",
        "llvm.x86.avx512.mask.pmaxs.q.256" => "__builtin_ia32_pmaxsq256_mask",
        "llvm.x86.avx512.mask.pmaxs.q.128" => "__builtin_ia32_pmaxsq128_mask",
        "llvm.x86.avx512.max.ps.512" => "__builtin_ia32_maxps512_mask",
        "llvm.x86.avx512.max.pd.512" => "__builtin_ia32_maxpd512_mask",
        "llvm.x86.avx512.mask.pmaxu.q.256" => "__builtin_ia32_pmaxuq256_mask",
        "llvm.x86.avx512.mask.pmaxu.q.128" => "__builtin_ia32_pmaxuq128_mask",
        "llvm.x86.avx512.mask.pmins.q.256" => "__builtin_ia32_pminsq256_mask",
        "llvm.x86.avx512.mask.pmins.q.128" => "__builtin_ia32_pminsq128_mask",
        "llvm.x86.avx512.min.ps.512" => "__builtin_ia32_minps512_mask",
        "llvm.x86.avx512.min.pd.512" => "__builtin_ia32_minpd512_mask",
        "llvm.x86.avx512.mask.pminu.q.256" => "__builtin_ia32_pminuq256_mask",
        "llvm.x86.avx512.mask.pminu.q.128" => "__builtin_ia32_pminuq128_mask",
        "llvm.fma.v16f32" => "__builtin_ia32_vfmaddps512_mask",
        "llvm.fma.v8f64" => "__builtin_ia32_vfmaddpd512_mask",
        "llvm.x86.avx512.vfmaddsub.ps.512" => "__builtin_ia32_vfmaddsubps512_mask",
        "llvm.x86.avx512.vfmaddsub.pd.512" => "__builtin_ia32_vfmaddsubpd512_mask",
        "llvm.x86.avx512.pternlog.d.512" => "__builtin_ia32_pternlogd512_mask",
        "llvm.x86.avx512.pternlog.d.256" => "__builtin_ia32_pternlogd256_mask",
        "llvm.x86.avx512.pternlog.d.128" => "__builtin_ia32_pternlogd128_mask",
        "llvm.x86.avx512.pternlog.q.512" => "__builtin_ia32_pternlogq512_mask",
        "llvm.x86.avx512.pternlog.q.256" => "__builtin_ia32_pternlogq256_mask",
        "llvm.x86.avx512.pternlog.q.128" => "__builtin_ia32_pternlogq128_mask",
        "llvm.x86.avx512.add.ps.512" => "__builtin_ia32_addps512_mask",
        "llvm.x86.avx512.add.pd.512" => "__builtin_ia32_addpd512_mask",
        "llvm.x86.avx512.sub.ps.512" => "__builtin_ia32_subps512_mask",
        "llvm.x86.avx512.sub.pd.512" => "__builtin_ia32_subpd512_mask",
        "llvm.x86.avx512.mul.ps.512" => "__builtin_ia32_mulps512_mask",
        "llvm.x86.avx512.mul.pd.512" => "__builtin_ia32_mulpd512_mask",
        "llvm.x86.avx512.div.ps.512" => "__builtin_ia32_divps512_mask",
        "llvm.x86.avx512.div.pd.512" => "__builtin_ia32_divpd512_mask",
        "llvm.x86.avx512.vfmadd.ps.512" => "__builtin_ia32_vfmaddps512_mask",
        "llvm.x86.avx512.vfmadd.pd.512" => "__builtin_ia32_vfmaddpd512_mask",

        // The above doc points to unknown builtins for the following, so override them:
        "llvm.x86.avx2.gather.d.d" => "__builtin_ia32_gathersiv4si",
        "llvm.x86.avx2.gather.d.d.256" => "__builtin_ia32_gathersiv8si",
        "llvm.x86.avx2.gather.d.ps" => "__builtin_ia32_gathersiv4sf",
        "llvm.x86.avx2.gather.d.ps.256" => "__builtin_ia32_gathersiv8sf",
        "llvm.x86.avx2.gather.d.q" => "__builtin_ia32_gathersiv2di",
        "llvm.x86.avx2.gather.d.q.256" => "__builtin_ia32_gathersiv4di",
        "llvm.x86.avx2.gather.d.pd" => "__builtin_ia32_gathersiv2df",
        "llvm.x86.avx2.gather.d.pd.256" => "__builtin_ia32_gathersiv4df",
        "llvm.x86.avx2.gather.q.d" => "__builtin_ia32_gatherdiv4si",
        "llvm.x86.avx2.gather.q.d.256" => "__builtin_ia32_gatherdiv4si256",
        "llvm.x86.avx2.gather.q.ps" => "__builtin_ia32_gatherdiv4sf",
        "llvm.x86.avx2.gather.q.ps.256" => "__builtin_ia32_gatherdiv4sf256",
        "llvm.x86.avx2.gather.q.q" => "__builtin_ia32_gatherdiv2di",
        "llvm.x86.avx2.gather.q.q.256" => "__builtin_ia32_gatherdiv4di",
        "llvm.x86.avx2.gather.q.pd" => "__builtin_ia32_gatherdiv2df",
        "llvm.x86.avx2.gather.q.pd.256" => "__builtin_ia32_gatherdiv4df",
        "" => "",
        // NOTE: this file is generated by https://github.com/GuillaumeGomez/llvmint/blob/master/generate_list.py
        _ => include!("archs.rs"),
    };

    let func = cx.context.get_target_builtin_function(gcc_name);
    cx.functions.borrow_mut().insert(gcc_name.to_string(), func);
    func
}
