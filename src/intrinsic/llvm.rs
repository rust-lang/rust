use std::borrow::Cow;

use gccjit::{Function, FunctionPtrType, RValue, ToRValue, UnaryOp};
use rustc_codegen_ssa::traits::BuilderMethods;

use crate::{context::CodegenCx, builder::Builder};

pub fn adjust_intrinsic_arguments<'a, 'b, 'gcc, 'tcx>(builder: &Builder<'a, 'gcc, 'tcx>, gcc_func: FunctionPtrType<'gcc>, mut args: Cow<'b, [RValue<'gcc>]>, func_name: &str, original_function_name: Option<&String>) -> Cow<'b, [RValue<'gcc>]> {
    // Some LLVM intrinsics do not map 1-to-1 to GCC intrinsics, so we add the missing
    // arguments here.
    if gcc_func.get_param_count() != args.len() {
        match &*func_name {
            // NOTE: the following intrinsics have a different number of parameters in LLVM and GCC.
            "__builtin_ia32_prold512_mask" | "__builtin_ia32_pmuldq512_mask" | "__builtin_ia32_pmuludq512_mask"
                | "__builtin_ia32_pmaxsd512_mask" | "__builtin_ia32_pmaxsq512_mask" | "__builtin_ia32_pmaxsq256_mask"
                | "__builtin_ia32_pmaxsq128_mask" | "__builtin_ia32_pmaxud512_mask" | "__builtin_ia32_pmaxuq512_mask"
                | "__builtin_ia32_pminsd512_mask" | "__builtin_ia32_pminsq512_mask" | "__builtin_ia32_pminsq256_mask"
                | "__builtin_ia32_pminsq128_mask" | "__builtin_ia32_pminud512_mask" | "__builtin_ia32_pminuq512_mask"
                | "__builtin_ia32_prolq512_mask" | "__builtin_ia32_prorq512_mask" | "__builtin_ia32_pslldi512_mask"
                | "__builtin_ia32_psrldi512_mask" | "__builtin_ia32_psllqi512_mask" | "__builtin_ia32_psrlqi512_mask"
                | "__builtin_ia32_pslld512_mask" | "__builtin_ia32_psrld512_mask" | "__builtin_ia32_psllq512_mask"
                | "__builtin_ia32_psrlq512_mask" | "__builtin_ia32_psrad512_mask" | "__builtin_ia32_psraq512_mask"
                | "__builtin_ia32_psradi512_mask" | "__builtin_ia32_psraqi512_mask" | "__builtin_ia32_psrav16si_mask"
                | "__builtin_ia32_psrav8di_mask" | "__builtin_ia32_prolvd512_mask" | "__builtin_ia32_prorvd512_mask"
                | "__builtin_ia32_prolvq512_mask" | "__builtin_ia32_prorvq512_mask" | "__builtin_ia32_psllv16si_mask"
                | "__builtin_ia32_psrlv16si_mask" | "__builtin_ia32_psllv8di_mask" | "__builtin_ia32_psrlv8di_mask"
                | "__builtin_ia32_permvarsi512_mask" | "__builtin_ia32_vpermilvarps512_mask"
                | "__builtin_ia32_vpermilvarpd512_mask" | "__builtin_ia32_permvardi512_mask"
                | "__builtin_ia32_permvarsf512_mask" | "__builtin_ia32_permvarqi512_mask"
                | "__builtin_ia32_permvarqi256_mask" | "__builtin_ia32_permvarqi128_mask"
                | "__builtin_ia32_vpmultishiftqb512_mask" | "__builtin_ia32_vpmultishiftqb256_mask"
                | "__builtin_ia32_vpmultishiftqb128_mask"
                => {
                let mut new_args = args.to_vec();
                let arg3_type = gcc_func.get_param_type(2);
                let first_arg = builder.current_func().new_local(None, arg3_type, "undefined_for_intrinsic").to_rvalue();
                new_args.push(first_arg);
                let arg4_type = gcc_func.get_param_type(3);
                let minus_one = builder.context.new_rvalue_from_int(arg4_type, -1);
                new_args.push(minus_one);
                args = new_args.into();
            },
            "__builtin_ia32_pmaxuq256_mask" | "__builtin_ia32_pmaxuq128_mask" | "__builtin_ia32_pminuq256_mask"
                | "__builtin_ia32_pminuq128_mask" | "__builtin_ia32_prold256_mask" | "__builtin_ia32_prold128_mask"
                | "__builtin_ia32_prord512_mask" | "__builtin_ia32_prord256_mask" | "__builtin_ia32_prord128_mask"
                | "__builtin_ia32_prolq256_mask" | "__builtin_ia32_prolq128_mask" | "__builtin_ia32_prorq256_mask"
                | "__builtin_ia32_prorq128_mask" | "__builtin_ia32_psraq256_mask" | "__builtin_ia32_psraq128_mask"
                | "__builtin_ia32_psraqi256_mask" | "__builtin_ia32_psraqi128_mask" | "__builtin_ia32_psravq256_mask"
                | "__builtin_ia32_psravq128_mask" | "__builtin_ia32_prolvd256_mask" | "__builtin_ia32_prolvd128_mask"
                | "__builtin_ia32_prorvd256_mask" | "__builtin_ia32_prorvd128_mask" | "__builtin_ia32_prolvq256_mask"
                | "__builtin_ia32_prolvq128_mask" | "__builtin_ia32_prorvq256_mask" | "__builtin_ia32_prorvq128_mask"
                | "__builtin_ia32_permvardi256_mask" | "__builtin_ia32_permvardf512_mask" | "__builtin_ia32_permvardf256_mask"
                | "__builtin_ia32_pmulhuw512_mask" | "__builtin_ia32_pmulhw512_mask" | "__builtin_ia32_pmulhrsw512_mask"
                | "__builtin_ia32_pmaxuw512_mask" | "__builtin_ia32_pmaxub512_mask" | "__builtin_ia32_pmaxsw512_mask"
                | "__builtin_ia32_pmaxsb512_mask" | "__builtin_ia32_pminuw512_mask" | "__builtin_ia32_pminub512_mask"
                | "__builtin_ia32_pminsw512_mask" | "__builtin_ia32_pminsb512_mask"
                | "__builtin_ia32_pmaddwd512_mask" | "__builtin_ia32_pmaddubsw512_mask" | "__builtin_ia32_packssdw512_mask"
                | "__builtin_ia32_packsswb512_mask" | "__builtin_ia32_packusdw512_mask" | "__builtin_ia32_packuswb512_mask"
                | "__builtin_ia32_pavgw512_mask" | "__builtin_ia32_pavgb512_mask" | "__builtin_ia32_psllw512_mask"
                | "__builtin_ia32_psllwi512_mask" | "__builtin_ia32_psllv32hi_mask" | "__builtin_ia32_psrlw512_mask"
                | "__builtin_ia32_psrlwi512_mask" | "__builtin_ia32_psllv16hi_mask" | "__builtin_ia32_psllv8hi_mask"
                | "__builtin_ia32_psrlv32hi_mask" | "__builtin_ia32_psraw512_mask" | "__builtin_ia32_psrawi512_mask"
                | "__builtin_ia32_psrlv16hi_mask" | "__builtin_ia32_psrlv8hi_mask" | "__builtin_ia32_psrav32hi_mask"
                | "__builtin_ia32_permvarhi512_mask" | "__builtin_ia32_pshufb512_mask" | "__builtin_ia32_psrav16hi_mask"
                | "__builtin_ia32_psrav8hi_mask" | "__builtin_ia32_permvarhi256_mask" | "__builtin_ia32_permvarhi128_mask"
                => {
                let mut new_args = args.to_vec();
                let arg3_type = gcc_func.get_param_type(2);
                let vector_type = arg3_type.dyncast_vector().expect("vector type");
                let zero = builder.context.new_rvalue_zero(vector_type.get_element_type());
                let num_units = vector_type.get_num_units();
                let first_arg = builder.context.new_rvalue_from_vector(None, arg3_type, &vec![zero; num_units]);
                new_args.push(first_arg);
                let arg4_type = gcc_func.get_param_type(3);
                let minus_one = builder.context.new_rvalue_from_int(arg4_type, -1);
                new_args.push(minus_one);
                args = new_args.into();
            },
            "__builtin_ia32_dbpsadbw512_mask" | "__builtin_ia32_dbpsadbw256_mask" | "__builtin_ia32_dbpsadbw128_mask" => {
                let mut new_args = args.to_vec();
                let arg4_type = gcc_func.get_param_type(3);
                let vector_type = arg4_type.dyncast_vector().expect("vector type");
                let zero = builder.context.new_rvalue_zero(vector_type.get_element_type());
                let num_units = vector_type.get_num_units();
                let first_arg = builder.context.new_rvalue_from_vector(None, arg4_type, &vec![zero; num_units]);
                new_args.push(first_arg);
                let arg5_type = gcc_func.get_param_type(4);
                let minus_one = builder.context.new_rvalue_from_int(arg5_type, -1);
                new_args.push(minus_one);
                args = new_args.into();
            },
            "__builtin_ia32_vplzcntd_512_mask" | "__builtin_ia32_vplzcntd_256_mask" | "__builtin_ia32_vplzcntd_128_mask"
                | "__builtin_ia32_vplzcntq_512_mask" | "__builtin_ia32_vplzcntq_256_mask" | "__builtin_ia32_vplzcntq_128_mask" => {
                let mut new_args = args.to_vec();
                // Remove last arg as it doesn't seem to be used in GCC and is always false.
                new_args.pop();
                let arg2_type = gcc_func.get_param_type(1);
                let vector_type = arg2_type.dyncast_vector().expect("vector type");
                let zero = builder.context.new_rvalue_zero(vector_type.get_element_type());
                let num_units = vector_type.get_num_units();
                let first_arg = builder.context.new_rvalue_from_vector(None, arg2_type, &vec![zero; num_units]);
                new_args.push(first_arg);
                let arg3_type = gcc_func.get_param_type(2);
                let minus_one = builder.context.new_rvalue_from_int(arg3_type, -1);
                new_args.push(minus_one);
                args = new_args.into();
            },
            "__builtin_ia32_vpconflictsi_512_mask" | "__builtin_ia32_vpconflictsi_256_mask"
                | "__builtin_ia32_vpconflictsi_128_mask" | "__builtin_ia32_vpconflictdi_512_mask"
                | "__builtin_ia32_vpconflictdi_256_mask" | "__builtin_ia32_vpconflictdi_128_mask" => {
                let mut new_args = args.to_vec();
                let arg2_type = gcc_func.get_param_type(1);
                let vector_type = arg2_type.dyncast_vector().expect("vector type");
                let zero = builder.context.new_rvalue_zero(vector_type.get_element_type());
                let num_units = vector_type.get_num_units();
                let first_arg = builder.context.new_rvalue_from_vector(None, arg2_type, &vec![zero; num_units]);
                new_args.push(first_arg);
                let arg3_type = gcc_func.get_param_type(2);
                let minus_one = builder.context.new_rvalue_from_int(arg3_type, -1);
                new_args.push(minus_one);
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
                | "__builtin_ia32_divps512_mask" | "__builtin_ia32_divpd512_mask"
                | "__builtin_ia32_maxps512_mask" | "__builtin_ia32_maxpd512_mask"
                |  "__builtin_ia32_minps512_mask" | "__builtin_ia32_minpd512_mask" => {
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
            "__builtin_ia32_vpermi2vard512_mask" | "__builtin_ia32_vpermi2vard256_mask"
                | "__builtin_ia32_vpermi2vard128_mask" | "__builtin_ia32_vpermi2varq512_mask"
                | "__builtin_ia32_vpermi2varq256_mask" | "__builtin_ia32_vpermi2varq128_mask"
                | "__builtin_ia32_vpermi2varps512_mask" | "__builtin_ia32_vpermi2varps256_mask"
                | "__builtin_ia32_vpermi2varps128_mask" | "__builtin_ia32_vpermi2varpd512_mask"
                | "__builtin_ia32_vpermi2varpd256_mask" | "__builtin_ia32_vpermi2varpd128_mask" | "__builtin_ia32_vpmadd52huq512_mask"
                | "__builtin_ia32_vpmadd52luq512_mask" | "__builtin_ia32_vpmadd52huq256_mask" | "__builtin_ia32_vpmadd52luq256_mask"
                | "__builtin_ia32_vpmadd52huq128_mask"
                => {
                let mut new_args = args.to_vec();
                let arg4_type = gcc_func.get_param_type(3);
                let minus_one = builder.context.new_rvalue_from_int(arg4_type, -1);
                new_args.push(minus_one);
                args = new_args.into();
            },
            "__builtin_ia32_cvtdq2ps512_mask" | "__builtin_ia32_cvtudq2ps512_mask"
                | "__builtin_ia32_sqrtps512_mask" | "__builtin_ia32_sqrtpd512_mask" => {
                let mut new_args = args.to_vec();
                let last_arg = new_args.pop().expect("last arg");
                let arg2_type = gcc_func.get_param_type(1);
                let undefined = builder.current_func().new_local(None, arg2_type, "undefined_for_intrinsic").to_rvalue();
                new_args.push(undefined);
                let arg3_type = gcc_func.get_param_type(2);
                let minus_one = builder.context.new_rvalue_from_int(arg3_type, -1);
                new_args.push(minus_one);
                new_args.push(last_arg);
                args = new_args.into();
            },
            "__builtin_ia32_stmxcsr" => {
                args = vec![].into();
            },
            "__builtin_ia32_addcarryx_u64" | "__builtin_ia32_sbb_u64" | "__builtin_ia32_addcarryx_u32" | "__builtin_ia32_sbb_u32" => {
                let mut new_args = args.to_vec();
                let arg2_type = gcc_func.get_param_type(1);
                let variable = builder.current_func().new_local(None, arg2_type, "addcarryResult");
                new_args.push(variable.get_address(None));
                args = new_args.into();
            },
            "__builtin_ia32_vpermt2varqi512_mask" | "__builtin_ia32_vpermt2varqi256_mask"
                | "__builtin_ia32_vpermt2varqi128_mask" | "__builtin_ia32_vpermt2varhi512_mask"
                | "__builtin_ia32_vpermt2varhi256_mask" | "__builtin_ia32_vpermt2varhi128_mask"
                => {
                let new_args = args.to_vec();
                let arg4_type = gcc_func.get_param_type(3);
                let minus_one = builder.context.new_rvalue_from_int(arg4_type, -1);
                args = vec![new_args[1], new_args[0], new_args[2], minus_one].into();
            },
            "__builtin_ia32_xrstor" | "__builtin_ia32_xsavec" => {
                let new_args = args.to_vec();
                let thirty_two = builder.context.new_rvalue_from_int(new_args[1].get_type(), 32);
                let arg2 = new_args[1] << thirty_two | new_args[2];
                let arg2_type = gcc_func.get_param_type(1);
                let arg2 = builder.context.new_cast(None, arg2, arg2_type);
                args = vec![new_args[0], arg2].into();
            },
            "__builtin_prefetch" => {
                let mut new_args = args.to_vec();
                new_args.pop();
                args = new_args.into();
            },
            _ => (),
        }
    }
    else {
        match &*func_name {
            "__builtin_ia32_rndscaless_mask_round" | "__builtin_ia32_rndscalesd_mask_round" => {
                let new_args = args.to_vec();
                let arg3_type = gcc_func.get_param_type(2);
                let arg3 = builder.context.new_cast(None, new_args[4], arg3_type);
                let arg4_type = gcc_func.get_param_type(3);
                let arg4 = builder.context.new_bitcast(None, new_args[2], arg4_type);
                args = vec![new_args[0], new_args[1], arg3, arg4, new_args[3], new_args[5]].into();
            },
            // NOTE: the LLVM intrinsic receives 3 floats, but the GCC builtin requires 3 vectors.
            // FIXME: the intrinsics like _mm_mask_fmadd_sd should probably directly call the GCC
            // instrinsic to avoid this.
            "__builtin_ia32_vfmaddss3_round" => {
                let new_args = args.to_vec();
                let arg1_type = gcc_func.get_param_type(0);
                let arg2_type = gcc_func.get_param_type(1);
                let arg3_type = gcc_func.get_param_type(2);
                let a = builder.context.new_rvalue_from_vector(None, arg1_type, &[new_args[0]; 4]);
                let b = builder.context.new_rvalue_from_vector(None, arg2_type, &[new_args[1]; 4]);
                let c = builder.context.new_rvalue_from_vector(None, arg3_type, &[new_args[2]; 4]);
                args = vec![a, b, c, new_args[3]].into();
            },
            "__builtin_ia32_vfmaddsd3_round" => {
                let new_args = args.to_vec();
                let arg1_type = gcc_func.get_param_type(0);
                let arg2_type = gcc_func.get_param_type(1);
                let arg3_type = gcc_func.get_param_type(2);
                let a = builder.context.new_rvalue_from_vector(None, arg1_type, &[new_args[0]; 2]);
                let b = builder.context.new_rvalue_from_vector(None, arg2_type, &[new_args[1]; 2]);
                let c = builder.context.new_rvalue_from_vector(None, arg3_type, &[new_args[2]; 2]);
                args = vec![a, b, c, new_args[3]].into();
            },
            "__builtin_ia32_vfmaddsubpd256" | "__builtin_ia32_vfmaddsubps" | "__builtin_ia32_vfmaddsubps256"
                | "__builtin_ia32_vfmaddsubpd" => {
                if let Some(original_function_name) = original_function_name {
                    match &**original_function_name {
                        "llvm.x86.fma.vfmsubadd.pd.256" | "llvm.x86.fma.vfmsubadd.ps" | "llvm.x86.fma.vfmsubadd.ps.256"
                            | "llvm.x86.fma.vfmsubadd.pd" => {
                            // NOTE: since both llvm.x86.fma.vfmsubadd.ps and llvm.x86.fma.vfmaddsub.ps maps to
                            // __builtin_ia32_vfmaddsubps, only add minus if this comes from a
                            // subadd LLVM intrinsic, e.g. _mm256_fmsubadd_pd.
                            let mut new_args = args.to_vec();
                            let arg3 = &mut new_args[2];
                            *arg3 = builder.context.new_unary_op(None, UnaryOp::Minus, arg3.get_type(), *arg3);
                            args = new_args.into();
                        },
                        _ => (),
                    }
                }
            },
            "__builtin_ia32_ldmxcsr" => {
                // The builtin __builtin_ia32_ldmxcsr takes an integer value while llvm.x86.sse.ldmxcsr takes a pointer,
                // so dereference the pointer.
                let mut new_args = args.to_vec();
                let uint_ptr_type = builder.uint_type.make_pointer();
                let arg1 = builder.context.new_cast(None, args[0], uint_ptr_type);
                new_args[0] = arg1.dereference(None).to_rvalue();
                args = new_args.into();
            },
            "__builtin_ia32_rcp14sd_mask" | "__builtin_ia32_rcp14ss_mask" | "__builtin_ia32_rsqrt14sd_mask"
                | "__builtin_ia32_rsqrt14ss_mask" => {
                let new_args = args.to_vec();
                args = vec![new_args[1], new_args[0], new_args[2], new_args[3]].into();
            },
            "__builtin_ia32_sqrtsd_mask_round" | "__builtin_ia32_sqrtss_mask_round" => {
                let new_args = args.to_vec();
                args = vec![new_args[1], new_args[0], new_args[2], new_args[3], new_args[4]].into();
            },
            _ => (),
        }
    }

    args
}

pub fn adjust_intrinsic_return_value<'a, 'gcc, 'tcx>(builder: &Builder<'a, 'gcc, 'tcx>, mut return_value: RValue<'gcc>, func_name: &str, args: &[RValue<'gcc>], args_adjusted: bool, orig_args: &[RValue<'gcc>]) -> RValue<'gcc> {
    match func_name {
        "__builtin_ia32_vfmaddss3_round" | "__builtin_ia32_vfmaddsd3_round" => {
            #[cfg(feature="master")]
            {
                let zero = builder.context.new_rvalue_zero(builder.int_type);
                return_value = builder.context.new_vector_access(None, return_value, zero).to_rvalue();
            }
        },
        "__builtin_ia32_addcarryx_u64" | "__builtin_ia32_sbb_u64" | "__builtin_ia32_addcarryx_u32" | "__builtin_ia32_sbb_u32" => {
            // Both llvm.x86.addcarry.32 and llvm.x86.addcarryx.u32 points to the same GCC builtin,
            // but only the former requires adjusting the return value.
            // Those 2 LLVM intrinsics differ by their argument count, that's why we check if the
            // arguments were adjusted.
            if args_adjusted {
                let last_arg = args.last().expect("last arg");
                let field1 = builder.context.new_field(None, builder.u8_type, "carryFlag");
                let field2 = builder.context.new_field(None, args[1].get_type(), "carryResult");
                let struct_type = builder.context.new_struct_type(None, "addcarryResult", &[field1, field2]);
                return_value = builder.context.new_struct_constructor(None, struct_type.as_type(), None, &[return_value, last_arg.dereference(None).to_rvalue()]);
            }
        },
        "__builtin_ia32_stmxcsr" => {
            // The builtin __builtin_ia32_stmxcsr returns a value while llvm.x86.sse.stmxcsr writes
            // the result in its pointer argument.
            // We removed the argument since __builtin_ia32_stmxcsr takes no arguments, so we need
            // to get back the original argument to get the pointer we need to write the result to.
            let uint_ptr_type = builder.uint_type.make_pointer();
            let ptr = builder.context.new_cast(None, orig_args[0], uint_ptr_type);
            builder.llbb().add_assignment(None, ptr.dereference(None), return_value);
            // The return value was assigned to the result pointer above. In order to not call the
            // builtin twice, we overwrite the return value with a dummy value.
            return_value = builder.context.new_rvalue_zero(builder.int_type);
        },
        _ => (),
    }

    return_value
}

pub fn ignore_arg_cast(func_name: &str, index: usize, args_len: usize) -> bool {
    // FIXME(antoyo): find a way to refactor in order to avoid this hack.
    match func_name {
        // NOTE: these intrinsics have missing parameters before the last one, so ignore the
        // last argument type check.
        "__builtin_ia32_maxps512_mask" | "__builtin_ia32_maxpd512_mask"
            | "__builtin_ia32_minps512_mask" | "__builtin_ia32_minpd512_mask" | "__builtin_ia32_sqrtps512_mask"
            | "__builtin_ia32_sqrtpd512_mask" | "__builtin_ia32_addps512_mask" | "__builtin_ia32_addpd512_mask"
            | "__builtin_ia32_subps512_mask" | "__builtin_ia32_subpd512_mask"
            | "__builtin_ia32_mulps512_mask" | "__builtin_ia32_mulpd512_mask"
            | "__builtin_ia32_divps512_mask" | "__builtin_ia32_divpd512_mask"
            | "__builtin_ia32_vfmaddsubps512_mask" | "__builtin_ia32_vfmaddsubpd512_mask"
            | "__builtin_ia32_cvtdq2ps512_mask" | "__builtin_ia32_cvtudq2ps512_mask" => {
                if index == args_len - 1 {
                    return true;
                }
            },
        "__builtin_ia32_rndscaless_mask_round" | "__builtin_ia32_rndscalesd_mask_round" => {
            if index == 2 || index == 3 {
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
        // NOTE: the LLVM intrinsic receives 3 floats, but the GCC builtin requires 3 vectors.
        "__builtin_ia32_vfmaddss3_round" | "__builtin_ia32_vfmaddsd3_round" => return true,
        "__builtin_ia32_vplzcntd_512_mask" | "__builtin_ia32_vplzcntd_256_mask" | "__builtin_ia32_vplzcntd_128_mask"
            | "__builtin_ia32_vplzcntq_512_mask" | "__builtin_ia32_vplzcntq_256_mask" | "__builtin_ia32_vplzcntq_128_mask" => {
            if index == args_len - 1 {
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
        "llvm.x86.xgetbv" | "llvm.x86.sse2.pause" => {
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
    match name {
        "llvm.prefetch" => {
            let gcc_name = "__builtin_prefetch";
            let func = cx.context.get_builtin_function(gcc_name);
            cx.functions.borrow_mut().insert(gcc_name.to_string(), func);
            return func
        },
        _ => (),
    }

    let gcc_name = match name {
        "llvm.x86.xgetbv" => "__builtin_ia32_xgetbv",
        // NOTE: this doc specifies the equivalent GCC builtins: http://huonw.github.io/llvmint/llvmint/x86/index.html
        "llvm.sqrt.v2f64" => "__builtin_ia32_sqrtpd",
        "llvm.x86.avx512.pmul.dq.512" => "__builtin_ia32_pmuldq512_mask",
        "llvm.x86.avx512.pmulu.dq.512" => "__builtin_ia32_pmuludq512_mask",
        "llvm.x86.avx512.max.ps.512" => "__builtin_ia32_maxps512_mask",
        "llvm.x86.avx512.max.pd.512" => "__builtin_ia32_maxpd512_mask",
        "llvm.x86.avx512.min.ps.512" => "__builtin_ia32_minps512_mask",
        "llvm.x86.avx512.min.pd.512" => "__builtin_ia32_minpd512_mask",
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
        "llvm.x86.avx512.sitofp.round.v16f32.v16i32" => "__builtin_ia32_cvtdq2ps512_mask",
        "llvm.x86.avx512.uitofp.round.v16f32.v16i32" => "__builtin_ia32_cvtudq2ps512_mask",
        "llvm.x86.avx512.mask.ucmp.d.512" => "__builtin_ia32_ucmpd512_mask",
        "llvm.x86.avx512.mask.ucmp.d.256" => "__builtin_ia32_ucmpd256_mask",
        "llvm.x86.avx512.mask.ucmp.d.128" => "__builtin_ia32_ucmpd128_mask",
        "llvm.x86.avx512.mask.cmp.d.512" => "__builtin_ia32_cmpd512_mask",
        "llvm.x86.avx512.mask.cmp.d.256" => "__builtin_ia32_cmpd256_mask",
        "llvm.x86.avx512.mask.cmp.d.128" => "__builtin_ia32_cmpd128_mask",
        "llvm.x86.avx512.mask.ucmp.q.512" => "__builtin_ia32_ucmpq512_mask",
        "llvm.x86.avx512.mask.ucmp.q.256" => "__builtin_ia32_ucmpq256_mask",
        "llvm.x86.avx512.mask.ucmp.q.128" => "__builtin_ia32_ucmpq128_mask",
        "llvm.x86.avx512.mask.cmp.q.512" => "__builtin_ia32_cmpq512_mask",
        "llvm.x86.avx512.mask.cmp.q.256" => "__builtin_ia32_cmpq256_mask",
        "llvm.x86.avx512.mask.cmp.q.128" => "__builtin_ia32_cmpq128_mask",
        "llvm.x86.avx512.mask.max.ss.round" => "__builtin_ia32_maxss_mask_round",
        "llvm.x86.avx512.mask.max.sd.round" => "__builtin_ia32_maxsd_mask_round",
        "llvm.x86.avx512.mask.min.ss.round" => "__builtin_ia32_minss_mask_round",
        "llvm.x86.avx512.mask.min.sd.round" => "__builtin_ia32_minsd_mask_round",
        "llvm.x86.avx512.mask.sqrt.ss" => "__builtin_ia32_sqrtss_mask_round",
        "llvm.x86.avx512.mask.sqrt.sd" => "__builtin_ia32_sqrtsd_mask_round",
        "llvm.x86.avx512.mask.getexp.ss" => "__builtin_ia32_getexpss_mask_round",
        "llvm.x86.avx512.mask.getexp.sd" => "__builtin_ia32_getexpsd_mask_round",
        "llvm.x86.avx512.mask.getmant.ss" => "__builtin_ia32_getmantss_mask_round",
        "llvm.x86.avx512.mask.getmant.sd" => "__builtin_ia32_getmantsd_mask_round",
        "llvm.x86.avx512.mask.rndscale.ss" => "__builtin_ia32_rndscaless_mask_round",
        "llvm.x86.avx512.mask.rndscale.sd" => "__builtin_ia32_rndscalesd_mask_round",
        "llvm.x86.avx512.mask.scalef.ss" => "__builtin_ia32_scalefss_mask_round",
        "llvm.x86.avx512.mask.scalef.sd" => "__builtin_ia32_scalefsd_mask_round",
        "llvm.x86.avx512.vfmadd.f32" => "__builtin_ia32_vfmaddss3_round",
        "llvm.x86.avx512.vfmadd.f64" => "__builtin_ia32_vfmaddsd3_round",
        "llvm.ceil.v4f64" => "__builtin_ia32_ceilpd256",
        "llvm.ceil.v8f32" => "__builtin_ia32_ceilps256",
        "llvm.floor.v4f64" => "__builtin_ia32_floorpd256",
        "llvm.floor.v8f32" => "__builtin_ia32_floorps256",
        "llvm.sqrt.v4f64" => "__builtin_ia32_sqrtpd256",
        "llvm.x86.sse.stmxcsr" => "__builtin_ia32_stmxcsr",
        "llvm.x86.sse.ldmxcsr" => "__builtin_ia32_ldmxcsr",
        "llvm.ctpop.v16i32" => "__builtin_ia32_vpopcountd_v16si",
        "llvm.ctpop.v8i32" => "__builtin_ia32_vpopcountd_v8si",
        "llvm.ctpop.v4i32" => "__builtin_ia32_vpopcountd_v4si",
        "llvm.ctpop.v8i64" => "__builtin_ia32_vpopcountq_v8di",
        "llvm.ctpop.v4i64" => "__builtin_ia32_vpopcountq_v4di",
        "llvm.ctpop.v2i64" => "__builtin_ia32_vpopcountq_v2di",
        "llvm.x86.addcarry.64" => "__builtin_ia32_addcarryx_u64",
        "llvm.x86.subborrow.64" => "__builtin_ia32_sbb_u64",
        "llvm.floor.v2f64" => "__builtin_ia32_floorpd",
        "llvm.floor.v4f32" => "__builtin_ia32_floorps",
        "llvm.ceil.v2f64" => "__builtin_ia32_ceilpd",
        "llvm.ceil.v4f32" => "__builtin_ia32_ceilps",
        "llvm.fma.v2f64" => "__builtin_ia32_vfmaddpd",
        "llvm.fma.v4f64" => "__builtin_ia32_vfmaddpd256",
        "llvm.fma.v4f32" => "__builtin_ia32_vfmaddps",
        "llvm.fma.v8f32" => "__builtin_ia32_vfmaddps256",
        "llvm.ctlz.v16i32" => "__builtin_ia32_vplzcntd_512_mask",
        "llvm.ctlz.v8i32" => "__builtin_ia32_vplzcntd_256_mask",
        "llvm.ctlz.v4i32" => "__builtin_ia32_vplzcntd_128_mask",
        "llvm.ctlz.v8i64" => "__builtin_ia32_vplzcntq_512_mask",
        "llvm.ctlz.v4i64" => "__builtin_ia32_vplzcntq_256_mask",
        "llvm.ctlz.v2i64" => "__builtin_ia32_vplzcntq_128_mask",
        "llvm.ctpop.v32i16" => "__builtin_ia32_vpopcountw_v32hi",
        "llvm.x86.fma.vfmsub.sd" => "__builtin_ia32_vfmsubsd3",
        "llvm.x86.fma.vfmsub.ss" => "__builtin_ia32_vfmsubss3",
        "llvm.x86.fma.vfmsubadd.pd" => "__builtin_ia32_vfmaddsubpd",
        "llvm.x86.fma.vfmsubadd.pd.256" => "__builtin_ia32_vfmaddsubpd256",
        "llvm.x86.fma.vfmsubadd.ps" => "__builtin_ia32_vfmaddsubps",
        "llvm.x86.fma.vfmsubadd.ps.256" => "__builtin_ia32_vfmaddsubps256",
        "llvm.x86.fma.vfnmadd.sd" => "__builtin_ia32_vfnmaddsd3",
        "llvm.x86.fma.vfnmadd.ss" => "__builtin_ia32_vfnmaddss3",
        "llvm.x86.fma.vfnmsub.sd" => "__builtin_ia32_vfnmsubsd3",
        "llvm.x86.fma.vfnmsub.ss" => "__builtin_ia32_vfnmsubss3",
        "llvm.x86.avx512.conflict.d.512" => "__builtin_ia32_vpconflictsi_512_mask",
        "llvm.x86.avx512.conflict.d.256" => "__builtin_ia32_vpconflictsi_256_mask",
        "llvm.x86.avx512.conflict.d.128" => "__builtin_ia32_vpconflictsi_128_mask",
        "llvm.x86.avx512.conflict.q.512" => "__builtin_ia32_vpconflictdi_512_mask",
        "llvm.x86.avx512.conflict.q.256" => "__builtin_ia32_vpconflictdi_256_mask",
        "llvm.x86.avx512.conflict.q.128" => "__builtin_ia32_vpconflictdi_128_mask",
        "llvm.x86.avx512.vpermi2var.qi.512" => "__builtin_ia32_vpermt2varqi512_mask",
        "llvm.x86.avx512.vpermi2var.qi.256" => "__builtin_ia32_vpermt2varqi256_mask",
        "llvm.x86.avx512.vpermi2var.qi.128" => "__builtin_ia32_vpermt2varqi128_mask",
        "llvm.x86.avx512.permvar.qi.512" => "__builtin_ia32_permvarqi512_mask",
        "llvm.x86.avx512.permvar.qi.256" => "__builtin_ia32_permvarqi256_mask",
        "llvm.x86.avx512.permvar.qi.128" => "__builtin_ia32_permvarqi128_mask",
        "llvm.x86.avx512.pmultishift.qb.512" => "__builtin_ia32_vpmultishiftqb512_mask",
        "llvm.x86.avx512.pmultishift.qb.256" => "__builtin_ia32_vpmultishiftqb256_mask",
        "llvm.x86.avx512.pmultishift.qb.128" => "__builtin_ia32_vpmultishiftqb128_mask",
        "llvm.ctpop.v16i16" => "__builtin_ia32_vpopcountw_v16hi",
        "llvm.ctpop.v8i16" => "__builtin_ia32_vpopcountw_v8hi",
        "llvm.ctpop.v64i8" => "__builtin_ia32_vpopcountb_v64qi",
        "llvm.ctpop.v32i8" => "__builtin_ia32_vpopcountb_v32qi",
        "llvm.ctpop.v16i8" => "__builtin_ia32_vpopcountb_v16qi",
        "llvm.x86.avx512.mask.vpshufbitqmb.512" => "__builtin_ia32_vpshufbitqmb512_mask",
        "llvm.x86.avx512.mask.vpshufbitqmb.256" => "__builtin_ia32_vpshufbitqmb256_mask",
        "llvm.x86.avx512.mask.vpshufbitqmb.128" => "__builtin_ia32_vpshufbitqmb128_mask",
        "llvm.x86.avx512.mask.ucmp.w.512" => "__builtin_ia32_ucmpw512_mask",
        "llvm.x86.avx512.mask.ucmp.w.256" => "__builtin_ia32_ucmpw256_mask",
        "llvm.x86.avx512.mask.ucmp.w.128" => "__builtin_ia32_ucmpw128_mask",
        "llvm.x86.avx512.mask.ucmp.b.512" => "__builtin_ia32_ucmpb512_mask",
        "llvm.x86.avx512.mask.ucmp.b.256" => "__builtin_ia32_ucmpb256_mask",
        "llvm.x86.avx512.mask.ucmp.b.128" => "__builtin_ia32_ucmpb128_mask",
        "llvm.x86.avx512.mask.cmp.w.512" => "__builtin_ia32_cmpw512_mask",
        "llvm.x86.avx512.mask.cmp.w.256" => "__builtin_ia32_cmpw256_mask",
        "llvm.x86.avx512.mask.cmp.w.128" => "__builtin_ia32_cmpw128_mask",
        "llvm.x86.avx512.mask.cmp.b.512" => "__builtin_ia32_cmpb512_mask",
        "llvm.x86.avx512.mask.cmp.b.256" => "__builtin_ia32_cmpb256_mask",
        "llvm.x86.avx512.mask.cmp.b.128" => "__builtin_ia32_cmpb128_mask",
        "llvm.x86.xrstor" => "__builtin_ia32_xrstor",
        "llvm.x86.xsavec" => "__builtin_ia32_xsavec",
        "llvm.x86.addcarry.32" => "__builtin_ia32_addcarryx_u32",
        "llvm.x86.subborrow.32" => "__builtin_ia32_sbb_u32",
        "llvm.x86.avx512.mask.compress.store.w.512" => "__builtin_ia32_compressstoreuhi512_mask",
        "llvm.x86.avx512.mask.compress.store.w.256" => "__builtin_ia32_compressstoreuhi256_mask",
        "llvm.x86.avx512.mask.compress.store.w.128" => "__builtin_ia32_compressstoreuhi128_mask",
        "llvm.x86.avx512.mask.compress.store.b.512" => "__builtin_ia32_compressstoreuqi512_mask",
        "llvm.x86.avx512.mask.compress.store.b.256" => "__builtin_ia32_compressstoreuqi256_mask",
        "llvm.x86.avx512.mask.compress.store.b.128" => "__builtin_ia32_compressstoreuqi128_mask",
        "llvm.x86.avx512.mask.compress.w.512" => "__builtin_ia32_compresshi512_mask",
        "llvm.x86.avx512.mask.compress.w.256" => "__builtin_ia32_compresshi256_mask",
        "llvm.x86.avx512.mask.compress.w.128" => "__builtin_ia32_compresshi128_mask",
        "llvm.x86.avx512.mask.compress.b.512" => "__builtin_ia32_compressqi512_mask",
        "llvm.x86.avx512.mask.compress.b.256" => "__builtin_ia32_compressqi256_mask",
        "llvm.x86.avx512.mask.compress.b.128" => "__builtin_ia32_compressqi128_mask",
        "llvm.x86.avx512.mask.expand.w.512" => "__builtin_ia32_expandhi512_mask",
        "llvm.x86.avx512.mask.expand.w.256" => "__builtin_ia32_expandhi256_mask",
        "llvm.x86.avx512.mask.expand.w.128" => "__builtin_ia32_expandhi128_mask",
        "llvm.x86.avx512.mask.expand.b.512" => "__builtin_ia32_expandqi512_mask",
        "llvm.x86.avx512.mask.expand.b.256" => "__builtin_ia32_expandqi256_mask",
        "llvm.x86.avx512.mask.expand.b.128" => "__builtin_ia32_expandqi128_mask",
        "llvm.fshl.v8i64" => "__builtin_ia32_vpshldv_v8di",
        "llvm.fshl.v4i64" => "__builtin_ia32_vpshldv_v4di",
        "llvm.fshl.v2i64" => "__builtin_ia32_vpshldv_v2di",
        "llvm.fshl.v16i32" => "__builtin_ia32_vpshldv_v16si",
        "llvm.fshl.v8i32" => "__builtin_ia32_vpshldv_v8si",
        "llvm.fshl.v4i32" => "__builtin_ia32_vpshldv_v4si",
        "llvm.fshl.v32i16" => "__builtin_ia32_vpshldv_v32hi",
        "llvm.fshl.v16i16" => "__builtin_ia32_vpshldv_v16hi",
        "llvm.fshl.v8i16" => "__builtin_ia32_vpshldv_v8hi",
        "llvm.fshr.v8i64" => "__builtin_ia32_vpshrdv_v8di",
        "llvm.fshr.v4i64" => "__builtin_ia32_vpshrdv_v4di",
        "llvm.fshr.v2i64" => "__builtin_ia32_vpshrdv_v2di",
        "llvm.fshr.v16i32" => "__builtin_ia32_vpshrdv_v16si",
        "llvm.fshr.v8i32" => "__builtin_ia32_vpshrdv_v8si",
        "llvm.fshr.v4i32" => "__builtin_ia32_vpshrdv_v4si",
        "llvm.fshr.v32i16" => "__builtin_ia32_vpshrdv_v32hi",
        "llvm.fshr.v16i16" => "__builtin_ia32_vpshrdv_v16hi",
        "llvm.fshr.v8i16" => "__builtin_ia32_vpshrdv_v8hi",
        "llvm.x86.fma.vfmadd.sd" => "__builtin_ia32_vfmaddsd3",
        "llvm.x86.fma.vfmadd.ss" => "__builtin_ia32_vfmaddss3",

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
        "llvm.x86.avx512.pslli.d.512" => "__builtin_ia32_pslldi512_mask",
        "llvm.x86.avx512.psrli.d.512" => "__builtin_ia32_psrldi512_mask",
        "llvm.x86.avx512.pslli.q.512" => "__builtin_ia32_psllqi512_mask",
        "llvm.x86.avx512.psrli.q.512" => "__builtin_ia32_psrlqi512_mask",
        "llvm.x86.avx512.psll.d.512" => "__builtin_ia32_pslld512_mask",
        "llvm.x86.avx512.psrl.d.512" => "__builtin_ia32_psrld512_mask",
        "llvm.x86.avx512.psll.q.512" => "__builtin_ia32_psllq512_mask",
        "llvm.x86.avx512.psrl.q.512" => "__builtin_ia32_psrlq512_mask",
        "llvm.x86.avx512.psra.d.512" => "__builtin_ia32_psrad512_mask",
        "llvm.x86.avx512.psra.q.512" => "__builtin_ia32_psraq512_mask",
        "llvm.x86.avx512.psra.q.256" => "__builtin_ia32_psraq256_mask",
        "llvm.x86.avx512.psra.q.128" => "__builtin_ia32_psraq128_mask",
        "llvm.x86.avx512.psrai.d.512" => "__builtin_ia32_psradi512_mask",
        "llvm.x86.avx512.psrai.q.512" => "__builtin_ia32_psraqi512_mask",
        "llvm.x86.avx512.psrai.q.256" => "__builtin_ia32_psraqi256_mask",
        "llvm.x86.avx512.psrai.q.128" => "__builtin_ia32_psraqi128_mask",
        "llvm.x86.avx512.psrav.d.512" => "__builtin_ia32_psrav16si_mask",
        "llvm.x86.avx512.psrav.q.512" => "__builtin_ia32_psrav8di_mask",
        "llvm.x86.avx512.psrav.q.256" => "__builtin_ia32_psravq256_mask",
        "llvm.x86.avx512.psrav.q.128" => "__builtin_ia32_psravq128_mask",
        "llvm.x86.avx512.psllv.d.512" => "__builtin_ia32_psllv16si_mask",
        "llvm.x86.avx512.psrlv.d.512" => "__builtin_ia32_psrlv16si_mask",
        "llvm.x86.avx512.psllv.q.512" => "__builtin_ia32_psllv8di_mask",
        "llvm.x86.avx512.psrlv.q.512" => "__builtin_ia32_psrlv8di_mask",
        "llvm.x86.avx512.permvar.si.512" => "__builtin_ia32_permvarsi512_mask",
        "llvm.x86.avx512.vpermilvar.ps.512" => "__builtin_ia32_vpermilvarps512_mask",
        "llvm.x86.avx512.vpermilvar.pd.512" => "__builtin_ia32_vpermilvarpd512_mask",
        "llvm.x86.avx512.permvar.di.512" => "__builtin_ia32_permvardi512_mask",
        "llvm.x86.avx512.permvar.di.256" => "__builtin_ia32_permvardi256_mask",
        "llvm.x86.avx512.permvar.sf.512" => "__builtin_ia32_permvarsf512_mask",
        "llvm.x86.avx512.permvar.df.512" => "__builtin_ia32_permvardf512_mask",
        "llvm.x86.avx512.permvar.df.256" => "__builtin_ia32_permvardf256_mask",
        "llvm.x86.avx512.vpermi2var.d.512" => "__builtin_ia32_vpermi2vard512_mask",
        "llvm.x86.avx512.vpermi2var.d.256" => "__builtin_ia32_vpermi2vard256_mask",
        "llvm.x86.avx512.vpermi2var.d.128" => "__builtin_ia32_vpermi2vard128_mask",
        "llvm.x86.avx512.vpermi2var.q.512" => "__builtin_ia32_vpermi2varq512_mask",
        "llvm.x86.avx512.vpermi2var.q.256" => "__builtin_ia32_vpermi2varq256_mask",
        "llvm.x86.avx512.vpermi2var.q.128" => "__builtin_ia32_vpermi2varq128_mask",
        "llvm.x86.avx512.vpermi2var.ps.512" => "__builtin_ia32_vpermi2varps512_mask",
        "llvm.x86.avx512.vpermi2var.ps.256" => "__builtin_ia32_vpermi2varps256_mask",
        "llvm.x86.avx512.vpermi2var.ps.128" => "__builtin_ia32_vpermi2varps128_mask",
        "llvm.x86.avx512.vpermi2var.pd.512" => "__builtin_ia32_vpermi2varpd512_mask",
        "llvm.x86.avx512.vpermi2var.pd.256" => "__builtin_ia32_vpermi2varpd256_mask",
        "llvm.x86.avx512.vpermi2var.pd.128" => "__builtin_ia32_vpermi2varpd128_mask",
        "llvm.x86.avx512.mask.add.ss.round" => "__builtin_ia32_addss_mask_round",
        "llvm.x86.avx512.mask.add.sd.round" => "__builtin_ia32_addsd_mask_round",
        "llvm.x86.avx512.mask.sub.ss.round" => "__builtin_ia32_subss_mask_round",
        "llvm.x86.avx512.mask.sub.sd.round" => "__builtin_ia32_subsd_mask_round",
        "llvm.x86.avx512.mask.mul.ss.round" => "__builtin_ia32_mulss_mask_round",
        "llvm.x86.avx512.mask.mul.sd.round" => "__builtin_ia32_mulsd_mask_round",
        "llvm.x86.avx512.mask.div.ss.round" => "__builtin_ia32_divss_mask_round",
        "llvm.x86.avx512.mask.div.sd.round" => "__builtin_ia32_divsd_mask_round",
        "llvm.x86.avx512.mask.cvtss2sd.round" => "__builtin_ia32_cvtss2sd_mask_round",
        "llvm.x86.avx512.mask.cvtsd2ss.round" => "__builtin_ia32_cvtsd2ss_mask_round",
        "llvm.x86.avx512.mask.range.ss" => "__builtin_ia32_rangess128_mask_round",
        "llvm.x86.avx512.mask.range.sd" => "__builtin_ia32_rangesd128_mask_round",
        "llvm.x86.avx512.rcp28.ss" => "__builtin_ia32_rcp28ss_mask_round",
        "llvm.x86.avx512.rcp28.sd" => "__builtin_ia32_rcp28sd_mask_round",
        "llvm.x86.avx512.rsqrt28.ss" => "__builtin_ia32_rsqrt28ss_mask_round",
        "llvm.x86.avx512.rsqrt28.sd" => "__builtin_ia32_rsqrt28sd_mask_round",
        "llvm.x86.avx512fp16.mask.add.sh.round" => "__builtin_ia32_addsh_mask_round",
        "llvm.x86.avx512fp16.mask.div.sh.round" => "__builtin_ia32_divsh_mask_round",
        "llvm.x86.avx512fp16.mask.getmant.sh" => "__builtin_ia32_getmantsh_mask_round",
        "llvm.x86.avx512fp16.mask.max.sh.round" => "__builtin_ia32_maxsh_mask_round",
        "llvm.x86.avx512fp16.mask.min.sh.round" => "__builtin_ia32_minsh_mask_round",
        "llvm.x86.avx512fp16.mask.mul.sh.round" => "__builtin_ia32_mulsh_mask_round",
        "llvm.x86.avx512fp16.mask.rndscale.sh" => "__builtin_ia32_rndscalesh_mask_round",
        "llvm.x86.avx512fp16.mask.scalef.sh" => "__builtin_ia32_scalefsh_mask_round",
        "llvm.x86.avx512fp16.mask.sub.sh.round" => "__builtin_ia32_subsh_mask_round",
        "llvm.x86.avx512fp16.mask.vcvtsd2sh.round" => "__builtin_ia32_vcvtsd2sh_mask_round",
        "llvm.x86.avx512fp16.mask.vcvtsh2sd.round" => "__builtin_ia32_vcvtsh2sd_mask_round",
        "llvm.x86.avx512fp16.mask.vcvtsh2ss.round" => "__builtin_ia32_vcvtsh2ss_mask_round",
        "llvm.x86.avx512fp16.mask.vcvtss2sh.round" => "__builtin_ia32_vcvtss2sh_mask_round",
        "llvm.x86.aesni.aesenc.256" => "__builtin_ia32_vaesenc_v32qi",
        "llvm.x86.aesni.aesenclast.256" => "__builtin_ia32_vaesenclast_v32qi",
        "llvm.x86.aesni.aesdec.256" => "__builtin_ia32_vaesdec_v32qi",
        "llvm.x86.aesni.aesdeclast.256" => "__builtin_ia32_vaesdeclast_v32qi",
        "llvm.x86.aesni.aesenc.512" => "__builtin_ia32_vaesenc_v64qi",
        "llvm.x86.aesni.aesenclast.512" => "__builtin_ia32_vaesenclast_v64qi",
        "llvm.x86.aesni.aesdec.512" => "__builtin_ia32_vaesdec_v64qi",
        "llvm.x86.aesni.aesdeclast.512" => "__builtin_ia32_vaesdeclast_v64qi",
        "llvm.x86.avx512bf16.cvtne2ps2bf16.128" => "__builtin_ia32_cvtne2ps2bf16_v8bf",
        "llvm.x86.avx512bf16.cvtne2ps2bf16.256" => "__builtin_ia32_cvtne2ps2bf16_v16bf",
        "llvm.x86.avx512bf16.cvtne2ps2bf16.512" => "__builtin_ia32_cvtne2ps2bf16_v32bf",
        "llvm.x86.avx512bf16.cvtneps2bf16.256" => "__builtin_ia32_cvtneps2bf16_v8sf",
        "llvm.x86.avx512bf16.cvtneps2bf16.512" => "__builtin_ia32_cvtneps2bf16_v16sf",
        "llvm.x86.avx512bf16.dpbf16ps.128" => "__builtin_ia32_dpbf16ps_v4sf",
        "llvm.x86.avx512bf16.dpbf16ps.256" => "__builtin_ia32_dpbf16ps_v8sf",
        "llvm.x86.avx512bf16.dpbf16ps.512" => "__builtin_ia32_dpbf16ps_v16sf",
        "llvm.x86.pclmulqdq.512" => "__builtin_ia32_vpclmulqdq_v8di",
        "llvm.x86.pclmulqdq.256" => "__builtin_ia32_vpclmulqdq_v4di",
        "llvm.x86.avx512.pmulhu.w.512" => "__builtin_ia32_pmulhuw512_mask",
        "llvm.x86.avx512.pmulh.w.512" => "__builtin_ia32_pmulhw512_mask",
        "llvm.x86.avx512.pmul.hr.sw.512" => "__builtin_ia32_pmulhrsw512_mask",
        "llvm.x86.avx512.pmaddw.d.512" => "__builtin_ia32_pmaddwd512_mask",
        "llvm.x86.avx512.pmaddubs.w.512" => "__builtin_ia32_pmaddubsw512_mask",
        "llvm.x86.avx512.packssdw.512" => "__builtin_ia32_packssdw512_mask",
        "llvm.x86.avx512.packsswb.512" => "__builtin_ia32_packsswb512_mask",
        "llvm.x86.avx512.packusdw.512" => "__builtin_ia32_packusdw512_mask",
        "llvm.x86.avx512.packuswb.512" => "__builtin_ia32_packuswb512_mask",
        "llvm.x86.avx512.pavg.w.512" => "__builtin_ia32_pavgw512_mask",
        "llvm.x86.avx512.pavg.b.512" => "__builtin_ia32_pavgb512_mask",
        "llvm.x86.avx512.psll.w.512" => "__builtin_ia32_psllw512_mask",
        "llvm.x86.avx512.pslli.w.512" => "__builtin_ia32_psllwi512_mask",
        "llvm.x86.avx512.psllv.w.512" => "__builtin_ia32_psllv32hi_mask",
        "llvm.x86.avx512.psllv.w.256" => "__builtin_ia32_psllv16hi_mask",
        "llvm.x86.avx512.psllv.w.128" => "__builtin_ia32_psllv8hi_mask",
        "llvm.x86.avx512.psrl.w.512" => "__builtin_ia32_psrlw512_mask",
        "llvm.x86.avx512.psrli.w.512" => "__builtin_ia32_psrlwi512_mask",
        "llvm.x86.avx512.psrlv.w.512" => "__builtin_ia32_psrlv32hi_mask",
        "llvm.x86.avx512.psrlv.w.256" => "__builtin_ia32_psrlv16hi_mask",
        "llvm.x86.avx512.psrlv.w.128" => "__builtin_ia32_psrlv8hi_mask",
        "llvm.x86.avx512.psra.w.512" => "__builtin_ia32_psraw512_mask",
        "llvm.x86.avx512.psrai.w.512" => "__builtin_ia32_psrawi512_mask",
        "llvm.x86.avx512.psrav.w.512" => "__builtin_ia32_psrav32hi_mask",
        "llvm.x86.avx512.psrav.w.256" => "__builtin_ia32_psrav16hi_mask",
        "llvm.x86.avx512.psrav.w.128" => "__builtin_ia32_psrav8hi_mask",
        "llvm.x86.avx512.vpermi2var.hi.512" => "__builtin_ia32_vpermt2varhi512_mask",
        "llvm.x86.avx512.vpermi2var.hi.256" => "__builtin_ia32_vpermt2varhi256_mask",
        "llvm.x86.avx512.vpermi2var.hi.128" => "__builtin_ia32_vpermt2varhi128_mask",
        "llvm.x86.avx512.permvar.hi.512" => "__builtin_ia32_permvarhi512_mask",
        "llvm.x86.avx512.permvar.hi.256" => "__builtin_ia32_permvarhi256_mask",
        "llvm.x86.avx512.permvar.hi.128" => "__builtin_ia32_permvarhi128_mask",
        "llvm.x86.avx512.pshuf.b.512" => "__builtin_ia32_pshufb512_mask",
        "llvm.x86.avx512.dbpsadbw.512" => "__builtin_ia32_dbpsadbw512_mask",
        "llvm.x86.avx512.dbpsadbw.256" => "__builtin_ia32_dbpsadbw256_mask",
        "llvm.x86.avx512.dbpsadbw.128" => "__builtin_ia32_dbpsadbw128_mask",
        "llvm.x86.avx512.vpmadd52h.uq.512" => "__builtin_ia32_vpmadd52huq512_mask",
        "llvm.x86.avx512.vpmadd52l.uq.512" => "__builtin_ia32_vpmadd52luq512_mask",
        "llvm.x86.avx512.vpmadd52h.uq.256" => "__builtin_ia32_vpmadd52huq256_mask",
        "llvm.x86.avx512.vpmadd52l.uq.256" => "__builtin_ia32_vpmadd52luq256_mask",
        "llvm.x86.avx512.vpmadd52h.uq.128" => "__builtin_ia32_vpmadd52huq128_mask",
        "llvm.x86.avx512.vpdpwssd.512" => "__builtin_ia32_vpdpwssd_v16si",
        "llvm.x86.avx512.vpdpwssd.256" => "__builtin_ia32_vpdpwssd_v8si",
        "llvm.x86.avx512.vpdpwssd.128" => "__builtin_ia32_vpdpwssd_v4si",
        "llvm.x86.avx512.vpdpwssds.512" => "__builtin_ia32_vpdpwssds_v16si",
        "llvm.x86.avx512.vpdpwssds.256" => "__builtin_ia32_vpdpwssds_v8si",
        "llvm.x86.avx512.vpdpwssds.128" => "__builtin_ia32_vpdpwssds_v4si",
        "llvm.x86.avx512.vpdpbusd.512" => "__builtin_ia32_vpdpbusd_v16si",
        "llvm.x86.avx512.vpdpbusd.256" => "__builtin_ia32_vpdpbusd_v8si",
        "llvm.x86.avx512.vpdpbusd.128" => "__builtin_ia32_vpdpbusd_v4si",
        "llvm.x86.avx512.vpdpbusds.512" => "__builtin_ia32_vpdpbusds_v16si",
        "llvm.x86.avx512.vpdpbusds.256" => "__builtin_ia32_vpdpbusds_v8si",
        "llvm.x86.avx512.vpdpbusds.128" => "__builtin_ia32_vpdpbusds_v4si",

        // NOTE: this file is generated by https://github.com/GuillaumeGomez/llvmint/blob/master/generate_list.py
        _ => include!("archs.rs"),
    };

    let func = cx.context.get_target_builtin_function(gcc_name);
    cx.functions.borrow_mut().insert(gcc_name.to_string(), func);
    func
}
