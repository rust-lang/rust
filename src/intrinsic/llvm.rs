use gccjit::Function;

use crate::context::CodegenCx;

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
        "llvm.x86.avx512.vfmaddsub.ps.512" => "__builtin_ia32_vfmaddps512_mask",
        "llvm.x86.avx512.vfmaddsub.pd.512" => "__builtin_ia32_vfmaddpd512_mask",
        "llvm.x86.avx512.rcp14.ps.256" => "__builtin_ia32_rcp14ps256_mask",
        "llvm.x86.avx512.rcp14.ps.128" => "__builtin_ia32_rcp14ps128_mask",
        "llvm.x86.avx512.rcp14.pd.256" => "__builtin_ia32_rcp14pd256_mask",
        "llvm.x86.avx512.rcp14.pd.128" => "__builtin_ia32_rcp14pd128_mask",
        "llvm.x86.avx512.rsqrt14.ps.256" => "__builtin_ia32_rsqrt14ps256_mask",
        "llvm.x86.avx512.rsqrt14.ps.128" => "__builtin_ia32_rsqrt14ps128_mask",
        "llvm.x86.avx512.rsqrt14.pd.256" => "__builtin_ia32_rsqrt14pd256_mask",
        "llvm.x86.avx512.rsqrt14.pd.128" => "__builtin_ia32_rsqrt14pd128_mask",
        "llvm.x86.avx512.mask.getexp.ps.512" => "__builtin_ia32_getexpps512_mask",
        "llvm.x86.avx512.mask.getexp.ps.256" => "__builtin_ia32_getexpps256_mask",
        "llvm.x86.avx512.mask.getexp.ps.128" => "__builtin_ia32_getexpps128_mask",
        "llvm.x86.avx512.mask.getexp.pd.512" => "__builtin_ia32_getexppd512_mask",
        "llvm.x86.avx512.mask.getexp.pd.256" => "__builtin_ia32_getexppd256_mask",
        "llvm.x86.avx512.mask.getexp.pd.128" => "__builtin_ia32_getexppd128_mask",
        "llvm.x86.avx512.mask.rndscale.ps.256" => "__builtin_ia32_rndscaleps_256_mask",
        "llvm.x86.avx512.mask.rndscale.ps.128" => "__builtin_ia32_rndscaleps_128_mask",
        "llvm.x86.avx512.mask.rndscale.pd.256" => "__builtin_ia32_rndscalepd_256_mask",
        "llvm.x86.avx512.mask.rndscale.pd.128" => "__builtin_ia32_rndscalepd_128_mask",
        "llvm.x86.avx512.mask.scalef.ps.512" => "__builtin_ia32_scalefps512_mask",
        "llvm.x86.avx512.mask.scalef.ps.256" => "__builtin_ia32_scalefps256_mask",
        "llvm.x86.avx512.mask.scalef.ps.128" => "__builtin_ia32_scalefps128_mask",

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
