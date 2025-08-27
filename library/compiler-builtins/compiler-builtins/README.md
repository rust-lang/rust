# `compiler-builtins`

This crate provides external symbols that the compiler expects to be available
when building Rust projects, typically software routines for basic operations
that do not have hardware support. It is largely a port of LLVM's
[`compiler-rt`].

It is distributed as part of Rust's sysroot. `compiler-builtins` does not need
to be added as an explicit dependency in `Cargo.toml`.

[`compiler-rt`]: https://github.com/llvm/llvm-project/tree/1b1dc505057322f4fa1110ef4f53c44347f52986/compiler-rt

## Configuration

`compiler-builtins` can be configured with the following environment variables when the `c` feature
is enabled:

- `LLVM_COMPILER_RT_LIB`
- `RUST_COMPILER_RT_ROOT`

See `build.rs` for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Progress

- [x] aarch64/chkstk.S
- [x] adddf3.c
- [x] addsf3.c
- [x] arm/addsf3.S
- [x] arm/aeabi_dcmp.S
- [x] arm/aeabi_fcmp.S
- [x] arm/aeabi_idivmod.S
- [x] arm/aeabi_ldivmod.S
- [x] arm/aeabi_memcpy.S
- [x] arm/aeabi_memmove.S
- [x] arm/aeabi_memset.S
- [x] arm/aeabi_uidivmod.S
- [x] arm/aeabi_uldivmod.S
- [ ] arm/chkstk.S
- [ ] arm/divmodsi4.S (generic version is done)
- [ ] arm/divsi3.S (generic version is done)
- [ ] arm/modsi3.S (generic version is done)
- [x] arm/softfloat-alias.list
- [ ] arm/udivmodsi4.S (generic version is done)
- [ ] arm/udivsi3.S (generic version is done)
- [ ] arm/umodsi3.S (generic version is done)
- [x] ashldi3.c
- [x] ashrdi3.c
- [ ] avr/divmodhi4.S
- [ ] avr/divmodqi4.S
- [ ] avr/mulhi3.S
- [ ] avr/mulqi3.S
- [ ] avr/udivmodhi4.S
- [ ] avr/udivmodqi4.S
- [x] bswapdi2.c
- [x] bswapsi2.c
- [x] bswapti2.c
- [x] clzdi2.c
- [x] clzsi2.c
- [x] clzti2.c
- [x] comparedf2.c
- [x] comparesf2.c
- [x] ctzdi2.c
- [x] ctzsi2.c
- [x] ctzti2.c
- [x] divdf3.c
- [x] divdi3.c
- [x] divmoddi4.c
- [x] divmodsi4.c
- [x] divmodti4.c
- [x] divsf3.c
- [x] divsi3.c
- [x] extendsfdf2.c
- [x] fixdfdi.c
- [x] fixdfsi.c
- [x] fixsfdi.c
- [x] fixsfsi.c
- [x] fixunsdfdi.c
- [x] fixunsdfsi.c
- [x] fixunssfdi.c
- [x] fixunssfsi.c
- [x] floatdidf.c
- [x] floatdisf.c
- [x] floatsidf.c
- [x] floatsisf.c
- [x] floatundidf.c
- [x] floatundisf.c
- [x] floatunsidf.c
- [x] floatunsisf.c
- [ ] i386/ashldi3.S
- [ ] i386/ashrdi3.S
- [x] i386/chkstk.S
- [ ] i386/divdi3.S
- [ ] i386/lshrdi3.S
- [ ] i386/moddi3.S
- [ ] i386/muldi3.S
- [ ] i386/udivdi3.S
- [ ] i386/umoddi3.S
- [x] lshrdi3.c
- [x] moddi3.c
- [x] modsi3.c
- [x] muldf3.c
- [x] muldi3.c
- [x] mulodi4.c
- [x] mulosi4.c
- [x] mulsf3.c
- [x] powidf2.c
- [x] powisf2.c
- [ ] riscv/muldi3.S
- [ ] riscv/mulsi3.S
- [x] subdf3.c
- [x] subsf3.c
- [x] truncdfsf2.c
- [x] udivdi3.c
- [x] udivmoddi4.c
- [x] udivmodsi4.c
- [x] udivsi3.c
- [x] umoddi3.c
- [x] umodsi3.c
- [x] x86_64/chkstk.S

These builtins are needed to support 128-bit integers.

- [x] ashlti3.c
- [x] ashrti3.c
- [x] divti3.c
- [x] fixdfti.c
- [x] fixsfti.c
- [x] fixunsdfti.c
- [x] fixunssfti.c
- [x] floattidf.c
- [x] floattisf.c
- [x] floatuntidf.c
- [x] floatuntisf.c
- [x] lshrti3.c
- [x] modti3.c
- [x] muloti4.c
- [x] multi3.c
- [x] udivmodti4.c
- [x] udivti3.c
- [x] umodti3.c

These builtins are needed to support `f16` and `f128`, which are in the process
of being added to Rust.

- [x] addtf3.c
- [x] comparetf2.c
- [x] divtf3.c
- [x] extenddftf2.c
- [x] extendhfsf2.c
- [x] extendhftf2.c
- [x] extendsftf2.c
- [x] fixtfdi.c
- [x] fixtfsi.c
- [x] fixtfti.c
- [x] fixunstfdi.c
- [x] fixunstfsi.c
- [x] fixunstfti.c
- [x] floatditf.c
- [x] floatsitf.c
- [x] floattitf.c
- [x] floatunditf.c
- [x] floatunsitf.c
- [x] floatuntitf.c
- [x] multf3.c
- [x] powitf2.c
- [x] subtf3.c
- [x] truncdfhf2.c
- [x] truncsfhf2.c
- [x] trunctfdf2.c
- [x] trunctfhf2.c
- [x] trunctfsf2.c


These builtins are used by the Hexagon DSP

- [ ] hexagon/common_entry_exit_abi1.S
- [ ] hexagon/common_entry_exit_abi2.S
- [ ] hexagon/common_entry_exit_legacy.S
- [x] hexagon/dfaddsub.S~~
- [x] hexagon/dfdiv.S~~
- [x] hexagon/dffma.S~~
- [x] hexagon/dfminmax.S~~
- [x] hexagon/dfmul.S~~
- [x] hexagon/dfsqrt.S~~
- [x] hexagon/divdi3.S~~
- [x] hexagon/divsi3.S~~
- [x] hexagon/fastmath2_dlib_asm.S~~
- [x] hexagon/fastmath2_ldlib_asm.S~~
- [x] hexagon/fastmath_dlib_asm.S~~
- [x] hexagon/memcpy_forward_vp4cp4n2.S~~
- [x] hexagon/memcpy_likely_aligned.S~~
- [x] hexagon/moddi3.S~~
- [x] hexagon/modsi3.S~~
- [x] hexagon/sfdiv_opt.S~~
- [x] hexagon/sfsqrt_opt.S~~
- [x] hexagon/udivdi3.S~~
- [x] hexagon/udivmoddi4.S~~
- [x] hexagon/udivmodsi4.S~~
- [x] hexagon/udivsi3.S~~
- [x] hexagon/umoddi3.S~~
- [x] hexagon/umodsi3.S~~

## Unimplemented functions

These builtins are for x87 `f80` floating-point numbers that are not supported
by Rust.

- ~~extendxftf2.c~~
- ~~fixunsxfdi.c~~
- ~~fixunsxfsi.c~~
- ~~fixunsxfti.c~~
- ~~fixxfdi.c~~
- ~~fixxfti.c~~
- ~~floatdixf.c~~
- ~~floattixf.c~~
- ~~floatundixf.c~~
- ~~floatuntixf.c~~
- ~~i386/floatdixf.S~~
- ~~i386/floatundixf.S~~
- ~~x86_64/floatdixf.c~~
- ~~x86_64/floatundixf.S~~

These builtins are for IBM "extended double" non-IEEE 128-bit floating-point
numbers.

- ~~ppc/divtc3.c~~
- ~~ppc/fixtfdi.c~~
- ~~ppc/fixtfti.c~~
- ~~ppc/fixunstfdi.c~~
- ~~ppc/fixunstfti.c~~
- ~~ppc/floatditf.c~~
- ~~ppc/floattitf.c~~
- ~~ppc/floatunditf.c~~
- ~~ppc/gcc_qadd.c~~
- ~~ppc/gcc_qdiv.c~~
- ~~ppc/gcc_qmul.c~~
- ~~ppc/gcc_qsub.c~~
- ~~ppc/multc3.c~~

These builtins are for 16-bit brain floating-point numbers that are not
supported by Rust.

- ~~truncdfbf2.c~~
- ~~truncsfbf2.c~~
- ~~trunctfxf2.c~~

These builtins involve complex floating-point types that are not supported by
Rust.

- ~~divdc3.c~~
- ~~divsc3.c~~
- ~~divtc3.c~~
- ~~divxc3.c~~
- ~~muldc3.c~~
- ~~mulsc3.c~~
- ~~multc3.c~~
- ~~mulxc3.c~~
- ~~powixf2.c~~

These builtins are never called by LLVM.

- ~~absvdi2.c~~
- ~~absvsi2.c~~
- ~~absvti2.c~~
- ~~addvdi3.c~~
- ~~addvsi3.c~~
- ~~addvti3.c~~
- ~~arm/aeabi_cdcmp.S~~
- ~~arm/aeabi_cdcmpeq_check_nan.c~~
- ~~arm/aeabi_cfcmp.S~~
- ~~arm/aeabi_cfcmpeq_check_nan.c~~
- ~~arm/aeabi_div0.c~~
- ~~arm/aeabi_drsub.c~~
- ~~arm/aeabi_frsub.c~~
- ~~arm/aeabi_memcmp.S~~
- ~~arm/bswapdi2.S~~
- ~~arm/bswapsi2.S~~
- ~~arm/clzdi2.S~~
- ~~arm/clzsi2.S~~
- ~~arm/comparesf2.S~~
- ~~arm/restore_vfp_d8_d15_regs.S~~
- ~~arm/save_vfp_d8_d15_regs.S~~
- ~~arm/switch16.S~~
- ~~arm/switch32.S~~
- ~~arm/switch8.S~~
- ~~arm/switchu8.S~~
- ~~cmpdi2.c~~
- ~~cmpti2.c~~
- ~~ffssi2.c~~
- ~~ffsdi2.c~~ - this is [called by gcc][jemalloc-fail] though!
- ~~ffsti2.c~~
- ~~mulvdi3.c~~
- ~~mulvsi3.c~~
- ~~mulvti3.c~~
- ~~negdf2.c~~
- ~~negdi2.c~~
- ~~negsf2.c~~
- ~~negti2.c~~
- ~~negvdi2.c~~
- ~~negvsi2.c~~
- ~~negvti2.c~~
- ~~paritydi2.c~~
- ~~paritysi2.c~~
- ~~parityti2.c~~
- ~~popcountdi2.c~~
- ~~popcountsi2.c~~
- ~~popcountti2.c~~
- ~~ppc/restFP.S~~
- ~~ppc/saveFP.S~~
- ~~subvdi3.c~~
- ~~subvsi3.c~~
- ~~subvti3.c~~
- ~~ucmpdi2.c~~
- ~~ucmpti2.c~~
- ~~udivmodti4.c~~

[jemalloc-fail]: https://travis-ci.org/rust-lang/rust/jobs/249772758

Rust only exposes atomic types on platforms that support them, and therefore does not need to fall back to software implementations.

- ~~arm/sync_fetch_and_add_4.S~~
- ~~arm/sync_fetch_and_add_8.S~~
- ~~arm/sync_fetch_and_and_4.S~~
- ~~arm/sync_fetch_and_and_8.S~~
- ~~arm/sync_fetch_and_max_4.S~~
- ~~arm/sync_fetch_and_max_8.S~~
- ~~arm/sync_fetch_and_min_4.S~~
- ~~arm/sync_fetch_and_min_8.S~~
- ~~arm/sync_fetch_and_nand_4.S~~
- ~~arm/sync_fetch_and_nand_8.S~~
- ~~arm/sync_fetch_and_or_4.S~~
- ~~arm/sync_fetch_and_or_8.S~~
- ~~arm/sync_fetch_and_sub_4.S~~
- ~~arm/sync_fetch_and_sub_8.S~~
- ~~arm/sync_fetch_and_umax_4.S~~
- ~~arm/sync_fetch_and_umax_8.S~~
- ~~arm/sync_fetch_and_umin_4.S~~
- ~~arm/sync_fetch_and_umin_8.S~~
- ~~arm/sync_fetch_and_xor_4.S~~
- ~~arm/sync_fetch_and_xor_8.S~~
- ~~arm/sync_synchronize.S~~
- ~~atomic.c~~
- ~~atomic_flag_clear.c~~
- ~~atomic_flag_clear_explicit.c~~
- ~~atomic_flag_test_and_set.c~~
- ~~atomic_flag_test_and_set_explicit.c~~
- ~~atomic_signal_fence.c~~
- ~~atomic_thread_fence.c~~

Miscellaneous functionality that is not used by Rust.

- ~~aarch64/fp_mode.c~~
- ~~aarch64/lse.S~~ (LSE atomics)
- ~~aarch64/sme-abi-init.c~~ (matrix extension)
- ~~aarch64/sme-abi.S~~ (matrix extension)
- ~~aarch64/sme-libc-routines.c~~ (matrix extension)
- ~~apple_versioning.c~~
- ~~arm/fp_mode.c~~
- ~~avr/exit.S~~
- ~~clear_cache.c~~
- ~~cpu_model/aarch64.c~~
- ~~cpu_model/x86.c~~
- ~~crtbegin.c~~
- ~~crtend.c~~
- ~~emutls.c~~
- ~~enable_execute_stack.c~~
- ~~eprintf.c~~
- ~~fp_mode.c~~ (float exception handling)
- ~~gcc_personality_v0.c~~
- ~~i386/fp_mode.c~~
- ~~int_util.c~~
- ~~loongarch/fp_mode.c~~
- ~~os_version_check.c~~
- ~~riscv/fp_mode.c~~
- ~~riscv/restore.S~~ (callee-saved registers)
- ~~riscv/save.S~~ (callee-saved registers)
- ~~trampoline_setup.c~~
- ~~ve/grow_stack.S~~
- ~~ve/grow_stack_align.S~~

Floating-point implementations of builtins that are only called from soft-float code. It would be better to simply use the generic soft-float versions in this case.

- ~~i386/floatdidf.S~~
- ~~i386/floatdisf.S~~
- ~~i386/floatundidf.S~~
- ~~i386/floatundisf.S~~
- ~~x86_64/floatundidf.S~~
- ~~x86_64/floatundisf.S~~
- ~~x86_64/floatdidf.c~~
- ~~x86_64/floatdisf.c~~

Unsupported in any current target: used on old versions of 32-bit iOS with ARMv5.

- ~~arm/adddf3vfp.S~~
- ~~arm/addsf3vfp.S~~
- ~~arm/divdf3vfp.S~~
- ~~arm/divsf3vfp.S~~
- ~~arm/eqdf2vfp.S~~
- ~~arm/eqsf2vfp.S~~
- ~~arm/extendsfdf2vfp.S~~
- ~~arm/fixdfsivfp.S~~
- ~~arm/fixsfsivfp.S~~
- ~~arm/fixunsdfsivfp.S~~
- ~~arm/fixunssfsivfp.S~~
- ~~arm/floatsidfvfp.S~~
- ~~arm/floatsisfvfp.S~~
- ~~arm/floatunssidfvfp.S~~
- ~~arm/floatunssisfvfp.S~~
- ~~arm/gedf2vfp.S~~
- ~~arm/gesf2vfp.S~~
- ~~arm/gtdf2vfp.S~~
- ~~arm/gtsf2vfp.S~~
- ~~arm/ledf2vfp.S~~
- ~~arm/lesf2vfp.S~~
- ~~arm/ltdf2vfp.S~~
- ~~arm/ltsf2vfp.S~~
- ~~arm/muldf3vfp.S~~
- ~~arm/mulsf3vfp.S~~
- ~~arm/nedf2vfp.S~~
- ~~arm/negdf2vfp.S~~
- ~~arm/negsf2vfp.S~~
- ~~arm/nesf2vfp.S~~
- ~~arm/subdf3vfp.S~~
- ~~arm/subsf3vfp.S~~
- ~~arm/truncdfsf2vfp.S~~
- ~~arm/unorddf2vfp.S~~
- ~~arm/unordsf2vfp.S~~

## License

Usage is allowed under the [MIT License] and the [Apache License, Version 2.0]
with the LLVM exception.

[MIT License]: https://opensource.org/license/mit
[Apache License, Version 2.0]: htps://www.apache.org/licenses/LICENSE-2.0

### Contribution

Contributions are licensed under the MIT License, the Apache License,
Version 2.0, and the Apache-2.0 license with the LLVM exception.

See [LICENSE.txt](../LICENSE.txt) for full details.
