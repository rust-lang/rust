# `compiler-builtins`

> Porting `compiler-rt` intrinsics to Rust

See [rust-lang/rust#35437][0].

[0]: https://github.com/rust-lang/rust/issues/35437

## When and how to use this crate?

If you are working with a target that doesn't have binary releases of std
available via rustup (this probably means you are building the core crate
yourself) and need compiler-rt intrinsics (i.e. you are probably getting linker
errors when building an executable: `undefined reference to __aeabi_memcpy`),
you can use this crate to get those intrinsics and solve the linker errors. To
do that, add this crate somewhere in the dependency graph of the crate you are
building:

```toml
# Cargo.toml
[dependencies]
compiler_builtins = { git = "https://github.com/rust-lang/compiler-builtins" }
```

```rust
extern crate compiler_builtins;

// ...
```

If you still get an "undefined reference to $INTRINSIC" error after that change,
that means that we haven't ported `$INTRINSIC` to Rust yet! Please open [an
issue] with the name of the intrinsic and the LLVM triple (e.g.
thumbv7m-none-eabi) of the target you are using. That way we can prioritize
porting that particular intrinsic.

If you've got a C compiler available for your target then while we implement
this intrinsic you can temporarily enable a fallback to the actual compiler-rt
implementation as well for unimplemented intrinsics:

```toml
[dependencies.compiler_builtins]
git = "https://github.com/rust-lang/compiler-builtins"
features = ["c"]
```

[an issue]: https://github.com/rust-lang/compiler-builtins/issues

## Contributing

1. Pick one or more intrinsics from the [pending list](#progress).
2. Fork this repository.
3. Port the intrinsic(s) and their corresponding [unit tests][1] from their
   [C implementation][2] to Rust.
4. Add a test to compare the behavior of the ported intrinsic(s) with their
   implementation on the testing host.
5. Add the intrinsic to `examples/intrinsics.rs` to verify it can be linked on
   all targets.
6. Send a Pull Request (PR).
7. Once the PR passes our extensive testing infrastructure, we'll merge it!
8. Celebrate :tada:

[1]: https://github.com/rust-lang/llvm-project/tree/9e3de9490ff580cd484fbfa2908292b4838d56e7/compiler-rt/test/builtins/Unit
[2]: https://github.com/rust-lang/llvm-project/tree/9e3de9490ff580cd484fbfa2908292b4838d56e7/compiler-rt/lib/builtins
[3]: https://github.com/rust-lang/compiler-builtins/actions

### Porting Reminders

1. [Rust][5a] and [C][5b] have slightly different operator precedence. C evaluates comparisons (`== !=`) before bitwise operations (`& | ^`), while Rust evaluates the other way.
2. C assumes wrapping operations everywhere. Rust panics on overflow when in debug mode. Consider using the [Wrapping][6] type or the explicit [wrapping_*][7] functions where applicable.
3. Note [C implicit casts][8], especially integer promotion. Rust is much more explicit about casting, so be sure that any cast which affects the output is ported to the Rust implementation.
4. Rust has [many functions][9] for integer or floating point manipulation in the standard library. Consider using one of these functions rather than porting a new one.

[5a]: https://doc.rust-lang.org/reference/expressions.html#expression-precedence
[5b]: http://en.cppreference.com/w/c/language/operator_precedence
[6]: https://doc.rust-lang.org/core/num/struct.Wrapping.html
[7]: https://doc.rust-lang.org/std/primitive.i32.html#method.wrapping_add
[8]: http://en.cppreference.com/w/cpp/language/implicit_conversion
[9]: https://doc.rust-lang.org/std/primitive.i32.html

## Testing

The easiest way to test locally is using Docker. This can be done by running
`./ci/run-docker.sh [target]`. If no target is specified, all targets will be
run.

In order to run the full test suite, you will also need the C compiler runtime
to test against, located in a directory called `compiler-rt`. This can be
obtained with the following:

```sh
curl -L -o rustc-llvm-19.1.tar.gz https://github.com/rust-lang/llvm-project/archive/rustc/19.1-2024-09-17.tar.gz
tar xzf rustc-llvm-19.1.tar.gz --strip-components 1 llvm-project-rustc-19.1-2024-09-17/compiler-rt
```

Local targets may also be tested with `./ci/run.sh [target]`.

Note that testing may not work on all hosts, in which cases it is acceptable to
rely on CI.

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

The compiler-builtins crate is dual licensed under both the University of
Illinois "BSD-Like" license and the MIT license.  As a user of this code you may
choose to use it under either license.  As a contributor, you agree to allow
your code to be used under both.

Full text of the relevant licenses is in LICENSE.TXT.
