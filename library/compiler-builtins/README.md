[![Build Status][status]](https://travis-ci.org/japaric/rustc-builtins)

[status]: https://travis-ci.org/japaric/rustc-builtins.svg?branch=master

# `rustc-builtins`

> [WIP] Porting `compiler-rt` intrinsics to Rust

See [rust-lang/rust#35437][0].

[0]: https://github.com/rust-lang/rust/issues/35437

## Contributing

1. Pick one or more intrinsics from the [pending list][/#progress].
2. Fork this repository
3. Port the intrinsic(s) and their corresponding [unit tests][1] from their [C implementation][2] to
   Rust.
4. Send a Pull Request (PR)
5. Once the PR passes our extensive [testing infrastructure][3], we'll merge it!
6. Celebrate :tada:

[1]: https://github.com/rust-lang/compiler-rt/tree/8598065bd965d9713bfafb6c1e766d63a7b17b89/test/builtins/Unit
[2]: https://github.com/rust-lang/compiler-rt/tree/8598065bd965d9713bfafb6c1e766d63a7b17b89/lib/builtins
[3]: https://travis-ci.org/japaric/rustc-builtins

## Progress

- [ ] adddf3.c
- [ ] addsf3.c
- [ ] arm/adddf3vfp.S
- [ ] arm/addsf3vfp.S
- [ ] arm/aeabi_dcmp.S
- [ ] arm/aeabi_fcmp.S
- [x] arm/aeabi_idivmod.S
- [x] arm/aeabi_ldivmod.S
- [x] arm/aeabi_memcpy.S
- [x] arm/aeabi_memmove.S
- [x] arm/aeabi_memset.S
- [x] arm/aeabi_uidivmod.S
- [x] arm/aeabi_uldivmod.S
- [ ] arm/divdf3vfp.S
- [ ] arm/divmodsi4.S
- [ ] arm/divsf3vfp.S
- [ ] arm/divsi3.S
- [ ] arm/eqdf2vfp.S
- [ ] arm/eqsf2vfp.S
- [ ] arm/extendsfdf2vfp.S
- [ ] arm/fixdfsivfp.S
- [ ] arm/fixsfsivfp.S
- [ ] arm/fixunsdfsivfp.S
- [ ] arm/fixunssfsivfp.S
- [ ] arm/floatsidfvfp.S
- [ ] arm/floatsisfvfp.S
- [ ] arm/floatunssidfvfp.S
- [ ] arm/floatunssisfvfp.S
- [ ] arm/gedf2vfp.S
- [ ] arm/gesf2vfp.S
- [ ] arm/gtdf2vfp.S
- [ ] arm/gtsf2vfp.S
- [ ] arm/ledf2vfp.S
- [ ] arm/lesf2vfp.S
- [ ] arm/ltdf2vfp.S
- [ ] arm/ltsf2vfp.S
- [ ] arm/modsi3.S
- [ ] arm/muldf3vfp.S
- [ ] arm/mulsf3vfp.S
- [ ] arm/nedf2vfp.S
- [ ] arm/negdf2vfp.S
- [ ] arm/negsf2vfp.S
- [ ] arm/nesf2vfp.S
- [ ] arm/softfloat-alias.list
- [ ] arm/subdf3vfp.S
- [ ] arm/subsf3vfp.S
- [ ] arm/truncdfsf2vfp.S
- [ ] arm/udivmodsi4.S
- [ ] arm/udivsi3.S
- [ ] arm/umodsi3.S
- [ ] arm/unorddf2vfp.S
- [ ] arm/unordsf2vfp.S
- [x] ashldi3.c
- [x] ashrdi3.c
- [ ] divdf3.c
- [ ] divdi3.c
- [ ] divsf3.c
- [ ] divsi3.c
- [ ] extendhfsf2.c
- [ ] extendsfdf2.c
- [ ] fixdfdi.c
- [ ] fixdfsi.c
- [ ] fixsfdi.c
- [ ] fixsfsi.c
- [ ] fixunsdfdi.c
- [ ] fixunsdfsi.c
- [ ] fixunssfdi.c
- [ ] fixunssfsi.c
- [ ] floatdidf.c
- [ ] floatdisf.c
- [ ] floatsidf.c
- [ ] floatsisf.c
- [ ] floatundidf.c
- [ ] floatundisf.c
- [ ] floatunsidf.c
- [ ] floatunsisf.c
- [ ] i386/ashldi3.S
- [ ] i386/ashrdi3.S
- [ ] i386/chkstk.S
- [ ] i386/chkstk2.S
- [ ] i386/divdi3.S
- [ ] i386/lshrdi3.S
- [ ] i386/moddi3.S
- [ ] i386/muldi3.S
- [ ] i386/udivdi3.S
- [ ] i386/umoddi3.S
- [x] lshrdi3.c
- [ ] moddi3.c
- [ ] modsi3.c
- [ ] muldf3.c
- [x] muldi3.c
- [x] mulodi4.c
- [x] mulosi4.c
- [ ] mulsf3.c
- [ ] powidf2.c
- [ ] powisf2.c
- [ ] subdf3.c
- [ ] subsf3.c
- [ ] truncdfhf2.c
- [ ] truncdfsf2.c
- [ ] truncsfhf2.c
- [x] udivdi3.c
- [x] udivmoddi4.c
- [x] udivmodsi4.c
- [x] udivsi3.c
- [x] umoddi3.c
- [x] umodsi3.c
- [x] x86_64/chkstk.S
- [x] x86_64/chkstk2.S

These builtins are needed to support 128-bit integers, which are in the process of being added to Rust.

- [ ] ashlti3.c
- [ ] ashrti3.c
- [ ] divti3.c
- [ ] fixdfti.c
- [ ] fixsfti.c
- [ ] fixunsdfti.c
- [ ] fixunssfti.c
- [ ] floattidf.c
- [ ] floattisf.c
- [ ] floatuntidf.c
- [ ] floatuntisf.c
- [ ] lshrti3.c
- [ ] modti3.c
- [ ] muloti4.c
- [ ] multi3.c
- [ ] udivmodti4.c
- [ ] udivti3.c
- [ ] umodti3.c

## Unimplemented functions

These builtins involve floating-point types ("`f128`", "`f80`" and complex numbers) that are not supported by Rust.

- ~~addtf3.c~~
- ~~comparetf2.c~~
- ~~divdc3.c~~
- ~~divsc3.c~~
- ~~divtc3.c~~
- ~~divtf3.c~~
- ~~divxc3.c~~
- ~~extenddftf2.c~~
- ~~extendsftf2.c~~
- ~~fixtfdi.c~~
- ~~fixtfsi.c~~
- ~~fixtfti.c~~
- ~~fixunstfdi.c~~
- ~~fixunstfsi.c~~
- ~~fixunstfti.c~~
- ~~fixunsxfdi.c~~
- ~~fixunsxfsi.c~~
- ~~fixunsxfti.c~~
- ~~fixxfdi.c~~
- ~~fixxfti.c~~
- ~~floatditf.c~~
- ~~floatdixf.c~~
- ~~floatsitf.c~~
- ~~floattixf.c~~
- ~~floatunditf.c~~
- ~~floatundixf.c~~
- ~~floatunsitf.c~~
- ~~floatuntixf.c~~
- ~~i386/floatdixf.S~~
- ~~i386/floatundixf.S~~
- ~~muldc3.c~~
- ~~mulsc3.c~~
- ~~multc3.c~~
- ~~multf3.c~~
- ~~mulxc3.c~~
- ~~powitf2.c~~
- ~~powixf2.c~~
- ~~ppc/divtc3.c~~
- ~~ppc/fixtfdi.c~~
- ~~ppc/fixunstfdi.c~~
- ~~ppc/floatditf.c~~
- ~~ppc/floatunditf.c~~
- ~~ppc/gcc_qadd.c~~
- ~~ppc/gcc_qdiv.c~~
- ~~ppc/gcc_qmul.c~~
- ~~ppc/gcc_qsub.c~~
- ~~ppc/multc3.c~~
- ~~subtf3.c~~
- ~~trunctfdf2.c~~
- ~~trunctfsf2.c~~
- ~~x86_64/floatdixf.c~~
- ~~x86_64/floatundixf.S~~

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
- ~~clzdi2.c~~
- ~~clzsi2.c~~
- ~~clzti2.c~~
- ~~cmpdi2.c~~
- ~~cmpti2.c~~
- ~~comparedf2.c~~
- ~~comparesf2.c~~
- ~~ctzdi2.c~~
- ~~ctzsi2.c~~
- ~~ctzti2.c~~
- ~~divmoddi4.c~~
- ~~divmodsi4.c~~
- ~~ffsdi2.c~~
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

- ~~apple_versioning.c~~
- ~~clear_cache.c~~
- ~~emutls.c~~
- ~~enable_execute_stack.c~~
- ~~eprintf.c~~
- ~~gcc_personality_v0.c~~
- ~~trampoline_setup.c~~

Floating-point implementations of builtins that are only called from soft-float code. It would be better to simply use the generic soft-float versions in this case.

- ~~i386/floatdidf.S~~
- ~~i386/floatdisf.S~~
- ~~i386/floatundidf.S~~
- ~~i386/floatundisf.S~~
- ~~x86_64/floatundidf.S~~
- ~~x86_64/floatundisf.S~~
- ~~x86_64/floatdidf.c~~
- ~~x86_64/floatdisf.c~~

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
