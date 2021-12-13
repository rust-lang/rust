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

``` toml
# Cargo.toml
[dependencies]
compiler_builtins = { git = "https://github.com/rust-lang/compiler-builtins" }
```

``` rust
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
4. Implement a [test generator][3] to compare the behavior of the ported intrinsic(s)
   with their implementation on the testing host. Note that randomized compiler-builtin tests
   should be run using `cargo test --features gen-tests`.
4. Send a Pull Request (PR).
5. Once the PR passes our extensive [testing infrastructure][4], we'll merge it!
6. Celebrate :tada:

[1]: https://github.com/rust-lang/compiler-rt/tree/8598065bd965d9713bfafb6c1e766d63a7b17b89/test/builtins/Unit
[2]: https://github.com/rust-lang/compiler-rt/tree/8598065bd965d9713bfafb6c1e766d63a7b17b89/lib/builtins
[3]: https://github.com/rust-lang/compiler-builtins/blob/0ba07e49264a54cb5bbd4856fcea083bb3fbec15/build.rs#L180-L265
[4]: https://travis-ci.org/rust-lang/compiler-builtins

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

## Progress

- [x] adddf3.c
- [x] addsf3.c
- [x] arm/adddf3vfp.S
- [x] arm/addsf3vfp.S
- [x] arm/aeabi_dcmp.S
- [x] arm/aeabi_fcmp.S
- [x] arm/aeabi_idivmod.S
- [x] arm/aeabi_ldivmod.S
- [x] arm/aeabi_memcpy.S
- [x] arm/aeabi_memmove.S
- [x] arm/aeabi_memset.S
- [x] arm/aeabi_uidivmod.S
- [x] arm/aeabi_uldivmod.S
- [x] arm/divdf3vfp.S
- [ ] arm/divmodsi4.S (generic version is done)
- [x] arm/divsf3vfp.S
- [ ] arm/divsi3.S (generic version is done)
- [x] arm/eqdf2vfp.S
- [x] arm/eqsf2vfp.S
- [x] arm/extendsfdf2vfp.S
- [ ] arm/fixdfsivfp.S
- [ ] arm/fixsfsivfp.S
- [ ] arm/fixunsdfsivfp.S
- [ ] arm/fixunssfsivfp.S
- [ ] arm/floatsidfvfp.S
- [ ] arm/floatsisfvfp.S
- [ ] arm/floatunssidfvfp.S
- [ ] arm/floatunssisfvfp.S
- [x] arm/gedf2vfp.S
- [x] arm/gesf2vfp.S
- [x] arm/gtdf2vfp.S
- [x] arm/gtsf2vfp.S
- [x] arm/ledf2vfp.S
- [x] arm/lesf2vfp.S
- [x] arm/ltdf2vfp.S
- [x] arm/ltsf2vfp.S
- [ ] arm/modsi3.S (generic version is done)
- [x] arm/muldf3vfp.S
- [x] arm/mulsf3vfp.S
- [x] arm/nedf2vfp.S
- [ ] arm/negdf2vfp.S
- [ ] arm/negsf2vfp.S
- [x] arm/nesf2vfp.S
- [x] arm/softfloat-alias.list
- [x] arm/subdf3vfp.S
- [x] arm/subsf3vfp.S
- [x] arm/truncdfsf2vfp.S
- [ ] arm/udivmodsi4.S (generic version is done)
- [ ] arm/udivsi3.S (generic version is done)
- [ ] arm/umodsi3.S (generic version is done)
- [ ] arm/unorddf2vfp.S
- [ ] arm/unordsf2vfp.S
- [x] ashldi3.c
- [x] ashrdi3.c
- [x] comparedf2.c
- [x] comparesf2.c
- [x] divdf3.c
- [x] divdi3.c
- [x] divmoddi4.c
- [x] divmodsi4.c
- [x] divsf3.c
- [x] divsi3.c
- [ ] extendhfsf2.c
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
- [x] i386/chkstk2.S
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
- [x] subdf3.c
- [x] subsf3.c
- [ ] truncdfhf2.c
- [x] truncdfsf2.c
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
- ~~ctzdi2.c~~
- ~~ctzsi2.c~~
- ~~ctzti2.c~~
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

The compiler-builtins crate is dual licensed under both the University of
Illinois "BSD-Like" license and the MIT license.  As a user of this code you may
choose to use it under either license.  As a contributor, you agree to allow
your code to be used under both.

Full text of the relevant licenses is in LICENSE.TXT.
