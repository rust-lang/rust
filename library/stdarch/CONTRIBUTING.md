This document is intended to be a guide for documenting the process of adding
new vendor intrinsics to this crate.

If you decide to implement a set of vendor intrinsics, please check the set of
open issues to make sure somebody else isn't already working on them. If no
such issue exists, then create a new issue and state the intrinsics you'd like
to implement.

At a high level, each vendor intrinsic should correspond to a single exported
Rust function with an appropriate `target_feature` attribute. Here's an
example for `_mm_adds_epi16`:

```rust
/// Add packed 16-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(paddsw))]
pub fn _mm_adds_epi16(a: i16x8, b: i16x8) -> i16x8 {
    unsafe { paddsw(a, b) }
}
```

Let's break this down:

* The `#[inline(always)]` is added because vendor intrinsic functions generally
  should always be inlined because the intent of a vendor intrinsic is to
  correspond to a single particular CPU instruction. A vendor intrinsic that
  is compiled into an actual function call could be quite disastrous for
  performance.
* The `#[target_feature = "+sse2"]` attribute intructs the compiler to generate
  code with the `sse2` target feature enabled, *regardless* of the target
  platform. That is, even if you're compiling for a platform that doesn't
  support `sse2`, the compiler will still generate code for `_mm_adds_epi16`
  *as if* `sse2` support existed. Without this attribute, the compiler might
  not generate the intended CPU instruction.
* The `#[cfg_attr(test, assert_instr(paddsw))]` attribute indicates that when
  we're testing the crate we'll assert that the `paddsw` instruction is
  generated inside this function, ensuring that the SIMD intrinsic truly is an
  intrinsic for the instruction!
* The types of the vectors given to the intrinsic should generally match the
  types as provided in the vendor interface. We'll talk about this more below.
* The implementation of the vendor intrinsic is generally very simple.
  Remember, the goal is to compile a call to `_mm_adds_epi16` down to a single
  particular CPU instruction. As such, the implementation typically defers to a
  compiler intrinsic (in this case, `paddsw`) when one is available. More on
  this below as well.

Once a function has been added, you should also add at least one test for basic
functionality. Here's an example for `_mm_adds_epi16`:

```rust
#[test]
fn _mm_adds_epi16() {
    let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
    let b = i16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
    let r = sse2::_mm_adds_epi16(a, b);
    let e = i16x8::new(8, 10, 12, 14, 16, 18, 20, 22);
    assert_eq!(r, e);
}
```

Finally, once that's done, check off the corresponding box in the `TODO.md`
file in the root of this repository.


## Determining types

Determining the function signature of each vendor intrinsic can be tricky
depending on the specificity of the vendor API. For SSE, Intel generally has
three types in their interface:

* `__m128` consists of 4 single-precision (32-bit) floating point numbers.
* `__m128d` consists of 2 double-precision (64-bit) floating point numbers.
* `__m128i` consists of `N` integers, where `N` can be 16, 8, 4 or 2. The
  corresponding bit sizes for each value of `N` are 8-bit, 16-bit, 32-bit and
  64-bit, respectively. Finally, there are signed and unsigned variants for
  each value of `N`, which means `__m128i` can be mapped to one of eight
  possible concrete integer types.

In terms of the `stdsimd` crate, the first two floating point types have a
straight-forward translation. `__m128` maps to `f32x4` while `__m128d` maps to
`f64x2`.

Unfortunately, since `__m128i` can correspond to any number of integer types
we need to actually inspect the vendor intrinsic to determine the type.
Sometimes this is hinted at in the name of intrinsic itself. Continuing with
our previous example, `_mm_adds_epi16`, we can infer that it is a signed
operation on an integer vector consisting of eight 16-bit integers. Namely, the
`epi` means signed (where as `epu` means unsigned) and `16` means 16-bit.

Fortunately, Clang (and LLVM) have determined the specific concrete integer
types for most of the vendor intrinsics already, but they aren't available in
any easily access away (as far as this author knows). For example, you can
see
[the types for `_mm_adds_epi16` in Clang's `emmintrin.h` header file](https://github.com/llvm-mirror/clang/blob/dcd8d797b20291f1a6b3e0ddda085aa2bbb382a8/lib/Headers/emmintrin.h#L2180-L2200).


## Writing the implementation

An implementation of an intrinsic (so far) generally has one of three shapes:

1. The vendor intrinsic does not have any corresponding compiler intrinsic, so
   you must write the implementation in such a way that the compiler will
   recognize it and produce the desired codegen. For example, the
   `_mm_add_epi16` intrinsic (note the missing `s` in `add`) is implemented
   via `a + b`, which compiles down to LLVM's cross platform SIMD vector API.
2. The vendor intrinsic *does* have a corresponding compiler intrinsic, so you
   must write an `extern` block to bring that intrinsic into scope and then
   call it. The example above (`_mm_adds_epi16`) uses this approach.
3. The vendor intrinsic has a parameter that must be a *constant* value when
   given to the CPU instruction, where that constant is often a parameter that
   impacts the operation of the intrinsic. This means the implementation of the
   vendor intrinsic must guarantee that a particular parameter be a constant.
   This is tricky because Rust doesn't (yet) have a stable way of doing this,
   so we have to do it ourselves. How you do it can vary, but one particularly
   gnarly example is
   [`_mm_cmpestri`](https://github.com/BurntSushi/stdsimd/blob/ff6021b72e8cc1e7db942847d99278fe0056c245/src/x86/sse42.rs#L286)
   (make sure to look at the `constify_imm8!` macro).


## References

The compiler intrinsics available to us through LLVM can be found here:
https://gist.github.com/anonymous/a25d3e3b4c14ee68d63bd1dcb0e1223c

The Intel vendor intrinsic API can be found here:
https://gist.github.com/anonymous/25d752fda8521d29699a826b980218fc

The Clang header files for vendor intrinsics can also be incredibly useful.
When in doubt, Do What Clang Does:
https://github.com/llvm-mirror/clang/tree/master/lib/Headers
