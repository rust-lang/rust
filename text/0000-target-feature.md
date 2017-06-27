- Feature Name: `target_feature` / `cfg_target_feature`
- Start Date: 2017-06-26
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Motivation and Summary
[summary]: #summary

Some `x86_64` CPUs, among others, support extra "features", like AVX SIMD vector
instructions, that not all `x86_64` CPUs do support. This RFC proposes extending
Rust with:

- **conditional compilation on target features**: choosing different code-paths
  at compile-time depending on the features being used by the compiler,
  
- **unconditional code generation**: unconditionally generating code that uses
  extra features that the host currently being targeted by the compiler does not
  support (allows embedding code optimized for different CPUs within the same
  binary), and
  
- (unresolved) **run-time feature detection**: querying whether a feature is
  supported by the host in which the binary runs.
  
This RFC also specifies the semantics of the backend options `-C
--target-feature/--target-cpu`.

# Detailed design
[design]: #detailed-design

This RFC proposes adding the following three constructs to the language:

- the `cfg!(target_feature = "feature_name")` macro which detects whether the
  current scope is being compiled with a target feature enabled/disabled, 
- the `#[target_feature = "+feature_name"]` attribute for `unsafe` functions
  which allows the compiler to generate the function's code assuming that the
  host in which the function is executed supports the feature, and
- (unresolved) a `std::cpuid` module for detecting at run-time whether a feature
  is supported by the current host.

## Target features

Each rustc target has a default set of target features; this set is
_implementation defined_.

This RFC does not add any target features to the language. It does however
specify the process for adding target features. Each target feature must:

- be proposed in its own mini-RFC, RFC, or rustc-issue and follow a FCP period,
- be behind its own feature macro of the form `target_feature_feature_name`
  (where `feature_name` should be replaced by the name of the feature ).
- (unresolved) when possible, be detectable at run-time via the `std::cpuid`
  module.
  
To use unstable features on nightly, crates must opt into them as usual by
writing, for example, `#![allow(target_feature_avx2)]`. Since this is currently
not required, a grace period of one full release cycle will be given in which
this will raise a soft error before turning this requirement into a hard error.


## Unconditional code generation: `#[target_feature]`

(note: `#[target_feature]` is similar to clang's and
gcc's
[`__attribute__ ((__target__ ("feature")))`](https://clang.llvm.org/docs/AttributeReference.html#target-gnu-target).)


The `unsafe` function attribute [`#[target_feature =
"+feature_name"]`](https://github.com/rust-lang/rust/pull/38079) _extends_ the
feature set of a function. That is, `#[target_feature = "+feature_name"] unsafe
fn foo(...)` allows the compiler to generate code for `foo` under the assumption
that `foo` will only run on binaries that support the feature `feature_name` in
addition to the default feature set of the target, which can be controlled
through the backend options `-C --target-feature/--target-cpu`.

Calling a function on a target that does not support its feature set is
_undefined behavior_; no diagnostic is required. Note: the sub-section "No
segfaults" within the "Alternatives" section discusses how to implement run-time
diagnostics but these incur a run-time cost.

> Note: Calling a function on a host that does not support its features invokes
undefined behavior in LLVM and the assembler, and also in some hardware. For
this reason, `#[target_feature]` cannot apply to safe functions.

Removing features from the default feature set using `#[target_feature =
"-feature_name"]` is illegal; a diagnostic is required.

The implementation will not inline code across mismatching target features.
Calling a function annotated with both `#[target_feature]` and
`#[always_inline]` from a context with a mismatching target feature is illegal;
a diagnostic is required.

Annotating a function with `#[target_feature]` potentially changes the ABI of
the function, for example, if portable vector types are used as function
arguments or return-types, or integer types like `_128` are used. Whether
calling a function annotated with `#[target_feature]` from a context with a
mismatching target feature is supported is _implementation defined_ (see
unresolved questions). Note: the implementation can either produce a hard-error,
or perform an implicit conversion that converts between ABIs (both GCC and Clang
do this implicitly and can optionally emit a warning which can be turned into a
hard error).

Taking a function pointer to a function annotated with `#[target_feature]` is
illegal if the ABI of the function does not match the ABI of the function
pointer; diagnostic required.

**Example 0 (basics):**

```rust
// This function will be optimized for different targets
#[always_inline] fn foo_impl() { ... }

// This generates a stub for CPUs that support SSE4:
#[target_feature = "+sse4"] unsafe fn foo_sse4() { foo_impl() } 

// This generates a stub for CPUs that support AVX:
#[target_feature = "+avx"] unsafe fn foo_avx() { foo_impl() } 

// This global function pointer can be used to dispatch at run-time to the "best"
// implementation of foo:
static global_foo_ptr: fn() -> () = initialize_foo();
// ^^^ this function pointer has no issues because it does not use any 
//     arguments or return values whose ABI might change in the presence
//     of target feature

// This function initializes the global function pointer 
fn initialize_global_foo_ptr() -> fn () -> () {
    if std::cpuid::has_feature("avx") { // (unresolved) run-time feature detection
      unsafe { foo_avx }
    } else if std::cpuid::has_feature("sse4") {
      unsafe { foo_sse4 }
    } else {
      foo_impl // use the default version
    }
}
```

**Example 1 (inlining):**

```rust
#[target_feature="+avx"] unsafe fn foo();
#[target_feature="+avx"] #[always_inline] unsafe fn bar();

fn has_feature() -> bool;

#[target_feature="+sse3"]
unsafe fn baz() {
  if std::cpuid::has_feature("avx") {
      foo(); // OK: foo is not inlined into baz
      bar(); // ERROR: cannot inline `#[always_inline]` function bar
             // bar requires target feature "avx" but baz only provides "sse3"
  }
}
```

**Example 2 (ABI):**

```rust
#[target_feature="+sse2"] 
unsafe fn foo_sse2(a: f32x8) -> f32x8 { a } // ABI: 2x 128bit registers

#[target_feature="+avx2"] 
unsafe fn foo_avx2(a: f32x8) -> f32x8 { // ABI: 1x 256bit register
  foo_sse2(a) // ABI mismatch: implicit conversion or hard error (unresolved)
}

#[target_feature="+sse2"]
unsafe fn bar() {
  type fn_ptr = fn(f32x8) -> f32x8;
  let mut p0: fn_ptr = foo_sse2; // OK
  let p1: fn_ptr = foo_avx2; // ERROR: mismatching ABI
  let p2 = foo_avx2; // OK
  p0 = p2; // ERROR: mismatching ABI
}
```

## Conditional compilation: `cfg!(target_feature)`

The
[`cfg!(target_feature = "feature_name")`](https://github.com/rust-lang/rust/issues/29717) macro
allows querying at compile-time whether a target feature is enabled in the
current scope. It returns `true` if the feature is enabled, and `false`
otherwise.

In a function annotated with `#[target_feature = "feature_name"]` the macro
`cfg!(target_feature = "feature_name")` _must_ expand to `true` if the generated
code for the function uses the feature. Otherwise, the value of the macro is
undefined. 

**Example 3 (conditional compilation):**

```rust 
fn bzhi_u32(x: u32, bit_position: u32) -> u32 {
    // Conditional compilation: both branches must be syntactically valid,
    // but it suffices that the true branch type-checks:
    #[cfg!(target_feature = "bmi2")] {
        // if this code is being compiled with BMI2 support, use a BMI2 instruction:
        unsafe { intrinsic::bmi2::bzhi(x, bit_position) }
    } 
    #[not(cfg!(target_feature = "bmi2"))] {
        // otherwise, call a portable emulation of the BMI2 instruction
        portable_emulation::bzhi(x, bit_position)
    } 
}

fn bzhi_u64(x: u64, bit_position: u64) -> u64 {
    // Here both branches must type-check and whether the false branch is removed
    // or not is left up to the optimizer.
    if cfg!(target_feature = "bmi2") {  // `cfg!` expands to `true` or `false` at compile-time
        // if target has the BMI2 instruction set, use a BMI2 instruction:
        unsafe { intrinsic::bmi2::bzhi(x, bit_position) }
    } else {
        // otherwise call an algorithm that emulates the instruction:
        portable_emulation::bzhi(x, bit_position)
    }
}
```

**Example 4 (value of `cfg!` within `#[target_feature]`):**

```rust
#[target_feature = "+avx"] 
unsafe fn foo() {
  if cfg!(target_feature = "avx") { /* this branch is always taken */ }
  else { /* this branch is never taken */ }
  #[not(cfg!(target_feature = "avx"))] {
    // this is dead code
  }
}
```

## (unresolved) Run-time feature detection

Writing safe wrappers around `unsafe` functions annotated with
`#[target_feature]` requires run-time feature detection. This section explores
the design space of run-time feature detection. Whether we need run-time feature
detection before stabilizing this RFC is an unresolved question.

The logic users (or procedural macros) will write for performing run-time
feature detection looks like this:

**Example 5 (run-time feature detection overview):**

```rust 
if std::cpuid::has_feature("feature_name") {
  // do something
} else {
  // do something else
}
```

All examples in this RFC call `std::cpuid::has_feature(&str) -> bool` to perform
run-time feature detection. This is a stub, multiple alternative exists.
Run-time feature detection can be provided:

- as a library:
  - in a third-party crate,
  - a crate in the nursery, or
  - as a standard library module, e.g., `std::cpuid`, or
- as a language feature (e.g. `cfg!(runtime_target_feature = "feature")`
  expanding to some library function call).

All of them work with this RFC as is (or with very minor changes). 

However, the best API for such a library or language feature is unknown. For
example, it might be better to use an `enum` to pattern-match on features
instead of a `&str` and a `bool`. These options have not been explored in the
ecosystem yet.

For this reason, ideally we would let solutions compete in third-party library
crates, and maybe at some point move the winner into the nursery or the standard
library. [Such crates already exist](https://docs.rs/cpuid/).

There are two main arguments for including run-time feature detection in the
standard library or the language:

- run-time feature detection is required to ensure memory safety when calling
  unsafe functions annotated with `#[target_feature]`. A popular third-party
  crate getting out-of-sync with rustc in a dangerous way could silently
  introduce memory unsafety in the rust ecosystem. And,

- implementing run-time feature detection in Rust requires using the `asm!`
  macro, which means that either crates implementing it will need to require
  unstable, or link to a library that hides this (like the cpuid does which
  links to the C libcpuid library).

As a consequence, we should consider whether we should stabilize some form of
run-time feature detection alongside this RFC and make it part of the standard
library or the language. Deciding whether we want to do this or not is left as
an unresolved question to be resolved before stabilization.

Note: the host detection library used by LLVM in `-march=native` is in
[`]llvm/lib/Support/Host.cpp`](https://github.com/llvm-mirror/llvm/blob/master/lib/Support/Host.cpp).
It is heavy-weight, and not intended to be used in the run-time of C-like
languages. There is a `__builtin_cpu_supports`
in
[compiler-rt/lib/builtins/cpu_model.c](https://github.com/llvm-mirror/compiler-rt/blob/27ebbdc985fb55567329aa2b510229cc19bd62c5/lib/builtins/cpu_model.c) that
was copy-pasted from LLVM in July 2016 and is x86 only.

## Backend compilation options

There are currently two ways of passing feature information to rustc's code
generation backend on stable rust. These are the proposed semantics for these
features:

- `-C --target-feature=+/-backend_target_feature_name`: where `+/-` add/remove
  features from the default feature set of the platform for the whole crate. The
  behavior for non-stabilized features is _implementation defined_. The behavior
  for stabilized features is:
  
  - to implicitly mark all functions in the crate with `#[target_feature =
    "+/-feature_name"]` (where `-` is still a hard error)
  
  - `cfg!(target_feature = "feature_name")` returns `true` if the feature is
    enabled. If the backend does not support the feature, the feature might be
    disabled even if the user explicitly enabled it, in which case `cfg!`
    returns false; a soft diagnostic is encouraged.

- `-C --target-cpu=backend_cpu_name`, which changes the default feature set of
  the crate to be that of `backend_cpu_name`.

Since the behavior for unstabilized features is _implementation defined_ and
currently no features are stabilized, rustc can continue to provide backwards
compatible behavior for all currently implemented features.

As features are stabilized, the behavior of the backend compilation options must
match the one specified on this RFC for those features. Whether different
feature names, or a warning period will be provided, is left to the RFCs
proposing the stabilization of concrete features and should be evaluated on a
per-feature basis depending on, e.g., impact.

It is recommended that the implementation emits a soft diagnostic when unstable
features are used via `--target-feature` on stable.

These options are already available on stable rust, so we are constrained on the
amount of changes that can be done here. An alternative would be to keep
`--target-feature` as is and introduce a new `--stable-target-feature` or
similar. Which approach is best is an unresolved question.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

There are two parts to this story, the low-level part, and the high-level part.

Starting with the high-level part, this is how users should be able to use
target features:

**Example 6 (high-level usage of target features):**

```rust
#[ifunc("default", "sse4", "avx", "avx2")]
fn foo() {}

... foo(); // dispatches to the best implementation at run-time

#[cfg!(target_feature = "sse4")] {
  foo(); // dispatches to the sse4 implementation at compile-time
}
```

Here, `ifunc` is a procedural macro that abstracts all the complexity and
unsafety of dealing with target features (some of these
macros
[already](https://github.com/alexcrichton/cfg-specialize/blob/master/cfg-specialize-macros) [exist](https://github.com/parched/runtime-target-feature-rs)).

To explain the low-lever part of the story, this is what `ifunc!` could expand
to:

**Example 7 (ifunc expansion):**

```rust
// Copy-pastes "foo" and generates code for multiple target features:
unsafe fn foo_default() { ...foo tokens... }
#[target_feature="+sse4"] unsafe fn foo_sse4() { ...foo tokens... }
#[target_feature="+avx"]  unsafe fn foo_avx() { ...foo tokens... }
#[target_feature="+avx2"] unsafe fn foo_avx2() { ...foo tokens... }

// Initializes `foo` on binary initialization
static foo_ptr: fn() -> () = initialize_foo();

fn initialize_foo() -> typeof(foo) {
    // run-time feature detection:
    if std::cpuid::has_feature("avx2")  { return unsafe { foo_avx2 } }
    if std::cpuid::has_feature("avx")  { return unsafe { foo_avx } }
    if std::cpuid::has_feature("sse4")  { return unsafe { foo_sse4 } }
    foo_default
}

// Wrap foo to do compile-time dispatch
#[always_inline] fn foo() {
  #[cfg!(target_feature = "avx2")] 
  { unsafe { foo_avx2() } } 
  #[and(cfg!(target_feature = "avx"), not(cfg!(target_feature = "avx2")))] 
  { unsafe { foo_avx() } } 
  #[and(cfg!(target_feature = "sse4"), not(cfg!(target_feature = "avx")))] 
  { unsafe { foo_sse4() } } 
  #[not(cfg!(target_feature = "sse4"))] 
  { foo_ptr() }
}
```

Note that there are many solutions to this problem and they have different
trade-offs, but these can be explored in procedural macros. When wrapping unsafe
intrinsics, conditional compilation can be used to create zero-cost wrappers:

**Example 8 (three-layered approach to target features):**

```rust
// Raw unsafe intrinsic: in LLVM, std::intrinsic, etc.
// Calling this on an unsupported target is undefined behavior. 
extern "C" { fn raw_intrinsic_function(f64, f64) -> f64; }

// Software emulation of the intrinsic, 
// works on all architectures.
fn software_emulation_of_raw_intrinsic_function(f64, f64) -> f64;

// Safe zero-cost wrapper over the intrinsic
// (i.e. can be inlined)
fn my_intrinsic(a: f64, b: f64) -> f64 {
  #[cfg!(target_feature = "some_feature")] {
    // If "some_feature" is enabled, it is safe to call the 
    // raw intrinsic function
    unsafe { raw_intrinsic_function(a, b) }
  }
  #[not(cfg!(target_feature = "some_feature"))] {
     // if "some_feature" is disabled calling 
     // the raw intrinsic function is undefined behavior (per LLVM), 
     // we call the safe software emulation of the intrinsic:
     software_emulation_of_raw_intrinsic_function(a, b)
  }
}

#[ifunc("default", "avx")]
fn my_intrinsic_rt(a: f64, b: f64) -> f64 { my_intrinsic(a, b) }
```

Note that `ifunc!` is not part of this proposal, it is a procedural macro that
can be built on top and provided as a library. It is expected that similar
macros for common usage patterns will emerge. Those needing something specific
can still write unsafe code directly and deal (or not, its up to them) with it.

Due to the low-level and high-level nature of these feature we will need two
kinds of documentation.

For the low level part:

- a section in the book listing the stabilized target features, and including an
  example application that uses both `cfg!` and `#[target_feature]` to explain
  how they work, 
- extend the section on `cfg!` with `target_feature`, and
- (unresolved) document run-time feature detection.

For the high-level part we should aim to bring third-party crates implementing
`ifunc!` or similar close to 1.0 releases before stabilization. 

# Drawbacks
[drawbacks]: #drawbacks

- Obvious increase in language complexity. 

The main drawback of not solving this issue is that many libraries that require
conditional feature-dependent compilation or run-time selection of code for
different features (SIMD, BMI, AES, ...) cannot be written efficiently in stable
Rust.


# Alternatives
[alternatives]: #alternatives

## Can we make `#[target_feature]` safe ?

Calling a function annotated with `#[target_feature]` on a host that does not
support the feature invokes undefined behavior in LLVM, the assembler, and
possibly the
CPU.
[See this comment](https://github.com/rust-lang/rfcs/pull/2045#issuecomment-311325202).

## Relax inlining restrictions of `#[target_feature]`

The rules proposed in this RFC for the interaction between `#[target_feature]`,
inlining, and `#[always_inline]` are modeled after the rules in LLVM and are
consistent with clang, e.g., see
this [nice clang compilation error](https://godbolt.org/g/2kSY4R)
([gcc also errors but the message is less nice](https://godbolt.org/g/tkPgCU)).

Rust, the language, could relax these rules. Whether doing so would require
changes to LLVM hasn't been explored here.

In particular, we could relax the rules to allow the "Example 1 (inlining)" to
compile by using target feature boundaries within which code can be inlined, but
across which code cannot be reordered:

```rust
#[target_feature="+sse3"]
// -- sse3 boundary start (applies to fn arguments as well)
unsafe fn baz() {
  if std::cpuid::has_feature("avx") {
    // -- sse3 boundary ends
    // -- avx boundary starts
    foo(); // might or might not be inlined, up to the inliner
    // -- avx boundary ends
    // -- avx boundary starts (the end and start of the boundary could be merged)
    // bar is inlined here, its code cannot be reordered across the avx boudnary 
    // -- avx boundary ends
    // -- sse3 boundary starts
    }
  } 
}
// -- sse3 boundary ends (applies to drop as well)
```

## No segfaults from `unsafe` code

Calling a `#[target_feature]`-annotated function on a platform that does not
support it invokes undefined behavior. A friendly compiler could use run-time
feature detection to check whether calling the function is safe and emit a nice
`panic!` message. 

This can be done, for example, by desugaring this: 

```rust
#[target_feature = "+avx"] unsafe fn foo();
```

into this:

```rust
#[target_feature = "+avx"] unsafe fn foo_impl() { ...foo tokens... };

// this function will be called if avx is not available:
fn foo_fallback() { 
    panic!("calling foo() requires a target with avx support")
}

// run-time feature detection on initialization
static foo_ptr: fn() -> () = if std::cpuid::has_feature("avx") { 
    unsafe { foo_impl }
} else {
    foo_fallback
};

// dispatches foo via function pointer to produce nice diagnostic
unsafe fn foo() { foo_ptr() }
```

Since this is not required for safety, and can probably be done without an RFC,
it is not proposed nor discussed further in this RFC.

# Unresolved questions
[unresolved]: #unresolved-questions

- Do we need to stabilize some form of run-time feature detection together with
  this RFC? If so, what is the best API for run-time feature detection (should
  be pursued in a different RFC)?

- Should calling a function from a context with a mismatching feature that
  involves a mismatching ABI fail or implicitly convert values between the
  different ABIs?

- Does it make sense to support removing features from the default feature set
  using `#[target_feature="-feature"]`? Can we do this while preventing
  non-sensical examples [like this one](https://internals.rust-lang.org/t/pre-rfc-stabilization-of-target-feature/5176/26)?

- What do we do about mutually exclusive features like `+/-thumb_mode` ?

- Can we incrementally improve `-C --target-feature` without breaking backwards
  compatibility? Or do we need to keep it as is and add a `-C
  --stable-target-feature` (or similar) flag?

# Acknowledgements
[acknowledgments]: #acknowledgements

@parched @burntsushi @alexcrichton @est31 @pedrocr @chandlerc

- `#[target_feature]` Pull-Request: https://github.com/rust-lang/rust/pull/38079
- `cfg_target_feature` tracking issue: https://github.com/rust-lang/rust/issues/29717
