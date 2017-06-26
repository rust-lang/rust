- Feature Name: `target_feature` / `cfg_target_feature`
- Start Date: 2017-06-26
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Motivation and Summary
[summary]: #summary

Some `x86_64` CPUs support extra "features", like AVX SIMD vector instructions,
that not all `x86_64` CPUs do support. This RFC proposes extending Rust with:

- **conditional compilation on target features**: choosing different code-paths at
  compile-time depending on the features being target by the compiler, and
  
- **unconditional code generation**: unconditionally generating code that uses some
  set of target features independently of the features currently being targeted
  by the compiler. This allows embedding code optimized for different CPUs
  within the same binary.
  
This RFC also specifies the semantics of the backend options `-C
--target-feature/--target-cpu`.


# Detailed design
[design]: #detailed-design

This RFC proposes adding the following two constructs to the language:

- the `cfg!(target_feature = "feature_name")` macro, and
- the `#[target_feature = "+feature_name") ` function attribute for `unsafe`
  functions.

## Target features

Each rustc target has a default set of target features enabled and this set is
_implementation defined_.

This RFC does not propose adding any features to the language, so this
constructs won't be able to do anything useful with this RFC as is. 

Each target feature should:

- be proposed in its own mini-RFC, RFC, or rustc-issue.
- all unstable target features should be behind their own feature macro of the
  form: `target_feature_feature_name` (where `feature_name` should be replaced
  by the name of the feature ).
  
To use unstable features on nightly, crates must opt into them as usual by
writing, for example, `#![allow(target_feature_avx2)]`.


## Unconditional code generation: `#[target_feature]`

The `unsafe` function attribute [`#[target_feature =
"+feature_name"]`](https://github.com/rust-lang/rust/pull/38079) _extends_ the
feature set of a function. That is, `#[target_feature = "+feature_name"] unsafe
fn foo(...)` means that the compiler is allowed to generate code for `foo` that
uses features from `feature_name` in addition to the default feature set of the
target.

> Note: the rationale behind `target_feature` applying to unsafe functions only
> is discussed in the "Alternatives" section.

Removing features from the default feature set, e.g., using `#[target_feature =
"-feature_name"]` is illegal (to prevent ABI issues, see below); a hard error is
required.

Also, executing any code on a target that doesn't support the features it was
compiled with is _undefined behavior_; no diagnostic required. Providing
best-effort compile-time or run-time errors is left as a Quality of
Implementation issue.

This example shows how to use unconditional code generation and
the [`cpuid` crate](https://docs.rs/cpuid/) (not part of this RFC) to
unconditionally generate code using different target features for a function,
and dispatch to the correct implementation at run-time:


```rust
// This function will be optimized for different targets
#[always_inline] fn foo_impl(...) { ... }

// This generate code for CPUs that support SSE4:
#[target_feature = "+sse4"] unsafe fn foo_sse4(...) { foo_impl(...) } 

// This generates code for CPUs that support AVX:
#[target_feature = "+avx"] unsafe fn foo_avx(...) { foo_impl(...) } 

// This global function pointer can be used to dispatch to the fastest
// implementation of foo:
static global_foo_ptr: fn(...) -> ... = initialize_foo();

fn initialize_global_foo_ptr() -> fn (...) -> ... {
    // This function initializes the global function pointer using the cpuid crate:
    let info = std::cpuid::identify().unwrap();
    match info {
        info.has_feature(cpuid::CpuFeature::AVX) => unsafe { foo_avx }, 
        info.has_feature(cpuid::CpuFeature::SSE4) => unsafe { foo_sse4 }, 
        _ => foo_impl  //< Use the default version
    }
}


// This wrapper function does feature detection and dispatches the call to the
// best implementation:
fn foo(...) -> ... {
    static info: CpuInfo = cpuid::identify().unwrap();
    match info {
        info.has_feature(cpuid::CpuFeature::AVX) => unsafe { foo_avx(...) }, 
        info.has_feature(cpuid::CpuFeature::SSE4) => unsafe { foo_sse4(...) }, 
        _ => foo_impl(...)  // call the default version
    }
}
```

The `#[target_feature]` attribute is akin to the clang and
gcc
[`__attribute__ ((__target__ ("feature")))`](https://clang.llvm.org/docs/AttributeReference.html#target-gnu-target) 
attribute.

### Interaction with inlining and `#[always_inline]`

TL;DR: inlining, and using `#[always_inline]`, shall not change
the semantics of the program on the presence of `#[target_feature]`.

The implementation will not inline code across mismatching target features.

Calling a function annotated with both `#[target_feature]` and
`#[always_inline]` from a context with a mismatching target feature is illegal;
and a diagnostic required.

Example:

```rust
#[target_feature="+avx"] fn foo();

fn has_feature() -> bool;

#[target_feature="+sse3"]
fn baz() {
  if has_feature() {
    foo(); // OK: foo is not inlined into baz
    bar(); // ERROR: cannot inline `#[always_inline]` function bar
           // bar requires target feature "avx" but baz only provides "sse3"
  }
}
```

This rules are consistent with what LLVM currently provides. Currently, if a
function is annotated with both `#[always_inline]` and `#[target_feature]`,
trying to call it from a context with a mismatching target feature results
in [a nice clang compilation error](https://godbolt.org/g/3ELXyJ).

We could relax this rules in the future to allow inlining as long as the
semantics of the program aren't changed. That is, we could allow inlining `foo`
above as long as the compiler cannot re-order code across mismatching feature
boundaries:

```rust
#[target_feature="+sse3"]
// -- sse3 boundary start (applies to fn arguments as well)
fn bar() {
  if has_feature() {
    // -- sse3 boundary ends
    // -- avx boundary starts
    // foo is inlined here 
    // -- avx boundary ends
    // -- sse3 boundary starts
    }
  } 
}
// -- sse3 boundary ends (applies to drop as well)
```

Such a relaxed rule can be proposed later, but is probably currently not
actionable since it would require changes to LLVM.

## Conditional compilation: `cfg!(target_feature)`

The
[`cfg!(target_feature = "feature_name")`](https://github.com/rust-lang/rust/issues/29717) macro
allows querying specific hardware features of the target _at compile-time_. 

The macro `cfg!(target_feature = "feature_name")` returns:

-  `true` if the feature is enabled,
-  `false` if the feature is disabled.


Since the result is known at compile-time, this information can be used for
conditional compilation:


```rust 
// Conditional compilation: both branches must be syntactically valid,
// but it suffices that the true branch type-checks:
#[cfg!(target_feature = "bmi2")] {
    // if target has the BMI2 instruction set, use a BMI2 instruction:
    unsafe { intrinsic::bmi2::bzhi(x, bit_position) }
} 
#[not(cfg!(target_feature = "bmi2"))] {
    // otherwise call an algorithm that emulates the instruction:
    software_fallback::bzhi(x, bit_position)
} 

// Here both branches must type-check:
if cfg!(target_feature = "bmi2") {  // `cfg!` expands to `true` or `false` at compile-time
    // if target has the BMI2 instruction set, use a BMI2 instruction:
    unsafe { intrinsic::bmi2::bzhi(x, bit_position) }
} else {
    // otherwise call an algorithm that emulates the instruction:
    software_fallback::bzhi(x, bit_position)
}

#[target_feature = "+avx"] 
unsafe fn foo() {
  if cfg!(target_feature = "avx") { /* this branch is always taken */ }
  else { /* this branch is never taken */ }
  #[not(cfg!(target_feature = "avx"))] {
    // this is dead code
  }
}

```

### Backend compilation options

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
features are used via `--target-feature`.

These options are already on stable rust, so we are constrained on the amount of
changes that can be done here. An alternative is proposed in the alternatives section.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

These features are low-level in nature. We expect that:

- for the most commonly used platforms, an ecosystem of cpuid crates will
emerge, that will allow querying features at run-time
(e.g. [the `cpuid` crate](https://docs.rs/cpuid/)), or that we will maintain one
of these crates either in the nursery or as a module in `std` that can be kept
in sync with rustc.

- procedural macros like
  the
  [runtime-target-feature-rs crate](https://github.com/parched/runtime-target-feature-rs) will
  emerge and make the generation of target dependent code as easy as:


```rust
#[ifunc("sse4", "avx", "avx2")]
fn foo() -> ... {}
```

where `ifunc` is a procedural macro that:

- creates copies of foo `foo_sse4/_avx/_avx2` annotated with the corresponding
  `#[target_feature = "+xxx"]`,
- on binary initialization does run-time feature detection using some cpuid
  libraries, does error handling, and initializes a global function pointer
- such that `foo` can then be safely called without any run-time overhead, and
  automatically dispatches to the most efficient implementation.

These crates all build up on `cfg!` and `#[target_feature]` to generate target
dependent code and optimize away run-time checks when the target features are
known at compile-time, and on platform specific crates to do the run-time
feature detection.

The advantage of providing this behavior as libraries is that it allows users
for which these libraries are not enough, e.g. because run-time feature
detection is not possible for a particular platform, to still solve their
problems using `#[target_feature]` and a bit of unsafe code.

Here is a full example of how this might look like which uses a different
`ifunc!` and shows its expansion:

```rust
// Raw intrinsic function: dispatches to LLVM directly. 
// Calling this on an unsupported target is undefined behavior. 
extern "C" { fn raw_intrinsic_function(f64, f64) -> f64; }

// Software emulation of the intrinsic, 
// works on all architectures.
fn software_emulation_of_raw_intrinsic_function(f64, f64) -> f64;

// Safe zero-cost wrapper over the intrinsic
// (i.e. can be inlined -- whether this is a good idea 
// is an unresolved question).
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

// Provides run-time dispatch to the best implementation:
// ifunc! is a procedural macro that generates copies 
/// of `my_intrinsic` with different target features,
// does run-time feature detection on binary initialization, 
// and sets my_intrinsic_rt  to dispatch to the appropriate 
// implementation:
static my_intrinsic_rt = ifunc!(my_intrinsic, ["default_target", "some_feature"]);

// This procedural macro expands to the following:

// Copies the tokens of `my_intrinsic` for each feature
// into a different function and annotates it with
// #[target_feature]. Due to this, calling this function
// is unsafe, since doing so on targets without the feature
// introduces undefined behavior.
#[target_feature = "some_feature"]
unsafe fn my_intrinsic_some_feature_wrapper(a: f64, b: f64) 
{
  #[cfg!(target_feature = "some_feature")] {
    // this branch will always be taken because 
    // `#[target_feautre = "..."]` defines `cfg!(target_feature = "...") == true`
    unsafe { raw_intrinsic_function(a, b) }
  }
  #[not(cfg!(target_feature = "some_feature"))] {
     // dead code: this branch will never be taken
     software_emulation_of_raw_intrinsic_function(a, b)
  }
}

// This function does run-time feature detection to return a function pointer
// to the "best" implementation (run-time feature detection is not part of this RFC):
fn initialize_my_intrinsic_fn_ptr() {
  if std::cpuid::has_feature("some_feature") -> typeof(my_intrinsic) {
    // Since we have made sure that the target the binary is running on 
    // has the feature, calling my_intrinsic_some_feature_wrapper is safe: 
    unsafe { 
        my_intrinsic_some_feature_wrapper 
        /* do we need a cast here?: as typeof(my_intrinsic) */ 
    }
  } else { 
    // because we passed "default_target" to ifunc we fall back 
    // to the safe implementation. We could otherwise return a 
    // function pointer to a stub function: 
    // fn my_intrinsic_fallback(f64,f64) -> f64 { panic!("nice error message") }
    my_intrinsic
  }
}
```

This also means that we need two kinds of documentation. The low level part:

- a section in the book listing the stabilized target features, and including an
  example application that uses both `cfg!` and `#[target_feature]` to explain
  how they work, and
- extend the section on `cfg!` with `target_feature`

And a high level part, which could be achieved by getting some of the
fundamental libraries close to a 1.0 release before `target_feature` is stabilized.

The important thing about teaching the low level library parts is teaching how
these features layer up:

- `cfg!` allows you to implement different algorithms using conditional compilation

- `#[target_feature]` allows instantiating this code for different target features

- a crate allows run-time feature detection and dispatching on appropriate functions

```rust
// conditional compilation:
#[always_inline] fn foo_impl() {
    #[cfg(target_feature("avx"))] {
      // AVX implementation
    }
    #[not(cfg(target_feature("avx")))] {
      // fallback for when AVX is disabled
    }
}

// unconditional code generation:
#[target_feature = "+avx"] fn foo_avx() { foo_impl() }
fn foo_fallback() { foo_impl() }

// using some third-party cpuid crate:
fn foo() {
  if std::cpuid::has_target_feature("avx") {
    foo_avx()
  } else {
    foo_fallback() 
  }
}
```

# Drawbacks
[drawbacks]: #drawbacks

- Obvious increase in language complexity. 
- `#[target_feature]` only allowed for `unsafe fn`s

The main drawback of not solving this issue is that many libraries that require
conditional feature-dependent compilation or run-time selection of code for
different features (SIMD, BMI, AES, ...) cannot be written efficiently in stable
Rust.


# Alternatives
[alternatives]: #alternatives

## Make `#[target_feature]` safe 

(TODO: there is an unresolved question of whether `#[target_feature]` is
inherently unsafe)

To make `#[target_feature]` safe we would need to ensure that calling a function
marked with the `#[target_feature]` attribute cannot result in memory unsafety.

There are multiple problems:

- 1. Functions with different target features might have a different ABI (this
  should be resolved before stabilization, see unresolved questions below)

- 2. Hardware undefined behavior: attempting to execute an illegal instruction
  on some (less safe) platforms (like AVR) invokes undefined behavior in hardware.
  
Supposing that we can solve 1., 2. still would mean that invoking a
`#[target_feature]` function on the wrong platform can produce memory unsafety.

On most widely used platforms memory unsafety can never occur
because the CPU will raise an illegal instruction exception (SIGILL) crashing
the process. 

The multiple alternatives considered are:

- run-time feature detection, and
- making the `unsafe`ty of `#[target_feature]` feature dependent.

These are discussed below. 

Because both solutions have drawbacks, and both can be implemented in a
backwards compatible way, this RFC proposes to initially allow
`#[target_feature]` on unsafe functions only.

### Run-time feature detection

(TODO: there is an unresolved question of whether `#[target_feature]` is
inherently unsafe)

First, less safe platforms in which an illegal instruction introduces hardware
undefined behavior do not allow querying of CPU features at run-time, so in
those platforms this is not a solution.

Second, run-time feature detection introduces a run-time cost. Consider the
following code:

```rust
use std::io;

#[target_feature = "+avx2"]
unsafe fn foo(); 

fn main() {
    let mut s = String::new();
    io::stdin().read_line(&mut s).unwrap();

    match s.trim_right().parse::<i32>() {
        Ok(i) => { 
          if i == 1337 { unsafe { foo(); } }
        },
        Err(_) => println!("Invalid number."),
    }
}
```

This code works fine on all x86 CPUs, unless the user passes `1337` as input, in
which case it only works for `AVX2` CPUs. The implementation can check whether
the CPU supports AVX2 on initialization, but even if it doesn't, the program is
correct unless the user inputs `1337`, so the fatal check must be done when
calling any functions using `#[target_feature]`. Now replace `1337` with the
result of some system `cpuid` library, there is no way the compiler can know
that a result value from a third-party library correspond to a target feature
unless we decide to encode these in the type system (which is an option not
proposed here).

### Target-feature dependent unsafety

(TODO: there is an unresolved question of whether `#[target_feature]` is
inherently unsafe)

This would allow us to make `#[target_feature]` safe for all the major
platforms, requiring unsafe only for those platforms in which memory unsafety
can be introduced due to `#[target_feature]`. Since run-time feature detection
in these platforms is not available, users still not have a safe way of using
`#[target_feature]` beyond "being careful", but this just reflects the reality
of using unsafe hardware. 

The big contra is that sometimes `#[target_feature]` requires an `unsafe` function
and sometimes it does not.

For me, this is the most appealing alternative to make `#[target_feature]` safe,
since it "just works" on most commonly used platforms, and it reflects a reality
of the hardware on less safe ones.

## Extra guarantees: no segfaults

Just because we make `#[target_feature]` safe does not mean that rust code won't
segfault or that we will get great error messages. We could try to go the extra
mile and try to guarantee "no segfaults" due to calling `#[target_feature]`
functions on the wrong platforms

Given the flexible nature of `#[target_feature]`, the only way to do this with
the feature as proposed is to perform run-time feature detection.

The main issues with run-time feature detection are:

- increased binary size: acceptable only if only those using `target_feature`
  have to pay for it (doable).
- run-time check on potentially every call site (including loops, etc.): not
  acceptable, probably unsolvable.

This check cannot be moved to the initialization of the binary, since just
because a function is in the binary doesn't mean it will be called.

I would like to see this check enabled in debug builds (or some other types of
builds), and once we gain experience with it, we might be able to enable it on
all builds without incurring a performance cost. I just don't think that with
the feature as proposed it is possible to do so.

## Removing features from the default feature set

Allowing `#[target_feature = "-feature_name"]` can result
in
[non-sensical behavior](https://internals.rust-lang.org/t/pre-rfc-stabilization-of-target-feature/5176/26?u=gnzlbg).

The backend might always assume that at least the default feature set for the
current platform is available. In these cases it might be better to globally
choose a different default using `-C --target-cpu` / `-C --target-feature`.

As we resolve the ABI issues mentioned above we'll gain more experience on
whether it is possible to do this safely in practice, as well as maybe some
situations in which doing this is useful.

## Not break backwards compatibility with `--target-feature`

A reliable way of avoiding breaking backwards compatibility with the current
behavior of `-C --target-feature` would be to add a new option for stabilized
features `-C --stable-target-feature`. It is, however, unfortunate that the
verbose alternative will be come the correct one. 

## Make `--target-feature/--target-cpu` unsafe

Rename them to `--unsafe-target-feature/--unsafe-target-cpu`. The rationale
would be that a binary for an architecture compiled with the wrong flags would
still be able to run in that architecture, but because using illegal
instructions on some architectures might lead to memory unsafety, then these
operations are unsafe.

## Run-time feature detection

This RFC proposes that `#[target_feature]` only applies to `unsafe fn`s. To
write safe wrappers around those some sort of run-time target feature detection
is required.

This RFC doesn't propose any system for this, leaving this to third-party crates.

There are, however, many approaches to this problem:

- third-party crates (works with this RFC as is)
- curated crate in the nursery (works with this RFC as is)
- as a standard library module (works with this RFC as is)
- as a language feature (works with this RFC as is).

In a nutshell, the logic that users (or procedural macros) must write is:

```rust 
if std::cpuid::has_runtime_feature("feature_name") {
  // do something
} else {
  // do something else
}
```

Whether this function is in a third-party crate, a crate in the nursery, the
standard library, or in the language somehow (e.g. an `if
cfg!(runtime_target_feature = "...")` analogous to `cfg!(target_feature)` that
desugars into some library call) is a mostly cosmetic issue. Also, whether we
just want to return `true` or `false` or offer pattern matching on features
(e.g. using an `enum`), or even a "feature hierarchy" are cosmetic issues as
well.

This RFC chooses not to proposes a solution to this problem. We currently only
have one crate that does runtime feature detection on
`x86`, [the `cpuid` crate](https://docs.rs/cpuid/). Whether its API is the best
fit for the problem, or better APIs do exist, is an open problem. 


While it is appealing to explore this on third-party crates, remember that
correct runtime feature detection is required to ensure memory safety when using
`#[target_feature]`. That is, a third-party crate getting out-of-sync with rustc
can result in memory unsafety. 

This is a very valid concern raised during the discussion in internals, and I
think that before stabilization we should explore the different approaches and
choose one that prevents memory unsafety. 

In my opinion, having a module in the standard library that is maintained in
sync with the stable target features is a nice compromise, but this would
require modifying this RFC to require that new target features _must_
incorporate some sort of run-time feature detection.

# Unresolved questions
[unresolved]: #unresolved-questions

Before stabilization the following issues _must_ be resolved:

- Is it possible to make `#[target_feature]` both safe and zero-cost?

On the most widely used platforms the answer is yes, this is possible. On other platforms like AVR the answer is probably not, but we are not sure.

- Is it possible to provide good error messages on misues of `#[target_feature]` at zero-cost ?

The answer is probably not: just because a binary includes a function does not mean that the function will be called. Hence the only way to check whether an invalid call happens is to add a check before each function call which incurs a potentially significant cost (e.g. in the form of missed optimizations).

- Is it possible to automatically detect and correct ABI mismatches between target features?

The implementation must ensure that this cannot invoke undefined behavior. That
is, the implementation must detect this at compile-time, and translate arguments
from one ABI to another. Since this hasn't been implemented yet, it is unclear
how to do this, or if it can be done for all cases.

- Does the unsafety leak? 

That is, when a user writes `if cfg!(target_feature) { ... use some target
features here ... }` or `if std::cpuid::has_target_feature("...") { ... use some
target features here ... }` the compiler might hoist some instrunctions _before_
the check, producing segfaults that shouldn't happen. 

The same can occur if a function with `#[target_feature]` is inlined, and its
instructions are hoisted before `if` checks that detect whether it is valid to
call the function. 

- Does run-time feature detection need to be baked in from the start?

Run-time feature detection is a critical piece of the puzzle required to make
calling `#[target_feature = "..."]` safe. There are different approaches to bake
this in: as part of the standard library, via `cfg!(runtime_target_feature =
"...")`, on a third-party crate, ... Before stabilizing this RFC it should at
least be decided which of these approaches should be pursued, since some of them
might need to be stabilized along with this RFC. 

- Can we lift the restriction on `target_feature="-feature"` to provide
  substractive features?

Some features like `-d16` would need to be exposed as `+d32=-d16` in the current
proposal. There are also mutually exclusive features like `+/-thumb_mode`

- Is calling a `#[target_feature]` function on a platform that doesn't support
  it undefined behavior in LLVM and/or the assembler? 
  
  The issues mentioned
  in
  [this comment](https://github.com/rust-lang/rfcs/pull/2045#issuecomment-311325202) seem
  to indicate that this is the case, and hence that calling `#[target_feature]`
  functions is inherently unsafe.

## Path towards stabilization

1. Enable a warning on nightly on usage of target features without the
   corresponding `feature(target_feature_xxx)` feature flag.
2. After a transition period (e.g. 1 release cycle) make this a hard error on
   nightly.
3. After all unresolved questions are resolved, stabilize `cfg!` and
   `#[target_feature]`
4. ..N: stabilize those `target_feature_xxx`s that we want to have on stable
   following either a mini-RFC in the rust-lang issues for these features or the
   normal RFC process.

# Acknowledgements
[acknowledgments]: #acknowledgements

@parched @burntsushi @alexcrichton @est31 @pedrocr

- `#[target_feature]` Pull-Request: https://github.com/rust-lang/rust/pull/38079
- `cfg_target_feature` tracking issue: https://github.com/rust-lang/rust/issues/29717
