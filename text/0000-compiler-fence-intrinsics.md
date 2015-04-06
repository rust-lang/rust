- Feature Name: compiler_fence_intrinsics
- Start Date: 2015-02-19
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add intrinsics for single-threaded memory fences.

# Motivation

Rust currently supports memory barriers through a set of intrinsics,
`atomic_fence` and its variants, which generate machine instructions and are
suitable as cross-processor fences. However, there is currently no compiler
support for single-threaded fences which do not emit machine instructions.

Certain use cases require that the compiler not reorder loads or stores across a
given barrier but do not require a corresponding hardware guarantee, such as
when a thread interacts with a signal handler which will run on the same thread.
By omitting a fence instruction, relatively costly machine operations can be
avoided.

The C++ equivalent of this feature is `std::atomic_signal_fence`.

# Detailed design

Add four language intrinsics for single-threaded fences:

 * `atomic_compilerfence`
 * `atomic_compilerfence_acq`
 * `atomic_compilerfence_rel`
 * `atomic_compilerfence_acqrel`

These have the same semantics as the existing `atomic_fence` intrinsics but only
constrain memory reordering by the compiler, not by hardware.

The existing fence intrinsics are exported in libstd with safe wrappers, but
this design does not export safe wrappers for the new intrinsics. The existing
fence functions will still perform correctly if used where a single-threaded
fence is called for, but with a slight reduction in efficiency. Not exposing
these new intrinsics through a safe wrapper reduces the possibility for
confusion on which fences are appropriate in a given situation, while still
providing the capability for users to opt in to a single-threaded fence when
appropriate.

# Alternatives

 * Do nothing. The existing fence intrinsics support all use cases, but with a
   negative impact on performance in some situations where a compiler-only fence
   is appropriate.

 * Recommend inline assembly to get a similar effect, such as `asm!("" :::
   "memory" : "volatile")`. LLVM provides an IR item specifically for this case
   (`fence singlethread`), so I believe taking advantage of that feature in LLVM is
   most appropriate, since its semantics are more rigorously defined and less
   likely to yield unexpected (but not necessarily wrong) behavior.

# Unresolved questions

These intrinsics may be better represented with a different name, such as
`atomic_signal_fence` or `atomic_singlethread_fence`. The existing
implementation of atomic intrinsics in the compiler precludes the use of
underscores in their names and I believe it is clearer to refer to this
construct as a "compiler fence" rather than a "signal fence" because not all use
cases necessarily involve signal handlers, hence the current choice of name.
