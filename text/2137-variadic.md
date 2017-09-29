- Feature Name: variadic
- Start Date: 2017-08-21
- RFC PR: https://github.com/rust-lang/rfcs/pull/2137
- Rust Issue: https://github.com/rust-lang/rust/issues/44930

# Summary
[summary]: #summary

Support defining C-compatible variadic functions in Rust, via new intrinsics.
Rust currently supports declaring external variadic functions and calling them
from unsafe code, but does not support writing such functions directly in Rust.
Adding such support will allow Rust to replace a larger variety of C libraries,
avoid requiring C stubs and error-prone reimplementation of platform-specific
code, improve incremental translation of C codebases to Rust, and allow
implementation of variadic callbacks.

# Motivation
[motivation]: #motivation

Rust can currently call any possible C interface, and export *almost* any
interface for C to call. Variadic functions represent one of the last remaining
gaps in the latter. Currently, providing a variadic function callable from C
requires writing a stub function in C, linking that function into the Rust
program, and arranging for that stub to subsequently call into Rust.
Furthermore, even with the arguments packaged into a `va_list` structure by C
code, extracting arguments from that structure requires exceptionally
error-prone, platform-specific code, for which the crates.io ecosystem provides
only partial solutions for a few target architectures.

This RFC does not propose an interface intended for native Rust code to pass
variable numbers of arguments to a native Rust function, nor an interface that
provides any kind of type safety. This proposal exists primarily to allow Rust
to provide interfaces callable from C code.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

C code allows declaring a function callable with a variable number of
arguments, using an ellipsis (`...`) at the end of the argument list. For
compatibility, unsafe Rust code may export a function compatible with this
mechanism.

Such a declaration looks like this:

```rust
pub unsafe extern "C" fn func(arg: T, arg2: T2, mut args: ...) {
    // implementation
}
```

The use of `...` as the type of `args` at the end of the argument list declares
the function as variadic. This must appear as the last argument of the
function, and the function must have at least one argument before it. The
function must use `extern "C"`, and must use `unsafe`. To expose such a
function as a symbol for C code to call directly, the function may want to use
`#[no_mangle]` as well; however, Rust code may also pass the function to C code
expecting a function pointer to a variadic function.

The `args` named in the function declaration has the type
`core::intrinsics::VaList<'a>`, where the compiler supplies a lifetime `'a`
that prevents the arguments from outliving the variadic function.

To access the arguments, Rust provides the following public interfaces in
`core::intrinsics` (also available via `std::intrinsics`):

```rust
/// The argument list of a C-compatible variadic function, corresponding to the
/// underlying C `va_list`. Opaque.
pub struct VaList<'a> { /* fields omitted */ }

// Note: the lifetime on VaList is invariant
impl<'a> VaList<'a> {
    /// Extract the next argument from the argument list. T must have a type
    /// usable in an FFI interface.
    pub unsafe fn arg<T>(&mut self) -> T;

    /// Copy the argument list. Destroys the copy after the closure returns.
    pub fn copy<'ret, F, T>(&self, F) -> T
    where
        F: for<'copy> FnOnce(VaList<'copy>) -> T, T: 'ret;
}
```

The type returned from `VaList::arg` must have a type usable in an `extern "C"`
FFI interface; the compiler allows all the same types returned from
`VaList::arg` that it allows in the function signature of an `extern "C"`
function.

All of the corresponding C integer and float types defined in the `libc` crate
consist of aliases for the underlying Rust types, so `VaList::arg` can also
extract those types.

Note that extracting an argument from a `VaList` follows the C rules for
argument passing and promotion. In particular, C code will promote any argument
smaller than a C `int` to an `int`, and promote `float` to `double`. Thus,
Rust's argument extractions for the corresponding types will extract an `int`
or `double` as appropriate, and convert appropriately.

Like the underlying platform `va_list` structure in C, `VaList` has an opaque,
platform-specific representation.

A variadic function may pass the `VaList` to another function. However, the
lifetime attached to the `VaList` will prevent the variadic function from
returning the `VaList` or otherwise allowing it to outlive that call to the
variadic function. Similarly, the closure called by `copy` cannot return the
`VaList` passed to it or otherwise allow it to outlive the closure.

A function declared with `extern "C"` may accept a `VaList` parameter,
corresponding to a `va_list` parameter in the corresponding C function. For
instance, the `libc` crate could define the `va_list` variants of `printf` as
follows:

```rust
extern "C" {
    pub unsafe fn vprintf(format: *const c_char, ap: VaList) -> c_int;
    pub unsafe fn vfprintf(stream: *mut FILE, format: *const c_char, ap: VaList) -> c_int;
    pub unsafe fn vsprintf(s: *mut c_char, format: *const c_char, ap: VaList) -> c_int;
    pub unsafe fn vsnprintf(s: *mut c_char, n: size_t, format: *const c_char, ap: VaList) -> c_int;
}
```

Note that, per the C semantics, after passing `VaList` to these functions, the
caller can no longer use it, hence the use of the `VaList` type to take
ownership of the object. To continue using the object after a call to these
functions, use `VaList::copy` to pass a copy of it instead.

Conversely, an `unsafe extern "C"` function written in Rust may accept a
`VaList` parameter, to allow implementing the `v` variants of such functions in
Rust. Such a function must not specify the lifetime.

Defining a variadic function, or calling any of these new functions, requires a
feature-gate, `c_variadic`.

Sample Rust code exposing a variadic function:

```rust
#![feature(c_variadic)]

#[no_mangle]
pub unsafe extern "C" fn func(fixed: u32, mut args: ...) {
    let x: u8 = args.arg();
    let y: u16 = args.arg();
    let z: u32 = args.arg();
    println!("{} {} {} {}", fixed, x, y, z);
}
```

Sample C code calling that function:

```c
#include <stdint.h>

void func(uint32_t fixed, ...);

int main(void)
{
    uint8_t x = 10;
    uint16_t y = 15;
    uint32_t z = 20;
    func(5, x, y, z);
    return 0;
}
```

Compiling and linking these two together will produce a program that prints:

```text
5 10 15 20
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

LLVM already provides a set of intrinsics, implementing `va_start`, `va_arg`,
`va_end`, and `va_copy`. The compiler will insert a call to the `va_start`
intrinsic at the start of the function to provide the `VaList` argument (if
used), and a matching call to the `va_end` intrinsic on any exit from the
function. The implementation of `VaList::arg` will call `va_arg`. The
implementation of `VaList::copy` wil call `va_copy`, and then `va_end` after
the closure exits.

`VaList` may become a language item (`#[lang="VaList"]`) to attach the
appropriate compiler handling.

The compiler may need to handle the type `VaList` specially, in order to
provide the desired parameter-passing semantics at FFI boundaries. In
particular, some platforms define `va_list` as a single-element array, such
that declaring a `va_list` allocates storage, but passing a `va_list` as a
function parameter occurs by pointer. The compiler must arrange to handle both
receiving and passing `VaList` parameters in a manner compatible with the C
ABI.

The C standard requires that the call to `va_end` for a `va_list` occur in the
same function as the matching `va_start` or `va_copy` for that `va_list`. Some
C implementations do not enforce this requirement, allowing for functions that
call `va_end` on a passed-in `va_list` that they did not create. This RFC does
not define a means of implementing or calling non-standard functions like these.

Note that on some platforms, these LLVM intrinsics do not fully implement the
necessary functionality, expecting the invoker of the intrinsic to provide
additional LLVM IR code. On such platforms, rustc will need to provide the
appropriate additional code, just as clang does.

This RFC intentionally does not specify or expose the mechanism used to limit
the use of `VaList::arg` only to specific types. The compiler should provide
errors similar to those associated with passing types through FFI function
calls.

# Drawbacks
[drawbacks]: #drawbacks

This feature is highly unsafe, and requires carefully written code to extract
the appropriate argument types provided by the caller, based on whatever
arbitrary runtime information determines those types. However, in this regard,
this feature provides no more unsafety than the equivalent C code, and in fact
provides several additional safety mechanisms, such as automatic handling of
type promotions, lifetimes, copies, and cleanup.

# Rationale and Alternatives
[alternatives]: #alternatives

This represents one of the few C-compatible interfaces that Rust does not
provide. Currently, Rust code wishing to interoperate with C has no alternative
to this mechanism, other than hand-written C stubs. This also limits the
ability to incrementally translate C to Rust, or to bind to C interfaces that
expect variadic callbacks.

Rather than having the compiler invent an appropriate lifetime parameter, we
could simply require the unsafe code implementing a variadic function to avoid
ever allowing the `VaList` structure to outlive it. However, if we can provide
an appropriate compile-time lifetime check, doing would make it easier to
correctly write the appropriate unsafe code.

Rather than naming the argument in the variadic function signature, we could
provide a `VaList::start` function to return one. This would also allow calling
`start` more than once. However, this would complicate the lifetime handling
required to ensure that the `VaList` does not outlive the call to the variadic
function.

We could use several alternative syntaxes to declare the argument in the
signature, including `...args`, or listing the `VaList` or `VaList<'a>` type
explicitly. The latter, however, would require care to ensure that code could
not reference or alias the lifetime.

# Unresolved questions
[unresolved]: #unresolved-questions

When implementing this feature, we will need to determine whether the compiler
can provide an appropriate lifetime that prevents a `VaList` from outliving its
corresponding variadic function.

Currently, Rust does not allow passing a closure to C code expecting a pointer
to an `extern "C"` function. If this becomes possible in the future, then
variadic closures would become useful, and we should add them at that time.

This RFC only supports the platform's native `"C"` ABI, not any other ABI. Code
may wish to define variadic functions for another ABI, and potentially more
than one such ABI in the same program. However, such support should not
complicate the common case. LLVM has extremely limited support for this, for
only a specific pair of platforms (supporting the Windows ABI on platforms that
use the System V ABI), with no generalized support in the underlying
intrinsics. The LLVM intrinsics only support using the ABI of the containing
function. Given the current state of the ecosystem, this RFC only proposes
supporting the native `"C"` ABI for now. Doing so will not prevent the
introduction of support for non-native ABIs in the future.
