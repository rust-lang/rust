- Feature Name: variadic
- Start Date: 2017-08-21
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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
function.  The function must use `extern "C"`, and must use `unsafe`. To expose
such a function as a symbol for C code to call directly, the function may want
to use `#[no_mangle]` as well; however, Rust code may also pass the function to
C code expecting a function pointer to a variadic function.

The `args` named in the function declaration has the type
`core::intrinsics::VaList<'a>`, where the compiler supplies a lifetime `'a`
that prevents the arguments from outliving the variadic function.

To access the arguments, Rust provides the following public interfaces in
`core::intrinsics` (also available via `std::intrinsics`):

```rust
/// The argument list of a C-compatible variadic function, corresponding to the
/// underlying C `va_list`. Opaque.
pub struct VaList<'a>;

impl<'a> VaList<'a> {
    /// Extract the next argument from the argument list. T must have a type
    /// usable in an FFI interface.
    pub unsafe fn arg<T>(&mut self) -> T;
}

impl<'a> Clone for VaList<'a>;
```

The type returned from `VaList::arg` must have a type usable in an FFI
interface: `i8`, `i16`, `i32`, `i64`, `isize`, `u8`, `u16`, `u32`, `u64`,
`usize`, `f32`, `f64`, `*const _`, `*mut _`, a `#[repr(C)]` struct or union, or
a non-value-carrying enum with a `repr`-specified discriminant type.

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
variadic function.

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
functions, pass a clone of it instead.

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
used). The implementation of `VaList::arg` will call `va_arg`.  The
implementation of `Clone` for `VaList` wil call `va_copy`. The compiler will
ensure that a call to `va_end` occurs exactly once on every `VaList` at an
appropriate time, similar to drop semantics. (This may occur via an
implementation of `Drop`, but this must take into account the semantics of
passing a `VaList` to or from an `extern "C"` function.)

The compiler may need to handle the type `VaList` specially, in order to
provide the desired drop semantics at FFI boundaries. In particular, some
platforms pass `va_list` by value, and others pass it by pointer; the standard
leaves unspecified whether changes made in the called function appear in the
caller's copy. Rust must match the underlying C semantics, to allow passing
VaList values between C and Rust. To avoid forcing variadic functions to cope
with these platform-specific differences, the compiler should ensure that the
type behaves as if it has `Drop` semantics, and has `va_end` called on the
original `VaList` as well as any clones of it. For instance, passing a `VaList`
to a `vprintf` function as declared in this RFC semantically passes ownership
of that `VaList`, and prevents calling the `arg` function again in the caller,
but it remains the caller's responsibility to call `va_end`, so the compiler
needs to insert the appropriate drop glue at the FFI boundary.

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
type promotions, lifetimes, copies, and destruction.

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
