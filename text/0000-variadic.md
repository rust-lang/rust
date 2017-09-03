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
pub unsafe extern "C" fn func(arg: T, arg2: T2, ...) {
    // implementation
}
```

The `...` at the end of the argument list declares the function as variadic.
The function must use `extern "C"`, and must use `unsafe`. To expose
such a function as a symbol for C code to call directly, the function may want
to use `#[no_mangle]` as well; however, Rust code may also pass the function to
C code expecting a function pointer to a variadic function.

To access the arguments, Rust provides the following public interfaces in
`core::intrinsics` (also available via `std::intrinsics`):

```rust
/// The argument list of a C-compatible variadic function, corresponding to the
/// underlying C `va_list`. Opaque.
pub struct VaList<'a>;

impl<'a> VaList<'a> {
    /// Obtain the variable arguments of the current function. Produces a
    /// compile-time error if called from a non-variadic function. The compiler
    /// will supply the appropriate lifetime when called, and prevent that
    /// lifetime from outliving the variadic function.
    pub unsafe fn start() -> VaList<'a>;

    /// Extract the next argument from the argument list.
    pub unsafe fn<T: VaArg> arg(&mut self) -> T;
}

impl<'a> Clone for VaList<'a>;
impl<'a> Drop for VaList<'a>;

/// The type of arguments extractable from VaList
trait VaArg;

impl VaArg for i8;
impl VaArg for i16;
impl VaArg for i32;
impl VaArg for i64;
impl VaArg for isize;

impl VaArg for u8;
impl VaArg for u16;
impl VaArg for u32;
impl VaArg for u64;
impl VaArg for usize;

impl VaArg for f32;
impl VaArg for f64;

impl<T> VaArg for *const T;
impl<T> VaArg for *mut T;
```

The `libc` crate additionally provides implementations of the `VaArg` trait for
the raw C types corresponding to the Rust integer and float types above:

```rust
impl VaArg for c_char;
impl VaArg for c_schar;
impl VaArg for c_uchar;

impl VaArg for c_short;
impl VaArg for c_ushort;

impl VaArg for c_int;
impl VaArg for c_uint;

impl VaArg for c_long;
impl VaArg for c_ulong;

impl VaArg for c_longlong;
impl VaArg for c_ulonglong;

impl VaArg for c_float;
impl VaArg for c_double;

impl VaArg for int8_t;
impl VaArg for int16_t;
impl VaArg for int32_t;
impl VaArg for int64_t;

impl VaArg for uint8_t;
impl VaArg for uint16_t;
impl VaArg for uint32_t;
impl VaArg for uint64_t;

impl VaArg for size_t;
impl VaArg for ssize_t;
```

Note that extracting an argument from a `VaList` follows the platform-specific
rules for argument passing and promotion. In particular, many platforms promote
any argument smaller than a C `int` to an `int`. On such platforms, extracting
the corresponding type will extract an `int` and convert appropriately.

Like the underlying platform `va_list` structure in C, `VaList` has an opaque,
platform-specific representation.

A variadic function may call `VaList::start` more than once, and traverse the
argument list more than once.

A variadic function may pass the `VaList` to another function. However, it may
not return the `VaList` or otherwise allow it to outlive that call to the
variadic function.

A function declared with `extern "C"` may accept a `VaList` parameter,
corresponding to a `va_list` parameter in the corresponding C function. For
instance, the `libc` crate could define the `va_list` variants of `printf` as
follows:

```rust
pub unsafe extern "C" fn vprintf(format: *const c_char, ap: VaList) -> c_int;
pub unsafe extern "C" fn vfprintf(stream: *mut FILE, format: *const c_char, ap: VaList) -> c_int;
pub unsafe extern "C" fn vsprintf(s: *mut c_char, format: *const c_char, ap: VaList) -> c_int;
pub unsafe extern "C" fn vsnprintf(s: *mut c_char, n: size_t, format: *const c_char, ap: VaList) -> c_int;
```

Defining a variadic function, or calling any of these new functions, requires a
feature-gate, `c_variadic`.

Sample Rust code exposing a variadic function:

```rust
#![feature(c_variadic)]
use std::intrinsics::{VaArg, VaList};

#[no_mangle]
pub unsafe extern "C" fn func(fixed: u32, ...) {
    let args = VaList::start();
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
`va_end`, and `va_copy`. The implementation of `VaList::start` will call the
`va_start` intrinsic. The implementation of `VaList::arg` will call `va_arg`.
The implementation of `Clone` for `VaList` wil call `va_copy`. The
implementation of `Drop` for `VaList` wil call `va_end`.

This RFC intentionally does not specify the mechanism used to implement the
`VaArg` trait, as the compiler may need to natively implement `VaList::arg`
with appropriate understanding of platform-specific conventions. Code outside
of `core`, `std`, and `libc` may not implement this trait for any other type.

Note that on some platforms, these LLVM intrinsics do not fully implement the
necessary functionality, expecting the invoker of the intrinsic to provide
additional LLVM IR code. On such platforms, rustc will need to provide the
appropriate additional code, just as clang does.

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

Rather than defining a `VaList::start` function, we could require specifying a
name along with the `...`:

```rust
pub unsafe extern "C" fn func(fixed: u32, ...args) {
    // implementation
}
```

This might simplify the provision of an appropriate lifetime, and would avoid
the need to provide a `VaList::start` function and only allow calling it from
within a variadic function.

However, such an approach would not expose any means of calling `va_start`
multiple times in the same variadic function. Note that doing so has a
different semantic than calling `va_copy`, as calling `va_start` again iterates
over the arguments from the beginning rather than the current point. Given that
this mechanism exists for the sole purpose of interoperability with C, more
closely matching the underlying C interface seems appropriate.

# Unresolved questions
[unresolved]: #unresolved-questions

When implementing this feature, we will need to determine whether the compiler
can provide an appropriate lifetime that prevents a `VaList` from outliving its
corresponding variadic function.

Currently, Rust does not allow passing a closure to C code expecting a pointer
to an `extern "C"` function. If this becomes possible in the future, then
variadic closures would become useful, and we should add them at that time.
