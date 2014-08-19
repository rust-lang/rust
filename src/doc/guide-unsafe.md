% Writing Unsafe and Low-Level Code in Rust

# Introduction

Rust aims to provide safe abstractions over the low-level details of
the CPU and operating system, but sometimes one needs to drop down and
write code at that level. This guide aims to provide an overview of
the dangers and power one gets with Rust's unsafe subset.

Rust provides an escape hatch in the form of the `unsafe { ... }`
block which allows the programmer to dodge some of the compiler's
checks and do a wide range of operations, such as:

- dereferencing [raw pointers](#raw-pointers)
- calling a function via FFI ([covered by the FFI guide](guide-ffi.html))
- casting between types bitwise (`transmute`, aka "reinterpret cast")
- [inline assembly](#inline-assembly)

Note that an `unsafe` block does not relax the rules about lifetimes
of `&` and the freezing of borrowed data.

Any use of `unsafe` is the programmer saying "I know more than you" to
the compiler, and, as such, the programmer should be very sure that
they actually do know more about why that piece of code is valid.  In
general, one should try to minimize the amount of unsafe code in a
code base; preferably by using the bare minimum `unsafe` blocks to
build safe interfaces.

> **Note**: the low-level details of the Rust language are still in
> flux, and there is no guarantee of stability or backwards
> compatibility. In particular, there may be changes that do not cause
> compilation errors, but do cause semantic changes (such as invoking
> undefined behaviour). As such, extreme care is required.

# Pointers

## References

One of Rust's biggest features is memory safety.  This is achieved in
part via [the lifetime system](guide-lifetimes.html), which is how the
compiler can guarantee that every `&` reference is always valid, and,
for example, never pointing to freed memory.

These restrictions on `&` have huge advantages. However, they also
constrain how we can use them. For example, `&` doesn't behave
identically to C's pointers, and so cannot be used for pointers in
foreign function interfaces (FFI). Additionally, both immutable (`&`)
and mutable (`&mut`) references have some aliasing and freezing
guarantees, required for memory safety.

In particular, if you have an `&T` reference, then the `T` must not be
modified through that reference or any other reference. There are some
standard library types, e.g. `Cell` and `RefCell`, that provide inner
mutability by replacing compile time guarantees with dynamic checks at
runtime.

An `&mut` reference has a different constraint: when an object has an
`&mut T` pointing into it, then that `&mut` reference must be the only
such usable path to that object in the whole program. That is, an
`&mut` cannot alias with any other references.

Using `unsafe` code to incorrectly circumvent and violate these
restrictions is undefined behaviour. For example, the following
creates two aliasing `&mut` pointers, and is invalid.

```
use std::mem;
let mut x: u8 = 1;

let ref_1: &mut u8 = &mut x;
let ref_2: &mut u8 = unsafe { mem::transmute(&mut *ref_1) };

// oops, ref_1 and ref_2 point to the same piece of data (x) and are
// both usable
*ref_1 = 10;
*ref_2 = 20;
```

## Raw pointers

Rust offers two additional pointer types "raw pointers", written as
`*const T` and `*mut T`. They're an approximation of C's `const T*` and `T*`
respectively; indeed, one of their most common uses is for FFI,
interfacing with external C libraries.

Raw pointers have much fewer guarantees than other pointer types
offered by the Rust language and libraries. For example, they

- are not guaranteed to point to valid memory and are not even
  guaranteed to be non-null (unlike both `Box` and `&`);
- do not have any automatic clean-up, unlike `Box`, and so require
  manual resource management;
- are plain-old-data, that is, they don't move ownership, again unlike
  `Box`, hence the Rust compiler cannot protect against bugs like
  use-after-free;
- are considered sendable (if their contents is considered sendable),
  so the compiler offers no assistance with ensuring their use is
  thread-safe; for example, one can concurrently access a `*mut int`
  from two threads without synchronization.
- lack any form of lifetimes, unlike `&`, and so the compiler cannot
  reason about dangling pointers; and
- have no guarantees about aliasing or mutability other than mutation
  not being allowed directly through a `*const T`.

Fortunately, they come with a redeeming feature: the weaker guarantees
mean weaker restrictions. The missing restrictions make raw pointers
appropriate as a building block for implementing things like smart
pointers and vectors inside libraries. For example, `*` pointers are
allowed to alias, allowing them to be used to write shared-ownership
types like reference counted and garbage collected pointers, and even
thread-safe shared memory types (`Rc` and the `Arc` types are both
implemented entirely in Rust).

There are two things that you are required to be careful about
(i.e. require an `unsafe { ... }` block) with raw pointers:

- dereferencing: they can have any value: so possible results include
  a crash, a read of uninitialised memory, a use-after-free, or
  reading data as normal.
- pointer arithmetic via the `offset` [intrinsic](#intrinsics) (or
  `.offset` method): this intrinsic uses so-called "in-bounds"
  arithmetic, that is, it is only defined behaviour if the result is
  inside (or one-byte-past-the-end) of the object from which the
  original pointer came.

The latter assumption allows the compiler to optimize more
effectively. As can be seen, actually *creating* a raw pointer is not
unsafe, and neither is converting to an integer.

### References and raw pointers

At runtime, a raw pointer `*` and a reference pointing to the same
piece of data have an identical representation. In fact, an `&T`
reference will implicitly coerce to an `*const T` raw pointer in safe code
and similarly for the `mut` variants (both coercions can be performed
explicitly with, respectively, `value as *const T` and `value as *mut T`).

Going the opposite direction, from `*const` to a reference `&`, is not
safe. A `&T` is always valid, and so, at a minimum, the raw pointer
`*const T` has to point to a valid instance of type `T`. Furthermore,
the resulting pointer must satisfy the aliasing and mutability laws of
references. The compiler assumes these properties are true for any
references, no matter how they are created, and so any conversion from
raw pointers is asserting that they hold. The programmer *must*
guarantee this.

The recommended method for the conversion is

```
let i: u32 = 1;
// explicit cast
let p_imm: *const u32 = &i as *const u32;
let mut m: u32 = 2;
// implicit coercion
let p_mut: *mut u32 = &mut m;

unsafe {
    let ref_imm: &u32 = &*p_imm;
    let ref_mut: &mut u32 = &mut *p_mut;
}
```

The `&*x` dereferencing style is preferred to using a `transmute`.
The latter is far more powerful than necessary, and the more
restricted operation is harder to use incorrectly; for example, it
requires that `x` is a pointer (unlike `transmute`).



## Making the unsafe safe(r)

There are various ways to expose a safe interface around some unsafe
code:

- store pointers privately (i.e. not in public fields of public
  structs), so that you can see and control all reads and writes to
  the pointer in one place.
- use `assert!()` a lot: since you can't rely on the protection of the
  compiler & type-system to ensure that your `unsafe` code is correct
  at compile-time, use `assert!()` to verify that it is doing the
  right thing at run-time.
- implement the `Drop` for resource clean-up via a destructor, and use
  RAII (Resource Acquisition Is Initialization). This reduces the need
  for any manual memory management by users, and automatically ensures
  that clean-up is always run, even when the task fails.
- ensure that any data stored behind a raw pointer is destroyed at the
  appropriate time.

As an example, we give a reimplementation of owned boxes by wrapping
`malloc` and `free`. Rust's move semantics and lifetimes mean this
reimplementation is as safe as the `Box` type.

```
#![feature(unsafe_destructor)]

extern crate libc;
use libc::{c_void, size_t, malloc, free};
use std::mem;
use std::ptr;

// Define a wrapper around the handle returned by the foreign code.
// Unique<T> has the same semantics as Box<T>
pub struct Unique<T> {
    // It contains a single raw, mutable pointer to the object in question.
    ptr: *mut T
}

// Implement methods for creating and using the values in the box.

// NB: For simplicity and correctness, we require that T has kind Send
// (owned boxes relax this restriction, and can contain managed (GC) boxes).
// This is because, as implemented, the garbage collector would not know
// about any shared boxes stored in the malloc'd region of memory.
impl<T: Send> Unique<T> {
    pub fn new(value: T) -> Unique<T> {
        unsafe {
            let ptr = malloc(mem::size_of::<T>() as size_t) as *mut T;
            // we *need* valid pointer.
            assert!(!ptr.is_null());
            // `*ptr` is uninitialized, and `*ptr = value` would
            // attempt to destroy it `overwrite` moves a value into
            // this memory without attempting to drop the original
            // value.
            ptr::write(&mut *ptr, value);
            Unique{ptr: ptr}
        }
    }

    // the 'r lifetime results in the same semantics as `&*x` with
    // Box<T>
    pub fn borrow<'r>(&'r self) -> &'r T {
        // By construction, self.ptr is valid
        unsafe { &*self.ptr }
    }

    // the 'r lifetime results in the same semantics as `&mut *x` with
    // Box<T>
    pub fn borrow_mut<'r>(&'r mut self) -> &'r mut T {
        unsafe { &mut *self.ptr }
    }
}

// A key ingredient for safety, we associate a destructor with
// Unique<T>, making the struct manage the raw pointer: when the
// struct goes out of scope, it will automatically free the raw pointer.
//
// NB: This is an unsafe destructor, because rustc will not normally
// allow destructors to be associated with parameterized types, due to
// bad interaction with managed boxes. (With the Send restriction,
// we don't have this problem.) Note that the `#[unsafe_destructor]`
// feature gate is required to use unsafe destructors.
#[unsafe_destructor]
impl<T: Send> Drop for Unique<T> {
    fn drop(&mut self) {
        unsafe {
            // Copy the object out from the pointer onto the stack,
            // where it is covered by normal Rust destructor semantics
            // and cleans itself up, if necessary
            ptr::read(self.ptr as *const T);

            // clean-up our allocation
            free(self.ptr as *mut c_void)
        }
    }
}

// A comparison between the built-in `Box` and this reimplementation
fn main() {
    {
        let mut x = box 5i;
        *x = 10;
    } // `x` is freed here

    {
        let mut y = Unique::new(5i);
        *y.borrow_mut() = 10;
    } // `y` is freed here
}
```

Notably, the only way to construct a `Unique` is via the `new`
function, and this function ensures that the internal pointer is valid
and hidden in the private field. The two `borrow` methods are safe
because the compiler statically guarantees that objects are never used
before creation or after destruction (unless you use some `unsafe`
code...).

# Inline assembly

For extremely low-level manipulations and performance reasons, one
might wish to control the CPU directly. Rust supports using inline
assembly to do this via the `asm!` macro. The syntax roughly matches
that of GCC & Clang:

```ignore
asm!(assembly template
   : output operands
   : input operands
   : clobbers
   : options
   );
```

Any use of `asm` is feature gated (requires `#![feature(asm)]` on the
crate to allow) and of course requires an `unsafe` block.

> **Note**: the examples here are given in x86/x86-64 assembly, but
> all platforms are supported.

## Assembly template

The `assembly template` is the only required parameter and must be a
literal string (i.e `""`)

```
#![feature(asm)]

#[cfg(target_arch = "x86")]
#[cfg(target_arch = "x86_64")]
fn foo() {
    unsafe {
        asm!("NOP");
    }
}

// other platforms
#[cfg(not(target_arch = "x86"),
      not(target_arch = "x86_64"))]
fn foo() { /* ... */ }

fn main() {
    // ...
    foo();
    // ...
}
```

(The `feature(asm)` and `#[cfg]`s are omitted from now on.)

Output operands, input operands, clobbers and options are all optional
but you must add the right number of `:` if you skip them:

```
# #![feature(asm)]
# #[cfg(target_arch = "x86")] #[cfg(target_arch = "x86_64")]
# fn main() { unsafe {
asm!("xor %eax, %eax"
    :
    :
    : "eax"
   );
# } }
```

Whitespace also doesn't matter:

```
# #![feature(asm)]
# #[cfg(target_arch = "x86")] #[cfg(target_arch = "x86_64")]
# fn main() { unsafe {
asm!("xor %eax, %eax" ::: "eax");
# } }
```

## Operands

Input and output operands follow the same format: `:
"constraints1"(expr1), "constraints2"(expr2), ..."`. Output operand
expressions must be mutable lvalues:

```
# #![feature(asm)]
# #[cfg(target_arch = "x86")] #[cfg(target_arch = "x86_64")]
fn add(a: int, b: int) -> int {
    let mut c = 0;
    unsafe {
        asm!("add $2, $0"
             : "=r"(c)
             : "0"(a), "r"(b)
             );
    }
    c
}
# #[cfg(not(target_arch = "x86"), not(target_arch = "x86_64"))]
# fn add(a: int, b: int) -> int { a + b }

fn main() {
    assert_eq!(add(3, 14159), 14162)
}
```

## Clobbers

Some instructions modify registers which might otherwise have held
different values so we use the clobbers list to indicate to the
compiler not to assume any values loaded into those registers will
stay valid.

```
# #![feature(asm)]
# #[cfg(target_arch = "x86")] #[cfg(target_arch = "x86_64")]
# fn main() { unsafe {
// Put the value 0x200 in eax
asm!("mov $$0x200, %eax" : /* no outputs */ : /* no inputs */ : "eax");
# } }
```

Input and output registers need not be listed since that information
is already communicated by the given constraints. Otherwise, any other
registers used either implicitly or explicitly should be listed.

If the assembly changes the condition code register `cc` should be
specified as one of the clobbers. Similarly, if the assembly modifies
memory, `memory` should also be specified.

## Options

The last section, `options` is specific to Rust. The format is comma
separated literal strings (i.e `:"foo", "bar", "baz"`). It's used to
specify some extra info about the inline assembly:

Current valid options are:

1. **volatile** - specifying this is analogous to `__asm__ __volatile__ (...)` in gcc/clang.
2. **alignstack** - certain instructions expect the stack to be
   aligned a certain way (i.e SSE) and specifying this indicates to
   the compiler to insert its usual stack alignment code
3. **intel** - use intel syntax instead of the default AT&T.

# Avoiding the standard library

By default, `std` is linked to every Rust crate. In some contexts,
this is undesirable, and can be avoided with the `#![no_std]`
attribute attached to the crate.

```ignore
// a minimal library
#![crate_type="lib"]
#![no_std]
# // fn main() {} tricked you, rustdoc!
```

Obviously there's more to life than just libraries: one can use
`#[no_std]` with an executable, controlling the entry point is
possible in two ways: the `#[start]` attribute, or overriding the
default shim for the C `main` function with your own.

The function marked `#[start]` is passed the command line parameters
in the same format as C:

```
#![no_std]
#![feature(lang_items)]

// Pull in the system libc library for what crt0.o likely requires
extern crate libc;

// Entry point for this program
#[start]
fn start(_argc: int, _argv: *const *const u8) -> int {
    0
}

// These functions are invoked by the compiler, but not
// for a bare-bones hello world. These are normally
// provided by libstd.
#[lang = "stack_exhausted"] extern fn stack_exhausted() {}
#[lang = "eh_personality"] extern fn eh_personality() {}
# // fn main() {} tricked you, rustdoc!
```

To override the compiler-inserted `main` shim, one has to disable it
with `#![no_main]` and then create the appropriate symbol with the
correct ABI and the correct name, which requires overriding the
compiler's name mangling too:

```ignore
#![no_std]
#![no_main]
#![feature(lang_items)]

extern crate libc;

#[no_mangle] // ensure that this symbol is called `main` in the output
pub extern fn main(argc: int, argv: *const *const u8) -> int {
    0
}

#[lang = "stack_exhausted"] extern fn stack_exhausted() {}
#[lang = "eh_personality"] extern fn eh_personality() {}
# // fn main() {} tricked you, rustdoc!
```


The compiler currently makes a few assumptions about symbols which are available
in the executable to call. Normally these functions are provided by the standard
library, but without it you must define your own.

The first of these two functions, `stack_exhausted`, is invoked whenever stack
overflow is detected.  This function has a number of restrictions about how it
can be called and what it must do, but if the stack limit register is not being
maintained then a task always has an "infinite stack" and this function
shouldn't get triggered.

The second of these two functions, `eh_personality`, is used by the failure
mechanisms of the compiler. This is often mapped to GCC's personality function
(see the [libstd implementation](std/rt/unwind/index.html) for more
information), but crates which do not trigger failure can be assured that this
function is never called.

## Using libcore

> **Note**: the core library's structure is unstable, and it is recommended to
> use the standard library instead wherever possible.

With the above techniques, we've got a bare-metal executable running some Rust
code. There is a good deal of functionality provided by the standard library,
however, that is necessary to be productive in Rust. If the standard library is
not sufficient, then [libcore](core/index.html) is designed to be used
instead.

The core library has very few dependencies and is much more portable than the
standard library itself. Additionally, the core library has most of the
necessary functionality for writing idiomatic and effective Rust code.

As an example, here is a program that will calculate the dot product of two
vectors provided from C, using idiomatic Rust practices.

```
#![no_std]
#![feature(globs)]
#![feature(lang_items)]

# extern crate libc;
extern crate core;

use core::prelude::*;

use core::mem;

#[no_mangle]
pub extern fn dot_product(a: *const u32, a_len: u32,
                          b: *const u32, b_len: u32) -> u32 {
    use core::raw::Slice;

    // Convert the provided arrays into Rust slices.
    // The core::raw module guarantees that the Slice
    // structure has the same memory layout as a &[T]
    // slice.
    //
    // This is an unsafe operation because the compiler
    // cannot tell the pointers are valid.
    let (a_slice, b_slice): (&[u32], &[u32]) = unsafe {
        mem::transmute((
            Slice { data: a, len: a_len as uint },
            Slice { data: b, len: b_len as uint },
        ))
    };

    // Iterate over the slices, collecting the result
    let mut ret = 0;
    for (i, j) in a_slice.iter().zip(b_slice.iter()) {
        ret += (*i) * (*j);
    }
    return ret;
}

#[lang = "begin_unwind"]
extern fn begin_unwind(args: &core::fmt::Arguments,
                       file: &str,
                       line: uint) -> ! {
    loop {}
}

#[lang = "stack_exhausted"] extern fn stack_exhausted() {}
#[lang = "eh_personality"] extern fn eh_personality() {}
# #[start] fn start(argc: int, argv: *const *const u8) -> int { 0 }
# fn main() {}
```

Note that there is one extra lang item here which differs from the examples
above, `begin_unwind`. This must be defined by consumers of libcore because the
core library declares failure, but it does not define it. The `begin_unwind`
lang item is this crate's definition of failure, and it must be guaranteed to
never return.

As can be seen in this example, the core library is intended to provide the
power of Rust in all circumstances, regardless of platform requirements. Further
libraries, such as liballoc, add functionality to libcore which make other
platform-specific assumptions, but continue to be more portable than the
standard library itself.

# Interacting with the compiler internals

> **Note**: this section is specific to the `rustc` compiler; these
> parts of the language may never be fully specified and so details may
> differ wildly between implementations (and even versions of `rustc`
> itself).
>
> Furthermore, this is just an overview; the best form of
> documentation for specific instances of these features are their
> definitions and uses in `std`.

The Rust language currently has two orthogonal mechanisms for allowing
libraries to interact directly with the compiler and vice versa:

- intrinsics, functions built directly into the compiler providing
  very basic low-level functionality,
- lang-items, special functions, types and traits in libraries marked
  with specific `#[lang]` attributes

## Intrinsics

> **Note**: intrinsics will forever have an unstable interface, it is
> recommended to use the stable interfaces of libcore rather than intrinsics
> directly.

These are imported as if they were FFI functions, with the special
`rust-intrinsic` ABI. For example, if one was in a freestanding
context, but wished to be able to `transmute` between types, and
perform efficient pointer arithmetic, one would import those functions
via a declaration like

```
# #![feature(intrinsics)]
# fn main() {}

extern "rust-intrinsic" {
    fn transmute<T, U>(x: T) -> U;

    fn offset<T>(dst: *const T, offset: int) -> *const T;
}
```

As with any other FFI functions, these are always `unsafe` to call.

## Lang items

> **Note**: lang items are often provided by crates in the Rust distribution,
> and lang items themselves have an unstable interface. It is recommended to use
> officially distributed crates instead of defining your own lang items.

The `rustc` compiler has certain pluggable operations, that is,
functionality that isn't hard-coded into the language, but is
implemented in libraries, with a special marker to tell the compiler
it exists. The marker is the attribute `#[lang="..."]` and there are
various different values of `...`, i.e. various different "lang
items".

For example, `Box` pointers require two lang items, one for allocation
and one for deallocation. A freestanding program that uses the `Box`
sugar for dynamic allocations via `malloc` and `free`:

```
#![no_std]
#![feature(lang_items)]

extern crate libc;

extern {
    fn abort() -> !;
}

#[lang="exchange_malloc"]
unsafe fn allocate(size: uint, _align: uint) -> *mut u8 {
    let p = libc::malloc(size as libc::size_t) as *mut u8;

    // malloc failed
    if p as uint == 0 {
        abort();
    }

    p
}
#[lang="exchange_free"]
unsafe fn deallocate(ptr: *mut u8, _size: uint, _align: uint) {
    libc::free(ptr as *mut libc::c_void)
}

#[start]
fn main(argc: int, argv: *const *const u8) -> int {
    let x = box 1i;

    0
}

#[lang = "stack_exhausted"] extern fn stack_exhausted() {}
#[lang = "eh_personality"] extern fn eh_personality() {}
```

Note the use of `abort`: the `exchange_malloc` lang item is assumed to
return a valid pointer, and so needs to do the check internally.

Other features provided by lang items include:

- overloadable operators via traits: the traits corresponding to the
  `==`, `<`, dereferencing (`*`) and `+` (etc.) operators are all
  marked with lang items; those specific four are `eq`, `ord`,
  `deref`, and `add` respectively.
- stack unwinding and general failure; the `eh_personality`, `fail_`
  and `fail_bounds_checks` lang items.
- the traits in `std::kinds` used to indicate types that satisfy
  various kinds; lang items `send`, `sync` and `copy`.
- the marker types and variance indicators found in
  `std::kinds::markers`; lang items `covariant_type`,
  `contravariant_lifetime`, `no_sync_bound`, etc.

Lang items are loaded lazily by the compiler; e.g. if one never uses
`Box` then there is no need to define functions for `exchange_malloc`
and `exchange_free`. `rustc` will emit an error when an item is needed
but not found in the current crate or any that it depends on.
