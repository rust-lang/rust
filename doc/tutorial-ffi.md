% Rust Foreign Function Interface Tutorial

# Introduction

Because Rust is a systems programming language, one of its goals is to
interoperate well with C code.

We'll start with an example, which is a bit bigger than usual. We'll
go over it one piece at a time. This is a program that uses OpenSSL's
`SHA1` function to compute the hash of its first command-line
argument, which it then converts to a hexadecimal string and prints to
standard output. If you have the OpenSSL libraries installed, it
should compile and run without any extra effort.

~~~~ {.xfail-test}
extern mod std;
use core::libc::c_uint;

extern mod crypto {
    fn SHA1(src: *u8, sz: c_uint, out: *u8) -> *u8;
}

fn as_hex(data: ~[u8]) -> ~str {
    let mut acc = ~"";
    for data.each |&byte| { acc += fmt!("%02x", byte as uint); }
    return acc;
}

fn sha1(data: ~str) -> ~str {
    unsafe {
        let bytes = str::to_bytes(data);
        let hash = crypto::SHA1(vec::raw::to_ptr(bytes),
                                vec::len(bytes) as c_uint,
                                ptr::null());
        return as_hex(vec::from_buf(hash, 20));
    }
}

fn main() {
    io::println(sha1(core::os::args()[1]));
}
~~~~

# Foreign modules

Before we can call the `SHA1` function defined in the OpenSSL library, we have
to declare it. That is what this part of the program does:

~~~~ {.xfail-test}
extern mod crypto {
    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8; }
~~~~

An `extern` module declaration containing function signatures introduces the
functions listed as _foreign functions_. Foreign functions differ from regular
Rust functions in that they are implemented in some other language (usually C)
and called through Rust's foreign function interface (FFI). An extern module
like this is called a foreign module, and implicitly tells the compiler to
link with a library that contains the listed foreign functions, and has the
same name as the module.

In this case, the Rust compiler changes the name `crypto` to a shared library
name in a platform-specific way (`libcrypto.so` on Linux, for example),
searches for the shared library with that name, and links the library into the
program. If you want the module to have a different name from the actual
library, you can use the `"link_name"` attribute, like:

~~~~ {.xfail-test}
#[link_name = "crypto"]
extern mod something {
    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
}
~~~~

# Foreign calling conventions

Most foreign code is C code, which usually uses the `cdecl` calling
convention, so that is what Rust uses by default when calling foreign
functions. Some foreign functions, most notably the Windows API, use other
calling conventions. Rust provides the `"abi"` attribute as a way to hint to
the compiler which calling convention to use:

~~~~
#[cfg(target_os = "win32")]
#[abi = "stdcall"]
extern mod kernel32 {
    fn SetEnvironmentVariableA(n: *u8, v: *u8) -> int;
}
~~~~

The `"abi"` attribute applies to a foreign module (it cannot be applied
to a single function within a module), and must be either `"cdecl"`
or `"stdcall"`. We may extend the compiler in the future to support other
calling conventions.

# Unsafe pointers

The foreign `SHA1` function takes three arguments, and returns a pointer.

~~~~ {.xfail-test}
# extern mod crypto {
fn SHA1(src: *u8, sz: libc::c_uint, out: *u8) -> *u8;
# }
~~~~

When declaring the argument types to a foreign function, the Rust
compiler has no way to check whether your declaration is correct, so
you have to be careful. If you get the number or types of the
arguments wrong, you're likely to cause a segmentation fault. Or,
probably even worse, your code will work on one platform, but break on
another.

In this case, we declare that `SHA1` takes two `unsigned char*`
arguments and one `unsigned long`. The Rust equivalents are `*u8`
unsafe pointers and an `uint` (which, like `unsigned long`, is a
machine-word-sized type).

The standard library provides various functions to create unsafe pointers,
such as those in `core::cast`. Most of these functions have `unsafe` in their
name.  You can dereference an unsafe pointer with the `*` operator, but use
caution: unlike Rust's other pointer types, unsafe pointers are completely
unmanaged, so they might point at invalid memory, or be null pointers.

# Unsafe blocks

The `sha1` function is the most obscure part of the program.

~~~~
# pub mod crypto {
#   pub fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8 { out }
# }
# fn as_hex(data: ~[u8]) -> ~str { ~"hi" }
fn sha1(data: ~str) -> ~str {
    unsafe {
        let bytes = str::to_bytes(data);
        let hash = crypto::SHA1(vec::raw::to_ptr(bytes),
                                vec::len(bytes), ptr::null());
        return as_hex(vec::from_buf(hash, 20));
    }
}
~~~~

First, what does the `unsafe` keyword at the top of the function
mean? `unsafe` is a block modifier—it declares the block following it
to be known to be unsafe.

Some operations, like dereferencing unsafe pointers or calling
functions that have been marked unsafe, are only allowed inside unsafe
blocks. With the `unsafe` keyword, you're telling the compiler 'I know
what I'm doing'. The main motivation for such an annotation is that
when you have a memory error (and you will, if you're using unsafe
constructs), you have some idea where to look—it will most likely be
caused by some unsafe code.

Unsafe blocks isolate unsafety. Unsafe functions, on the other hand,
advertise it to the world. An unsafe function is written like this:

~~~~
unsafe fn kaboom() { ~"I'm harmless!"; }
~~~~

This function can only be called from an `unsafe` block or another
`unsafe` function.

# Pointer fiddling

The standard library defines a number of helper functions for dealing
with unsafe data, casting between types, and generally subverting
Rust's safety mechanisms.

Let's look at our `sha1` function again.

~~~~
# pub mod crypto {
#     pub fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8 { out }
# }
# fn as_hex(data: ~[u8]) -> ~str { ~"hi" }
# fn x(data: ~str) -> ~str {
# unsafe {
let bytes = str::to_bytes(data);
let hash = crypto::SHA1(vec::raw::to_ptr(bytes),
                        vec::len(bytes), ptr::null());
return as_hex(vec::from_buf(hash, 20));
# }
# }
~~~~

The `str::to_bytes` function is perfectly safe: it converts a string to a
`~[u8]`. The program then feeds this byte array to `vec::raw::to_ptr`, which
returns an unsafe pointer to its contents.

This pointer will become invalid at the end of the scope in which the vector
it points to (`bytes`) is valid, so you should be very careful how you use
it. In this case, the local variable `bytes` outlives the pointer, so we're
good.

Passing a null pointer as the third argument to `SHA1` makes it use a
static buffer, and thus save us the effort of allocating memory
ourselves. `ptr::null` is a generic function that, in this case, returns an
unsafe null pointer of type `*u8`. (Rust generics are awesome
like that: they can take the right form depending on the type that they
are expected to return.)

Finally, `vec::from_buf` builds up a new `~[u8]` from the
unsafe pointer that `SHA1` returned. SHA1 digests are always
twenty bytes long, so we can pass `20` for the length of the new
vector.

# Passing structures

C functions often take pointers to structs as arguments. Since Rust
`struct`s are binary-compatible with C structs, Rust programs can call
such functions directly.

This program uses the POSIX function `gettimeofday` to get a
microsecond-resolution timer.

~~~~
extern mod std;
use core::libc::c_ulonglong;

struct timeval {
    tv_sec: c_ulonglong,
    tv_usec: c_ulonglong
}

#[nolink]
extern mod lib_c {
    fn gettimeofday(tv: *mut timeval, tz: *()) -> i32;
}
fn unix_time_in_microseconds() -> u64 {
    unsafe {
        let mut x = timeval {
            tv_sec: 0 as c_ulonglong,
            tv_usec: 0 as c_ulonglong
        };
        lib_c::gettimeofday(&mut x, ptr::null());
        return (x.tv_sec as u64) * 1000_000_u64 + (x.tv_usec as u64);
    }
}

# fn main() { assert!(fmt!("%?", unix_time_in_microseconds()) != ~""); }
~~~~

The `#[nolink]` attribute indicates that there's no foreign library to
link in. The standard C library is already linked with Rust programs.

In C, a `timeval` is a struct with two 32-bit integer fields. Thus, we
define a `struct` type with the same contents, and declare
`gettimeofday` to take a pointer to such a `struct`.

This program does not use the second argument to `gettimeofday` (the time
 zone), so the `extern mod` declaration for it simply declares this argument
 to be a pointer to the unit type (written `()`). Since all null pointers have
 the same representation regardless of their referent type, this is safe.

