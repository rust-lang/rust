# Interacting with foreign code

One of Rust's aims, as a system programming language, is to
interoperate well with C code.

We'll start with an example. It's a bit bigger than usual, and
contains a number of new concepts. We'll go over it one piece at a
time.

This is a program that uses OpenSSL's `SHA1` function to compute the
hash of its first command-line argument, which it then converts to a
hexadecimal string and prints to standard output. If you have the
OpenSSL libraries installed, it should 'just work'.

    use std;
    import std::{vec, str};
    
    native "cdecl" mod ssl {
        fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
    }
    
    fn as_hex(data: [u8]) -> str {
        let acc = "";
        for byte in data { acc += #fmt("%02x", byte as uint); }
        ret acc;
    }

    fn sha1(data: str) -> str unsafe {
        let bytes = str::bytes(data);
        let hash = ssl::SHA1(vec::unsafe::to_ptr(bytes),
                             vec::len(bytes), std::ptr::null());
        ret as_hex(vec::unsafe::from_buf(hash, 20u));
    }
    
    fn main(args: [str]) {
        std::io::println(sha1(args[1]));
    }

## Native modules

Before we can call `SHA1`, we have to declare it. That is what this
part of the program is responsible for:

    native "cdecl" mod ssl {
        fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;
    }

A `native` module declaration tells the compiler that the program
should be linked with a library by that name, and that the given list
of functions are available in that library.

In this case, it'll change the name `ssl` to a shared library name in
a platform-specific way (`libssl.so` on Linux, for example), and link
that in. If you want the module to have a different name from the
actual library, you can say `native "cdecl" mod something = "ssl" {
... }`.

The `"cdecl"` word indicates the calling convention to use for
functions in this module. Most C libraries use cdecl as their calling
convention. You can also specify `"x86stdcall"` to use stdcall
instead.

FIXME: Mention c-stack variants? Are they going to change?

## Unsafe pointers

The native `SHA1` function is declared to take three arguments, and
return a pointer.

    fn SHA1(src: *u8, sz: uint, out: *u8) -> *u8;

When declaring the argument types to a foreign function, the Rust
compiler has no way to check whether your declaration is correct, so
you have to be careful. If you get the number or types of the
arguments wrong, you're likely to get a segmentation fault. Or,
probably even worse, your code will work on one platform, but break on
another.

In this case, `SHA1` is defined as taking two `unsigned char*`
arguments and one `unsigned long`. The rust equivalents are `*u8`
unsafe pointers and an `uint` (which, like `unsigned long`, is a
machine-word-sized type).

Unsafe pointers can be created through various functions in the
standard lib, usually with `unsafe` somewhere in their name. You can
dereference an unsafe pointer with `*` operator, but use
caution—unlike Rust's other pointer types, unsafe pointers are
completely unmanaged, so they might point at invalid memory, or be
null pointers.

## Unsafe blocks

The `sha1` function is the most obscure part of the program.

    fn sha1(data: str) -> str unsafe {
        let bytes = str::bytes(data);
        let hash = ssl::SHA1(vec::unsafe::to_ptr(bytes),
                             vec::len(bytes), std::ptr::null());
        ret as_hex(vec::unsafe::from_buf(hash, 20u));
    }

Firstly, what does the `unsafe` keyword at the top of the function
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

    unsafe fn kaboom() { log "I'm harmless!"; }

This function can only be called from an unsafe block or another
unsafe function.

## Pointer fiddling

The standard library defines a number of helper functions for dealing
with unsafe data, casting between types, and generally subverting
Rust's safety mechanisms.

Let's look at our `sha1` function again.

    let bytes = str::bytes(data);
    let hash = ssl::SHA1(vec::unsafe::to_ptr(bytes),
                         vec::len(bytes), std::ptr::null());
    ret as_hex(vec::unsafe::from_buf(hash, 20u));

The `str::bytes` function is perfectly safe, it converts a string to
an `[u8]`. This byte array is then fed to `vec::unsafe::to_ptr`, which
returns an unsafe pointer to its contents.

This pointer will become invalid as soon as the vector it points into
is cleaned up, so you should be very careful how you use it. In this
case, the local variable `bytes` outlives the pointer, so we're good.

Passing a null pointer as third argument to `SHA1` causes it to use a
static buffer, and thus save us the effort of allocating memory
ourselves. `ptr::null` is a generic function that will return an
unsafe null pointer of the correct type (Rust generics are awesome
like that—they can take the right form depending on the type that they
are expected to return).

Finally, `vec::unsafe::from_buf` builds up a new `[u8]` from the
unsafe pointer that was returned by `SHA1`. SHA1 digests are always
twenty bytes long, so we can pass `20u` for the length of the new
vector.

## Passing structures

C functions often take pointers to structs as arguments. Since Rust
records are binary-compatible with C structs, Rust programs can call
such functions directly.

This program uses the Posix function `gettimeofday` to get a
microsecond-resolution timer.

    use std;
    type timeval = {tv_sec: u32, tv_usec: u32};
    native "cdecl" mod libc = "" {
        fn gettimeofday(tv: *mutable timeval, tz: *()) -> i32;
    }
    fn unix_time_in_microseconds() -> u64 unsafe {
        let x = {tv_sec: 0u32, tv_usec: 0u32};
        libc::gettimeofday(std::ptr::addr_of(x), std::ptr::null());
        ret (x.tv_sec as u64) * 1000_000_u64 + (x.tv_usec as u64);
    }

The `libc = ""` sets the name of the native module to the empty string
to prevent the rust compiler from trying to link it. The standard C
library is already linked with Rust programs.

A `timeval`, in C, is a struct with two 32-bit integers. Thus, we
define a record type with the same contents, and declare
`gettimeofday` to take a pointer to such a record.

The second argument to `gettimeofday` (the time zone) is not used by
this program, so it simply declares it to be a pointer to the nil
type. Since null pointer look the same, no matter which type they are
supposed to point at, this is safe.
