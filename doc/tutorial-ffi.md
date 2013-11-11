% Rust Foreign Function Interface Tutorial

# Introduction

This tutorial will use the [snappy](https://code.google.com/p/snappy/)
compression/decompression library as an introduction to writing bindings for
foreign code. Rust is currently unable to call directly into a C++ library, but
snappy includes a C interface (documented in
[`snappy-c.h`](https://code.google.com/p/snappy/source/browse/trunk/snappy-c.h)).

The following is a minimal example of calling a foreign function which will compile if snappy is
installed:

~~~~ {.xfail-test}
use std::libc::size_t;

#[link_args = "-lsnappy"]
extern {
    fn snappy_max_compressed_length(source_length: size_t) -> size_t;
}

fn main() {
    let x = unsafe { snappy_max_compressed_length(100) };
    println!("max compressed length of a 100 byte buffer: {}", x);
}
~~~~

The `extern` block is a list of function signatures in a foreign library, in this case with the
platform's C ABI. The `#[link_args]` attribute is used to instruct the linker to link against the
snappy library so the symbols are resolved.

Foreign functions are assumed to be unsafe so calls to them need to be wrapped with `unsafe {}` as a
promise to the compiler that everything contained within truly is safe. C libraries often expose
interfaces that aren't thread-safe, and almost any function that takes a pointer argument isn't
valid for all possible inputs since the pointer could be dangling, and raw pointers fall outside of
Rust's safe memory model.

When declaring the argument types to a foreign function, the Rust compiler will not check if the
declaration is correct, so specifying it correctly is part of keeping the binding correct at
runtime.

The `extern` block can be extended to cover the entire snappy API:

~~~~ {.xfail-test}
use std::libc::{c_int, size_t};

#[link_args = "-lsnappy"]
extern {
    fn snappy_compress(input: *u8,
                       input_length: size_t,
                       compressed: *mut u8,
                       compressed_length: *mut size_t) -> c_int;
    fn snappy_uncompress(compressed: *u8,
                         compressed_length: size_t,
                         uncompressed: *mut u8,
                         uncompressed_length: *mut size_t) -> c_int;
    fn snappy_max_compressed_length(source_length: size_t) -> size_t;
    fn snappy_uncompressed_length(compressed: *u8,
                                  compressed_length: size_t,
                                  result: *mut size_t) -> c_int;
    fn snappy_validate_compressed_buffer(compressed: *u8,
                                         compressed_length: size_t) -> c_int;
}
~~~~

# Creating a safe interface

The raw C API needs to be wrapped to provide memory safety and make use of higher-level concepts
like vectors. A library can choose to expose only the safe, high-level interface and hide the unsafe
internal details.

Wrapping the functions which expect buffers involves using the `vec::raw` module to manipulate Rust
vectors as pointers to memory. Rust's vectors are guaranteed to be a contiguous block of memory. The
length is number of elements currently contained, and the capacity is the total size in elements of
the allocated memory. The length is less than or equal to the capacity.

~~~~ {.xfail-test}
pub fn validate_compressed_buffer(src: &[u8]) -> bool {
    unsafe {
        snappy_validate_compressed_buffer(vec::raw::to_ptr(src), src.len() as size_t) == 0
    }
}
~~~~

The `validate_compressed_buffer` wrapper above makes use of an `unsafe` block, but it makes the
guarantee that calling it is safe for all inputs by leaving off `unsafe` from the function
signature.

The `snappy_compress` and `snappy_uncompress` functions are more complex, since a buffer has to be
allocated to hold the output too.

The `snappy_max_compressed_length` function can be used to allocate a vector with the maximum
required capacity to hold the compressed output. The vector can then be passed to the
`snappy_compress` function as an output parameter. An output parameter is also passed to retrieve
the true length after compression for setting the length.

~~~~ {.xfail-test}
pub fn compress(src: &[u8]) -> ~[u8] {
    unsafe {
        let srclen = src.len() as size_t;
        let psrc = vec::raw::to_ptr(src);

        let mut dstlen = snappy_max_compressed_length(srclen);
        let mut dst = vec::with_capacity(dstlen as uint);
        let pdst = vec::raw::to_mut_ptr(dst);

        snappy_compress(psrc, srclen, pdst, &mut dstlen);
        vec::raw::set_len(&mut dst, dstlen as uint);
        dst
    }
}
~~~~

Decompression is similar, because snappy stores the uncompressed size as part of the compression
format and `snappy_uncompressed_length` will retrieve the exact buffer size required.

~~~~ {.xfail-test}
pub fn uncompress(src: &[u8]) -> Option<~[u8]> {
    unsafe {
        let srclen = src.len() as size_t;
        let psrc = vec::raw::to_ptr(src);

        let mut dstlen: size_t = 0;
        snappy_uncompressed_length(psrc, srclen, &mut dstlen);

        let mut dst = vec::with_capacity(dstlen as uint);
        let pdst = vec::raw::to_mut_ptr(dst);

        if snappy_uncompress(psrc, srclen, pdst, &mut dstlen) == 0 {
            vec::raw::set_len(&mut dst, dstlen as uint);
            Some(dst)
        } else {
            None // SNAPPY_INVALID_INPUT
        }
    }
}
~~~~

For reference, the examples used here are also available as an [library on
GitHub](https://github.com/thestinger/rust-snappy).

# Stack management

Rust tasks by default run on a "large stack". This is actually implemented as a
reserving a large segment of the address space and then lazily mapping in pages
as they are needed. When calling an external C function, the code is invoked on
the same stack as the rust stack. This means that there is no extra
stack-switching mechanism in place because it is assumed that the large stack
for the rust task is plenty for the C function to have.

A planned future improvement (net yet implemented at the time of this writing)
is to have a guard page at the end of every rust stack. No rust function will
hit this guard page (due to rust's usage of LLVM's __morestack). The intention
for this unmapped page is to prevent infinite recursion in C from overflowing
onto other rust stacks. If the guard page is hit, then the process will be
terminated with a message saying that the guard page was hit.

For normal external function usage, this all means that there shouldn't be any
need for any extra effort on a user's perspective. The C stack naturally
interleaves with the rust stack, and it's "large enough" for both to
interoperate. If, however, it is determined that a larger stack is necessary,
there are appropriate functions in the task spawning API to control the size of
the stack of the task which is spawned.

# Destructors

Foreign libraries often hand off ownership of resources to the calling code,
which should be wrapped in a destructor to provide safety and guarantee their
release.

A type with the same functionality as owned boxes can be implemented by
wrapping `malloc` and `free`:

~~~~
use std::cast;
use std::libc::{c_void, size_t, malloc, free};
use std::ptr;
use std::unstable::intrinsics;

// a wrapper around the handle returned by the foreign code
pub struct Unique<T> {
    priv ptr: *mut T
}

impl<T: Send> Unique<T> {
    pub fn new(value: T) -> Unique<T> {
        unsafe {
            let ptr = malloc(std::mem::size_of::<T>() as size_t) as *mut T;
            assert!(!ptr::is_null(ptr));
            // `*ptr` is uninitialized, and `*ptr = value` would attempt to destroy it
            intrinsics::move_val_init(&mut *ptr, value);
            Unique{ptr: ptr}
        }
    }

    // the 'r lifetime results in the same semantics as `&*x` with ~T
    pub fn borrow<'r>(&'r self) -> &'r T {
        unsafe { cast::copy_lifetime(self, &*self.ptr) }
    }

    // the 'r lifetime results in the same semantics as `&mut *x` with ~T
    pub fn borrow_mut<'r>(&'r mut self) -> &'r mut T {
        unsafe { cast::copy_mut_lifetime(self, &mut *self.ptr) }
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Unique<T> {
    fn drop(&mut self) {
        unsafe {
            let x = intrinsics::init(); // dummy value to swap in
            // moving the object out is needed to call the destructor
            ptr::replace_ptr(self.ptr, x);
            free(self.ptr as *c_void)
        }
    }
}

// A comparison between the built-in ~ and this reimplementation
fn main() {
    {
        let mut x = ~5;
        *x = 10;
    } // `x` is freed here

    {
        let mut y = Unique::new(5);
        *y.borrow_mut() = 10;
    } // `y` is freed here
}
~~~~

# Linking

In addition to the `#[link_args]` attribute for explicitly passing arguments to the linker, an
`extern mod` block will pass `-lmodname` to the linker by default unless it has a `#[nolink]`
attribute applied.

# Unsafe blocks

Some operations, like dereferencing unsafe pointers or calling functions that have been marked
unsafe are only allowed inside unsafe blocks. Unsafe blocks isolate unsafety and are a promise to
the compiler that the unsafety does not leak out of the block.

Unsafe functions, on the other hand, advertise it to the world. An unsafe function is written like
this:

~~~~
unsafe fn kaboom(ptr: *int) -> int { *ptr }
~~~~

This function can only be called from an `unsafe` block or another `unsafe` function.

# Accessing foreign globals

Foreign APIs often export a global variable which could do something like track
global state. In order to access these variables, you declare them in `extern`
blocks with the `static` keyword:

~~~{.xfail-test}
use std::libc;

#[link_args = "-lreadline"]
extern {
    static rl_readline_version: libc::c_int;
}

fn main() {
    println!("You have readline version {} installed.",
             rl_readline_version as int);
}
~~~

Alternatively, you may need to alter global state provided by a foreign
interface. To do this, statics can be declared with `mut` so rust can mutate
them.

~~~{.xfail-test}
use std::libc;
use std::ptr;

#[link_args = "-lreadline"]
extern {
    static mut rl_prompt: *libc::c_char;
}

fn main() {
    do "[my-awesome-shell] $".as_c_str |buf| {
        unsafe { rl_prompt = buf; }
        // get a line, process it
        unsafe { rl_prompt = ptr::null(); }
    }
}
~~~

# Foreign calling conventions

Most foreign code exposes a C ABI, and Rust uses the platform's C calling convention by default when
calling foreign functions. Some foreign functions, most notably the Windows API, use other calling
conventions. Rust provides a way to tell the compiler which convention to use:

~~~~
#[cfg(target_os = "win32", target_arch = "x86")]
#[link_name = "kernel32"]
extern "stdcall" {
    fn SetEnvironmentVariableA(n: *u8, v: *u8) -> int;
}
~~~~

This applies to the entire `extern` block. The list of supported ABI constraints
are:

* `stdcall`
* `aapcs`
* `cdecl`
* `fastcall`
* `Rust`
* `rust-intrinsic`
* `system`
* `C`

Most of the abis in this list are self-explanatory, but the `system` abi may
seem a little odd. This constraint selects whatever the appropriate ABI is for
interoperating with the target's libraries. For example, on win32 with a x86
architecture, this means that the abi used would be `stdcall`. On x86_64,
however, windows uses the `C` calling convention, so `C` would be used. This
means that in our previous example, we could have used `extern "system" { ... }`
to define a block for all windows systems, not just x86 ones.

# Interoperability with foreign code

Rust guarantees that the layout of a `struct` is compatible with the platform's representation in C.
A `#[packed]` attribute is available, which will lay out the struct members without padding.
However, there are currently no guarantees about the layout of an `enum`.

Rust's owned and managed boxes use non-nullable pointers as handles which point to the contained
object. However, they should not be manually created because they are managed by internal
allocators. Borrowed pointers can safely be assumed to be non-nullable pointers directly to the
type. However, breaking the borrow checking or mutability rules is not guaranteed to be safe, so
prefer using raw pointers (`*`) if that's needed because the compiler can't make as many assumptions
about them.

Vectors and strings share the same basic memory layout, and utilities are available in the `vec` and
`str` modules for working with C APIs. However, strings are not terminated with `\0`. If you need a
NUL-terminated string for interoperability with C, you should use the `c_str::to_c_str` function.

The standard library includes type aliases and function definitions for the C standard library in
the `libc` module, and Rust links against `libc` and `libm` by default.
