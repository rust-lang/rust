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

#[fixed_stack_segment]
fn main() {
    let x = unsafe { snappy_max_compressed_length(100) };
    println(fmt!("max compressed length of a 100 byte buffer: %?", x));
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

Finally, the `#[fixed_stack_segment]` annotation that appears on
`main()` instructs the Rust compiler that when `main()` executes, it
should request a "very large" stack segment.  More details on
stack management can be found in the following sections.

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
#[fixed_stack_segment]
#[inline(never)]
pub fn validate_compressed_buffer(src: &[u8]) -> bool {
    unsafe {
        snappy_validate_compressed_buffer(vec::raw::to_ptr(src), src.len() as size_t) == 0
    }
}
~~~~

The `validate_compressed_buffer` wrapper above makes use of an `unsafe` block, but it makes the
guarantee that calling it is safe for all inputs by leaving off `unsafe` from the function
signature.

The `validate_compressed_buffer` wrapper is also annotated with two
attributes `#[fixed_stack_segment]` and `#[inline(never)]`. The
purpose of these attributes is to guarantee that there will be
sufficient stack for the C function to execute. This is necessary
because Rust, unlike C, does not assume that the stack is allocated in
one continuous chunk. Instead, we rely on a *segmented stack* scheme,
in which the stack grows and shrinks as necessary.  C code, however,
expects one large stack, and so callers of C functions must request a
large stack segment to ensure that the C routine will not run off the
end of the stack.

The compiler includes a lint mode that will report an error if you
call a C function without a `#[fixed_stack_segment]` attribute. More
details on the lint mode are given in a later section.

You may be wondering why we include a `#[inline(never)]` directive.
This directive informs the compiler never to inline this function.
While not strictly necessary, it is usually a good idea to use an
`#[inline(never)]` directive in concert with `#[fixed_stack_segment]`.
The reason is that if a fn annotated with `fixed_stack_segment` is
inlined, then its caller also inherits the `fixed_stack_segment`
annotation. This means that rather than requesting a large stack
segment only for the duration of the call into C, the large stack
segment would be used for the entire duration of the caller. This is
not necessarily *bad* -- it can for example be more efficient,
particularly if `validate_compressed_buffer()` is called multiple
times in a row -- but it does work against the purpose of the
segmented stack scheme, which is to keep stacks small and thus
conserve address space.

The `snappy_compress` and `snappy_uncompress` functions are more complex, since a buffer has to be
allocated to hold the output too.

The `snappy_max_compressed_length` function can be used to allocate a vector with the maximum
required capacity to hold the compressed output. The vector can then be passed to the
`snappy_compress` function as an output parameter. An output parameter is also passed to retrieve
the true length after compression for setting the length.

~~~~ {.xfail-test}
pub fn compress(src: &[u8]) -> ~[u8] {
    #[fixed_stack_segment]; #[inline(never)];

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
    #[fixed_stack_segment]; #[inline(never)];

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

# Automatic wrappers

Sometimes writing Rust wrappers can be quite tedious.  For example, if
function does not take any pointer arguments, often there is no need
for translating types. In such cases, it is usually still a good idea
to have a Rust wrapper so as to manage the segmented stacks, but you
can take advantage of the (standard) `externfn!` macro to remove some
of the tedium.

In the initial section, we showed an extern block that added a call
to a specific snappy API:

~~~~ {.xfail-test}
use std::libc::size_t;

#[link_args = "-lsnappy"]
extern {
    fn snappy_max_compressed_length(source_length: size_t) -> size_t;
}

#[fixed_stack_segment]
fn main() {
    let x = unsafe { snappy_max_compressed_length(100) };
    println(fmt!("max compressed length of a 100 byte buffer: %?", x));
}
~~~~

To avoid the need to create a wrapper fn for `snappy_max_compressed_length()`,
and also to avoid the need to think about `#[fixed_stack_segment]`, we
could simply use the `externfn!` macro instead, as shown here:

~~~~ {.xfail-test}
use std::libc::size_t;

externfn!(#[link_args = "-lsnappy"]
          fn snappy_max_compressed_length(source_length: size_t) -> size_t)

fn main() {
    let x = unsafe { snappy_max_compressed_length(100) };
    println(fmt!("max compressed length of a 100 byte buffer: %?", x));
}
~~~~

As you can see from the example, `externfn!` replaces the extern block
entirely. After macro expansion, it will create something like this:

~~~~ {.xfail-test}
use std::libc::size_t;

// Automatically generated by
//   externfn!(#[link_args = "-lsnappy"]
//             fn snappy_max_compressed_length(source_length: size_t) -> size_t)
unsafe fn snappy_max_compressed_length(source_length: size_t) -> size_t {
    #[fixed_stack_segment]; #[inline(never)];
    return snappy_max_compressed_length(source_length);

    #[link_args = "-lsnappy"]
    extern {
        fn snappy_max_compressed_length(source_length: size_t) -> size_t;
    }
}

fn main() {
    let x = unsafe { snappy_max_compressed_length(100) };
    println(fmt!("max compressed length of a 100 byte buffer: %?", x));
}
~~~~

# Segmented stacks and the linter

By default, whenever you invoke a non-Rust fn, the `cstack` lint will
check that one of the following conditions holds:

1. The call occurs inside of a fn that has been annotated with
   `#[fixed_stack_segment]`;
2. The call occurs inside of an `extern fn`;
3. The call occurs within a stack closure created by some other
   safe fn.

All of these conditions ensure that you are running on a large stack
segment. However, they are sometimes too strict. If your application
will be making many calls into C, it is often beneficial to promote
the `#[fixed_stack_segment]` attribute higher up the call chain.  For
example, the Rust compiler actually labels main itself as requiring a
`#[fixed_stack_segment]`. In such cases, the linter is just an
annoyance, because all C calls that occur from within the Rust
compiler are made on a large stack. Another situation where this
frequently occurs is on a 64-bit architecture, where large stacks are
the default. In cases, you can disable the linter by including a
`#[allow(cstack)]` directive somewhere, which permits violations of
the "cstack" rules given above (you can also use `#[warn(cstack)]` to
convert the errors into warnings, if you prefer).

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
        #[fixed_stack_segment];
        #[inline(never)];

        unsafe {
            let ptr = malloc(std::sys::size_of::<T>() as size_t) as *mut T;
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
        #[fixed_stack_segment];
        #[inline(never)];

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
    println(fmt!("You have readline version %d installed.",
                 rl_readline_version as int));
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
#[cfg(target_os = "win32")]
#[link_name = "kernel32"]
extern "stdcall" {
    fn SetEnvironmentVariableA(n: *u8, v: *u8) -> int;
}
~~~~

This applies to the entire `extern` block, and must be either `"cdecl"` or
`"stdcall"`. The compiler may eventually support other calling conventions.

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
