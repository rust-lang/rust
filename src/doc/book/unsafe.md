% Unsafe

Rust’s main draw is its powerful static guarantees about behavior. But safety
checks are conservative by nature: there are some programs that are actually
safe, but the compiler is not able to verify this is true. To write these kinds
of programs, we need to tell the compiler to relax its restrictions a bit. For
this, Rust has a keyword, `unsafe`. Code using `unsafe` has less restrictions
than normal code does.

Let’s go over the syntax, and then we’ll talk semantics. `unsafe` is used in
four contexts. The first one is to mark a function as unsafe:

```rust
unsafe fn danger_will_robinson() {
    // scary stuff
}
```

All functions called from [FFI][ffi] must be marked as `unsafe`, for example.
The second use of `unsafe` is an unsafe block:

[ffi]: ffi.html

```rust
unsafe {
    // scary stuff
}
```

The third is for unsafe traits:

```rust
unsafe trait Scary { }
```

And the fourth is for `impl`ementing one of those traits:

```rust
# unsafe trait Scary { }
unsafe impl Scary for i32 {}
```

It’s important to be able to explicitly delineate code that may have bugs that
cause big problems. If a Rust program segfaults, you can be sure it’s somewhere
in the sections marked `unsafe`.

# What does ‘safe’ mean?

Safe, in the context of Rust, means ‘doesn’t do anything unsafe’. It’s also
important to know that there are certain behaviors that are probably not
desirable in your code, but are expressly _not_ unsafe:

* Deadlocks
* Leaks of memory or other resources
* Exiting without calling destructors
* Integer overflow

Rust cannot prevent all kinds of software problems. Buggy code can and will be
written in Rust. These things aren’t great, but they don’t qualify as `unsafe`
specifically.

In addition, the following are all undefined behaviors in Rust, and must be
avoided, even when writing `unsafe` code:

* Data races
* Dereferencing a null/dangling raw pointer
* Reads of [undef][undef] (uninitialized) memory
* Breaking the [pointer aliasing rules][aliasing] with raw pointers.
* `&mut T` and `&T` follow LLVM’s scoped [noalias][noalias] model, except if
  the `&T` contains an `UnsafeCell<U>`. Unsafe code must not violate these
  aliasing guarantees.
* Mutating an immutable value/reference without `UnsafeCell<U>`
* Invoking undefined behavior via compiler intrinsics:
  * Indexing outside of the bounds of an object with `std::ptr::offset`
    (`offset` intrinsic), with
    the exception of one byte past the end which is permitted.
  * Using `std::ptr::copy_nonoverlapping_memory` (`memcpy32`/`memcpy64`
    intrinsics) on overlapping buffers
* Invalid values in primitive types, even in private fields/locals:
  * Null/dangling references or boxes
  * A value other than `false` (0) or `true` (1) in a `bool`
  * A discriminant in an `enum` not included in its type definition
  * A value in a `char` which is a surrogate or above `char::MAX`
  * Non-UTF-8 byte sequences in a `str`
* Unwinding into Rust from foreign code or unwinding from Rust into foreign
  code.

[noalias]: http://llvm.org/docs/LangRef.html#noalias
[undef]: http://llvm.org/docs/LangRef.html#undefined-values
[aliasing]: http://llvm.org/docs/LangRef.html#pointer-aliasing-rules

# Unsafe Superpowers

In both unsafe functions and unsafe blocks, Rust will let you do three things
that you normally can not do. Just three. Here they are:

1. Access or update a [static mutable variable][static].
2. Dereference a raw pointer.
3. Call unsafe functions. This is the most powerful ability.

That’s it. It’s important that `unsafe` does not, for example, ‘turn off the
borrow checker’. Adding `unsafe` to some random Rust code doesn’t change its
semantics, it won’t start accepting anything. But it will let you write
things that _do_ break some of the rules.

You will also encounter the `unsafe` keyword when writing bindings to foreign
(non-Rust) interfaces. You're encouraged to write a safe, native Rust interface
around the methods provided by the library.

Let’s go over the basic three abilities listed, in order.

## Access or update a `static mut`

Rust has a feature called ‘`static mut`’ which allows for mutable global state.
Doing so can cause a data race, and as such is inherently not safe. For more
details, see the [static][static] section of the book.

[static]: const-and-static.html#static

## Dereference a raw pointer

Raw pointers let you do arbitrary pointer arithmetic, and can cause a number of
different memory safety and security issues. In some senses, the ability to
dereference an arbitrary pointer is one of the most dangerous things you can
do. For more on raw pointers, see [their section of the book][rawpointers].

[rawpointers]: raw-pointers.html

## Call unsafe functions

This last ability works with both aspects of `unsafe`: you can only call
functions marked `unsafe` from inside an unsafe block.

This ability is powerful and varied. Rust exposes some [compiler
intrinsics][intrinsics] as unsafe functions, and some unsafe functions bypass
safety checks, trading safety for speed.

I’ll repeat again: even though you _can_ do arbitrary things in unsafe blocks
and functions doesn’t mean you should. The compiler will act as though you’re
upholding its invariants, so be careful!

[intrinsics]: intrinsics.html
