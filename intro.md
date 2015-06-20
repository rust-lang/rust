% The Unsafe Rust Programming Language

**This document is about advanced functionality and low-level development practices
in the Rust Programming Language. Most of the things discussed won't matter
to the average Rust programmer. However if you wish to correctly write unsafe
code in Rust, this text contains invaluable information.**

This document seeks to complement [The Rust Programming Language Book][] (TRPL).
Where TRPL introduces the language and teaches the basics, TURPL dives deep into
the specification of the language, and all the nasty bits necessary to write
Unsafe Rust. TURPL does not assume you have read TRPL, but does assume you know
the basics of the language and systems programming. We will not explain the
stack or heap, we will not explain the syntax.




# Chapters

* [Data Layout](data.html)
* [Ownership and Lifetimes](lifetimes.html)
* [Conversions](conversions.html)
* [Uninitialized Memory](uninitialized.html)
* [Ownership-oriented resource management (RAII)](raii.html)
* [Concurrency](concurrency.html)




# A Tale Of Two Languages

Rust can be thought of as two different languages: Safe Rust, and Unsafe Rust.
Any time someone opines the guarantees of Rust, they are almost surely talking about
Safe Rust. However Safe Rust is not sufficient to write every program. For that,
we need the Unsafe Rust superset.

Most fundamentally, writing bindings to other languages
(such as the C exposed by your operating system) is never going to be safe. Rust
can't control what other languages do to program execution! However Unsafe Rust is
also necessary to construct fundamental abstractions where the type system is not
sufficient to automatically prove what you're doing is sound.

Indeed, the Rust standard library is implemented in Rust, and it makes substantial
use of Unsafe Rust for implementing IO, memory allocation, collections,
synchronization, and other low-level computational primitives.

Upon hearing this, many wonder why they would not simply just use C or C++ in place of
Rust (or just use a "real" safe language). If we're going to do unsafe things, why not
lean on these much more established languages?

The most important difference between C++ and Rust is a matter of defaults:
Rust is 100% safe by default. Even when you *opt out* of safety in Rust, it is a modular
action. In deciding to work with unchecked uninitialized memory, this does not
suddenly make dangling or null pointers a problem. When using unchecked indexing on `x`,
one does not have to suddenly worry about indexing out of bounds on `y`.
C and C++, by contrast, have pervasive unsafety baked into the language. Even the
modern best practices like `unique_ptr` have various safety pitfalls.

It should also be noted that writing Unsafe Rust should be regarded as an exceptional
action. Unsafe Rust is often the domain of *fundamental libraries*. Anything that needs
to make FFI bindings or define core abstractions. These fundamental libraries then expose
a *safe* interface for intermediate libraries and applications to build upon. And these
safe interfaces make an important promise: if your application segfaults, it's not your
fault. *They* have a bug.

And really, how is that different from *any* safe language? Python, Ruby, and Java libraries
can internally do all sorts of nasty things. The languages themselves are no
different. Safe languages regularly have bugs that cause critical vulnerabilities.
The fact that Rust is written with a healthy spoonful of Unsafe Rust is no different.
However it *does* mean that Rust doesn't need to fall back to the pervasive unsafety of
C to do the nasty things that need to get done.




# What does `unsafe` mean?

Rust tries to model memory safety through the `unsafe` keyword. Interestingly,
the meaning of `unsafe` largely revolves around what
its *absence* means. If the `unsafe` keyword is absent from a program, it should
not be possible to violate memory safety under *any* conditions. The presence
of `unsafe` means that there are conditions under which this code *could*
violate memory safety.

To be more concrete, Rust cares about preventing the following things:

* Dereferencing null/dangling pointers
* Reading uninitialized memory
* Breaking the pointer aliasing rules (TBD) (llvm rules + noalias on &mut and & w/o UnsafeCell)
* Invoking Undefined Behaviour (in e.g. compiler intrinsics)
* Producing invalid primitive values:
    * dangling/null references
    * a `bool` that isn't 0 or 1
    * an undefined `enum` discriminant
    * a `char` larger than char::MAX
    * A non-utf8 `str`
* Unwinding into an FFI function
* Causing a data race

That's it. That's all the Undefined Behaviour in Rust. Libraries are free to
declare arbitrary requirements if they could transitively cause memory safety
issues, but it all boils down to the above actions. Rust is otherwise
quite permisive with respect to other dubious operations. Rust considers it
"safe" to:

* Deadlock
* Leak memory
* Fail to call destructors
* Access private fields
* Overflow integers
* Delete the production database

However any program that does such a thing is *probably* incorrect. Rust just isn't
interested in modeling these problems, as they are much harder to prevent in general,
and it's literally impossible to prevent incorrect programs from getting written.

There are several places `unsafe` can appear in Rust today, which can largely be
grouped into two categories:

* There are unchecked contracts here. To declare you understand this, I require
you to write `unsafe` elsewhere:
    * On functions, `unsafe` is declaring the function to be unsafe to call. Users
      of the function must check the documentation to determine what this means,
      and then have to write `unsafe` somewhere to identify that they're aware of
    the danger.
    * On trait declarations, `unsafe` is declaring that *implementing* the trait
      is an unsafe operation, as it has contracts that other unsafe code is free to
      trust blindly.

* I am declaring that I have, to the best of my knowledge, adhered to the
unchecked contracts:
    * On trait implementations, `unsafe` is declaring that the contract of the
      `unsafe` trait has been upheld.
    * On blocks, `unsafe` is declaring any unsafety from an unsafe
      operation within to be handled, and therefore the parent function is safe.

There is also `#[unsafe_no_drop_flag]`, which is a special case that exists for
historical reasons and is in the process of being phased out. See the section on
destructors for details.

Some examples of unsafe functions:

* `slice::get_unchecked` will perform unchecked indexing, allowing memory
  safety to be freely violated.
* `ptr::offset` in an intrinsic that invokes Undefined Behaviour if it is
  not "in bounds" as defined by LLVM (see the lifetimes section for details).
* `mem::transmute` reinterprets some value as having the given type,
  bypassing type safety in arbitrary ways. (see the conversions section for details)
* All FFI functions are `unsafe` because they can do arbitrary things.
  C being an obvious culprit, but generally any language can do something
  that Rust isn't happy about. (see the FFI section for details)

As of Rust 1.0 there are exactly two unsafe traits:

* `Send` is a marker trait (it has no actual API) that promises implementors
  are safe to send to another thread.
* `Sync` is a marker trait that promises that threads can safely share
  implementors through a shared reference.

All other traits that declare any kind of contract *really* can't be trusted
to adhere to their contract when memory-safety is at stake. For instance Rust has
`PartialOrd` and `Ord` to differentiate between types which can "just" be
compared and those that implement a total ordering. However you can't actually
trust an implementor of `Ord` to actually provide a total ordering if failing to
do so causes you to e.g. index out of bounds. But if it just makes your program
do a stupid thing, then it's "fine" to rely on `Ord`.

The reason this is the case is that `Ord` is safe to implement, and it should be
impossible for bad *safe* code to violate memory safety. Rust has traditionally
avoided making traits unsafe because it makes `unsafe` pervasive in the language,
which is not desirable. The only reason `Send` and `Sync` are unsafe is because
thread safety is a sort of fundamental thing that a program can't really guard
against locally (even by-value message passing still requires a notion Send).




# Working with unsafe

Rust generally only gives us the tools to talk about safety in a scoped and
binary manner. Unfortunately reality is significantly more complicated than that.
For instance, consider the following toy function:

```rust
fn do_idx(idx: usize, arr: &[u8]) -> Option<u8> {
    if idx < arr.len() {
        unsafe {
            Some(*arr.get_unchecked(idx))
        }
    } else {
        None
    }
}
```

Clearly, this function is safe. We check that the index is in bounds, and if it
is, index into the array in an unchecked manner. But even in such a trivial
function, the scope of the unsafe block is questionable. Consider changing the
`<` to a `<=`:

```rust
fn do_idx(idx: usize, arr: &[u8]) -> Option<u8> {
    if idx <= arr.len() {
        unsafe {
            Some(*arr.get_unchecked(idx))
        }
    } else {
        None
    }
}
```

This program is now unsound, an yet *we only modified safe code*. This is the
fundamental problem of safety: it's non-local. The soundness of our unsafe
operations necessarily depends on the state established by "safe" operations.
Although safety *is* modular (we *still* don't need to worry about about
unrelated safety issues like uninitialized memory), it quickly contaminates the
surrounding code.

Trickier than that is when we get into actual statefulness. Consider a simple
implementation of `Vec`:

```rust
// Note this defintion is insufficient. See the section on lifetimes.
struct Vec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

// Note this implementation does not correctly handle zero-sized types.
// We currently live in a nice imaginary world of only postive fixed-size
// types.
impl<T> Vec<T> {
    fn new() -> Self {
        Vec { ptr: heap::EMPTY, len: 0, cap: 0 }
    }

    fn push(&mut self, elem: T) {
        if self.len == self.cap {
            // not important for this example
            self.reallocate();
        }
        unsafe {
            ptr::write(self.ptr.offset(len as isize), elem);
            self.len += 1;
        }
    }

    fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            self.len -= 1;
            unsafe {
                Some(ptr::read(self.ptr.offset(self.len as isize)))
            }
        } else {
            None
        }
    }
}
```

This code is simple enough to reasonably audit and verify. Now consider
adding the following method:

```rust
    fn make_room(&mut self) {
        // grow the capacity
        self.cap += 1;
    }
```

This code is safe, but it is also completely unsound. Changing the capacity
violates the invariants of Vec (that `cap` reflects the allocated space in the
Vec). This is not something the rest of `Vec` can guard against. It *has* to
trust the capacity field because there's no way to verify it.

`unsafe` does more than pollute a whole function: it pollutes a whole *module*.
Generally, the only bullet-proof way to limit the scope of unsafe code is at the
module boundary with privacy.

