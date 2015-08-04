- Feature Name: `catch_panic`
- Start Date: 2015-07-24
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Stabilize `std::thread::catch_panic` after removing the `Send` and `'static`
bounds from the closure parameter.

# Motivation

In today's stable Rust it's not currently possible to catch a panic. There are a
number of situations, however, where catching a panic is either required for
correctness or necessary for building a useful abstraction:

* It is currently defined as undefined behavior to have a Rust program panic
  across an FFI boundary. For example if C calls into Rust and Rust panics, then
  this is undefined behavior. Being able to catch a panic will allow writing
  robust C apis in Rust.
* Abstactions like thread pools want to catch the panics of tasks being run
  instead of having the thread torn down (and having to spawn a new thread).

Stabilizing the `catch_panic` function would enable these two use cases, but
let's also take a look at the current signature of the function:

```rust
fn catch_panic<F, R>(f: F) -> thread::Result<R>
    where F: FnOnce() -> R + Send + 'static
```

This function will run the closure `f` and if it panics return `Err(Box<Any>)`.
If the closure doesn't panic it will return `Ok(val)` where `val` is the
returned value of the closure. The closure, however, is restricted to only close
over `Send` and `'static` data. This can be overly restrictive at times and it's
also not clear what purpose the bounds are serving today, hence the desire to
remove these bounds.

Historically Rust has purposefully avoided the foray into the situation of
catching panics, largely because of a problem typically referred to as
"exception safety". To further understand the motivation of stabilization and
relaxing the bounds, let's review what exception safety is and what it means for
Rust.

# Background: What is exception safety?

Languages with exceptions have the property that a function can "return" early
if an exception is thrown. This is normally not something that needs to be
worried about, but this form of control flow can often be surprising and
unexpected. If an exception ends up causing unexpected behavior or a bug then
code is said to not be **exception safe**.

The idea of throwing an exception causing bugs may sound a bit alien, so it's
helpful to drill down into exactly why this is the case. Bugs related to
exception safety are comprised of two critical components:

1. An invariant of a data structure is broken.
2. This broken invariant is the later observed.

Exceptional control flow often exacerbates this first component of breaking
invariants. For example many data structures often have a number of invariants
that are dynamically upheld for correctness, and the type's routines can
temporarily break these invariants to be fixed up before the function returns.
If, however, an exception is thrown in this interim period the broken invariant
could be accidentally exposed.

The second component, observing a broken invariant, can sometimes be difficult
in the face of exceptions, but languages often have constructs to enable these
sorts of witnesses. Two primary methods of doing so are something akin to
finally blocks (code run on a normal or exceptional return) or just catching the
exception. In both cases code which later runs that has access to the original
data structure then it will see the broken invariants.

Now that we've got a better understanding of how an exception might cause a bug
(e.g. how code can be "exception unsafe"), let's take a look how we can make
code exception safe. To be exception safe, code needs to be prepared for an
exception to be thrown whenever an invariant it relies on is broken, for
example:

* Code can be audited to ensure it only calls functions which are statically
  known to not throw an exception.
* Local "cleanup" handlers can be placed on the stack to restore invariants
  whenever a function returns, either normally or exceptionally. This can be
  done through finally blocks in some languages for via destructors in others.
* Exceptions can be caught locally to perform cleanup before possibly re-raising
  the exception.

With all that in mind, we've now identified problems that can arise via
exceptions (an invariant is broken and then observed) as well as methods to
ensure that prevent this from happening. In languages like C++ this means that
we can be memory safe in the face of exceptions and in languages like Java we
can ensure that our logical invariants are upheld. Given this background let's
take a look at how any of this applies to Rust.

# Background: What is exception safety in Rust?

> Note: This section describes the current state of Rust today without this RFC
>       implemented

Up to now we've been talking about exceptions and exception safety, but from a
Rust perspective we can just replace this with panics and panic safety. Panics
in Rust are currently implemented essentially as a C++ exception under the hood.
As a result, **exception safety is something that needs to be handled in Rust
code**.

One of the primary examples where panics need to be handled in Rust is unsafe
code. Let's take a look at an example where this matters:

```rust
pub fn push_ten_more<T: Clone>(v: &mut Vec<T>, t: T) {
    unsafe {
        v.reserve(10);
        let len = v.len();
        v.set_len(len + 10);
        for i in 0..10 {
            ptr::write(v.as_mut_ptr().offset(len + i), t.clone());
        }
    }
}
```

While this code may look correct, it's actually not memory safe.
`Vec` has an internal invariant that its first `len` elements are safe to drop
at any time. Our function above has temporarily broken this invariant with the
call to `set_len` (the next 10 elements are uninitialized). If the type `T`'s
`clone` method panics then this broken invariant will escape the function. The
broken `Vec` is then observed during its destructor, leading to the eventual
memory unsafety.

It's important to keep in mind that panic safety in Rust is not solely limited
to memory safety. *Logical invariants* are often just as critical to keep correct
during execution and no `unsafe` code in Rust is needed to break a logical
invariant. In practice, however, these sorts of bugs are rarely observed due to
Rust's design:

* Rust doesn't expose uninitialized memory
* Panics cannot be caught in a thread
* Across threads data is poisoned by default on panics
* Idiomatic Rust must opt in to extra amounts of sharing data across boundaries

With these mitigation tactics, it ends up being the case that **safe Rust code
can mostly ignore exception safety concerns**. That being said, it does not mean
that safe Rust code can *always* ignore exception safety issues. There are a
number of methods to subvert the mitigation strategies listed above:

1. When poisoning data across threads, antidotes are available to access
   poisoned data. Namely the [`PoisonError` type][pet] allows safe access to the
   poisoned information.
2. Single-threaded types with interior mutability, such as `RefCell`, allow for
   sharing data across stack frames such that a broken invariant could
   eventually be observed.
3. Whenever a thread panics, the destructors for its stack variables will be run
   as the thread unwinds. Destructors may have access to data which was also
   accessible lower on the stack (such as through `RefCell` or `Rc`) which has a
   broken invariant, and the destructor may then witness this.

[pet]: http://doc.rust-lang.org/std/sync/struct.PoisonError.html

Despite these methods to subvert the mitigations placed by default in Rust, a
key part of exception safety in Rust is that **safe code can never lead to
memory unsafety**, regardless of whether it panics or not. Memory unsafety
triggered as part of a panic can always be traced back to an `unsafe` block.

With all that background out of the way now, let's take a look at the guts of
this RFC.

# Detailed design

At its heard, the change this RFC is proposing is to stabilize
`std::thread::catch_panic` after removing the `Send` and `'static` bounds from
the closure parameter, modifying the signature to be:

```rust
fn catch_panic<F: FnOnce() -> R, R>(f: F) -> thread::Result<R>
```

More generally, however, this RFC also claims that this stable function does
not radically alter Rust's exception safety story (explained above).

### Exception safety mitigation

A mitigation strategy for exception safety listed above is that a panic cannot
be caught within a thread, and this change would move that bullet to the list of
"methods to subvert the mitigation strategies" instead. Catching a panic (and
not having `'static` on the bounds list) makes it easier to observe broken
invariants of data structures shared across the `catch_panic` boundary, which
can possibly increase the likelihood of exception safety issues arising.

One of the key reasons Rust doesn't provide an exhaustive set of mitigation
strategies is that the design of the language and standard library lead to
idiomatic code not having to worry about exception safety. The use cases for
`catch_panic` are relatively niche, and it is not expected for `catch_panic` to
overnight become the idiomatic method of handling errors in Rust.

Essentially, the addition of `catch_panic`:

* Does not mean that *only now* does Rust code need to consider exception
  safety. This is something that already must be handled today.
* Does not mean that safe code everywhere must start worrying about exception
  safety. This function is not the primary method to signal errors in Rust
  (discussed later) and only adds a minor bullet to the list of situations that
  safe Rust already needs to worry about exception safety in.

### Will Rust have exceptions?

In a technical sense this RFC is not "adding exceptions to Rust" as they
already exist in the form of panics. What this RFC is adding, however, is a
construct via which to catch these exceptions, bringing the standard library
closer to the exception support in other languages. Idiomatic usage of Rust,
however, will continue to follow the guidelines listed below for using a Result
vs using a panic (which also do not need to change to account for this RC).

It's likely that the `catch_panic` function will only be used where it's
absolutely necessary, like FFI boundaries, instead of a general-purpose error
handling mechanism in all code.

# Drawbacks

A drawback of this RFC is that it can water down Rust's error handling story.
With the addition of a "catch" construct for exceptions, it may be unclear to
library authors whether to use panics or `Result` for their error types. There
are fairly clear guidelines and conventions about using a `Result` vs a `panic`
today, however, and they're summarized below for completeness.

### Result vs Panic

There are two primary strategies for signaling that a function can fail in Rust
today:

* `Results` represent errors/edge-cases that the author of the library knew
  about, and expects the consumer of the library to handle.
* `panic`s represent errors that the author of the library did not expect to
  occur, and therefore does not expect the consumer to handle in any particular
  way.

Another way to put this division is that:

* `Result`s represent errors that carry additional contextual information. This
  information allows them to be handled by the caller of the function producing
  the error, modified with additional contextual information, and eventually
  converted into an error message fit for a top-level program.
* `panic`s represent errors that carry no contextual information (except,
  perhaps, debug information). Because they represented an unexpected error,
  they cannot be easily handled by the caller of the function or presented to
  the top-level program (except to say "something unexpected has gone wrong").

Some pros of `Result` are that it signals specific edge cases that you as a
consumer should think about handling and it allows the caller to decide
precisely how to handle the error. A con with `Result` is that defining errors
and writing down `Result` + `try!` is not always the most ergonomic.

The pros and cons of `panic` are essentially the opposite of `Result`, being
easy to use (nothing to write down other than the panic) but difficult to
determine when a panic can happen or handle it in a custom fashion.

### Result? Or panic?

These divisions justify the use of `panic`s for things like out-of-bounds
indexing: such an error represents a programming mistake that (1) the author of
the library was not aware of, by definition, and (2) cannot be easily handled by
the caller.

In terms of heuristics for use, `panic`s should rarely if ever be used to report
routine errors for example through communication with the system or through IO.
If a Rust program shells out to `rustc`, and `rustc` is not found, it might be
tempting to use a panic because the error is unexpected and hard to recover
from. A user of the program, however, would benefit from intermediate code
adding contextual information about the in-progress operation, and the program
could report the error in terms a they can understand. While the error is
rare, **when it happens it is not a programmer error**. In short, panics are
roughly analogous to an opaque "an unexpected error has occurred" message.

Another key reason to choose `Result` over a panic is that the compiler is
likely to soon grow an option to map a panic to an abort. This is motivated for
portability, compile time, binary size, and a number of other factors, but it
fundamentally means that a library which signals errors via panics (and relies
on consumers using `catch_panic`) will not be usable in this context.

# Alternatives

One alternative, which is somewhat more of an addition, is to have the standard
library entirely abandon all exception safety mitigation tactics. As explained
in the motivation section, exception safety will not lead to memory unsafety
unless paired with unsafe code, so it is perhaps within the realm of possibility
to remove the tactics of poisoning from mutexes and simply require that
consumers deal with exception safety 100% of the time.

This alternative is often motivated by saying that there are enough methods to
subvert the default mitigation tactics that it's not worth trying to plug some
holes and not others. Upon closer inspection, however, the areas where safe code
needs to worry about exception safety are isolated to the single-threaded
situations. For example `RefCell`, destructors, and `catch_panic` all only
expose data possibly broken through a panic in a single thread.

Once a thread boundary is crossed, the only current way to share data mutably is
via `Mutex` or `RwLock`, both of which are poisoned by default. This sort of
sharing is fundamental to threaded code, and poisoning by default allows safe
code to freely use many threads without having to consider exception safety
across threads (as poisoned data will tear down all connected threads).

This property of multithreaded programming in Rust is seen as strong enough that
poisoning should not be removed by default, and in fact a new hypothetical
`thread::scoped` API (a rough counterpart of `catch_panic`) could also propagate
panics by default (like poisoning) with an ability to opt out (like
`PoisonError`).

# Unresolved questions

None currently.
