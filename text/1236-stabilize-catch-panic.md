- Feature Name: `recover`
- Start Date: 2015-07-24
- RFC PR: [rust-lang/rfcs#1236](https://github.com/rust-lang/rfcs/pull/1236)
- Rust Issue: [rust-lang/rust#27719](https://github.com/rust-lang/rust/issues/27719)

# Summary

Move `std::thread::catch_panic` to `std::panic::recover` after replacing the
`Send + 'static` bounds on the closure parameter with a new `PanicSafe`
marker trait.

# Motivation

In today's stable Rust it's not possible to catch a panic on the thread that
caused it. There are a number of situations, however, where this is
either required for correctness or necessary for building a useful abstraction:

* It is currently defined as undefined behavior to have a Rust program panic
  across an FFI boundary. For example if C calls into Rust and Rust panics, then
  this is undefined behavior. Being able to catch a panic will allow writing
  C APIs in Rust that do not risk aborting the process they are embedded into.

* Abstractions like thread pools want to catch the panics of tasks being run
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
over `Send` and `'static` data. These bounds can be overly restrictive, and due
to thread-local storage [they can be subverted][tls-subvert], making it unclear
what purpose they serve. This RFC proposes to remove the bounds as well.

[tls-subvert]: https://github.com/rust-lang/rust/issues/25662

Historically Rust has purposefully avoided the foray into the situation of
catching panics, largely because of a problem typically referred to as
"exception safety". To further understand the motivation of stabilization and
relaxing the bounds, let's review what exception safety is and what it means for
Rust.

# Background: What is exception safety?

Languages with exceptions have the property that a function can "return" early
if an exception is thrown. While exceptions aren't too hard to reason about when
thrown explicitly, they can be problematic when they are thrown by code being
called -- especially when that code isn't known in advance. Code is **exception
safe** if it works correctly even when the functions it calls into throw
exceptions.

The idea of throwing an exception causing bugs may sound a bit alien, so it's
helpful to drill down into exactly why this is the case. Bugs related to
exception safety are comprised of two critical components:

1. An invariant of a data structure is broken.
2. This broken invariant is the later observed.

Exceptional control flow often exacerbates this first component of breaking
invariants. For example many data structures have a number of invariants that
are dynamically upheld for correctness, and the type's routines can temporarily
break these invariants to be fixed up before the function returns.  If, however,
an exception is thrown in this interim period the broken invariant could be
accidentally exposed.

The second component, observing a broken invariant, can sometimes be difficult
in the face of exceptions, but languages often have constructs to enable these
sorts of witnesses. Two primary methods of doing so are something akin to
finally blocks (code run on a normal or exceptional return) or just catching the
exception. In both cases code which later runs that has access to the original
data structure will see the broken invariants.

Now that we've got a better understanding of how an exception might cause a bug
(e.g. how code can be "exception unsafe"), let's take a look how we can make
code exception safe. To be exception safe, code needs to be prepared for an
exception to be thrown whenever an invariant it relies on is broken, for
example:

* Code can be audited to ensure it only calls functions which are statically
  known to not throw an exception.
* Local "cleanup" handlers can be placed on the stack to restore invariants
  whenever a function returns, either normally or exceptionally. This can be
  done through finally blocks in some languages or via destructors in others.
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
code today**.

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
to memory safety. *Logical invariants* are often just as critical to keep
correct during execution and no `unsafe` code in Rust is needed to break a
logical invariant. In practice, however, these sorts of bugs are rarely observed
due to Rust's design:

* Rust doesn't expose uninitialized memory
* Panics cannot be caught in a thread
* Across threads data is poisoned by default on panics
* Idiomatic Rust must opt in to extra sharing across boundaries (e.g. `RefCell`)
* Destructors are relatively rare and uninteresting in safe code

These mitigations all address the *second* aspect of exception unsafety:
observation of broken invariants. With the tactics in place, it ends up being
the case that **safe Rust code can largely ignore exception safety
concerns**. That being said, it does not mean that safe Rust code can *always*
ignore exception safety issues. There are a number of methods to subvert the
mitigation strategies listed above:

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

But all of these "subversions" fall outside the realm of normal, idiomatic, safe
Rust code, and so they all serve as a "heads up" that panic safety might be an
issue. Thus, in practice, Rust programmers worry about exception safety far less
than in languages with full-blown exceptions.

Despite these methods to subvert the mitigations placed by default in Rust, a
key part of exception safety in Rust is that **safe code can never lead to
memory unsafety**, regardless of whether it panics or not. Memory unsafety
triggered as part of a panic can always be traced back to an `unsafe` block.

With all that background out of the way now, let's take a look at the guts of
this RFC.

# Detailed design

At its heart, the change this RFC is proposing is to move
`std::thread::catch_panic` to a new `std::panic` module and rename the function
to `recover`. Additionally, the `Send + 'static` bounds on the closure parameter
will be replaced with a new trait `PanicSafe`, modifying the signature to
be:

```rust
fn recover<F: FnOnce() -> R + PanicSafe, R>(f: F) -> thread::Result<R>
```

Before analyzing this new signature, let's take a look at this new
`PanicSafe` trait.

## A `PanicSafe` marker trait

As discussed in the motivation section above, the current bounds of `Send +
'static` on the closure parameter are too restrictive for common use cases, but
they can serve as a "speed bump" (like poisoning on mutexes) to add to the
repertoire of mitigation strategies that Rust has by default for dealing with
panics.

The purpose of this marker trait will be to identify patterns which do not need
to worry about exception safety and allow them by default. In situations where
exception safety *may* be concerned then an explicit annotation will be needed
to allow the usage. In other words, this marker trait will act similarly to a
"targeted `unsafe` block".

For the implementation details, the following items will be added to the
`std::panic` module.

```rust
pub trait PanicSafe {}
impl PanicSafe for .. {}

impl<'a, T> !PanicSafe for &'a mut T {}
impl<'a, T: NoUnsafeCell> PanicSafe for &'a T {}
impl<T> PanicSafe for Mutex<T> {}

pub trait NoUnsafeCell {}
impl NoUnsafeCell for .. {}
impl<T> !NoUnsafeCell for UnsafeCell<T> {}

pub struct AssertPanicSafe<T>(pub T);
impl<T> PanicSafe for AssertPanicSafe<T> {}

impl<T> Deref for AssertPanicSafe<T> {
    type Target = T;
    fn deref(&self) -> &T { &self.0 }
}
impl<T> DerefMut for AssertPanicSafe<T> {
    fn deref_mut(&mut self) -> &mut T { &mut self.0 }
}
```

Let's take a look at each of these items in detail:

* `impl PanicSafe for .. {}` - this makes this trait a marker trait, implying
  that a the trait is implemented for all types by default so long as the
  consituent parts implement the trait.
* `impl<T> !PanicSafe for &mut T {}` - this indicates that exception safety
  needs to be handled when dealing with mutable references. Thinking about the
  `recover` function, this means that the pointer could be modified inside the
  block, but once it exits the data may or may not be in an invalid state.
* `impl<T: NoUnsafeCell> PanicSafe for &T {}` - similarly to the above
  implementation for `&mut T`, the purpose here is to highlight points where
  data can be mutated across a `recover` boundary. If `&T` does not contains an
  `UnsafeCell`, then no mutation should be possible and it is safe to allow.
* `impl<T> PanicSafe for Mutex<T> {}` - as mutexes are poisoned by default, they
  are considered exception safe.
* `pub struct AssertPanicSafe<T>(pub T);` - this is the "opt out" structure of
  exception safety. Wrapping something in this type indicates an assertion that
  it is exception safe and shouldn't be warned about when crossing the `recover`
  boundary. Otherwise this type simply acts like a `T`.

### Example usage

The only consumer of the `PanicSafe` bound is the `recover` function on the
closure type parameter, and this ends up meaning that the *environment* needs to
be exception safe. In terms of error messages, this causes the compiler to emit
an error per closed-over-variable to indicate whether or not it is exception
safe to share across the boundary.

It is also a critical design aspect that usage of `PanicSafe` or
`AssertPanicSafe` does not require `unsafe` code. As discussed above, panic
safety does not directly lead to memory safety problems in otherwise safe code.

In the normal usage of `recover`, neither `PanicSafe` nor `AssertPanicSafe`
should be necessary to mention. For example when defining an FFI function:

```rust
#[no_mangle]
pub extern fn called_from_c(ptr: *const c_char, num: i32) -> i32 {
    let result = panic::recover(|| {
        let s = unsafe { CStr::from_ptr(ptr) };
        println!("{}: {}", s, num);
    });
    match result {
        Ok(..) => 0,
        Err(..) => 1,
    }
}
```

Additionally, if FFI functions instead use normal Rust types, `AssertPanicSafe`
still need not be mentioned at all:

```rust
#[no_mangle]
pub extern fn called_from_c(ptr: &i32) -> i32 {
    let result = panic::recover(|| {
        println!("{}", *ptr);
    });
    match result {
        Ok(..) => 0,
        Err(..) => 1,
    }
}
```

If, however, types are coming in which are flagged as not exception safe then
the `AssertPanicSafe` wrapper can be used to leverage `recover`:

```rust
fn foo(data: &RefCell<i32>) {
    panic::recover(|| {
        println!("{}", data.borrow()); //~ ERROR RefCell is not panic safe
    });
}
```

This can be fixed with a simple assertion that the usage here is indeed
exception safe:

```rust
fn foo(data: &RefCell<i32>) {
    let data = AssertPanicSafe(data);
    panic::recover(|| {
        println!("{}", data.borrow()); // ok
    });
}
```

### Future extensions

In the future, this RFC proposes adding the following implementation of
`PanicSafe`:

```rust
impl<T: Send + 'static> PanicSafe for T {}
```

This implementation block encodes the "exception safe" boundary of
`thread::spawn` but is unfortunately not allowed today due to coherence rules.
If available, however, it would possibly reduce the number of false positives
which require using `AssertPanicSafe`.

### Global complexity

Adding a new marker trait is a pretty hefty move for the standard library. The
current marker traits, `Send` and `Sync`, are well known and are ubiquitous
throughout the ecosystem and standard library. Due to the way that these
properties are derived, adding a new marker trait can lead to a multiplicative
increase in global complexity (as all types must consider the marker trait).

With `PanicSafe`, however, it is expected that this is not the case. The
`recover` function is not intented to be used commonly outside of FFI or thread
pool-like abstractions. Within FFI the `PanicSafe` trait is typically not
mentioned due to most types being relatively simple. Thread pools, on the other
hand, will need to mention `AssertPanicSafe`, but will likely propagate panics
to avoid exposing `PanicSafe` as a bound.

Overall, the expected idiomatic usage of `recover` should mean that `PanicSafe`
is rarely mentioned, if at all. It is intended that `AssertPanicSafe` is ideally
only necessary where it actually needs to be considered (which idiomatically
isn't too often) and even then it's lightweight to use.

## Will Rust have exceptions?

In a technical sense this RFC is not "adding exceptions to Rust" as they already
exist in the form of panics. What this RFC is adding, however, is a construct
via which to catch these exceptions within a thread, bringing the standard
library closer to the exception support in other languages.

Catching a panic makes it easier to observe broken invariants of data structures
shared across the `catch_panic` boundary, which can possibly increase the
likelihood of exception safety issues arising.

The risk of this step is that catching panics becomes an idiomatic way to deal
with error-handling, thereby making exception safety much more of a headache
than it is today (as it's more likely that a broken invariant is later
witnessed). The `catch_panic` function is intended to only be used
where it's absolutely necessary, e.g. for FFI boundaries, but how can it be
ensured that `catch_panic` isn't overused?

There are two key reasons `catch_panic` likely won't become idiomatic:

1. There are already strong and established conventions around error handling,
   and in particular around the use of panic and `Result` with stabilized usage
   of them in the standard library. There is little chance these conventions
   would change overnight.

2. There has long been a desire to treat every use of `panic!` as an abort
   which is motivated by portability, compile time, binary size, and a number of
   other factors. Assuming this step is taken, it would be extremely unwise for
   a library to signal expected errors via panics and rely on consumers using
   `catch_panic` to handle them.

For reference, here's a summary of the conventions around `Result` and `panic`,
which still hold good after this RFC:

### Result vs Panic

There are two primary strategies for signaling that a function can fail in Rust
today:

* `Results` represent errors/edge-cases that the author of the library knew
  about, and expects the consumer of the library to handle.

* `panic`s represent errors that the author of the library did not expect to
  occur, such as a contract violation, and therefore does not expect the
  consumer to handle in any particular way.

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
determine when a panic can happen or handle it in a custom fashion, even with
`catch_panic`.

These divisions justify the use of `panic`s for things like out-of-bounds
indexing: such an error represents a programming mistake that (1) the author of
the library was not aware of, by definition, and (2) cannot be meaningfully
handled by the caller.

In terms of heuristics for use, `panic`s should rarely if ever be used to report
routine errors for example through communication with the system or through IO.
If a Rust program shells out to `rustc`, and `rustc` is not found, it might be
tempting to use a panic because the error is unexpected and hard to recover
from. A user of the program, however, would benefit from intermediate code
adding contextual information about the in-progress operation, and the program
could report the error in terms a they can understand. While the error is
rare, **when it happens it is not a programmer error**. In short, panics are
roughly analogous to an opaque "an unexpected error has occurred" message.

Stabilizing `catch_panic` does little to change the tradeoffs around `Result`
and `panic` that led to these conventions.

# Drawbacks

A drawback of this RFC is that it can water down Rust's error handling story.
With the addition of a "catch" construct for exceptions, it may be unclear to
library authors whether to use panics or `Result` for their error types. As we
discussed above, however, Rust's design around error handling has always had to
deal with these two strategies, and our conventions don't materially change by
stabilizing `catch_panic`.

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

- Is it worth keeping the `'static` and `Send` bounds as a mitigation measure in
  practice, even if they aren't enforceable in theory? That would require thread
  pools to use unsafe code, but that could be acceptable.

- Should `catch_panic` be stabilized within `std::thread` where it lives today,
  or somewhere else?
