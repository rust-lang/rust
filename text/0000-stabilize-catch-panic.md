- Feature Name: `catch_panic`
- Start Date: 2015-07-24
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Stabilize `std::thread::catch_panic` after removing the `Send` and `'static`
bounds from the closure parameter.

# Motivation

In today's Rust it's not currently possible to catch a panic by design. There
are a number of situations, however, where catching a panic is either required
for correctness or necessary for building a useful abstraction:

* It is currently defined as undefined behavior to have a Rust program panic
  across an FFI boundary. For example if C calls into Rust and Rust panics, then
  this is undefined behavior. Being able to catch a panic will allow writing
  robust C apis in Rust.
* Abstactions like thread pools want to catch the panics of tasks being run
  instead of having the thread torn down (and having to spawn a new thread).

The purpose of the unstable `thread::catch_panic` function is to solve these
problems by enabling you to catch a panic in Rust before control flow is
returned back over to C. As a refresher, the signature of the function looks
like:

```rust
fn catch_panic<F, R>(f: F) -> thread::Result<R>
    where F: FnOnce() -> R + Send + 'static
```

This function will run the closure `f` and if it panics return `Err(Box<Any>)`.
If the closure doesn't panic it will return `Ok(val)` where `val` is the
returned value of the closure. Most of these aspects "pretty much make sense",
but an odd part about this signature is the `Send` and `'static` bounds on the
closure provided. At a high level, these two bounds are intended to mitigate
problems related to something many programmers call "exception safety". To
understand why let's first briefly review exception safety in Rust.

### Exception Safety

The problem of exception safety often plagues many C++ programmers (and other
languages), and it essentially means that code needs to be ready to handle
exceptional control flow. This primarily matters when an invariant is
temporarily broken in a region of code which can have exceptional control flow.
What this largely boils down to is that a block of code having only one entry
point but possibly many exit points, and invariants need to be upheld on all
exit points.

For Rust this means that code needs to be prepared to handle panics as any
unknown function call can cause a thread to panic. For example:

```rust
let mut foo = true;
bar();
foo = false;
```

It may be intuitive to say that this block of code returns that `foo`'s value is
always `false` (e.g. a local invariant of ours). If, however, the `bar` function
panics, then the block of code will "return" (because of unwinding), but the
value of `foo` is still `true`.  Let's take a look at a more harmful example to
see how this can go wrong:

```
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

While this code may look correct, it's actually not memory safe. If the type
`T`'s `clone` method panics, then this vector will point to uninitialized data.
`Vec` has an internal invariant that the first `len` elements are safe to drop
at any time, and we have broken that invariant temporarily with a call to
`set_len`. If a call to `clone` panics then we'll exit this block before
reaching the end, causing the invariant breakage to be leaked.

The problem with this code is that it's not **exception safe**. There are a
number of common strategies to help mitigate this problem:

* Use a "finally" block or some other equivalent mechanism to restore invariants
  on all exit paths. In Rust this typically manifests itself as a destructor on
  a structure as the compiler will ensure that this is run whenever a panic
  happens.
* Avoid calling code which can panic (e.g. functions with assertions or
  functions with statically unknown implementations) whenever an invariant is
  broken.

In our example of `push_ten_more` we can take the second round of avoiding code
which can panic when an invariant is broken. If we call `set_len` on each
iteration of the loop with `len + i` then the vector's invariant will always bee
respected.

### Catching Exceptions

In languages with `catch` blocks exception unsafe code can often cause problems
more frequently. The core problem here is that shared state in the "try" block
and the "catch" block can end up getting corrupted. Due to a panic possibly
happening at any time, data may not often prepare for the panic and the catch
(or finally) block will then read this corrupt data.

Rust has not had to deal with this problem much because there's no stable way to
catch a panic. One primary area this comes up is dealing with cross-thread
panics, and the standard library poisons mutexes and rwlocks by default to help
deal with this situation. The `catch_panic` function proposed in this RFC,
however, is exactly "catch for Rust". To see how this function is not making
Rust memory unsafe, let's take a look at how memory safety and exception safety
interact.

### Exception Safety and Memory Safety

If this is the first time you've ever heard about exception safety, this may
sound pretty bad! Chances are you haven't considered how Rust code can "exit" at
many points in a function beyond just the points where you wrote down `return`.
The good news is that Rust by default **is still memory safe** in the face of
this exception safety problem.

All safe code in Rust is guaranteed to not cause any memory unsafety due to a
panic. There is never any invalid intermediate state which can then be read due
to a destructor running on a panic. As we've also seen, however, it's possible
to cause memory unsafety through panics when dealing with `unsafe` code. The key
part of this is that you have to have `unsafe` somewhere to inject the memory
unsafety, and you largely just need to worry about exception safety in the
context of unsafe code.

Even though mixing safe Rust and panics cannot cause undefined behavior, it's
possible for a **logical** invariant to be violated as a result of a panic.
These sorts of situations can often become serious bugs and are difficult to
audit for, so it means that exception safety in Rust is unfortunately not a
situation that can be completely sidestepped.

### Exception Safety in Rust

Rust does not provide many primitives today to deal with exception safety, but
it's a situation you'll see handled in many locations when browsing unsafe
collections-related code, for example. One case where Rust does help you with
this is an aspect of Mutexes called [**poisoining**][poison].

[poison]: http://doc.rust-lang.org/std/sync/struct.Mutex.html#poisoning

Poisoning is a mechanism for propagating panics among threads to ensure that
inconsistent state is not read. A mutex becomes poisoned if a thread holds the
lock and then panics. Most usage of a mutex simply `unwrap`s the result of
`lock()`, causing a panic in one thread to be propagated to all others that are
reachable.

A key design aspect of poisoning, however, is that you can opt-out of poisoning.
The `Err` variant of the [`lock` method] provides the ability to gain access to
the mutex anyway. As explained above, exception safety can only lead to memory
unsafety when intermingled with unsafe code. This means that fundamentally
poisoning a Mutex is **not** guaranteeing memory safety, and hence getting
access to a poisoned mutex is not an unsafe operation.

[`lock` method]: http://doc.rust-lang.org/std/sync/struct.Mutex.html#method.lock

Exception safety is rarely considered when writing code in Rust, so the standard
library strives to help out as much as possible when it can. Poisoning mutexes
is a good example of this where ignoring panics in remote threads means that
mutexes could very commonly contain corrupted data (not memory unsafe, just
logically corrupt). There's typically an opt-out to these mechanisms, but by
default the standard library provides them.

### `Send` and `'static` on `catch_panic`

Alright, now that we've got a bit of background, let's explore why these bounds
were originally added to the `catch_panic` function. It was thought that these
two bounds would provide basically the same level of exception safety protection
that spawning a new thread does (e.g. today this requires both of these bounds).
This in theory meant that the addition of `catch_panic` to the standard library
would not exascerbate the concerns of exception safety.

It [was discovered][cp-issue], however, that TLS can be used to bypass this
theoretical "this is the same as spawning a thread" boundary. Using TLS means
that you can share non-`Send` data across the `catch_panic` boundary, meaning
the caller of `catch_panic` may see invalid state.

[cp-issue]: https://github.com/rust-lang/rust/issues/25662

As a result, these two bounds have been called into question, and this RFC is
recommending removing both bounds from the `catch_panic` function.

### Is `catch_panic` unsafe?

With the removal of the two bounds on this function, we can freely share state
across a "panic boundary". This means that we don't always know for sure if
arbitrary data is corrupted or not. As we've seen above, however, if we're only
dealing with safe Rust then this will not lead to memory unsafety. For memory
unsafety to happen it would require interaction with `unsafe` code at which
point the `unsafe` code is responsible for dealing with exception safety.

The standard library has a clear definition for what functions are `unsafe`, and
it's precisely those which can lead to memory unsafety in otherwise safe Rust.
Because that is not the case for `catch_panic` it will not be declared as an
`unsafe` function.

### What about other bounds?

It has been discussed that there may be possible other bounds or mitigation
strategies for `catch_panic` (to help with the TLS problem described above), and
although it's somewhat unclear as to what this may precisely mean it's still the
case that the standard library will want a `catch_panic` with no bounds in
*some* form or another.

The standard library is providing the lowest-level tools to create robust APIs,
and inevitably it should not forbid patterns that are safe. Rust itself does
this via the `unsafe` subset by allowing you to build up a safe abstraction on
unsafe underpinnings. Similarly any bound on `catch_panic` will eventually be
too restrictive for someone even though their usage is 100% safe. As a result
the standard library will always want (and was always going to have) a no-bounds
version of this function.

As a result this RFC proposes not attempting to go through hoops to find a more
restrictive, but more helpful with exception safety, set of bounds for this
function and instead stabilize the no-bounds version.

# Detailed design

Stabilize `std::thread::catch_panic` after removing the `Send` and `'static`
bounds from the closure parameter, modifying the signature to be:

```rust
fn catch_panic<F, R>(f: F) -> thread::Result<R> where F: FnOnce() -> R
```

# Drawbacks

A major drawback of this RFC is that it can mitigate Rust's error handling
story. On one hand this function can be seen as adding exceptions to Rust as
it's now possible to both throw (panic) and catch (`catch_panic`). The track
record of exceptions in languages like C++, Java, and Python hasn't been great,
and a drawing point of Rust for many has been the lack of exceptions. To help
understand what's going on, let's go through a brief overview of error handling
in Rust today:

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
  converted into an error message fit for a human consumer of the top-level
  program.
* `panic`s represent errors that carry no contextual information (except,
  perhaps, debug information). Because they represented an unexpected error,
  they cannot be easily handled by the caller of the function or presented to a
  human consumer of the top-level program (except to say "something unexpected
  has gone wrong").

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
the caller, except perhaps to indicate to the human user that an unexpected
error has occurred.

In terms of heuristics for use:

* `panic`s should rarely if ever be used to report errors that occurred through
  communication with the system or through IO. For example, if a Rust program
  shells out to `rustc`, and `rustc` is not found, it might be tempting to use a
  panic, because the error is unexpected and hard to recover from. However, a
  human consumer of the program would benefit from intermediate code adding
  contextual information about the in-progress operation, and the program could
  report the error in terms a human can understand. While the error is rare,
  **when it happens it is not a programmer error**.
* assertions can produce `panic`s, because the programmer is saying that if the
  assertion fails, it means that he has made an unexpected mistake.

In short, if it would make sense to report an error as a context-free `500
Internal Server Error` or a red an unknown error has occurred in all cases, it's
an appropriate panic.

Another key reason to choose `Result` over a panic is that the compiler is
likely to soon grow an option to map a panic to an abort. This is motivated for
portability, compile time, binary size, and a number of factors, but it
fundamentally means that a library which signals errors via panics (and relies
on consumers using `catch_panic`) will not be usable in this context.

### Will Rust have exceptions?

After reviewing the cases for `Result` and `panic`, there's still clearly a
niche that both of these two systems are filling, so it's not the case that we
want to scrap one for the other. Rust will indeed have the ability to catch
exceptions to a greater extent than it does today with this RFC, but idiomatic
Rust will continue to follow the above rules for when to use a panic vs a result.

It's likely that the `catch_panic` function will only be used where it's
absolutely necessary, like FFI boundaries, instead of a general-purpose error
handling mechanism in all code.

# Alternatives

One alternative, which is somewhat more of an addition, is to have the standard
library entirely abandon all exception safety mitigation tactics. As explained
in the motivation section, exception safety will not lead to memory unsafety
unless paired with unsafe code, so it is perhaps within the realm of possibility
to remove the tactics of poisoning from mutexes and simply require that
consumers deal with exception safety 100%.

This alternative is often motivated by saying that there are holes in our
poisoning story or the problem space is too large to tackle via targeted APIs.
This section will look a little bit more in detail about what's going on here.

For the purpose of this discussion, let's use the term *dangerous* to
refer to code that can produce problems related to exception safety. Exception
safety means we're exposing the following possibly dangerous situation:

> Dangerous code allows code that uses interior mutability to be interrupted in
> the process of making a mutation, and then allow other code to see the
> incomplete change.

Today, most Rust code is protected from this danger from two angles:

* If a piece of code acquires interior mutability through &mut and a panic
  occurs, that panic will propagate through the owner of the original value.
  Since there can be no outstanding & references to the same value, nobody can
  see the incomplete change.
* If a piece of code acquires interior mutability through Mutex and a
  panic occurs, attempts by another thread to read the value through
  normal means will propagate the panic.

There are areas in Rust that are not covered by these cases:

* RefCell (especially with destructors) allows code to get access to a value
  with an incomplete change.
* Generally speaking, destructors can observe an incomplete change.
* The Mutex API provides an alternate mechanism of reading a value with an
  incomplete change.
* The proposed `catch_panic` API allows the propagation of panics to a boundary
  that does not have any ownership restrictions.

One open question that this question affects:

* Should a theoretical `Thread::scoped` API propagate panics?

Looking at these cases that aren't covered in Rust by default, and assuming that
`Thread::scoped` propagates panics by default (with an analogous API to
`PoisonError::into_inner`), we get a table that looks like:

![img](https://www.evernote.com/l/AAJdvryuzOVFrakUiK6i0IBASP7wysYHN0sB/image.png)

The main point here is that although this problem space seems sprawling, it is,
in reality, restricted to interior mutability. Enumerating the "dangerous" APIs
seems to be a tractable problem. Calling `RefCell` and `catch_panic` "dangerous"
(with the incomplete mutation problem) would not be problematic. `Mutex` or
`Thread::scoped` would not be dangerous because of the benefits associated with
detecting panics across threads, and this aligns with the table above. Note that
implementations of Drop, because they run during stack unwinding, should be
considered "dangerous" for the purposes of this summary.

It may not be surprising that the threaded APIs ended up being protected via
APIs, because this kind of sharing is fundamental to threaded code. Making
them "dangerous" would make almost anything you would want to do with threads
"dangerous", and instead we ask users to learn about the danger only when they
try to access the possibly dangerous data.

In contrast, both `RefCell` and `catch_panic` are more niche tools, making it
reasonable to ask users to learn about the danger when they begin using the
tools in the first place, and then making the access more ergonomic. Despite
labeling being "dangerous" there are strategies to mitigate this such as
building abstractions on top of these primitive which only use `RefCell` or
`catch_panic` as an implementation detail. These higher-level abstractions will
have fewer edge cases and risks associated with them.

# Unresolved questions

None currently.
