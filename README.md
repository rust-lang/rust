% The Unsafe Rust Programming Language

# NOTE: This is a draft document, and may contain serious errors

**This document is about advanced functionality and low-level development practices
in the Rust Programming Language. Most of the things discussed won't matter
to the average Rust programmer. However if you wish to correctly write unsafe
code in Rust, this text contains invaluable information.**

The Unsafe Rust Programming Language (TURPL) seeks to complement
[The Rust Programming Language Book][trpl] (TRPL).
Where TRPL introduces the language and teaches the basics, TURPL dives deep into
the specification of the language, and all the nasty bits necessary to write
Unsafe Rust. TURPL does not assume you have read TRPL, but does assume you know
the basics of the language and systems programming. We will not explain the
stack or heap, we will not explain the basic syntax.





# Meet Safe and Unsafe

Safe and Unsafe are Rust's chief engineers.

TODO: ADORABLE PICTURES OMG

Unsafe handles all the dangerous internal stuff. They build the foundations
and handle all the dangerous materials. By all accounts, Unsafe is really a bit
unproductive, because the nature of their work means that they have to spend a
lot of time checking and double-checking everything. What if there's an earthquake
on a leap year? Are we ready for that? Unsafe better be, because if they get
*anything* wrong, everything will blow up! What Unsafe brings to the table is
*quality*, not quantity. Still, nothing would ever get done if everything was
built to Unsafe's standards!

That's where Safe comes in. Safe has to handle *everything else*. Since Safe needs
to *get work done*, they've grown to be fairly carless and clumsy! Safe doesn't worry
about all the crazy eventualities that Unsafe does, because life is too short to deal
with leap-year-earthquakes. Of course, this means there's some jobs that Safe just
can't handle. Safe is all about quantity over quality.

Unsafe loves Safe to bits, but knows that tey *can never trust them to do the
right thing*. Still, Unsafe acknowledges that not every problem needs quite the
attention to detail that they apply. Indeed, Unsafe would *love* if Safe could do
*everything* for them. To accomplish this, Unsafe spends most of their time
building *safe abstractions*. These abstractions handle all the nitty-gritty
details for Safe, and choose good defaults so that the simplest solution (which
Safe will inevitably use) is usually the *right* one. Once a safe abstraction is
built, Unsafe ideally needs to never work on it again, and Safe can blindly use
it in all their work.

Unsafe's attention to detail means that all the things that they mark as ok for
Safe to use can be combined in arbitrarily ridiculous ways, and all the rules
that Unsafe is forced to uphold will never be violated. If they *can* be violated
by Safe, that means *Unsafe*'s the one in the wrong. Safe can work carelessly,
knowing that if anything blows up, it's not *their* fault. Safe can also call in
Unsafe at any time if there's a hard problem they can't quite work out, or if they
can't meet the client's quality demands. Of course, Unsafe will beg and plead Safe
to try their latest safe abstraction first!

In addition to being adorable, Safe and Unsafe are what makes Rust possible.
Rust can be thought of as two different languages: Safe Rust, and Unsafe Rust.
Any time someone opines the guarantees of Rust, they are almost surely talking about
Safe. However Safe is not sufficient to write every program. For that,
we need the Unsafe superset.

Most fundamentally, writing bindings to other languages
(such as the C exposed by your operating system) is never going to be safe. Rust
can't control what other languages do to program execution! However Unsafe is
also necessary to construct fundamental abstractions where the type system is not
sufficient to automatically prove what you're doing is sound.

Indeed, the Rust standard library is implemented in Rust, and it makes substantial
use of Unsafe for implementing IO, memory allocation, collections,
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

It cannot be emphasized enough that Unsafe should be regarded as an exceptional
thing, not a normal one. Unsafe is often the domain of *fundamental libraries*: anything that needs
to make FFI bindings or define core abstractions. These fundamental libraries then expose
a safe interface for intermediate libraries and applications to build upon. And these
safe interfaces make an important promise: if your application segfaults, it's not your
fault. *They* have a bug.

And really, how is that different from *any* safe language? Python, Ruby, and Java libraries
can internally do all sorts of nasty things. The languages themselves are no
different. Safe languages *regularly* have bugs that cause critical vulnerabilities.
The fact that Rust is written with a healthy spoonful of Unsafe is no different.
However it *does* mean that Rust doesn't need to fall back to the pervasive unsafety of
C to do the nasty things that need to get done.





# What do Safe and Unsafe really mean?

Rust cares about preventing the following things:

* Dereferencing null or dangling pointers
* Reading [uninitialized memory][]
* Breaking the [pointer aliasing rules][]
* Producing invalid primitive values:
    * dangling/null references
    * a `bool` that isn't 0 or 1
    * an undefined `enum` discriminant
    * a `char` larger than char::MAX (TODO: check if stronger restrictions apply)
    * A non-utf8 `str`
* Unwinding into another language
* Causing a [data race][]
* Invoking Misc. Undefined Behaviour (in e.g. compiler intrinsics)

That's it. That's all the Undefined Behaviour in Rust. Libraries are free to
declare arbitrary requirements if they could transitively cause memory safety
issues, but it all boils down to the above actions. Rust is otherwise
quite permisive with respect to other dubious operations. Rust considers it
"safe" to:

* Deadlock
* Have a Race Condition
* Leak memory
* Fail to call destructors
* Overflow integers
* Delete the production database

However any program that does such a thing is *probably* incorrect. Rust
provides lots of tools to make doing these things rare, but these problems are
considered impractical to categorically prevent.

Rust models the seperation between Safe and Unsafe with the `unsafe` keyword.
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
[destructors][] for details.

Some examples of unsafe functions:

* `slice::get_unchecked` will perform unchecked indexing, allowing memory
  safety to be freely violated.
* `ptr::offset` is an intrinsic that invokes Undefined Behaviour if it is
  not "in bounds" as defined by LLVM (see the lifetimes section for details).
* `mem::transmute` reinterprets some value as having the given type,
  bypassing type safety in arbitrary ways. (see [conversions][] for details)
* All FFI functions are `unsafe` because they can do arbitrary things.
  C being an obvious culprit, but generally any language can do something
  that Rust isn't happy about.

As of Rust 1.0 there are exactly two unsafe traits:

* `Send` is a marker trait (it has no actual API) that promises implementors
  are safe to send to another thread.
* `Sync` is a marker trait that promises that threads can safely share
  implementors through a shared reference.

The need for unsafe traits boils down to the fundamental lack of trust that Unsafe
has for Safe. All safe traits are free to declare arbitrary contracts, but because
implementing them is a job for Safe, Unsafe can't trust those contracts to actually
be upheld.

For instance Rust has `PartialOrd` and `Ord` traits to try to differentiate
between types which can "just" be compared, and those that actually implement a
*total* ordering. Pretty much every API that wants to work with data that can be
compared *really* wants Ord data. For instance, a sorted map like BTreeMap
*doesn't even make sense* for partially ordered types. If you claim to implement
Ord for a type, but don't actually provide a proper total ordering, BTreeMap will
get *really confused* and start making a total mess of itself. Data that is
inserted may be impossible to find!

But that's ok. BTreeMap is safe, so it guarantees that even if you give it a
*completely* garbage Ord implementation, it will still do something *safe*. You
won't start reading uninitialized memory or unallocated memory. In fact, BTreeMap
manages to not actually lose any of your data. When the map is dropped, all the
destructors will be successfully called! Hooray!

However BTreeMap is implemented using a modest spoonful of Unsafe (most collections
are). That means that it is not necessarily *trivially true* that a bad Ord
implementation will make BTreeMap behave safely. Unsafe most be sure not to rely
on Ord *where safety is at stake*, because Ord is provided by Safe, and memory
safety is not Safe's responsibility to uphold. *It must be impossible for Safe
code to violate memory safety*.

But wouldn't it be grand if there was some way for Unsafe to trust *some* trait
contracts *somewhere*? This is the problem that unsafe traits tackle: by marking
*the trait itself* as unsafe *to implement*, Unsafe can trust the implementation
to be correct (because Unsafe can trust themself).

Rust has traditionally avoided making traits unsafe because it makes Unsafe
pervasive, which is not desirable. Send and Sync are unsafe is because
thread safety is a *fundamental property* that Unsafe cannot possibly hope to
defend against in the same way it would defend against a bad Ord implementation.
The only way to possibly defend against thread-unsafety would be to *not use
threading at all*. Making every operation atomic isn't even sufficient, because
it's possible for complex invariants between disjoint locations in memory.

Even concurrent paradigms that are traditionally regarded as Totally Safe like
message passing implicitly rely on some notion of thread safety -- are you
really message-passing if you send a *pointer*? Send and Sync therefore require
some *fundamental* level of trust that Safe code can't provide, so they must be
unsafe to implement. To help obviate the pervasive unsafety that this would
introduce, Send (resp. Sync) is *automatically* derived for all types composed only
of Send (resp. Sync) values. 99% of types are Send and Sync, and 99% of those
never actually say it (the remaining 1% is overwhelmingly synchronization
primitives).




# Working with Unsafe

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
// We currently live in a nice imaginary world of only positive fixed-size
// types.
impl<T> Vec<T> {
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



[trpl]: https://doc.rust-lang.org/book/
[pointer aliasing rules]: lifetimes.html#references
[uninitialized memory]: uninitialized.html
[data race]: concurrency.html
[destructors]: raii.html
[conversions]: conversions.html
