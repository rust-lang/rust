% What do Safe and Unsafe really mean?

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



[pointer aliasing rules]: lifetimes.html#references
[uninitialized memory]: uninitialized.html
[data race]: concurrency.html
[destructors]: raii.html
[conversions]: conversions.html