% How Safe and Unsafe Interact

So what's the relationship between Safe and Unsafe Rust? How do they interact?

Rust models the separation between Safe and Unsafe Rust with the `unsafe`
keyword, which can be thought as a sort of *foreign function interface* (FFI)
between Safe and Unsafe Rust. This is the magic behind why we can say Safe Rust
is a safe language: all the scary unsafe bits are relegated exclusively to FFI
*just like every other safe language*.

However because one language is a subset of the other, the two can be cleanly
intermixed as long as the boundary between Safe and Unsafe Rust is denoted with
the `unsafe` keyword. No need to write headers, initialize runtimes, or any of
that other FFI boiler-plate.

There are several places `unsafe` can appear in Rust today, which can largely be
grouped into two categories:

* There are unchecked contracts here. To declare you understand this, I require
you to write `unsafe` elsewhere:
    * On functions, `unsafe` is declaring the function to be unsafe to call.
      Users of the function must check the documentation to determine what this
      means, and then have to write `unsafe` somewhere to identify that they're
      aware of the danger.
    * On trait declarations, `unsafe` is declaring that *implementing* the trait
      is an unsafe operation, as it has contracts that other unsafe code is free
      to trust blindly. (More on this below.)

* I am declaring that I have, to the best of my knowledge, adhered to the
unchecked contracts:
    * On trait implementations, `unsafe` is declaring that the contract of the
      `unsafe` trait has been upheld.
    * On blocks, `unsafe` is declaring any unsafety from an unsafe
      operation within to be handled, and therefore the parent function is safe.

There is also `#[unsafe_no_drop_flag]`, which is a special case that exists for
historical reasons and is in the process of being phased out. See the section on
[drop flags] for details.

Some examples of unsafe functions:

* `slice::get_unchecked` will perform unchecked indexing, allowing memory
  safety to be freely violated.
* every raw pointer to sized type has intrinsic `offset` method that invokes
  Undefined Behaviour if it is not "in bounds" as defined by LLVM.
* `mem::transmute` reinterprets some value as having the given type,
  bypassing type safety in arbitrary ways. (see [conversions] for details)
* All FFI functions are `unsafe` because they can do arbitrary things.
  C being an obvious culprit, but generally any language can do something
  that Rust isn't happy about.

As of Rust 1.0 there are exactly two unsafe traits:

* `Send` is a marker trait (it has no actual API) that promises implementors
  are safe to send (move) to another thread.
* `Sync` is a marker trait that promises that threads can safely share
  implementors through a shared reference.

The need for unsafe traits boils down to the fundamental property of safe code:

**No matter how completely awful Safe code is, it can't cause Undefined
Behavior.**

This means that Unsafe Rust, **the royal vanguard of Undefined Behavior**, has to be
*super paranoid* about generic safe code. To be clear, Unsafe Rust is totally free to trust
specific safe code. Anything else would degenerate into infinite spirals of
paranoid despair. In particular it's generally regarded as ok to trust the standard library
to be correct. `std` is effectively an extension of the language, and you
really just have to trust the language. If `std` fails to uphold the
guarantees it declares, then it's basically a language bug.

That said, it would be best to minimize *needlessly* relying on properties of
concrete safe code. Bugs happen! Of course, I must reinforce that this is only
a concern for Unsafe code. Safe code can blindly trust anyone and everyone
as far as basic memory-safety is concerned.

On the other hand, safe traits are free to declare arbitrary contracts, but because
implementing them is safe, unsafe code can't trust those contracts to actually
be upheld. This is different from the concrete case because *anyone* can
randomly implement the interface. There is something fundamentally different
about trusting a particular piece of code to be correct, and trusting *all the
code that will ever be written* to be correct.

For instance Rust has `PartialOrd` and `Ord` traits to try to differentiate
between types which can "just" be compared, and those that actually implement a
total ordering. Pretty much every API that wants to work with data that can be
compared wants Ord data. For instance, a sorted map like BTreeMap
*doesn't even make sense* for partially ordered types. If you claim to implement
Ord for a type, but don't actually provide a proper total ordering, BTreeMap will
get *really confused* and start making a total mess of itself. Data that is
inserted may be impossible to find!

But that's okay. BTreeMap is safe, so it guarantees that even if you give it a
completely garbage Ord implementation, it will still do something *safe*. You
won't start reading uninitialized or unallocated memory. In fact, BTreeMap
manages to not actually lose any of your data. When the map is dropped, all the
destructors will be successfully called! Hooray!

However BTreeMap is implemented using a modest spoonful of Unsafe Rust (most collections
are). That means that it's not necessarily *trivially true* that a bad Ord
implementation will make BTreeMap behave safely. BTreeMap must be sure not to rely
on Ord *where safety is at stake*. Ord is provided by safe code, and safety is not
safe code's responsibility to uphold.

But wouldn't it be grand if there was some way for Unsafe to trust some trait
contracts *somewhere*? This is the problem that unsafe traits tackle: by marking
*the trait itself* as unsafe to implement, unsafe code can trust the implementation
to uphold the trait's contract. Although the trait implementation may be
incorrect in arbitrary other ways.

For instance, given a hypothetical UnsafeOrd trait, this is technically a valid
implementation:

```rust
# use std::cmp::Ordering;
# struct MyType;
# unsafe trait UnsafeOrd { fn cmp(&self, other: &Self) -> Ordering; }
unsafe impl UnsafeOrd for MyType {
    fn cmp(&self, other: &Self) -> Ordering {
        Ordering::Equal
    }
}
```

But it's probably not the implementation you want.

Rust has traditionally avoided making traits unsafe because it makes Unsafe
pervasive, which is not desirable. The reason Send and Sync are unsafe is because thread
safety is a *fundamental property* that unsafe code cannot possibly hope to defend
against in the same way it would defend against a bad Ord implementation. The
only way to possibly defend against thread-unsafety would be to *not use
threading at all*. Making every load and store atomic isn't even sufficient,
because it's possible for complex invariants to exist between disjoint locations
in memory. For instance, the pointer and capacity of a Vec must be in sync.

Even concurrent paradigms that are traditionally regarded as Totally Safe like
message passing implicitly rely on some notion of thread safety -- are you
really message-passing if you pass a pointer? Send and Sync therefore require
some fundamental level of trust that Safe code can't provide, so they must be
unsafe to implement. To help obviate the pervasive unsafety that this would
introduce, Send (resp. Sync) is automatically derived for all types composed only
of Send (resp. Sync) values. 99% of types are Send and Sync, and 99% of those
never actually say it (the remaining 1% is overwhelmingly synchronization
primitives).




[drop flags]: drop-flags.html
[conversions]: conversions.html
