% Language FAQ

# General language issues

## Safety oriented

* Memory safe: no null pointers, dangling pointers, use-before-initialize or use-after-move
* Expressive mutability control. Immutable by default, statically verified freezing for Owned types
* No shared mutable state across tasks
* Dynamic execution safety: task failure / unwinding, trapping, RAII / dtors
* Safe interior pointer types with lifetime analysis

## Concurrency and efficiency oriented

* Lightweight tasks (coroutines) with expanding stacks
* Fast asynchronous, copyless message passing
* Optional garbage collected pointers
* All types may be explicitly allocated on the stack or interior to other types
* Static, native compilation using LLVM
* Direct and simple interface to C code

## Practicality oriented

* Multi-paradigm: pure-functional, concurrent-actor, imperative-procedural, OO
 * First-class functions, cheap non-escaping closures
 * Algebraic data types (called enums) with pattern matching
 * Method implementations on any type
 * Traits, which share aspects of type classes and interfaces
* Multi-platform. Developed on Windows, Linux, OS X
* UTF-8 strings, assortment of machine-level types
* Works with existing native toolchains, GDB, Valgrind, Instruments, etc
* Rule-breaking is allowed if explicit about where and how

## What does it look like?

The syntax is still evolving, but here's a snippet from the hash map in core::hashmap.

~~~
struct LinearMap<K,V> {
    k0: u64,
    k1: u64,
    resize_at: uint,
    size: uint,
    buckets: ~[Option<Bucket<K,V>>],
}

enum SearchResult {
    FoundEntry(uint), FoundHole(uint), TableFull
}

fn linear_map_with_capacity<K:Eq + Hash,V>(capacity: uint) -> LinearMap<K,V> {
    let r = rand::Rng();
    linear_map_with_capacity_and_keys(r.gen_u64(), r.gen_u64(), capacity)
}

impl<K:Hash + IterBytes + Eq, V> LinearMap<K,V> {

    fn contains_key(&self, k: &K) -> bool {
        match self.bucket_for_key(self.buckets, k) {
            FoundEntry(_) => true,
            TableFull | FoundHole(_) => false
        }
    }

    fn clear(&mut self) {
        for bkt in self.buckets.mut_iter() {
            *bkt = None;
        }
        self.size = 0;
    }

...
}
~~~

## Are there any big programs written in it yet? I want to read big samples.

There aren't many large programs yet. The Rust [compiler][rustc], 60,000+ lines at the time of writing, is written in Rust. As the oldest body of Rust code it has gone through many iterations of the language, and some parts are nicer to look at than others. It may not be the best code to learn from, but [borrowck] and [resolve] were written recently.

[rustc]: https://github.com/mozilla/rust/tree/master/src/librustc
[resolve]: https://github.com/mozilla/rust/blob/master/src/librustc/middle/resolve.rs
[borrowck]: https://github.com/mozilla/rust/blob/master/src/librustc/middle/borrowck/

A research browser engine called [Servo][servo], currently 30,000+ lines across more than a dozen crates, will be exercising a lot of Rust's distinctive type-system and concurrency features, and integrating many native libraries.

[servo]: https://github.com/mozilla/servo

Some examples that demonstrate different aspects of the language:

* [sprocketnes], an NES emulator with no GC, using modern Rust conventions
* The language's general-purpose [hash] function, SipHash-2-4. Bit twiddling, OO, macros
* The standard library's [HashMap], a sendable hash map in an OO style
* The extra library's [json] module. Enums and pattern matching

[sprocketnes]: https://github.com/pcwalton/sprocketnes
[hash]: https://github.com/mozilla/rust/blob/master/src/libstd/hash.rs
[HashMap]: https://github.com/mozilla/rust/blob/master/src/libstd/hashmap.rs
[json]: https://github.com/mozilla/rust/blob/master/src/libextra/json.rs

You may also be interested in browsing [GitHub's Rust][github-rust] page.

[github-rust]: https://github.com/languages/Rust

## Does it run on Windows?

Yes. All development happens in lock-step on all 3 target platforms. Using MinGW, not Cygwin. Note that the windows implementation currently has some limitations: in particular tasks [cannot unwind on windows][unwind], and all Rust executables [require a MinGW installation at runtime][libgcc].

[unwind]: https://github.com/mozilla/rust/issues/908
[libgcc]: https://github.com/mozilla/rust/issues/1603

## Have you seen this Google language, Go? How does Rust compare?

Rust and Go have similar syntax and task models, but they have very different type systems. Rust is distinguished by greater type safety and memory safety guarantees, more control over memory layout, and robust generics.

Rust has several key features that aren't shared by Go:

* No shared mutable state - Shared mutable state allows data races, a large class of bad bugs. In Rust there is no sharing of mutable data, but ownership of data can be efficiently transferred between tasks.
* Minimal GC impact - By not having shared mutable data, Rust can avoid global GC, hence Rust never stops the world to collect garbage. With multiple allocation options, individual tasks can completely avoid GC.
* No null pointers - Accidentally dereferencing null pointers is a big bummer, so Rust doesn't have them.
* Type parametric code - Generics prove useful time and again, though they are inevitably complex to greater or lesser degrees.

Some of Rust's advantages come at the cost of a more intricate type system than Go's.

Go has its own strengths and in particular has a great user experience that Rust still lacks.

## I like the language but it really needs _$somefeature_.

At this point we are focusing on removing and stabilizing features rather than adding them. File a bug if you think it's important in terms of meeting the existing goals or making the language passably usable. Reductions are more interesting than additions, though.

# Specific language issues

## Is it OO? How do I do this thing I normally do in an OO language?

It is multi-paradigm. Not everything is shoe-horned into a single abstraction. Many things you can do in OO languages you can do in Rust, but not everything, and not always using the same abstraction you're accustomed to.

## How do you get away with "no null pointers"?

Data values in the language can only be constructed through a fixed set of initializer forms. Each of those forms requires that its inputs already be initialized. A liveness analysis ensures that local variables are initialized before use.

## What is the relationship between a module and a crate?

* A crate is a top-level compilation unit that corresponds to a single loadable object.
* A module is a (possibly nested) unit of name-management inside a crate.
* A crate contains an implicit, un-named top-level module.
* Recursive definitions can span modules, but not crates.
* Crates do not have global names, only a set of non-unique metadata tags.
* There is no global inter-crate namespace; all name management occurs within a crate.
 * Using another crate binds the root of _its_ namespace into the user's namespace.

## Why is failure unwinding non-recoverable within a task? Why not try to "catch exceptions"?

In short, because too few guarantees could be made about the dynamic environment of the catch block, as well as invariants holding in the unwound heap, to be able to safely resume; we believe that other methods of signalling and logging errors are more appropriate, with tasks playing the role of a "hard" isolation boundary between separate heaps.

Rust provides, instead, three predictable and well-defined options for handling any combination of the three main categories of "catch" logic:

* Failure _logging_ is done by the integrated logging subsystem.
* _Recovery_ after a failure is done by trapping a task failure from _outside_ the task, where other tasks are known to be unaffected.
* _Cleanup_ of resources is done by RAII-style objects with destructors.

Cleanup through RAII-style destructors is more likely to work than in catch blocks anyways, since it will be better tested (part of the non-error control paths, so executed all the time).

## Why aren't modules type-parametric?

We want to maintain the option to parametrize at runtime. We may make eventually change this limitation, but initially this is how type parameters were implemented.

## Why aren't values type-parametric? Why only items?

Doing so would make type inference much more complex, and require the implementation strategy of runtime parametrization.

## Why are enumerations nominal and closed?

We don't know if there's an obvious, easy, efficient, stock-textbook way of supporting open or structural disjoint unions. We prefer to stick to language features that have an obvious and well-explored semantics.

## Why aren't channels synchronous?

There's a lot of debate on this topic; it's easy to find a proponent of default-sync or default-async communication, and there are good reasons for either. Our choice rests on the following arguments:

* Part of the point of isolating tasks is to decouple tasks from one another, such that assumptions in one task do not cause undue constraints (or bugs, if violated!) in another. Temporal coupling is as real as any other kind; async-by-default relaxes the default case to only _causal_ coupling.
* Default-async supports buffering and batching communication, reducing the frequency and severity of task-switching and inter-task / inter-domain synchronization.
* Default-async with transmittable channels is the lowest-level building block on which more-complex synchronization topologies and strategies can be built; it is not clear to us that the majority of cases fit the 2-party full-synchronization pattern rather than some more complex multi-party or multi-stage scenario. We did not want to force all programs to pay for wiring the former assumption into all communications.

## Why are channels half-duplex (one-way)?

Similar to the reasoning about default-sync: it wires fewer assumptions into the implementation, that would have to be paid by all use-cases even if they actually require a more complex communication topology.

## Why are strings UTF-8 by default? Why not UCS2 or UCS4?

The `str` type is UTF-8 because we observe more text in the wild in this encoding -- particularly in network transmissions, which are endian-agnostic -- and we think it's best that the default treatment of I/O not involve having to recode codepoints in each direction.

This does mean that indexed access to a Unicode codepoint inside a `str` value is an O(n) operation. On the one hand, this is clearly undesirable; on the other hand, this problem is full of trade-offs and we'd like to point a few important qualifications:

* Scanning a `str` for ASCII-range codepoints can still be done safely octet-at-a-time, with each indexing operation pulling out a `u8` costing only O(1) and producing a value that can be cast and compared to an ASCII-range `char`. So if you're (say) line-breaking on `'\n'`, octet-based treatment still works. UTF8 was well-designed this way.
* Most "character oriented" operations on text only work under very restricted language assumptions sets such as "ASCII-range codepoints only". Outside ASCII-range, you tend to have to use a complex (non-constant-time) algorithm for determining linguistic-unit (glyph, word, paragraph) boundaries anyways. We recommend using an "honest" linguistically-aware, Unicode-approved algorithm.
* The `char` type is UCS4. If you honestly need to do a codepoint-at-a-time algorithm, it's trivial to write a `type wstr = [char]`, and unpack a `str` into it in a single pass, then work with the `wstr`. In other words: the fact that the language is not "decoding to UCS4 by default" shouldn't stop you from decoding (or re-encoding any other way) if you need to work with that encoding.

## Why are strings, vectors etc. built-in types rather than (say) special kinds of trait/impl?

In each case there is one or more operator, literal constructor, overloaded use or integration with a built-in control structure that makes us think it would be awkward to phrase the type in terms of more-general type constructors. Same as, say, with numbers! But this is partly an aesthetic call, and we'd be willing to look at a worked-out proposal for eliminating or rephrasing these special cases.

## Can Rust code call C code?

Yes. Since C code typically expects a larger stack than Rust code does, the stack may grow before the call. The Rust domain owning the task that makes the call will block for the duration of the call, so if the call is likely to be long-lasting, you should consider putting the task in its own domain (thread or process).

## Can C code call Rust code?

Yes. The Rust code has to be exposed via an `extern` declaration, which makes it C-ABI compatible. Its address can then be taken and passed to C code. When C calls Rust back, the callback occurs in very restricted circumstances.

## How do Rust's task stacks work?

They start small (ideally in the hundreds of bytes) and expand dynamically by calling through special frames that allocate new stack segments. This is known as the "spaghetti stack" approach.

## What is the difference between a managed box pointer (`@`) and an owned box pointer (`~`)?

* Managed boxes live in the garbage collected task-local heap
* Owned boxes live in the global exchange heap
* Managed boxes may be referred to by multiple managed box references
* Owned boxes have unique ownership and there may only be a single unique pointer to a unique box at a time
* Managed boxes may not be shared between tasks
* Owned boxes may be transferred (moved) between tasks

## What is the difference between a reference (`&`) and managed and owned boxes?

* References point to the interior of a stack _or_ heap allocation
* References can only be formed when it will provably be outlived by the referent
* References to managed box pointers keep the managed boxes alive
* References to owned boxes prevent their ownership from being transferred
* References employ region-based alias analysis to ensure correctness

## Why aren't function signatures inferred? Why only local slots?

* Mechanically, it simplifies the inference algorithm; inference only requires looking at one function at a time.
* The same simplification goes double for human readers. A reader does not need an IDE running an inference algorithm across an entire crate to be able to guess at a function's argument types; it's always explicit and nearby.
* Parameters in Rust can be passed by reference or by value. We can't automatically infer which one the programmer means.

## Why does a type parameter need explicit trait bounds to invoke methods on it, when C++ templates do not?

* Requiring explicit bounds means that the compiler can type-check the code at the point where the type-parametric item is *defined*, rather than delaying to when its type parameters are instantiated.  You know that *any* set of type parameters fulfilling the bounds listed in the API will compile. It's an enforced minimal level of documentation, and results in very clean error messages.

* Scoping of methods is also a problem.  C++ needs [Koenig (argument dependent) lookup](http://en.wikipedia.org/wiki/Argument-dependent_name_lookup), which comes with its own host of problems. Explicit bounds avoid this issue: traits are explicitly imported and then used as bounds on type parameters, so there is a clear mapping from the method to its implementation (via the trait and the instantiated type).

  * Related to the above point: since a parameter explicitly names its trait bounds, a single type is able to implement traits whose sets of method names overlap, cleanly and unambiguously.

* There is further discussion on [this thread on the Rust mailing list](https://mail.mozilla.org/pipermail/rust-dev/2013-September/005603.html).

## Will Rust implement automatic semicolon insertion, like in Go?

For simplicity, we do not plan to do so. Implementing automatic semicolon insertion for Rust would be tricky because the absence of a trailing semicolon means "return a value".
