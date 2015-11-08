% The Rust Language FAQ

## Are there any big programs written in it yet? I want to read big samples.

There aren't many large programs yet. The Rust [compiler][rustc], 60,000+ lines at the time of writing, is written in Rust. As the oldest body of Rust code it has gone through many iterations of the language, and some parts are nicer to look at than others. It may not be the best code to learn from, but [borrowck] and [resolve] were written recently.

[rustc]: https://github.com/rust-lang/rust/tree/master/src/librustc
[resolve]: https://github.com/rust-lang/rust/tree/master/src/librustc_resolve
[borrowck]: https://github.com/rust-lang/rust/tree/master/src/librustc_borrowck/borrowck

A research browser engine called [Servo][servo], currently 30,000+ lines across more than a dozen crates, will be exercising a lot of Rust's distinctive type-system and concurrency features, and integrating many native libraries.

[servo]: https://github.com/servo/servo

Some examples that demonstrate different aspects of the language:

* [sprocketnes], an NES emulator with no GC, using modern Rust conventions
* The language's general-purpose [hash] function, SipHash-2-4. Bit twiddling, OO, macros
* The standard library's [HashMap], a sendable hash map in an OO style
* The standard library's [json] module. Enums and pattern matching

[sprocketnes]: https://github.com/pcwalton/sprocketnes
[hash]: https://github.com/rust-lang/rust/tree/master/src/libcore/hash
[HashMap]: https://github.com/rust-lang/rust/tree/master/src/libstd/collections/hash
[json]: https://github.com/rust-lang/rust/blob/master/src/libserialize/json.rs

You may also be interested in browsing [trending Rust repositories][github-rust] on GitHub.

[github-rust]: https://github.com/trending?l=rust

## Is anyone using Rust in production?

Yes. For example (incomplete):

* [OpenDNS](http://labs.opendns.com/2013/10/04/zeromq-helping-us-block-malicious-domains/)
* [Skylight](http://skylight.io)
* [wit.ai](https://github.com/wit-ai/witd)
* [Codius](https://codius.org/blog/codius-rust/)
* [MaidSafe](http://maidsafe.net/)
* [Terminal.com](https://terminal.com)

## Does it run on Windows?

Yes. All development happens in lockstep on all 3 target platforms (using MinGW, not Cygwin).

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

## Why is panic unwinding non-recoverable within a thread? Why not try to "catch exceptions"?

In short, because too few guarantees could be made about the dynamic environment of the catch block, as well as invariants holding in the unwound heap, to be able to safely resume; we believe that other methods of signalling and logging errors are more appropriate, with threads playing the role of a "hard" isolation boundary between separate heaps.

Rust provides, instead, three predictable and well-defined options for handling any combination of the three main categories of "catch" logic:

* Failure _logging_ is done by the integrated logging subsystem.
* _Recovery_ after a panic is done by trapping a thread panic from _outside_
  the thread, where other threads are known to be unaffected.
* _Cleanup_ of resources is done by RAII-style objects with destructors.

Cleanup through RAII-style destructors is more likely to work than in catch blocks anyways, since it will be better tested (part of the non-error control paths, so executed all the time).

## Why aren't modules type-parametric?

We want to maintain the option to parameterize at runtime. We may eventually change this limitation, but initially this is how type parameters were implemented.

## Why aren't values type-parametric? Why only items?

Doing so would make type inference much more complex, and require the implementation strategy of runtime parameterization.

## Why are enumerations nominal and closed?

We don't know if there's an obvious, easy, efficient, stock-textbook way of supporting open or structural disjoint unions. We prefer to stick to language features that have an obvious and well-explored semantics.

## Why aren't channels synchronous?

There's a lot of debate on this topic; it's easy to find a proponent of default-sync or default-async communication, and there are good reasons for either. Our choice rests on the following arguments:

* Part of the point of isolating threads is to decouple threads from one another, such that assumptions in one thread do not cause undue constraints (or bugs, if violated!) in another. Temporal coupling is as real as any other kind; async-by-default relaxes the default case to only _causal_ coupling.
* Default-async supports buffering and batching communication, reducing the frequency and severity of thread-switching and inter-thread / inter-domain synchronization.
* Default-async with transmittable channels is the lowest-level building block on which more-complex synchronization topologies and strategies can be built; it is not clear to us that the majority of cases fit the 2-party full-synchronization pattern rather than some more complex multi-party or multi-stage scenario. We did not want to force all programs to pay for wiring the former assumption into all communications.

## Why are channels half-duplex (one-way)?

Similar to the reasoning about default-sync: it wires fewer assumptions into the implementation, that would have to be paid by all use-cases even if they actually require a more complex communication topology.

## Why are strings UTF-8 by default? Why not UCS2 or UCS4?

The `str` type is UTF-8 because we observe more text in the wild in this encoding – particularly in network transmissions, which are endian-agnostic – and we think it's best that the default treatment of I/O not involve having to recode codepoints in each direction.

This does mean that indexed access to a Unicode codepoint inside a `str` value is an O(n) operation. On the one hand, this is clearly undesirable; on the other hand, this problem is full of trade-offs and we'd like to point a few important qualifications:

* Scanning a `str` for ASCII-range codepoints can still be done safely octet-at-a-time. If you use `.as_bytes()`, pulling out a `u8` costs only O(1) and produces a value that can be cast and compared to an ASCII-range `char`. So if you're (say) line-breaking on `'\n'`, octet-based treatment still works. UTF8 was well-designed this way.
* Most "character oriented" operations on text only work under very restricted language assumptions sets such as "ASCII-range codepoints only". Outside ASCII-range, you tend to have to use a complex (non-constant-time) algorithm for determining linguistic-unit (glyph, word, paragraph) boundaries anyways. We recommend using an "honest" linguistically-aware, Unicode-approved algorithm.
* The `char` type is UCS4. If you honestly need to do a codepoint-at-a-time algorithm, it's trivial to write a `type wstr = [char]`, and unpack a `str` into it in a single pass, then work with the `wstr`. In other words: the fact that the language is not "decoding to UCS4 by default" shouldn't stop you from decoding (or re-encoding any other way) if you need to work with that encoding.

## Why are `str`s, slices, arrays etc. built-in types rather than (say) special kinds of trait/impl?

In each case there is one or more operator, literal constructor, overloaded use or integration with a built-in control structure that makes us think it would be awkward to phrase the type in terms of more-general type constructors. Same as, say, with numbers! But this is partly an aesthetic call, and we'd be willing to look at a worked-out proposal for eliminating or rephrasing these special cases.

## Can Rust code call C code?

Yes. Calling C code from Rust is simple and exactly as efficient as calling C code from C.

## Can C code call Rust code?

Yes. The Rust code has to be exposed via an `extern` declaration, which makes it C-ABI compatible. Such a function can be passed to C code as a function pointer or, if given the `#[no_mangle]` attribute to disable symbol mangling, can be called directly from C code.

## Why aren't function signatures inferred? Why only local variables?

* Mechanically, it simplifies the inference algorithm; inference only requires looking at one function at a time.
* The same simplification goes double for human readers. A reader does not need an IDE running an inference algorithm across an entire crate to be able to guess at a function's argument types; it's always explicit and nearby.

## Why does a type parameter need explicit trait bounds to invoke methods on it, when C++ templates do not?

* Requiring explicit bounds means that the compiler can type-check the code at the point where the type-parametric item is *defined*, rather than delaying to when its type parameters are instantiated.  You know that *any* set of type parameters fulfilling the bounds listed in the API will compile. It's an enforced minimal level of documentation, and results in very clean error messages.

* Scoping of methods is also a problem.  C++ needs [Koenig (argument dependent) lookup](http://en.wikipedia.org/wiki/Argument-dependent_name_lookup), which comes with its own host of problems. Explicit bounds avoid this issue: traits are explicitly imported and then used as bounds on type parameters, so there is a clear mapping from the method to its implementation (via the trait and the instantiated type).

  * Related to the above point: since a parameter explicitly names its trait bounds, a single type is able to implement traits whose sets of method names overlap, cleanly and unambiguously.

* There is further discussion on [this thread on the Rust mailing list](https://mail.mozilla.org/pipermail/rust-dev/2013-September/005603.html).

## Will Rust implement automatic semicolon insertion, like in Go?

For simplicity, we do not plan to do so. Implementing automatic semicolon insertion for Rust would be tricky because the absence of a trailing semicolon means "return a value".

## How do I get my program to display the output of logging macros?

**Short Answer**: Set the `RUST_LOG` environment variable to the name of your source file, sans extension.

```sh
rustc hello.rs
export RUST_LOG=hello
./hello
```

**Long Answer**: `RUST_LOG` takes a 'logging spec' that consists of a
comma-separated list of paths, where a path consists of the crate name and
sequence of module names, each separated by double-colons. For standalone `.rs`
files, the crate is implicitly named after the source file, so in the above
example we were setting `RUST_LOG` to the name of the hello crate. Multiple paths
can be combined to control the exact logging you want to see. For example, when
debugging linking in the compiler, you might set the following:

```sh
RUST_LOG=rustc::metadata::creader,rustc::util::filesearch,rustc::back::rpath
```

For a full description, see [the logging crate][1].

## How fast is Rust?

As always, this question is difficult to answer. There's still a lot of work to
do on speed, and depending on what you're benchmarking, Rust has variable
performance.

That said, it is an explicit goal of Rust to be as fast as C++ for most things.
Language decisions are made with performance in mind, and we want Rust to be as
fast as possible. Given that Rust is built on top of LLVM, any performance
improvements in it also help Rust become faster.

[1]:log/index.html
