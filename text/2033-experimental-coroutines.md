- Feature Name: `coroutines`
- Start Date: 2017-06-15
- RFC PR: https://github.com/rust-lang/rfcs/pull/2033
- Rust Issue: https://github.com/rust-lang/rust/issues/43122

# Summary
[summary]: #summary

This is an **experimental RFC** for adding a new feature to the language,
coroutines (also commonly referred to as generators). This RFC is intended to be
relatively lightweight and bikeshed free as it will be followed by a separate
RFC in the future for stabilization of this language feature. The intention here
is to make sure everyone's on board with the *general idea* of
coroutines/generators being added to the Rust compiler and available for use on
the nightly channel.

# Motivation
[motivation]: #motivation

One of Rust's 2017 roadmap goals is ["Rust should be well-equipped for writing
robust, high-scale servers"][goal]. A [recent survey][survey] has shown that
the biggest blocker to robust, high-scale servers is ergonomic usage of async
I/O (futures/Tokio/etc). Namely, the lack of async/await syntax. Syntax like
async/await is essentially the defacto standard nowadays when working with async
I/O, especially in languages like C#, JS, and Python. Adding such a feature to
rust would be a huge boon to productivity on the server and make significant
progress on the 2017 roadmap goal as one of the largest pain points, creating
and returning futures, should be as natural as writing blocking code.

[goal]: https://github.com/rust-lang/rfcs/blob/master/text/1774-roadmap-2017.md#rust-should-be-well-equipped-for-writing-robust-high-scale-servers
[survey]: https://users.rust-lang.org/t/what-does-rust-need-today-for-server-workloads/11114

With our eyes set on async/await the next question is how would we actually
implement this? There's sort of two main sub-questions that we have to answer to
make progress here though, which are:

* What's the actual syntax for async/await? Should we be using new keywords in
  the language or pursuing syntax extensions instead?

* How do futures created with async/await support suspension? Essentially while
  you're waiting for some sub-future to complete, how does the future created by
  the async/await syntax return back up the stack and support coming back and
  continuing to execute?

The focus of this experimental RFC is predominately on the second, but before we
dive into more motivation there it may be worth to review the expected syntax
for async/await.

### Async/await syntax

Currently it's intended that **no new keywords are added to
Rust yet** to support async/await. This is done for a number of reasons, but
one of the most important is flexibility. It allows us to stabilize features
more quickly and experiment more quickly as well.

Without keywords the intention is that async/await will be implemented with
macros, both procedural and `macro_rules!` style. We should be able to leverage
[procedural macros][pmac] to give a near-native experience. Note that procedural
macros are only available on the nightly channel today, so this means that
"stable async/await" will have to wait for procedural macros (or at least a
small slice) to stabilize.

[pmac]: https://github.com/rust-lang/rfcs/blob/master/text/1566-proc-macros.md

With that in mind, the expected syntax for async/await is:

```rust
#[async]
fn print_lines() -> io::Result<()> {
    let addr = "127.0.0.1:8080".parse().unwrap();
    let tcp = await!(TcpStream::connect(&addr))?;
    let io = BufReader::new(tcp);

    #[async]
    for line in io.lines() {
        println!("{}", line);
    }

    Ok(())
}
```

The notable pieces here are:

* `#[async]` is how you tag a function as "this returns a future". This is
  implemented with a `proc_macro_attribute` directive and allows us to change
  the function to actually returning a future instead of a `Result`.

* `await!` is usable inside of an `#[async]` function to block on a future. The
  `TcpStream::connect` function here can be thought of as returning a future of
  a connected TCP stream, and `await!` will block execution of the `print_lines`
  function until it becomes available. Note the trailing `?` propagates errors
  as the `?` does today.

* Finally we can implement more goodies like `#[async]` `for` loops which
  operate over the `Stream` trait in the `futures` crate. You could also imagine
  pieces like `async!` blocks which are akin to `catch` for `?`.

The intention with this syntax is to be as familiar as possible to existing Rust
programmers and disturb control flow as little as possible. To that end all
that's needed is to tag functions that may block (e.g. return a future) with
`#[async]` and then use `await!` internally whenever blocking is needed.

Another critical detail here is that the API exposed by async/await is quite
minimal! You'll note that this RFC is an experimental RFC for coroutines and we
haven't mentioned coroutines at all with the syntax! This is an intentional
design decision to keep the implementation of `#[async]` and `await!` as
flexible as possible.

### Suspending in async/await

With a rough syntax in mind the next question was how do we actually suspend
these futures? The function above will desugar to:

```rust
fn print_lines() -> impl Future<Item = (), Error = io::Error> {
    // ...
}
```

and this means that we need to create a `Future` *somehow*. If written with
combinators today we might desugar this to:

```rust
fn print_lines() -> impl Future<Item = (), Error = io::Error> {
    lazy(|| {
        let addr = "127.0.0.1:8080".parse().unwrap();
        TcpStream::connect(&addr).and_then(|tcp| {
            let io = BufReader::new(tcp);

            io.lines().for_each(|line| {
                println!("{}", line);
                Ok(())
            })
        })
    })
}
```

Unfortunately this is actually quite a difficult transformation to do
(translating to combinators) and it's actually not quite as optimal as we might
like! We can see here though some important points about the semantics that we
expect:

* When called, `print_lines` doesn't actually do anything. It immediately just
  returns a future, in this case created via [`lazy`].
* When `Future::poll` is first called, it'll create the `addr` and then call
  `TcpStream::connect`. Further calls to `Future::poll` will then delegate to
  the future returned by `TcpStream::connect`.
* After we've connected (the `connect` future resolves) we continue our
  execution with further combinators, blocking on each line being read from the
  socket.

[`lazy`]: https://docs.rs/futures/0.1.14/futures/future/fn.lazy.html

A major benefit of the desugaring above is that there are no hidden allocations.
Combinators like `lazy`, `and_then`, and `for_each` don't add that sort of
overhead. A problem, however, is that there's a bunch of nested state machines
here (each combinator is its own state machine). This means that our in-memory
representation can be a bit larger than it needs to be and take some time to
traverse. Finally, this is also very difficult for an `#[async]` implementation
to generate! It's unclear how, with unusual control flow, you'd implement all
the paradigms.

Before we go on to our final solution below it's worth pointing out that a
popular solution to this problem of generating a future is to side step
this completely with the concept of green threads. With a green thread you can
suspend a thread by simply context switching away and there's no need to
generate state and such as an allocated stack implicitly holds all this state.
While this does indeed solve our problem of "how do we translate `#[async]`
functions" it unfortunately violates Rust's general theme of "zero cost
abstractions" because the allocated stack on the side can be quite costly.

At this point we've got some decent syntax and rough (albeit hard) way we want
to translate our `#[async]` functions into futures. We've also ruled out
traditional solutions like green threads due to their costs, so we just need a
way to easily create the optimal state machine for a future that combinators
would otherwise emulate.

### State machines as "stackless coroutines"

Up to this point we haven't actually mentioned coroutines all that much which
after all is the purpose of this RFC! The intention of the above motivation,
however, is to provide a strong case for *why coroutines?* At this point,
though, this RFC will mostly do a lot of hand-waving. It should suffice to say,
though, that the feature of "stackless coroutines" in the compiler is precisely
targeted at generating the state machine we wanted to write by hand above,
solving our problem!

Coroutines are, however, a little lower level than futures themselves. The
stackless coroutine feature can be used not only for futures but also other
language primitives like iterators. As a result let's take a look at what a
hypothetical translation of our original `#[async]` function might look like.
Keep in mind that this is not a specification of syntax, it's just a strawman
possibility for how we'd write the above.

```rust
fn print_lines() -> impl Future<Item = (), Error = io::Error> {
    CoroutineToFuture(|| {
        let addr = "127.0.0.1:8080".parse().unwrap();
        let tcp = {
            let mut future = TcpStream::connect(&addr);
            loop {
                match future.poll() {
                    Ok(Async::Ready(e)) => break Ok(e),
                    Ok(Async::NotReady) => yield,
                    Err(e) => break Err(e),
                }
            }
        }?;

        let io = BufReader::new(tcp);

        let mut stream = io.lines();
        loop {
            let line = {
                match stream.poll()? {
                    Async::Ready(Some(e)) => e,
                    Async::Ready(None) => break,
                    Async::NotReady => {
                        yield;
                        continue
                    }
                }
            };
            println!("{}", line);
        }

        Ok(())
    })
}
```

The most prominent addition here is the usage of `yield` keywords. These are
inserted here to inform the compiler that the coroutine should be suspended for
later resumption. Here this happens precisely where futures are themselves
`NotReady`. Note, though, that we're not working directly with futures (we're
working with coroutines!). That leads us to this funky `CoroutineToFuture` which
might look like so:

```rust
struct CoroutineToFuture<T>(T);

impl<T: Coroutine> Future for CoroutineToFuture {
    type Item = T::Item;
    type Error = T::Error;

    fn poll(&mut self) -> Poll<T::Item, T::Error> {
        match Coroutine::resume(&mut self.0) {
            CoroutineStatus::Return(Ok(result)) => Ok(Async::Ready(result)),
            CoroutineStatus::Return(Err(e)) => Err(e),
            CoroutineStatus::Yield => Ok(Async::NotReady),
        }
    }
}
```

Note that some details here are elided, but the basic idea is that we can pretty
easily translate all coroutines into futures through a small adapter struct.

As you may be able to tell by this point, we've now solved our problem of code
generation! This last transformation of `#[async]` to coroutines is much more
straightforward than the translations above, and has in fact [already been
implemented][futures-await].

[futures-await]: https://github.com/alexcrichton/futures-await

To reiterate where we are at this point, here's some of the highlights:

* One of Rust's roadmap goals for 2017 is pushing Rust's usage on the server.
* A major part of this goal is going to be implementing async/await syntax for
  Rust with futures.
* The async/await syntax has a relatively straightforward syntactic definition
  (borrowed from other languages) with procedural macros.
* The procedural macro itself can produce optimal futures through the usage of
  *stackless coroutines*

Put another way: if the compiler implements stackless coroutines as a feature,
we have now achieved async/await syntax!

### Features of stackless coroutines

At this point we'll start to tone down the emphasis of servers and async I/O
when talking about stackless coroutines. It's important to keep them in mind
though as motivation for coroutines as they guide the design constraints of
coroutines in the compiler.

At a high-level, though, stackless coroutines in the compiler would be
implemented as:

* No implicit memory allocation
* Coroutines are translated to state machines internally by the compiler
* The standard library has the traits/types necessary to support the coroutines
  language feature.

Beyond this, though, there aren't many other constraints at this time. Note that
a critical feature of async/await is that **the syntax of stackless coroutines
isn't all that important**. In other words, the implementation detail of
coroutines isn't actually exposed through the `#[async]` and `await!`
definitions above. They purely operate with `Future` and simply work internally
with coroutines. This means that if we can all broadly agree on async/await
there's no need to bikeshed and delay coroutines. Any implementation of
coroutines should be easily adaptable to async/await syntax.

# Detailed design
[design]: #detailed-design

Alright hopefully now we're all pumped to get coroutines into the compiler so we
can start playing around with async/await on the nightly channel. This RFC,
however, is explicitly an **experimental RFC** and is not intended to be a
reference for stability. It is not intended that stackless coroutines will ever
become a stable feature of Rust without a further RFC. As coroutines are such a
large feature, however, testing the feature and gathering usage data needs to
happen on the nightly channel, meaning we need to land something in the
compiler!

This RFC is different from the previous [RFC 1823] and [RFC 1832] in that this
detailed design section will be mostly devoid of implementation details for
generators. This is intentionally done so to avoid bikeshedding about various
bits of syntax related to coroutines. While critical to stabilization of
coroutines these features are, as explained earlier, irrelevant to the "apparent
stability" of async/await and can be determined at a later date once we have
more experience with coroutines.

In other words, the intention of this RFC is to emphasize that point that **we
will focus on adding async/await through procedural macros and coroutines**. The
driving factor for stabilization is the real-world and high-impact use case of
async/await, and zero-cost futures will be an overall theme of the continued
work here.

It's worth briefly mentioning, however, some high-level design goals of the
concept of stackless coroutines:

* Coroutines should be compatible with libcore. That is, they should not require
  any runtime support along the lines of allocations, intrinsics, etc.
* As a result, coroutines will roughly compile down to a state machine that's
  advanced forward as its resumed. Whenever a coroutine yields it'll leave
  itself in a state that can be later resumed from the yield statement.
* Coroutines should work similarly to closures in that they allow for capturing
  variables and don't impose dynamic dispatch costs. Each coroutine will be
  compiled separately (monomorphized) in the way that closures are today.
* Coroutines should also support some method of communicating arguments in and
  out of itself. For example when yielding a coroutine should be able to yield a
  value. Additionally when resuming a coroutine may wish to require a value is
  passed in on resumption.

[RFC 1823]: https://github.com/rust-lang/rfcs/pull/1823
[RFC 1832]: https://github.com/rust-lang/rfcs/pull/1832

As a reference point @Zoxc has implemented generators in a [fork of
rustc][fork], and has been a critical stepping stone in experimenting with the
`#[async]` macro in the motivation section. This implementation may end up being
the original implementation of coroutines in the compiler, but if so it may
still change over time.

[fork]: https://github.com/Zoxc/rust/tree/gen

One important note is that we haven't had many experimental RFCs yet, so this
process is still relatively new to us! We hope that this RFC is lighter weight
and can go through the RFC process much more quickly as the ramifications of it
landing are much more minimal than a new stable language feature being added.

Despite this, however, there is also a desire to think early on about corner
cases that language features run into and plan for a sort of reference test
suite to exist ahead of time. Along those lines this RFC proposes a list of
tests accompanying any initial implementation of coroutines in the compiler,
covering. Finally this RFC also proposes a list of unanswered questions related
to coroutines which likely wish to be considered before stabilization

##### Open Questions - coroutines

* What is the precise syntax for coroutines?
* How are coroutines syntactically and functionally constructed?
* What do the traits related to coroutines look like?
* Is "coroutine" the best name?
* Are coroutines sufficient for implementing iterators?
* How do various traits like "the coroutine trait", the `Future` trait, and
  `Iterator` all interact? Does coherence require "wrapper struct" instances to
  exist?

##### Open Questions - async/await

* Is using a syntax extension too much considered to be creating a
  "sub-language"? Does async/await usage feel natural in Rust?
* What precisely do you write in a signature of an async function? Do you
  mention the future aspect?
* Can `Stream` implementations be created with similar syntax? Is async/await
  with coroutines too specific to futures?
*

##### Tests - Basic usage

* Coroutines which don't yield at all and immediately return results
* Coroutines that yield once and then return a result
* Creating a coroutine which closes over a value, and then returning it
* Returning a captured value after one yield
* Destruction of a coroutine drops closed over variables
* Create a coroutine, don't run it, and drop it
* Coroutines are `Send` and `Sync` like closures are wrt captured variables
* Create a coroutine on one thread, run it on another

##### Tests - Basic compile failures

* Coroutines cannot close over data that is destroyed before the coroutine is
  itself destroyed.
* Coroutines closing over non-`Send` data are not `Send`

##### Test - Interesting control flow

* Yield inside of a `for` loop a set number of times
* Yield on one branch of an `if` but not the other (take both branches here)
* Yield on one branch of an `if` inside of a `for` loop
* Yield inside of the condition expression of an `if`

##### Tests - Panic safety

* Panicking in a coroutine doesn't kill everything
* Resuming a panicked coroutine is memory safe
* Panicking drops local variables correctly

##### Tests - Debuginfo

* Inspecting variables before/after yield points works
* Breaking before/after yield points works

Suggestions for more test are always welcome!

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Coroutines are not, and will not become a stable language feature as a result of
this RFC. They are primarily designed to be used through async/await notation
and are otherwise transparent. As a result there are no specific plans at this
time for teaching coroutines in Rust. Such plans must be formulated, however,
prior to stabilization.

Nightly-only documentation will be available as part of the unstable book about
basic usage of coroutines and their abilities, but it likely won't be exhaustive
or the best learning for resource for coroutines yet.

# Drawbacks
[drawbacks]: #drawbacks

Coroutines are themselves a significant feature for the compiler. This in turns
brings with it maintenance burden if the feature doesn't pan out and can
otherwise be difficult to design around. It is thought, though, that coroutines
are highly likely to pan out successfully with futures and async/await notation
and are likely to be coalesced around as a stable compiler feature.

# Alternatives
[alternatives]: #alternatives

The alternatives to list here, as this is an experimental RFC, are more targeted
as alternatives to the motivation rather than the feature itself here. Along
those lines, you could imagine quite a few alternatives to the goal of tackling
the 2017 roadmap goal targeted in this RFC. There's quite a bit of discussion on
the [original rfc thread][rfc], but some highlight alternatives are:

* "Stackful coroutines" aka green threads. This strategy has, however, been
  thoroughly explored in historical versions of Rust. Rust long ago had green
  threads and libgreen, and consensus was later reached that it should be
  removed. There are many tradeoffs with an approach like this, but it's safe to
  say that we've definitely gained a lot of experimental and anecdotal evidence
  historically!

* User-mode-scheduling is another possibility along the line of green threads.
  Unfortunately this isn't implemented in all mainstream operating systems
  (Linux/Mac/Windows) and as a result isn't a viable alternative at this time.

* ["Resumable expressions"][cpp] is a proposal in C++ which attempts to deal
  with some of the "viral" concerns of async/await, but it's unclear how
  applicable or easy it would apply to Rust.

[rfc]: https://github.com/rust-lang/rfcs/pull/2033
[cpp]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0114r0.pdf

Overall while there are a number of alternatives, the most plausible ones have a
large amount of experimental and anecdotal evidence already (green
threads/stackful coroutines). The next-most-viable alternative (stackless
coroutines) we do not have much experience with. As a result it's believed that
it's time to explore and experiment with an alternative to M:N threading with
stackless coroutines, and continue to push on the 2017 roadmap goal.

Some more background about this motivation for exploring async/await vs
alternatives can also be found [in a comment on the RFC thread][comment].

[comment]: https://github.com/rust-lang/rfcs/pull/2033#issuecomment-309603972

# Unresolved questions
[unresolved]: #unresolved-questions

The precise semantics, timing, and procedure of an experimental RFC are still
somewhat up in the air. It may be unclear what questions need to be decided on
as part of an experimental RFC vs a "real RFC". We're hoping, though, that we
can smooth out this process as we go along!
