- Feature Name: nonportable
- Start Date: 2016-11-15
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

There has long been a desire to expand the number of platform- and
architecture-specific APIs in the standard library, and to offer subsets of the
standard library for working in constrained environments. At the same time, we
want to retain the property that Rust code is portable by default.

This RFC proposes a new *portability lint*, which threads the needle between
these two desires. The lint piggybacks on the existing `cfg` system, so that
using APIs involving `cfg` will generate a warning unless there is explicit
acknowledgment of the portability implications.

The lint is intended to make the existing `std::os` module obsolete, to allow
expansion (and subsetting) of the standard library, and to provide deeper
checking for portability across the ecosystem.

# Motivation
[motivation]: #motivation

## Background: portability and the standard library

One of the goals of the standard library is to provide an interface to hardware
and system services. In doing so, there were several competing principles that
we wanted to embrace:

- Rust should provide ergonomic and productive APIs for system services.
- Rust should encourage portability by default.
- Rust should provide zero-cost access to low-level system services.
- Rust should be usable in a wide range of contexts, including
  resource-constrained and kernel environments.

The way we balanced these principles was roughly as follows:

- We identified a set of "mainstream" platforms, consisting of 32- and 64-bit
  machines running Windows, Linux, or macOS. "Portability by default" thus more
  specifically means portability *to mainstream platforms*.

- We present an ergonomic, primary API surface which is portable across these
  mainstream platforms (see `std::{fs, net, env, process, sync}` etc.).

- We *also* provide separate access to low-level or OS-specific services via the
  `std::os` module. APIs in this module are largely traits that extend the
  cross-platform APIs, and in particular can expose their OS-level
  representation. The fact that these APIs require explicitly importing from
  `std::os` provided a small "speed bump" for venturing out of guaranteed
  mainstream platform portability.

- Finally, for working in low-level and embedded contexts, we stabilized
  `libcore`, a subset of `libstd` that excludes all OS services and allocation,
  but *still* makes some hardware assumptions (e.g. about atomics and floating
  point support).

## Problems with the status quo

The above strategy has served us fairly well in the first year since Rust 1.0,
but it's increasingly holding us back from enhancements we'd like to make, and
even for the needs it covers, it's suboptimal in a few ways.

**Problems with `std::os`**:

* The `std::os` module has submodules that correspond to a hierarchy of OS
  types. For example, there is a `unix` submodule that applies to several
  operating systems, but there's also a `linux` submodule with Linux-specific
  extensions. There are a couple of problems with such an organization. Most
  importantly, it's not at all clear how to use the module hierarchy to organize
  features like [fixed-size atomic types][more-atomics], where the types
  available vary in a fine-grained way based on the CPU family; [SIMD] is even
  worse. But even just for operating systems, organizing into a hierarchy
  becomes difficult as we gain more and more APIs, some of which are only
  available on particular *versions* of a given operating system.

* The "speed bump" for using `std::os` is minimal and easy to miss; it's just an
  import that looks the same as any other. Moreover, it doesn't provide any help
  with the ecosystem beyond `std`. There's no simple way to tell whether a crate
  you're relying on is portable to the same degree as `std` is, and the `os`
  submodule pattern has not really caught on in the wider ecosystem.

* Platform-specific APIs don't live in their "natural location". The majority of
  `std::os` works through extension traits to enhance the functionality of
  standard primitives. For example `std::os::unix::io::AsRawFd` is a trait with
  the `as_raw_fd` method (to extract a file descriptor). If you were to ignore
  Windows, however, one might expect this API instead to live as a method
  directly on types like `File`, `TcpStream`, etc. Forcing code to live in
  `std::os` thus comes at a mild cost for both ergonomics and discoverability.
  This problem is even worse for features like adding more atomic types or SIMD.

**Problems with `libcore`/the facade**:

* Embedded libraries typically wish to never use functions in the standard
  library that abort on allocation failure (e.g. `Vec::push`). We'd like to
  provide some way for these libraries to use and interoperate with the standard
  collection types, but only have access to an alternative API surface (e.g. a
  `try_push` method provided via an extension trait). It's not clear how to do
  that with the current [facade] setup.

* Kernels and embedded environments often want to
  [disable floating point][no floats], but the floating point types are
  currently treated as primitive and shipped in `libcore`.

* There are platforms like emscripten where much of the standard library exists
  for consumption, but APIs like `std::thread` are unimplementable.  Today these
  functions simply panic on use, but a compiler error would be better.

* We'd like to open the door to a growing number of subsets of `std` and `core`,
  dropping hardware features like atomics, or perhaps even supporting 16-bit
  architectures. But again, it's not clear how to fit this into the [facade]
  model without introducing a sprawling, unwieldy collection of crates.

[more-atomics]: https://github.com/rust-lang/rfcs/pull/1543
[unix sockets]: https://github.com/rust-lang/rfcs/pull/1479
[SIMD]: https://github.com/rust-lang/rfcs/pull/1199
[no floats]: https://github.com/rust-lang/rfcs/pull/1596
[facade]: https://github.com/rust-lang/rfcs/pull/40

## What are our portability goals?

Taking a step back from the specific problems with the status quo, **it's worth
thinking about what it means for Rust to be "portable", and what is realistic to
achieve**. We should be asking this question not just for the standard library,
but for the Rust library ecosystem in general.

The premise of this RFC is that there are roughly three desired portability
levels for a library. In order of increasing portability:

- **Platform-specific**. These are libraries whose fundamental purpose
  depends on a given platform, for which portability doesn't make
  sense. Examples include the `libc` crate, the winapi crates, and crates
  designed for particular embedded devices.

- **Mainstream portability**. Most libraries take portability as a secondary
  concern, and in particular don't want to take a productivity hit just for the
  sake of maximizing portability. On the other hand, these libraries tend not to
  use obscure platform features, and it's usually not too much of a hardship to
  work across common platforms.

- **Maximal portability**. In some cases, a library author is motivated to push
  for a greater degree of portability, for example allowing their code to work
  in the `no_std` ecosystem. Depending on the library, this may entail a
  significant amount of work.

There's a fundamental tradeoff here. On the one hand, we want Rust libraries to
be as portable as possible. On the other hand, achieving *maximal* portability
can be a big burden for library authors.  Our approach so far has been to
identify "mainstream platform assumptions", as mentioned above, and *guide* code
to work on all mainstream platforms by default; by convention, such portability
is the default expectation of libraries on crates.io. This RFC formalizes that
approach in a deeper way.

An important point: while we can expect library authors who are striving for
portability to test their code on a variety of target platforms, we can't make
that assumption for the average library. In other words, **if we want to guide all
Rust code toward at least mainstream portability, we will need to do so in a way
that doesn't require actually compiling and testing for all mainstream
scenarios**.

# Detailed design
[design]: #detailed-design

## The basic idea

The core problem we want to solve is:

- We want to make non-mainstream APIs available in their natural location,
  e.g. as inherent methods directly on standard library types.

- We want to have some kind of "speed bump" before using such APIs, so that
  users realize that they may be giving up mainstream portability.

- We want to do this *without* requiring testing on platforms that lack the API.

Let's take a concrete example: the `as_raw_fd` method. We'd like to provide this
API as an inherent method on things like files. But it's not a "mainstream" API;
it only works on Unix. If you tried to use it and compiled your code on Windows,
you would discover the problem right away, since the API would not be available
due to `cfg`. But if you were only testing on Linux, you might never notice,
since the API is available there.

**The basic idea of this RFC is to provide an additional layer of checking on
top of the existing `cfg` system, to avoid usage of an API *accidentally working*
because you happen to be compiling for a given target platform**. This checking
is performed through a new **portability lint**, which warns when invoking APIs
marked with `cfg` unless you've explicitly acknowledged the portability
implications. We'll see how you do that in a moment.

Going back to our example, we'd like to define methods on `File` like:

```rust
impl File {
    #[cfg(unix)]
    fn as_raw_fd(&self) -> RawFd { ... }

    #[cfg(windows)]
    fn as_raw_handle(&self) -> RawHandle { ... }
}
```

If you attempted to call `as_raw_fd`, when compiling on Unix you'd get a warning
from the portability lint that you're calling an API not available on all
mainstream platforms. There are basically three ways to react (all of which will
make the warning go away):

- Decide not to use the API, after discovering that it would reduce portability.

- Decide to use the API, putting the function using it within a `cfg(unix)` as
  well (which will flag that function as Unix-specific).

- Decide to use the API *in a cross-platform way*, e.g. by providing a Windows
  version of the same functionality. In that case you `allow` the lint,
  explicitly acknowledging that your code may involve platform-specific APIs but
  claiming that all platforms of the current `cfg` are handled. (See the
  appendix at the end for a possible extension that does more checking).

In code, we'd have:

```rust
////////////////////////////////////////////////////////////////////////////////
// The code we might have written initially:
////////////////////////////////////////////////////////////////////////////////

fn unlabeled() {
    // Would generate a warning: calling a `unix`-only API while only
    // assuming a mainstream platform
    let fd = File::open("foo.txt").unwrap().as_raw_fd();
}

////////////////////////////////////////////////////////////////////////////////
// Code that opts into platform-specificness:
////////////////////////////////////////////////////////////////////////////////

#[cfg(unix)]
fn foo() {
    // No warning: we're within code that assumes `unix`
    let fd = File::open("foo.txt").unwrap().as_raw_fd();
}

#[cfg(windows)]
fn foo() {
    // No warning: we're within code that assumes `windows`
    let handle = File::open("foo.txt").unwrap().as_raw_handle();
}

#[cfg(linux)]
fn linux_only() {
    // No warning: we're within code that assumes `linux`, which implies `unix`
    let fd = File::open("foo.txt").unwrap().as_raw_fd();
}

////////////////////////////////////////////////////////////////////////////////
// Code that provides a cross-platform abstraction
////////////////////////////////////////////////////////////////////////////////

// No `cfg` label here; it's a cross-platform function, which we claim
// via the `allow`
#[allow(nonportable)]
fn cross_platform() {
    // invoke an item with a more restrictive `cfg`
    foo()
}
```

As with many lints, the portability lint is *best effort*: it is not required to
provide airtight guarantees about portability. However, the RFC sketches a
plausible implementation route that should cover the vast majority of cases.

With that overview in mind, let's dig into the details.

## The lint definition

The lint is structured somewhat akin to a type and effect system: roughly
speaking, items that are labeled with a given `cfg` assumption can only be used
within code making that same `cfg` assumption.

More precisely, each item has a *portability*, consisting of all the
lexically-nested uses of `cfg`. If there are multiple uses of `cfg`, the
portability is taken to be their *conjunction*:

```rust
#[cfg(unix)]
mod foo {
    #[cfg(target_pointer_width = "32")]
    fn bar() {
        // the portability of `bar` is `all(unix, target_pointer_width = "32")`
    }
}
```

The portability only considers built-in `cfg` attributes (like `target_os`),
*not* Cargo features (which are treated as automatically true for the lint
purposes).

The lint is then straightforward to define at a high level: it walks over item
definitions and checks that the item's portability is *narrower* than the
portability of items it references or invokes. For example, `bar` in the above
could invoke an item with portability `unix` and/or `target_pointer_width =
"32"`, but not one with portability `linux`.

To fully define the lint, though, we need to give more details about what
"narrower" means, and how referenced item portability is determined.

### Comparing portabilities

**What does it mean for a portability to be narrower?** In general, portability
is a logical expression, using the operators `all`, `any`, `not` on top of
primitive expressions like `unix`. Portability `P` is narrower than portability
`Q` if `P` *implies* `Q` as a logic formula.

In general, comparing two portabilities is equivalent to solving SAT, an
NP-complete problem -- a frightening prospect for a lint! However, note that
worst-case execution is exponential in *the number of variables* (i.e.,
primitive `cfg` constraints), not the number/complexity of clauses, and most
comparisons should involve a very small number of variables. We can likely get
away with a naive SAT implementation, perhaps with a handful of optimiziations
specific to our use-case. In the limit, there are also many well-known
techniques for solving SAT efficiently even on very large examples that arise in
real-world usage.

#### Axioms

Another aspect of portability comparison is the relationship between things like
`unix` and `linux`. In logical terms, we want to assume that `linux` implies
`unix`, for example.

The primitive portabilities we'll be comparing are all *built in* (since we are
not including Cargo features). The solver can thus build in a number of
assumptions about these portabilities. The end result is that code like the
following should pass the lint:

```rust
#[cfg(unix)]
fn unix_only() { .. }

#[cfg(linux)]
fn linux_only() {
    // permitted since `linux` implies `unix`
    unix_only()
}
```

The precise details of how these implications are specified---and what
implications are desired---are left as implementation details.

### Determining the portability of referenced items

**How is the portability of a referenced item determined?** The lint will
resolve an item to its definition, and use the portability of that definition,
which will be recorded in metadata. For the case of trait items, however, this
will involve attempting to resolve the invocation to a particular impl, to look
up the portability of that impl. We can set up trait selection to yield
portability information with the selected impl, which will allow us to catch
cases like the following:

```rust
trait Foo {
    fn foo();
}

struct MyType;

#[cfg(unix)]
impl Foo for MyType {
    fn foo() { .. }
}

fn use_foo<T: Foo>() {
    T::foo()
}

fn invoke() {
    // invokes a `cfg(unix)` item via a generic function, but we can catch it
    // when checking that `MyType: Foo`, since selection will say that we need
    // our context to imply `unix`
    use_foo::<MyType>();
}
```

## The story for `std`

With these basic mechanisms in hand, let's sketch out how we might apply them to
the standard library to achieve our initial goals. This part of the RFC should
not be considered normative; it's left to the implementation to make the final
determination about how to set up the standard library.

### The mainstream platform

The "mainstream platform" will be expressed via a new primitive `cfg` pattern
called `std`. This is the **default portability of all crates**, unless
opted-out (see below on "subsetting `std`"). Likewise, most items in `std` will
initially be exported at `std` portability level. These two facts together mean
that existing uses of `std` will continue to work without issuing any warnings.

The `std` portability will include several implications, e.g.:

- `std` implies `any(windows, macos, linux)`
- `std` implies `any(target_pointer_width = "32", target_pointer_width = "64")`

and so on. That means, in particular, that a `match_cfg` expression that covers
*all* of Windows, macOS and Linux will be considered to have "mainstream
portability" automatically.

### Expanding `std`

With the above setup, handling extensions to `std` with APIs like `as_raw_fd` is
straightforward. In particular, we can write:

```rust
impl File {
    #[cfg(unix)]
    fn as_raw_fd(&self) -> RawFd { ... }

    #[cfg(windows)]
    fn as_raw_handle(&self) -> RawHandle { ... }
}
```

and the portability of `as_raw_fd` will be `all(std, unix)`. Thus, any code
using `as_raw_fd` will need to be in a `unix` context.

We can thus deprecate the `std::os` module in favor of these in-place
APIs. Doing so leverages the fact that we're using a portability *lint*: these
new inherent methods will shadow the existing ones in `std::os`, and may
generate new warnings, but this is considered an acceptable change. After all,
lints on dependencies are automatically capped, and the lint will not prevent
code from compiling--and can be silenced.

Expanding to include new atomics, SIMD, and other desired extensions should
amount to a straightforward use of `cfg`.

### Subsetting `std`

What about subsets of `std` (or `core`)? First of all, if you apply `cfg` to
your *crate* definition, you opt out of the default `std` portability level in
favor of the `cfg` you write. Doing so will deny access to many APIs in `std`.

Over time, APIs within `std` and `core` will be labeled with new, more narrow
portabilities. Let's take, for example, threading--which we wish to not provide
on platforms like Emscripten. The `std` threading APIs might be revised to use
`cfg(threads)`, rather than `cfg(std)`; at the same time, `std` would be set up
to imply `threads`, so that no new warnings would be generated. To check for
compatibility with Emscripten, you can opt out of the `std` scenario, and avoid
opting into `threads` (or use `match_cfg` if you want to do so only optionally,
for example to use optional parallelism).

Similarly, `libcore` can be annotated with new `cfg`s, like `cfg(float)` for
floating point support.

Thus, library authors shooting for maximal portability should opt out of
`cfg(std)`, and use `cfg` as little as possible. And over time, we can allow
increasingly fine-grained subsets of `std` by introducing new `cfg` flags.

## Backwards compatibility and lint evolution

The fact that the portability lint is a *lint* gives us a lot of
flexibility. Take, for example, the assumption that `std` implies `any(windows,
macos, linux)`. Conceivably, we may want to add more mainstream platforms in the
future. Doing so may generate new warnings--particularly for people who had used
`match_cfg` previously to exhaustively match against these cases. But (1) it's
merely a new *warning*, which is fine to introduce, and can be silenced; (2)
this is actually a highly desirable outcome if we ever did add a new mainstream
platform, since existing code would get a heads-up that it may not longer be
compatible with all mainstream platforms.

# Drawbacks
[drawbacks]: #drawbacks

There are several potential drawbacks to the approach of this RFC:

* It adds a significant level of pedanticness about portability to Rust.
* It does not provide airtight guarantees.
* It may create compiler performance issues, due to the use of SAT solving.

The fact that it's a lint offers some help with the first two points; the use of
`std` as a default portability level should also help quite a bit with
pedanticness.

The worry about SAT solving is harder to mitigate; there's not much concrete
evidence in either direction. But it is yet another place where the fact that
it's a lint could help: we may be able to simply skip checking pathological
cases, if they indeed arise in practice. In any case, it's hard to know how
concerned to be until we try it.

While the fact that it's a lint gives us more leeway to experiment, it's also a
lint that could produce widespread warnings throughout the ecosystem, so we need
to exercise care.

# Alternatives
[alternatives]: #alternatives

The main alternatives are:

- **Give up on encouraging "portability by default"**, and instead just land
  APIs in their natural location using today's `cfg` system. This is certainly
  the less costly way to go. It's also *forward-compatible* with implementing
  the proposed lint, so we should discuss the possibility of landing APIs under
  `cfg` even before the lint is implemented.

- **Use a less precise checking strategy.** In particular, rather than trying to
  compare portabilities in a detailed, item-level way, we might just require
  some crate-level "opt in". That could either take the form of acknowledging
  "this code makes assumptions beyond the mainstream platform", or might list
  the specific `cfg` assumptions the code is allowed to make. Of course, the
  downside is that you get much less help making sure that your APIs are
  properly labeled in place.

# How we teach this
[how-we-teach-this]: #how-we-teach-this

For people simply using libraries, this feature "teaches itself" by generating
warnings. Those warnings should make clear what to do to fix the problem, and
ideally provide extended error information that describes the system in more
detail.

For library authors, the documentation for `cfg` and `match_cfg` would explain
the implications for the lint, and walk through several examples illustrating
the scenarios that arise in practice.

# Unresolved questions
[unresolved]: #unresolved-questions

### External libraries

It's not clear what the story should be for a library like `libc`, which
currently involves intricate uses of `cfg`. We should have some idea for how to
approach such cases before landing the RFC.

### The standard library

To what extent does this proposal obviate the need for the `std` facade? Might
it be possible to deprecate `libcore` in favor of the "subsetting `std`" approach?

# Appendix: possible extensions

## `match_cfg`

The original version of this RFC was more expansive, and proposed a `match_cfg`
macro that provided some additional checking.

The `match_cfg` macro takes a sequence of `cfg` patterns, followed by `=>` and
an expression. Its syntax and semantics resembles that of `match`. However,
there are some special considerations when checking portability:

* When descending into an arm of a `match_cfg`, the arm is checked against
  portability that includes the pattern for the arm.

* The portability for the `match_cfg` itself is understood as `any(p1, ...,
  p_n)` where the `match_cfg` patterns are `p1` through `p_n`.

Thus, for example, the following code will pass the lint:

```rust
#[cfg(windows)]
fn windows_only() { .. }

#[cfg(unix)]
fn unix_only() { .. }

#[cfg(any(windows, unix))]
fn portable() {
    // the expression here has portability `any(windows, unix)`
    match_cfg! {
        windows => {
            // allowed because we are within a scope with
            // portability `all(any(windows, unix), windows)`
            windows_only()
        }
        unix => {
            // allowed because we are within a scope with
            // portability `all(any(windows, unix), unix)`
            unix_only()
        }
    }
}
```

If you have a `match_case` that covers *all* cases (like `windows` and
`not(windows)`), then it imposes *no* portability constraints on its context.

On more reflection, though, this extension doesn't seem so worthwhile: while it
provides some additional checking, the fact remains that only the
currently-enabled `cfg` is fully checked, so the additional guarantee you get is
somewhat mixed. It's also a rare (maybe non-existent) error to explicitly write
code that's broken down by platforms, but forget one of the platforms you wish
to cover.

We can, however, add `match_cfg` as a backwards-compatible extension at any time.
