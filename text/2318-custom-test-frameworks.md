- Feature Name: `custom_test_frameworks`
- Start Date: 2018-01-25
- RFC PR: [rust-lang/rfcs#2318](https://github.com/rust-lang/rfcs/pull/2318)
- Rust Issue: [rust-lang/rust#50297](https://github.com/rust-lang/rust/issues/50297)

# Summary
[summary]: #summary

This is an *experimental RFC* for adding the ability to integrate custom test/bench/etc frameworks ("test frameworks") in Rust.

# Motivation
[motivation]: #motivation

Currently, Rust lets you write unit tests with a `#[test]` attribute. We also have an unstable `#[bench]` attribute which lets one write benchmarks.

In general it's not easy to use your own testing strategy. Implementing something that can work
within a `#[test]` attribute is fine (`quickcheck` does this with a macro), but changing the overall
strategy is hard. For example, `quickcheck` would work even better if it could be done as:

```rust
#[quickcheck]
fn test(input1: u8, input2: &str) {
    // ...
}
```

If you're trying to do something other than testing, you're out of luck -- only tests, benches, and examples
get the integration from `cargo` for building auxiliary binaries the correct way. [cargo-fuzz] has to
work around this by creating a special fuzzing crate that's hooked up the right way, and operating inside
of that. Ideally, one would be able to just write fuzz targets under `fuzz/`.

[Compiletest] (rustc's test framework) would be another kind of thing that would be nice to
implement this way. Currently it compiles the test cases by manually running `rustc`, but it has the
same problem as cargo-fuzz where getting these flags right is hard. This too could be implemented as
a custom test framework.

A profiling framework may want to use this mode to instrument the binary in a certain way. We
can already do this via proc macros, but having it hook through `cargo test` would be neat.

Overall, it would be good to have a generic framework for post-build steps that can support use
cases like `#[test]` (both the built-in one and quickcheck), `#[bench]` (both built in and custom
ones like [criterion]), `examples`, and things like fuzzing. While we may not necessarily rewrite
the built in test/bench/example infra in terms of the new framework, it should be possible to do so.

The main two features proposed are:

 - An API for crates that generate custom binaries, including
   introspection into the target crate.
 - A mechanism for `cargo` integration so that custom test frameworks
are at the same level of integration as `test` or `bench` as
   far as build processes are concerned.

 [cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
 [criterion]: https://github.com/japaric/criterion.rs
 [Compiletest]: http://github.com/laumann/compiletest-rs

# Detailed proposal
[detailed-proposal]: #detailed-proposal

(As an eRFC I'm merging the "guide-level/reference-level" split for now; when we have more concrete
ideas we can figure out how to frame it and then the split will make more sense)

The basic idea is that crates can define test frameworks, which specify 
how to transform collected test functions and construct a `main()` function,
and then crates using these can declare them in their Cargo.toml, which will let
crate developers invoke various test-like steps using the framework.


## Procedural macro for a new test framework

A test framework is like a procedural macro that is evaluated after all other macros in the target
crate have been evaluated. The exact mechanism is left up to the experimentation phase, however we
have some proposals at the end of this RFC.


A crate may only define a single framework.

## Cargo integration

Alternative frameworks need to integrate with cargo.
In particular, when crate `a` uses a crate `b` which provides an
framework, `a` needs to be able to specify when `b`'s framework
should be used. Furthermore, cargo needs to understand that when
`b`'s framework is used, `b`'s dependencies must also be linked.

Crates which define a test framework must have a `[testing.framework]`
key in their `Cargo.toml`. They cannot be used as regular dependencies.
This section works like this:

```rust
[testing.framework]
kind = "test" # or bench
```

`lib` specifies if the `--lib` mode exists for this framework by default,
and `folders` specifies which folders the framework applies to. Both can be overridden
by consumers.

`single-target` indicates that only a single target can be run with this
framework at once (some tools, like cargo-fuzz, run forever, and so it
does not make sense to specify multiple targets).

Crates that wish to *use* a custom test framework, do so by including a framework
under a new `[[testing.frameworks]]` section in their
`Cargo.toml`:

```toml
[[testing.frameworks]]
provider = { quickcheck = "1.0" }
```

This pulls in the framework  from the "quickcheck" crate.  By default, the following
framework is defined:

```toml
[[testing.frameworks]]
provider = { test = "1.0" }
```

(We may define a default framework for bench in the future)

Declaring a test framework will replace the existing default one. You cannot declare
more than one test or bench framework.

To invoke a particular framework, a user invokes `cargo test` or `cargo bench`. Any additional
arguments are passed to the testing binary. By convention, the first position argument should allow
filtering which targets (tests/benchmarks/etc.) are run.

## To be designed

This contains things which we should attempt to solve in the course of this experiment, for which this eRFC
does not currently provide a concrete proposal.

## Procedural macro design


We have a bunch of concrete proposals here, but haven't yet chosen one.

### main() function generation with test collector

One possible design is to have a proc macro that simply generates `main()`

It is passed the `TokenStream` for every element in the
target crate that has a set of attributes the test framework has
registered interest in. For example, to declare a test framework
called `mytest`:

```rust
extern crate proc_macro;
use proc_macro::{TestFrameworkContext, TokenStream};

// attributes() is optional
#[test_framework]
pub fn test(context: &TestFrameworkContext) -> TokenStream {
    // ...
}
```

where

```rust
struct TestFrameworkContext<'a> {
    items: &'a [AnnotatedItem],
    // ... (may be added in the future)
}

struct AnnotatedItem
    tokens: TokenStream,
    span: Span,
    attributes: TokenStream,
    path: SomeTypeThatRepresentsPathToItem
}
```

`items` here contains an `AnnotatedItem` for every item in the
target crate that has one of the attributes declared in `attributes`
along with attributes sharing the name of the framework (`test`, here --
the function must be named either `test` or `bench`).

The annotated function _must_ be named "test" for a test framework and
"bench" for a bench framework. We currently do not support
any other kind of framework, but we may in the future.

So an example transformation would be to take something like this:

```rust
#[test]
fn foo(x: u8) {
    // ...
}

mod bar {
    #[test]
    fn bar(x: String, y: u8) {
        // ...
    }
}
```

and output a `main()` that does something like:

```rust
fn main() {
    // handles showing failures, etc
    let mut runner = quickcheck::Runner();

    runner.iter("foo", |random_source| foo(random_source.next().into()));
    runner.iter("bar::bar", |random_source| bar::bar(random_source.next().into(),
                                                     random_source.next().into()));
    runner.finish();
}
```

The compiler will make marked items `pub(crate)` (i.e. by making
all their parent modules public). `#[test]` and `#[bench]` items will only exist
with `--cfg test` (or bench), which is automatically set when running tests.


### Whole-crate procedural macro

An alternative proposal was to expose an extremely general whole-crate proc macro:

```rust
#[test_framework(attributes(foo, bar))]
pub fn mytest(crate: TokenStream) -> TokenStream {
    // ...
}
```

and then we can maintain a helper crate, out of tree, that uses `syn` to provide a nicer
API, perhaps something like:

```rust
fn clean_entry_point(tree: syn::ItemMod) -> syn::ItemMod;

trait TestCollector {
    fn fold_function(&mut self, path: syn::Path, func: syn::ItemFn) -> syn::ItemFn;
}

fn collect_tests<T: TestCollector>(collector: &mut T, tree: syn::ItemMod) -> ItemMod;
```

This lets us continue to develop things outside of tree without perma-stabilizing an API;
and it also lets us provide a friendlier API via the helper crate.

It also lets crates like `cargo-fuzz` introduce things like a `#![no_main]` attribute or do
other antics.

Finally, it handles the "profiling framework" case as mentioned in the motivation. On the other hand,
these tools usually operate at a different layer of abstraction so it might not be necessary.

A major drawback of this proposal is that it is very general, and perhaps too powerful. We're currently using the
more focused API in the eRFC, and may switch to this during experimentation if a pressing need crops up.

### Alternative procedural macro with minimal compiler changes

The above proposal can be made even more general, minimizing the impact on the compiler.

This assumes that `#![foo]` ("inner attribute") macros work on modules and on crates.

The idea is that the compiler defines no new proc macro surface, and instead simply exposes
a `--attribute` flag. This flag, like `-Zextra-plugins`, lets you attach a proc macro attribute
to the whole crate before compiling. (This flag actually generalizes a bunch of flags that the
compiler already has)

Test crates are now simply proc macro attributes:

```rust
#[test_framework(attributes(test, foo, bar))]
pub fn harness(crate: TokenStream) -> TokenStream {
  // ...
}
```

The cargo functionality will basically compile the file with the right dependencies
and `--attribute=your_crate::harness`.


### Standardizing the output

We should probably provide a crate with useful output formatters and stuff so that if test harnesses desire, they can
use the same output formatting as a regular test. This also provides a centralized location to standardize things
like json output and whatnot.

@killercup is working on a proposal for this which I will try to work in.

# Drawbacks
[drawbacks]: #drawbacks

 - This adds more sections to `Cargo.toml`.
 - This complicates the execution path for cargo, in that it now needs
   to know about testing frameworks.
 - Flags and command-line parameters for test and bench will now vary
   between testing frameworks, which may confuse users as they move
   between crates.

# Rationale and alternatives
[alternatives]: #alternatives

We could stabilize `#[bench]` and extend libtest with setup/teardown and
other requested features. This would complicate the in-tree libtest,
introduce a barrier for community contributions, and discourage other
forms of testing or benchmarking.

# Unresolved questions
[unresolved]: #unresolved-questions

These are mostly intended to be resolved during the experimental
feature. Many of these have strawman proposals -- unlike the rest of this RFC,
these proposals have not been discussed as thoroughly. If folks feel like
there's consensus on some of these we can move them into the main RFC.

## Integration with doctests

Documentation tests are somewhat special, in that they cannot easily be
expressed as `TokenStream` manipulations. In the first instance, the
right thing to do is probably to have an implicitly defined framework
 called `doctest` which is included in the testing set
`test` by default (as proposed above).

Another argument for punting on doctests is that they are intended to
demonstrate code that the user of a library would write. They're there
to document *how* something should be used, and it then makes somewhat
less sense to have different "ways" of running them.

## Standardizing the output

We should probably provide a crate with useful output formatters and
stuff so that if test harnesses desire, they can use the same output
formatting as a regular test. This also provides a centralized location
to standardize things like json output and whatnot.

## Namespacing

Currently, two frameworks can both declare interest in the same
attributes. How do we deal with collisions (e.g., most test crates will
want the attribute `#[test]`). Do we namespace the attributes by the
framework name (e.g., `#[mytest::test]`)? Do we require them to be behind
`#[cfg(mytest)]`?

## Runtime dependencies and flags

The code generated by the framework may itself have dependencies.
Currently there's no way for the framework to specify this. One
proposal is for the crate to specify  _runtime_ dependencies of the
framework via:

```toml
[testing.framework.dependencies]
libfuzzer-sys = ...
```

If a crate is currently running this framework, its
dev-dependencies will be semver-merged with the frameworks's
`framework.dependencies`. However, this may not be strictly necessary.
Custom derives have a similar problem and they solve it by just asking
users to import the correct crate.

## Naming

The general syntax and toml stuff should be approximately settled on before this eRFC merges, but
iterated on later. Naming the feature is hard, some candidates are:

 - testing framework
 - post-build context
 - build context
 - execution context

None of these are particularly great, ideas would be nice.

## Bencher

Should we be shipping a bencher by default at all (i.e., in libtest)? Could we instead default
`cargo bench` to a `rust-lang-nursery` `bench` crate?

If this RFC lands and [RFC 2287] is rejected, we should probably try to stabilize
`test::black_box` in some form (maybe `mem::black_box` and `mem::clobber` as detailed
in [this amendment]).

## Cargo integration

A previous iteration of this RFC allowed for test frameworks to declare new attributes
and folders, so you would have `cargo test --kind quickcheck` look for tests in the
`quickcheck/` folder that were annotated with `#[quickcheck]`.

This is no longer the case, but we may wish to add this again.

 [RFC 2287]: https://github.com/rust-lang/rfcs/pull/2287
 [this amendment]: https://github.com/Manishearth/rfcs/pull/1
