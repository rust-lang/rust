- Feature Name: post_build_contexts
- Start Date: 2018-01-25
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This is an *experimental RFC* for adding the ability to integrate custom test/bench/etc frameworks ("post-build frameworks") in Rust.

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

The main two problems that we need to solve are:

 - Having a nice API for generating custom post-build binaries
 - Having good `cargo` integration so that custom tests are at the same level of integration as regular tests as far as build processes are concerned

 [cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
 [criterion]: https://github.com/japaric/criterion.rs
 [Compiletest]: http://github.com/laumann/compiletest-rs

# Detailed proposal
[detailed-proposal]: #detailed-proposal

(As an eRFC I'm merging the "guide-level/reference-level" split for now; when we have more concrete
ideas we can figure out how to frame it and then the split will make more sense)

## Procedural macro for a new post-build context

A custom post-build context is essentially a whole-crate procedural
macro that is evaluated after all other macros in the target crate have
been evaluated. It is passed the `TokenStream` for every element in the
target crate that has a set of attributes the post-build context has
registered interest in. Essentially:

```rust
extern crate proc_macro;
use proc_macro::TokenStream;

// attributes() is optional
#[post_build_context(test, attributes(foo, bar))]
pub fn like_todays_test(items: &[AnnotatedItem]) -> TokenStream {
    // ...
}
```

where

```rust
struct AnnotatedItem
    tokens: TokenStream,
    span: Span,
    attributes: TokenStream,
    path: SomeTypeThatRepresentsPathToItem
}
```

`items` here contains an `AnnotatedItem` for every element in the
target crate that has one of the attributes declared in `attributes`
along with attributes sharing the name of the context (`test`, here).

A post-build context could declare that it reacts to multiple different
attributes, in which case it would get all items with any of the
listed attributes. These items be modules, functions, structs,
statics, or whatever else the post-build context wants to support. Note
that the post-build context function can only see all the annotated
items, not modify them; modification would have to happen with regular
procedural macros The returned `TokenStream` will become the `main()`
when this post-build context is used.

Because this procedural macro is only loaded when it is used as the
post-build context, the `#[test]` annotation should probably be kept
behind `#[cfg(test)]` so that you don't get unknown attribute warnings
whilst loading. (We could change this by asking attributes to be
registered in Cargo.toml, but we don't find this necessary)

## Cargo integration

Alternative post-build contexts need to integrate with cargo.
In particular, when crate `a` uses a crate `b` which provides an
post-build context, `a` needs to be able to specify when `b`'s post-build
context should be used. Furthermore, cargo needs to understand that when
`b`'s post-build context is used, `b`'s dependencies must also be linked.
Note that `b` could potentially provide multiple post-build contexts ---
these are named according to the name of their `#[post_build_context]`
function.

Crates which define a post-build context must have an `post-build-context = true`
key.

For crates that wish to *use* a custom post-build context, they do so by
defining a new post-build context under a new `post-build` section in
their `Cargo.toml`:

```toml
[post-build.context.fuzz]
provider = { rust-fuzz = "1.0" }
folder = "fuzz/"
specify-single-target = true    # false by default
```

This defines a post-build context named `fuzz`, which uses the
implementation provided by the `rust-fuzz` crate. When run, it will be
applies to all files in the `fuzz` directory. `specify-single-target`
addresses whether it must be run with a single target. If true, you will
be forced to run `cargo post-build foobar --test foo`. This is useful for cases
like `cargo-fuzz` where running tests on everything isn't possible.

By default, the following contexts are defined:

```toml
[post-build.context.test]
provider = { test = "1.0", context = "test" }
folder = "tests/"

[post-build.context.bench]
provider = { test = "1.0", context = "bench" }
folder = ["benchmarks/", "morebenchmarks/"]
```

These can be overridden by a crate's `Cargo.toml`. The `context`
property is used to disambiguate when a single crate has multiple
functions tagged `#[post_build_context]` (if we were using the example
post-build provider further up, we'd give `like_todays_test` here).
`test` here is `libtest`, though note that it could be maintained
out-of-tree, and shipped with rustup.

To invoke a particular post-build context, a user invokes `cargo post-build
<context>`. `cargo test` and `cargo bench` are aliases for `cargo
post-build test` and `cargo post-build bench` respectively. Any additional
arguments are passed to the post-build context binary. By convention, the
first position argument should allow filtering which
test/benchmarks/etc. are run.


By default, the crate has an implicit "test", "bench", and "example" context that use the default libtest stuff.
(example is a no-op context that just runs stuff). However, declaring a context with the name `test`
will replace the existing `test` context. In case you wish to supplement the context, use a different
name.

By default, `cargo test` will run doctests and the `test` and `examples` context. This can be customized:

```toml
[post-build.set.test]
contexts = [test, quickcheck, examples]
```

This means that `cargo test` will, aside from doctests, run `cargo post-build test`, `cargo post-build quickcheck`,
and `cargo post-build examples` (and similar stuff for `cargo bench`). It is not possible to make `cargo test`
_not_ run doctests.

There are currently only two custom post-build sets (test and bench).

Custom test targets can be declared via `[[post-build.target]]`

```toml
[[post-build.target]]
context = fuzz
path = "foo.rs"
name = "foo"
```

`[[test]]` is an alias for `[[post-build.target]] context = test` (same goes for `[[bench]]` and `[[example]]`).


The generated test binary should be able to take one identifier argument, used for narrowing down what tests to run.
I.e. `cargo test --kind quickcheck my_test_fn` will build the test(s) and call them with `./testbinary my_test_fn`.
Typically, this argument is used to filter tests further; test harnesses should try to use it for the same purpose.


## To be designed

This contains things which we should attempt to solve in the course of this experiment, for which this eRFC
does not currently provide a concrete proposal.

### Standardizing the output

We should probably provide a crate with useful output formatters and stuff so that if test harnesses desire, they can
use the same output formatting as a regular test. This also provides a centralized location to standardize things
like json output and whatnot.

@killercup is working on a proposal for this which I will try to work in.

### Configuration

Currently we have `cfg(test)` and `cfg(bench)`. Should `cfg(test)` be applied to all? Should `cfg(nameofharness)`
be used instead? Ideally we'd have a way when declaring a framework to declare what cfgs it should be built with.

# Drawbacks
[drawbacks]: #drawbacks

 - This adds more sections to `Cargo.toml`.
 - This complicates the execution path for cargo, in that it now needs
   to know about post-build contexts and sets.
 - Flags and command-line parameters for test and bench will now vary
   between post-build contexts, which may confuse users as they move
   between crates.

# Rationale and alternatives
[alternatives]: #alternatives

We should either do this or stabilize the existing bencher.

## Alternative procedural macro

An alternative proposal was to expose an extremely general whole-crate proc macro:

```rust
#[post_build_context(test, attributes(foo, bar))]
pub fn context(crate: TokenStream) -> TokenStream {
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
these tools usually operate at a differeny layer of abstraction so it might not be necessary.

A major drawback of this proposal is that it is very general, and perhaps too powerful. We're currently using the
more focused API in the eRFC, and may switch to this during experimentation if a pressing need crops up.

## Alternative procedural macro with minimal compiler changes

The above proposal can be made even more general, minimizing the impact on the compiler.

This assumes that `#![foo]` ("inner attribute") macros work on modules and on crates.

The idea is that the compiler defines no new proc macro surface, and instead simply exposes
a `--attribute` flag. This flag, like `-Zextra-plugins`, lets you attach a proc macro attribute
to the whole crate before compiling. (This flag actually generalizes a bunch of flags that the
compiler already has)

Test crates are now simply proc macro attributes:

```rust
#[proc_macro_attr(attributes(test, foo, bar))]
pub fn harness(crate: TokenStream) -> TokenStream {
  // ...
}
```

The cargo functionality will basically compile the file with the right dependencies
and `--attribute=your_crate::harness`.

# Unresolved questions
[unresolved]: #unresolved-questions

These are mostly intended to be resolved during the experimental
feature. Many of these have strawman proposals -- unlike the rest of this RFC,
these proposals have not been discussed as thoroughly. If folks feel like
there's consensus on some of these we can move them into the main RFC.

## Integration with doctests

Documentation tests are somewhat special, in that they cannot easily be
expressed as `TokenStream` manipulations. In the first instance, the
right thing to do is probably to have an implicitly defined execution
context called `doctest` which is included in the execution context set
`test` by default.

Another argument for punting on doctests is that they are intended to
demonstrate code that the user of a library would write. They're there
to document *how* something should be used, and it then makes somewhat
less sense to have different "ways" of running them.

## Translating existing cargo test flags

Today, `cargo test` takes a number of flags such as `--lib`, `--test
foo`, and `--doc`. As it would be a breaking change to change these,
cargo should recognize them and map to the appropriate execution
contexts.

Currently, `cargo test` lets you pick a single testing target via `--test`,
and `cargo bench` via `--bench`. We'll need to create an agnostic flag
for `cargo post-build` (we cannot use `--target` because it is already used for
the target architecture, and `--test` is too specific for tests). `--post-build-target`
is one rather verbose suggestion.

## Standardizing the output

We should probably provide a crate with useful output formatters and
stuff so that if test harnesses desire, they can use the same output
formatting as a regular test. This also provides a centralized location
to standardize things like json output and whatnot.

## Configuration

Currently we have `cfg(test)` and `cfg(bench)`. Should `cfg(test)` be
applied to all? Should `cfg(post_build_context)` be used instead?
Ideally we'd have a way when declaring a post-build context to declare
what cfgs it should be built with.

## Runtime dependencies and flags

The generated harness itself may have some dependencies. Currently there's
no way for the post-build context to specify this. One proposal is for the crate
to specify  _runtime_ dependencies of the post-build context via:

```toml
[post-build.dependencies]
libfuzzer-sys = ...
```

If a crate is currently running this post-build context, its dev-dependencies
will be semver-merged with the post-build-context.dependencies.

However, this may not be strictly necessary. Custom derives have
a similar problem and they solve it by just asking users to import the correct
crate and keep it in their dev-dependencies.

## Naming

The general syntax and toml stuff should be approximately settled on before this eRFC merges, but
iterated on later. Naming the feature is hard, some candidates are:

 - test framework
 - post-build context
 - execution context

None of these are particularly great, ideas would be nice.

## Default folders and sets

Should a post-build context be able to declare "defaults" for what folders and post-build sets it
should be added to? This might save users from some boilerplate in a large number of situations.

This could be done in the Cargo.toml as:

```toml
[post-build.defaults]
folder = "tests/"
set = "test" # will automatically be added to the `test` set
```

This is useful if a crate wishes to standardize things.

## Bencher

Should we be shipping a bencher by default at all (i.e., in libtest)? Could we instead default
`cargo bench` to a `rust-lang-nursery` `bench` crate?

If this RFC lands and [RFC 2287] is rejected, we should probably try to stabilize
`test::black_box` in some form (maybe `mem::black_box` and `mem::clobber` as detailed
in [this amendment]).



 [RFC 2287]: https://github.com/rust-lang/rfcs/pull/2287
 [this amendment]: https://github.com/Manishearth/rfcs/pull/1

## Specify-single-target

`specify-single-target = true` probably should be specified by the execution context itself, not the
consumer. It's also questionable if it's necessary -- cargo-fuzz is going to need a wrapper script
anyway, so it's fine if the CLI isn't as ergonomic for that use case.

If we do `post-build.defaults` it would just make sense to include that there.





 