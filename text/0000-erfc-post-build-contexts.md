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

The main two features proposed are:

 - An API for crates that generate custom binaries, including
   introspection into the target crate.
 - A mechanism for `cargo` integration so that custom post-build
   contexts are at the same level of integration as `test` or `bench` as
   far as build processes are concerned.

 [cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
 [criterion]: https://github.com/japaric/criterion.rs
 [Compiletest]: http://github.com/laumann/compiletest-rs

# Detailed proposal
[detailed-proposal]: #detailed-proposal

(As an eRFC I'm merging the "guide-level/reference-level" split for now; when we have more concrete
ideas we can figure out how to frame it and then the split will make more sense)

The basic idea is that crates can define post-build contexts, which specify 
how to transform collected test functions and construct a `main()` function,
and then crates using these can declare them in their Cargo.toml, which will let
crate developers invoke various test-like post-build steps using the post-build
context.


## Procedural macro for a new post-build context

A custom post-build context is like a whole-crate procedural
macro that is evaluated after all other macros in the target crate have
been evaluated. It is passed the `TokenStream` for every element in the
target crate that has a set of attributes the post-build context has
registered interest in. For example, to declare a post-build context
called `mytest`:

```rust
extern crate proc_macro;
use proc_macro::TokenStream;

// attributes() is optional
#[post_build_context(attributes(foo, bar))]
pub fn mytest(items: &[AnnotatedItem]) -> TokenStream {
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

`items` here contains an `AnnotatedItem` for every item in the
target crate that has one of the attributes declared in `attributes`
along with attributes sharing the name of the context (`mytest`, here).

A post-build context could declare that it reacts to multiple different
attributes, in which case it would get all items with any of the
listed attributes. These items be modules, functions, structs,
statics, or whatever else the post-build context wants to support. Note
that the post-build context function can only see all the annotated
items, not modify them; modification would have to happen with regular
procedural macros. The returned `TokenStream` must declare the `main()`
that is to become the entry-point for the binary produced when this
post-build context is used.

So an example transformation would be to take something like this:

```rust
#[quickcheck]
fn foo(x: u8) {
    // ...
}

mod bar {
    #[quickcheck]
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

Because this procedural macro is only loaded when it is used as the
post-build context, the `#[mytest]` annotation should probably be kept
behind `#[cfg(mytest)]` (which is automatically set when the `mytest`
context is used) so that you don't get unknown attribute warnings
whilst loading, and to avoid conflicts with other post-build contexts
that may use the same attributes. (We could change this by asking
attributes to be registered in Cargo.toml, but we don't find this
necessary)

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
key in their `Cargo.toml`. It can also specify

```rust
single-target = true    # false by default
```

`single-target` indicates that only a single target can be run with this
context at once (some tools, like cargo-fuzz, run forever, and so it
does not make sense to specify multiple targets).

Crates that wish to *use* a custom post-build context, do so by defining
a new post-build context under a new `build-context` section in their
`Cargo.toml`:

```toml
[post-build.context.fuzz]
provider = { rust-fuzz = "1.0" }
folders = ["fuzz/"]
```

This defines a post-build context named `fuzz`, which uses the
implementation provided by the `rust-fuzz` crate. When run, it will be
applied to all files in the `fuzz` directory. By default, the following
contexts are defined:

```toml
[post-build.context.test]
provider = { test = "1.0", context = "test" }
folders = ["tests/"]

[post-build.context.bench]
provider = { test = "1.0", context = "bench" }
folders = ["benchmarks/"]
```

There's also an `example` context defined that just runs the `main()` of
any files given.

These can be overridden by a crate's `Cargo.toml`. The `context`
property is used to disambiguate when a single crate has multiple
functions tagged `#[post_build_context]` (if we were using the example
post-build context further up as a provider, we'd give `mytest` here).
`test` here is `libtest`, though note that it could be maintained
out-of-tree, and shipped with rustup.

To invoke a particular post-build context, a user invokes `cargo context
<context>`. `cargo test` and `cargo bench` are aliases for `cargo
context test` and `cargo context bench` respectively. Any additional
arguments are passed to the post-build context binary. By convention, the
first position argument should allow filtering which targets
(tests/benchmarks/etc.) are run.

To run multiple contexts at once, a crate can declare post-build context
*sets*. One such example is the `test` post-build context set, which
will run doctests and the `test` and `examples` context. This can be
customized:

```toml
[post-build.set.test]
contexts = [test, quickcheck, examples]
```

This means that `cargo test` will, aside from doctests, run `cargo
context test`, `cargo context quickcheck`, and `cargo context examples`
(and similar stuff for `cargo bench`). It is not possible to make `cargo
test` _not_ run doctests. If both a context and a set exists with a
given name, the set takes precedence.

`[[test]]` and `[[example]]` in a crate's `Cargo.toml` add files to the
`test` and `example` contexts respectively.

## To be designed

This contains things which we should attempt to solve in the course of this experiment, for which this eRFC
does not currently provide a concrete proposal.

### Standardizing the output

We should probably provide a crate with useful output formatters and stuff so that if test harnesses desire, they can
use the same output formatting as a regular test. This also provides a centralized location to standardize things
like json output and whatnot.

@killercup is working on a proposal for this which I will try to work in.

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

We could stabilize `#[bench]` and extend libtest with setup/teardown and
other requested features. This would complicate the in-tree libtest,
introduce a barrier for community contributions, and discourage other
forms of testing or benchmarking.

## Alternative procedural macro

An alternative proposal was to expose an extremely general whole-crate proc macro:

```rust
#[post_build_context(attributes(foo, bar))]
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
right thing to do is probably to have an implicitly defined post-build
context called `doctest` which is included in the post-build context set
`test` by default (as proposed above).

Another argument for punting on doctests is that they are intended to
demonstrate code that the user of a library would write. They're there
to document *how* something should be used, and it then makes somewhat
less sense to have different "ways" of running them.

## Translating existing cargo test flags

Today, `cargo test` takes a number of flags such as `--lib`, `--test
foo`, and `--doc`. As it would be a breaking change to change these,
cargo should recognize them and map to the appropriate post-build
contexts.

Furthermore, `cargo test` lets you pick a single testing target via `--test`,
and `cargo bench` via `--bench`. We'll need to create an agnostic flag
for `cargo context` (we cannot use `--target` because it is already used for
the target architecture, and `--test` is too specific for tests). `--post-build-target`
is one rather verbose suggestion.

We also need to settle on a command name, `cargo context` and `cargo post-build`
don't quite capture what's going on.

## Standardizing the output

We should probably provide a crate with useful output formatters and
stuff so that if test harnesses desire, they can use the same output
formatting as a regular test. This also provides a centralized location
to standardize things like json output and whatnot.

## Namespacing

Currently, two post-build contexts can both declare interest in the same
attributes. How do we deal with collisions (e.g., most test crates will
want the attribute `#[test]`). Do we namespace the attributes by the
context name (e.g., `#[mytest::test]`)? Do we require them to be behind
`#[cfg(mytest)]`?

## Runtime dependencies and flags

The code generated by the post-build context may itself have dependencies.
Currently there's no way for the post-build context to specify this. One
proposal is for the crate to specify  _runtime_ dependencies of the
post-build context via:

```toml
[context-dependencies]
libfuzzer-sys = ...
```

If a crate is currently running this post-build context, its
dev-dependencies will be semver-merged with the post-build context's
`context-dependencies`. However, this may not be strictly necessary.
Custom derives have a similar problem and they solve it by just asking
users to import the correct crate.

## Naming

The general syntax and toml stuff should be approximately settled on before this eRFC merges, but
iterated on later. Naming the feature is hard, some candidates are:

 - test framework
 - build context
 - execution context

None of these are particularly great, ideas would be nice.

## Default folders and sets

Should a post-build context be able to declare "defaults" for what folders and post-build sets it
should be added to? This might save users from some boilerplate in a large number of situations.

This could be done in the Cargo.toml as:

```toml
[post-build.defaults]
folders = ["tests/"]
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
