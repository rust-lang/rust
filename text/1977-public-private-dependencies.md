- Feature Name: `public_private_dependencies`
- Start Date: 2017-04-03
- RFC PR: https://github.com/rust-lang/rfcs/pull/1977
- Rust Issue: https://github.com/rust-lang/rust/issues/44663

# Summary
[summary]: #summary

Introduce a public/private distinction to crate dependencies.

# Motivation
[motivation]: #motivation

The crates ecosystem has greatly expanded since Rust 1.0. With that, a few patterns for
dependencies have evolved that challenge the currently existing dependency declaration
system in Cargo and Rust. The most common problem is that a crate `A` depends on another
crate `B` but some of the types from crate `B` are exposed through the API in crate `A`.
This causes problems in practice if that dependency `B` is also used by the user's code
itself, crate `B` resolves to different versions for each usage, and the values of types
from the two crate `B` instances need to be used together but don't match. In this case,
the user's code will refuse to compile because different versions of those libraries are
requested, and the compiler messages are less than clear.

The introduction of an explicit distinction between public and private dependencies can
solve some of these issues. This distinction should also let us lift some restrictions and
make some code compile that previously was prevented from compiling.

**Q: What is a public dependency?**<br>
A: A dependency is public if some of the types or traits of that dependency are themselves
exported through the public API of main crate. The most common places where this happens
are return values and function parameters. The same applies to trait implementations and
many other things. Because "public" can be tricky to determine for a user, this RFC
proposes to extend the compiler infrastructure to detect the concept of a "public
dependency". This will help the user understand this concept so they can avoid making
mistakes in the `Cargo.toml`.

Effectively, the idea is that if you bump a public dependency's version, it's a breaking
change of your *own* crate.

**Q: What is a private dependency?**<br>
A: On the other hand, a private dependency is contained within your crate and effectively
invisible for users of your crate. As a result, private dependencies can be freely
duplicated in the dependency graph and won't cause compilation errors. This distinction
will also make it possible to relax some restrictions that currently exist in Cargo which
sometimes prevent crates from compiling.

**Q: Can public become private later?**<br>
A: Public dependencies are public within a reachable subgraph but can become private if a
crate stops exposing a public dependency. For instance, it is very possible to have a
family of crates that all depend on a utility crate that provides common types which is a
public dependency for all of them. However, if your own crate ends up being a user of this
utility crate but none of its types or traits become part of your own API, then this
utility crate dependency is marked private.

**Q: Where is public / private defined?**<br>
Dependencies are private by default and are made public through a `public` flag on the
dependency in the `Cargo.toml` file. This also means that crates created before the
implementation of this RFC will have all their dependencies private.

**Q: How is backwards compatibility handled?**<br>
A: It will continue to be permissible to "leak" dependencies (and there are even some use
cases of this), however, the compiler or Cargo will emit warnings if private dependencies
are part of the public API. Later, it might even become invalid to publish new crates
without explicitly silencing these warnings or marking the dependencies as public.

**Q: Can I export a type from a private dependency as my own?**<br>
A: For now, it will not be strictly permissible to privately depend on a crate and export a
type from there as your own. The reason for this is that at the moment it is not possible
to force this type to be distinct. This means that users of the crate might accidentally
start depending on that type to be compatible if the user starts to depend on the crate
that actually implements that type. The limitations from the previous answer apply (e.g.:
you can currently overrule the restrictions).

**Q: How do semver and dependencies interact?**<br>
A: It is already the case that changing your own dependencies would require a semver bump
for your own library because your API contract to the outside world changes. This RFC,
however, makes it possible to only have this requirement for public dependencies and would
permit Cargo to prevent new crate releases with semver violations.

# Detailed design
[design]: #detailed-design

There are a few areas that need to be changed for this RFC:

* The compiler needs to be extended to understand when crate dependencies are
  considered a public dependency
* The `Cargo.toml` manifest needs to be extended to support declaring public
  dependencies. This will start as an unstable cargo feature available on nightly
  and only via opt-in.
* The `public` attribute of dependencies needs to appear in the Cargo index in order
  to be used by Cargo during version resolution
* Cargo's version resolution needs to change to reject crate graph resolutions where
  two versions of a crate are publicly reachable to each other.
* The `cargo publish` process needs to be changed to warn (or prevent) the publishing
  of crates that have undeclared public dependencies
* `cargo publish` will resolve dependencies to the *lowest* possible versions in order to
  check that the minimal version specified in `Cargo.toml` is correct.
* Crates.io should show public dependencies more prominently than private ones.

## Compiler Changes

The main change to the compiler will be to accept a new parameter that Cargo
supplies which is a list of public dependencies. The flag will be called
`--extern-public`. The compiler then emits warnings if it encounters private
dependencies leaking to the public API of a crate. `cargo publish` might change
this warning into an error in its lint step.

Additionally, later on, the warning can turn into a hard error in general.

In some situations, it can be necessary to allow private dependencies to become
part of the public API. In that case one can permit this with
`#[allow(external_private_dependency)]`. This is particularly useful when
paired with `#[doc(hidden)]` and other already existing hacks.

This most likely will also be necessary for the more complex relationship of
`libcore` and `libstd` in Rust itself.

## Changes to `Cargo.toml`

The `Cargo.toml` file will be amended to support the new `public` parameter on
dependencies. Old Cargo versions will emit a warning when this key is encountered
but otherwise continue. Since the default for a dependency to be private only,
public ones will need to be tagged which should be the minority.

This will start as an unstable Cargo feature available on nightly only that authors
will need to opt into via a feature specified in `Cargo.toml` before Cargo will
start using the `public` attribute to change the way versions are resolved. The
Cargo unstable feature will turn on a corresponding rustc unstable feature for
the compiler changes noted above.

Example dependency:

```toml
[dependencies]
url = { version = "1.4.0", public = true }
```

## Changes to the Cargo Index

The [Cargo index](https://github.com/rust-lang/crates.io-index) used by Cargo when
resolving versions will contain the `public` attribute on dependencies as specified
in `Cargo.toml`. For example, an index line for a crate named `example` that
publicly depends on the `url` crate would look like (JSON prettified for legibility):

```json
{
    "name":"example",
    "vers":"0.1.0",
    "deps":[
        {
            "name":"url",
            "req":"^1.4.0",
            "public":"true",
            "features":[],
            "optional":false,
            "default_features":true,
            "target":null,
            "kind":"normal"
        }
    ]
}
```

## Changes to Cargo Version Resolution

Cargo will specifically reject graphs that contain two different versions of the
same crate being publicly depended upon and reachable from each other. This will
prevent the strange errors possible today at version resolution time rather than at
compile time.

How this will work:

* First, a resolution graph has a bunch of nodes. These nodes are "package ids"
  which are a triple of (name, source, version). Basically this means that different
  versions of the same crate are different nodes, and different sources of the same
  name (e.g. git and crates.io) are also different nodes.
* There are *directed edges* between nodes. A directed edge represents a dependency.
  For example if A depends on B then there's a directed edge from A to B.
* With public/private dependencies, we can now say that every edge is either tagged
  with public or private.
* This means that we can have a collection of subgraphs purely connected by public
  dependency edges. The directionality of the public dependency edges within the
  subgraph doesn't matter. Each of these subgraphs represents an "ecosystem" of
  crates publicly depending on each other. These subgraphs are "pools of public
  types" where if you have access to the subgraph, you have access to all types
  within that pool of types.
* We can place a constraint that each of these "publicly connected subgraphs" are
  required to have exactly one version of all crates internally. For example, each
  subgraph can only have one version of Hyper.
* Finally, we can consider all pairs of edges coming out of one node in the
  resolution graph. If the two edges point to *two distinct publicly connected
  subgraphs from above* and those subgraphs contain two different versions of the
  same crate, we consider that an error. This basically means that if you privately
  depend on Hyper 0.3 and Hyper 0.4, that's an error.

## Changes to Cargo Publish: Warnings

When a new crate version is published, Cargo will warn about types and traits that
the compiler determined to be public but did not come from a public dependency. For
now, it should be possible to publish anyways but in some period in the future it
will be necessary to explicitly mark all public dependencies as such or explicitly
mark them with `#[allow(external_private_dependency)]`.

## Changes to Cargo Publish: Lowest Version Resolution

A very common situation today is that people write the initial version of a
dependency in their Cargo.toml, but never bother to update it as they take advantage
of new features in newer versions. This works out okay because (1) Cargo will
generally use the largest version it can find, compatible with constraints, and (2)
upper bounds on constraints (at least within a particular minor version) are
relatively rare. That means, in particular, that Cargo.toml is not a fully accurate
picture of version dependency information; in general it's a lower bound at best.
There can be "invisible" dependencies that don't cause resolution failures but can
create compilation errors as APIs evolve.

Public dependencies exacerbate the above problem, because you can end up relying on
features of a "new API" from a crate you didn't even know you depended on! For
example:

- A depends on:
  - B 1.0 which publicly depends on C ^1.0
  - D 1.0, which has no dependencies
- When A is initially built, it resolves to B 1.0 and C 1.1.
  - Because C's APIs are available to A via re-exports in B, A effectively depends
    on C 1.1 now, even though B only claims to depend on C ^1.0
  - In particular, the code in A might depend on APIs only available in C 1.1
  - However, if A is a library, we don't check in any lockfile for it, so this
    information is lost.
- Now we change A to depend on D 1.1, which depends on C =1.0
  - A fresh copy of A, when built, will now resolve the crate graph to B 1.0, D 1.1,
    C 1.0
  - But now A may suddenly fail to compile, because it was implicitly depending on C
    1.1 features via B.

This example and others like it rely on a common ingredient: a crate somewhere using
an API that only is available in a newer version of a crate than the version listed
in Cargo.toml.

To attempt to surface this problem earlier, `cargo publish` will attempt to resolve
the graph while picking the smallest versions compatible with constraints. If the
crate fails to build with this resolution graph, the publish will fail.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

From the user's perspective, the initial scope of the RFC will be quite transparent,
but it will definitely show up for users as a question of what the new restrictions
mean. In particular, a common way to leak out types from APIs that most crates do is
error handling. Quite frequently it happens that users wrap errors from other
libraries in their own types. It might make sense to identify common cases of where
type leakage happens and provide hints in the lint about how to deal with it.

Cases that I anticipate that should be explained separately:

* Type leakage through errors: This should be easy to spot for a lint because the
  wrapper type will implement `std::error::Error`. The recommendation should most
  likely be to encourage wrapping the internal error.
* Traits from other crates: In particular, serde and some other common crates will
  show up frequently. It might make sense to separately explain types and traits.
* Type leakage through derive: Users might not be aware they have a dependency on
  a type when they derive a trait (think `serde_derive`). The lint might want to
  call this out separately.

The feature will be called `public_private_dependencies` and it comes with one
lint flag called `external_private_dependency`. For all intents and purposes, this
should be the extent of the new terms introduced in the beginning. This RFC, however,
lays the groundwork for later providing aliasing so that a private dependency could
be forcefully re-exported as the crate's own types. As such, it might make sense to
consider how to refer to this.

It is assumed that this feature will eventually become quite popular due to patterns
that already exist in the crate ecosystem. It's likely that it will evoke some
negative opinions initially. As such, it would be a good idea to make a run with
cargobomb/crater to see what the actual impact of the new linter warnings is and
how far away we are from making them errors.

Crates.io should be updated to render public and private dependencies separately.

# End user experience
[end-user-experience]: #end-user-experience

## Author of a crate with one dependency

Assume today that an author of a library crate `onedep` has a
dependency on the `url` crate and the `url::Url` type is exposed in
`onedep`'s public API.

`onedep`'s `Cargo.toml`:

```toml
[package]
name = "onedep"
version = "0.1.0"

[dependencies]
url = "1.0.0"
```

`onedep`'s `src/lib.rs`:

```rust
extern crate url;
use url::Origin;

use std::collections::HashMap;

#[derive(Default)]
pub struct OriginTracker {
    origin_counts: HashMap<Origin, usize>,
}

impl OriginTracker {
    pub fn log_origin(&mut self, origin: Origin) {
        let counter = self.origin_counts.entry(origin).or_insert(0);
        *counter += 1;
    }
}
```

When the author of `onedep` upgrades Rust/Cargo to a version where this RFC is
completely implemented, the author will notice two changes:

1. When they run `cargo build`, the build will succeed but they will get a warning
that a private dependency (the `url` crate specifically) is used in their public API
(the `url::Origin` type in the `pub fn log_origin` function specifically) and that
they should consider adding `public = true` to their `Cargo.toml`. Ideally the
warning would say something like:

    ```
        consider changing dependency:

        ```
        url = "1.0.0"
        ```

        to:

        ```
        url = { version = "1.0.0", public = true }
        ```
    ```

The warning could also encourage the author to then bump their crate's major
version since adding public dependencies is a breaking change.

2. When they run `cargo publish`, the build check that happens after packaging will
fail and the publish will fail. This is because [deriving `Hash` on `url::Origin`
wasn't added until v1.5.1 of the url
crate](https://github.com/servo/rust-url/commit/42603254fac8d4c446183cba73bbaeb2c3b416c2).
The author of `onedep` has been running `cargo update` periodically, and their
`Cargo.lock` has url 1.5.1, but they never updated `Cargo.toml` to indicate that
they have a new lower bound. Since `cargo publish` will try to resolve dependencies
to the lowest possible versions, it will choose version 1.0.0 of the url crate,
which doesn't implement `Hash` on `Origin`.

There should be a clear error message for this case that indicates Cargo has
resolved crates to their lowest possible versions, that this might be the cause of
the compilation failure, and that the author should investigate the versions of
their dependencies in `Cargo.toml` to see if they should be updated. This command
should change the Cargo.lock so that running `cargo build` will reproduce the error
for the author to fix.

## Author of a crate with multiple dependencies

`twodep`'s `Cargo.toml`:

```toml
[package]
name = "twodep"
version = "0.1.0"

[dependencies]
// this is the version of onedep above using a public dep on url 1.5.1
onedep = "1.0.0"
url = "1.0.0"
```

`twodep`'s `src/main.rs`:

```rust
extern crate url;
use url::Origin;

extern crate onedep;

fn main() {
    let mut origin_tracker = onedep::OriginTracker::default();

    loop {
        println!("Please enter a URL!");
        // pseudocode because I'm lazy
        let url = stdin::readline().unwrap();
        let url = Url::parse(url).unwrap();
        origin_tracker.log_origin(url.origin());
        // other stuff
    }
    println!("Here are all the origins you mentioned: {:#?}", origin_tracker);
}
```

Before upgrading Rust/Cargo to a version where this RFC has been implemented, this
code might have been getting a compilation error if Cargo had resolved the direct
dependency on the url crate to a different version than the version of onedep
resolved to. Or it might have been resolving and compiling fine if the versions had
resolved to be the same.

After upgrading Rust/Cargo, if this code had a compilation error, it would now have
a version resolution problem that cargo would either automatically resolve or prompt
the user to change version constraints/`cargo update` to resolve. If the code was
compiling before, that must mean the previous resolution graph was good, so nothing
will change on upgrading.

This crate is a binary and doesn't have a public API, so it won't get any warnings
about crates not being marked public.

If the author publishes to crates.io after upgrading Rust/Cargo, since onedep's
public dependency on url now has a lower bound of 1.5.1, the only valid graphs that
Cargo will generate will be with url 1.5.1 or greater, which is also compatible with
the url 1.0.0 direct dependency. Publish will work without any errors or further
changes.

# Drawbacks
[drawbacks]: #drawbacks

I believe that there are no drawbacks if implemented well (this assumes good
linters and error messages).

# Alternatives
[alternatives]: #alternatives

For me, the biggest alternative to this RFC would be a variation of it where type
and trait aliasing becomes immediately part of it. This would mean that a crate
can have a private dependency and re-export it as its own type, hiding where it
came from originally. This would most likely be easier to teach users and can get
rid of a few "cul-de-sac" situations users can end up in where their only way
out is to introduce a public dependency for now. The assumption is that if trait
and type aliasing is available, the `external_public_dependency` would not need to
exist.

# Unresolved questions
[unresolved]: #unresolved-questions

There are a few open questions about how to best hook into the compiler and Cargo
infrastructure:

* What is the impact of this change going to be? This most likely can be answered
  running cargobomb/crater.
* Since changing public dependency pins/ranges requires a change in semver, it might
  be worth exploring if Cargo could prevent the user from publishing new crate
  versions that violate that constraint.
* If this is implemented before [the RFC to deprecate `extern crate`](https://github.com/rust-lang/rfcs/pull/2126), how would this work if you're not using `--extern`?
