- Feature Name: `public_private_dependencies`
- Start Date: 2017-04-03
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Introduce a public/private distinction to crate dependencies.

# Motivation
[motivation]: #motivation

The crates ecosystem has greatly expanded since Rust 1.0 and with that a few patterns for
dependencies have evolved that challenge the currently existing dependency declaration
system in cargo and rust.  The most common problem is that a crate `A` depends on
another crate `B` but some of the types from crate `B` are exposed through the API in
crate `A`.  This causes problems in practice if that dependency `B` is also used by the
user's code itself which often leaves users in less than ideal situations where either
their code refuses to compile because different versions of those libraries are requested
or where compiler messages are less than clear.

The introduction of an explicit distinction between public and private dependencies can
solve some of these issues and also let us lift some restrictions that should make some
code compile that previously was prevented from compiling by restrictions in cargo.

**Q: What is a public dependency?**<br>
A: a dependency is public if some of the types or trait of that dependency is itself
exported through the main crate.  The most common places where this happens is obviously
return values and function parameter but obviously the same applies to trait implementations
and many other things.  Because public can be tricky to determine for a user this RFC
proposes to extend the compiler infrastructure to detect the concept of "public dependency".
This will help the user understanding this concept and avoid making mistakes in
the `Cargo.toml`

Effectively the idea is that if your own library bumps a public dependency it means that
it's a breaking change of your *own* crate.

**Q: What is a private dependency?**<br>
A: On the other hand a private dependency is contained within your crate and effectively
invisible for users of your crate.  As a result private dependencies can be freely
duplicated.  This distinction will also make it possible to relax some restrictions that
currently exist in Cargo which sometimes prevent crates from compiling.

**Q: Can public become private later?**<br>
A: Public dependencies are public within a reachable subgraph but can become private if a
crate stops exposing a public dependency.  For instance it is very possible to have a
family of crates that all depend on a utility crate that provides common types which is
a public dependency for all of them.  However your own crate only becomes a user of this
utility crate through another dependency that itself does not expose any of the types
from that utility crate and as such the dependency is marked private.

**Q: Where is public / private defined?**<br>
Dependencies are private by default and are made public through a `public` flag in the
dependency in the `Cargo.toml` file.  This also means that crates created before the
implementation of this RFC will have all their dependencies private.

**Q: How is backwards compatibility handled?**<br>
A: It will continue to be permissible to "leak" dependencies and there are even some
use cases of this, however the compiler or cargo will emit warnings if private
dependencies become part of the public API.  Later it might even become invalid to
publish new crates without explicitly silencing these warnings or marking the
dependencies as public.

**Q: Can I export a type from a private dependency as my own?**<br>
For now it will not be strictly permissible to privately depend on a crate and export
a type from their as your own.  The reason for this is that at the moment it is not
possible to force this type to be distinct.  This means that users of the crate might
accidentally start depending on that type to be compatible if the user starts to depend
on the crate that actually implements that type.

# Detailed design
[design]: #detailed-design

There are a few areas that require to be changed for this RFC:

* The compiler needs to be extended to understand when crate dependencies are
  considered a public dependency
* The `Cargo.toml` manifest needs to be extended to support declaring public
  dependencies
* The cargo publish process needs to be changed to warn (or prevent) the publishing
  of crates that have undeclared public dependencies
* crates.io should show public dependencies more prominently than private ones.

## Compiler Changes

The main change to the compiler will be to accept a new parameter that cargo
supplies which is a list of public dependencies.  The compiler then emits
warnings if it encounters private dependencies leaking to the public API of a
crate.  `cargo publish` might change this warning into an error in its lint
step.

Additionally later on the warning can turn into a hard error in general.

In some situations it can be necessary to allow private dependencies to become
part of the public API.  In that case one can permit this with
`#[allow(external_private_dependency)]`.  This is particularly useful when
paired with `#[doc(hidden)]` and other already existing hacks.

This most likely will also be necessary for the more complex relationship of
`libcore` and `libstd` in Rust itself.

## Changes to `Cargo.toml`

The `Cargo.toml` file will be amended to support the new `public` parameter on
dependencies.  Old cargo versions will emit a warning when this key is encountered
but otherwise continue.  Since the default for a dependency to be private only
public ones will need to be tagged which should be the minority.

Example dependency:

```toml
[dependencies]
url = { version = "1.4.0", public = true }
```

## Changes to Cargo Publishing

When a new crate version is published Cargo will warn about types and traits that
the compiler determined to be public but did not come from a public dependency.  For
now it should be possible to publish anyways but in some period in the future it will
be necessary to explicitly mark all public dependencies as such or explicitly
mark them with `#[allow(external_private_dependency)]`.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

From the user's perspective the initial scope of the RFC will be quite transparent
but it will definitely show up for users as a question of what the new restrictions
mean.  In particular a common way to leak out types from APIs that most crates do
is error handling.  Quite frequently it happens that users wrap errors from other
libraries in their own types.  It might make sense to identify common cases of where
type leakage happens and provide hints in the lint about how to deal with it.

Cases that I anticipate that should be explained separately:

* type leakage through errors. This should be easy to spot for a lint because the
  wrapper type will implement `std::error::Error`.  The recommendation should most
  likely be to encourage containing the internal error.
* traits from other crates.  In particular serde and some other common crates will
  show up frequently and it might make sense to separately explain types and traits.
* type leakage through derive.  Users might not be aware they have a dependency to
  a type when they derive a trait (think `serde_derive`).  The lint might want to
  call this out separately.

The feature will be called `public_private_dependencies` and it comes with one
lint flag called `external_private_dependency`.  For all intents and purposes this
should be the extend of the new terms introduced in the beginning.  This RFC however
lays the groundwork for later providing aliasing so that a private dependencies could
be forcefully re-exported as own types.  As such it might make sense to already
consider what this will be referred to.

It is assumed that this feature will eventually become quite popular due to patterns
that already exist in the crate ecosystem but it's likely that it will evoke some
negative opinions initially.  As such it would be a good idea to make a run with
cargobomb/crater to see what the actual impact of the new linter warnings is and
how far we are off to making them errors.

crates.io should most likely be updated to render public and private dependencies
separately.

# Drawbacks
[drawbacks]: #drawbacks

I believe that there are no drawbacks if implemented well (this assumes good
linters and error messages).

# Alternatives
[alternatives]: #alternatives

For me the biggest alternative to this RFC would be a variation of it where type
and trait aliasing becomes immediately part of it.  This would meant that a crate
can have a private dependency and re-export it as its own type, hiding where it
came from originally.  This would most likely be easier to teach users and can get
rid of a few "cul-de-sac" situations where users can end up in and their only way
out is to introduce a public dependency for now.  The assumption is that if trait
and type aliasing is available the `external_public_dependency` would not need to
exist.

# Unresolved questions
[unresolved]: #unresolved-questions

There are a few open questions about how to best hook into the compiler and cargo
infrastructure:

* is passing in the last of public dependencies the correct way to get around it?
  If yes, what is the parameter supposed to be called.
* what is the impact of this change going to be. This most likely can be answered
  running cargobomb/crater.
* since changing public dependency pins/ranges requires a change in semver it might
  be worth exploring if cargo could prevent the user in pushing up new crate
  versions that violate that constraint.
