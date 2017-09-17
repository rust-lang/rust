- Feature Name: TBD
- Start Date: 2017-08-24
- RFC PR: https://github.com/rust-lang/rfcs/pull/2126
- Rust Issue: https://github.com/rust-lang/rust/issues/44660

# Summary
[summary]: #summary

This RFC seeks to clarify and streamline Rust's story around paths and visibility for modules and crates. That story will look as follows:

- Absolute paths should begin with a crate name, where the keyword `crate` refers to the current crate (other forms are linted, see below)
- `extern crate` is no longer necessary, and is linted (see below); dependencies are available at the root unless shadowed.
- The `crate` keyword also acts as a visibility modifier, equivalent to today's `pub(crate)`. Consequently, uses of bare `pub` on items that are not actually publicly exported are linted, suggesting `crate` visibility instead.
- A `foo.rs` and `foo/` subdirectory may coexist; `mod.rs` is no longer needed when placing submodules in a subdirectory.

**These changes do not require a new epoch**. The new features are purely additive. They can ship with **allow-by-default** lints, which can gradually be moved to warn-by-default and deny-by-default over time, as better tooling is developed and more code has actively made the switch.

*This RFC incorporates some text written by @withoutboats and @cramertj, who have both been involved in the long-running discussions on this topic.*

[new epoch]: https://github.com/rust-lang/rfcs/pull/2052

# Motivation
[motivation]: #motivation

A major theme of this year's [roadmap] is improving the learning curve and
ergonomics of the core language. That's based on overwhelming feedback that the
single biggest barrier to Rust adoption is its learning curve.

[roadmap]: https://github.com/rust-lang/rfcs/pull/1774

One part of Rust that has long been a source of friction for some is its
module system. There are two related perspectives for improvement here:
learnability and productivity:

- Modules are not a place that Rust was trying to innovate at 1.0, but they are
  nevertheless often reported as one of the major stumbling blocks to learning
  Rust. We should fix that.

- Even for seasoned Rustaceans, the module system has some deficiencies, as
  we’ll dig into below. Ideally, we can solve these problems while also making
  modules easier to learn.

## The core problems

This RFC does not attempt to *comprehensively* solve the problems that have been
raised in today's module system. The focus is instead high-impact problems with
noninvasive solutions.

### Defining versus bringing into scope

A persistent point of confusion is the relationship between *defining an item*
and *bringing an item into scope*. First, let's look at the rules as they exist
today:

- When you refer to items within definitions (e.g. a `fn` signature or body),
  those items must be **in scope** (unless you use a leading `::` or `super`).

- Defining an item "mounts" its name within the current crate's module
  hierarchy, making it available through absolute paths.

- All items defined within a module are also in scope throughout that
  module. This includes `use` statements, which actually *define* (i.e. mount)
  items within the current module.

- Additional names are brought into scope through things like function
  parameters or generics.

There's a beautiful uniformity and sparseness in these rules that makes them
appealing. And they turn out to be reasonably intuitive for items whose full
definition is given within the module (e.g. `struct` definitions).

The struggle tends to instead be with items like `extern crate` and `mod foo;`
which "bring in" other crates or files. This RFC focuses on the former, so let's
explore that in more detail.

When you write `extern crate futures` in your crate root, there are two consequences per
the above rules:

- The external crate `futures` is "mounted" at the root absolute path.
- The external crate `futures` is brought into scope for the top-level module.

When writing code at crate root, you're able to freely refer to `futures` to start
paths in *both* `use` statements *and* in references to items:

```rust
extern crate futures;

use futures::Future;

fn my_poll() -> futures::Poll { ... }
```

These consequences make it easy to build an incorrect mental model, in which
`extern crate` *globally* adds the external crate name as something you can
start *any* path with--made worse because it's half true. (This confusion is
undoubtedly influenced by the way that external package references work in many
other languages, where absolute paths *always* begin with a package reference.)
This wrong mental model works fine in the crate root, but breaks down as soon as
you try it in a submodule:

```rust
extern crate futures;

mod submodule {
    // this still works fine!
    use futures::Future;

    // but suddenly this doesn't...
    fn my_poll() -> futures::Poll { ... }
}
```

The fact that adding a `use futures;` statement to the submodule makes the `fn`
declaration work is almost worse: it reinforces the idea that external crates
define names in the root namespace, but that *sometimes* you need to write `use
futures` to refer to them... but not to refer to them in `use` declarations!
This is the point where some people get exasperated by the module system, which
seems to be enforcing some mysterious and pedantic distinctions. And this is
perhaps worst with `std`, in which there's an *implicit* `extern crate` in the
root module, so that `fn make_vec() -> std::vec::Vec<u8>` works fine in crate
root but requires `use std` elsewhere.

In other words, while there are simple and consistent *rules* defining the
module system, their *consequences* can feel inconsistent, counterintuitive and
mysterious.

It's tempting to say that we can fully address these problems by better
documentation and compiler diagnostics--and surely we should improve them! But
for folks trying out Rust, there's already plenty to learn, and there's a sense
that the module system is "getting in the way" early on, forcing you to stop and
try to understand its particular set of rules before you can get back to trying
to understand ownership and other aspects of Rust.

This RFC instead tweaks the handling of external crates and absolute paths, so
that when you apply the general rules of the module system, you get an outcome
that feels more consistent and intuitive, and requires less front-loading of
explanation. As we'll see below, in practice these changes will also improve
clarity and readability even for users with a full understanding of the rules.

(We'll revisit this example at the end of the Guide section to explain how the
RFC helps.)

### Nonlocal reasoning

There are at least two ways in which today's module system doesn't support local
reasoning. These affect newcomers and old hands alike.

- **Is a `use` path talking about this crate or an external one?** When reading
  `use` statements, to know the source of the import you need to have in your
  head a list of external crates and/or top-level modules for the current
  crate. It has long been idiomatic to visually group imports from the current
  crate separately from external imports. In general, this suggests a certain
  muddiness around the root namespace.

- **Is an item marked `pub` *actually* public?** It's a fairly common idiom
  today to have a private module that contains `pub` items used by its parent
  and siblings only. This idiom arises in part because of ergonomic concerns;
  writing `pub(super)` or `pub(crate)` on these internal items feels
  heavier. But the consequence is that, when reading code, visibility
  annotations tell you less than you might hope, and in general you have to walk
  up the module tree looking for re-exports to know exactly how public an item
  is.

### The `mod.rs` file

A final issue, though far less important, is the use of `mod.rs` files when
creating a directory containing submodules. There are several downsides:

- From a learnability perspective, the fact that the paths in the module system
  aren't *quite* in direct correspondence with the file system is another small
  speedbump, and in particular makes `mod foo;` declarations entail extra
  ceremony (since the parent module must be moved into a new directory). A
  simpler rule would be: the path to a module's file is the path to it within
  Rust code, with `.rs` appended.
- From an ergonomics perspective, one often ends up with many `mod.rs` files
  open, and thus must depend on editor smarts to easily navigate between
  them. Again, a minor but nontrivial papercut.
- When refactoring code to introduce submodules, having to use `mod.rs` means
  you often have to move existing files around. Another papercut.

The main *benefit* to `mod.rs` is that the code for a parent module and its
children live more closely together (not necessarily desirable!) and that it
provides a consistent story with `lib.rs`.

## Some evidence of learning struggles

In the survey data collected in both 2016 and 2017, learnability and ergonomics
issues were one of the major challenges for people using or considering
Rust. While there were other features that were raised more frequently than the
module system (lifetimes for example), ideally the module system, which isn't
*meant* to be novel, would not be a learnability problem at all!

Here are some select quotes (these are not the only responses that mention the module
system):

> Also the module system is confusing (not that I say is wrong, just confusing
> until you are experienced in it).

> a colleague of mine that started rust got really confused over the module
> system

> You had to import everything in the main module, but you also had to in
> submodules, but if it was only imported in a submodule it wouldn't work.

> I especially find the modules and crates design weird and verbose

> fix the module system

One user states that the reason they stopped using Rust was that the
"module system is really unintuitive." Similar data is present in the 2016 survey.

Experiences along similar lines can be found in Rust forums, StackOverflow, and
similar, some of which has been collected into [a gist][learning-modules].

[learning-modules]: https://gist.github.com/aturon/2f10f19f084f39330cfe2ee028b2ea0c

The problems presented above represent a boiled down subset of the problems
raised in this feedback.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## As we would teach it

The following sections sketch a plausible way of teaching the module system once
this RFC has been fully implemented.

### Using external dependencies

To add an external dependency, record it in the `[dependencies]` section of
`Cargo.toml`:

```toml
[dependencies]
serde = "1.0.0"
```

By default, crates have an automatic dependency on `std`, the standard library.

Once your dependency has been added, you can bring it or its exports into scope with
`use` declarations:

```rust
use std; // bring `std` itself into scope
use std::vec::Vec;

use serde::Serialize;
```

Note that these `use` declarations all begin with a crate name.

Once an item is in scope, you can reference it directly within definitions:

```rust
// Both of these work, because we brought `std` and `Vec` into scope:
fn make_vec() -> Vec<u8> { ... }
fn make_vec() -> std::vec::Vec<u8> { ... }

// Only the first of these work, because we didn't bring `serde` into scope:
impl Serialize for MyType { ... }
impl serde::Serialize for MyType { ... } // the name `serde` is not in scope here
```

You can also reference items from a crate without bringing them into scope by
writing a **fully qualified path**, designated by a leading `::`, as follows:

```rust
impl ::serde::Serialize for MyType { ... }
```

All `use` declarations are interpreted as fully qualified paths, making the
leading `::` optional for them.

> **Note: that means that you can write `use serde::Serialize` in *any* module
without trouble, as long as `serde` is an external dependency!**

### Adding a new file to your crate

Rust crates have a distinguished entry point (generally called `main.rs` or
`lib.rs`) which is used to determine the crate's structure. Other files and
directories within `src/` are *not* automatically included in the crate.
Instead, you explicitly declare *submodules* using `mod` declarations.

Let's see how this looks with an example. First, we might set up a directory
structure like the following:

```
src
├── cli
│   ├── parse.rs
│   └── usage.rs
├── cli.rs
├── main.rs
├── process
│   ├── read.rs
│   └── write.rs
└── process.rs
```

The intent is for the crate to have two top-level modules, `cli` and `process`,
each of which contain two submodules. To turn these files into submodules, we
use `mod` declarations as follows:

```rust
// src/main.rs
mod cli;
mod process;
```

```rust
// src/cli.rs
mod parse;
mod usage;
```

```rust
// src/process.rs
mod read;
mod write;
```

Note how these declarations follow the structure of the filesystem (except that
the entry point, `main.rs`, has its children modules as sibling files). By
default, `mod` declarations assume this kind of direct mapping to the
filesystem; they are used to tell Rust to incorporate those files, and to set
attributes on the resulting modules (as we'll see in a moment).

### Importing items from other parts of your crate

In Rust, all items defined in a module are *private* by default, which means
they can only be accessed by the module defining them (or any of its
submodules). If you want an item to have greater visibility, you can use a
*visibility modifier*. The two most important of these are:

- `crate`, which makes an item visible anywhere within the current crate, but
  not outside of it.
- `pub`, which makes an item public, i.e. visible everywhere.

For binary crates (which have no consumers), `crate` and `pub` are equivalent.

Going back to the earlier example, we might instead write:

```rust
// src/main.rs
pub mod cli;
pub mod process;
```

```rust
// src/cli.rs
pub mod parse;
pub mod usage;
```

```rust
// src/cli/usage.rs
pub fn print_usage() { ... }
```

```rust
// src/process.rs
pub mod read;
pub mod write;
```

To refer to an item within your own crate, you can use a fully qualified path
that starts with one of the following:

- `crate`, to start at the root of your crate, e.g. `crate::cli::usage::print_usage`
- `self`, to start at the current module
- `super`, to start at the current module's parent

So we could write in `main.rs`:

```rust
use crate::cli::usage;

fn main() {
    // ...
    usage::print_usage()
    // ...
}
```

In general, then, fully qualified paths always start with an initial location: an external
crate name, or `crate`/`self`/`super`.

## Guide-level thoughts when comparing to today's system

Let's revisit one of the motivating examples. Today, you might write:

```rust
extern crate futures;
fn my_poll() -> futures::Poll { ... }
```

and then be confused when the following doesn't work:

```rust
extern crate futures;
mod submodule {
    fn my_poll() -> futures::Poll { ... }
}
```

because you've been led to think that `extern crate` brings the name into scope
everywhere.

After this RFC, you would no longer write `extern crate futures`. You might try to write just:

```rust
fn my_poll() -> futures::Poll { ... }
```

but the compiler would produce an error, saying that there's no `futures` in
scope; maybe you meant the external dependency, which you can bring into scope
by writing `use futures;`? So you do that:

```rust
use futures;
fn my_poll() -> futures::Poll { ... }
```

and now, when you refactor, you're much more likely to understand that the `use`
should come along for the ride:

```rust
mod submodule {
    use futures;
    fn my_poll() -> futures::Poll { ... }
}
```

Together with the fact that you use `crate::` in `use` declarations, this
strongly reinforces the idea that:

- `use` brings items into scope, based on paths that start by identifying the crate
- an item needs to be in scope before you can refer to it

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

First, a bit of terminology: a *fully qualified path* is a path starting with
`::`, which *all* paths in `use` do implicitly.

The actual changes in this RFC are fairly small tweaks to the current module
system; most of the complexity comes from the migration plans.

The proposed migration plan is minimally disruptive; **it does not require an
epoch**.

## Basic changes

- You can write `mod bar;` statements even when not in a `mod.rs` or equivalent;
  in this case, the submodules must appear within a subdirectory with
  the same name as the current module. Thus, `foo.rs` can contain `mod bar;` if
  there is also a `foo/bar.rs`.
  - It is not permitted to have both `foo.rs` and `foo/mod.rs` at the same point
    in the file system.
  - The use of `mod.rs` continues to be allowed without any deprecation. It is
    expected that tooling like Clippy will push for at least style consistency
    within a project, and perhaps eventually across the ecosystem.

- We introduce `crate` as a new visibility specifier, shorthand for `pub(crate)`
  visibility.

- We introduce `crate` as a new path component which designates the root of the
  current crate.

- In a path fully qualified path `::foo`, resolution will first attempt to
  resolve to a top-level definition of `foo`, and otherwise fall back to
  available external crates.

- Cargo will provide a new `alias` key for aliasing dependencies, so that
  e.g. users who want to use the `rand` crate but call its library crate
  `random` instead can now write `rand = { version = "0.3", alias = "random" }`.

- We introduce several lints, which all start out allow-by-default but are
  expected to ratchet up over time:

  - A lint for fully qualified paths that do not begin with one of: an external
  crate name, `crate`, `super`, or `self`.

  - A lint for use of `extern crate`.

  - A lint against use of bare `pub` for items which are not reachable via some
  fully-`pub` path. That is, bare `pub` should truly mean *public*, and `crate`
  should be used for crate-level visibility.

## Resolving fully-qualified paths

The only way to refer to an external crate without using `extern crate` is
through a fully-qualified path.

When resolving a fully-qualified path that begins with a name (and not `crate`,
`super` or `self`, we go through a two-stage process:

- First, attempt to resolve the name as an item defined in the top-level module.
  - If successful, issue a deprecation warning, saying that the `crate` prefix
    should be used.
- Otherwise, attempt to resolve the name as an external crate, exactly as we do
  with `extern crate` today.

In particular, no change to the compilation model or interface between `rustc`
and Cargo/the ambient build system is needed.

This approach is designed for backwards compatibility, but it means that you
cannot have a top-level module and an external crate with the same
name. Allowing that would require all fully-qualified paths into the current
crate to start with `crate`, which can only be done on a future epoch. We can
and should consider making such a change eventually, but it is not required for
this RFC.

[epoch]: https://github.com/rust-lang/rfcs/pull/2052
[macros 2.0]: https://github.com/rust-lang/rfcs/blob/master/text/1561-macro-naming.md#importing-macros
[previous RFC]: https://github.com/rust-lang/rfcs/pull/2088

## Migration experience

We will provide a high-fidelity `rustfix` tool that makes changes to the a crate
such that the lints proposed in this RFC would not fire. In particular, the tool
will introduce `crate::` prefixes, downgrade from `pub` to `crate` where
appropriate, and remove `extern crate`. It must be sound (i.e. keep the meaning
of code intact and keep it compiling) but may not be complete (i.e. you may
still get some deprecation warnings after running it).

Such a tool should be working at with very high coverage before we consider
changing any of the lints to warn-by-default.

# Drawbacks
[drawbacks]: #drawbacks

The most important drawback is that this RFC pushes toward *ultimately* changing
most Rust code in existence. There is risk of this reintroducing a sense that
Rust is unstable, if not handled properly. However, that risk is mitigated by
several factors:

- The fact that existing forms continue to work indefinitely.
- The fact that we will provide migration tooling with high coverage.
- The fact that nudges toward new forms (in the forms of lints) are introduced
  gradually, and only after strong tooling exists.

Imports from within your crate become more verbose, since they require a leading
`crate`. However, this downside is considerably mitigated if [nesting in `use`]
is permitted.

[nesting in `use`]: https://github.com/rust-lang/rfcs/pull/2128

There is some concern that introducing and encouraging the use of `crate` as a
visibility will, counter to the goals of the RFC, lead to people *increasing*
the visibility of items rather than decreasing it (and hence increasing
inter-module coupling). This could happen if, for example, an item needs to be
exposed to a cousin module, where a Rust user might hesitate to make it `pub`
but feel that `crate` is sufficiently "safe" (when really a refactoring is
called for). While this is indeed a possibility, it's offset by some other
cultural and design factors: Rust's design strongly encourages narrow access
rights (privacy by default; immutability by default), and this orientation has a
strong cultural sway within the Rust community.

In previous discussions about deprecating `extern crate`, there were concerns
about the impact on non-Cargo tooling, and in overall explicitness. This RFC
fully addresses both concerns by leveraging the new, unambiguous nature of fully
qualified paths.

Moving crate renaming externally has implications for procedural macros with
dependencies: their clients must include those dependencies without renaming
them.

# Rationale and Alternatives
[alternatives]: #alternatives

The core rationale here should be clear given the detailed analysis in the
motivation. The crucial insight of the design is that, by making absolute paths
unambiguous about which crate they draw from, we can solve a number of
confusions and papercuts with the module system.

## Epoch-based migration story

We can avoid the need for fallback in resolution by leveraging epochs instead.
On the current epoch, we would make `crate::` paths available and start warning
about *not* using them for crate-internal paths, but we would not issue warnings
about `extern crate`. In the next epoch, we would change absolute path
interpretations, such that warning-free code on the previous epoch would
continue to compile and have the same meaning.

## Bike-sheddy choices

There are a few aspects of this proposal that could be colored a bit differently
without fundamental change.

- Rather than `crate::top_level_module`, we could consider `extern::serde` or
  something like it, which would eliminate the need for any fallback in name
  resolution. That would come with some significant downsides, though.
  - First, having paths typically start with a crate name, with `crate`
    referring to the current crate, provides a *very simple* and easy to
    understand model for paths---and its one that's pretty commonly used in other languages.
  - Second, one benefit of `crate` is that it helps reduce confusion about paths
    appearing in `use` versus references to names elsewhere. In particular, it
    serves as a reminder that `use` paths are absolute.

- Rather than using `crate` as a visibility specifier, we could use something
  like `local`. (If we used it purely as a visibility specifier, we could make
  it a contextual keyword). That might be preferable, since `local` is an
  adjective and is arguably more intuitive. This is an unresolved question.

- The lint checking for `pub` items that are not actually public could be
  extended to check for *all* visibility levels. The RFC stuck with just `pub`
  because the ergonomics of `crate` make it more feasible to go from `pub` to
  `crate`, which should always work. It seems less feasible to ask people to
  annotate definitions with e.g. `pub(super)`, though maybe this is a sign that
  the `pub(restricted)` syntax is too unergonomic or underused.

## The community discussion around modules

For the past several months, the Rust community has been investigating the
module system, its weaknesses, strengths, and areas of potential
improvement. The discussion is far too wide-ranging to summarize here, so I'll
just present links.

Two blog posts serve as milestones in the discussion, laying out a
part of the argument in favor of improving the module system:

* [The Rust module system is too confusing][too-confusing] by @withoutboats
* [Revisiting Rust's modules][revisiting] by @aturon

[too-confusing]: https://withoutboats.github.io/blog/rust/2017/01/04/the-rust-module-system-is-too-confusing.html
[revisiting]: https://aturon.github.io/blog/2017/07/26/revisiting-rusts-modules/

And in addition there's been extensive discussion on internals:

- [Revisiting Rust’s modules](https://internals.rust-lang.org/t/revisiting-rusts-modules/5628) - aturon, Jul 26
- [Revisiting Rust’s modules, part 2](https://internals.rust-lang.org/t/revisiting-rust-s-modules-part-2/5700?u=carols10cents) - aturon, Aug 2
- [Revisting Modules, take 3](https://internals.rust-lang.org/t/revisiting-modules-take-3/5715?u=carols10cents) - withoutboats, Aug 4
- [pre-RFC: inline mod](https://internals.rust-lang.org/t/pre-rfc-inline-mod/5716?u=carols10cents) - ahmedcharles, Aug 4
- [My Preferred Module System (a fusion of earlier proposals)](https://internals.rust-lang.org/t/my-preferred-module-system-a-fusion-of-earlier-proposals/5718?u=carols10cents) - phaylon, Aug 5
- [[Pre-RFC] Yet another take on modules](https://internals.rust-lang.org/t/pre-rfc-yet-another-take-on-modules/5717?u=carols10cents) - newpavlov, Aug 5
- [pre-RFC: from crate use item](https://internals.rust-lang.org/t/pre-rfc-from-crate-use-item/5719?u=carols10cents) - ahmedcharles, Aug 5
- [Decoupled Module Improvements](https://internals.rust-lang.org/t/decoupled-module-improvements/5724?u=carols10cents) - phaylon, Aug 6
- [Revisiting modules – `[other_crate]::path` syntax](https://internals.rust-lang.org/t/revisiting-modules-other-crate-path-syntax/5728) - le-jzr, Aug 7
- [Poll: Which other-crate-relative-path syntax do you prefer?](https://internals.rust-lang.org/t/poll-which-other-crate-relative-path-syntax-do-you-prefer/5744?u=carols10cents) - elahn, Aug 9

These discussions ultimately led to [two](https://github.com/rust-lang/rfcs/pull/2108) [failed](https://github.com/rust-lang/rfcs/pull/2121) RFCs.

These earlier RFCs were shooting for a more comprehensive set of improvements
around the module system, and in particular both involved eliminating the need
for `mod` declarations in common cases. However, there are enough concerns and
open questions about that direction that we chose to split those more ambitious
ideas off into a separate *experimental* RFC:

> We recognize that this is a major point of controversy and so will put aside trying to complete a full RFC on the topic at this time; however, we believe the idea has enough merit that it's worth an experimental implementation in the compiler that we can use to gather more data, e.g. around the impact on workflow. We would still like to do this before the impl period, so that we can do that exploration during the impl period. (To be clear: experimental RFCs are to approve landing unstable features that seem promising but where we need more experience; they require a standard RFC to be merged before they can be stabilized.)

# Unresolved questions
[unresolved]: #unresolved-questions

- How should we approach migration? Via a fallback, as proposed, or via epochs?
  It is probably best to make this determination with more experience,
  e.g. after we have a `rustfix` tool in hand.
