- Feature Name: TBD
- Start Date: 2017-08-24
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This RFC seeks to clarify and streamline Rust's story around paths and visibility for modules and crates. When the RFC is *fully* implemented (which will require a [new epoch]), that story will look as follows:

- Absolute paths *always* begin with a crate name, where the keyword `crate` refers to the current crate.
- `extern crate` is deprecated. Usage of external crates is always unambiguous, since it must somewhere involve an absolute path beginning with the crate's name.
- The `crate` keyword also acts as a visibility modifier, equivalent to today's `pub(crate)`. Consequently, uses of bare `pub` on items that are not actually publicly exported trigger a lint warning suggesting `crate` visibility instead.
- A `foo.rs` and `foo/` subdirectory may coexist; `mod.rs` is no longer needed when placing submodules in a subdirectory.

The vast majority of these changes can be done without a new epoch; only the "always" on the first bullet requires a new epoch.

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
  ceremony (since the parent module must be moved into a new directory).
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

There are two basic ways we could do migration. The first one, pioneered by [RFC
2088], treats external crates as a kind of "fallback" when interpeting fully
qualified paths (to retain compatibility with today's code). That solution is
detailed in the alternatives section.

[RFC 2088]: https://github.com/rust-lang/rfcs/pull/2088

This RFC, however, spells out a different alternative which avoids the need for
any kind of fallback in name resolution, or indeed any "implicit" linking of
crates at all. This is achieved via a two-stage approach: one within the current
[epoch], and one for the next.

[epoch]: https://github.com/rust-lang/rfcs/pull/2052

## Stage 1 (current epoch)

The core idea of stage 1 is to make all of the changes in this RFC *except* for
removing the need for `extern crate`. In particular, this allows people to
migrate paths to use `crate::` in preparation for the next stage.

In detail, at stage 1:

- You can write `mod bar;` statements even when not in a `mod.rs` or equivalent;
  in this case, the submodules must appear within a subdirectory with
  the same name as the current module. Thus, `foo.rs` can contain `mod bar;` if
  there is also a `foo/bar.rs`.
  - It is not permitted to have both `foo.rs` and `foo/mod.rs` at the same point
    in the file system.

- We introduce `crate` as a new visibility specifier, shorthand for `pub(crate)`
  visibility.

- We introduce a lint against use of bare `pub` for items which are not
  reachable via some fully-`pub` path. That is, bare `pub` should truly mean
  *public*, and `crate` should be used for crate-level visibility.

- We introduce `crate` as a new path component which designates the root of the
  current crate, and deprecate fully qualified paths that do not begin with one
  of: an external crate name, `crate`, `super`, or `self`.
  - However, at this stage, external crate names *must* be provided by
    *explicit* `extern crate` declarations at the root of the crate.
  - We deprecate any use of `extern crate` *not* at crate root.

- Cargo will provide a new `alias` key for dependencies, so that e.g. users who
  want to use the `rand` crate but call it `random` instead can now write `rand
  = { version = "0.3", alias = "random" }`.
  - `extern crate foo as bar` is deprecated.

However, at stage 1, we do *not* deprecate `extern crate` outright, which is still
required for mounting external crates.

Thus, at the end of stage 1, idiomatic code should look as follows:

```rust
extern crate serde; // we'll drop this in stage 2
use serde::Serialize;
use crate::top_level_module::SomeItem;
```

The idea is that, if you have warning-free code at this point, transitioning to
the next stage should just be a matter of removing `extern crate` declarations
(in some cases replacing them with `use` declarations).

## Stage 2 (next epoch)

In stage 2, we interpret fully-qualified paths strictly according to this RFC,
which means that it's *always* clear whether they are referring to the current
crate or an external crate. In detail:

- *All* fully-qualified paths *must* begin with one of: an external crate name,
  `crate`, `super`, or `self`.
  - If a path begins with a name that does *not* correspond to an external
    crate, the compiler will generate an error. If there is a top-level module
    with the same name, the compiler error will point this out and suggest
    adding `crate`.

- `extern crate foo;` is fully deprecated. Its semantics is now identical to `use foo;`.
  - There is one exception: `#[macro_use] extern crate foo` at the top-level
    will not generate a warning. This will ultimately be supplanted by [macros
    2.0].
  - Note that macro authors can still emit `extern crate` with an attribute to
    silence the warning.

- The compilation model for external crates is essentially the same as today:
  they are linked only if references through an absolute path (e.g. a `use`
  statement) or `extern crate` declaration. In either case, the fact that the
  reference is to an external crate is completely unambiguous, and use of
  `extern crate` is no longer needed.

Note that, unlike the [previous RFC] involving `extern crate`, we do not need to
change the interface between Cargo and `rustc` here at all, because there is no
need to "generate" bindings based on the external crates provided. Instead, as
today, extern crates are brought in *by demand* using absolute paths.

Thus, at the end of stage 2, idiomatic code should look as follows:

```rust
use serde::Serialize;
use crate::top_level_module::SomeItem;
```

A consequence is that you can discover all usage of external crates from a
project's source by grepping for `use` statements or absolute paths that do not
begin with `crate`/`super`/`self`. Put differently, usage of external crates is
still fully explicit in the source, even without consulting `Cargo.toml`.

[macros 2.0]: https://github.com/rust-lang/rfcs/blob/master/text/1561-macro-naming.md#importing-macros
[previous RFC]: https://github.com/rust-lang/rfcs/pull/2088

## Migration experience

The two stages are carefully designed to retain key properties of the module
system and compilation throughout, and to minimize the introduction of hard
errors across epochs. The only thing that you *must* do to opt in to the next
epoch is change absolute paths to use the new format (starts with crate name,
`crate`, `super`, or `self`).

We can and should provide a high-fidelity `rustfix` tool that performs at least
this change, and ideally lowers `pub` to `crate` when possible as well.

In the error specified for invalid fully qualified paths, in which the compiler
checks for a top-level module with the same name, the error should *also*
suggest that the code may be coming from an older epoch, and point to (1) the
`rustfix` tool to automate migration and (2) the ability to specify an older
epoch in `Cargo.toml`.

# Drawbacks
[drawbacks]: #drawbacks

The most clear drawback here is that a new epoch is required. The RFC strives to
make the transition as narrow as possible, both by introducing as few new errors
as possible, and by providing additional mitigations through migration tooling
and compiler diagnostics (as well as the epoch system itself, of course).

Another drawback is that imports from within your crate become more verbose,
since they require a leading `crate`. However, this downside is considerably
mitigated if [nesting in `use`] is permitted.

[nesting in `use`]: https://github.com/rust-lang/rfcs/issues/1400

In previous discussions about deprecating `extern crate`, there were concerns
about the impact on non-Cargo tooling, and in overall explicitness. This RFC
fully addresses both concerns by leveraging the new, unambiguous nature of fully
qualified paths.

# Rationale and Alternatives
[alternatives]: #alternatives

The core rationale here should be clear given the detailed analysis in the
motivation. The crucial insight of the design is that, by making absolute paths
unambiguous about which crate they draw from, we can solve a number of
confusions and papercuts with the module system.

## Bike-sheddy choices

There are a few aspects of this proposal that could be colored a bit differently
without fundamental change.

- Rather than `crate::top_level_module`, we could consider `extern::serde` or
  something like it. That would come with some significant downsides, though.
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

## The migration plan

This RFC lays out a two stage plan using epochs, which aims to avoid the need
for any kind of name resolution fallback or change of interface between `rustc`
and `cargo`. That's partly in response to feedback on [RFC 2088].

That said, we could instead adopt the [RFC 2088] approach:

> External crates can be passed to the rust compiler using the
> `--extern CRATE_NAME=PATH` flag.
> For example, `cargo build`-ing a crate `my_crate` with a dependency on `rand`
> results in a call to rustc that looks something like
> `rustc --crate-name mycrate src/main.rs --extern rand=/path/to/librand.rlib ...`.
>
> When an external crate is specified this way, it will be automatically
> available to any module in the current crate through `use` statements or
> absolute paths (e.g. `::rand::random()`). It will _not_ be automatically
> imported at root level as happens with current `extern crate`.
> None of this behavior will occur when including a library using the `-l`
> or `-L` flags.
>
> We will continue to support the current `extern crate` syntax for backwards
> compatibility. `extern crate foo;` will behave just like it does currently.
> Writing `extern crate foo;` will not affect the availability of `foo` in
> `use` and absolute paths as specified by this RFC.
>
> Additionally, items such as modules, types, or functions that conflict with
> the names of implicitly imported crates will result in a warning and will
> require the external crate to be brought in manually using `extern crate`.
> Note that this is different from the current behavior of the
> implicitly-imported `std` module.
> Currently, creating a root-level item named `std` results in a name conflict
> error. For consistency with other crates, this error will be removed.
> Creating a root-level item named `std` will prevent `std` from being included,
> and will trigger a warning.
>
> When compiling, an external crate is only linked if it is used
> (through either `extern crate`, `use`, or absolute paths).
> This prevents unused crates from being linked, which is helpful in a number of
> scenarios:
> - Some crates have both `lib` and `bin` targets and want to avoid linking both
> `bin` and `lib` dependencies.
> - `no_std` crates need a way to avoid accidentally linking `std`-using crates.
> - Other crates have a large number of possible dependencies (such as
> [the current Rust Playground](https://users.rust-lang.org/t/the-official-rust-playground-now-has-the-top-100-crates-available/11817)),
> and want to avoid linking all of them.
>
> In order to prevent linking of unused crates,
> after macro expansion has occurred, the compiler will resolve
> `use`, `extern crate`, and absolute paths looking for a reference to external
> crates or items within them. Crates which are unreferenced in these paths
> will not be linked.

The upside of this approach is that *no epoch is required*. That said, an epoch
may still be desirable: the mental model of this RFC strongly suggests that you
should be able to have an external crate and a top-level module with the same
name, and doing that requires an epochal change. Still, if we took the [RFC
2088] approach, we could get the vast majority of the benefits of this RFC, and
make the epochal change much later, when we have confidence that very little
code in active development would be impacted.

Regardless, the actual changes to code are the same, and we'd want to provide a
`rustfix` tool to make them painless.

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

- Bikeshedding on `crate` as visibility attribute
- Precise migration story (two options are laid out above)
- Should `mod.rs` be deprecated?
