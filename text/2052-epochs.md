- Feature Name: N/A
- Start Date: 2017-06-26
- RFC PR: https://github.com/rust-lang/rfcs/pull/2052
- Rust Issue: https://github.com/rust-lang/rust/issues/44581

# Summary
[summary]: #summary

Rust's ecosystem, tooling, documentation, and compiler are constantly improving. To make it easier to follow development, and to provide a clear, coherent "rallying point" for this work, this RFC proposes that we declare a *epoch* every two or three years. Epochs are designated by the year in which they occur, and represent a release in which several elements come together:

- A significant, coherent set of new features and APIs have been stabilized since the previous epoch.
- Error messages and other important aspects of the user experience around these features are fully polished.
- Tooling (IDEs, rustfmt, Clippy, etc) has been updated to work properly with these new features.
- There is a guide to the new features, explaining why they're important and how they should influence the way you write Rust code.
- The book has been updated to cover the new features.
  - Note that this is already [required](https://github.com/rust-lang/rfcs/pull/1636) prior to stabilization, but in general these additions are put in an appendix; updating the book itself requires *significant* work, because new features can change the book in deep and cross-cutting ways. We don't block stabilization on that.
- The standard library and other core ecosystem crates have been updated to use the new features as appropriate.
- A new edition of the Rust Cookbook has been prepared, providing an updated set of guidance for which crates to use for various tasks.

Sometimes a feature we want to make available in a new epoch would require backwards-incompatible changes, like introducing a new keyword. In that case, the feature is only available by explicitly opting in to the new epoch. Existing code continues to compile, and crates can freely mix dependencies using different epochs.

# Motivation
[motivation]: #motivation

## The status quo

Today, Rust evolution happens steadily through a combination of several mechanisms:

- **The nightly/stable release channel split**. Features that are still under
  development are usable *only* on the nightly channel, preventing *de facto*
  lock-in and thus leaving us free to iterate in ways that involve code breakage
  before "stabilizing" the feature.

- **The rapid (six week) release process**. Frequent releases on the stable
  channel allow features to stabilize as they become ready, rather than as part
  of a massive push toward an infrequent "feature-based" release. Consequently,
  Rust evolves in steady, small increments.

- **Deprecation**. Compiler support for deprecating language features and
  library APIs makes it possible to nudge people toward newer idioms without
  breaking existing code.

All told, the tools work together quite nicely to allow Rust to change and grow
over time, while keeping old code working (with only occasional, very minor
adjustments to account for things like changes to type inference.)

## What's missing

So, what's the problem?

There are a few desires that the current process doesn't have a good story for:

- **Lack of clear "chapters" in the evolutionary story**. A downside to rapid
  releases is that, while the constant small changes eventually add up to large
  shifts in idioms, there's not an agreed upon line of demarcation between these
  major shifts. Nor is there a clear point at which tooling, books, and other
  artifacts are all fully updated and in sync around a given set of
  features. This is not a huge problem for those following Rust development
  carefully (e.g., readers of this RFC!), but many users and potential users
  don't. Providing greater clarity and coherence around the "chapters" of Rust
  evolution will make it easier to provide an overall narrative arc, and to
  refer easily to large sets of changes.

- **Lack of community rallying points**. The six week release process tends to
  make each individual release a somewhat ho hum affair. On the one hand, that's
  the whole point--we want to avoid marathon marches toward huge, feature-based
  releases, and instead ship things in increments as they become ready. But in
  doing so, we lose an opportunity to, every so often, come together as an
  entire community and produce a "major release" that is polished, coherent, and
  meaningful in a way that each six week increment is not. The [roadmap process]
  does provide some of this flavor, but it's hard to beat the power of working
  together toward a point-in-time release. The challenge is doing so *without*
  losing the benefits of our incremental working style.

- **Changes that may require some breakage in corner cases**. The simplest
  example is adding new keywords: the current implementation of `catch` uses the
  syntax `do catch` because `catch` is not a keyword, and cannot be added even
  as a contextual keyword without potential breakage. There are plenty of
  examples of "superficial" breakage like this that do not fit well into the
  current evolution mechanisms.

[roadmap process]: https://github.com/rust-lang/rfcs/pull/1728

At the same time, the commitment to stability and rapid releases has been an
incredible boon for Rust, and we don't want to give up those existing mechanisms
or their benefits.

This RFC proposes *epochs* as a mechanism we can layer on top of our existing
release process, keeping its guarantees while addressing its gaps.

# Detailed design
[design]: #detailed-design

## The basic idea

To make it easier to follow Rust's evolution, and to provide a clear, coherent
"rallying point" for the community, the project declares a *epoch* every
two or three years. Epochs are designated by the year in which they occur,
and represent a release in which several elements come together:

- A significant, coherent set of new features and APIs have been stabilized since the previous epoch.
- Error messages and other important aspects of the user experience around these features are fully polished.
- Tooling (IDEs, rustfmt, Clippy, etc) has been updated to work properly with these new features.
- There is a guide to the new features, explaining why they're important and how they should influence the way you write Rust code.
- The book has been updated to cover the new features.
  - Note that this is already [required](https://github.com/rust-lang/rfcs/pull/1636) prior to stabilization, but in general these additions are put in an appendix; updating the book itself requires *significant* work, because new features can change the book in deep and cross-cutting ways. We don't block stabilization on that.
- The standard library and other core ecosystem crates have been updated to use the new features as appropriate.
- A new edition of the Rust Cookbook has been prepared, providing an updated set of guidance for which crates to use for various tasks.

The precise list of elements going into an epoch is expected to evolve over
time, as the Rust project and ecosystem grow.

Sometimes a feature we want to make available in a new epoch would require
backwards-incompatible changes, like introducing a new keyword. In that case,
the feature is only available by explicitly opting in to the new
epoch. Each **crate** can declare an epoch in its `Cargo.toml` like
`epoch = "2019"`; otherwise it is assumed to have epoch 2015,
coinciding with Rust 1.0. Thus, new epochs are *opt in*, and the
dependencies of a crate may use older or newer epochs than the crate
itself.

To be crystal clear: Rust compilers must support *all* extant epochs, and
a crate dependency graph may involve several different epochs
simultaneously. Thus, **epochs do not split the ecosystem nor do they break
existing code**.

Furthermore:

- As with today, each new version of the compiler may gain stabilizations and deprecations.
- When opting in to a new epoch, existing deprecations *may* turn into hard
  errors, and the compiler may take advantage of that fact to repurpose existing
  usage, e.g. by introducing a new keyword. **This is the only kind of *breaking* change a
  epoch opt-in can make.**

Thus, code that compiles without warnings on the previous epoch (under the latest
compiler release) will compile without errors on the next epoch (modulo the
[usual caveats] about type inference changes and so on).

[usual caveats]: https://github.com/rust-lang/rfcs/blob/master/text/1122-language-semver.md

Alternatively, you can continue working with the previous epoch on new
compiler releases indefinitely, but your code may not have access to new
features that require new keywords and the like. New features that *are*
backwards compatible, however, will be available on older epochs.

## Epoch timing, stabilizations, and the roadmap process

As mentioned above, we want to retain our rapid release model, in which new
features and other improvements are shipped on the stable release channel as
soon as they are ready. So, to be clear, **we do not hold features back until
the next epoch**.

Rather, epochs, as their name suggests, represent a point of *global
coherence*, where documentation, tooling, the compiler, and core libraries are
all fully aligned on a new set of (already stabilized!) features and other
changes. This alignment can happen incrementally, but an epoch signals that
it *has* happened.

At the same time, epochs serve as a rallying point for making sure this
alignment work gets done in a timely fashion--and helping set scope as
needed. To make this work, we use the roadmap process:

- As today, each year has a [roadmap setting out that year's vision]. Some
  years---like 2017---the roadmap is mostly about laying down major new
  groundwork. Some years, however, they roadmap explicitly proposes to produce a
  new epoch during the year.

- Epoch years are focused primarily on *stabilization*, *polish*, and
  *coherence*, rather than brand new ideas. We are trying to put together and
  ship a coherent product, complete with documentation and a well-aligned
  ecosystem. These goals will provide a rallying point for the whole community,
  to put our best foot forward as we publish a significant new version of the
  project.

[roadmap laying out that year's vision]: https://github.com/rust-lang/rfcs/pull/1728

In short, epochs are striking a delicate balance: they're not a cutoff for
stabilization, which continues every six weeks, but they still provide a strong
impetus for coming together as a community and putting together a polished product.

### The preview period

There's an important tension around stabilization and epochs:

- We want to enable new features, including those that require an epoch
  opt-in, to be available on the stable channel as they become ready.

  - That means that we must enable some form of the opt in before the epoch
    is fully ready to ship.

- We want to retain our promise that code compiling on stable will continue to
  do so with new versions of the compiler, with minimum hassle.

  - That means that, once *any* form of the opt in is shipped, it cannot introduce *new* hard errors.

Thus, at some point within an epoch year, we will enable the opt-in on the
stable release channel, which must include *all* of the hard errors that will be
introduced in the next epoch, but not yet all of the stabilizations (or
other artifacts that go into the full epoch release). This is the *preview
period* for the epoch, which ends when a release is produced that
synchronizes all of the elements that go into an epoch and the epoch is
formally announced.

## A broad policy on epoch changes

There are numerous reasons to limit the scope of changes for new epochs, among them:

- **Limiting churn**. Even if you aren't *forced* to update your code, even if there are automated tools to do so, churn is still a pain for existing users. It also invalidates, or at least makes harder to use, existing content on the internet, like StackOverflow answers and blog posts. And finally, it plays against the important and hard work we've done to make Rust stable in both reality and perception. In short, while epochs avoid *ecosystem* splits and make churn opt-in, they do not eliminate *all* drawbacks.

- **Limiting technical debt**. The compiler retains compatibility for old epochs, and thus must have distinct "modes" for dealing with them. We need to strongly limit the amount and complexity of code needed for these modes, or the compiler will become very difficult to maintain.

- **Limiting deep conceptual changes**. Just as we want to keep the compiler maintainable, so too do we want to keep the conceptual model sustainable. That is, if we make truly radical changes in a new epoch, it will be very difficult for people to reason about code involving different epochs, or to remember the precise differences.

These lead to some hard and soft constraints.

### Hard constraints

**TL;DR: Warning-free code on epoch N must compile on epoch N+1 and have the
same behavior.**

There are only two things a new epoch can do that a normal release cannot:

- Change an existing deprecation into a hard error.
  - This option is only available when the deprecation is expected to hit a relatively small percentage of code.
- Change an existing deprecation to *deny* by default, and leverage the corresponding lint setting to produce error messages *as if* the feature were removed entirely.

The second option is to be preferred whenever possible. Note that warning-free code in one epoch might produce warnings in the next epoch, but it should still compile successfully.

The Rust compiler supports multiple epochs, but **must only support a single version of "core Rust"**. We identify "core Rust" as being, roughly, MIR and the core trait system; this specification will be made more precise over time.  The implication is that the "epoch modes" boil down to keeping around multiple desugarings into this core Rust, which greatly limits the complexity and technical debt involved. Similar, core Rust encompasses the core *conceptual* model of the language, and this constraint guarantees that, even when working with multiple epochs, those core concepts remain fixed.

### Soft constraints

**TL;DR: *Most* code *with* warnings on epoch N should, after running `rustfix`, compile on epoch N+1 and have the same behavior.**

The core epoch design avoids an ecosystem split, which is very important. But it's *also* important that upgrading your own code to a new epoch is minimally disruptive. The basic principle is that **changes that cannot be automated must be required only in a small minority of crates, and even there not require extensive work**. This principle applies not just to epochs, but also to cases where we'd like to make a widespread deprecation.

Note that a `rustfix` tool will never be perfect, because of conditional compilation and code generation. So it's important that, in the cases it inevitably fails, the manual fixes are not too onerous.

In addition, migrations that affect a large percentage of code must be "small tweaks" (e.g. clarifying syntax), and as above, must keep the old form intact (though they can enact a deny-by-default lint on it).

These are "soft constraints" because they use terms like "small minority" and "small tweaks", which are open for interpretation. More broadly, the more disruption involved, the higher the bar for the change.

### Positive examples: What epoch opt-ins can do

Given those principles, let's look in more detail at a few examples of the kinds of
changes epoch opt-ins enable. **These are just examples---this RFC doesn't
entail any commitment to these language changes**.

#### Example: new keywords

We've taken as a running example introducing new keywords, which sometimes
cannot be done backwards compatibly (because a contextual keyword isn't
possible). Let's see how this works out for the case of `catch`, assuming that
we're currently in epoch 2015.

- First, we deprecate uses of `catch` as identifiers, preparing it to become a new keyword.
- We may, as today, implement the new `catch` feature using a temporary syntax
  for nightly (like `do catch`).
- When the epoch opt-in for `2019` is released, opting into it makes `catch` into a
  keyword, regardless of whether the `catch` feature has been implemented. This
  means that opting in may require some adjustment to your code.
- The `catch` syntax can be hooked into an implementation usable on nightly within the `2019` epoch.
- When we're confident in the `catch` feature on nightly, we can stabilize it
  *onto the stable channel for users opting into `2019`*. It cannot be stabilized onto the `2015` epoch,
  since it requires a new keyword.
- `catch` is now a part of Rust, but may not be *fully* integrated into e.g. the book, IDEs, etc.
- At some point, epoch `2019` is fully shipped, and `catch` is now fully
  incorporated into tooling, documentation, and core libraries.

To make this even more concrete, let's imagine the following (aligned with the diagram above):

| Rust version | Latest available epoch | Status of `catch` in `2015` | Status of `catch` in latest epoch
| ------------ | ---------------------- | -- | -- |
| 1.15 | 2015 | Valid identifier | Valid identifier
| 1.21 | 2015 | Valid identifier; deprecated | Valid identifier; deprecated
| 1.23 | 2019 (preview period) | Valid identifier; deprecated | Keyword, unimplemented
| 1.25 | 2019 (preview period) | Valid identifier; deprecated | Keyword, implemented
| 1.27 | 2019 (final) | Valid identifier; deprecated | Keyword, implemented

Now, suppose you have the following code:

```
Cargo.toml:

epoch = "2015"
```

```rust
// main.rs:

fn main() {
    let catch = "gotcha";
    println!("{}", catch);
}
```

- This code will compile **as-is** on *all* Rust versions. On versions 1.21 and
above, it will yield a warning, saying that `catch` is deprecated as an
identifier.

- On version 1.23, if you change `Cargo.toml` to use `2019`, the
  code will fail to compile due to `catch` being a keyword.

- However, if you leave it at `2015`, you can upgrade to Rust 1.27 **and
  use libraries that opt in to the `2019` epoch** with no problem.

#### Example: repurposing corner cases

A similar story plays out for more complex modifications that repurpose existing
usages. For example, some suggested module system improvements deduce the module
hierarchy from the filesystem. But there is a corner case today of providing
both a `lib.rs` and a `bin.rs` directly at the top level, which doesn't play
well with the new feature.

Using epochs, we can deprecate such usage (in favor of the `bin` directory),
then make it an error during the preview period. The module system change could then
be made available (and ultimately stabilized) within the preview period, before
fully shipping on the next epoch.

#### Example: repurposing syntax

A more radical example: changing the syntax for trait objects and `impl
Trait`. In particular, we have
sometimes [discussed](https://github.com/rust-lang/rfcs/pull/1603):

- Using `dyn Trait` for trait objects (e.g. `Box<dyn Iterator<Item = u32>>`)
- Repurposing "bare `Trait` to use instead of `impl Trait`, so you can write `fn
  foo() -> Iterator<Item = u32>` instead of `fn foo -> impl Iterator<Item =
  u32>`

Suppose we wanted to carry out such a change. We could do it over multiple steps:

- First, introduce and stabilize `dyn Trait`.
- Deprecate bare `Trait` syntax in favor of `dyn Trait`.
- In an epoch preview period, make it an error to use bare `Trait` syntax.
- Ship the new epoch, and wait until bare `Trait` syntax is obscure.
- Re-introduce bare `Trait` syntax, stabilize it, and deprecate `impl Trait` in
  favor of it.

Of course, this RFC isn't suggesting that such a course of action is a *good*
one, just that it is *possible* to do without breakage. The policy around such
changes is left as an open question.

#### Example: type inference changes

There are a number of details about type inference that seem suboptimal:

- Currently multi-parameter traits like `AsRef<T>` will infer the value of one
  parameter on the basis of the other. We would at least like an opt-out, but
  employing it for `AsRef` is backwards-incompatible.
- Coercions don’t always trigger when we wish they would, but altering the rules
  may cause other programs to stop compiling.
- In trait selection, where-clauses take precedence over impls; changing this is backwards-incompatible.

We may or may not be able to change these details on the existing epoch. With
enough effort, we could probably deprecate cases where type inference rules
might change and request explicit type annotations, and then—in the new
epoch—tweak those rules.

### Negative examples: What epoch opt-ins can't do

There are also changes that epochs don't help with, due to the constraints
we impose. These limitations are extremely important for keeping the compiler
maintainable, the language understandable, and the ecosystem compatible.

#### Example: changes to coherence rules

Trait coherence rules, like the "orphan" rule, provide a kind of protocol about
which crates can provide which `impl`s. It's not possible to change protocol
incompatibly, because existing code will assume the current protocol and provide
impls accordingly, and there's no way to work around that fact via deprecation.

More generally, this means that epochs can only be used to make changes to the
language that are applicable *crate-locally*; they cannot impose new
requirements or semantics on external crates, since we want to retain
compatibility with the existing ecosystem.

#### Example: `Error` trait downcasting

See [rust-lang/rust#35943](https://github.com/mozilla/rust/issues/35943). Due to
a silly oversight, you can’t currently downcast the “cause” of an error to
introspect what it is. We can’t make the trait have stricter requirements; it
would break existing impls. And there's no way to do so only in a newer epoch,
because we must be compatible with the older one, meaning that we cannot rely on
downcasting.

This is essentially another example of a non-crate-local change.

More generally, breaking changes to the standard library are not possible.

## The full mechanics

We'll wrap up with the full details of the mechanisms at play.

- `rustc` will take a new flag, `--epoch`, which can specify the epoch to
  use. This flag will default to epoch 2015.
  - This flag should not affect the behavior of the core trait system or passes at the MIR level.
- `Cargo.toml` can include an `epoch` value, which is used to pass to `rustc`.
  - If left off, it will assume epoch 2015.
- `cargo new` will produce a `Cargo.toml` with the latest `epoch` value
  (including an epoch currently in its preview period).

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

First and foremost, if we accept this RFC, we should publicize the plan widely,
including on the main Rust blog, in a style simlar to [previous posts] about our
release policy. This will require extremely careful messaging, to make clear
that epochs are *not* about breaking Rust code, but instead *primarily*
about putting together a globally coherent, polished product on a regular basis,
while providing some opt-in ways to allow for evolution not possible today.

In addition, the book should talk about the basics from a user perspective,
including:

- The fact that, if you do nothing, your code should continue to compile (with
  minimum hassle) when upgrading the compiler.
- If you resolve deprecations as they occur, moving to a new epoch should also
  require minimum hassle.
- Best practices about upgrading epochs (TBD).

[previous posts]: https://blog.rust-lang.org/2014/10/30/Stability.html

# Drawbacks
[drawbacks]: #drawbacks

There are several drawbacks to this proposal:

- Most importantly, it risks muddying our story about stability, which we've
  worked very hard to message clearly.

  - To mitigate this, we need to put front and center that, **if you do nothing,
  updating to a new `rustc` should not be a hassle**, and **staying on an old
  epoch doesn't cut you off from the ecosystem**.

- It adds a degree of complication to an evolution story that is already
  somewhat complex (with release channels and rapid releases).

  - On the other hand, epoch releases provide greater clarity about major
    steps in Rust evolution, for those who are not following development
    closely.

- New epochs can invalidate existing blog posts and documentation, a problem we
  suffered a lot around the 1.0 release

  - However, this situation already obtains in the sense of changing idioms; a
    blog post using `try!` these days already feels like it's using "old
    Rust". Notably, though, the code still compiles on current Rust.

  - A saving grace is that, with epochs, it's more likely that a post will
    mention what epoch is being used, for context. Moreover, with sufficient
    work on error messages, it seems plausible to detect that code was intended
    for an earlier epochs and explain the situation.

These downsides are most problematic in cases that involve "breakage" if they
were done without opt in. They indicate that, even if we do adopt epochs, we
should use them judiciously.

# Alternatives
[alternatives]: #alternatives

## Within the basic epoch structure

There was a significant amount of discussion on the RFC thread about using "2.0"
rather than "2019". It's difficult to concisely summarize this discussion, but
in a nutshell, some feel that 2.0 (with a guarantee of backwards compatibility)
is more honest and easier to understand, while others worry that it will be
misconstrued no matter how much we caveat it, and that we cannot risk Rust being
perceived as unstable or risky.

  - The "epoch" terminology and current framing arose from this discussion,
    as a way of clarifying what we intend -- i.e., that the concept is
    *primarily* about putting together a coherent package -- and as a heads up
    that the model is different from that of other languages.

Sticking with the basic idea of epochs, there are a couple alternative setups
that avoid "preview" epochs:

- Rather than locking in a set of deprecations up front, we could provide
  "stable channel feature gates", allowing users to opt in to features of the
  next epoch in a fine-grained way, which may introduce new errors.  When
  the new epoch is released, one would then upgrade to it and remove all of
  the gates.

  - The main downside is lack of clarity about what the current "stable Rust"
    is; each combination of gates gives you a slightly different language. While
    this fine-grained variation is acceptable for nightly, since it's meant for
    experimentation, it cuts against some of the overall goals of this proposal
    to introduce such fragmentation on the stable channel. There's risk that
    people would use a mixture of gates in perpetuity, essentially picking their
    preferred dialect of the language.

  - It's feasible to introduce such a fine-grained scheme later on, if it proves
    necessary. Given the risks involved, it seems best to start with a
    coarse-grained flag at the outset.

- We could stabilize features using undesirable syntax at first, making way for
  better syntax only when the new epoch is released, then deprecate the "bad"
  syntax in favor of the "good" syntax.

  - For `catch`, this would look like:
    - Stabilize `do catch`.
    - Deprecate `catch` as an identifier.
    - Ship new epoch, which makes `catch` a keyword.
    - Stabilize `catch` as a syntax for the `catch` feature, and deprecate `do catch` in favor of it.
  - This approach involves significantly more churn than the one proposed in the RFC.

- Finally, we could just wait to stabilize features like `catch` until the
  moment the epoch is released.

  - This approach seems likely to introduce all the downsides of "feature-based"
    releases, making the epoch release extremely high stakes, and preventing
    usage of "ready to go" feature on the stable channel until the epoch is
    shipped.

## Alternatives to epochs

The larger alternatives include, of course, not trying to solve the problems
laid out in the motivation, and instead finding creative alternatives.

- For cases like `catch` that require a new keyword, it's not clear how to do
this without ending up with suboptimal syntax.

The other main alternative is to issue major releases in the semver sense: Rust
2.0. This strategy could potentially be coupled with a `rustfix`, depending on
what kinds of changes we want to allow. Downsides:

- Lack of clarity around ecosystem compatibility. If we allow both 1.0 and 2.0
  crates to interoperate, we arrive at something like this RFC. If we don't, we
  risk splitting the ecosystem, which is extremely dangerous.

- Likely significant blowback based on abandoning stability as a core principle
  of Rust. Even if we provide a perfect `rustfix`, the message is significantly muddied.

- Much greater temptation to make sweeping changes, and continuous litigation
  over what those changes should be.

# Unresolved questions
[unresolved]: #unresolved-questions

- What impact is there, if any, on breakage permitted today for bug fixing or
  soundness holes? In many cases these are more disruptive than introducing a
  new keyword.

- Is "epoch" the right key in Cargo.toml? Would it be more clear to just say `rust = "2019"`?

- Will we ever consider dropping support for very old epochs? Given the
  constraints in this RFC, it seems unlikely to ever be worth it.

- Should `rustc` default to the latest epoch instead?

- How do we handle macros, particularly procedural macros, that may mix source
  from multiple epochs?
