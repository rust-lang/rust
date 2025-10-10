# Walkthrough: a typical contribution

There are _a lot_ of ways to contribute to the Rust compiler, including fixing
bugs, improving performance, helping design features, providing feedback on
existing features, etc. This chapter does not claim to scratch the surface.
Instead, it walks through the design and implementation of a new feature. Not
all of the steps and processes described here are needed for every
contribution, and I will try to point those out as they arise.

In general, if you are interested in making a contribution and aren't sure
where to start, please feel free to ask!

## Overview

The feature I will discuss in this chapter is the `?` Kleene operator for
macros. Basically, we want to be able to write something like this:

```rust,ignore
macro_rules! foo {
    ($arg:ident $(, $optional_arg:ident)?) => {
        println!("{}", $arg);

        $(
            println!("{}", $optional_arg);
        )?
    }
}

fn main() {
    let x = 0;
    foo!(x); // ok! prints "0"
    foo!(x, x); // ok! prints "0 0"
}
```

So basically, the `$(pat)?` matcher in the macro means "this pattern can occur
0 or 1 times", similar to other regex syntaxes.

There were a number of steps to go from an idea to stable Rust feature. Here is
a quick list.  We will go through each of these in order below. As I mentioned
before, not all of these are needed for every type of contribution.

- **Idea discussion/Pre-RFC**  A Pre-RFC is an early draft or design discussion
  of a feature. This stage is intended to flesh out the design space a bit and
  get a grasp on the different merits and problems with an idea. It's a great
  way to get early feedback on your idea before presenting it to the wider
  audience. You can find the original discussion [here][prerfc].
- **RFC**  This is when you formally present your idea to the community for
  consideration. You can find the RFC [here][rfc].
- **Implementation** Implement your idea unstably in the compiler. You can
  find the original implementation [here][impl1].
- **Possibly iterate/refine** As the community gets experience with your
  feature on the nightly compiler and in `std`, there may be additional
  feedback about design choice that might be adjusted. This particular feature
  went [through][impl2] a [number][impl3] of [iterations][impl4].
- **Stabilization** When your feature has baked enough, a Rust team member may
  [propose to stabilize it][merge]. If there is consensus, this is done.
- **Relax** Your feature is now a stable Rust feature!

[prerfc]: https://internals.rust-lang.org/t/pre-rfc-at-most-one-repetition-macro-patterns/6557
[rfc]: https://github.com/rust-lang/rfcs/pull/2298
[impl1]: https://github.com/rust-lang/rust/pull/47752
[impl2]: https://github.com/rust-lang/rust/pull/49719
[impl3]: https://github.com/rust-lang/rust/pull/51336
[impl4]: https://github.com/rust-lang/rust/pull/51587
[merge]: https://github.com/rust-lang/rust/issues/48075#issuecomment-433177613

## Pre-RFC and RFC

> NOTE: In general, if you are not proposing a _new_ feature or substantial
> change to Rust or the ecosystem, you don't need to follow the RFC process.
> Instead, you can just jump to [implementation](#impl).
>
> You can find the official guidelines for when to open an RFC [here][rfcwhen].

[rfcwhen]: https://github.com/rust-lang/rfcs#when-you-need-to-follow-this-process

An RFC is a document that describes the feature or change you are proposing in
detail. Anyone can write an RFC; the process is the same for everyone,
including Rust team members.

To open an RFC, open a PR on the
[rust-lang/rfcs](https://github.com/rust-lang/rfcs) repo on GitHub. You can
find detailed instructions in the
[README](https://github.com/rust-lang/rfcs#what-the-process-is).

Before opening an RFC, you should do the research to "flesh out" your idea.
Hastily-proposed RFCs tend not to be accepted. You should generally have a good
description of the motivation, impact, disadvantages, and potential
interactions with other features.

If that sounds like a lot of work, it's because it is. But no fear! Even if
you're not a compiler hacker, you can get great feedback by doing a _pre-RFC_.
This is an _informal_ discussion of the idea. The best place to do this is
internals.rust-lang.org. Your post doesn't have to follow any particular
structure.  It doesn't even need to be a cohesive idea. Generally, you will get
tons of feedback that you can integrate back to produce a good RFC.

(Another pro-tip: try searching the RFCs repo and internals for prior related
ideas. A lot of times an idea has already been considered and was either
rejected or postponed to be tried again later. This can save you and everybody
else some time)

In the case of our example, a participant in the pre-RFC thread pointed out a
syntax ambiguity and a potential resolution. Also, the overall feedback seemed
positive. In this case, the discussion converged pretty quickly, but for some
ideas, a lot more discussion can happen (e.g. see [this RFC][nonascii] which
received a whopping 684 comments!). If that happens, don't be discouraged; it
means the community is interested in your idea, but it perhaps needs some
adjustments.

[nonascii]: https://github.com/rust-lang/rfcs/pull/2457

The RFC for our `?` macro feature did receive some discussion on the RFC thread
too.  As with most RFCs, there were a few questions that we couldn't answer by
discussion: we needed experience using the feature to decide. Such questions
are listed in the "Unresolved Questions" section of the RFC. Also, over the
course of the RFC discussion, you will probably want to update the RFC document
itself to reflect the course of the discussion (e.g. new alternatives or prior
work may be added or you may decide to change parts of the proposal itself).

In the end, when the discussion seems to reach a consensus and die down a bit,
a Rust team member may propose to move to "final comment period" (FCP) with one
of three possible dispositions. This means that they want the other members of
the appropriate teams to review and comment on the RFC. More discussion may
ensue, which may result in more changes or unresolved questions being added. At
some point, when everyone is satisfied, the RFC enters the FCP, which is the
last chance for people to bring up objections. When the FCP is over, the
disposition is adopted. Here are the three possible dispositions:

- _Merge_: accept the feature. Here is the proposal to merge for our [`?` macro
  feature][rfcmerge].
- _Close_: this feature in its current form is not a good fit for rust. Don't
  be discouraged if this happens to your RFC, and don't take it personally.
  This is not a reflection on you, but rather a community decision that rust
  will go a different direction.
- _Postpone_: there is interest in going this direction but not at the moment.
  This happens most often because the appropriate Rust team doesn't have the
  bandwidth to shepherd the feature through the process to stabilization. Often
  this is the case when the feature doesn't fit into the team's roadmap.
  Postponed ideas may be revisited later.

[rfcmerge]: https://github.com/rust-lang/rfcs/pull/2298#issuecomment-360582667

When an RFC is merged, the PR is merged into the RFCs repo. A new _tracking
issue_ is created in the [rust-lang/rust] repo to track progress on the feature
and discuss unresolved questions, implementation progress and blockers, etc.
Here is the tracking issue on for our [`?` macro feature][tracking].

[tracking]: https://github.com/rust-lang/rust/issues/48075

<a id="impl"></a>

## Implementation

To make a change to the compiler, open a PR against the [rust-lang/rust] repo.

[rust-lang/rust]: https://github.com/rust-lang/rust

Depending on the feature/change/bug fix/improvement, implementation may be
relatively-straightforward or it may be a major undertaking. You can always ask
for help or mentorship from more experienced compiler devs.  Also, you don't
have to be the one to implement your feature; but keep in mind that if you
don't, it might be a while before someone else does.

For the `?` macro feature, I needed to go understand the relevant parts of
macro expansion in the compiler. Personally, I find that [improving the
comments][comments] in the code is a helpful way of making sure I understand
it, but you don't have to do that if you don't want to.

[comments]: https://github.com/rust-lang/rust/pull/47732

I then [implemented][impl1] the original feature, as described in the RFC. When
a new feature is implemented, it goes behind a _feature gate_, which means that
you have to use `#![feature(my_feature_name)]` to use the feature. The feature
gate is removed when the feature is stabilized.

**Most bug fixes and improvements** don't require a feature gate. You can just
make your changes/improvements.

When you open a PR on the [rust-lang/rust], a bot will assign your PR to a
reviewer. If there is a particular Rust team member you are working with, you can
request that reviewer by leaving a comment on the thread with `r?
@reviewer-github-id` (e.g. `r? @eddyb`). If you don't know who to request,
don't request anyone; the bot will assign someone automatically based on which files you changed.

The reviewer may request changes before they approve your PR, they may mark the PR with label 
"S-waiting-on-author" after leaving comments, this means that the PR is blocked on you to make 
some requested changes. When you finished iterating on the changes, you can mark the PR as 
`S-waiting-on-review` again by leaving a comment with `@rustbot ready`, this will remove the 
`S-waiting-on-author` label and add the `S-waiting-on-review` label.

Feel free to ask questions or discuss things you don't understand or disagree with. However,
recognize that the PR won't be merged unless someone on the Rust team approves
it. If a reviewer leave a comment like `r=me after fixing ...`, that means they approve the PR and 
you can merge it with comment with `@bors r=reviewer-github-id`(e.g. `@bors r=eddyb`) to merge it 
after fixing trivial issues. Note that `r=someone` requires permission and bors could say 
something like "🔑 Insufficient privileges..." when commenting `r=someone`. In that case, 
you have to ask the reviewer to revisit your PR.

When your reviewer approves the PR, it will go into a queue for yet another bot
called `@bors`. `@bors` manages the CI build/merge queue. When your PR reaches
the head of the `@bors` queue, `@bors` will test out the merge by running all
tests against your PR on GitHub Actions. This takes a lot of time to
finish. If all tests pass, the PR is merged and becomes part of the next
nightly compiler!

There are a couple of things that may happen for some PRs during the review process

- If the change is substantial enough, the reviewer may request an FCP on
  the PR. This gives all members of the appropriate team a chance to review the
  changes.
- If the change may cause breakage, the reviewer may request a [crater] run.
  This compiles the compiler with your changes and then attempts to compile all
  crates on crates.io with your modified compiler. This is a great smoke test
  to check if you introduced a change to compiler behavior that affects a large
  portion of the ecosystem.
- If the diff of your PR is large or the reviewer is busy, your PR may have
  some merge conflicts with other PRs that happen to get merged first. You
  should fix these merge conflicts using the normal git procedures.

[crater]: ./tests/crater.html

If you are not doing a new feature or something like that (e.g. if you are
fixing a bug), then that's it! Thanks for your contribution :)

## Refining your implementation

As people get experience with your new feature on nightly, slight changes may
be proposed and unresolved questions may become resolved. Updates/changes go
through the same process for implementing any other changes, as described
above (i.e. submit a PR, go through review, wait for `@bors`, etc).

Some changes may be major enough to require an FCP and some review by Rust team
members.

For the `?` macro feature, we went through a few different iterations after the
original implementation: [1][impl2], [2][impl3], [3][impl4].

Along the way, we decided that `?` should not take a separator, which was
previously an unresolved question listed in the RFC. We also changed the
disambiguation strategy: we decided to remove the ability to use `?` as a
separator token for other repetition operators (e.g. `+` or `*`). However,
since this was a breaking change, we decided to do it over an edition boundary.
Thus, the new feature can be enabled only in edition 2018. These deviations
from the original RFC required [another
FCP](https://github.com/rust-lang/rust/issues/51934).

## Stabilization

Finally, after the feature had baked for a while on nightly, a language team member
[moved to stabilize it][stabilizefcp].

[stabilizefcp]: https://github.com/rust-lang/rust/issues/48075#issuecomment-433177613

A _stabilization report_ needs to be written that includes

- brief description of the behavior and any deviations from the RFC
- which edition(s) are affected and how
- links to a few tests to show the interesting aspects

The stabilization report for our feature is [here][stabrep].

[stabrep]: https://github.com/rust-lang/rust/issues/48075#issuecomment-433243048

After this, [a PR is made][stab] to remove the feature gate, enabling the feature by
default (on the 2018 edition). A note is added to the [Release notes][relnotes]
about the feature.

[stab]: https://github.com/rust-lang/rust/pull/56245

Steps to stabilize the feature can be found at [Stabilizing Features](./stabilization_guide.md).

[relnotes]: https://github.com/rust-lang/rust/blob/master/RELEASES.md
