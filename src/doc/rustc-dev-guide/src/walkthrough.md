# Walkthrough: a typical contribution

There are _a lot_ of ways to contribute to the rust compiler, including fixing
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

```rust
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

There were a number of steps to go from an idea to stable rust feature. Here is
a quick list.  We will go through each of these in order below. As I mentioned
before, not all of these are needed for every type of contribution.

- **Idea discussion/Pre-RFC**  A Pre-RFC is an early draft or design discussion
  of a feature. This stage is intended to flesh out the design space a bit and
  get a grasp on the different merits and problems with an idea. It's a great
  way to get early feedback on your idea before presenting it the wider
  audience. You can find the original discussion [here][prerfc].
- **RFC**  This is when you formally present your idea to the community for
  consideration. You can find the RFC [here][rfc].
- **Implementation** Implement your idea unstabley in the compiler. You can
  find the original implementation [here][impl1].
- **Possibly iterate/refine** As the community gets experience with your
  feature on the nightly compiler and in `libstd`, there may be additional
  feedback about design choice that might be adjusted. This particular feature
  went [through][impl2] a [number][impl3] of [iterations][impl4].
- **Stabilization** When your feature has baked enough, a rust team member may
  [propose to stabilize it][merge]. If there is consensus, this is done.
- **Relax** Your feature is now a stable rust feature!

[prerfc]: https://internals.rust-lang.org/t/pre-rfc-at-most-one-repetition-macro-patterns/6557
[rfc]: https://github.com/rust-lang/rfcs/pull/2298
[impl1]: https://github.com/rust-lang/rust/pull/47752
[impl2]: https://github.com/rust-lang/rust/pull/49719
[impl3]: https://github.com/rust-lang/rust/pull/51336
[impl4]: https://github.com/rust-lang/rust/pull/51587
[merge]: https://github.com/rust-lang/rust/issues/48075#issuecomment-433177613

## Pre-RFC and RFC

> NOTE: In general, if you are not proposing a _new_ feature or substantial
> change to rust or the ecosystem, you don't need to follow the RFC process.
> Instead, you can just jump to [implementation](#impl).
>
> You can find the official guidelines for when to open an RFC [here][rfcwhen].

[rfcwhen]: https://github.com/rust-lang/rfcs#when-you-need-to-follow-this-process

An RFC is a document that describes the feature or change you are proposing in
detail. Anyone can write an RFC; the process is the same for everyone,
including rust team members.

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
a rust team member may propose [to merge the RFC][rfcmerge]. This means that
they want the other members of the appropriate teams to review and comment on
the RFC.  More changes may be proposed. At some point, when everyone is
satisfied, the RFC enters the "final comment period" (FCP), which is the last
chance for people to bring up objections. When the FCP is over, the RFC is
"merged" (or accepted).

[rfcmerge]: https://github.com/rust-lang/rfcs/pull/2298#issuecomment-360582667

Some other possible outcomes might be for a team member to propose to

- _Close_: this feature in its current form is not a good fit for rust. Don't
  be discouraged if this happens to your RFC, and don't take it personally.
  This is not a reflection on you, but rather a community decision that rust
  will go a different direction.
- _Postpone_: there is interest in going this direction but not at the moment.
  This happens most often because the appropriate rust team doesn't have the
  bandwidth to shepherd the feature through the process to stabilization. Often
  this is the case when the feature doesn't fit into the team's roadmap.
  Postponed ideas may be revisited later.

<a name="impl"></a>

## Implementation

TODO
