- Feature Name: N/A
- Start Date: 2016-06-07
- RFC PR: https://github.com/rust-lang/rfcs/pull/1643
- Rust Issue: N/A

# Summary
[summary]: #summary

Incorporate a strike team dedicated to preparing rules and guidelines
for writing unsafe code in Rust (commonly referred to as Rust's
"memory model"), in cooperation with the lang team. The discussion
will generally proceed in phases, starting with establishing
high-level principles and gradually getting down to the nitty gritty
details (though some back and forth is expected). The strike team will
produce various intermediate documents that will be submitted as
normal RFCs.

# Motivation
[motivation]: #motivation

Rust's safe type system offers very strong aliasing information that
promises to be a rich source of compiler optimization.  For example,
in safe code, the compiler can infer that if a function takes two
`&mut T` parameters, those two parameters must reference disjoint
areas of memory (this allows optimizations similar to C99's `restrict`
keyword, except that it is both automatic and fully enforced). The
compiler also knows that given a shared reference type `&T`, the
referent is immutable, except for data contained in an `UnsafeCell`.

Unfortunately, there is a fly in the ointment. Unsafe code can easily
be made to violate these sorts of rules. For example, using unsafe
code, it is trivial to create two `&mut` references that both refer to
the same memory (and which are simultaneously usable).  In that case,
if the unsafe code were to (say) return those two points to safe code,
that would undermine Rust's safety guarantees -- hence it's clear that
this code would be "incorrect".

But things become more subtle when we just consider what happens
*within* the abstraction. For example, is unsafe code allowed to use
two overlapping `&mut` references internally, without returning it to
the wild? Is it all right to overlap with `*mut`? And so forth.

It is the contention of this RFC that a complete guidelines for unsafe
code are far too big a topic to be fruitfully addressed in a single
RFC. Therefore, this RFC proposes the formation of a dedicated
**strike team** (that is, a temporary, single-purpose team) that will
work on hammering out the details over time. Precise membership of
this team is not part of this RFC, but will be determined by the lang
team as well as the strike team itself.

The unsafe guidelines work will proceed in rough stages, described
below. An initial goal is to produce a **high-level summary detailing
the general approach of the guidelines.** Ideally, this summary should
be sufficient to help guide unsafe authors in best practices that are
most likely to be forwards compatible. Further work will then expand
on the model to produce a more **detailed set of rules**, which may in
turn require revisiting the high-level summary if contradictions are
uncovered.

This new "unsafe code" strike team is intended to work in
collaboration with the existing lang team. Ultimately, whatever rules
are crafted must be adopted with the **general consensus of both the
strike team and the lang team**. It is expected that lang team members
will be more involved in the early discussions that govern the overall
direction and less involved in the fine details.

#### History and recent discussions

The history of optimizing C can be instructive. All code in C is
effectively unsafe, and so in order to perform optimizations,
compilers have come to lean heavily on the notion of "undefined
behavior" as well as various ad-hoc rules about what programs ought
not to do (see e.g. [these][cl1] [three][cl2] [posts][cl3] entitled
"What Every C Programmer Should Know About Undefined Behavior", by
Chris Lattner). This can cause some very surprising behavior (see e.g.
["What Every Compiler Author Should Know About Programmers"][cap] or
[this blog post by John Regehr][jr], which is quite humorous).  Note that
Rust has a big advantage over C here, in that only the authors of
unsafe code should need to worry about these rules.

[cl1]: http://blog.llvm.org/2011/05/what-every-c-programmer-should-know.html
[cl2]: http://blog.llvm.org/2011/05/what-every-c-programmer-should-know_14.html
[cl3]: http://blog.llvm.org/2011/05/what-every-c-programmer-should-know_21.html
[cap]: http://www.complang.tuwien.ac.at/kps2015/proceedings/KPS_2015_submission_29.pdf
[jr]: http://blog.regehr.org/archives/761

In terms of Rust itself, there has been a large amount of discussion
over the years. Here is a (non-comprehensive) set of relevant links,
with a strong bias towards recent discussion:

- [RFC Issue #1447](https://github.com/rust-lang/rfcs/issues/1447) provides
  a general set of links as well as some discussion.
- [RFC #1578](https://github.com/rust-lang/rfcs/pull/1578) is an initial
  proposal for a Rust memory model by ubsan.
- The
  [Tootsie Pop](http://smallcultfollowing.com/babysteps/blog/2016/05/27/the-tootsie-pop-model-for-unsafe-code/)
  blog post by nmatsakis proposed an alternative approach, building on
  [background about unsafe abstractions](http://smallcultfollowing.com/babysteps/blog/2016/05/23/unsafe-abstractions/)
  described in an earlir post. There is also a lot of valuable
  discussion in
  [the corresponding internals thread](http://smallcultfollowing.com/babysteps/blog/2016/05/23/unsafe-abstractions/).

#### Other factors

Another factor that must be considered is the interaction with weak
memory models. Most of the links above focus purely on sequential
code: Rust has more-or-less adopted the C++ memory model for governing
interactions across threads. But there may well be subtle cases that
arise we delve deeper. For more on the C++ memory model, see
[Hans Boehm's excellent webpage](http://www.hboehm.info/c++mm/).

# Detailed design
[design]: #detailed-design

## Scope

Here are some of the issues that should be resolved as part of these
unsafe code guidelines. The following list is not intended as
comprehensive (suggestions for additions welcome):

- Legal aliasing rules and patterns of memory accesses
  - e.g., which of the patterns listed in [rust-lang/rust#19733](https://github.com/rust-lang/rust/issues/19733)
    are legal?
  - can unsafe code create (but not use) overlapping `&mut`?
  - under what conditions is it legal to dereference a `*mut T`?
  - when can an `&mut T` legally alias an `*mut T`?
- Struct layout guarantees
- Interactions around zero-sized types
  - e.g., what pointer values can legally be considered a `Box<ZST>`?
- Allocator dependencies

One specific area that we can hopefully "outsource" is detailed rules
regarding the interaction of different threads. Rust exposes atomics
that roughly correspond to C++11 atomics, and the intention is that we
can layer our rules for sequential execution atop those rules for
parallel execution.

## Termination conditions

The unsafe code guidelines team is intended as a temporary strike team
with the goal of producing the documents described below. Once the RFC
for those documents have been approved, responsibility for maintaining
the documents falls to the lang team.

## Time frame

Working out a a set of rules for unsafe code is a detailed process and
is expected to take months (or longer, depending on the level of
detail we ultimately aim for). However, the intention is to publish
preliminary documents as RFCs as we go, so hopefully we can be
providing ever more specific guidance for unsafe code authors.

Note that even once an initial set of guidelines is adopted, problems
or inconsistencies may be found. If that happens, the guidelines will
be adjusted as needed to correct the problem, naturally with an eye
towards backwards compatibility. In other words, the unsafe
guidelines, like the rules for Rust language itself, should be
considered a "living document".

As a note of caution, experience from other languages such as Java or
C++ suggests that the work on memory models can take years. Moreover,
even once a memory model is adopted, it can be unclear whether
[common compiler optimizations are actually permitted](http://www.di.ens.fr/~zappa/readings/c11comp.pdf)
under the model. The hope is that by focusing on sequential and
Rust-specific issues we can sidestep some of these quandries.

## Intermediate documents

Because hammering out the finer points of the memory model is expected
to possibly take some time, it is important to produce intermediate
agreements. This section describes some of the documents that may be
useful. These also serve as a rough guideline to the overall "phases"
of discussion that are expected, though in practice discussion will
likely go back and forth:

- **Key examples and optimizations**: highlighting code examples that
  ought to work, or optimizations we should be able to do, as well as
  some that will not work, or those whose outcome is in doubt.
- **High-level design**: describe the rules at a high-level. This
  would likely be the document that unsafe code authors would read to
  know if their code is correct in the majority of scenarios. Think of
  this as the "user's guide".
- **Detailed rules**: More comprehensive rules. Think of this as the
  "reference manual".
  
Note that both the "high-level design" and "detailed rules", once
considered complete, will be submitted as RFCs and undergo the usual
final comment period.

### Key examples and optimizations

Probably a good first step is to agree on some key examples and
overall principles. Examples would fall into several categories:

- Unsafe code that we feel **must** be considered **legal** by any model
- Unsafe code that we feel **must** be considered **illegal** by any model
- Unsafe code that we feel **may or may not** be considered legal
- Optimizations that we **must** be able to perform
- Optimizations that we **should not** expect to be able to perform
- Optimizations that it would be nice to have, but which may be sacrificed
  if needed

Having such guiding examples naturally helps to steer the effort, but
it also helps to provide guidance for unsafe code authors in the
meantime. These examples illustrate patterns that one can adopt with
reasonable confidence.

Deciding about these examples should also help in enumerating the
guiding principles we would like to adhere to. The design of a memory
model ultimately requires balancing several competing factors and it
may be useful to state our expectations up front on how these will be
weighed:

- **Optimization.** The stricter the rules, the more we can optimize.
  - on the other hand, rules that are overly strict may prevent people
    from writing unsafe code that they would like to write, ultimately
    leading to slower exeution.
- **Comprehensibility.** It is important to strive for rules that end
  users can readily understand. If learning the rules requires diving
  into academic papers or using Coq, it's a non-starter.
- **Effect on existing code.** No matter what model we adopt, existing
  unsafe code may or may not comply. If we then proceed to optimize,
  this could cause running code to stop working. While
  [RFC 1122](https://github.com/rust-lang/rfcs/blob/master/text/1122-language-semver.md)
  explicitly specified that the rules for unsafe code may change, we
  will have to decide where to draw the line in terms of how much to
  weight backwards compatibility.

It is expected that the lang team will be **highly involved** in this discussion.

It is also expected that we will gather examples in the following ways:

- survey existing unsafe code;
- solicit suggestions of patterns from the Rust-using public:
  - scenarios where they would like an official judgement; 
  - interesting questions involving the standard library.

### High-level design

The next document to produce is to settle on a high-level
design. There have already been several approaches floated. This phase
should build on the examples from before, in that proposals can be
weighed against their effect on the examples and optimizations.

There will likely also be some feedback between this phase and the
previosu: as new proposals are considered, that may generate new
examples that were not relevant previously.

Note that even once a high-level design is adopted, it will be
considered "tentative" and "unstable" until the detailed rules have
been worked out to a reasonable level of confidence.

Once a high-level design is adopted, it may also be used by the
compiler team to inform which optimizations are legal or illegal.
However, if changes are later made, the compiler will naturally have
to be adjusted to match.

It is expected that the lang team will be **highly involved** in this discussion.

### Detailed rules

Once we've settled on a high-level path -- and, no doubt, while in the
process of doing so as well -- we can begin to enumerate more detailed
rules. It is also expected that working out the rules may uncover
contradictions or other problems that require revisiting the
high-level design.

### Lints and other checkers

Ideally, the team will also consider whether automated checking for
conformance is possible. It is not a responsibility of this strike
team to produce such automated checking, but automated checking is
naturally a big plus!

## Repository

In general, the memory model discussion will be centered on a specific
repository (perhaps
<https://github.com/nikomatsakis/rust-memory-model>, but perhaps moved
to the rust-lang organization). This allows for multi-faced
discussion: for example, we can open issues on particular questions,
as well as storing the various proposals and litmus tests in their own
directories. We'll work out and document the procedures and
conventions here as we go.

# Drawbacks
[drawbacks]: #drawbacks

The main drawback is that this discussion will require time and energy
which could be spent elsewhere. The justification for spending time on
developing the memory model instead is that it is crucial to enable
the compiler to perform aggressive optimizations. Until now, we've
limited ourselves by and large to conservative optimizations (though
we do supply some LLVM aliasing hints that can be affected by unsafe
code). As the transition to MIR comes to fruition, it is clear that we
will be in a place to perform more aggressive optimization, and hence
the need for rules and guidelines is becoming more acute. We can
continue to adopt a conservative course, but this risks growing an
ever larger body of code dependent on the compiler not performing
aggressive optimization, which may close those doors forever.

# Alternatives
[alternatives]: #alternatives

- Adopt a memory model in one fell swoop:
  - considered too complicated
- Defer adopting a memory model for longer:
  - considered too risky

# Unresolved questions
[unresolved]: #unresolved-questions

None.
