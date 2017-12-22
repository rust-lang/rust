- Feature Name: libsyntax2
- Start Date: 2017-12-30
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)


>I think the lack of reusability comes in object-oriented languages,
>not functional languages. Because the problem with object-oriented
>languages is they’ve got all this implicit environment that they
>carry around with them. You wanted a banana but what you got was a
>gorilla holding the banana and the entire jungle.
>
>If you have referentially transparent code, if you have pure
>functions — all the data comes in its input arguments and everything
>goes out and leave no state behind — it’s incredibly reusable.
>
> **Joe Armstrong**

# Summary
[summary]: #summary

The long-term plan is to rewrite libsyntax parser and syntax tree data
structure to create a software component independent of the rest of
rustc compiler and suitable for the needs of IDEs and code
editors. This RFCs is the first step of this plan, whose goal is to
find out if this is possible at least in theory. If it is possible,
the next steps would be a prototype implementation as a crates.io
crate and a separate RFC for integrating the prototype with rustc,
other tools, and eventual libsyntax removal.

Note that this RFC does not propose to stabilize any API for working
with rust syntax: the semver version of the hypothetical library would
be `0.1.0`.


# Motivation
[motivation]: #motivation

"Reusable software component" part is addressed first "IDE ready part"
second.


In theory, parsing can be a pure function, which takes a `&str` as an
input, and produces a `ParseTree` as an output.

This is great for reusability: for example, you can compile this
function to WASM and use it for fast client-side validation of syntax
on the rust playground, or you can develop tools like `rustfmt` on
stable Rust outside of rustc repository, or you can embed the parser
into your favorite IDE or code editor.

This is also great for correctness: with such simple interface, it's
possible to write property-based tests to thoroughly compare two
different implementations of the parser. It's also straightforward to
create a comprehensive test suite, because all the inputs and outputs
are trivially serializable to human-readable text.

Another benefit is performance: with this signature, you can cache a
parse tree for each file, with trivial strategy for cache invalidation
(invalidate an entry when the underling file changes). On top of such
a cache it is possible to build a smart code indexer which maintains
the set of symbols in the project, watches files for changes and
automatically reindexes only changed files.

Unfortunately, the current libsyntax is far from this ideal. For
example, even the lexer makes use of the `FileMap` which is
essentially a global state of the compiler which represents all know
files. As a data point, it turned out to be easier to move `rustfmt`
inside of main `rustc` repository than to move libsyntax outside!










# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Explain the proposal as if it was already included in the language and you were teaching it to another Rust programmer. That generally means:

- Introducing new named concepts.
- Explaining the feature largely in terms of examples.
- Explaining how Rust programmers should *think* about the feature, and how it should impact the way they use Rust. It should explain the impact as concretely as possible.
- If applicable, provide sample error messages, deprecation warnings, or migration guidance.
- If applicable, describe the differences between teaching this to existing Rust programmers and new Rust programmers.

For implementation-oriented RFCs (e.g. for compiler internals), this section should focus on how compiler contributors should think about the change, and give examples of its concrete impact. For policy RFCs, this section should provide an example-driven introduction to the policy, and explain its impact in concrete terms.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This is the technical portion of the RFC. Explain the design in sufficient detail that:

- Its interaction with other features is clear.
- It is reasonably clear how the feature would be implemented.
- Corner cases are dissected by example.

The section should return to the examples given in the previous section, and explain more fully how the detailed proposal makes those examples work.

# Drawbacks
[drawbacks]: #drawbacks

Why should we *not* do this?

# Rationale and alternatives
[alternatives]: #alternatives

- Why is this design the best in the space of possible designs?
- What other designs have been considered and what is the rationale for not choosing them?
- What is the impact of not doing this?

# Unresolved questions
[unresolved]: #unresolved-questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?
- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
- What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?
