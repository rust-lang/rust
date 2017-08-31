- Feature Name: (fill me in with a unique ident, my_awesome_feature)
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This RFC proposes the addition of the support for alternative crates.io servers to be used
alongside the public crates.io server. This would allow users to publish crates to their
own private instance of crates.io, while still able to use the public instance of crates.io.

# Motivation
[motivation]: #motivation

Cargo currently has support for getting crates from a public server, which works well for open
source projects using Rust, however is problematic for closed source code. A workaround for
this is to use Git repositories to specify the packages, but that means that the helpful
versioning and discoverability that Cargo and crates.io provides is lost. We would like to
change this such that it is possible to have a local crates.io server which crates can be
pushed to, while still making use of the public crates.io server.

We would also like to support the use of crates.io mirrors. These differ from alternative
registries in that a mirror completely replicates the functionality and content of
crates.io. A mirror would be useful if we ever need a fallback for when crates.io
goes down, or in areas of the world where crates.io is blocked.

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

# Rationale and Alternatives
[alternatives]: #alternatives

- Why is this design the best in the space of possible designs?
- What other designs have been considered and what is the rationale for not choosing them?
- What is the impact of not doing this?

# Unresolved questions
[unresolved]: #unresolved-questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?
- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
- What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?
