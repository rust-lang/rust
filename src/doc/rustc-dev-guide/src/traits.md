# Trait solving (new-style)

🚧 This chapter describes "new-style" trait solving. This is still in
the process of being implemented; this chapter serves as a kind of
in-progress design document. If you would prefer to read about how the
current trait solver works, check out
[this chapter](./trait-resolution.html).🚧

Trait solving is based around a few key ideas:

- [Canonicalization](./traits-canonicalization.html), which allows us to
  extract types that contain inference variables in them from their
  inference context, work with them, and then bring the results back
  into the original context.
- [Lowering to logic](./traits-lowering-to-logic.html), which expresses
  Rust traits in terms of standard logical terms.

*more to come*
