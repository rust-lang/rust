# Trait solving (new-style)

🚧 This chapter describes "new-style" trait solving. This is still in the
[process of being implemented][wg]; this chapter serves as a kind of
in-progress design document. If you would prefer to read about how the
current trait solver works, check out
[this other chapter](./traits/resolution.html). (By the way, if you
would like to help in hacking on the new solver, you will find
instructions for getting involved in the
[Traits Working Group tracking issue][wg].) 🚧

[wg]: https://github.com/rust-lang/rust/issues/48416

Trait solving is based around a few key ideas:

- [Lowering to logic](./traits/lowering-to-logic.html), which expresses
  Rust traits in terms of standard logical terms.
  - The [goals and clauses](./traits/goals-and-clauses.html) chapter
    describes the precise form of rules we use, and
    [lowering rules](./traits/lowering-rules.html) gives the complete set of
    lowering rules in a more reference-like form.
- [Canonical queries](./traits/canonical-queries.html), which allow us
  to solve trait problems (like "is `Foo` implemented for the type
  `Bar`?") once, and then apply that same result independently in many
  different inference contexts.
- [Lazy normalization](./traits/associated-types.html), which is the
  technique we use to accommodate associated types when figuring out
  whether types are equal.
- [Region constraints](./traits/regions.html), which are accumulated
  during trait solving but mostly ignored. This means that trait
  solving effectively ignores the precise regions involved, always –
  but we still remember the constraints on them so that those
  constraints can be checked by thet type checker.
  
Note: this is not a complete list of topics. See the sidebar for more.
