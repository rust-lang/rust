# Trait solving (new-style)

🚧 This chapter describes "new-style" trait solving. This is still in the
[process of being implemented][wg]; this chapter serves as a kind of
in-progress design document. If you would prefer to read about how the
current trait solver works, check out
[this other chapter](./resolution.html). (By the way, if you
would like to help in hacking on the new solver, you will find
instructions for getting involved in the
[Traits Working Group tracking issue][wg].) 🚧

[wg]: https://github.com/rust-lang/rust/issues/48416

Trait solving is based around a few key ideas:

- [Lowering to logic](./lowering-to-logic.html), which expresses
  Rust traits in terms of standard logical terms.
  - The [goals and clauses](./goals-and-clauses.html) chapter
    describes the precise form of rules we use, and
    [lowering rules](./lowering-rules.html) gives the complete set of
    lowering rules in a more reference-like form.
  - [Lazy normalization](./associated-types.html), which is the
    technique we use to accommodate associated types when figuring out
    whether types are equal.
  - [Region constraints](./regions.html), which are accumulated
    during trait solving but mostly ignored. This means that trait
    solving effectively ignores the precise regions involved, always –
    but we still remember the constraints on them so that those
    constraints can be checked by the type checker.
- [Canonical queries](./canonical-queries.html), which allow us
  to solve trait problems (like "is `Foo` implemented for the type
  `Bar`?") once, and then apply that same result independently in many
  different inference contexts.
  
Note: this is not a complete list of topics. See the sidebar for more.

The design of the new-style trait solving currently happens in two places:
* The [chalk][chalk] repository is where we experiment with new ideas and
  designs for the trait system. It basically consists of a unit testing framework
  for the correctness and feasibility of the logical rules defining the new-style
  trait system. It also provides the [`chalk_engine`][chalk_engine] crate, which
  defines the new-style trait solver used both in the unit testing framework and
  in rustc.
* Once we are happy with the logical rules, we proceed to implementing them in
  rustc. This mainly happens in [`librustc_traits`][librustc_traits].

[chalk]: https://github.com/rust-lang-nursery/chalk
[chalk_engine]: https://github.com/rust-lang-nursery/chalk/tree/master/chalk-engine
[librustc_traits]: https://github.com/rust-lang/rust/tree/master/src/librustc_traits
