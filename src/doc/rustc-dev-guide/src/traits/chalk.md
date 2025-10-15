# Chalk-based trait solving

[Chalk][chalk] is an experimental trait solver for Rust that is
(as of <!-- date-check --> May 2022) under development by the [Types team].
Its goal is to enable a lot of trait system features and bug fixes
that are hard to implement (e.g. GATs or specialization). If you would like to
help in hacking on the new solver, drop by on the rust-lang Zulip in the [`#t-types`]
channel and say hello!

[Types team]: https://github.com/rust-lang/types-team
[`#t-types`]: https://rust-lang.zulipchat.com/#narrow/stream/144729-t-types

The new-style trait solver is based on the work done in [chalk]. Chalk
recasts Rust's trait system explicitly in terms of logic programming. It does
this by "lowering" Rust code into a kind of logic program we can then execute
queries against.

The key observation here is that the Rust trait system is basically a
kind of logic, and it can be mapped onto standard logical inference
rules. We can then look for solutions to those inference rules in a
very similar fashion to how e.g. a [Prolog] solver works. It turns out
that we can't *quite* use Prolog rules (also called Horn clauses) but
rather need a somewhat more expressive variant.

[Prolog]: https://en.wikipedia.org/wiki/Prolog

You can read more about chalk itself in the
[Chalk book](https://rust-lang.github.io/chalk/book/) section.

## Ongoing work
The design of the new-style trait solving happens in two places:

**chalk**. The [chalk] repository is where we experiment with new ideas
and designs for the trait system.

**rustc**. Once we are happy with the logical rules, we proceed to
implementing them in rustc. We map our struct, trait, and impl declarations
into logical inference rules in the lowering module in rustc.

[chalk]: https://github.com/rust-lang/chalk
[rustc_traits]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_traits
