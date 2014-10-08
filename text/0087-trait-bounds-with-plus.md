- Start Date: 2014-05-22
- RFC PR: [rust-lang/rfcs#87](https://github.com/rust-lang/rfcs/pull/87)
- Rust Issue: [rust-lang/rust#12778](https://github.com/rust-lang/rust/issues/12778)

# Summary

Bounds on trait objects should be separated with `+`.

# Motivation

With DST there is an ambiguity between the following two forms:

    trait X {
        fn f(foo: b);
    }


and

    trait X {
        fn f(Trait: Share);
    }

See Rust issue #12778 for details.

Also, since kinds are now just built-in traits, it makes sense to treat a bounded trait object as just a combination of traits. This could be extended in the future to allow objects consisting of arbitrary trait combinations.

# Detailed design

Instead of `:` in trait bounds for first-class traits (e.g. `&Trait:Share + Send`), we use `+` (e.g. `&Trait + Share + Send`).

`+` will not be permitted in `as` without parentheses. This will be done via a special *restriction* in the type grammar: the special `TYPE` production following `as` will be the same as the regular `TYPE` production, with the exception that it does not accept `+` as a binary operator.

# Drawbacks

* It may be that `+` is ugly.

* Adding a restriction complicates the type grammar more than I would prefer, but the community backlash against the previous proposal was overwhelming.

# Alternatives

The impact of not doing this is that the inconsistencies and ambiguities above remain.

# Unresolved questions

Where does the `'static` bound fit into all this?
