- Start Date: 2014-05-22
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

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

# Drawbacks

It may be that `+` is ugly. Also, it messes with the precedence of `as`. (See Rust PR #14365 for the fallout of the latter.)

# Alternatives

The impact of not doing this is that the inconsistencies and ambiguities above remain.

# Unresolved questions

Where does the `'static` bound fit into all this?
