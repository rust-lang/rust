- Feature Name: deprecate_anonymous_parameters
- Start Date: 2016-07-19
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Currently Rust allows anonymous parameters in trait methods:

```Rust
trait T {
    fn foo(i32);

    fn bar_with_default_impl(String, String) {

    }
}
```

This RFC proposes to deprecate this syntax. This RFC intentionally does not
propose to remove this syntax.

# Motivation
[motivation]: #motivation

Anonymous parameters are a [historic accident]. They do not pose any
significant problems, but lead to a number of annoyances.

1. Surprising pattern syntax in traits

    ```Rust
    trait T {
        fn foo(x: i32);        // Ok
        fn bar(&x: &i32);      // Ok
        fn baz(&&x: &&i32);    // Ok
        fn quux(&&&x: &&&i32); // Syntax error
    }
    ```

    That is, patterns more complex than `_, foo, &foo, &&foo, mut foo` are
    forbidden.

2. Inconsistency between default implementations in traits and implementations
   in impl blocks

    ```Rust
    trait T {
        fn foo((x, y): (usize, usize)) { // Syntax error
        }
    }

    impl T for S {
        fn foo((x, y): (usize, usize)) { // Ok
        }
    }
    ```
3. Inconsistency between method declarations in traits and in extern blocks

    ```Rust
    trait T {
        fn foo(i32);  // Ok
    }

    extern "C" {
        fn foo(i32); // Syntax error
    }
    ```

4. Slightly more complicated syntax analysis for LL style parsers. The parser
   must guess if it currently parses a pattern or a type.

5. Small complications for source code analyzers (e.g. [IntelliJ Rust]) and
   potential alternative implementations.

6. Potential future parsing ambiguities with named and default parameters
   syntax.


None of these issues is significant, but they exist.

[historic accident]: https://github.com/rust-lang/rust/pull/29406#issuecomment-151859611
[IntelliJ Rust]: https://github.com/intellij-rust/intellij-rust/commit/1bb65c47341a04aecef5fa6817e8b2b56bfc9abb#diff-66f3ba596f0ecf74a2942b3223789ab5R41


# Detailed design
[design]: #detailed-design

Removing anonymous parameters from the language is formally a breaking change.
The breakage can be trivially and automatically fixed by adding `_:` (suggested by @nagisa):

```Rust
trait T {
    fn foo(_: i32);

    fn bar_with_default_impl(_: String, _: String) {

    }
}
```

However this is also a major breaking change from the practical point of view.
Parameter names are rarely omitted, but it happens. For example,
`std::fmt::Display` is currently defined as follows:

```Rust
trait Display {
    fn fmt(&self, &mut Formatter) -> Result;
}
```

So the proposal is just to deprecate this syntax in the hope that the removal
would become feasible and practical in the future. The hypothetical future may
include:

* Rust 2.0 release.
* A tool to automatically fix deprecation warnings.
* Storing crates on crates.io in "elaborated" syntax independent format.

Enabling deprecation early makes potential future removal easier in practice.


# Drawbacks
[drawbacks]: #drawbacks

* Deprecation will require code changes without bringing any immediate benefits.
  Until the deprecation is turned into a hard error the underlying issues will
  stay.

* It is not clear if it will ever be possible to remove this syntax entirely.


# Alternatives
[alternatives]: #alternatives

* Status quo.

* Decide on the precise removal plan prior to deprecation.

* Try to solve the underlying annoyances in some other way. For example,
  unbounded look ahead can be used in the parser to allow both anonymous
  parameters and the full pattern syntax.


# Unresolved questions
[unresolved]: #unresolved-questions

* How often are anonymous parameters used in practice? There is a rough
  estimate: Servo and its dependencies omit parameter names 34 times.

* Is there a consensus that anonymous parameters are not a useful language
  feature?
