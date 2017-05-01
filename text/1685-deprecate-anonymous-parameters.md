- Feature Name: deprecate_anonymous_parameters
- Start Date: 2016-07-19
- RFC PR: https://github.com/rust-lang/rfcs/pull/1685
- Rust Issue: https://github.com/rust-lang/rust/issues/41686

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

Anonymous parameters are a [historic accident]. They cause a number of technical
annoyances.

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


Even if we exclude these technical drawbacks, it can be argued that allowing to
omit parameter names unnecessary complicates the language. It is unnecessary
because it does not make Rust more expressive and does not provide noticeable
ergonomic improvements. It is trivial to add parameter name, and only a small
fraction of method declarations actually omits it.

Another drawback of this syntax is its impact on the learning curve. One needs
to have a C background to understand that `fn foo(T);` means a function with
single parameter of type `T`. If one comes from dynamically typed language like
Python or JavaScript, this `T` looks more like a parameter name.

Anonymous parameters also cause inconsistencies between trait definitions and
implementations. One way to write an implementation is to copy the method
prototypes from the trait into the impl block. With anonymous parameters this
leads to syntax errors.


[historic accident]: https://github.com/rust-lang/rust/pull/29406#issuecomment-151859611
[IntelliJ Rust]: https://github.com/intellij-rust/intellij-rust/commit/1bb65c47341a04aecef5fa6817e8b2b56bfc9abb#diff-66f3ba596f0ecf74a2942b3223789ab5R41


# Detailed design
[design]: #detailed-design


## Backward compatibility

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

Of the 5560 packages from crates.io, 416 include at least one usage of
an anonymous parameter ([full report]).

[full report]: https://github.com/rust-lang/rfcs/pull/1685#issuecomment-238954434


## Benefits of deprecation

So the proposal is just to deprecate this syntax. Phasing the syntax out of
usage will mostly solve the learning curve problems. The technical problems
would not be solved until the actual removal becomes feasible and
practical. This hypothetical future may include:

* Rust 2.0 release.
* A widely deployed tool to automatically fix deprecation warnings.
* Storing crates on crates.io in "elaborated" syntax independent format.

Enabling deprecation early makes potential future removal easier in practice.


## Deprecation strategy

There are two possible ways to deprecate this syntax:

### Hard deprecation

One option is to produce a warning for anonymous parameters. This is backwards
compatible, but in practice will force crate authors to actively change their
code to avoid the warnings, causing code churn.

### Soft deprecation

Another option is to clearly document this syntax as deprecated and add an
allow-by-default lint, a clippy lint, and an IntelliJ Rust inspection, but do
not produce compiler warnings by default. This will make the update process more
gradual, but will delay the benefits of deprecation.

### Automatic transition

Rustfmt and IntelliJ Rust can automatically change anonymous parameters to
`_`. However it is better to manually add real names to make it obvious what
name is expected on the `impl` side.

# Drawbacks
[drawbacks]: #drawbacks

* Hard deprecation will cause code churn.

* Soft deprecation might not be as efficient at removing the syntax from usage.

* The technical issues can not be solved nicely until the deprecation is turned
  into a hard error.

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

* What deprecation strategy should be chosen?
