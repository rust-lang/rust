# `lint_reasons`

The tracking issue for this feature is: [#54503]

[#54503]: https://github.com/rust-lang/rust/issues/54503

---

This feature guards two functions related to lint level attributes, as described
in [RFC 2383]:

[RFC 2383]: https://rust-lang.github.io/rfcs/2383-lint-reasons.html

## Reason information on lint level attributes

With this feature, it's possible to add a reason text to all lint level
attributes. This reason is displayed as part of the lint message, if the
affected lint is emitted.

Here is an example how the reason can be added and how it'll be displayed as
part of the emitted lint message:

```rust,compile_fail
#![feature(lint_reasons)]

fn main() {
    #[deny(unused_variables, reason = "unused variables, should be removed")]
    let unused = "How much wood would a woodchuck chuck?";
}
```

Which produces the output:

```text
error: unused variable: `unused`
 --> src/main.rs:5:9
  |
5 |     let unused = "How much wood would a woodchuck chuck?";
  |         ^^^^^^ help: if this is intentional, prefix it with an underscore: `_unused`
  |
  = note: unused variables, should be removed
note: the lint level is defined here
 --> src/main.rs:4:12
  |
4 |     #[deny(unused_variables, reason = "unused variables, should be removed")]
  |            ^^^^^^^^^^^^^^^^
```

Defining a reason is especially useful for `#[allow]` or `#[expect]` attributes
to explain why, the attribute has been added. Here is an example, how the reason
can be used in practice:

```rust
fn get_file_path() -> PathBuf {
    #[allow(unused_mut, reason = "the path will be modified on windows, but not on other platforms")]
    let mut path = std::env::current_exe()
        .expect("current executable path invalid")
        .with_file_name("some_file_name");

    #[cfg(target_os = "windows")]
    path.set_extension("exe");

    path
}
```

## Expecting lint emissions

The new `#[expect]` attribute is part of this feature. With this attribute, a user
can suppress a lint in the same way that `#[allow]` works. However, if the lint is
not emitted in the defined scope, a new `unfulfilled_lint_expectation` lint is
emitted.

Here is an example, that fulfills the expectation and compiles successfully:

```rust
#![feature(lint_reasons)]

fn main() {
    #[expect(unused_variables, reason = "WIP, I'll use this value later")]
    let message = "How much wood would a woodchuck chuck?";
}
```

If we change the code, to use the value like this:

```rust
#![feature(lint_reasons)]

fn main() {
    #[expect(unused_variables, reason = "WIP, I'll use this value later")]
    let message = "How much wood would a woodchuck chuck?";
    println!("{message}")
}
```

We'll receive the following output:

```text
warning: this lint expectation is unfulfilled
 --> src/main.rs:4:14
  |
4 |     #[expect(unused_variables, reason = "WIP, I'll use this value later")]
  |              ^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unfulfilled_lint_expectations)]` on by default
  = note: WIP, I'll use this value later
```

Note, that the `unfulfilled_lint_expectations` lint can't be expected. However,
it can be suppressed by the `#[allow]` attribute as usual.
