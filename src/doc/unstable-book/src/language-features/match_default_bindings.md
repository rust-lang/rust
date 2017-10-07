# `match_default_bindings`

The tracking issue for this feature is: [#42640]

[#42640]: https://github.com/rust-lang/rust/issues/42640

------------------------

Match default bindings (also called "default binding modes in match") improves ergonomics for
pattern-matching on references by introducing automatic dereferencing (and a corresponding shift
in binding modes) for large classes of patterns that would otherwise not compile.

For example, under match default bindings,

```rust
#![feature(match_default_bindings)]

fn main() {
    let x: &Option<_> = &Some(0);

    match x {
        Some(y) => {
            println!("y={}", *y);
        },
        None => {},
    }
}
```

compiles and is equivalent to either of the below:

```rust
fn main() {
    let x: &Option<_> = &Some(0);

    match *x {
        Some(ref y) => {
            println!("y={}", *y);
        },
        None => {},
    }
}
```

or

```rust
fn main() {
    let x: &Option<_> = &Some(0);

    match x {
        &Some(ref y) => {
            println!("y={}", *y);
        },
        &None => {},
    }
}
```
