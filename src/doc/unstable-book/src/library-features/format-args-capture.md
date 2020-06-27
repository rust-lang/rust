# `format_args_capture`

The tracking issue for this feature is: [#67984]

[#67984]: https://github.com/rust-lang/rust/issues/67984

------------------------

Enables `format_args!` (and macros which use `format_args!` in their implementation, such
as `format!`, `print!` and `panic!`) to capture variables from the surrounding scope.
This avoids the need to pass named parameters when the binding in question
already exists in scope.

```rust
#![feature(format_args_capture)]

let (person, species, name) = ("Charlie Brown", "dog", "Snoopy");

// captures named argument `person`
print!("Hello {person}");

// captures named arguments `species` and `name`
format!("The {species}'s name is {name}.");
```

This also works for formatting parameters such as width and precision:

```rust
#![feature(format_args_capture)]

let precision = 2;
let s = format!("{:.precision$}", 1.324223);

assert_eq!(&s, "1.32");
```

A non-exhaustive list of macros which benefit from this functionality include:
- `format!`
- `print!` and `println!`
- `eprint!` and `eprintln!`
- `write!` and `writeln!`
- `panic!`
- `unreachable!`
- `unimplemented!`
- `todo!`
- `assert!` and similar
- macros in many thirdparty crates, such as `log`
