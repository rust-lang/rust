# `default_field_values`

The tracking issue for this feature is: [#132162]

[#132162]: https://github.com/rust-lang/rust/issues/132162

The RFC for this feature is: [#3681]

[#3681]: https://github.com/rust-lang/rfcs/blob/master/text/3681-default-field-values.md

------------------------

The `default_field_values` feature allows users to specify a const value for
individual fields in struct definitions, allowing those to be omitted from
initializers.

## Examples

```rust
#![feature(default_field_values)]

#[derive(Default)]
struct Pet {
    name: Option<String>, // impl Default for Pet will use Default::default() for name
    age: i128 = 42, // impl Default for Pet will use the literal 42 for age
}

fn main() {
    let a = Pet { name: Some(String::new()), .. }; // Pet { name: Some(""), age: 42 }
    let b = Pet::default(); // Pet { name: None, age: 42 }
    assert_eq!(a.age, b.age);
    // The following would be a compilation error: `name` needs to be specified
    // let _ = Pet { .. };
}
```

## `#[derive(Default)]`

When deriving Default, the provided values are then used. On enum variants,
the variant must still be marked with `#[default]` and have all its fields
with default values.

```rust
#![feature(default_field_values)]

#[derive(Default)]
enum A {
    #[default]
    B {
        x: i32 = 0,
        y: i32 = 0,
    },
    C,
}
```

## Enum variants

This feature also supports enum variants for both specifying default values
and `#[derive(Default)]`.

## Interaction with `#[non_exhaustive]`

A struct or enum variant marked with `#[non_exhaustive]` is not allowed to
have default field values.

## Lints

When manually implementing the `Default` trait for a type that has default
field values, if any of these are overridden in the impl the
`default_overrides_default_fields` lint will trigger. This lint is in place
to avoid surprising diverging behavior between `S { .. }` and
`S::default()`, where using the same type in both ways could result in
different values. The appropriate way to write a manual `Default`
implementation is to use the functional update syntax:

```rust
#![feature(default_field_values)]

struct Pet {
    name: String,
    age: i128 = 42, // impl Default for Pet will use the literal 42 for age
}

impl Default for Pet {
    fn default() -> Pet {
        Pet {
            name: "no-name".to_string(),
            ..
        }
    }
}
```
