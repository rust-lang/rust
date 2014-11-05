- Start Date: 2014-10-25
- RFC PR: [rust-lang/rfcs#418](https://github.com/rust-lang/rfcs/pull/418)
- Rust Issue: [rust-lang/rust#18641](https://github.com/rust-lang/rust/issues/18641)

# Summary

Just like structs, variants can come in three forms - unit-like, tuple-like,
or struct-like:
```rust
enum Foo {
    Foo,
    Bar(int, String),
    Baz { a: int, b: String }
}
```
The last form is currently feature gated. This RFC proposes to remove that gate
before 1.0.

# Motivation

Tuple variants with multiple fields can become difficult to work with,
especially when the types of the fields don't make it obvious what each one is.
It is not an uncommon sight in the compiler to see inline comments used to help
identify the various variants of an enum, such as this snippet from
`rustc::middle::def`:
```rust
pub enum Def {
    // ...
    DefVariant(ast::DefId /* enum */, ast::DefId /* variant */, bool /* is_structure */),
    DefTy(ast::DefId, bool /* is_enum */),
    // ...
}
```
If these were changed to struct variants, this ad-hoc documentation would move
into the names of the fields themselves. These names are visible in rustdoc,
so a developer doesn't have to go source diving to figure out what's going on.
In addition, the fields of struct variants can have documentation attached.
```rust
pub enum Def {
    // ...
    DefVariant {
        enum_did: ast::DefId,
        variant_did: ast::DefId,
        /// Identifies the variant as tuple-like or struct-like
        is_structure: bool,
    },
    DefTy {
        did: ast::DefId,
        is_enum: bool,
    },
    // ...
}
```

As the number of fields in a variant increases, it becomes increasingly crucial
to use struct variants. For example, consider this snippet from
`rust-postgres`:
```rust
enum FrontendMessage<'a> {
    // ...
    Bind {
        pub portal: &'a str,
        pub statement: &'a str,
        pub formats: &'a [i16],
        pub values: &'a [Option<Vec<u8>>],
        pub result_formats: &'a [i16]
    },
    // ...
}
```
If we convert `Bind` to a tuple variant:
```rust
enum FrontendMessage<'a> {
    // ...
    Bind(&'a str, &'a str, &'a [i16], &'a [Option<Vec<u8>>], &'a [i16]),
    // ...
}
```
we run into both the documentation issues discussed above, as well as ergonomic
issues. If code only cares about the `values` and `formats` fields, working
with a struct variant is nicer:
```rust
match msg {
    // you can reorder too!
    Bind { values, formats, .. } => ...
    // ...
}
```
versus
```rust
match msg {
    Bind(_, _, formats, values, _) => ...
    // ...
}
```

This feature gate was originally put in place because there were many serious
bugs in the compiler's support for struct variants. This is not the case today.
The issue tracker does not appear have any open correctness issues related to
struct variants and many libraries, including rustc itself, have been using
them without trouble for a while.

# Detailed design

Change the `Status` of the `struct_variant` feature from `Active` to
`Accepted`.

The fields of struct variants use the same style of privacy as normal struct
fields - they're private unless tagged `pub`. This is inconsistent with tuple
variants, where the fields have inherited visibility. Struct variant fields
will be changed to have inhereted privacy, and `pub` will no longer be allowed.

# Drawbacks

Adding formal support for a feature increases the maintenance burden of rustc.

# Alternatives

If struct variants remain feature gated at 1.0, libraries that want to ensure
that they will continue working into the future will be forced to avoid struct
variants since there are no guarantees about backwards compatibility of
feature-gated parts of the language.

# Unresolved questions

N/A
