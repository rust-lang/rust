# `diagnostic_namespace`

The tracking issue for this feature is: [#111996]

[#111996]: https://github.com/rust-lang/rust/issues/111996

------------------------

The `diagnostic_namespace` feature permits customization of compilation errors.

## diagnostic::on_unimplemented

With [#114452] support for `diagnostic::on_unimplemented` was added.

When used on a trait declaration, the following options are available:

* `message` to customize the primary error message
* `note` to add a customized note message to an error message
* `label` to customize the label part of the error message

The attribute will hint to the compiler to use these in error messages:
```rust
// some library
#![feature(diagnostic_namespace)]

#[diagnostic::on_unimplemented(
    message = "cannot insert element",
    label = "cannot be put into a table",
    note = "see <link> for more information about the Table api"
)]
pub trait Element {
    // ...
}
```

```rust,compile_fail,E0277
# #![feature(diagnostic_namespace)]
#
# #[diagnostic::on_unimplemented(
#    message = "cannot insert element",
#    label = "cannot be put into a table",
#    note = "see <link> for more information about the Table api"
# )]
# pub trait Element {
#    // ...
# }
# struct Table;
# impl Table {
#    fn insert<T: Element>(&self, element: T) {
#        // ..
#    }
# }
# fn main() {
#    let table = Table;
#    let element = ();
// user code
table.insert(element);
# }
```

```text
error[E0277]: cannot insert element
  --> src/main.rs:24:18
   |
24 |     table.insert(element);
   |           ------ ^^^^^^^ cannot be put into a table
   |           |
   |           required by a bound introduced by this call
   |
   = help: the trait `Element` is not implemented for `<type>`
   = note: see <link> for more information about the Table api
note: required by a bound in `Table::insert`
  --> src/main.rs:15:18
   |
15 |     fn insert<T: Element>(&self, element: T) {
   |                  ^^^^^^^ required by this bound in `Table::insert`

For more information about this error, try `rustc --explain E0277`.
```

See [RFC 3368] for more information.

[#114452]: https://github.com/rust-lang/rust/pull/114452
[RFC 3368]: https://github.com/rust-lang/rfcs/blob/master/text/3368-diagnostic-attribute-namespace.md
