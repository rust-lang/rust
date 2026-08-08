# `alloc_token_hint`

The tracking issue for this feature is: [#159111]

[#159111]: https://github.com/rust-lang/rust/issues/159111

------------------------

The `alloc_token_hint` feature allows the user to define the allocation token
hint (i.e., the contains-pointer classification and type name encoding) for
user-defined types.

It allows the user to place a type in a different partition than the one
assigned by the selected heap partitioning scheme, or to use different names for
types that otherwise would be required to have the same name as used in
externally defined C types.

## Examples

```rust
#![feature(alloc_token_hint)]

#[alloc_token_hint(contains_pointers = false)]
pub struct Counters([usize; 8]);

#[alloc_token_hint(type_name = "Foo", contains_pointers = true)]
#[repr(C)]
pub struct Type1 {
    next: *mut Type1,
}
```
