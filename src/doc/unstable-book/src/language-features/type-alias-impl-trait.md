# `type_alias_impl_trait`

The tracking issue for this feature is: [#63063]

------------------------

> This feature is not to be confused with [`trait_alias`] or [`impl_trait_in_assoc_type`].

### What is `impl Trait`?

`impl Trait` in return position is useful for declaring types that are constrained by traits, but whose concrete type should be hidden:

```rust
use std::fmt::Debug;

fn new() -> impl Debug {
    42
}

fn main() {
    let thing = new();
    // What actually is a `thing`?
    // No idea but we know it implements `Debug`, so we can debug print it
    println!("{thing:?}");
}
```

See the [reference] for more information about `impl Trait` in return position.

### `type_alias_impl_trait`

However, we might want to use an `impl Trait` in multiple locations but actually use the same concrete type everywhere while keeping it hidden.
This can be useful in libraries where you want to hide implementation details.

The `#[define_opaque]` attribute must be used to explicitly list opaque items constrained by the item it's on.

```rust
#![feature(type_alias_impl_trait)]
# #![allow(unused_variables, dead_code)]
trait Trait {}

struct MyType;

impl Trait for MyType {}

type Alias = impl Trait;

#[define_opaque(Alias)] // To constrain the type alias to `MyType`
fn new() -> Alias {
    MyType
}

#[define_opaque(Alias)] // So we can name the concrete type inside this item
fn main() {
    let thing: MyType = new();
}

// It can be a part of a struct too
struct HaveAlias {
    stuff: String,
    thing: Alias,
}
```

In this example, the concrete type referred to by `Alias` is guaranteed to be the same wherever `Alias` occurs.

> Originally this feature included type aliases as an associated type of a trait. In [#110237] this was split off to [`impl_trait_in_assoc_type`].

### `type_alias_impl_trait` in argument position.

Note that using `Alias` as an argument type is *not* the same as argument-position `impl Trait`, as `Alias` refers to a unique type, whereas the concrete type for argument-position `impl Trait` is chosen by the caller.

```rust
# #![feature(type_alias_impl_trait)]
# #![allow(unused_variables)]
# pub mod x {
# pub trait Trait {}
#
# struct MyType;
#
# impl Trait for MyType {}
#
# pub type Alias = impl Trait;
#
# #[define_opaque(Alias)]
# pub fn new() -> Alias {
#     MyType
# }
# }
# use x::*;
// this...
pub fn take_alias(x: Alias) {
    // ...
}

// ...is *not* the same as
pub fn take_impl(x: impl Trait) {
    // ...
}
# fn main(){}
```

```rust,compile_fail,E0308
# #![feature(type_alias_impl_trait)]
# #![allow(unused_variables)]
# pub mod x {
# pub trait Trait {}
#
# struct MyType;
#
# impl Trait for MyType {}
#
# pub type Alias = impl Trait;
#
# #[define_opaque(Alias)]
# pub fn new() -> Alias {
#     MyType
# }
# }
# use x::*;
# pub fn take_alias(x: Alias) {
#     // ...
# }
#
# pub fn take_impl(x: impl Trait) {
#    // ...
# }
#
// a user's crate using the trait and type alias
struct UserType;
impl Trait for UserType {}

# fn main(){
let x = UserType;
take_alias(x);
// ERROR expected opaque type, found `UserType`
// this function *actually* takes a `MyType` as is constrained in `new`

let x = UserType;
take_impl(x);
// OK

let x = new();
take_alias(x);
// OK

let x = new();
take_impl(x);
// OK
# }
```

Note that the user cannot use `#[define_opaque(Alias)]` to reify the opaque type because only the crate where the type alias is declared may do so. But if this happened in the same crate and the opaque type was reified, they'd get a familiar error: "expected `MyType`, got `UserType`".

[#63063]: https://github.com/rust-lang/rust/issues/63063
[#110237]: https://github.com/rust-lang/rust/pull/110237
[reference]: https://doc.rust-lang.org/stable/reference/types/impl-trait.html#abstract-return-types
[`trait_alias`]: ./trait-alias.md
[`impl_trait_in_assoc_type`]: ./impl-trait-in-assoc-type.md
