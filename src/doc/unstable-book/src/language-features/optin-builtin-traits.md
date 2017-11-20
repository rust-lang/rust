# `optin_builtin_traits`

The tracking issue for this feature is [#13231] 

[#13231]: https://github.com/rust-lang/rust/issues/13231

----

The `optin_builtin_traits` feature gate allows you to define auto traits.

Auto traits, like [`Send`] or [`Sync`] in the standard library, are marker traits
that are automatically implemented for every type, unless the type, or a type it contains, 
has explictly opted out via a negative impl. 

[`Send`]: https://doc.rust-lang.org/std/marker/trait.Send.html
[`Sync`]: https://doc.rust-lang.org/std/marker/trait.Sync.html

```rust,ignore
impl !Type for Trait
```

[#46108] added the `DynSized` trait, which is an implicit bound for all traits. Auto traits may not
have bounds, so you must explicitly remove this bound with `?DynSized`. [#44917] adds a `Move` trait
which is also implicit, so when that lands, you will have to add `?Move` as well.

[#46108]: https://github.com/rust-lang/rust/pull/46108
[#44917]: https://github.com/rust-lang/rust/pull/44917


Example:

```rust
#![feature(optin_builtin_traits, dynsized)]

use std::marker::DynSized;

auto trait Valid: ?DynSized {}

struct True;
struct False;

impl !Valid for False {}

struct MaybeValid<T>(T);

fn must_be_valid<T: Valid>(_t: T) { }

fn main() {
    // works
    must_be_valid( MaybeValid(True) );
    // compiler error - trait bound not satisfied
    // must_be_valid( MaybeValid(False) );
}
```
