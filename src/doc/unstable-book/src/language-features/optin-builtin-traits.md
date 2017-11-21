# `optin_builtin_traits`

The tracking issue for this feature is [#13231] 

[#13231]: https://github.com/rust-lang/rust/issues/13231

----

The `optin_builtin_traits` feature gate allows you to define auto traits.

Auto traits, like [`Send`] or [`Sync`] in the standard library, are marker traits
that are automatically implemented for every type, unless the type, or a type it contains, 
has explicitly opted out via a negative impl. 

[`Send`]: https://doc.rust-lang.org/std/marker/trait.Send.html
[`Sync`]: https://doc.rust-lang.org/std/marker/trait.Sync.html

```rust,ignore
impl !Type for Trait
```

Example:

```rust
#![feature(optin_builtin_traits)]

auto trait Valid {}

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
