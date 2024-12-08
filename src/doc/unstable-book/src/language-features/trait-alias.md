# `trait_alias`

The tracking issue for this feature is: [#41517]

[#41517]: https://github.com/rust-lang/rust/issues/41517

------------------------

The `trait_alias` feature adds support for trait aliases. These allow aliases
to be created for one or more traits (currently just a single regular trait plus
any number of auto-traits), and used wherever traits would normally be used as
either bounds or trait objects.

```rust
#![feature(trait_alias)]

trait Foo = std::fmt::Debug + Send;
trait Bar = Foo + Sync;

// Use trait alias as bound on type parameter.
fn foo<T: Foo>(v: &T) {
    println!("{:?}", v);
}

pub fn main() {
    foo(&1);

    // Use trait alias for trait objects.
    let a: &Bar = &123;
    println!("{:?}", a);
    let b = Box::new(456) as Box<dyn Foo>;
    println!("{:?}", b);
}
```
