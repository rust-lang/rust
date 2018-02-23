- Feature Name: impl-only-use
- Start Date: 2017-10-01
- RFC PR: [rust-lang/rfcs#2166](https://github.com/rust-lang/rfcs/pull/2166)
- Rust Issue: [rust-lang/rust#48216](https://github.com/rust-lang/rust/issues/48216)

# Summary
[summary]: #summary

The `use …::{… as …}` syntax can now accept `_` as alias to a trait to only import the
implementations of such a trait.

# Motivation
[motivation]: #motivation

Sometimes, we might need to `use` a trait to be able to use its methods on a type in our code.
However, we might also not want to import the trait symbol (because we redefine it, for instance):

```rust
// in zoo.rs
pub trait Zoo {
  fn zoo(&self) -> u32;
}

// several impls here
// …
```

```rust
// in main.rs
struct Zoo {
  // …
}

fn main() {
  let x = "foo";
  let y = x.zoo(); // won’t compile because `zoo::Zoo` not in scope
}
```

To solve this, we need to import the trait:

```rust
// in main.rs
use zoo::Zoo;

struct Zoo { // wait, what happens here?
  // …
}

fn main() {
  let x = "foo";
  let y = x.zoo();
}
```

However, you can see that we’ll hit a problem here, because we define an ambiguous symbol. We have
two solutions:

- Change the name of the `struct` to something else.
- Qualify the `use`.

The problem is that if we qualify the `use`, what name do we give the trait? We’re not even
referring to it directly.

```rust
use zoo::Zoo as ZooTrait;
```

This will work but seems a bit like a hack because rustc forces us to give a name to something we
won’t use in our types.

This RFC suggests to solve this by adding the possibility to explictly state that we won’t directly
refer to that trait, but we want the impls:

```rust
use zoo::Zoo as _;
```

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Qualifying a `use` with `_` on a trait imports the trait’s `impl`s but not the symbol directly. It’s
handy if you don’t use the trait’s symbol in your type and if you redefine the symbol to something
else.

The `_` means that you “don’t care about the name rustc will use for that qualified `use`“.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

`use Trait as _` needs to desugar into `use Trait as SomeGenSym`. With this scheme, global imports
and exports can work properly with such items, i.e. import / re-export them.

```rust
mod m {
  pub use Trait as _;

  // `Trait` is in scope
}

use m::*;

// `Trait` is in scope too
```

In the case where the symbol is not a *trait*, it works the exact same way. However, a warning must
be emitted by the compiler to state the unused import (as types don’t have `impl`!).

In the same way, it’s possible to use the same mechanism with `extern crate` for linking-only
crates:

```rust
extern crate my_crate as _;
```

# Drawbacks
[drawbacks]: #drawbacks

This RFC tries to solve a very specific problem (when you *must* alias a trait use). It’s just a
nit to make the syntax more *“rust-ish”* (it’s very easy to think such a thing would work given the
way `_` works pretty much everywhere else).

# Rationale and alternatives
[alternatives]: #alternatives

The simple alternative is to let the programmer give a name to the qualified import, which is not a
big deal, but is a bit ugly.

# Unresolved questions
[unresolved]: #unresolved-questions
