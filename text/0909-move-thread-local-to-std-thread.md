- Feature Name: N/A
- Start Date: 2015-02-25
- RFC PR: https://github.com/rust-lang/rfcs/pull/909
- Rust Issue: https://github.com/rust-lang/rust/issues/23547

# Summary

Move the contents of `std::thread_local` into `std::thread`. Fully
remove `std::thread_local` from the standard library.

# Motivation

Thread locals are directly related to threading. Combining the modules
would reduce the number of top level modules, combine related concepts,
and make browsing the docs easier. It also would have the potential to
slightly reduce the number of `use` statementsl

# Detailed design

The contents of`std::thread_local` module would be moved into to
`std::thread::local`. `Key` would be renamed to `LocalKey`, and
`scoped` would also be flattened, providing `ScopedKey`, etc. This
way, all thread related code is combined in one module.

It would also allow using it as such:

```rust
use std::thread::{LocalKey, Thread};
```

# Drawbacks

It's pretty late in the 1.0 release cycle. This is a mostly bike
shedding level of a change. It may not be worth changing it at this
point and staying with two top level modules in `std`. Also, some users
may prefer to have more top level modules.

# Alternatives

An alternative (as the RFC originally proposed) would be to bring
`thread_local` in as a submodule, rather than flattening. This was
decided against in an effort to keep hierarchies flat, and because of
the slim contents on the `thread_local` module.

# Unresolved questions

The exact strategy for moving the contents into `std::thread`
