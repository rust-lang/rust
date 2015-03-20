- Feature Name: N/A
- Start Date: 2015-02-25
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Move the contents of `std::thread_local` into `std::thread`. Fully
remove `std::thread_local` from the standard library.

# Motivation

Thread locals are directly related to threading. Combining the modules
would reduce the number of top level modules, combine related concepts,
and make browsing the docs easier. It also would have the potential to
slightly reduce the number of `use` statementsl

# Detailed design

The `std::thread_local` module would be renamed to `std::thread::local`.
All contents of the module would remain the same. This way, all thread
related code is combined in one module.

It would also allow using it as such:

```rust
use std::thread::{local, Thread};
```

# Drawbacks

It's pretty late in the 1.0 release cycle. This is a mostly bike
shedding level of a change. It may not be worth changing it at this
point and staying with two top level modules in `std`. Also, some users
may prefer to have more top level modules.

# Alternatives

Another strategy for moving `std::thread_local` would be to move it
directly into `std::thread` without scoping it in a dedicated module.
There are no naming conflicts, but the names would not be ideal anymore.
One way to mitigate would be to rename the types to something like
`LocalKey` and `LocalState`.

# Unresolved questions

The exact strategy for moving the contents into `std::thread`
