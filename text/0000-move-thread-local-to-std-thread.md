- Feature Name: N/A
- Start Date: 2015-02-25
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Move the contents of `std::thread_local` into `std::thread`. Fully
remove `std::thread_local` from the standard library.

# Motivation

Thread locals are directly related to threading. Combining the modules
would reduce the number of top level modules, making browsing the docs
easier as well as reduce the number of `use` statements.

# Detailed design

The goal is to move the contents of `std::thread_local` into
`std::thread`. There are a few possible strategies that could be used to
achieve this.

One option would be to move the contents as is into `std::thread`. This
would leave `Key` and `State` as is. There would be no naming conflict,
but the names would be less ideal since the containing module is not
directly related to thread locals anymore. This could be handled by
renaming the types to something like `LocalKey` and `LocalState`.

Another option would be to move the contents into a dedicated sub module
such as `std::thread::local`. This would mean some code would still have
an extra `use` statement for pulling in thread local related types, but
it would also enable doing:

`use std::thread::{local, Thread};`

# Drawbacks

It's pretty late in the 1.0 release cycle. This is a mostly bike
shedding level of a change. It may not be worth changing it at this
point and staying with two top level modules in `std`. Also, some users
may prefer to have more top level modules.

# Alternatives

Leaving `std::thread_local` in its own module.

# Unresolved questions

The exact strategy for moving the contents into `std::thread`
