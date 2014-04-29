- Start Date: 2014-03-28
- RFC PR #: 
- Rust Issue #: 

# Summary

I propose removing support for weak pointers from `Rc<T>`, and adding it to a
separate type `WRc<T>`.

# Motivation

`Rc<T>` currently uses two words for reference counting, to support an
infrequently used feature - weak pointers. A search of the rust and servo
codebases show zero uses of weak pointers, and the vast majority of reference
counted code does not create cycles. However, outside of the spirit of not
paying for what you don't use, every single `Rc<T>` keeps an extra word,
the weak count, than it needs to.

Hopefully this will help decrease the memory usage of rustc after all of the
managed boxes get turned into `Rc<T>`s.

# Detailed design

I just want to rename `Rc<T>` to `WRc<T>`, and create a similar datastructure
with its old name `Rc<T>` which just does not have a weak count. It will
support all of the same operations except for downgrading.

A similar transformation will also be done on `ARc<T>`, with a renaming to
`AWRc<T>`.

# Alternatives

Of course, we could do nothing. That's fine, but using all this memory for no
good reason is kind of unfortunate.

# Unresolved questions

Bikeshedding over the naming.
