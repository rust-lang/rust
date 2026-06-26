# `allocator_ext`

The tracking issue for this feature is: [#163177]

[#163177]: https://github.com/rust-lang/rust/issues/163177

------------------------

Sometimes you want the memory for one collection to use a different
allocator than the memory for another collection. In this case,
replacing the global allocator is not a workable option. Instead,
you need to pass in an instance of an `AllocRef` to each collection
for which you want a custom allocator.

TBD
