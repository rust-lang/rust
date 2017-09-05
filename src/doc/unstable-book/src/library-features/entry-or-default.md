# `entry_or_default`

The tracking issue for this feature is: [#44324]

[#44324]: https://github.com/rust-lang/rust/issues/44324

------------------------

The `entry_or_default` feature adds a new method to `hash_map::Entry`
and `btree_map::Entry`, `or_default`, when `V: Default`. This method is
semantically identical to `or_insert_with(Default::default)`, and will
insert the default value for the type if no entry exists for the current
key.
