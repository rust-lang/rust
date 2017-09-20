# `entry_and_modify`

The tracking issue for this feature is: [#44733]

[#44733]: https://github.com/rust-lang/rust/issues/44733

------------------------

This introduces a new method for the Entry API of maps
(`std::collections::HashMap` and `std::collections::BTreeMap`), so that
occupied entries can be modified before any potential inserts into the
map.

For example:

```rust
#![feature(entry_and_modify)]
# fn main() {
use std::collections::HashMap;

struct Foo {
    new: bool,
}

let mut map: HashMap<&str, Foo> = HashMap::new();

map.entry("quux")
   .and_modify(|e| e.new = false)
   .or_insert(Foo { new: true });
# }
```

This is not possible with the stable API alone since inserting a default
_before_ modifying the `new` field would mean we would lose the default state:

```rust
# fn main() {
use std::collections::HashMap;

struct Foo {
    new: bool,
}

let mut map: HashMap<&str, Foo> = HashMap::new();

map.entry("quux").or_insert(Foo { new: true }).new = false;
# }
```

In the above code the `new` field will never be `true`, even though we only
intended to update that field to `false` for previously extant entries.

To achieve the same effect as `and_modify` we would have to manually match
against the `Occupied` and `Vacant` variants of the `Entry` enum, which is
a little less user-friendly, and much more verbose:

```rust
# fn main() {
use std::collections::HashMap;
use std::collections::hash_map::Entry;

struct Foo {
    new: bool,
}

let mut map: HashMap<&str, Foo> = HashMap::new();

match map.entry("quux") {
    Entry::Occupied(entry) => {
        entry.into_mut().new = false;
    },
    Entry::Vacant(entry) => {
        entry.insert(Foo { new: true });
    },
};
# }
```
