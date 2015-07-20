- Feature Name: collection_recovery
- Start Date: 2015-07-08
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add element-recovery methods to the set types in `std`. Add key-recovery methods to the map types
in `std` in order to facilitate this.

# Motivation

Sets are sometimes used as a cache keyed on a certain property of a type, but programs may need to
access the type's other properties for efficiency or functionality. The sets in `std` do not expose
their elements (by reference or by value), making this use-case impossible.

Consider the following example:

```rust
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

// The `Widget` type has two fields that are inseparable.
#[derive(PartialEq, Eq, Hash)]
struct Widget {
    foo: Foo,
    bar: Bar,
}

#[derive(PartialEq, Eq, Hash)]
struct Foo(&'static str);

#[derive(PartialEq, Eq, Hash)]
struct Bar(u32);

// Widgets are normally considered equal if all their corresponding fields are equal, but we would
// also like to maintain a set of widgets keyed only on their `bar` field. To this end, we create a
// new type with custom `{PartialEq, Hash}` impls.
struct MyWidget(Widget);

impl PartialEq for MyWidget {
    fn eq(&self, other: &Self) -> bool { self.0.bar == other.0.bar }
}

impl Eq for MyWidget {}

impl Hash for MyWidget {
    fn hash<H: Hasher>(&self, h: &mut H) { self.0.bar.hash(h); }
}

fn main() {
    // In our program, users are allowed to interactively query the set of widgets according to
    // their `bar` field, as well as insert, replace, and remove widgets.

    let mut widgets = HashSet::new();

    // Add some default widgets.
    widgets.insert(MyWidget(Widget { foo: Foo("iron"), bar: Bar(1) }));
    widgets.insert(MyWidget(Widget { foo: Foo("nickel"), bar: Bar(2) }));
    widgets.insert(MyWidget(Widget { foo: Foo("copper"), bar: Bar(3) }));

    // At this point, the user enters commands and receives output like:
    //
    // ```
    // > get 1
    // Some(iron)
    // > get 4
    // None
    // > remove 2
    // removed nickel
    // > add 2 cobalt
    // added cobalt
    // > add 3 zinc
    // replaced copper with zinc
    // ```
    //
    // However, `HashSet` does not expose its elements via its `{contains, insert, remove}`
    // methods,  instead providing only a boolean indicator of the elements's presence in the set,
    // preventing us from implementing the desired functionality.
}
```

# Detailed design

Add the following element-recovery methods to `std::collections::{BTreeSet, HashSet}`:

```rust
impl<T> Set<T> {
    // Like `contains`, but returns a reference to the element if the set contains it.
    fn element<Q: ?Sized>(&self, element: &Q) -> Option<&T>;

    // Like `remove`, but returns the element if the set contained it.
    fn remove_element<Q: ?Sized>(&mut self, element: &Q) -> Option<T>;

    // Like `insert`, but replaces the element with the given one and returns the previous element
    // if the set contained it.
    fn replace(&mut self, element: T) -> Option<T>;
}
```

In order to implement the above methods, add the following key-recovery methods to
`std::collections::{BTreeMap, HashMap}`:

```rust
impl<K, V> Map<K, V> {
    // Like `get`, but additionally returns a reference to the entry's key.
    fn key_value<Q: ?Sized>(&self, key: &Q) -> Option<(&K, &V)>;

    // Like `get_mut`, but additionally returns a reference to the entry's key.
    fn key_value_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<(&K, &mut V)>;

    // Like `remove`, but additionally returns the entry's key.
    fn remove_key_value<Q: ?Sized>(&mut self, key: &Q) -> Option<(K, V)>;

    // Like `insert`, but additionally replaces the key with the given one and returns the previous
    // key and value if the map contained it.
    fn replace(&mut self, key: K, value: V) -> Option<(K, V)>;
}
```

Add the following key-recovery methods to `std::collections::{btree_map, hash_map}::OccupiedEntry`:

```rust
impl<'a, K, V> OccupiedEntry<'a, K, V> {
    // Like `get`, but additionally returns a reference to the entry's key.
    fn key_value(&self) -> (&K, &V);

    // Like `get_mut`, but additionally returns a reference to the entry's key.
    fn key_value_mut(&mut self) -> (&K, &mut V);

    // Like `into_mut`, but additionally returns a reference to the entry's key.
    fn into_key_value_mut(self) -> (&'a K, &'a mut V);

    // Like `remove`, but additionally returns the entry's key.
    fn remove_key_value(self) -> (K, V);
}
```

Add the following key-recovery methods to `std::collections::{btree_map, hash_map}::VacantEntry`:

```rust
impl<'a, K, V> VacantEntry<'a, K, V> {
    /// Returns a reference to the entry's key.
    fn key(&self) -> &K;

    // Like `insert`, but additionally returns a reference to the entry's key.
    fn insert_key_value(self, value: V) -> (&'a K, &'a mut V);

    // Returns the entry's key without inserting it into the map.
    fn into_key(self) -> K;
}
```

# Drawbacks

This complicates the collection APIs.

The distinction between `insert` and `replace` may be confusing. It would be more consistent to
call `Set::replace` `Set::insert_element` and `Map::replace` `Map::insert_key_value`, but
`BTreeMap` and `HashMap` do not replace equivalent keys in their `insert` methods, so rather than
have `insert` and `insert_key_value` behave differently in that respect, `replace` is used instead.

# Alternatives

Do nothing.

# Unresolved questions

Are these the best method names?

Should `{BTreeMap, HashMap}::insert` be changed to replace equivalent keys? This could break code
relying on the old behavior, and would add an additional inconsistency to `OccupiedEntry::insert`.
