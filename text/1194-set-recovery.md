- Feature Name: `set_recovery`
- Start Date: 2015-07-08
- RFC PR: [rust-lang/rfcs#1194](https://github.com/rust-lang/rfcs/pull/1194)
- Rust Issue: [rust-lang/rust#28050](https://github.com/rust-lang/rust/issues/28050)

# Summary

Add element-recovery methods to the set types in `std`.

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
    fn get<Q: ?Sized>(&self, element: &Q) -> Option<&T>;

    // Like `remove`, but returns the element if the set contained it.
    fn take<Q: ?Sized>(&mut self, element: &Q) -> Option<T>;

    // Like `insert`, but replaces the element with the given one and returns the previous element
    // if the set contained it.
    fn replace(&mut self, element: T) -> Option<T>;
}
```

# Drawbacks

This complicates the collection APIs.

# Alternatives

Do nothing.
