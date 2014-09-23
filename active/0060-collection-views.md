- Start Date: 2014-08-28
- RFC PR: (https://github.com/rust-lang/rfcs/pull/216)
- Rust Issue: (https://github.com/rust-lang/rust/issues/17320)

# Summary

Add additional iterator-like Entry objects to collections.
Entries provide a composable mechanism for in-place observation and mutation of a
single element in the collection, without having to "re-find" the element multiple times.
This deprecates several "internal mutation" methods like hashmap's `find_or_insert_with`.

# Motivation

As we approach 1.0, we'd like to normalize the standard APIs to be consistent, composable,
and simple. However, this currently stands in opposition to manipulating the collections in
an *efficient* manner. For instance, if one wishes to build an accumulating map on top of one
of the concrete maps, they need to distinguish between the case when the element they're inserting
is *already* in the map, and when it's *not*. One way to do this is the following:

```
if map.contains_key(&key) {
    *map.find_mut(&key).unwrap() += 1;
} else {
    map.insert(key, 1);
}
```

However, searches for `key` *twice* on every operation.
The second search can be squeezed out the `update` re-do by matching on the result
of `find_mut`, but the `insert` case will always require a re-search.

To solve this problem, Rust currently has an ad-hoc mix of "internal mutation" methods which
take multiple values or closures for the collection to use contextually. Hashmap in particular
has the following methods:

```
fn find_or_insert<'a>(&'a mut self, k: K, v: V) -> &'a mut V
fn find_or_insert_with<'a>(&'a mut self, k: K, f: |&K| -> V) -> &'a mut V
fn insert_or_update_with<'a>(&'a mut self, k: K, v: V, f: |&K, &mut V|) -> &'a mut V
fn find_with_or_insert_with<'a, A>(&'a mut self, k: K, a: A, found: |&K, &mut V, A|, not_found: |&K, A| -> V) -> &'a mut V
```

Not only are these methods fairly complex to use, but they're over-engineered and
combinatorially explosive. They all seem to return a mutable reference to the region
accessed "just in case", and `find_with_or_insert_with` takes a magic argument `a` to
try to work around the fact that the *two* closures it requires can't both close over
the same value (even though only one will ever be called). `find_with_or_insert_with`
is also actually performing the role of `insert_with_or_update_with`,
suggesting that these aren't well understood.

Rust has been in this position before: internal iteration. Internal iteration was (author's note: I'm told)
confusing and complicated. However the solution was simple: external iteration. You get
all the benefits of internal iteration, but with a much simpler interface, and greater
composability. Thus, this RFC proposes the same solution to the internal mutation problem.

# Detailed design

A fully tested "proof of concept" draft of this design has been implemented on top of hashmap,
as it seems to be the worst offender, while still being easy to work with. It sits as a pull request
[here](https://github.com/rust-lang/rust/pull/17378).

All the internal mutation methods are replaced with a single method on a collection: `entry`.
The signature of `entry` will depend on the specific collection, but generally it will be similar to
the signature for searching in that structure. `entry` will in turn return an `Entry` object, which
captures the *state* of a completed search, and allows mutation of the area.

For convenience, we will use the hashmap draft as an example.

```
/// Get an Entry for where the given key would be inserted in the map
pub fn entry<'a>(&'a mut self, key: K) -> Entry<'a, K, V>;

/// A view into a single occupied location in a HashMap
pub struct OccupiedEntry<'a, K, V>{ ... }

/// A view into a single empty location in a HashMap
pub struct VacantEntry<'a, K, V>{ ... }

/// A view into a single location in a HashMap
pub enum Entry<'a, K, V> {
    /// An occupied Entry
    Occupied(OccupiedEntry<'a, K, V>),
    /// A vacant Entry
    Vacant(VacantEntry<'a, K, V>),
}
```

Of course, the real meat of the API is in the Entry's interface (impl details removed):

```
impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value of this Entry
    pub fn get(&self) -> &V;

    /// Gets a mutable reference to the value of this Entry
    pub fn get_mut(&mut self) -> &mut V;

    /// Converts the entry into a mutable reference to its value
    pub fn into_mut(self) -> &'a mut V;

    /// Sets the value stored in this Entry
    pub fn set(&mut self, value: V) -> V;

    /// Takes the value stored in this Entry
    pub fn take(self) -> V;
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// Set the value stored in this Entry, and returns a reference to it
    pub fn set(self, value: V) -> &'a mut V;
}
```

There are definitely some strange things here, so let's discuss the reasoning!

First, `entry` takes a `key` by value, because this is the observed behaviour of the internal mutation
methods. Further, taking the `key` up-front allows implementations to avoid *validating* provided keys if
they require an owned `key` later for insertion. This key is effectively a *guarantor* of the entry.

Taking the key by-value might change once collections reform lands, and Borrow and ToOwned are available.
For now, it's an acceptable solution, because in particular, the primary use case of this functionality
is when you're *not sure* if you need to insert, in which case you should be prepared to insert.
Otherwise, `find_mut` is likely sufficient.

The result is actually an enum, that will either be Occupied or Vacant. These two variants correspond
to concrete types for when the key matched something in the map, and when the key didn't, repsectively.

If there isn't a match, the user has exactly one option: insert a value using `set`, which will also insert
the guarantor, and destroy the Entry. This is to avoid the costs of maintaining the structure, which
otherwise isn't particularly interesting anymore.

If there is a match, a more robust set of options is provided. `get` and `get_mut` provide access to the
value found in the location. `set` behaves as the vacant variant, but without destroying the entry. 
It also yields the old value. `take` simply removes the found value, and destroys the entry for similar reasons as `set`.

Let's look at how we one now writes `insert_or_update`:

There are two options. We can either do the following:

```
// cleaner, and more flexible if logic is more complex
let val = match map.entry(key) {
    Vacant(entry) => entry.set(0),
    Occupied(entry) => entry.into_mut(),
};
*val += 1;
```

or

```
// closer to the original, and more compact
match map.entry(key) {
    Vacant(entry) => { entry.set(1); },
    Occupied(mut entry) => { *entry.get_mut() += 1; },
}
```

Either way, one can now write something equivalent to the "intuitive" inefficient code, but it is now as efficient as the complex
`insert_or_update` methods. In fact, this matches so closely to the inefficient manipulation
that users could reasonable ignore Entries *until performance becomes an issue*, at which point
it's an almost trivial migration. Closures also aren't needed to dance around the fact that one may
want to avoid generating some values unless they have to, because that falls naturally out of
normal control flow.

If you look at the actual patch that does this, you'll see that Entry itself is exceptionally
simple to implement. Most of the logic is trivial. The biggest amount of work was just
capturing the search state correctly, and even that was mostly a cut-and-paste job.

With Entries, the gate is also opened for... *adaptors*!
Really want `insert_or_update` back? That can be written on top of this generically with ease.
However, such discussion is out-of-scope for this RFC. Adaptors can
be tackled in a back-compat manner after this has landed, and usage is observed. Also, this
proposal does not provide any generic trait for Entries, preferring concrete implementations for
the time-being.

# Drawbacks

* More structs, and more methods in the short-term

* More collection manipulation "modes" for the user to think about

* `insert_or_update_with` is kind of convenient for avoiding the kind of boiler-plate
found in the examples

# Alternatives

* Just put our foot down, say "no efficient complex manipulations", and drop
all the internal mutation stuff without a replacement.

* Try to build out saner/standard internal manipulation methods.

* Try to make this functionality a subset of [Cursors](http://discuss.rust-lang.org/t/pseudo-rfc-cursors-reversible-iterators/386/7),
which would be effectively a bi-directional mut_iter
where the returned references borrow the cursor preventing aliasing/safety issues,
so that mutation can be performed at the location of the cursor.
However, preventing invalidation would be more expensive, and it's not clear that
cursor semantics would make sense on e.g. a HashMap, as you can't insert *any* key
in *any* location.

* This RFC originally [proposed a design without enums that was substantially more complex]
(https://github.com/Gankro/rust/commit/6d6804a6d16b13d07934f0a217a3562384e55612).
However it had some interesting ideas about Key manipulation, so we mention it here for
historical purposes.

# Unresolved questions

Naming bikesheds!
