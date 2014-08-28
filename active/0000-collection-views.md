- Start Date: 2014-08-28
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add additional iterator-like View objects to collections. 
Views provide a composable mechanism for in-place observation and mutation of a
single element in the collection, without having to "re-find" the element multiple times.
This deprecates several "internal mutation" methods like hashmap's `find_or_insert_with`.

# Motivation

As we approach 1.0, we'd like to normalize the standard APIs to be consistent, composable,
and simple. However, this currently stands in opposition to manipulating the collections in
an *efficient* manner. For instance, if we wish to build an accumulating map on top of one
of our concrete maps, we need to distinguish between the case when the element we're inserting
is *already* in the map, and when it's *not*. One way to do this is the following:

```
if map.contains_key(&key) {
    let v = map.find_mut(&key).unwrap();
    let new_v = *v + 1;
    *v = new_v;
} else {
    map.insert(key, 1);
}
```

However, this requires us to search for `key` *twice* on every operation.
We might be able to squeeze out the `update` re-do by matching on the result
of `find_mut`, but the `insert` case will always require a re-search.

To solve this problem, we have an ad-hoc mix of "internal mutation" methods which
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

Rust has been in this position before: internal iteration. Internal iteration was (I'm told)
confusing and complicated. However the solution was simple: external iteration. You get
all the benefits of internal iteration, but with a much simpler interface, and greater
composability. Thus, we propose the same solution to the internal mutation problem.

# Detailed design

A fully tested "proof of concept" draft of this design has been implemented on top of pczarn's
pending hashmap PR, as hashmap seems to be the worst offender, while still being easy
to work with. You can 
[read the diff here](https://github.com/Gankro/rust/commit/6d6804a6d16b13d07934f0a217a3562384e55612).

We replace all the internal mutation methods with a single method on a collection: `view`.
The signature of `view` will depend on the specific collection, but generally it will be similar to
the signature for searching in that structure. `view` will in turn return a `View` object, which
captures the *state* of a completed search, and allows mutation of the area. 

For convenience, we will use the hashmap draft as an example.

```
pub fn view<'a>(&'a mut self, key: K) -> Entry<'a, K, V>;
```

Of course, the real meat of the API is in the View's interface (impl details removed):

```
impl<'a, K, V> Entry<'a, K, V> {
    /// Get a reference to the value at the Entry's location
    pub fn get(&self) -> Option<&V>;

    /// Get a mutable reference to the value at the Entry's location
    pub fn get_mut(&mut self) -> Option<&mut V>;

    /// Get a reference to the key at the Entry's location
    pub fn get_key(&self) -> Option<&K>;

    /// Return whether the Entry's location contains anything
    pub fn is_empty(&self) -> bool;
    
    /// Get a reference to the Entry's key
    pub fn key(&self) -> &K;

    /// Set the key and value of the location pointed to by the Entry, and return any old
    /// key and value that might have been there
    pub fn set(self, value: V) -> Option<(K, V)>;

    /// Retrieve the Entry's key
    pub fn into_key(self) -> K;
}
```

There are definitely some strange things here, so let's discuss the reasoning! 

First, `view` takes a `key` by value, because we observe that this is how all the internal mutation 
methods work. Further, taking the `key` up-front allows us to avoid *validating* provided keys if 
we require an owned `key` later. This key is effectively a *guarantor* of the view. 
To compensate, we provide an `into_key` method which converts the entry back into its guarantor.
We also provide a `key` method for getting an immutable reference to the guarantor, in case its
value effects any computations one wishes to perform. 

Taking the key by-value might change once associated types land, 
and we successfully tackle the "equiv" problem. For now, it's an acceptable solution in our mind.
In particular, the primary use case of this functionality is when you're *not sure* if you need to
insert, in which case you should be prepared to insert. Otherwise, `find_mut` is likely sufficient.

Next, we provide a nice simple suite of "standard" methods: 
`get`, `get_mut`, `get_key`, and `is_empty`.
These do exactly what you would expect, and allow you to query the view to see if it is logically
empty, and if not, what it contains.

Finally, we provide a `set` method which inserts the provided value using the guarantor key, 
and yields the old key-value pair if it existed. Note that `set` consumes the View, because 
we lose the guarantor, and the collection might have to shift around a lot to compensate. 
Maintaining the entry after an insertion would add significant cost and complexity for no 
clear gain.

Let's look at how we now `insert_or_update`:

```
let mut view = map.view(key);
if view.is_empty() {
    let v = view.get_mut().unwrap();
    let new_v = *v + 1;
    *v = new_v;
} else {
    view.set(1);
}
```

We can now write our "intuitive" inefficient code, but it is now as efficient as the complex
`insert_or_update` methods. In fact, this matches so closely to the inefficient manipulation
that users could reasonable ignore views *until performance becomes an issue*. At which point
it's an almost trivial migration. We also don't need closures to dance around the fact that we
want to avoid generating some values unless we have to, because that falls naturally out of our
normal control flow.

If you look at the actual patch that does this, you'll see that Entry itself is exceptional
simple to implement. Most of the logic is trivial. The biggest amount of work was just
capturing the search state correctly, and even that was mostly a cut-and-paste job. 

# Drawbacks

* More structs, and more methods in the short-term 

* More collection manipulation "modes" for the user to think about

* `insert_or_update_with` is kind of convenient for avoiding the kind of boiler-plate
found in the examples

# Alternatives

* We can just put our foot down, say "no efficient complex manipulations", and drop 
all the internal mutation stuff without a replacement.

* We can try to build out saner/standard internal manipulation methods.

# Unresolved questions

One thing omitted from the design was a "take" method on the Entry. The reason for this
is primarily that this doesn't seem to be a thing people are interested in having for
internal manipulation. However, it also just would have meant more complexity, especially
if it *didn't* consume the View. Do we want this functionality?

The internal mutation methods cannot actually be implemented in terms of the View, because
they return a mutable reference at the end, and there's no way to do that with the current
View design. However, it's not clear why this is done by them. We believe it's simply to
validate what the method *actually did*. If this is the case, then Views make this functionality
obsolete. However, if this is *still* desirable, we could tweak `set` to do this as well.
Do we want this functionality?

Naming bikesheds!
