- Feature Name: embrace-extend-extinguish
- Start Date: 2015-02-13
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Extend the Extend trait to take IntoIterator, and make all collections
`impl<'a, T: Clone> Extend<&'a T>`. This enables both `vec.extend(&[1, 2, 3])`, and
`vec.extend(&hash_set)`. This provides a more expressive replacement for
`Vec::push_all` with literally no ergonomic loss, while leveraging established APIs.

# Motivation

Vec::push_all is kinda random and specific. Partially motivated by performance concerns,
but largely just "nice" to not have to do something like
`vec.extend([1, 2, 3].iter().cloned())`. The performance argument falls flat
(we *must* make iterators fast, and trusted_len should get us there). The ergonomics
argument is salient, though. Working with Plain Old Data types in Rust is super annoying
because generic APIs and semantics are tailored for non-Copy types.

Even with Extend upgraded to take IntoIterator, that won't work with &[Copy],
because a slice can't be moved out of. Collections would have to take `IntoIterator<&T>`,
and clone out of the reference. So, do exactly that.

As a bonus, this is more expressive than `push_all`, because you can feed in any
collection by-reference to clone the data out of it.

# Detailed design

Here's a quick hack to get this working today:

```
/// A type growable from an `Iterator` implementation
pub trait Extend<T> {
    fn extend<It: Iterator<Item=T>, I: IntoIterator<IntoIter=It>>
        (&mut self, iterator: I);
}
```

This isn't the signature we'd like longterm, but it's what works with today's
IntoIterator and where clauses. Longterm (like, tomorrow) this should work:

```
/// A type growable from an `Iterator` implementation
pub trait Extend<T> {
    fn extend<I: IntoIterator<Item=T>>(&mut self, iterator: I);
}
```

And here's usage:

```
use std::iter::IntoIterator;

impl<'a, T: Clone> Extend<&'a T> for Vec<T> {
    fn extend<It: Iterator<Item=&'a T>, I: IntoIterator<IntoIter=It>>
            (&mut self, iterator: I){
        self.extend(iterator.into_iter().cloned())
    }
}


fn main() {
    let mut foo = vec![1];
    foo.extend(&[1, 2, 3, 4]);
    let bar = vec![1, 2, 3];
    foo.extend(&bar);
    foo.extend(bar.iter());

    println!("{:?}", foo);
}
```

# Drawbacks

Mo' generics, mo' magic. How you gonna discover it?

Hidden clones?

# Alternatives

Nope.

# Unresolved questions

FromIterator could also be extended in the same manner, but this is less useuful for
two reasons:

* FromIterator is always called by calling `collect`, and IntoIterator doesn't really
"work" right in `self` position.
* Introduces ambiguities in some cases. What is `let foo: Vec<_> = [1, 2, 3].iter().collect()`?

Of course, context might disambiguate in many cases, and
`let foo: Vec<i32> = [1, 2, 3].iter().collect()` might still be nicer than
`let foo: Vec<_> = [1, 2, 3].iter().cloned().collect()`.
