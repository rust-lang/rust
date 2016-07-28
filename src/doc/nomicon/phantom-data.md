% PhantomData

When working with unsafe code, we can often end up in a situation where
types or lifetimes are logically associated with a struct, but not actually
part of a field. This most commonly occurs with lifetimes. For instance, the
`Iter` for `&'a [T]` is (approximately) defined as follows:

```rust,ignore
struct Iter<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
}
```

However because `'a` is unused within the struct's body, it's *unbounded*.
Because of the troubles this has historically caused, unbounded lifetimes and
types are *forbidden* in struct definitions. Therefore we must somehow refer
to these types in the body. Correctly doing this is necessary to have
correct variance and drop checking.

We do this using `PhantomData`, which is a special marker type. `PhantomData`
consumes no space, but simulates a field of the given type for the purpose of
static analysis. This was deemed to be less error-prone than explicitly telling
the type-system the kind of variance that you want, while also providing other
useful such as the information needed by drop check.

Iter logically contains a bunch of `&'a T`s, so this is exactly what we tell
the PhantomData to simulate:

```
use std::marker;

struct Iter<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
    _marker: marker::PhantomData<&'a T>,
}
```

and that's it. The lifetime will be bounded, and your iterator will be variant
over `'a` and `T`. Everything Just Works.

Another important example is Vec, which is (approximately) defined as follows:

```
struct Vec<T> {
    data: *const T, // *const for variance!
    len: usize,
    cap: usize,
}
```

Unlike the previous example, it *appears* that everything is exactly as we
want. Every generic argument to Vec shows up in at least one field.
Good to go!

Nope.

The drop checker will generously determine that `Vec<T>` does not own any values
of type T. This will in turn make it conclude that it doesn't need to worry
about Vec dropping any T's in its destructor for determining drop check
soundness. This will in turn allow people to create unsoundness using
Vec's destructor.

In order to tell dropck that we *do* own values of type T, and therefore may
drop some T's when *we* drop, we must add an extra PhantomData saying exactly
that:

```
use std::marker;

struct Vec<T> {
    data: *const T, // *const for covariance!
    len: usize,
    cap: usize,
    _marker: marker::PhantomData<T>,
}
```

Raw pointers that own an allocation is such a pervasive pattern that the
standard library made a utility for itself called `Unique<T>` which:

* wraps a `*const T` for variance
* includes a `PhantomData<T>`
* auto-derives Send/Sync as if T was contained
* marks the pointer as NonZero for the null-pointer optimization
