- Start Date: 2014-06-09
- RFC PR #: (leave this empty)
- Rust Issue #: #6515

# Summary

`Index` should be split into `Index`, `IndexMut`, and `IndexAssign`

# Motivation

Currently, the `Index` trait is not suitable for most array indexing tasks. The slice functionality cannot be replicated using it, and as a result the new `Vec` has to use `.get()` and `.get_mut()` methods.

Additionally, this simply follows the `Deref`/`DerefMut` split that has been implemented for a while.

# Detailed design

We split `Index` into three traits (borrowed from @nikomatsakis):

    // self[element] -- if used as rvalue, implicitly a deref of the result
    trait Index<E,R> {
        fn index<'a>(&'a self, element: &E) -> &'a R;
    }

    // &mut self[element] -- when used as a mutable lvalue
    trait IndexMut<E,R> {
        fn index_mut<'a>(&'a mut self, element: &E) -> &'a mut R;
    }

    // self[element] = value
    trait IndexSet<E,V> {
        fn index_set(&mut self, element: E, value: V);
    }

# Drawbacks

* The number of lang. items increases.

* This design doesn't support moving out of a vector-like object.

# Alternatives

The impact of not doing this is that the `[]` notation will not be available to `Vec`.

# Unresolved questions

None that I'm aware of.
