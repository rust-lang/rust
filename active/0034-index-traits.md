- Start Date: 2014-06-09
- RFC PR: [rust-lang/rfcs#111](https://github.com/rust-lang/rfcs/pull/111)
- Rust Issue: [rust-lang/rust#6515](https://github.com/rust-lang/rust/issues/6515)

# Summary

`Index` should be split into `Index` and `IndexMut`.

# Motivation

Currently, the `Index` trait is not suitable for most array indexing tasks. The slice functionality cannot be replicated using it, and as a result the new `Vec` has to use `.get()` and `.get_mut()` methods.

Additionally, this simply follows the `Deref`/`DerefMut` split that has been implemented for a while.

# Detailed design

We split `Index` into two traits (borrowed from @nikomatsakis):

    // self[element] -- if used as rvalue, implicitly a deref of the result
    trait Index<E,R> {
        fn index<'a>(&'a self, element: &E) -> &'a R;
    }

    // &mut self[element] -- when used as a mutable lvalue
    trait IndexMut<E,R> {
        fn index_mut<'a>(&'a mut self, element: &E) -> &'a mut R;
    }

# Drawbacks

* The number of lang. items increases.

* This design doesn't support moving out of a vector-like object. This can be added backwards compatibly.

* This design doesn't support hash tables because there is no assignment operator. This can be added backwards compatibly.

# Alternatives

The impact of not doing this is that the `[]` notation will not be available to `Vec`.

# Unresolved questions

None that I'm aware of.
