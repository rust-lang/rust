- Start Date: 2014-07-02
- RFC PR: [rust-lang/rfcs#151](https://github.com/rust-lang/rfcs/pull/151)
- Rust Issue: [rust-lang/rust#12831](https://github.com/rust-lang/rust/issues/12831)

# Summary

Closures should capture their upvars by value unless the `ref` keyword is used.

# Motivation

For unboxed closures, we will need to syntactically distinguish between captures by value and captures by reference.

# Detailed design

This is a small part of #114, split off to separate it from the rest of the discussion going on in that RFC.

Closures should capture their upvars (closed-over variables) by value unless the `ref` keyword precedes the opening `|` of the argument list. Thus `|x| x + 2` will capture `x` by value (and thus, if `x` is not `Copy`, it will move `x` into the closure), but `ref |x| x + 2` will capture `x` by reference.

In an unboxed-closures world, the immutability/mutability of the borrow (as the case may be) is inferred from the type of the closure: `Fn` captures by immutable reference, while `FnMut` captures by mutable reference. In a boxed-closures world, the borrows are always mutable.

# Drawbacks

It may be that `ref` is unwanted complexity; it only changes the semantics of 10%-20% of closures, after all. This does not add any core functionality to the language, as a reference can always be made explicitly and then captured. However, there are a *lot* of closures, and the workaround to capture a reference by value is painful.

# Alternatives

As above, the impact of not doing this is that reference semantics would have to be achieved. However, the diff against current Rust was thousands of lines of pretty ugly code.

Another alternative would be to annotate each individual upvar with its capture semantics, like capture clauses in C++11. This proposal does not preclude adding that functionality should it be deemed useful in the future. Note that C++11 provides a syntax for capturing all upvars by reference, exactly as this proposal does.

# Unresolved questions

None.
