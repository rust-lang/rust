- Feature Name: `manually_drop`
- Start Date: 2017-01-20
- RFC PR: [rust-lang/rfcs#1860](https://github.com/rust-lang/rfcs/pull/1860)
- Rust Issue: [rust-lang/rust#40673](https://github.com/rust-lang/rust/issues/40673)

# Summary
[summary]: #summary

Include the `ManuallyDrop` wrapper in `core::mem`.

# Motivation
[motivation]: #motivation

Currently Rust does not specify the order in which the destructors are run. Furthermore, this order
differs depending on context. RFC issue [#744](https://github.com/rust-lang/rfcs/issues/744)
exposed the fact that the current, but unspecified behaviour is relied onto for code validity and
that there’s at least a few instances of such code in the wild.

While a move to stabilise and document the order of destructor evaluation would technically fix the
problem described above, there’s another important aspect to consider here – implicitness. Consider
such code:

```rust
struct FruitBox {
    peach: Peach,
    banana: Banana,
}
```

Does this structure depend on `Peach`’s destructor being run before `Banana` for correctness?
Perhaps its the other way around and it is `Banana`’s destructor that has to run first? In the
common case structures do not have any such dependencies between fields, and therefore it is easy
to overlook such a dependency while changing the code above to the snippet below (e.g. so the
fields are sorted by name).

```rust
struct FruitBox {
    banana: Banana,
    peach: Peach,
}
```

For structures with dependencies between fields it is worthwhile to have ability to explicitly
annotate the dependencies somehow.

# Detailed design
[design]: #detailed-design

This RFC proposes adding following `union` to the `core::mem` (and by extension the `std::mem`)
module. `mem` module is a most suitable place for such type, as the module already a place for
functions very similar in purpose: `drop` and `forget`.

```rust
/// Inhibits compiler from automatically calling `T`’s destructor.
#[unstable(feature = "manually_drop", reason = "recently added", issue = "0")]
#[allow(unions_with_drop_fields)]
pub union ManuallyDrop<T>{ value: T }

impl<T> ManuallyDrop<T> {
    /// Wraps a value to be manually dropped.
    #[unstable(feature = "manually_drop", reason = "recently added", issue = "0")]
    pub fn new(value: T) -> ManuallyDrop<T> {
        ManuallyDrop { value: value }
    }

    /// Extracts the value from the ManuallyDrop container.
    #[unstable(feature = "manually_drop", reason = "recently added", issue = "0")]
    pub fn into_inner(self) -> T {
        unsafe {
            self.value
        }
    }

    /// Manually drops the contained value.
    ///
    /// # Unsafety
    ///
    /// This function runs the destructor of the contained value and thus makes any further action
    /// with the value within invalid. The fact that this function does not consume the wrapper
    /// does not statically prevent further reuse.
    #[unstable(feature = "manually_drop", reason = "recently added", issue = "0")]
    pub unsafe fn drop(slot: &mut ManuallyDrop<T>) {
        ptr::drop_in_place(&mut slot.value)
    }
}

impl<T> Deref for ManuallyDrop<T> {
    type Target = T;
    // ...
}

impl<T> DerefMut for ManuallyDrop<T> {
    // ...
}

// Other common impls such as `Debug for T: Debug`.
```

Let us apply this union to a somewhat expanded example from the motivation:

```rust
struct FruitBox {
    // Immediately clear there’s something non-trivial going on with these fields.
    peach: ManuallyDrop<Peach>,
    melon: Melon, // Field that’s independent of the other two.
    banana: ManuallyDrop<Banana>,
}

impl Drop for FruitBox {
    fn drop(&mut self) {
        unsafe {
            // Explicit ordering in which field destructors are run specified in the intuitive
            // location – the destructor of the structure containing the fields.
            // Moreover, one can now reorder fields within the struct however much they want.
            ManuallyDrop::drop(&mut self.peach);
            ManuallyDrop::drop(&mut self.banana);
        }
        // After destructor for `FruitBox` runs (this function), the destructor for Melon gets
        // invoked in the usual manner, as it is not wrapped in `ManuallyDrop`.
    }
}
```

It is proposed that this pattern would become idiomatic for structures where fields must be dropped
in a particular order.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

It is expected that the functions and wrapper added as a result of this RFC would be seldom
necessary.

In addition to the usual API documentation, `ManuallyDrop` should be mentioned in
reference/nomicon/elsewhere as the solution to the desire of explicit control of the order in which
the structure fields gets dropped.

<!--
# Drawbacks
[drawbacks]: #drawbacks

No drawbacks known at the time.
-->

# Alternatives
[alternatives]: #alternatives

* Stabilise some sort of drop order and make people to write code that’s hard to figure out at a
glance;
* Bikeshed colour;
* Stabilise union and let people implement this themselves:
    * Precludes (or makes it much harder) from recommending this pattern as the idiomatic way to
    implement destructors with dependencies.

# Unresolved questions
[unresolved]: #unresolved-questions

None known.
