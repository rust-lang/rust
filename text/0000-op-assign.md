- Feature Name: op_assign
- Start Date: 2015-03-08
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add the family of `[Op]Assign` traits to allow overloading assignment
operations like `a += b`.

# Motivation

We already let users overload the binary operations, letting them overload the
assignment version is the next logical step. Plus, this sugar is important to
make mathematical libraries more palatable.

# Detailed design

Add the following **unstable** traits to libcore and reexported them in stdlib:

```
// `+=`
#[lang = "add_assign"]
trait AddAssign<Rhs=Self> {
    fn add_assign(&mut self, &Rhs);
}

// the remaining traits have the same signature
// (lang items have been omitted for brevity)
trait BitAndAssign { .. }  // `&=`
trait BitOrAssign { .. }   // `|=`
trait BitXorAssign { .. }  // `^=`
trait DivAssign { .. }     // `/=`
trait MulAssign { .. }     // `*=`
trait RemAssign { .. }     // `%=`
trait ShlAssign { .. }     // `<<=`
trait ShrAssign { .. }     // `>>=`
trait SubAssign { .. }     // `-=`
```

Implement these traits for the primitive numeric types *without* overloading,
i.e. only `impl AddAssign<i32> for i32 { .. }`.

Add an `op_assign` feature gate. When the feature gate is enabled, the compiler
will consider these traits when typecheking `a += b`. Without the feature gate
the compiler will enforce that `a` and `b` must be primitives of the same
type/category as it does today.

Once we feel comfortable with the implementation we'll remove the feature gate
and mark the traits as stable. This can be done after 1.0 as this change is
backwards compatible.

## RHS: By ref vs by value

This RFC proposes that the assignment operations take the RHS always by ref;
instead of by value like the "normal" binary operations (e.g. `Add`) do. The
rationale is that, as far as the author has seen in practice [1], one never
wants to mutate the RHS or consume it, or in other words an immutable view into
the RHS is enough to perform the operation. Therefore, this RFC follows in the
footsteps of the `Index` traits, where the same situation arises with the
indexing value, and by ref was chosen over by value.

[1] It could be possible that the author is not aware of use cases where taking
RHS by value is necessary. Feedback on this matter would be appreciated. (See
the first unresolved question)

# Drawbacks

None that I can think of.

# Alternatives

Alternatively, we could change the traits to take the RHS by value. This makes
them more "flexible" as the user can pick by value or by reference, but makes
the use slightly unergonomic in the by ref case as the borrow must be explicit
e.g. `x += &big_float;` vs `x += big_float;`.

# Unresolved questions

Are there any use cases of assignment operations where the RHS has to be taken
by value for the operation to be performant (e.g. to avoid internal cloning)?

Should we overload `ShlAssign` and `ShrAssign`, e.g.
`impl ShlAssign<u8> for i32`, since we have already overloaded the `Shl` and
`Shr` traits?

Should we overload all the traits for references, e.g.
`impl<'a> AddAssign<&'a i32> for i32` to allow `x += &0;`?
