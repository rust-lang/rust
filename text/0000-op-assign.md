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

Add the following **unstable** traits to libcore and reexported them in libstd:

```
// `+=`
#[lang = "add_assign"]
trait AddAssign<Rhs=Self> {
    fn add_assign(&mut self, Rhs);
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

## RHS: By value vs by ref

Taking the RHS by value is more flexible. The implementations allowed with
a by value RHS are a superset of the implementations allowed with a by ref RHS.
An example where taking the RHS by value is necessary would be operator sugar
for extending a collection with an iterator [1]: `vec ++= iter` where
`vec: Vec<T>` and `iter impls Iterator<T>`. This can't be implemented with the
by ref version as the iterator couldn't be advanced in that case.

[1] Where `++` is the "combine" operator that has been proposed [elsewhere].
Note that this RFC doesn't propose adding that particular operator or adding
similar overloaded operations (`vec += iter`) to stdlib's collections, but it
leaves the door open to the possibility of adding them in the future (if
desired).

[elsewhere]: https://github.com/rust-lang/rfcs/pull/203

# Drawbacks

None that I can think of.

# Alternatives

Take the RHS by ref. This is less flexible than taking the RHS by value but, in
some instances, it can save writing `&rhs` when the RHS is owned and the
implementation demands a reference. However, this last point will be moot if we
implement auto-referencing for binary operators, as `lhs += rhs` would actually
call `add_assign(&mut lhs, &rhs)` if `Lhs impls AddAssign<&Rhs>`.

# Unresolved questions

Should we overload `ShlAssign` and `ShrAssign`, e.g.
`impl ShlAssign<u8> for i32`, since we have already overloaded the `Shl` and
`Shr` traits?

Should we overload all the traits for references, e.g.
`impl<'a> AddAssign<&'a i32> for i32` to allow `x += &0;`?
