- Start Date: 2014-04-05
- RFC PR: [rust-lang/rfcs#34](https://github.com/rust-lang/rfcs/pull/34)
- Rust Issue: [rust-lang/rust#15759](https://github.com/rust-lang/rust/issues/15759)

# Summary

Check all types for well-formedness with respect to the bounds of type variables.

Allow bounds on formal type variable in structs and enums. Check these bounds
are satisfied wherever the struct or enum is used with actual type parameters.

# Motivation

Makes type checking saner. Catches errors earlier in the development process.
Matches behaviour with built-in bounds (I think).

Currently formal type variables in traits and functions may have bounds and
these bounds are checked whenever the item is used against the actual type
variables. Where these type variables are used in types, these types
should be checked for well-formedness with respect to the type definitions.
E.g.,

```
trait U {}
trait T<X: U> {}
trait S<Y> {
    fn m(x: ~T<Y>) {}  // Should be flagged as an error
}
```

Formal type variables in structs and enums may not have bounds. It is possible
to use these type variables in the types of fields, and these types cannot be
checked for well-formedness until the struct is instantiated, where each field
must be checked.

```
struct St<X> {
    f: ~T<X>, // Cannot be checked
}
```

Likewise, impls of structs are not checked. E.g.,

```
impl<X> St<X> {  // Cannot be checked
    ...
}
```

Here, no struct can exist where `X` is replaced by something implementing `U`,
so in the impl, `X` can be assumed to have the bound `U`. But the impl does not
indicate this. Note, this is sound, but does not indicate programmer intent very
well.

# Detailed design

Whenever a type is used it must be checked for well-formedness. For polymorphic
types we currently check only that the type exists. I would like to also check
that any actual type parameters are valid. That is, given a type `T<U>` where
`T` is declared as `T<X: B>`, we currently only check that `T` does in fact
exist somewhere (I think we also check that the correct number of type
parameters are supplied, in this case one). I would also like to check that `U`
satisfies the bound `B`.

Work on built-in bounds is (I think) in the process of adding this behaviour for
built-in bounds. I would like to apply this to user-specified bounds too.

I think no fewer programs can be expressed. That is, any errors we catch with
this new check would have been caught later in the existing scheme, where
exactly would depend on where the type was used. The only exception would be if
the formal type variable was not used.

We would allow bounds on type variable in structs and enums. Wherever a concrete
struct or enum type appears, check the actual type variables against the bounds
on the formals (the type well-formedness check).

From the above examples:

```
trait U {}
trait T<X: U> {}
trait S1<Y> {
    fn m(x: ~T<Y>) {}  //~ ERROR
}
trait S2<Y: U> {
    fn m(x: ~T<Y>) {}
}

struct St<X: U> {
    f: ~T<X>,
}

impl<X: U> St<X> {
    ...
}
```

# Alternatives

Keep the status quo.

We could add bounds on structs, etc. But not check them in impls. This is safe
since the implementation is more general than the struct. It would mean we allow
impls to be un-necessarily general.

# Unresolved questions

Do we allow and check bounds in type aliases? We currently do not. We should
probably continue not to since these type variables (and indeed the type
aliases) are substituted away early in the type checking process. So if we think
of type aliases as almost macro-like, then not checking makes sense. OTOH, it is
still a little bit inconsistent.
