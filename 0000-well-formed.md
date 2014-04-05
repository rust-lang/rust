- Start Date: 2014-04-05
- RFC PR #:
- Rust Issue #:

# Summary

Check all types for well-formedness with respect to the bounds of type variables.

# Motivation

Makes type checking saner. Catches errors earlier in the development process. Matches behaviour with built-in bounds (I think).

# Detailed design

Whenever a type is used it must be checked for well-formedness. For polymorphic types we currently check only that the type exists. I would like to also check that any actual type parameters are valid. That is, given a type `T<U>` where `T` is declared as `T<X: B>`, we currently only check that `T` does in fact exist somewhere (I think we also check that the correct number of type parameters are supplied, in this case one). I would also like to check that `U` satisfies the bound `B`.

Work on built-in bounds is (I think) in the process of adding this behaviour for built-in bounds. I would like to apply this to user-specified bounds too.

I think no fewer programs can be expressed. That is, any errors we catch with this new check would have been caught later in the existing scheme, where exactly would depend on where the type was used. The only exception would be if the formal type variable was not used.

# Alternatives

Keep the status quo.

# Unresolved questions

