- Start Date: 2015-2-3
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)
- Feature: `ascription`

# Summary

Add type ascription to expressions. (An earlier version of this RFC covered type
ascription in patterns too, that has been postponed).

Type ascription on expression has already been implemented.

See also discussion on [#354](https://github.com/rust-lang/rfcs/issues/354) and
[rust issue 10502](https://github.com/rust-lang/rust/issues/10502).


# Motivation

Type inference is imperfect. It is often useful to help type inference by
annotating a sub-expression with a type. Currently, this is only possible by
extracting the sub-expression into a variable using a `let` statement and/or
giving a type for a whole expression or pattern. This is un- ergonomic, and
sometimes impossible due to lifetime issues. Specifically, where a variable has
lifetime of its enclosing scope, but a sub-expression's lifetime is typically
limited to the nearest semi-colon.

Typical use cases are where a function's return type is generic (e.g., collect)
and where we want to force a coercion.

Type ascription can also be used for documentation and debugging - where it is
unclear from the code which type will be inferred, type ascription can be used
to precisely communicate expectations to the compiler or other programmers.

By allowing type ascription in more places, we remove the inconsistency that
type ascription is currently only allowed on top-level patterns.

## Examples:

(Somewhat simplified examples, in these cases there are sometimes better
solutions with the current syntax).

Generic return type:

```
// Current.
let z = if ... {
    let x: Vec<_> = foo.enumerate().collect();
    x
} else {
    ...
};

// With type ascription.
let z = if ... {
    foo.enumerate().collect(): Vec<_>
} else {
    ...
};
```

Coercion:

```
fn foo<T>(a: T, b: T) { ... }

// Current.
let x = [1u32, 2, 4];
let y = [3u32];
...
let x: &[_] = &x;
let y: &[_] = &y;
foo(x, y);

// With type ascription.
let x = [1u32, 2, 4];
let y = [3u32];
...
foo(x: &[_], y: &[_]);
```

Generic return type and coercion:

```
// Current.
let x: T = {
    let temp: U<_> = foo();
    temp
};

// With type ascription.
let x: T = foo(): U<_>;
```


# Detailed design

The syntax of expressions is extended with type ascription:

```
e ::= ... | e: T
```

where `e` is an expression and `T` is a type. Type ascription has the same
precedence as explicit coercions using `as`.

When type checking `e: T`, `e` must have type `T`. The `must have type` test
includes implicit coercions and subtyping, but not explicit coercions. `T` may
be any well-formed type.

At runtime, type ascription is a no-op, unless an implicit coercion was used in
type checking, in which case the dynamic semantics of a type ascription
expression are exactly those of the implicit coercion.

@eddyb has implemented the expressions part of this RFC,
[PR](https://github.com/rust-lang/rust/pull/21836).

This feature should land behind the `ascription` feature gate.


### coercion and `as` vs `:`

A downside of type ascription is the overlap with explicit coercions (aka casts,
the `as` operator). To the programmer, type ascription makes implicit coercions
explicit (however, the compiler makes no distinction between coercions due to
type ascription and other coercions). In RFC 401, it is proposed that all valid
implicit coercions are valid explicit coercions. However, that may be too
confusing for users, since there is no reason to use type ascription rather than
`as` (if there is some coercion). Furthermore, if programmers do opt to use `as`
as the default whether or not it is required, then it loses its function as a
warning sign for programmers to beware of.

To address this I propose two lints which check for: trivial casts and trivial
numeric casts. Other than these lints we stick with the proposal from #401 that
unnecessary casts will no longer be an error.

A trivial cast is a cast `x as T` where `x` has type `U` and `x` can be
implicitly coerced to `T` or is already a subtype of `T`.

A trivial numeric cast is a cast `x as T` where `x` has type `U` and `x` is
implicitly coercible to `T` or `U` is a subtype of `T`, and both `U` and `T` are
numeric types.

Like any lints, these can be customised per-crate by the programmer. Both lints
are 'warn' by default.

Although this is a somewhat complex scheme, it allows code that works today to
work with only minor adjustment, it allows for a backwards compatible path to
'promoting' type conversions from explicit casts to implicit coercions, and it
allows customisation of a contentious kind of error (especially so in the
context of cross-platform programming).


### Type ascription and temporaries

There is an implementation choice between treating `x: T` as an lvalue or
rvalue. Note that when a rvalue is used in lvalue context (e.g., the subject of
a reference operation), then the compiler introduces a temporary variable.
Neither option is satisfactory, if we treat an ascription expression as an
lvalue (i.e., no new temporary), then there is potential for unsoundness:

```
let mut foo: S = ...;
{
    let bar = &mut (foo: T);  // S <: T, no coercion required
    *bar = ... : T;
}
// Whoops, foo has type T, but the compiler thinks it has type S, where potentially T </: S
```

If we treat ascription expressions as rvalues (i.e., create a temporary in
lvalue position), then we don't have the soundness problem, but we do get the
unexpected result that `&(x: T)` is not in fact a reference to `x`, but a
reference to a temporary copy of `x`.

The proposed solution is that type ascription expressions are rvalues, but
taking a reference of such an expression is forbidden. I.e., type asciption is
forbidden in the following contexts (where `<expr>` is a type ascription
expression):

```
&[mut] <expr>
let ref [mut] x = <expr>
match <expr> { .. ref [mut] x .. => { .. } .. }
<expr>.foo() // due to autoref
```

Like other rvalues, type ascription would not be allowed as the lhs of assignment.

Note that, if type asciption is required in such a context, an lvalue can be
forced by using `{}`, e.g., write `&mut { foo: T }`, rather than `&mut (foo: T)`.


# Drawbacks

More syntax, another feature in the language.

Interacts poorly with struct initialisers (changing the syntax for struct
literals has been [discussed and rejected](https://github.com/rust-lang/rfcs/pull/65)
and again in [discuss](http://internals.rust-lang.org/t/replace-point-x-3-y-5-with-point-x-3-y-5/198)).

If we introduce named arguments in the future, then it would make it more
difficult to support the same syntax as field initialisers.


# Alternatives

We could do nothing and force programmers to use temporary variables to specify
a type. However, this is less ergonomic and has problems with scopes/lifetimes.

Rely on explicit coercions - the current plan [RFC 401](https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md)
is to allow explicit coercion to any valid type and to use a customisable lint
for trivial casts (that is, those given by subtyping, including the identity
case). If we allow trivial casts, then we could always use explicit coercions
instead of type ascription. However, we would then lose the distinction between
implicit coercions which are safe and explicit coercions, such as narrowing,
which require more programmer attention. This also does not help with patterns.

We could use a different symbol or keyword instead of `:`, e.g., `is`.


# Unresolved questions

Is the suggested precedence correct?

Should we remove integer suffixes in favour of type ascription?

Style guidelines - should we recommend spacing or parenthesis to make type
ascription syntax more easily recognisable?
