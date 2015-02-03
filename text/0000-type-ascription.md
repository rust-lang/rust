- Start Date: 2015-2-3
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add type ascription to expressions and patterns.

Type ascription on expression has already been implemented. Type ascription on
patterns can probably wait until post-1.0.

See also discussion on [#354](https://github.com/rust-lang/rfcs/issues/354) and
[rust issue 10502](https://github.com/rust-lang/rust/issues/10502).


# Motivation

Type inference is imperfect. It is often useful to help type inference by
annotating a sub-expression or sub-pattern with a type. Currently, this is only
possible by extracting the sub-expression into a variable using a `let`
statement and/or giving a type for a whole expression or pattern. This is un-
ergonomic, and sometimes impossible due to lifetime issues. Specifically, a
variable has lifetime of its enclosing scope, but a sub-expression's lifetime is
typically limited to the nearest semi-colon.

Typical use cases are where a function's return type is generic (e.g., collect)
and where we want to force a coercion.

Type ascription can also be used for documentation and debugging - where it is
unclear from the code which type will be inferred, type ascription can be used
to precisely communicate expectations to the compiler or other programmers.

By allowing type ascription in more places, we remove the inconsistency that
type ascription is currently only allowed on top-level patterns.

## Examples:

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

In patterns:

```
struct Foo<T> { a: T, b: String }

// Current
fn foo(Foo { a, .. }: Foo<i32>) { ... }

// With type ascription.
fn foo(Foo { a: i32, .. }) { ... }
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

The syntax of sub-patterns is extended to include an optional type ascription.
Old syntax:

```
P ::= SP: T | SP
SP ::= var | 'box' SP | ...
```

where `P` is a pattern, `SP` is a sub-pattern, `T` is a type, and `var` is a
variable name.

New syntax:

```
P ::= SP: T | SP
SP ::= var | 'box' P | ...
```

Type ascription in patterns has the narrowest precedence, e.g., `box x: T` means
`box (x: T)`.

In type checking, if an expression is matched against a pattern, when matching
a sub-pattern the matching sub-expression must have the ascribed type (again,
this check includes subtyping and implicit coercion). Types in patterns play no
role at runtime.

@eddyb has implemented the expressions part of this RFC,
[PR](https://github.com/rust-lang/rust/pull/21836).


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
Patterns can be given a type as a whole rather than annotating a part of the
pattern.

We could allow type ascription in expressions but not patterns. This is a
smaller change and addresses most of the motivation.

Rely on explicit coercions - the current plan [RFC 401](https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md)
is to allow explicit coercion to any valid type and to use a customisable lint
for trivial casts (that is, those given by subtyping, including the identity
case). If we allow trivial casts, then we could always use explicit coercions
instead of type ascription. However, we would then lose the distinction between
implicit coercions which are safe and explicit coercions, such as narrowing,
which require more programmer attention. This also does not help with patterns.


# Unresolved questions

Is the suggested precedence correct? Especially for patterns.

Does type ascription on patterns have backwards compatibility issues?

