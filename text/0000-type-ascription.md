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
let x: T foo(): U<_>;
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

The syntax of patterns is extended to include an optional type ascription.
Old syntax:

```
PT ::= P: T
P ::= var | 'box' P | ...
e ::= 'let' (PT | P) = ... | ...
```

where `PT` is a pattern with optional type, `P` is a sub-pattern, `T` is a type,
and `var` is a variable name. (Formal arguments are `PT`, patterns in match arms
are `P`).

New syntax:

```
PT ::= P: T | P
P ::= var | 'box' PT | ...
e ::= 'let' PT = ... | ...
```

Type ascription in patterns has the narrowest precedence, e.g., `box x: T` means
`box (x: T)`. In particular, in a struct initialiser or patter, `x : y : z` is
parsed as `x : (y: z)`, i.e., a field named `x` is initialised with a value `y`
and that value must have type `z`. If only `x: y` is given, that is considered
to be the field name and the field's contents, with no type ascription.

The chagnes to pattern syntax mean that in some contexts where a pattern
previously required a type annotation, it is no longer required if all variables
can be assigned types via the ascription. Examples,

```
struct Foo {
    a: Bar,
    b: Baz,
}
fn foo(x: Foo); // Ok, type of x given by type of whole pattern
fn foo(Foo { a: x, b: y}: Foo) // Ok, types of x and y found by destructuring
fn foo(Foo { a: x: Bar, b: y: Baz}) // Ok, no type annotation, but types given as ascriptions
fn foo(Foo { a: x: Bar, _ }) // Ok, we can still deduce the type of x and the whole argument
fn foo(Foo { a: x, b: y}) // Ok, type of x and y given by Foo

struct Qux<X> {
    a: Bar,
    b: X,
}
fn foo(x: Qux<Baz>); // Ok, type of x given by type of whole pattern
fn foo(Qux { a: x, b: y}: Qux<Baz>) // Ok, types of x and y found by destructuring
fn foo(Qux { a: x: Bar, b: y: Baz}) // Ok, no type annotation, but types given as ascriptions
fn foo(Qux { a: x: Bar, _ }) // Error, can't find the type of the whole argument
fn foo(Qux { a: x, b: y}) // Error can't find type of y or the whole argument
```

Note the above changes mean moving some errors from parsing to later in type
checking. For example, all uses of patterns have optional types, and it is a
type error if there must be a type (e.g., in function arguments) but it is not
fully specified (currently it would be a parsing error).

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

We could use a different symbol or keyword instead of `:`, e.g., `is`.

# Unresolved questions

Is the suggested precedence correct? Especially for patterns.

Does type ascription on patterns have backwards compatibility issues?

Given the potential confusion with struct literal syntax, it is perhaps worth
re-opening that discussion. But given the timing, probably not.

Should remove integer suffixes in favour of type ascription?

### `as` vs `:`

A downside of type ascription is the overlap with explicit coercions (aka casts,
the `as` operator). Type ascription makes implicit coercions explicit. In RFC
401, it is proposed that all valid implicit coercions are valid explicit
coercions. However, that may be too confusing for users, since there is no
reason to use type ascription rather than `as` (if there is some coercion). It
might be a good idea to revisit that decision (it has not yet been implemented).
Then it is clear that the user uses `as` for explicit casts and `:` for non-
coercing ascription and implicit casts. Although there is no hard guideline for
which operations are implicit or explicit, the intuition is that if the
programmer ought to be aware of the change (i.e., the invariants of using the
type change to become less safe in any way) then coercion should be explicit,
otherwise it can be implicit.

Alternatively we could remove `as` and require `:` for explicit coercions, but
not for implicit ones (they would keep the same rules as they currently have).
The only loss would be that `:` doesn't stand out as much as `as` and there
would be no lint for trivial coercions. Another (backwards compatible)
alternative would be to keep `as` and `:` as synonyms and recommend against
using `as`.
