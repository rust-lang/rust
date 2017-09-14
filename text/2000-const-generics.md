- Feature Name: const_generics
- Start Date: 2017-05-01
- RFC PR: https://github.com/rust-lang/rfcs/pull/2000
- Rust Issue: https://github.com/rust-lang/rust/issues/44580

# Summary
[summary]: #summary

Allow types to be generic over constant values; among other things this will
allow users to write impls which are abstract over all array types.

# Motivation
[motivation]: #motivation

Rust currently has one type which is parametric over constants: the built-inf
array type `[T; LEN]`. However, because const generics are not a first class
feature, users cannot define their own types which are generic over constant
values, and cannot implement traits for all arrays.

As a result of this limitation, the standard library only contains trait
implementations for arrays up to a length of 32; as a result, arrays are often
treated as a second-class language feature. Even if the length of an array
might be statically known, it is more common to heap allocate it using a
vector than to use an array type (which has certain performance trade offs).

Const parameters can also be used to allow users to more naturally specify
variants of a generic type which are more accurately reflected as values,
rather than types. For example, if a type takes a name as a parameter for
configuration or other reasons, it may make more sense to take a `&'static str`
than take a unit type which provides the name (through an associated const or
function). This can simplify APIs.

Lastly, consts can be used as parameters to make certain values determined at
typecheck time. By limiting which values a trait is implemented over, the
orphan rules can enable a crate to ensure that only some safe values are used,
with the check performed at compile time (this is especially relevant to
cryptographic libraries for example).

# Detailed design
[design]: #detailed-design

Today, types in Rust can be parameterized by two kinds: types and lifetimes. We
will additionally allow types to be parameterized by values, so long as those
values can be computed at compile time. A single constant parameter must be of
a single, particular type, and can be validly substituted with any value of
that type which can be computed at compile time and the type meets the equality
requirements laid out later in this RFC.

(Exactly which expressions are evaluable at compile time is orthogonal to this
RFC. For our purposes we assume that integers and their basic arithmetic
operations can be computed at compile time, and we will use them in all
examples.)

## Glossary

* __Const (constant, const value):__ A Rust value which is guaranteed to be
fully evaluated at compile time. Unlike statics, consts will be inlined at
their use sites rather than existing in the data section of the compiled
binary.

* __Const parameter (generic const):__ A const which a type or function is
abstract over; this const is input to the concrete type of the item, such as
the length parameter of a static array.

* __Associated const:__ A const associated with a trait, similar to an
associated type. Unlike a const parameter, an associated const is *determined*
by a type.

* __Const variable:__ Either a const parameter or an associated const,
contrast with concrete const; a const which is undetermined in this context
(prior to monomorphization).

* __Concrete const:__ In contrast to a const variable, a const which has a
known and singular value in this context.

* __Const expression:__ An expression which evaluates to a const. This may be
an identity expression or a more complex expression, so long as it can be
evaluated by Rust's const system.

* __Abstract const expression:__ A const expression which involves a const
variable (and therefore the value that it evaluates to cannot be determined
until after monomorphization).

* __Const projection:__ The value of an abstract const expression (which cannot
be determined in a generic context because it is dependent on a const
variable).

* __Identity expression:__ An expression which cannot be evaluated further
except by substituting it with names in scope. This includes all literals as
well all idents - e.g. `3`, `"Hello, world"`, `foo_bar`.

## Declaring a const parameter

In any sequence of type parameter declarations (such as in the definition of a
type or on the `impl` header of an impl block) const parameters can also be
declared. Const parameters declarations take the form `const $ident: $ty`:

```rust
struct RectangularArray<T, const WIDTH: usize, const HEIGHT: usize> {
    array: [[T; WIDTH]; HEIGHT],
}
```

The idents declared are the names used for these const parameters
(interchangeably called "const variables" in this RFC text), and all values
must be of the type ascribed to it. Which types can be ascribed to const
parameters is restricted later in this RFC.

The const parameter is in scope for the entire body of the item (type, impl,
function, method, etc) in which it is declared.

## Applying a const as a parameter

Any const expression of the type ascribed to a const parameter can be applied
as that parameter. When applying an expression as const parameter (except for
arrays), which is not an identity expression, the expression must be contained
within a block. This syntactic restriction is necessary to avoid requiring
infinite lookahead when parsing an expression inside of a type.

```rust
const X: usize = 7;

let x: RectangularArray<i32, 2, 4>;
let y: RectangularArray<i32, X, {2 * 2}>;
```

### Arrays
Arrays have a special construction syntax: `[T; CONST]`. In array syntax,
braces are not needed around any const expressions; `[i32; N * 2]` is a
syntactically valid type.

## When a const variable can be used

A const variable can be used as a const in any of these contexts:

1. As an applied const to any type which forms a part of the signature of
the item in question: `fn foo<const N: usize>(arr: [i32; N])`.
2. As part of a const expression used to define an associated const, or as a
parameter to an associated type.
3. As a value in any runtime expression in the body of any functions in the
item.
4. As a parameter to any type used in the body of any functions in the item,
as in `let x: [i32; N]` or `<[i32; N] as Foo>::bar()`.
5. As a part of the type of any fields in the item (as in
`struct Foo<const N: usize>([i32; N]);`).

In general, a const variable can be used where a const can. There is one
significant exception: const variables cannot be used in the construction of
consts, statics, functions, or types inside a function body. That is, these
are invalid:

```rust
fn foo<const X: usize>() {
    const Y: usize = X * 2;
    static Z: (usize, usize)= (X, X);

    struct Foo([i32; X]);
}
```

This restriction can be analogized to the restriction on using type variables
in types constructed in the body of functions - all of these declarations,
though private to this item, must be independent of it, and do not have any
of its parameters in scope.

## Theory of equality for type equality of two consts

During unification and the overlap check, it is essential to determine when two
types are equivalent or not. Because types can now be dependent on consts, we
must define how we will compare the equality of two constant expressions.

For most cases, the equality of two consts follows the same reasoning you would
expect - two constant values are equal if they are equal to one another. But
there are some particular caveats.

### Structural equality

Const equality is determined according to the definition of structural equality
defined in [RFC 1445][1445]. Only types which have the "structural match"
property can be used as const parameters. This would exclude floats, for
example.

The structural match property is intended as a stopgap until a final solution
for matching against consts has been arrived at. It is important for the
purposes of type equality that whatever solution const parameters use will
guarantee that the equality is *reflexive*, so that a type is always the same
type as itself. (The standard definition of equality for floating point numbers
is not reflexive.)

This may diverge someday from the definition used by match; it is not necessary
that matching and const parameters use the same definition of equality, but the
definition of equality used by match today is good enough for our purposes.

Because consts must have the structural match property, and this property
cannot be enforced for a type variable, it is not possible to introduce a const
parameter which is ascribed to a type variable (`Foo<T, const N: T>` is not
valid).

### Equality of two abstract const expressions

When comparing the equality of two abstract const expressions (that is, those
that depend on a variable) we cannot compare the equality of their values
because their values are determined by a const variable, the value of which is
unknown prior to monomorphization.

For this reason we will (initially, at least) treat the return value of const
expressions as *projections* - values determined by the input, but which are
not themselves known. This is similar to how we treat associated types today.
When comparing the evaluation of an abstract const expression - which we'll
call a *const projection* - to another const of the same type, its equality is
always unknown.

Each const expression generates a new projection, which is inherently
anonymous. It is not possible to unify two anonymous projections (imagine two
associated types on a generic - `T::Assoc` and `T::Item`: you can't prove or
disprove that they are the same type). For this reason, const expressions do
not unify with one another unless they are *literally references to the same
AST node*. That means that one instance of `N + 1` does not unify with another
instance of `N + 1` in a type.

To be clearer, this does not typecheck, because `N + 1` appears in two
different types:

```rust
fn foo<const N: usize>() -> [i32; N + 1] {
    let x: [i32; N + 1] = [0; N + 1];
    x
}
```

But this does, because it appears only once:

```rust
type Foo<const N: usize> = [i32; N + 1];

fn foo<const N: usize>() -> Foo<N> {
    let x: Foo<N> = Default::default();
    x
}
```

#### Future extensions

Someday we could introduce knowledge of the basic properties of some operations
- such as the commutativity of addition and multiplication - to begin making
smarter judgments on the equality of const projections. However, this RFC does
not proposing building any knowledge of that sort into the language and doing
so would require a future RFC.

## Specialization on const parameters

It is also necessary for specialization that const parameters have a defined
ordering of specificity. For this purpose, literals are defined as more
specific than other expressions, otherwise expressions have an indeterminate
ordering.

Just as we could some day support more advanced notions of equality between
const projections, we could some day support more advanced definitions of
specificity. For example, given the type `(i32, i32)`, we could determine that
`(0, PARAM2)` is more specific than `(PARAM1, PARAM2)` - roughly the analog
of understanding that `(i32, U)` is more specific than the type `(T, U)`. We
could also someday support intersectional and other more advanced definitions
of specialization on constants.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Const generics is a large feature, and will require significant educational
materials - it will need to be documented in both the book and the reference,
and will probably need its own section in the book. Documenting const generics
will be a big project in itself.

However, const generics should be treated as an advanced feature, and it should
not be something we expose to new users early in their use of Rust.

# Drawbacks
[drawbacks]: #drawbacks

This feature adds a significant amount of complexity to the type system,
allowing types to be determined by constants. It requires determining the rules
around abstract const equality, which result in surprising edge cases. It adds
a lot of syntax to the language. The language would definitely be simpler if we
don't adopt this feature.

However, we have already introduced a type which is determined by a constant -
the array type. Generalizing this feature seems natural and even inevitable
given that early decision.

# Alternatives
[alternatives]: #alternatives

There are not really alternatives other than not doing this, or staging it
differently.

We could limit const generics to the type `usize`, but this would not make the
implementation simpler.

We could move more quickly to more complex notions of equality between consts,
but this would make the implementation more complex up front.

We could choose a slightly different syntax, such as separating consts from
types with a semicolon.

# Unresolved questions
[unresolved]: #unresolved-questions

- **Unification of abstract const expressions:** This RFC performs the most
  minimal unification of abstract const expressions possible - it essentially
  doesn't unify them. Possibly this will be an unacceptable UX for
  stabilization and we will want to perform some more advanced unification
  before we stabilize this feature.
- **Well formedness of const expressions:** Types should be considered well
  formed only if during monomorphization they will not panic. This is tricky
  for overflow and out of bound array access. However, we can only actually
  provide well formedness constraints of expressions in the signature of
  functions; what to do about abstract const expressions appearing in function
  bodies in regards to well formedness is currently unclear & is delayed to
  implementation.
- **Ordering and default parameters:** Do all const parameters come last, or
  can they be mixed with types? Do all parameters with defaults have to come
  after parameters without defaults? We delay this decision to implementation
  of the grammar.

[1445]: https://github.com/rust-lang/rfcs/blob/master/text/1445-restrict-constants-in-patterns.md
