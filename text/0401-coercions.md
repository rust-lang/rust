- Start Date: 2014-10-30
- RFC PR #: https://github.com/rust-lang/rfcs/pull/401
- Rust Issue #: https://github.com/rust-lang/rust/issues/18469

# Summary

Describe the various kinds of type conversions available in Rust and suggest
some tweaks.

Provide a mechanism for smart pointers to be part of the DST coercion system.

Reform coercions from functions to closures.

The `transmute` intrinsic and other unsafe methods of type conversion are not
covered by this RFC.


# Motivation

It is often useful to convert a value from one type to another. This conversion
might be implicit or explicit and may or may not involve some runtime action.
Such conversions are useful for improving reuse of code, and avoiding unsafe
transmutes.

Our current rules around type conversions are not well-described. The different
conversion mechanisms interact poorly and the implementation is somewhat ad-hoc.

# Detailed design

Rust has several kinds of type conversion: subtyping, coercion, and casting.
Subtyping and coercion are implicit, there is no syntax. Casting is explicit,
using the `as` keyword. The syntax for a cast expression is:

```
e_cast ::= e as U
```

Where `e` is any valid expression and `U` is any valid type (note that we
restrict in type checking the valid types for `U`).

These conversions (and type equality) form a total order in terms of their
strength. For any types `T` and `U`, if `T == U` then `T` is also a subtype of
`U`. If `T` is a subtype of `U`, then `T` coerces to `U`, and if `T` coerces to
`U`, then `T` can be cast to `U`.

There is an additional kind of coercion which does not fit into that total order
- implicit coercions of receiver expressions. (I will use 'expression coercion'
when I need to distinguish coercions in non-receiver position from coercions of
receivers). All expression coercions are valid receiver coercions, but not all
receiver coercions are valid casts.

Finally, I will discuss function polymorphism, which is something of a coercion
edge case.

## Subtyping

Subtyping is implicit and can occur at any stage in type checking or inference.
Subtyping in Rust is very restricted and occurs only due to variance with
respect to lifetimes and between types with higher ranked lifetimes. If we were
to erase lifetimes from types, then the only subtyping would be due to type
equality.


## Coercions

A coercion is implicit and has no syntax. A coercion can only occur at certain
coercion sites in a program, these are typically places where the desired type
is explicit or can be dervied by propagation from explicit types (without type
inference). The base cases are:

* In `let` statements where an explicit type is given: in `let _: U = e;`, `e`
  is coerced to to have type `U`;

* In statics and consts, similarly to `let` statements;

* In argument position for function calls. The value being coerced is the actual
  parameter and it is coerced to the type of the formal parameter. For example,
  where `foo` is defined as `fn foo(x: U) { ... }` and is called with `foo(e);`,
  `e` is coerced to have type `U`;

* Where a field of a struct or variant is instantiated. E.g., where `struct Foo
  { x: U }` and the instantiation is `Foo { x: e }`, `e` is coerced to to have
  type `U`;

* The result of a function, either the final line of a block if it is not semi-
  colon terminated or any expression in a `return` statement. For example, for
  `fn foo() -> U { e }`, `e` is coerced to to have type `U`;

If the expression in one of these coercion sites is a coercion-propagating
expression, then the relevant sub-expressions in that expression are also
coercion sites. Propagation recurses from these new coercion sites. Propagating
expressions and their relevant sub-expressions are:

* array literals, where the array has type `[U, ..n]`, each sub-expression in
  the array literal is a coercion site for coercion to type `U`;

* array literals with repeating syntax, where the array has type `[U, ..n]`, the
  repeated sub-expression is a coercion site for coercion to type `U`;

* tuples, where a tuple is a coercion site to type `(U_0, U_1, ..., U_n)`, each
  sub-expression is a coercion site for the respective type, e.g., the zero-th
  sub-expression is a coercion site to `U_0`;

* the box expression, if the expression has type `Box<U>`, the sub-expression is
  a coercion site to `U` (I expect this to be generalised when `box` expressions
  are);

* parenthesised sub-expressions (`(e)`), if the expression has type `U`, then
  the sub-expression is a coercion site to `U`;

* blocks, if a block has type `U`, then the last expression in the block (if it
  is not semicolon-terminated) is a coercion site to `U`. This includes blocks
  which are part of control flow statements, such as `if`/`else`, if the block
  has a known type.


Note that we do not perform coercions when matching traits (except for
receivers, see below). If there is an impl for some type `U` and `T` coerces to
`U`, that does not constitute an implementation for `T`. For example, the
following will not type check, even though it is OK to coerce `t` to `&T` and
there is an impl for `&T`:

```
struct T;
trait Trait {}

fn foo<X: Trait>(t: X) {}

impl<'a> Trait for &'a T {}


fn main() {
    let t: &mut T = &mut T;
    foo(t); //~ ERROR failed to find an implementation of trait Trait for &mut T
}
```

In a cast expression, `e as U`, the compiler will first attempt to coerce `e` to
`U`, only if that fails will the conversion rules for casts (see below) be
applied.

Coercion is allowed between the following types:

* `T` to `U` if `T` is a subtype of `U` (the 'identity' case);

* `T_1` to `T_3` where `T_1` coerces to `T_2` and `T_2` coerces to `T_3`
  (transitivity case);

* `&mut T` to `&T`;

* `*mut T` to `*const T`;

* `&T` to `*const T`;

* `&mut T` to `*mut T`;

* `T` to `U` if `T` implements `CoerceUnsized<U>` (see below) and `T = Foo<...>`
  and `U = Foo<...>` (for any `Foo`, when we get HKT I expect this could be a
  constraint on the `CoerceUnsized` trait, rather than being checked here);

* From TyCtor(`T`) to TyCtor(coerce_inner(`T`)) (these coercions could be
  provided by implementing `CoerceUnsized` for all instances of TyCtor);

where TyCtor(`T`) is one of `&T`, `&mut T`, `*const T`, `*mut T`, or `Box<T>`.
And where coerce_inner is defined as

* coerce_inner(`[T, ..n]`) = `[T]`;

* coerce_inner(`T`) = `U` where `T` is a concrete type which implements the
  trait `U`;

* coerce_inner(`T`) = `U` where `T` is a sub-trait of `U`;

* coerce_inner(`Foo<..., T, ...>`) = `Foo<..., coerce_inner(T), ...>` where
  `Foo` is a struct and only the last field has type `T` and `T` is not part of
  the type of any other fields;

* coerce_inner(`(..., T)`) = `(..., coerce_inner(T))`.

Note that coercing from sub-trait to a super-trait is a new coercion and is non-
trivial. One implementation strategy which avoids re-computation of vtables is
given in RFC PR #250.

A note for the future: although there hasn't been an RFC nor much discussion, it
is likely that post-1.0 we will add type ascription to the language (see #354).
That will (probably) allow any expression to be annotated with a type (e.g,
`foo(a, b: T, c)` a function call where the second argument has a type
annotation).

Type ascription is purely descriptive and does not cast the sub-expression to
the required type. However, it seems sensible that type ascription would be a
coercion site, and thus type ascription would be a way to make implicit
coercions explicit. There is a danger that such coercions would be confused with
casts. I hope the rule that casting should change the type and type ascription
should not is enough of a discriminant. Perhaps we will need a style guideline
to encourage either casts or type ascription to force an implicit coercion.
Perhaps type ascription should not be a coercion site. Or perhaps we don't need
type ascription at all if we allow trivial casts.


### Custom unsizing coercions

It should be possible to coerce smart pointers (e.g., `Rc`) in the same way as
the built-in pointers. In order to do so, we provide two traits and an intrinsic
to allow users to make their smart pointers work with the compiler's coercions.
It might be possible to implement some of the coercions described for built-in
pointers using this machinery, whether that is a good idea or not is an
implementation detail.

```
// Cannot be impl'ed - it really is quite a magical trait, see the cases below.
trait Unsize<Sized? U> for Sized? {}
```

The `Unsize` trait is a marker trait and a lang item. It should not be
implemented by users and user implementations will be ignored. The compiler will
assume the following implementations, these correspond to the definition of
coerce_inner, above; note that these cannot be expressed in real Rust:

```
impl<T, n: int> Unsize<[T]> for [T, ..n] {}

// Where T is a trait
impl<Sized? T, U: T> Unsize<T> for U {}

// Where T and U are traits
impl<Sized? T, Sized? U: T> Unsize<T> for U {}

// Where T and U are structs ... following the rules for coerce_inner
impl Unsize<T> for U {}

impl Unsize<(..., T)> for (..., U)
    where U: Unsize(T) {}
```

The `CoerceUnsized` trait should be implemented by smart pointers and containers
which want to be part of the coercions system.

```
trait CoerceUnsized<U> {
    fn coerce(self) -> U;
}
```

To help implement `CoerceUnsized`, we provide an intrinsic -
`fat_pointer_convert`. This takes and returns raw pointers. The common case will
be to take a thin pointer, unsize the contents, and return a fat pointer. But
the exact behaviour depends on the types involved. This will perform any
computation associated with a coercion (for example, adjusting or creating
vtables). The implementation of fat_pointer_convert will match what the
compiler must do in coerce_inner as described above.

```
intrinsic fn fat_pointer_convert<Sized? T, Sized? U>(t: *const T) -> *const U
    where T : Unsize<U>;
```

Here is an example implementation of `CoerceUnsized` for `Rc`:

```
impl<Sized? T, Sized? U> CoerceUnsized<Rc<T>> for Rc<U> {
    where U: Unsize<T>

    fn coerce(self) -> Rc<T> {
        let new_ptr: *const RcBox<T> = fat_pointer_convert(self._ptr);
        Rc { _ptr: new_ptr }
    }
}
```

## Coercions of receiver expressions

These coercions occur when matching the type of the receiver of a method call
with the self type (i.e., the type of `e` in `e.m(...)`) or in field access.
These coercions can be thought of as a feature of the `.` operator, they do not
apply when using the UFCS form with the self argument in argument position. Only
an expression before the dot is coerced as a receiver. When using the UFCS form
of method call, arguments are only coerced according to the expression coercion
rules. This matches the rules for dispatch - dynamic dispatch only happens using
the `.` operator, not the UFCS form.

In method calls the target type of the coercion is the concrete type of the impl
in which the method is defined, modified by the type of `self`. Assuming the
impl is for `T`, the target type is given by:

 self             | target type
------------------|------------
 `self`           | `T`
 `&self`          | `&T`
 `&mut self`      | `&mut T`
 `self: Box<Self>`| `Box<T>`

and likewise with any variations of the self type we might add in the future.

For field access, the target type is `&T`, `&mut T` for field assignment,
where `T` is a struct with the named field.

A receiver coercion consists of some number of dereferences (either compiler
built-in (of a borrowed reference or `Box` pointer, not raw pointers) or custom,
given by the `Deref` trait), one or zero applications of `coerce_inner` or use
of the `CoerceUnsized` trait (as defined above, note that this requires we are
at a type which has neither references nor dereferences at the top level), and
up to two address-of operations (i.e., `T` to `&T`, `&mut T`, `*const T`, or
`*mut T`, with a fresh lifetime.). The usual mutability rules for taking a
reference apply. (Note that the implementation of the coercion isn't so simple,
it is embedded in the search for candidate methods, but from the point of view
of type conversions, that is not relevant).

Alternatively, a receiver coercion may be thought of as a two stage process.
First, we dereference and then take the address until the source type has the
same shape (i.e., has the same kind and number of indirection) as the target
type. Then we try to coerce the adjusted source type to the target type using
the usual coercion machinery. I believe, but have not proved, that these two
descriptions are equivalent.


## Casts

Casting is indicated by the `as` keyword. A cast `e as U` is valid if one of the
following holds:

* `e` has type `T` and `T` coerces to `U`; *coercion-cast*
* `e` has type `*T`, `U` is `*U_0`, and either `U_0: Sized` or
   unsize_kind(`T`) = unsize_kind(`U_0`); *ptr-ptr-cast*
* `e` has type `*T` and `U` is a numeric type, while `T: Sized`; *ptr-addr-cast*
* `e` has type `usize` and `U` is `*U_0`, while `U_0: Sized`; *addr-ptr-cast*
* `e` has type `T` and `T` and `U` are any numeric types; *numeric-cast*
* `e` is a C-like enum and `U` is an integer type or `bool`; *enum-cast*
* `e` has type `bool` and `U` is an integer; *bool-cast*
* `e` has type `u8` and `U` is `char`; *u8-char-cast*
* `e` has type `&.[T; n]` and `U` is `*T`, and `e` is a mutable
  reference if `U` is. *array-ptr-cast*
* `e` is a function pointer type and `U` has type `*T`,
  while `T: Sized`; *fptr-ptr-cast*
* `e` is a function pointer type and `U` is an integer; *fptr-addr-cast*

where `&.T` and `*T` are references of either mutability,
and where unsize_kind(`T`) is the kind of the unsize info
in `T` - a vtable or a length (or `()` if `T: Sized`).

Casting is not transitive, that is, even if `e as U1 as U2` is a valid
expression, `e as U2` is not necessarily so (in fact it will only be valid if
`U1` coerces to `U2`).

A cast may require a runtime conversion.

There will be a lint for trivial casts. A trivial cast is a cast `e as T` where
`e` has type `U` and `U` is a subtype of `T`. The lint will be warn by default.


## Function type polymorphism

Currently, functions may be used where a closure is expected by coercing a
function to a closure. We will remove this coercion and instead use the
following scheme:

* Every function item has its own fresh type. This type cannot be written by the
  programmer (i.e., it is expressible but not denotable).
* Conceptually, for each fresh function type, there is an automatically generated
  implementation of the `Fn`, `FnMut`, and `FnOnce` traits.
* All function types are implicitly coercible to a `fn()` type with the
  corresponding parameter types.
* Conceptually, there is an implementation of `Fn`, `FnMut`, and `FnOnce` for
  every `fn()` type.
* `Fn`, `FnMut`, or `FnOnce` trait objects and references to type parameters
  bounded by these traits may be considered to have the corresponding unboxed
  closure type. This is a desugaring (alias), rather than a coercion. This is
  an existing part of the unboxed closures work.

These steps should allow for functions to be stored in variables with both
closure and function type. It also allows variables with function type to be
stored as a variable with closure type. Note that these have different
dynamic semantics, as described below. For example,

```
fn foo() { ... }         // `foo` has a fresh and non-denotable type.

fn main() {
    let x: fn() = foo;   // `foo` is coerced to `fn()`.
    let y: || = x;       // `x` is coerced to `&Fn` (a closure object),
                         // legal due to the `fn()` auto-impls.

    let z: || = foo;     // `foo` is coerced to `&T` where `T` is fresh and
                         // bounded by `Fn`. Legal due to the fresh function
                         // type auto-impls.
}
```

The two kinds of auto-generated impls are rather different: the first case (for
the fresh and non-denotable function types) is a static call to `Fn::Call`,
which in turn calls the function with the given arguments. The first call would
be inlined (in fact, the impls and calls to them may be special-cased by the
compiler). In the second case (for `fn()` types), we must execute a virtual call
to find the implementing method and then call the function itself because the
function is 'wrapped' in a closure object.


## Changes required

* Add cast from unsized slices to raw pointers (`&[V] to *V`);

* allow coercions as casts and add lint for trivial casts;

* ensure we support all coercion sites;

* remove [T, ..n] to &[T]/*[T] coercions;

* add raw pointer coercions;

* add sub-trait coercions;

* add unsized tuple coercions;

* add all transitive coercions;

* receiver coercions - add referencing to raw pointers, remove triple
  referencing for slices;

* remove function coercions, add function type polymorphism;

* add DST/custom coercions.


# Drawbacks

We are adding and removing some coercions. There is always a trade-off with
implicit coercions on making Rust ergonomic vs making it hard to comprehend due
to magical conversions. By changing this balance we might be making some things
worse.


# Alternatives

These rules could be tweaked in any number of ways.

Specifically for the DST custom coercions, the compiler could throw an error if
it finds a user-supplied implementation of the `Unsize` trait, rather than
silently ignoring them.

# Unresolved questions
