- Feature Name: `infer_outlives`
- Start Date: 2017-08-02
- RFC PR: https://github.com/rust-lang/rfcs/pull/2093
- Rust Issue: https://github.com/rust-lang/rust/issues/44493

# Summary
[summary]: #summary

Remove the need for explicit `T: 'x` annotations on structs. We will
infer their presence based on the fields of the struct. In short, if
the struct contains a reference, directly or indirectly, to `T` with
lifetime `'x`, then we will infer that `T: 'x` is a requirement:

```rust
struct Foo<'x, T> {
  // inferred: `T: 'x`
  field: &'x T
}  
```

Explicit annotations remain as an option used to control trait object
lifetime defaults, and simply for backwards compatibility.

# Motivation
[motivation]: #motivation

Today, when you write generic struct definitions that contain
references, those structs require where-clauses of the form `T:
'a`:

```rust
struct SharedRef<'a, T>
  where T: 'a // <-- currently required
{
  data: &'a T
}
```

These clauses are called *outlives requirements*, and the next section
("Background") goes into a bit more detail on what they mean
semantically.  **The overriding goal of this RFC is to make these
`where T: 'a` annotations unnecessary by inferring them.**

Anecdotally, these annotations are not well understood. Instead, the
most common thing is to wait and add the where-clauses when the
compiler requests that you do so. This is annoying, of course, but the
annotations also clutter up the code, and add to the perception of
Rust's complexity.

Experienced Rust users may have noticed that the compiler already
performs a similar seeming kind of inference in other settings. In
particular, in function definitions or impls, outlives requirements
are rarely needed. This is due to the mechanism of known as *implied
bounds* (also explained in more detail in the next section), which
allows a function (resp. impl) to infer outlives requirements based on
the types of its parameters (resp. input types):

```rust
fn foo<'a, T>(r: SharedRef<'a, T>) {
  // Gets to assume that `T: 'a` holds, because it is a requirement
  // of the parameter type `SharedRef<'a, T>`.
}  
```

This RFC proposes a mechanism for also inferring the outlives
requirements on structs. This is not an extension of the implied
bounds system; in general, field types of a struct are not considered
"inputs" to the struct definition, and hence implied bounds do not
apply. Indeed, the annotations that we are attempting to infer are
used to drive the implied bounds system. Instead, to infer these
outlives requirements on structs, we will use a specialized,
fixed-point inference similar to [variance inference].

[variance inference]: https://github.com/rust-lang/rfcs/blob/master/text/0738-variance.md

There is one other, relatively obscure, place where explicit lifetime
annotations are used today: trait object lifetime defaults
([RFC 599][]). The interaction there is discussed in the Guide-Level
Explanation below.

[RFC 599]: https://github.com/rust-lang/rfcs/blob/master/text/0599-default-object-bound.md

### Background: outlives requirements today

[RFC 34][] established the current rules around "outlives
requirements". Specifically, in order for a reference type `&'a T` to
be "well formed" (valid), the compiler must know that the type `T`
"outlives" the lifetime `'a` -- meaning that all references contained
in the type `T` must be valid for the lifetime `'a`. So, for example,
the type `i32` outlives any lifetime, including `'static`, since it
has no references at all. (The "outlives" rules were later tweaked by
[RFC 1214][] to be more syntactic in nature.)

[RFC 34]: https://github.com/nikomatsakis/rfcs/blob/master/text/0034-bounded-type-parameters.md
[RFC 1214]: https://github.com/rust-lang/rfcs/blob/master/text/1214-projections-lifetimes-and-wf.md

In practice, this means that in Rust, when you define a struct that
contains references to a generic type, or references to other
references, you need to add various where clauses for that struct type
to be considered valid. For example, consider the following (currently invalid)
struct `SharedRef`:

```rust
struct SharedRef<'a, T> {
  data: &'a T
}
```    

In general, for a struct definition to be valid, its field types must be
known to be well-formed, based only on the struct's where-clauses. In this case,
the field `data` has the `&'a T` -- for that to be well-formed, we must know that
`T: 'a` holds. Since we do not know what `T` is, we require that a where-clause be
added to the struct header to assert that `T: 'a` must hold:

```rust
struct SharedRef<'a, T>
  where T: 'a // currently required...
{
  data: &'a T // ...so that we know that this field's type is well-formed
}
```

In principle, similar where clauses would be required on generic
functions or impl to ensure that their parameters or inputs are
well-formed.  However, as you may have noticed, this is not the
case. For example, the following function is valid as written:

```rust
fn foo<'a, T>(x: &'a T) {
  ..
}  
```

This is due to Rust's support for **implied bounds** -- in particular,
every function and impl **assumes** that the types of its inputs are
well-formed. In this case, since `foo` can assume that `&'a T` is
well-formed, it can also deduce that `T: 'a` must hold, and hence we
do not require where-clauses asserting this fact. (Currently, implied
bounds are only used for lifetime requirements; pending [RFC 2089]
proposes to extend this mechanism to other sorts of bounds.)

[RFC 2089]: https://github.com/rust-lang/rfcs/pull/2089

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

This RFC does not introduce any new concepts -- rather, it (mostly)
removes the need to be actively aware of outlives requirements. In
particular, the compiler will infer the `T: 'a` requirements on behalf
of the programmer.  Therefore, the `SharedRef` struct we have seen in
the previous section would be accepted without any annotation:

```rust
struct SharedRef<'a, T> {
    r: &'a T
}
```

The compiler would infer that `T: 'a` must hold for the type
`SharedRef<'a, T>` to be valid. In some cases, the requirement may be
inferred through several structs. So, for the struct `Indirect` below,
we would also infer that `T: 'a` is required, because `Indirect` contains
a `SharedRef<'a, T>`:

```rust
struct Indirect<'a, T> {
  r: SharedRef<'a, T>
}
```

### Where explicit annotations would still be required

Explicit outlives annotations would primarily be required in cases
where the lifetime and the type are combined within the value of an
associated type, but not in one of the impl's input types. For
example:

```
trait MakeRef<'a> {
  type Type;
}

impl<'a, T> MakeRef<'a> for Vec<T>
  where T: 'a // still required
{
  type Type = &'a T;
}
```

In this case, the impl has two inputs -- the lifetime `'a` and the
type `Vec<T>` (note that `'a` and `T` are the impl parameters; the
inputs come from the parameters of the trait that is being
implemented). Neither of these inputs requires that `T: 'a`. So, when
we try to specify the value of the associated type as `&'a T`, we
still require a where clause to infer that `T: 'a` must hold.

In turn, if this associated type were used in a struct, where-clauses
would be required. As we'll see in the reference-level explanation,
this is a consequence of the fact that we do inference without regard
for associated type normalization, but it makes for a relatively
simple rule -- explicit where clauses are needed in the preseence of
impls like the one above:

```rust
struct Foo<'a, T>
  where T: 'a // still required, not inferred from `field`
{
  field: <Vec<T> as MakeRef<'a>>::Type
}    
```

As the algorithm is currently framed, outlives requirements written on
traits must also be explicitly propagated; however, this will typically
occur as part of the existing bounds:

```rust
trait Trait<'a> where Self: 'a {
  type Type;
}

struct Foo<'a, T>
  where T: Trait<'a> // implies `T: 'a` already, so no error
{
  r: <T as Trait<'a>>::Type // requires that `T: 'a` to be WF
}
```

### Trait object lifetime defaults

[RFC 599][] (later amended by [RFC 1156]) specified the defaulting
rules for trait object types. Typically, a trait object type that
appears as a parameter to a struct is given the implicit bound
`'static`; hence `Box<Debug>` defaults to `Box<Debug +
'static>`. References to trait objects, however, are given by default
the lifetime of the reference; hence `&'a Debug` defaults to `&'a
(Debug + 'a)`.

Structs that contain explicit `T: 'a` where-clauses, however, use the
default given lifetime `'a` as the default for trait objects.
Therefore, given a struct definition like the following:

```rust
struct Ref<'a, T> where T: 'a + ?Sized { .. }
```

The type `Ref<'x, Debug>` defaults to `Ref<'x, Debug + 'x>` and not
`Ref<'x, Debug + 'static>`. Effectively the `where T: 'a` declaration
acts as a kind of signal that `Ref` acts as a "reference to `T`".

This RFC does not change these defaulting rules. In particular, these
defaults are applied **before** where-clause inference takes place,
and hence are not affected by the results. Trait object defaulting
therefore requires an explicit `where T: 'a` declaration on the
struct; in fact, such explicit declarations can be thought of as
existing primarily for the purpose of informing trait object lifetime
defaults, since they are typically not needed otherwise.

[RFC 1156]: https://github.com/rust-lang/rfcs/blob/master/text/1156-adjust-default-object-bounds.md

### Long-range errors, and why they are considered unlikely

Initially, we avoided inferring the `T: 'a` annotations on struct
types in part out of a fear of "long-range" error messages, where it
becomes hard to see the origin of an outlives requirement.  Consider
for example a setup like this one:

```rust
struct Indirect<'a, T> {
  field: Direct<'a, T>
}

struct Direct<'a, T> {
  field: &'a T
}
```

Here, both of these structs require that `T: 'a`, but the requirement
is not written explicitly. If you have access to the full definition
of `Direct`, it might be obvious that the requirement arises from the
`&'a T` type, but discovering this for `Indirect` requires looking
deeply into the definitions of all types that it references.

In principle, such errors can occur, but there are many reasons to
believe that "long-range errors" will not be a source of problems in
practice:

- The inferred bounds approach ensures that code that is given (e.g.,
  as a parameter) an existing `Indirect` or `Direct` value will
  already be able to assume the required outlives relationship holds.
- Code that creates an `Indirect` or `Direct` value must also create
  the `&'a T` reference found in `Direct`, and creating *that* reference 
  would only be legal if `T: 'a`.
  
Put another way, think back on your experience writing Rust code: how
often do you get an error that is solved by writing `where T: 'a` or
`where 'a: 'b` **outside of a struct definition**? At least in the
author's experience, such errors are quite infrequent.

That said, long-range errors *can* still occur, typically around impls
and associated type values, as mentioned in the previous section. For example,
the following impl would not compile:

```rust
trait MakeRef<'a> {
  type Type;
}

impl<'a, T> MakeRef<'a> for Vec<T> {
  type Type = Indirect<'a, T>;
}
```

Here, we would be missing a where-clause that `T: 'a` due to the type
`Indirect<'a, T>`, just as we saw in the previous section. In such
cases, tweaking the wording of the error could help to make the cause
clearer. Similarly to auto traits, the idea would be to help trace the
path that led to the `T: 'a` requirement on the user's behalf:

```
error[E0309]: the type `T` may not live long enough
 --> src/main.rs:6:3
   |
 6 |   type Type = Indirect<'a, T>;
   |   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the type `Indirect<'a, T>` requires that `T: 'a`
   |
   = note: `Indirect<'a, T>` requires that `T: 'a` because it contains a field of type `Direct<'a, T>`
   = note: `Direct<'a, T>` requires that `T: 'a` because it contains a field of type `&'a T`
```

### Impact on semver

Due to the implied bounds rules, it is currently the case that
removing `where T: 'a` annotations is potentially a breaking
change. After this RFC, the rule is a bit more subtle: removing an
annotation is still potentially a breaking change (even if it would be
inferred), due to the trait object rules; but also, adding or removing
a field of type `&'a T` could affect the results of inference, and
hence may be a breaking change. As an example, consider a struct like
the following:

```rust
struct Iter<'a, T> {
  vec: &'a Vec<T> // Implies: `T: 'a`
}
```

Now imagine a function that takes `Iter` as an argument:

```rust
fn foo<'a, T>(iter: Iter<'a, T>) { .. }
```

Under this RFC, this function can assume that `T: 'a` due to the
implied bounds of its parameter type. But if `Iter<'a, T>` were
changed to (e.g.) remove the field `vec`, then it may no longer
require that `T: 'a` holds, and hence `foo()` would no longer have the
implied bound that `T: 'a` holds.

This situation is considerd unlikely: typically, if a struct has a
lifetime parameter (such as the `Iter` struct), then the fact that
it contains (or may contain) a borrowed reference is rather
fundamental to how it works. If that borrowed refernce were to be
removed entirely, then the struct's API will likely be changing in
other incompatible ways, since that implies that the struct is now
taking ownership of data it used to borrow (or else has access to less
data than it did before).

**Note:** This is not the only case where changes to private field
types can cause downstream errors: introducing object types can
inhibit auto traits like `Send` and `Sync`. What these have in common
is that they are both entangled with Rust's memory safety checking. It
is commonly observed that parallelim is anti-encapsulation, in that,
to know if two bits of code can be run in parallel, you must know what
data they access, but for the strongest encapsulation, you wish to
hide that fact. Memory safety has a similar property: to guarantee
that references are always valid, we need to know where they appear,
even if it is deeply nested within a struct hierarchy. Probably the
best way to mitigate these sorts of subtle semver complications is to
have a tool that detects and warns for incompatible changes.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The intention is that the outlives inference takes place at the same
time in the compiler pipeline as variance inference. In particular,
this is after the point where we have been able to construct
"semantics" or "internal" types from the HIR (so we don't have to
define the inference in a purely syntactic fashion). However, this is
still relatively early, so we wish to avoid doing things like solving
traits. Like variance inference, the new inference is an iterative
algorithm that continues to infer additional requirements until a
fixed point is reached.

For each struct declared by the user, we will infer a set of implicit
outlives annotations. These annotations take one of several forms:

- `'a: 'b` -- two lifetimes (typically parameters of the trait) are
  required to outlive one another
- `T: 'a` -- a type parameter `T` of the trait is required to outlive
  the lifetime `'a`, which is either a parameter of the trait or `'static`
- `<T as Trait<..>>::Item: 'a` -- the value of an associated type is
  required to outlive the lifetime `'a`, which is either a parameter
  of the trait or `'static` (here `T` represents an arbitrary type).

We will infer a minimal set of annotations `A[S]` for each struct `S`.
This set must meet the constraints derived by the following algorithm.

First, if the struct contains a where-clause `C` matching the above
forms, then we add the constraint that `C in A[S]`. So, for example,
in the following struct:

```rust
struct Foo<'a, T> where T: 'a { .. }
```

we would add the constraint that `(T: 'a) in A[S]`.

Next, for each field `f` of type `T_f` of the struct `S`, we derive
each outlives requirement that is needed for `T_f` to be well-formed
and require that those be included in `A[S]`. **This is done on the
unnormalized type `T_f`**.  These rules can be derived in a fairly
straightforward way from the inference rules given in [RFC 1214][]. We
won't give an exhaustive accounting of the rules, but will just note
the outlines of the algorithm:

- A field containing a reference type like `&'a T` naturally requires
  that `T: 'a` must be satisfied (here `T` represents "some type" and
  not necessarily a type parameter; for example, `&'a &'b i32` would
  lead to the outlives requirement that `'b: 'a`).
- A reference to a struct like `Foo<'a, T>` may also require outlives
  requirements. This is determined by checking the (current) value of
  `A[Foo]`, after substituting its parameters.
- For an associated type reference like `<T as BarTrait<'a>>::Type`, we do
  not attempt normalization, but rather just check that `T` is well-formed.
  - This is partly looking forward to a time when, at this stage, we
    may not know which trait is being projected from (in the compiler
    as currently implemented, we already do).
  - Note that we do not infer additional requirements on traits, we simply
    use the values given by users.
  - Note further that where-clauses declared on impls are never relevant here.

Once inference is complete, the implicit outlives requirements
inferred as part of `A` become part of the predicates on the struct
for all intents and purposes after this point.

Note that inference is not "complete" -- i.e., it is not guaranteed to
find all the outlives requirements that are ultimately required (in
particular, it does not find those that arise through
normalization). Furthermore, it only covers outlives requirements, and
not other sorts of well-formedness rules (e.g., trait requirements
like `T: Eq`). Therefore, after inference completes, we still check
that each type is well-formed just as today, but with the inferred
outlives requirements in scope.

### Example 1: A reference

The simplest example is one where we have a reference type directly
contained in the struct:

```rust
struct Foo<'a, T> {
  bar: &'a [T]
}
```

Here, the reference type requires that `[T]: 'a` which in turn is true
if `T: 'a`. Hence we will create a single constraint, that `(T: 'a) in
A[Foo]`.

### Example 2: Projections

In some cases, the outlives requirements are not of the form `T: 'a`,
as in this example:

```rust
struct Foo<'a, T: Iterator> {
  bar: &'a T::Item
}
```

Here, the requirement will be that `<T as Iterator>::Item: 'a`.

### Example 3: Explicit where-clauses

In some cases, we may have constraints that arise from explicit where-clauses
and not from field types, as in the following example:

```rust
struct Foo<'b, U> {
  bar: Bar<'b, U>
}

struct Bar<'a, T> where T: 'a {
  x: &'a (),
  y: T
}
```

Here, `Bar` is declared with the where clause that `T: 'a`. This
results in the requirement that `(T: 'a) in A[Bar]`. `Foo`, meanwhile,
requires that any outlives requirements for `Bar<'b, U>` are
satisfied, and hence as the rule that `('a => 'b, T => U) (A[Bar]) <=
A[Foo]`. The minimal solution to this is:

- `A[Foo] = (U: 'b)`
- `A[Bar] = (T: 'a)`

This means that we would infer an implicit outlives requirements of
`U: 'b` for `Foo`; for `Bar` we would infer `T: 'a` but that was
explicitly declared.

### Example 4: Normalization or lack thereof

Let us revisit the case where the where-clause is due
to an impl:

```rust
trait MakeRef<'a> {
  type Type;
}

impl<'a, T> MakeRef<'a> for Vec<T>
  where T: 'a
{
  type Type = &'a T;
}

struct Foo<'a, T> { // Results in an error
  foo: <Vec<T> as MakeRef<'a>::Type
}
```

Here, for the struct `Foo<'a, T>`, we will in fact create no
constraints for its where-clause set, and hence we will infer an empty
set. This is because we encounter the field type `<Vec<T> as
MakeRef<'a>>::Type`, and in such a case we ignore the trait reference
itself and just require that `Vec<T>` is well-formed, which does not
result in any outlives requirements as it contains no references. 

Now, when we go to check the full well-formedness rules for `Foo`, we will
get an error -- this is because, in that context, we will try to normalize
the associated type reference, but we will fail in doing so because we do not
have any where-clause stating that `T: 'a` (which the impl requires).

### Example 5: Multiple regions

Sometimes the outlives relationship can be inferred between multiple
regions, not only type parameters. Consider the following:

```rust
struct Foo<'a,'b,T> {
    x: &'a &'b T
}
```

Here the WF rules for the type `&'a &'b T` require that both:

- `'b: 'a` holds, because of the outer reference; and,
- `T: 'b` holds, because of the inner reference.

# Drawbacks
[drawbacks]: #drawbacks

The primary drawbacks were covered in depth in the guide-level explanation,
which also covers why they are not considered to be major problems:

- Long-range errors
  - can be readily mitigated by better explanations
- Removing fields can affect semver compatibility
  - considered unlikely to occur frequently in practice
  - already true that changing field types can affect semver compatibility
  - semver-like tool could help to mitigate

# Rationale and Alternatives
[alternatives]: #alternatives

Naturally, we might choose to retain the status quo, and continue to
require outlives annotations on structs. Assuming however that we wish
to remove them, the primary alternative is to consider going *farther*
than this RFC in various ways.

We might make try to infer outlives requirements for impls as well,
and thus eliminate the final place where `T: 'a` requirements are
needed. However, this would introduce complications in the
implementation -- in order to propagate requirements from impls to
structs, we must be able to do associated type normalization and hence
trait solving, but we would have to do before we know the full WF
requirements for each struct. The current setup avoids this
complication.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
