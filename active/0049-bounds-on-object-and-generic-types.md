- Start Date: 2014-08-06
- RFC PR: https://github.com/rust-lang/rfcs/pull/192
- Rust Issue: https://github.com/rust-lang/rust/issues/16462

# Summary

- Remove the special-case bound `'static` and replace with a generalized
  *lifetime bound* that can be used on objects and type parameters.
- Remove the rules that aim to prevent references from being stored
  into objects and replace with a simple lifetime check.
- Tighten up type rules pertaining to reference lifetimes and
  well-formed types containing references.
- Introduce explicit lifetime bounds (`'a:'b`), with the meaning that
  the lifetime `'a` outlives the lifetime `'b`. These exist today but
  are always inferred; this RFC adds the ability to specify them
  explicitly, which is sometimes needed in more complex cases.

# Motivation

Currently, the type system is not supposed to allow references to
escape into object types. However, there are various bugs where it
fails to prevent this from happening. Moreover, it is very useful (and
frequently necessary) to store a reference into an object. Moreover,
the current treatment of generic types is in some cases naive and not
obviously sound.

# Detailed design

## Lifetime bounds on parameters

The heart of the new design is the concept of a *lifetime bound*. In fact,
this (sort of) exists today in the form of the `'static` bound:

     fn foo<A:'static>(x: A) { ... }
     
Here, the notation `'static` means "all borrowed content within `A`
outlives the lifetime `'static`". (Note that when we say that
something outlives a lifetime, we mean that it lives *at least that
long*. In other words, for any lifetime `'a`, `'a` outlives `'a`. This
is similar to how we say that every type `T` is a subtype of itself.)

In the newer design, it is possible to use an arbitrary lifetime as a
bound, and not just `'static`:

     fn foo<'a, A:'a>(x: A) { ... }

Explicit lifetime bounds are in fact only rarely necessary, for two
reasons:

1. The compiler is often able to infer this relationship from the argument
   and return types. More on this below.
2. It is only important to bound the lifetime of a generic type like
   `A` when one of two things is happening (and both of these are
   cases where the inference generally is sufficient):
   - A borrowed pointer to an `A` instance (i.e., value of type `&A`)
     is being consumed or returned.
   - A value of type `A` is being closed over into an object reference
     (or closure, which per the unboxed closures RFC is really the
     same thing).

Note that, per RFC 11, these lifetime bounds may appear in types as
well (this is important later on). For example, an iterator might be
declared:

    struct Items<'a, T:'a> {
        v: &'a Collection<T>
    }
    
Here, the constraint `T:'a` indicates that the data being iterated
over must live at least as long as the collection (logically enough).

## Lifetime bounds on object types

Like parameters, all object types have a lifetime bound. Unlike
parameter types, however, object types are *required* to have exactly
one bound. This bound can be either specified explicitly or derived
from the traits that appear in the object type. In general, the rule is
as follows:

- If an explicit bound is specified, use that.
- Otherwise, let S be the set of lifetime bounds we can derive.
- Otherwise, if S contains 'static, use 'static.
- Otherwise, if S is a singleton set, use that.
- Otherwise, error.

Here are some examples:

    trait IsStatic : 'static { }
    trait Is<'a> : 'a { }
    
    // Type               Bounds
    // IsStatic           'static
    // Is<'a>             'a
    // IsStatic+Is<'a>    'static+'a
    // IsStatic+'a        'static+'a
    // IsStatic+Is<'a>+'b 'static,'a,'b

Object types must have exactly one bound -- zero bounds is not
acceptable. Therefore, if an object type with no derivable bounds
appears, we will supply a default lifetime using the normal rules:

    trait Writer { /* no derivable bounds */ }
    struct Foo<'a> {
        Box<Writer>,      // Error: try Box<Writer+'static> or Box<Writer+'a>
        Box<Writer+Send>, // OK: Send implies 'static
        &'a Writer,       // Error: try &'a (Writer+'a)
    }

    fn foo(a: Box<Writer>, // OK: Sugar for Box<Writer+'a> where 'a fresh
           b: &Writer)     // OK: Sugar for &'b (Writer+'c) where 'b, 'c fresh
    { ... }

This kind of annotation can seem a bit tedious when using object types
extensively, though type aliases can help quite a bit:

    type WriterObj = Box<Writer+'static>;
    type WriterRef<'a> = &'a (Writer+'a);

The unresolved questions section discussed possibles ways to lighten
the burden.

See Appendix B for the motivation on why object types are permitted to
have exactly one lifetime bound.

## Specifying relations between lifetimes

Currently, when a type or fn has multiple lifetime parameters, there
is no facility to explicitly specify a relationship between them. For
example, in a function like this:

    fn foo<'a, 'b>(...) { ... }
    
the lifetimes `'a` and `'b` are declared as independent. In some
cases, though, it can be important that there be a relation between
them. In most cases, these relationships can be inferred (and in fact
are inferred today, see below), but it is useful to be able to state
them explicitly (and necessary in some cases, see below).

A *lifetime bound* is written `'a:'b` and it means that "`'a` outlives
`'b`". For example, if `foo` were declared like so:

    fn foo<'x, 'y:'x>(...) { ... }
    
that would indicate that the lifetime '`x` was shorter than (or equal
to) `'y`.
  
## The "type must outlive" and well-formedness relation

Many of the rules to come make use of a "type must outlive" relation,
written `T outlives 'a`. This relation means primarily that all
borrowed data in `T` is known to have a lifetime of at least '`a`
(hence the name). However, the relation also guarantees various basic
lifetime constraints are met. For example, for every reference type
`&'b U` that is found within `T`, it would be required that `U
outlives 'b` (and that `'b` outlives `'a`). 

In fact, `T outlives 'a` is defined on another function `WF(T:'a)`,
which yields up a list of lifetime relations that must hold for `T` to
be well-formed and to outlive `'a`. It is not necessary to understand
the details of this relation in order to follow the rest of the RFC, I
will defer its precise specification to an appendix below.

For this section, it suffices to give some examples:

    // int always outlives any region
    WF(int : 'a) = []
    
    // a reference with lifetime 'a outlives 'b if 'a outlives 'b
    WF(&'a int : 'b) = ['a : 'b]

    // the outer reference must outlive 'c, and the inner reference
    // must outlive the outer reference
    WF(&'a &'b int : 'c) = ['a : 'c, 'b : 'a]

    // Object type with bound 'static
    WF(SomeTrait+'static : 'a) = ['static : 'a]

    // Object type with bound 'a 
    WF(SomeTrait+'a : 'b) = ['a : 'b]

## Rules for when object closure is legal

Whenever data of type `T` is closed over to form an object, the type
checker will require that `T outlives 'a` where `'a` is the primary
lifetime bound of the object type. 

## Rules for types to be well-formed

Currently we do not apply any tests to the types that appear in type
declarations. Per RFC 11, however, this should change, as we intend to
enforce trait bounds on types, wherever those types appear. Similarly,
we should be requiring that types are well-formed with respect to the
`WF` function. This means that a type like the following would be
illegal without a lifetime bound on the type parameter `T`:

    struct Ref<'a, T> { c: &'a T }

This is illegal because the field `c` has type `&'a T`, which is only
well-formed if `T:'a`. Per usual practice, this RFC does not propose
any form of inference on struct declarations and instead requires all
conditions to be spelled out (this is in contrast to fns and methods,
see below).

## Rules for expression type validity

We should add the condition that for every expression with lifetime
`'e` and type `T`, then `T outlives 'e`. We already enforce this in
many special cases but not uniformly.

## Inference

The compiler will infer lifetime bounds on both type parameters and
region parameters as follows. Within a function or method, we apply
the wellformedness function `WF` to each function or parameter type.
This yields up a set of relations that must hold. The idea here is
that the caller could not have type checked unless the types of the
arguments were well-formed, so that implies that the callee can assume
that those well-formedness constraints hold.

As an example, in the following function:

    fn foo<'a, A>(x: &'a A) { ... }
    
the callee here can assume that the type parameter `A` outlives the
lifetime `'a`, even though that was not explicitly declared.

Note that the inference also pulls in constraints that were declared
on the types of arguments. So, for example, if there is a type `Items`
declared as follows:

    struct Items<'a, T:'a> { ... }
    
And a function that takes an argument of type `Items`:

    fn foo<'a, T>(x: Items<'a, T>) { ... }

The inference rules will conclude that `T:'a` because the `Items` type
was declared with that bound.

In practice, these inference rules largely remove the need to manually
declare lifetime relations on types. When porting the existing library
and rustc over to these rules, I had to add explicit lifetime bounds
to exactly one function (but several types, almost exclusively
iterators).

Note that this sort of inference is already done. This RFC simply
proposes a more extensive version that also includes bounds of the
form `X:'a`, where `X` is a type parameter.

# What does all this mean in practice?

This RFC has a lot of details. The main implications for end users are:

1. Object types must specify a lifetime bound when they appear in a type.
   This most commonly means changing `Box<Trait>` to `Box<Trait+'static>`
   and `&'a Trait` to `&'a Trait+'a`.
2. For types that contain references to generic types, lifetime bounds
   are needed in the type definition. This comes up most often in iterators:

       struct Items<'a, T:'a> {
           x: &'a [T]
       }
       
   Here, the presence of `&'a [T]` within the type definition requires
   that the type checker can show that `T outlives 'a` which in turn
   requires the bound `T:'a` on the type definition. These bounds are
   rarely outside of type definitions, because they are almost always
   implied by the types of the arguments.
3. It is sometimes, but rarely, necessary to use lifetime bounds,
   specifically around double indirections (references to references,
   often the second reference is contained within a struct). For
   example:

       struct GlobalContext<'global> {
           arena: &'global Arena
       }
       
       struct LocalContext<'local, 'global:'local> {
           x: &'local mut Context<'global>
       }
       
   Here, we must know that the lifetime `'global` outlives `'local` in
   order for this type to be well-formed.

# Phasing

Some parts of this RFC require new syntax and thus must be phased in.
The current plan is to divide the implementation three parts:

1. Implement support for everything in this RFC except for region bounds
   and requiring that every expression type be well-formed. Enforcing
   the latter constraint leads to type errors that require lifetime
   bounds to resolve.
2. Implement support for `'a:'b` notation to be parsed under a feature
   gate `issue_5723_bootstrap`.
3. Implement the final bits of the RFC:
   - Bounds on lifetime parameters
   - Wellformedness checks on every expression
   - Wellformedness checks in type definitions

Parts 1 and 2 can be landed simultaneously, but part 3 requires a
snapshot. Parts 1 and 2 have largely been written. Depending on
precisely how the timing works out, it might make sense to just merge
parts 1 and 3.

# Drawbacks / Alternatives

If we do not implement some solution, we could continue with the
current approach (but patched to be sound) of banning references from
being closed over in object types. I consider this a non-starter.

# Unresolved questions

## Inferring wellformedness bounds

Under this RFC, it is required to write bounds on struct types which are
in principle inferable from their contents. For example, iterators
tend to follow a pattern like:

    struct Items<'a, T:'a> {
        x: &'a [T]
    }
    
Note that `T` is bounded by `'a`. It would be possible to infer these
bounds, but I've stuck to our current principle that type definitions
are always fully spelled out. The danger of inference is that it
becomes unclear *why* a particular constraint exists if one must
traverse the type hierarchy deeply to find its origin. This could
potentially be addressed with better error messages, though our track
record for lifetime error messages is not very good so far.

Also, there is a potential interaction between this sort of inference
and the description of default trait bounds below.

## Default trait bounds

When referencing a trait object, it is almost *always* the case that one follows
certain fixed patterns:

- `Box<Trait+'static>`
- `Rc<Trait+'static>` (once DST works)
- `&'a (Trait+'a)`
- and so on.

You might think that we should simply provide some kind of defaults
that are sensitive to where the `Trait` appears. The same is probably
true of struct type parameters (in other words, `&'a SomeStruct<'a>`
is a very common pattern).

However, there are complications:

- What about a type like `struct Ref<'a, T:'a> { x: &'a T }`? `Ref<'a,
  Trait>` should really work the same way as `&'a Trait`. One way that
  I can see to do this is to drive the defaulting based on the default
  trait bounds of the `T` type parameter -- but if we do that, it is
  both a non-local default (you have to consult the definition of
  `Ref`) and interacts with the potential inference described in the
  previous section.
- There *are* reasons to want a type like `Box<Trait+'a>`. For example,
  the macro parser includes a function like:
  
      fn make_macro_ext<'cx>(cx: &'cx Context, ...) -> Box<MacroExt+'cx>
    
  In other words, this function returns an object that closes over the
  macro context. In such a case, if `Box<MacroExt>` implies a static
  bound, then taking ownership of this macro object would require a signature
  like:
  
      fn take_macro_ext<'cx>(b: Box<MacroExt+'cx>) {  }
  
  Note that the `'cx` variable is only used in one place. It's purpose
  is just to disable the `'static` default that would otherwise be
  inserted.

# Appendix: Definition of the outlives relation and well-formedness

To make this more specific, we can "formally" model the Rust type
system as:

    T = scalar (int, uint, fn(...))   // Boring stuff
      | *const T                      // Unsafe pointer
      | *mut T                        // Unsafe pointer
      | Id<P>                         // Nominal type (struct, enum)
      | &'x T                         // Reference
      | &'x mut T                     // Mutable reference
      | {TraitReference<P>}+'x        // Object type
      | X                             // Type variable
    P = {'x} + {T}
    
We can define a function `WF(T : 'a)` which, given a type `T` and
lifetime `'a` yields a list of `'b:'c` or `X:'d` pairs. For each pair
`'b:'c`, the lifetime `'b` must outlive the lifetime `'c` for the type
`T` to be well-formed in a location with lifetime `'a`. For each pair
`X:'d`, the type parameter `X` must outlive the lifetime `'d`.

- `WF(int : 'a)` yields an empty list
- `WF(X:'a)` where `X` is a type parameter yields `(X:'a)`.
- `WF(Foo<P>:'a)` where `Foo<P>` is an enum or struct type yields:
  - For each lifetime parameter `'b` that is contravariant or invariant,
    `'b : 'a`.
  - For each type parameter `T` that is covariant or invariant, the
    results of `WF(T : 'a)`.
  - The lifetime bounds declared on `Foo`'s lifetime or type parameters.
  - The reasoning here is that if we can reach borrowed data with
    lifetime `'a` through `Foo<'a>`, then `'a` must be contra- or
    invariant.  Covariant lifetimes only occur in "setter"
    situations. Analogous reasoning applies to the type case.
- `WF(T:'a)` where `T` is an object type:
  - For the primary bound `'b`, `'b : 'a`.
  - For each derived bound `'c` of `T`, `'b : 'c`
    - Motivation: The primary bound of an object type implies that all
      other bounds are met. This simplifies some of the other
      formulations and does not represent a loss of expressiveness.

We can then say that `T outlives 'a` if all lifetime relations
returned by `WF(T:'a)` hold.

# Appendix B: Why object types must have exactly one bound

The motivation is that handling multiple bounds is overwhelmingly
complicated to reason about and implement. In various places,
constraints arise of the form `all i. exists j. R[i] <= R[j]`, where
`R` is a list of lifetimes. This is challenging for lifetime
inference, since there are many options for it to choose from, and
thus inference is no longer a fixed-point iteration. Moreover, it
doesn't seem to add any particular expressiveness.

The places where this becomes important	are:

- Checking lifetime bounds when data is closed over into an object type
- Subtyping between object types, which would most naturally be
  contravariant in the lifetime bound

Similarly, requiring that the "master" bound on object lifetimes outlives
all other bounds also aids inference. Now, given a type like the
following:

    trait Foo<'a> : 'a { }
    trait Bar<'b> : 'b { }
    
    ...
    
    let x: Box<Foo<'a>+Bar<'b>>

the inference engine can create a fresh lifetime variable `'0` for the
master bound and then say that `'0:'a` and `'0:'b`. Without the
requirement that `'0` be a master bound, it would be somewhat unclear
how `'0` relates to `'a` and `'b` (in fact, there would be no
necessary relation). But if there is no necessary relation, then when
closing over data, one would have to ensure that the closed over data
outlives *all* derivable lifetime bounds, which again creates a
constraint of the form `all i. exists j.`.
