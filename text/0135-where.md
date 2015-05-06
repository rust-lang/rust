- Start Date: 2014-09-30
- RFC PR #: https://github.com/rust-lang/rfcs/pull/135
- Rust Issue #: https://github.com/rust-lang/rust/issues/17657

# Summary

Add `where` clauses, which provide a more expressive means of
specifying trait parameter bounds. A `where` clause comes after a
declaration of a generic item (e.g., an impl or struct definition) and
specifies a list of bounds that must be proven once precise values are
known for the type parameters in question. The existing bounds
notation would remain as syntactic sugar for where clauses.

So, for example, the `impl` for `HashMap` could be changed from this:

    impl<K:Hash+Eq,V> HashMap<K, V>
    {
        ..
    }
    
to the following:    

    impl<K,V> HashMap<K, V>
        where K : Hash + Eq
    {
        ..
    }

The full grammar can be found in the detailed design.

# Motivation

The high-level bit is that the current bounds syntax does not scale to
complex cases. Introducing `where` clauses is a simple extension that
gives us a lot more expressive power. In particular, it will allow us
to refactor the operator traits to be in a convenient, multidispatch
form (e.g., so that user-defined mathematical types can be added to
`int` and vice versa). (It's also worth pointing out that, once #5527
lands at least, implementing where clauses will be very little work.)

Here is a list of limitations with the current bounds syntax that are
overcome with the `where` syntax:

- **It cannot express bounds on anything other than type parameters.**
  Therefore, if you have a function generic in `T`, you can write
  `T:MyTrait` to declare that `T` must implement `MyTrait`, but you
  can't write `Option<T> : MyTrait` or `(int, T) : MyTrait`. These
  forms are less commonly required but still important.

- **It does not work well with associated types.** This is because
  there is no space to specify the value of an associated type. Other
  languages use `where` clauses (or something analogous) for this
  purpose.
  
- **It's just plain hard to read.** Experience has shown that as the
  number of bounds grows, the current syntax becomes hard to read and
  format.
  
Let's examine each case in detail.  
  
### Bounds are insufficiently expressive

Currently bounds can only be declared on type parameters. But there
are situations where one wants to declare bounds not on the type
parameter itself but rather a type that includes the type parameter.

#### Partially generic types

One situation where this is occurs is when you want to write functions
where types are partially known and have those interact with other
functions that are fully generic. To explain the situation, let's
examine some code adapted from rustc.

Imagine I have a table parameterized by a value type `V` and a key
type `K`. There are also two traits, `Value` and `Key`, that describe
the keys and values. Also, each type of key is linked to a specific
value:

    struct Table<V:Value, K:Key<V>> { ... }
    trait Key<V:Value> { ... }
    trait Value { ... }

Now, imagine I want to write some code that operates over all keys
whose value is an `Option<T>` for some `T`:

    fn example<T,K:Key<Option<T>>(table: &Table<Option<T>, K>) { ... }
    
This seems reasonable, but this code will not compile. The problem is
that the compiler needs to know that the value type implements
`Value`, but here the value type is `Option<T>`. So we'd need to
declare `Option<T> : Value`, which we cannot do.

There are workarounds. I might write a new trait `OptionalValue`:

    trait OptionalValue<T> {
        fn as_option<'a>(&'a self) -> &'a Option<T>; // identity fn
    }

and then I could write my example as:

    fn example<T,O:OptionalValue<T>,K:Key<O>(table: &Table<O, K>) { ... }

But this is making my example function, already a bit complicated,
become quite obscure.

#### Multidispatch traits

Another situation where a similar problem is encountered is
*multidispatch traits* (aka, multiparameter type classes in Haskell).
The idea of a multidispatch trait is to be able to choose the impl
based not just on one type, as is the most common case, but on
multiple types (usually, but not always, two).

Multidispatch is rarely needed because the *vast* majority of traits
are characterized by a single type. But when you need it, you really
need it. One example that arises in the standard library is the traits
for binary operators like `+`. Today, the `Add` trait is defined using
only single-dispatch (like so):

```
pub trait Add<Rhs,Sum> {
    fn add(&self, rhs: &Rhs) -> Sum;
}
```

The expression `a + b` is thus sugar for `Add::add(&a, &b)`. Because
of how our trait system works, this means that only the type of the
left-hand side (the `Self` parameter) will be used to select the
impl. The type for the right-hand side (`Rhs`) along with the type of
their sum (`Sum`) are defined as trait parameters, which are always
*outputs* of the trait matching: that is, they are specified by the
impl and are not used to select which impl is used.

This setup means that addition is not as extensible as we would
like. For example, the standard library includes implementations of
this trait for integers and other built-in types:

```
impl Add<int,int> for int { ... }
impl Add<f32,f32> for f32 { ... }
```

The limitations of this setup become apparent when we consider how a
hypothetical user library might integrate. Imagine a library L that
defines a type `Complex` representing complex numbers:

```
struct Complex { ... }
```

Naturally, it should be possible to add complex numbers and integers.
Since complex number addition is commutative, it should be possible to
write both `1 + c` and `c + 1`. Thus one might try the following
impls:

```
impl Add<int,Complex> for Complex { ... }     // 1. Complex + int
impl Add<Complex,Complex> for int { ... }     // 2. int + Complex
impl Add<Complex,Complex> for Complex { ... } // 3. Complex + Complex
```

Due to the coherence rules, however, this setup will not work. There
are in fact three errors. The first is that there are two impls of
`Add` defined for `Complex` (1 and 3). The second is that there are
two impls of `Add` defined for `int` (the one from the standard
library and 2). The final error is that impl 2 violates the orphan
rule, since the type `int` is not defined in the current crate.

This is not a new problem. Object-oriented languages, with their focus
on single dispatch, have long had trouble dealing with binary
operators. One common solution is double dispatch, an awkward but
effective pattern in which no type ever implements `Add`
directly. Instead, we introduce "indirection" traits so that, e.g.,
`int` is addable to anything that implements `AddToInt` and so
on. This is not my preferred solution so I will not describe it in
detail, but rather refer readers to [this blog post][bp] where I
describe how it works.

An alternative to double dispatch is to define `Add` on tuple types
`(LHS, RHS)` rather than on a single value. Imagine that the `Add`
trait were defined as follows:

    trait Add<Sum> {
        fn add(self) -> Sum;
    }
    
    impl Add<int> for (int, int) {
        fn add(self) -> int {
            let (x, y) = self;
            x + y
        }
    }

Now the expression `a + b` would be sugar for `Add::add((a, b))`.
This small change has several interesting ramifications. For one
thing, the library L can easily extend `Add` to cover complex numbers:

```
impl Add<Complex> for (Complex, int)     { ... }
impl Add<Complex> for (int, Complex)     { ... }
impl Add<Complex> for (Complex, Complex) { ... }
```

These impls do not violate the coherence rules because they are all
applied to distinct types. Moreover, none of them violate the orphan
rule because each of them is a tuple involving at least one type local
to the library.

One downside of this `Add` pattern is that there is no way within the
trait definition to refer to the type of the left- or right-hand side
individually; we can only use the type `Self` to refer to the tuple of
both types. In the *Discussion* section below, I will introduce
an extended "multi-dispatch" pattern that addresses this particular
problem.

There is however another problem that where clauses help to
address. Imagine that we wish to define a function to increment
complex numbers:

    fn increment(c: Complex) -> Complex {
        1 + c
    }
    
This function is pretty generic, so perhaps we would like to
generalize it to work over anything that can be added to an int. We'll
use our new version of the `Add` trait that is implemented over
tuples:

    fn increment<T:...>(c: T) -> T {
        1 + c
    }

At this point we encounter the problem. What bound should we give for
`T`?  We'd like to write something like `(int, T) : Add<T>` -- that
is, `Add` is implemented for the tuple `(int, T)` with the sum type
`T`. But we can't write that, because the current bounds syntax is too
limited.

Where clauses give us an answer. We can write a generic version of
`increment` like so:

    fn increment<T>(c: T) -> T
        where (int, T) : Add<T>
    {
        1 + c
    }

### Associated types

It is unclear exactly what form associated types will have in Rust,
but it is [well documented][comparison] that our current design, in
which type parameters decorate traits, does not scale particularly
well. (For curious readers, there are [several][part1] [blog][part2]
[posts][pnkfelix] exploring the design space of associated types with
respect to Rust in particular.)

The high-level summary of associated types is that we can replace
a generic trait like `Iterator`:

    trait Iterator<E> {
        fn next(&mut self) -> Option<E>;
    }
    
With a version where the type parameter is a "member" of the
`Iterator` trait:

    trait Iterator {
        type E;
        
        fn next(&mut self) -> Option<E>;
    }
    
This syntactic change helps to highlight that, for any given type, the
type `E` is *fixed* by the impl, and hence it can be considered a
member (or output) of the trait. It also scales better as the number
of associated types grows.

One challenge with this design is that it is not clear how to convert
a function like the following:

    fn sum<I:Iterator<int>>(i: I) -> int {
        ...    
    }
    
With associated types, the reference `Iterator<int>` is no longer
valid, since the trait `Iterator` doesn't have type parameters.

The usual solution to this problem is to employ a where clause:

    fn sum<I:Iterator>(i: I) -> int
      where I::E == int
    {
        ...    
    }
  
We can also employ where clauses with object types via a syntax like
`&Iterator<where E=int>` (admittedly somewhat wordy)

## Readability

When writing very generic code, it is common to have a large number of
parameters with a large number of bounds. Here is some example
function extracted from `rustc`:

    fn set_var_to_merged_bounds<T:Clone + InferStr + LatticeValue,
                                V:Clone+Eq+ToStr+Vid+UnifyVid<Bounds<T>>>(
                                &self,
                                v_id: V,
                                a: &Bounds<T>,
                                b: &Bounds<T>,
                                rank: uint)
                                -> ures;

Definitions like this are very difficult to read (it's hard to even know
how to *format* such a definition).

Using a `where` clause allows the bounds to be separated from the list
of type parameters:

    fn set_var_to_merged_bounds<T,V>(&self,
                                     v_id: V,
                                     a: &Bounds<T>,
                                     b: &Bounds<T>,
                                     rank: uint)
                                     -> ures
        where T:Clone,         // it is legal to use individual clauses...
              T:InferStr,
              T:LatticeValue,
              V:Clone+Eq+ToStr+Vid+UnifyVid<Bounds<T>>, // ...or use `+`
    {                                     
        ..
    }
    
This helps to separate out the function signature from the extra
requirements that the function places on its types.

If I may step aside from the "impersonal voice" of the RFC for a
moment, I personally find that when writing generic code it is helpful
to focus on the types and signatures, and come to the bounds
later. Where clauses help to separate these distinctions. Naturally,
your mileage may vary. - nmatsakis

# Detailed design

### Where can where clauses appear?

Where clauses can be added to anything that can be parameterized with
type/lifetime parameters with the exception of trait method
definitions: `impl` declarations, `fn` declarations, and `trait` and
`struct` definitions. They appear as follows:

    impl Foo<A,B>
        where ...
    { }

    impl Foo<A,B> for C
        where ...
    { }

    impl Foo<A,B> for C
    {
        fn foo<A,B> -> C
            where ...
        { }
    }

    fn foo<A,B> -> C
        where ...
    { }

    struct Foo<A,B>
        where ...
    { }

    trait Foo<A,B> : C
        where ...
    { }
    
#### Where clauses cannot (yet) appear on trait methods

Note that trait method definitions were specifically excluded from the
list above. The reason is that including where clauses on a trait
method raises interesting questions for what it means to implement the
trait. Using where clauses it becomes possible to define methods that
do not necessarily apply to all implementations. We intend to enable
this feature but it merits a second RFC to delve into the details.

### Where clause grammar

The grammar for a `where` clause would be as follows (BNF):

    WHERE = 'where' BOUND { ',' BOUND } [,]
    BOUND = TYPE ':' TRAIT { '+' TRAIT } [+]
    TRAIT = Id [ '<' [ TYPE { ',' TYPE } [,] ] '>' ]
    TYPE  = ... (same type grammar as today)
    
### Semantics    

The meaning of a where clause is fairly straightforward. Each bound in
the where clause must be proven by the caller after substitution of
the parameter types.

One interesting case concerns trivial where clauses where the
self-type does not refer to any of the type parameters, such as the
following:

    fn foo()
        where int : Eq
    { ... }

Where clauses like these are considered an error. They have no
particular meaning, since the callee knows all types involved. This is
a conservative choice: if we find that we do desire a particular
interpretation for them, we can always make them legal later.

# Drawbacks

This RFC introduces two ways to declare a bound.

# Alternatives

**Remove the existing trait bounds.** I decided against this both to
avoid breaking lots of existing code and because the existing syntax
is convenient much of the time.

**Embed where clauses in the type parameter list.** One alternative
syntax that was proposed is to embed a where-like clause in the type
parameter list. Thus the `increment()` example

    fn increment<T>(c: T) -> T
        where () : Add<int,T,T>
    {
        1 + c
    }

would become something like:

    fn increment<T, ():Add<int,T,T>>(c: T) -> T
    {
        1 + c
    }

This is unfortunately somewhat ambiguous, since a bound like `T:Eq`
could either be declared a type parameter `T` or as a condition that
the (existing) type `T` implement `Eq`.

**Use a colon intead of the keyword.** There is some precedent for
this from the type state days. Unfortunately, it doesn't work with
traits due to the supertrait list, and it also doesn't look good with
the use of `:` as a trait-bound separator:

    fn increment<T>(c: T) -> T
        : () : Add<int,T,T>
    {
        1 + c
    }

[bp]: http://smallcultfollowing.com/babysteps/blog/2012/10/04/refining-traits-slash-impls/
[comparison]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.122
[pnkfelix]: http://blog.pnkfx.org/blog/2013/04/22/designing-syntax-for-associated-items-in-rust/#background
[part1]: http://www.smallcultfollowing.com/babysteps/blog/2013/04/02/associated-items/
[part2]: http://www.smallcultfollowing.com/babysteps/blog/2013/04/03/associated-items-continued/

