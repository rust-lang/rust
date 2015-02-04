- Start Date: 2015-02-04
- RFC PR: https://github.com/rust-lang/rfcs/pull/213
- Rust Issue: https://github.com/rust-lang/rust/issues/21939

# Summary

Rust currently includes feature-gated support for type parameters that
specify a default value. This feature is not well-specified. The aim
of this RFC is to fully specify the behavior of defaulted type
parameters:

1. Type parameters in any position can specify a default.
2. Within fn bodies, defaulted type parameters are used to drive inference.
3. Outside of fn bodies, defaulted type parameters supply fixed
   defaults.
4. `_` can be used to omit the values of type parameters and apply a
   suitable default:
   - In a fn body, any type parameter can be omitted in this way, and
     a suitable type variable will be used.
   - Outside of a fn body, only defaulted type parameters can be
     omitted, and the specified default is then used.

Points 2 and 4 extend the current behavior of type parameter defaults,
aiming to address some shortcomings of the current implementation.

This RFC would remove the feature gate on defaulted type parameters.

# Motivation

## Why defaulted type parameters

Defaulted type parameters are very useful in two main scenarios:

1. Extended a type without breaking existing clients.
2. Allowing customization in ways that many or most users do not care
   about.

Often, these two scenarios occur at the same time. A classic
historical example is the `HashMap` type from Rust's standard
library. This type now supports the ability to specify custom
hashers. For most clients, this is not particularly important and this
initial versions of the `HashMap` type were not customizable in this
regard. But there are some cases where having the ability to use a
custom hasher can make a huge difference. Having the ability to
specify defaults for type parameters allowed the `HashMap` type to add
a new type parameter `H` representing the hasher type without breaking
any existing clients and also without forcing all clients to specify
what hasher to use.

However, customization occurs in places other than types. Consider the
function `range()`. In early versions of Rust, there was a distinct
range function for each integral type (e.g. `uint::range`,
`int::range`, etc). These functions were eventually consolidated into
a single `range()` function that is defined generically over all
"enumerable" types:

    trait Enumerable : Add<Self,Self> + PartialOrd + Clone + One;
    pub fn range<A:Enumerable>(start: A, stop: A) -> Range<A> {
        Range{state: start, stop: stop, one: One::one()}
    }

This version is often more convenient to use, particularly in a
generic context.

However, the generic version does have the downside that when the
bounds of the range are integral, inference sometimes lacks enough
information to select a proper type:

    // ERROR -- Type argument unconstrained, what integral type did you want?
    for x in range(0, 10) { ... }

Thus users are forced to write:

    for x in range(0u, 10u) { ... }

This RFC describes how to integrate default type parameters with
inference such that the type parameter on `range` can specify a
default (`uint`, for example):

    pub fn range<A:Enumerable=uint>(start: A, stop: A) -> Range<A> {
        Range{state: start, stop: stop, one: One::one()}
    }

Using this definition, a call like `range(0, 10)` is perfectly legal.
If it turns out that the type argument is not other constraint, `uint`
will be used instead.

## Extending types without breaking clients.

Without defaults, once a library is released to "the wild", it is not
possible to add type parameters to a type without breaking all
existing clients. However, it frequently happens that one wants to
take an existing type and make it more flexible that it used to be.
This often entails adding a new type parameter so that some type which
was hard-coded before can now be customized. Defaults provide a means
to do this while having older clients transparently fallback to the
older behavior.

*Historical example:* Extending HashMap to support various hash
 algorithms.

# Detailed Design

## Remove feature gate

This RFC would remove the feature gate on defaulted type parameters.

## Type parameters with defaults

Defaults can be placed on any type parameter, whether it is declared
on a type definition (`struct`, `enum`), type alias (`type`), trait
definition (`trait`), trait implementation (`impl`), or a function or
method (`fn`).

Once a given type parameter declares a default value, all subsequent
type parameters in the list must declare default values as well:

    // OK. All defaulted type parameters come at the end.
    fn foo<A,B=uint,C=uint>() { .. }

    // ERROR. B has a default, but C does not.
    fn foo<A,B=uint,C>() { .. }

The default value of a type parameter `X` may refer to other type
parameters declared on the same item. However, it may only refer to
type parameters declared *before* `X` in the list of type parameters:

    // OK. Default value of `B` refers to `A`, which is not defaulted.
    fn foo<A,B=A>() { .. }

    // OK. Default value of `C` refers to `B`, which comes before
    // `C` in the list of parameters.
    fn foo<A,B=uint,C=B>() { .. }

    // ERROR. Default value of `B` refers to `C`, which comes AFTER
    // `B` in the list of parameters.
    fn foo<A,B=C,C=uint>() { .. }

## Instantiating defaults

This section specifies how to interpret a reference to a generic
type. Rather than writing out a rather tedious (and hard to
understand) description of the algorithm, the rules are instead
specified by a series of examples. The high-level idea of the rules is
as follows:

- Users must always provide *some* value for non-defaulted type parameters.
  Defaulted type parameters may be omitted.
- The `_` notation can always be used to *explicitly omit* the value
  of a type parameter:
  - Inside a fn body, any type parameter may be omitted. Inference is used.
  - Outside a fn body, only defaulted type parameters may be
    omitted. The default value is used.
  - *Motivation:* This is consistent with Rust tradition, which
    generally requires explicit types or a mechanical defaulting
    process outside of `fn` bodies.

### References to generic types

We begin with examples of references to the generic type `Foo`:

    struct Foo<A,B,C=DefaultHasher,D=C> { ... }

`Foo` defines four type parameters, the final two of which are
defaulted. First, let us consider what happens outside of a fn
body. It is mandatory to supply explicit values for all non-defaulted
type parameters:

    // ERROR: 2 parameters required, 0 provided.
    fn f(_: &Foo) { ... }

Defaulted type parameters are filled in based on the defaults given:

    // Legal: Equivalent to `Foo<int,uint,DefaultHasher,DefaultHasher>`
    fn f(_: &Foo<int,uint>) { ... }

Naturally it is legal to specify explicit values for the defaulted
type parameters if desired:

    // Legal: Equivalent to `Foo<int,uint,uint,char,u8>`
    fn f(_: &Foo<int,uint,char,u8>) { ... }

It is also legal to provide just one of the defaulted type parameters
and not the other:

    // Legal: Equivalent to `Foo<int,uint,char,char>`
    fn f(_: &Foo<int,uint,char>) { ... }

If the user wishes to supply the value of the type parameter `D`
explicitly, but not `C`, then `_` can be used to request the default:

    // Legal: Equivalent to `Foo<int,uint,DefaultHasher,uint>`
    fn f(_: &Foo<int,uint,_,uint>) { ... }

Note that, outside of a fn body, `_` can *only* be used with
defaulted type parameters:

    // ERROR: outside of a fn body, `_` cannot be
    // used for a non-defaulted type parameter
    fn f(_: &Foo<int,_>) { ... }

Inside a fn body, the rules are much the same, except that `_` is
legal everywhere. Every reference to `_` creates a fresh type
variable `$n`. If the type parameter whose value is omitted has an
associate default, that default is used as the *fallback* for `$n`
(see the section "Type variables with fallbacks" for more
information). Here are some examples:

    fn f() {
        // Error: `Foo` requires at least 2 type parameters, 0 supplied.
        let x: Foo = ...;

        // All of these 4 examples are OK and equivalent. Each
        // results in a type `Foo<$0,$1,$2,$3>` and `$0`-`$4` are type
        // variables. `$2` has a fallback of `DefaultHasher` and `$3`
        // has a fallback of `$2`.
        let x: Foo<_,_> = ...;
        let x: Foo<_,_,_> = ...;
        let x: Foo<_,_,_,_> = ...;

        // Results in a type `Foo<int,uint,$0,char>` where `$0`
        // has a fallback of `DefaultHasher`.
        let x: Foo<int,uint,_,char> = ...;
    }

### References to generic traits

The rules for traits are the same as the rules for types.  Consider a
trait `Foo`:

    trait Foo<A,B,C=uint,D=C> { ... }

References to this trait can omit values for `C` and `D` in precisely
the same way as was shown for types:

    // All equivalent to Foo<i8,u8,uint,uint>:
    fn foo<T:Foo<i8,u8>>() { ... }
    fn foo<T:Foo<i8,u8,_>>() { ... }
    fn foo<T:Foo<i8,u8,_,_>>() { ... }

    // Equivalent to Foo<i8,u8,char,char>:
    fn foo<T:Foo<i8,u8,char,_>>() { ... }

### References to generic functions

The rules for referencing generic functions are the same as for types,
except that it is legal to omit values for all type parameters if
desired. In that case, the behavior is the same as it would be if `_`
were used as the value for every type parameter. Note that functions
can only be referenced from within a fn body.

### References to generic impls

Users never explicitly "reference" an impl. Rather, the trait matching
system implicitly instantaites impls as part of trait matching. This
implies that all type parameters are always instantiated with type
variables. These type variables are assigned fallbacks according to
the defaults given.

## Type variables with fallbacks

We extend the inference system so that when a type variable is
created, it can optionally have a *fallback value*, which is another
type.

In the type checker, whenever we create a fresh type variable to
represent a type parameter with an associated default, we will use
that default as the fallback value for this type variable.

Example:

```
fn foo<A,B=A>(a: A, b: B) { ... }

fn bar() {
    // Here, the values of the type parameters are given explicitly.
    let f: fn(uint, uint) = foo::<uint, uint>;

    // Here the value of the first type parameter is given explicitly,
    // but not the second. Because the second specifies a default, this
    // is permitted. The type checker will create a fresh variable `$0`
    // and attempt to infer the value of this defaulted type parameter.
    let g: fn(uint, $0) = foo::<uint>;

    // Here, the values of the type parameters are not given explicitly,
    // and hence the type checker will create fresh variables
    // `$1` and `$2` for both of them.
    let h: fn($1, $2) = foo;
}
```

In this snippet, there are three references to the generic function
`foo`, each of which specifies progressively fewer types. As a result,
the type checker winds up creating three type variables, which are
referred to in the example as `$0`, `$1`, and `$2` (not that this `$`
notation is just for explanatory purposes and is not actual Rust
syntax).

The fallback values of `$0`, `$1`, and `$2` are as follows:

- `$0` was created to represent the type parameter `B` defined on
  `foo`.  This means that `$0` will have a fallback value of `uint`,
  since the type variable `A` was specified to be `uint` in the
  expression that created `$0`.
- `$1` was created to represent the type parameter `A`, which
  has no default. Therefore `$1` has no fallback.
- `$2` was created to represent the type parameter `B`. It will
  have the fallback value of `$1`, which was the value of `A`
  within the expression where `$2` was created.

## Trait resolution, fallbacking, and inference

Prior to this RFC, type-checking a function body proceeds roughly as
follows:

1. The function body is analyzed. This results in an accumulated set of
   type variables, constraints, and trait obligations.
2. Those trait obligations are then resolved until a fixed point
   is reached.
3. If any trait obligations remain unresolved, an error is reported.
4. If any type variables were never bound to a concrete value, an error
   is reported.

To accommodate fallback, the new procedure is somewhat different:

1. The function body is analyzed. This results in an accumulated set of
   type variables, constraints, and trait obligations.
2. Execute in a loop:
  1. Run trait resolution until a fixed point is reached.
  2. Create a (initially empty) set `UB` of unbound type and
     integral/float variables.  This set represents the set of
     variables for which fallbacks should be applied.
  3. Add all unbound integral and float variables to the set `UB`
  4. For each type variable `X`:
     - If `X` has no fallback defined, skip.
     - If `X` is not bound, add `X` to `UB`
     - If `X` is bound to an unbound integral variable `I`, add `X` to
       `UB` and remove `I` from `UB` (if present).
     - If `X` is bound to an unbound float variable `F`, add `X` to
       `UB` and remove `F` from `UB` (if present).
  5. If `UB` is the empty set, break out of the loop.
  6. For each member of `UB`:
     - If the member is an integral type variable `I`, set `I` to `int`.
     - If the member is a float variable `F`, set `I` to `f64`.
     - Otherwise, the member must be a variable `X` with a defined fallback.
       Set `X` to its fallback.
       - Note that this "set" operations can fail, which indicates
         conflicting defaults. A suitable error message should be
         given.
3. If any type parameters still have no value assigned to them, report an error.
4. If any trait obligations could not be resolved, report an error.

There are some subtle points to this algorithm:

**When defaults are to be applied, we first gather up the set of
variables that have applicable defaults (step 2.2) and then later
unconditionally apply those defaults (step 2.4).** In particular, we
do not loop over each type variable, check whether it is unbound, and
apply the default only if it is unbound. The reason for this is that
it can happen that there are contradictory defaults and we want to
ensure that this results in an error:

    fn foo<F:Default=uint>() -> F { }
    fn bar<B=int>(b: B) { }
    fn baz() {
        // Here, F is instantiated with $0=uint
        let x: $0 = foo();

        // Here, B is instantiated with $1=uint, and constraint $0 <: $1 is added.
        bar(x);
    }

In this example, two type variables are created. `$0` is the value of
`F` in the call to `foo()` and `$1` is the value of `B` in the call to
`bar()`. The fact that `x`, which has type `$0`, is passed as an
argument to `bar()` will add the constraint that `$0 <: $1`, but at no
point are any concrete types given. Therefore, once type checking is
complete, we will apply defaults. Using the algorithm given above, we
will determine that both `$0` and `$1` are unbound and have suitable
defaults. We will then unify `$0` with `uint`. This will succeed and,
because `$0 <: $1`, cause `$1` to be unified with `uint`. Next, we
will try to unify `$1` with its default, `int`. This will lead to an
error. If we combined the checking of whether `$1` was unbound with
the unification with the default, we would have first unified `$0` and
then decided that `$1` did not require unification.

**In the general case, a loop is required to continue resolving traits
and applying defaults in sequence.** Resolving traits can lead to
unifications, so it is clear that we must resolve all traits that we
can before we apply any defaults. However, it is also true that adding
defaults can create new trait obligations that must be resolved.

Here is an example where processing trait obligations creates
defaults, and processing defaults created trait obligations:

    trait Foo { }
    trait Bar { }

    impl<T:Bar=uint> Foo for Vec<T> { } // Impl 1
    impl Bar for uint { } // Impl 2

    fn takes_foo<F:Foo>(f: F) { }

    fn main() {
        let x = Vec::new(); // x: Vec<$0>
        takes_foo(x); // adds oblig Vec<$0> : Foo
    }

When we finish type checking `main`, we are left with a variable `$0`
and a trait obligation `Vec<$0> : Foo`. Processing the trait
obligation selects the impl 1 as the way to fulfill this trait
obligation. This results in:

1. a new type variable `$1`, which represents the parameter `T` on the impl.
   `$1` has a default, `uint`.
2. the constraint that `$0=$1`.
3. a new trait obligation `$1 : Bar`.

We cannot process the new trait obligation yet because the type
variable `$1` is still unbound. (We know that it is equated with `$0`,
but we do not have any concrete types yet, just variables.) After
trait resolution reaches a fixed point, defaults are applied.  `$1` is
equated with `uint` which in turn propagates to `$0`. At this point,
there is still an outstanding trait obligation `uint : Bar`.  This
trait obligation can be resolved to impl 2.

The previous example consisted of "1.5" iterations of the loop. That
is, although trait resolution runs twice, defaults are only needed one
time:

1. Trait resolution executed to resolve `Vec<$0> : Foo`.
2. Defaults were applied to unify `$1 = $0 = uint`.
3. Trait resolution executed to resolve `uint : Bar`
4. No more defaults to apply, done.

The next example does 2 full iterations of the loop.

    trait Foo { }
    trait Bar<U> { }
    trait Baz { }

    impl<U,T:Bar<U>=Vec<U>> Foo for Vec<T> { } // Impl 1
    impl<V=uint> Bar for Vec<V> { } // Impl 2

    fn takes_foo<F:Foo>(f: F) { }

    fn main() {
        let x = Vec::new(); // x: Vec<$0>
        takes_foo(x); // adds oblig Vec<$0> : Foo
    }

Here the process is as follows:

1. Trait resolution executed to resolve `Vec<$0> : Foo`. The result is
   two fresh variables, `$1` (for `U`) and `$2=Vec<$1>` (for `$T`), the
   constraint that `$0=$2`, and the obligation `$2 : Bar<$1>`.
2. Defaults are applied to unify `$2 = $0 = Vec<$1>`.
3. Trait resolution executed to resolve `$2 : Bar<$1>`. The result
   is a fresh variable `$3=uint` (for `$V`) and the constraint
   that `$1=$3`.
4. Defaults are applied to unify `$3 = $1 = uint`.

It should be clear that one can create examples in this vein so as to
require any number of loops.

**Interaction with integer/float literal fallback.** This RFC gives
defaulted type parameters precedence over integer/float literal
fallback. This seems preferable because such types can be more
specific. Below are some examples. See also the *alternatives*
section.

```
// Here the type of the integer literal 22 is inferred
// to `int` using literal fallback.
fn foo<T>(t: T) { ... }
foo(22)
```

```
// Here the type of the integer literal 22 is inferred
// to `uint` because the default on `T` overrides the
// standard integer literal fallback.
fn foo<T=uint>(t: T) { ... }
foo(22)
```

```
// Here the type of the integer literal 22 is inferred
// to `char`, leading to an error. This can be resolved
// by using an explicit suffix like `22i`.
fn foo<T=char>(t: T) { ... }
foo(22)
```

**Termination.** Any time that there is a loop, one must inquire after
termination. In principle, the loop above could execute indefinitely.
This is because trait resolution is not guaranteed to terminate --
basically there might be a cycle between impls such that we continue
creating new type variables and new obligations forever. The trait
matching system already defends against this with a recursion counter.
That same recursion counter is sufficient to guarantee termination
even when the default mechanism is added to the mix. This is because
the default mechanism can never itself create new trait obligations:
it can only cause previous ambiguous trait obligations to now be
matchable (because unbound variables become bound). But the actual
need to iteration through the loop is still caused by trait matching
generating recursive obligations, which have an associated depth
limit.

## Compatibility analysis

One of the major design goals of defaulted type parameters is to
permit new parameters to be added to existing types or methods in a
backwards compatible way. This remains possible under the current
design.

Note though that adding a default to an *existing* type parameter can
lead to type errors in clients. This can occur if clients were already
relying on an inference fallback from some other source and there is
now an ambiguity. Naturally clients can always fix this error by
specifying the value of the type parameter in question manually.

# Downsides and alternatives

## Avoid inference

Rather than adding the notion of *fallbacks* to type variables,
defaults could be mechanically added, even within fn bodies, as they
are today. But this is disappointing because it means that examples
like `range(0,10)`, where defaults could inform inference, still
require explicit annotation. Without the notion of fallbacks, it is
also difficult to say what defaulted type parameters in methods or
impls should mean.

## More advanced interaction between integer literal inference

There were some other proposals to have a more advanced interaction
between custom fallbacks and literal inference. For example, it is
possible to imagine that we allow literal inference to take precedence
over type default fallbacks, unless the fallback is itself integral.
The problem is that this is both complicated and possibly not forwards
compatible if we opt to allow a more general notion of literal
inference in the future (in other words, if integer literals may be
mapped to more than just the built-in integral types). Furthermore,
these rules would create strictly fewer errors, and hence can be added
in the future if desired.

## Notation

Allowing `_` notation outside of fn body means that it's meaning
changes somewhat depending on context. However, this is consistent
with the meaning of omitted lifetimes, which also change in the same
way (mechanical default outside of fn body, inference within).

An alternative design is to use the `K=V` notation proposed in the
associated items RFC for specify the values of default type
parameters. However, this is somewhat odd, because default type
parameters appear in a positional list, and thus it is suprising that
values for the non-defaulted parameters are given positionally, but
values for the defaulted type parameters are given with labels.

Another alternative would to simply prohibit users from specifying the
value of a defaulted type parameter unless values are given for all
previous defaulted typed parameters. But this is clearly annoying in
those cases where defaulted type parameters represent distinct axes of
customization.

# Hat Tip

eddyb introduced defaulted type parameters and also opened the first
pull request that used them to inform inference.
