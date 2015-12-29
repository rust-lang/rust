- Start Date: 2014-09-16
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)


# Summary

Add syntactic sugar for working with the `Result` type which models common exception handling constructs.

The new constructs are:

 * An `?` operator for explicitly propagating "exceptions".

 * A `try`..`catch` construct for conveniently catching and handling "exceptions".

The idea for the `?` operator originates from [RFC PR 204][204] by [@aturon](https://github.com/aturon).

[204]: https://github.com/rust-lang/rfcs/pull/204


# Motivation and overview

Rust currently uses the `enum Result` type for error
handling. This solution is simple, well-behaved, and easy to understand, but
often gnarly and inconvenient to work with. We would like to solve the latter
problem while retaining the other nice properties and avoiding duplication of
functionality.

We can accomplish this by adding constructs which mimic the exception-handling
constructs of other languages in both appearance and behavior, while improving
upon them in typically Rustic fashion. Their meaning can be specified by a straightforward
source-to-source translation into existing language constructs, plus a very
simple and obvious new one. (They may also, but need not necessarily, be
implemented in this way.)

These constructs are strict additions to the existing language, and apart from
the issue of keywords, the legality and behavior of all currently existing Rust
programs is entirely unaffected.

The most important additions are a postfix `?` operator for propagating
"exceptions" and a `try`..`catch` block for catching and handling them. By an
"exception", we essentially just mean the `Err` variant of a `Result`. (See the "Detailed design" section for more
precision.)


## `?` operator

The postfix `?` operator can be applied to `Result` values and is equivalent to the current `try!()` macro. It either
returns the `Ok` value directly, or performs an early exit and propagates
the `Err` value further out. (So given `my_result: Result<Foo, Bar>`, we
have `my_result?: Foo`.) This allows it to be used for e.g. conveniently
chaining method calls which may each "throw an exception":

    foo()?.bar()?.baz()

(Naturally, in this case the types of the "exceptions thrown by" `foo()` and
`bar()` must unify.)

When used outside of a `try` block, the `?` operator propagates the exception to
the caller of the current function, just like the current `try!` macro does. (If
the return type of the function isn't a `Result`, then this is a type error.) When used inside a `try`
block, it propagates the exception up to the innermost `try` block, as one would
expect.

Requiring an explicit `?` operator to propagate exceptions strikes a very
pleasing balance between completely automatic exception propagation, which most
languages have, and completely manual propagation, which we'd have apart from the `try!` macro. It means that function calls
remain simply function calls which return a result to their caller, with no
magic going on behind the scenes; and this also *increases* flexibility, because
one gets to choose between propagation with `?` or consuming the returned
`Result` directly.

The `?` operator itself is suggestive, syntactically lightweight enough to not
be bothersome, and lets the reader determine at a glance where an exception may
or may not be thrown. It also means that if the signature of a function changes
with respect to exceptions, it will lead to type errors rather than silent
behavior changes, which is a good thing. Finally, because exceptions are
tracked in the type system, and there is no silent propagation of exceptions, and
all points where an exception may be thrown are readily apparent visually, this
also means that we do not have to worry very much about "exception safety".


## `try`..`catch`

Like most other things in Rust, and unlike other languages that I know of,
`try`..`catch` is an *expression*. If no exception is thrown in the `try` block,
the  `try`..`catch` evaluates to the value of `try` block; if an exception is
thrown, it is passed to the `catch` block, and the `try`..`catch` evaluates to
the value of the `catch` block. As with `if`..`else` expressions, the types of
the `try` and `catch` blocks must therefore unify. Unlike other languages, only
a single type of exception may be thrown in the `try` block (a `Result` only has
a single `Err` type); all exceptions are always caught; and there may only be one `catch` block. This dramatically simplifies thinking about the behavior of exception-handling code.

There are two variations on this theme:

 1. `try { EXPR }`

    In this case the `try` block evaluates directly to a `Result`
    containing either the value of `EXPR`, or the exception which was thrown.
    For instance, `try { foo()? }` is essentially equivalent to `foo()`.
    This can be useful if you want to coalesce *multiple* potential exceptions -
    `try { foo()?.bar()?.baz()? }` - into a single `Result`, which you wish to
    then e.g. pass on as-is to another function, rather than analyze yourself.

 2. `try { EXPR } catch { PAT => EXPR, PAT => EXPR, ... }`

    For example:

        try {
            foo()?.bar()?
        } catch {
            Red(rex)  => baz(rex),
            Blue(bex) => quux(bex)
        }

    Here the `catch`
    performs a `match` on the caught exception directly, using any number of
    refutable patterns. This form is convenient for checking and handling the
    caught exception directly.


# Detailed design

The meaning of the constructs will be specified by a source-to-source
translation. We make use of an "early exit from any block" feature which doesn't
currently exist in the language, generalizes the current `break` and `return`
constructs, and is independently useful.


## Early exit from any block

The capability can be exposed either by generalizing `break` to take an optional
value argument and break out of any block (not just loops), or by generalizing
`return` to take an optional lifetime argument and return from any block, not
just the outermost block of the function. This feature is independently useful
and I believe it should be added, but as it is only used here in this RFC as an
explanatory device, and implementing the RFC does not require exposing it, I am
going to arbitrarily choose the `break` syntax for the following and won't
discuss the question further.

So we are extending `break` with an optional value argument: `break 'a EXPR`.
This is an expression of type `!` which causes an early return from the
enclosing block specified by `'a`, which then evaluates to the value `EXPR` (of
course, the type of `EXPR` must unify with the type of the last expression in
that block). This works for any block, not only loops.

A completely artificial example:

    'a: {
        let my_thing = if have_thing {
            get_thing()
        } else {
            break 'a None
        };
        println!("found thing: {}", my_thing);
        Some(my_thing)
    }

Here if we don't have a thing, we escape from the block early with `None`.

If no value is specified, it defaults to `()`: in other words, the current behavior.
We can also imagine there is a magical lifetime `'fn` which refers to the lifetime of the whole function: in this case, `break 'fn` is equivalent to `return`.


## Definition of constructs

Finally we have the definition of the new constructs in terms of a
source-to-source translation.

In each case except the first, I will provide two definitions: a single-step
"shallow" desugaring which is defined in terms of the previously defined new
constructs, and a "deep" one which is "fully expanded".

Of course, these could be defined in many equivalent ways: the below definitions
are merely one way.

 * Construct:

        EXPR?

   Shallow:

        match EXPR {
            Ok(a)  => a,
            Err(e) => break 'here Err(e)
        }

   Where `'here` refers to the innermost enclosing `try` block, or to `'fn` if
   there is none.

   The `?` operator has the same precedence as `.`.

 * Construct:

        try {
            foo()?.bar()
        }

   Shallow:

        'here: {
            Ok(foo()?.bar())
        }

   Deep:

        'here: {
            Ok(match foo() {
                Ok(a) => a,
                Err(e) => break 'here Err(e)
            }.bar())
        }

 * Construct:

        try {
            foo()?.bar()
        } catch {
            A(a) => baz(a),
            B(b) => quux(b)
        }

  Shallow:

        match (try {
            foo()?.bar()
        }) {
            Ok(a) => a,
            Err(e) => match e {
                A(a) => baz(a),
                B(b) => quux(b)
            }
        }

   Deep:

        match 'here: {
            Ok(match foo() {
                Ok(a) => a,
                Err(e) => break 'here Err(e)
            }.bar())
        } {
            Ok(a) => a,
            Err(e) => match e {
                A(a) => baz(a),
                B(b) => quux(b)
            }
        }

The fully expanded translations get quite gnarly, but that is why it's good that
you don't have to write them!

In general, the types of the defined constructs should be the same as the types
of their definitions.

(As noted earlier, while the behavior of the constructs can be *specified* using
a source-to-source translation in this manner, they need not necessarily be
*implemented* this way.)


## Laws

Without any attempt at completeness, here are some things which should be true:

 * `try { foo()      }                   ` = `Ok(foo())`
 * `try { Err(e)?    }                   ` = `Err(e)`
 * `try { foo()?     }                   ` = `foo()`
 * `try { foo()      } catch e {     e  }` = `foo()`
 * `try { Err(e)?    } catch e {     e  }` = `e`
 * `try { Ok(foo()?) } catch e { Err(e) }` = `foo()`


# Drawbacks

 * Increases the syntactic surface area of the language.

 * No expressivity is added, only convenience. Some object to "there's more than one way to do it" on principle.

 * If at some future point we were to add higher-kinded types and syntactic sugar
   for monads, a la Haskell's `do` or Scala's `for`, their functionality may overlap and result in redundancy.
   However, a number of challenges would have to be overcome for a generic monadic sugar to be able to
   fully supplant these features: the integration of higher-kinded types into Rust's type system in the
   first place, the shape of a `Monad` `trait` in a language with lifetimes and move semantics,
   interaction between the monadic control flow and Rust's native control flow (the "ambient monad"),
   automatic upcasting of exception types via `Into` (the exception (`Either`, `Result`) monad normally does not
   do this, and it's not clear whether it can), and potentially others.


# Alternatives

 * Don't.

 * Only add the `?` operator, but not `try`..`catch`.

 * Instead of a built-in `try`..`catch` construct, attempt to define one using
   macros. However, this is likely to be awkward because, at least, macros may
   only have their contents as a single block, rather than two. Furthermore,
   macros are excellent as a "safety net" for features which we forget to add
   to the language itself, or which only have specialized use cases; but generally
   useful control flow constructs still work better as language features.

 * Add [first-class checked exceptions][notes], which are propagated
   automatically (without an `?` operator).

   This has the drawbacks of being a more invasive change and duplicating
   functionality: each function must choose whether to use checked exceptions
   via `throws`, or to return a `Result`. While the two are isomorphic and
   converting between them is easy, with this proposal, the issue does not even
   arise, as exception handling is defined *in terms of* `Result`. Furthermore,
   automatic exception propagation raises the specter of "exception safety": how
   serious an issue this would actually be in practice, I don't know - there's
   reason to believe that it would be much less of one than in C++.

[notes]: https://github.com/glaebhoerl/rust-notes/blob/268266e8fbbbfd91098d3bea784098e918b42322/my_rfcs/Exceptions.txt

 * Wait (and hope) for HKTs and generic monad sugar.


# Future possibilities

## An additional `catch` form to bind the caught exception irrefutably

The `catch` described above immediately passes the caught exception into a `match` block.
It may sometimes be desirable to instead bind it directly to a single variable. That might
look like this:

    try { EXPR } catch IRR-PAT { EXPR }

Where `catch` is followed by any irrefutable pattern (as with `let`).

For example:

    try {
        foo()?.bar()?
    } catch e {
        let x = baz(e);
        quux(x, e);
    }

While it may appear to be extravagant to provide both forms, there is reason to
do so: either form on its own leads to unavoidable rightwards drift under some
circumstances.

The first form leads to rightwards drift if one wishes to do more complex
multi-statement work with the caught exception:

    try {
        foo()?.bar()?
    } catch {
        e => {
            let x = baz(e);
            quux(x, e);
        }
    }

This single case arm is quite redundant and unfortunate.

The second form leads to rightwards drift if one wishes to `match` on the caught
exception:

    try {
        foo()?.bar()?
    } catch e {
        match e {
            Red(rex)  => baz(rex),
            Blue(bex) => quux(bex)
        }
    }

This `match e` is quite redundant and unfortunate.

Therefore, neither form can be considered strictly superior to the other, and it
may be preferable to simply provide both.


## `throw` and `throws`

It is possible to carry the exception handling analogy further and also add
`throw` and `throws` constructs.

`throw` is very simple: `throw EXPR` is essentially the same thing as
`Err(EXPR)?`; in other words it throws the exception `EXPR` to the innermost
`try` block, or to the function's caller if there is none.

A `throws` clause on a function:

    fn foo(arg: Foo) -> Bar throws Baz { ... }

would mean that instead of writing `return Ok(foo)` and
`return Err(bar)` in the body of the function, one would write `return foo`
and `throw bar`, and these are implicitly turned into `Ok` or `Err` for the caller. This removes syntactic overhead from
both "normal" and "throwing" code paths and (apart from `?` to propagate
exceptions) matches what code might look like in a language with native
exceptions.


## Generalize over `Result`, `Option`, and other result-carrying types

`Option<T>` is completely equivalent to `Result<T, ()>` modulo names, and many common APIs
use the `Option` type, so it would make sense to extend all of the above syntax to `Option`,
and other (potentially user-defined) equivalent-to-`Result` types, as well.

This can be done by specifying a trait for types which can be used to "carry" either a normal
result or an exception. There are several different, equivalent ways
to formulate it, which differ in the set of methods provided, but the meaning in any case is essentially just
that you can choose some types `Normal` and `Exception` such that `Self` is isomorphic to `Result<Normal, Exception>`.

Here is one way:

    #[lang(result_carrier)]
    trait ResultCarrier {
        type Normal;
        type Exception;
        fn embed_normal(from: Normal) -> Self;
        fn embed_exception(from: Exception) -> Self;
        fn translate<Other: ResultCarrier<Normal=Normal, Exception=Exception>>(from: Self) -> Other;
    }

For greater clarity on how these methods work, see the section on `impl`s below. (For a
simpler formulation of the trait using `Result` directly, see further below.)

The `translate` method says that it should be possible to translate to any
*other* `ResultCarrier` type which has the same `Normal` and `Exception` types.
This may not appear to be very useful, but in fact, this is what can be used to inspect the result,
by translating it to a concrete type such as `Result<Normal, Exception>` and then, for example, pattern matching on it.

Laws:

 1. For all `x`,       `translate(embed_normal(x): A): B      ` = `embed_normal(x): B`.
 2. For all `x`,       `translate(embed_exception(x): A): B   ` = `embed_exception(x): B`.
 3. For all `carrier`, `translate(translate(carrier: A): B): A` = `carrier: A`.

Here I've used explicit type ascription syntax to make it clear that e.g. the
types of `embed_` on the left and right hand sides are different.

The first two laws say that embedding a result `x` into one result-carrying type and
then translating it to a second result-carrying type should be the same as embedding it
into the second type directly.

The third law says that translating to a different result-carrying type and then
translating back should be a no-op.


## `impl`s of the trait

    impl<T, E> ResultCarrier for Result<T, E> {
        type Normal = T;
        type Exception = E;
        fn embed_normal(a: T) -> Result<T, E> { Ok(a) }
        fn embed_exception(e: E) -> Result<T, E> { Err(e) }
        fn translate<Other: ResultCarrier<Normal=T, Exception=E>>(result: Result<T, E>) -> Other {
            match result {
                Ok(a)  => Other::embed_normal(a),
                Err(e) => Other::embed_exception(e)
            }
        }
    }

As we can see, `translate` can be implemented by deconstructing ourself and then
re-embedding the contained value into the other result-carrying type.

    impl<T> ResultCarrier for Option<T> {
        type Normal = T;
        type Exception = ();
        fn embed_normal(a: T) -> Option<T> { Some(a) }
        fn embed_exception(e: ()) -> Option<T> { None }
        fn translate<Other: ResultCarrier<Normal=T, Exception=()>>(option: Option<T>) -> Other {
            match option {
                Some(a) => Other::embed_normal(a),
                None    => Other::embed_exception(())
            }
        }
    }

Potentially also:

    impl ResultCarrier for bool {
        type Normal = ();
        type Exception = ();
        fn embed_normal(a: ()) -> bool { true }
        fn embed_exception(e: ()) -> bool { false }
        fn translate<Other: ResultCarrier<Normal=(), Exception=()>>(b: bool) -> Other {
            match b {
                true  => Other::embed_normal(()),
                false => Other::embed_exception(())
            }
        }
    }

The laws should be sufficient to rule out any "icky" impls. For example, an impl
for `Vec` where an exception is represented as the empty vector, and a normal
result as a single-element vector: here the third law fails, because if the
`Vec` has more than one element *to begin with*, then it's not possible to
translate to a different result-carrying type and then back without losing information.

The `bool` impl may be surprising, or not useful, but it *is* well-behaved:
`bool` is, after all, isomorphic to `Result<(), ()>`.

### Other miscellaneous notes about `ResultCarrier`

 * Our current lint for unused results could be replaced by one which warns for
   any unused result of a type which implements `ResultCarrier`.

 * If there is ever ambiguity due to the result-carrying type being underdetermined
   (experience should reveal whether this is a problem in practice), we could
   resolve it by defaulting to `Result`.

 * Translating between different result-carrying types with the same `Normal` and
   `Exception` types *should*, but may not necessarily *currently* be, a
   machine-level no-op most of the time.

   We could/should make it so that:

     * repr(`Option<T>`) = repr(`Result<T, ()>`)
     * repr(`bool`) = repr(`Option<()>`) = repr(`Result<(), ()>`)

   If these hold, then `translate` between these types could in theory be
   compiled down to just a `transmute`. (Whether LLVM is smart enough to do
   this, I don't know.)

 * The `translate()` function smells to me like a natural transformation between
   functors, but I'm not category theorist enough for it to be obvious.


### Alternative formulations of the `ResultCarrier` trait

All of these have the form:

    trait ResultCarrier {
        type Normal;
        type Exception;
        ...methods...
    }

and differ only in the methods, which will be given.

#### Explicit isomorphism with `Result`

    fn from_result(Result<Normal, Exception>) -> Self;
    fn to_result(Self) -> Result<Normal, Exception>;

This is, of course, the simplest possible formulation.

The drawbacks are that it, in some sense, privileges `Result` over other
potentially equivalent types, and that it may be less efficient for those types:
for any non-`Result` type, every operation requires two method calls (one into
`Result`, and one out), whereas with the `ResultCarrier` trait in the main text, they
only require one.

Laws:

  * For all `x`, `from_result(to_result(x))` = `x`.
  * For all `x`, `to_result(from_result(x))` = `x`.

Laws for the remaining formulations below are left as an exercise for the
reader.

#### Avoid privileging `Result`, most naive version

    fn embed_normal(Normal) -> Self;
    fn embed_exception(Exception) -> Self;
    fn is_normal(&Self) -> bool;
    fn is_exception(&Self) -> bool;
    fn assert_normal(Self) -> Normal;
    fn assert_exception(Self) -> Exception;

Of course this is horrible.

#### Destructuring with HOFs (a.k.a. Church/Scott-encoding)

    fn embed_normal(Normal) -> Self;
    fn embed_exception(Exception) -> Self;
    fn match_carrier<T>(Self, FnOnce(Normal) -> T, FnOnce(Exception) -> T) -> T;

This is probably the right approach for Haskell, but not for Rust.

With this formulation, because they each take ownership of them, the two
closures may not even close over the same variables!

#### Destructuring with HOFs, round 2

    trait BiOnceFn {
        type ArgA;
        type ArgB;
        type Ret;
        fn callA(Self, ArgA) -> Ret;
        fn callB(Self, ArgB) -> Ret;
    }

    trait ResultCarrier {
        type Normal;
        type Exception;
        fn normal(Normal) -> Self;
        fn exception(Exception) -> Self;
        fn match_carrier<T>(Self, BiOnceFn<ArgA=Normal, ArgB=Exception, Ret=T>) -> T;
    }

Here we solve the environment-sharing problem from above: instead of two objects
with a single method each, we use a single object with two methods! I believe
this is the most flexible and general formulation (which is however a strange
thing to believe when they are all equivalent to each other). Of course, it's
even more awkward syntactically.
