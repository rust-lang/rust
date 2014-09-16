- Start Date: 2014-09-16
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)


# Summary

Add sugar for working with existing algebraic datatypes such as `Result` and
`Option`. Put another way, use types such as `Result` and `Option` to model
common exception handling constructs.

Add a trait which precisely spells out the abstract interface and requirements
for such types.

The new constructs are:

 * An `?` operator for explicitly propagating exceptions.

 * A `try`..`catch` construct for conveniently catching and handling exceptions.

 * (Potentially) a `throw` operator, and `throws` sugar for function signatures.

The idea for the `?` operator originates from [RFC PR 204][204] by @aturon.

[204]: https://github.com/rust-lang/rfcs/pull/204


# Motivation and overview

Rust currently uses algebraic `enum` types `Option` and `Result` for error
handling. This solution is simple, well-behaved, and easy to understand, but
often gnarly and inconvenient to work with. We would like to solve the latter
problem while retaining the other nice properties and avoiding duplication of
functionality.

We can accomplish this by adding constructs which mimic the exception-handling
constructs of other languages in both appearance and behavior, while improving
upon them in typically Rustic fashion. These constructs are well-behaved in a
very precise sense and their meaning can be specified by a straightforward
source-to-source translation into existing language constructs (plus a very
simple and obvious new one). (They may also, but need not necessarily, be
implemented in this way.)

These constructs are strict additions to the existing language, and apart from
the issue of keywords, the legality and behavior of all currently existing Rust
programs is entirely unaffected.

The most important additions are a postfix `?` operator for propagating
"exceptions" and a `try`..`catch` block for catching and handling them. By an
"exception", we more or less just mean the `None` variant of an `Option` or the
`Err` variant of a `Result`. (See the "Detailed design" section for more
precision.)

## `?` operator

The postfix `?` operator can be applied to expressions of types like `Option`
and `Result` which contain either a "success" or an "exception" value, and can
be thought of as a generalization of the current `try! { }` macro. It either
returns the "success" value directly, or performs an early exit and propagates
the "exception" value further out. (So given `my_result: Result<Foo, Bar>`, we
have `my_result?: Foo`.) This allows it to be used for e.g. conveniently
chaining method calls which may each "throw an exception":

    foo()?.bar()?.baz()

(Naturally, in this case the types of the "exceptions thrown by" `foo()` and
`bar()` must unify.)

When used outside of a `try` block, the `?` operator propagates the exception to
the caller of the current function, just like the current `try!` macro does. (If
the return type of the function isn't one, like `Result`, that's capable of
carrying the exception, then this is a type error.) When used inside a `try`
block, it propagates the exception up to the innermost `try` block, as one would
expect.

Requiring an explicit `?` operator to propagate exceptions strikes a very
pleasing balance between completely automatic exception propagation, which most
languages have, and completely manual propagation, which we currently have
(apart from the `try!` macro to lessen the pain). It means that function calls
remain simply function calls which return a result to their caller, with no
magic going on behind the scenes; and this also *increases* flexibility, because
one gets to choose between propagation with `?` or consuming the returned
`Result` directly.

The `?` operator itself is suggestive, syntactically lightweight enough to not
be bothersome, and lets the reader determine at a glance where an exception may
or may not be thrown. It also means that if the signature of a function changes
with respect to exceptions, it will lead to type errors rather than silent
behavior changes, which is always a good thing. Finally, because exceptions are
tracked in the type system, there is no silent propagation of exceptions, and
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
a single `Err` type); and there may only be a single `catch` block, which
catches all exceptions. This dramatically simplifies matters and allows for nice
properties.

There are two variations on the `try`..`catch` theme, each of which is more
convenient in different circumstances.

 1. `try { EXPR } catch IRR-PAT { EXPR }`

    For example:

        try {
            foo()?.bar()?
        } catch e {
            let x = baz(e);
            quux(x, e);
        }

    Here the caught exception is bound to an irrefutable pattern immediately
    following the `catch`.
    This form is convenient when one does not wish to do case analysis on the
    caught exception.

 2. `try { EXPR } catch { PAT => EXPR, PAT => EXPR, ... }`

    For example:

        try {
            foo()?.bar()?
        } catch {
            Red(rex)  => baz(rex),
            Blue(bex) => quux(bex)
        }

    Here the `catch` is not immediately followed by a pattern; instead, its body
    performs a `match` on the caught exception directly, using any number of
    refutable patterns.
    This form is convenient when one *does* wish to do case analysis on the
    caught exception.

While it may appear to be extravagant to provide both forms, there is reason to
do so: either form on its own leads to unavoidable rightwards drift under some
circumstances.

The first form leads to rightwards drift if one wishes to `match` on the caught
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

The second form leads to rightwards drift if one wishes to do more complex
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

Therefore, neither form can be considered strictly superior to the other, and it
is preferable to simply provide both.

Finally, it is also possible to write a `try` block *without* a `catch` block:

 3. `try { EXPR }`

    In this case the `try` block evaluates directly to a `Result`-like type
    containing either the value of `EXPR`, or the exception which was thrown.
    For instance, `try { foo()? }` is essentially equivalent to `foo()`.
    This can be useful if you want to coalesce *multiple* potential exceptions -
    `try { foo()?.bar()?.baz()? }` - into a single `Result`, which you wish to
    then e.g. pass on as-is to another function, rather than analyze yourself.

## (Optional) `throw` and `throws`

It is possible to carry the exception handling analogy further and also add
`throw` and `throws` constructs.

`throw` is very simple: `throw EXPR` is essentially the same thing as
`Err(EXPR)?`; in other words it throws the exception `EXPR` to the innermost
`try` block, or to the function's caller if there is none.

A `throws` clause on a function:

    fn foo(arg; Foo) -> Bar throws Baz { ... }

would do two things:

 * Less importantly, it would make the function polymorphic over the
   `Result`-like type used to "carry" exceptions.

 * More importantly, it means that instead of writing `return Ok(foo)` and
   `return Err(bar)` in the body of the function, one would write `return foo`
   and `throw bar`, and these are implicitly embedded as the "success" or
   "exception" value in the carrier type. This removes syntactic overhead from
   both "normal" and "throwing" code paths and (apart from `?` to propagate
   exceptions) matches what code might look like in a language with native
   exceptions.

(This could potentially be extended to allow writing `throws` clauses on `fn`
and closure *types*, desugaring to a type parameter with a `Carrier` bound on
the parent item (e.g. a HOF), but this would be considerably more involved, and
it's not clear whether there is value in doing so.)


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
going to arbitrarily choose the `return` syntax for the following and won't
discuss the question further.

So we are extending `return` with an optional lifetime argument: `return 'a
EXPR`. This is an expression of type `!` which causes an early return from the
enclosing block specified by `'a`, which then evaluates to the value `EXPR` (of
course, the type of `EXPR` must unify with the type of the last expression in
that block).

A completely artificial example:

    'a: {
        let my_thing = if have_thing {
            get_thing()
        } else {
            return 'a None
        };
        println!("found thing: {}", my_thing);
        Some(my_thing)
    }

Here if we don't have a thing, we escape from the block early with `None`.

If no lifetime is specified, it defaults to returning from the whole function:
in other words, the current behavior. We can pretend there is a magical lifetime
`'fn` which refers to the outermost block of the current function, which is the
default.

## The trait

Here we specify the trait for types which can be used to "carry" either a normal
result or an exception. There are several different, completely equivalent ways
to formulate it, which differ only in the set of methods: for other
possibilities, see the appendix.

    #[lang(carrier)]
    trait Carrier {
        type Normal;
        type Exception;
        fn embed_normal(from: Normal) -> Self;
        fn embed_exception(from: Exception) -> Self;
        fn translate<Other: Carrier<Normal=Normal, Exception=Exception>>(from: Self) -> Other;
    }

This trait basically just states that `Self` is isomorphic to
`Result<Normal, Exception>` for some types `Normal` and `Exception`. For greater
clarity on how these methods work, see the section on `impl`s below. (For a
simpler formulation of the trait using `Result` directly, see the appendix.)

The `translate` method says that it should be possible to translate to any
*other* `Carrier` type which has the same `Normal` and `Exception` types. This
can be used to inspect the value by translating to a concrete type such as
`Result<Normal, Exception>` and then, for example, pattern matching on it.

Laws:

 1. For all `x`,       `translate(embed_normal(x): A): B      ` = `embed_normal(x): B`.
 2. For all `x`,       `translate(embed_exception(x): A): B   ` = `embed_exception(x): B`.
 3. For all `carrier`, `translate(translate(carrier: A): B): A` = `carrier: A`.

Here I've used explicit type ascription syntax to make it clear that e.g. the
types of `embed_` on the left and right hand sides are different.

The first two laws say that embedding a result `x` into one carrier type and
then translating it to a second carrier type should be the same as embedding it
into the second type directly.

The third law says that translating to a different carrier type and then
translating back should be the identity function.


## `impl`s of the trait

    impl<T, E> Carrier for Result<T, E> {
        type Normal = T;
        type Exception = E;
        fn embed_normal(a: T) -> Result<T, E> { Ok(a) }
        fn embed_exception(e: E) -> Result<T, E> { Err(e) }
        fn translate<Other: Carrier<Normal=T, Exception=E>>(result: Result<T, E>) -> Other {
            match result {
                Ok(a)  => Other::embed_normal(a),
                Err(e) => Other::embed_exception(e)
            }
        }
    }

As we can see, `translate` can be implemented by deconstructing ourself and then
re-embedding the contained value into the other carrier type.

    impl<T> Carrier for Option<T> {
        type Normal = T;
        type Exception = ();
        fn embed_normal(a: T) -> Option<T> { Some(a) }
        fn embed_exception(e: ()) -> Option<T> { None }
        fn translate<Other: Carrier<Normal=T, Exception=()>>(option: Option<T>) -> Other {
            match option {
                Some(a) => Other::embed_normal(a),
                None    => Other::embed_exception(())
            }
        }
    }

Potentially also:

    impl Carrier for bool {
        type Normal = ();
        type Exception = ();
        fn embed_normal(a: ()) -> bool { true }
        fn embed_exception(e: ()) -> bool { false }
        fn translate<Other: Carrier<Normal=(), Exception=()>>(b: bool) -> Other {
            match b {
                true  => Other::embed_normal(()),
                false => Other::embed_exception(())
            }
        }
    }

The laws should be sufficient to rule out any "icky" impls. For example, an impl
for `Vec` where an exception is represented as the empty vector, and a normal
result as a single-element vector: here the third law fails, because if the
`Vec` has more than element *to begin with*, then it's not possible to translate
to a different carrier type and then back without losing information.

The `bool` impl may be surprising, or not useful, but it *is* well-behaved:
`bool` is, after all, isomorphic to `Result<(), ()>`. This `impl` may be
included or not; I don't have a strong opinion about it.

## Definition of constructs

Finally we have the definition of the new constructs in terms of a
source-to-source translation.

In each case except the first, I will provide two definitions: a single-step
"shallow" desugaring which is defined in terms of the previously defined new
constructs, and a "deep" one which is "fully expanded".

Of course, these could be defined in many equivalent ways: the below definitions
are merely one way.

 * Construct:

        throw EXPR

   Shallow:

        return 'here Carrier::embed_exception(EXPR)

   Where `'here` refers to the innermost enclosing `try` block, or to `'fn` if
   there is none. As with `return`, `EXPR` may be omitted and defaults to `()`.

 * Construct:

        EXPR?

   Shallow:

        match translate(EXPR) {
            Ok(a)  => a,
            Err(e) => throw e
        }

   Deep:

        match translate(EXPR) {
            Ok(a)  => a,
            Err(e) => return 'here Carrier::embed_exception(e)
        }

 * Construct:

        try {
            foo()?.bar()
        }

   Shallow:

        'here: {
            Carrier::embed_normal(foo()?.bar())
        }

   Deep:

        'here: {
            Carrier::embed_normal(match translate(foo()) {
                Ok(a) => a,
                Err(e) => return 'here Carrier::embed_exception(e)
            }.bar())
        }

 * Construct:

        try {
            foo()?.bar()
        } catch e {
            baz(e)
        }

   Shallow:

        match try {
            foo()?.bar()
        } {
            Ok(a) => a,
            Err(e) => baz(e)
        }

   Deep:

        match 'here: {
            Carrier::embed_normal(match translate(foo()) {
                Ok(a) => a,
                Err(e) => return 'here Carrier::embed_exception(e)
            }.bar())
        } {
            Ok(a) => a,
            Err(e) => baz(e)
        }

 * Construct:

        try {
            foo()?.bar()
        } catch {
            A(a) => baz(a),
            B(b) => quux(b)
        }

   Shallow:

        try {
            foo()?.bar()
        } catch e {
            match e {
                A(a) => baz(a),
                B(b) => quux(b)
            }
        }

   Deep:

        match 'here: {
            Carrier::embed_normal(match translate(foo()) {
                Ok(a) => a,
                Err(e) => return 'here Carrier::embed_exception(e)
            }.bar())
        } {
            Ok(a) => a,
            Err(e) => match e {
                A(a) => baz(a),
                B(b) => quux(b)
            }
        }

 * Construct:

        fn foo(A) -> B throws C {
            CODE
        }

   Shallow:

        fn foo<Car: Carrier<Normal=B, Exception=C>>(A) -> Car {
            try {
                'fn: {
                    CODE
                }
            }
        }

   Deep:

        fn foo<Car: Carrier<Normal=B, Exception=C>>(A) -> Car {
            'here: {
                Carrier::embed_normal('fn: {
                    CODE
                })
            }
        }

   (Here our desugaring runs into a stumbling block, and we resort to a pun: the
   *whole function* should be conceptually wrapped in a `try` block, and a
   `return` inside `CODE` should be embedded as a successful result into the
   carrier, rather than escaping from the `try` block itself. We suggest this by
   putting the "magical lifetime" `'fn` *inside* the `try` block.)

The fully expanded translations get quite gnarly, but that is why it's good that
you don't have to write them!

In general, the types of the defined constructs should be the same as the types
of their definitions.

(As noted earlier, while the behavior of the constructs can be *specified* using
a source-to-source translation in this manner, they need not necessarily be
*implemented* this way.)

## Laws

Without any attempt at completeness, and modulo `translate()` between different
carrier types, here are some things which should be true:

 * `try { foo()      }                   ` = `Ok(foo())`
 * `try { throw e    }                   ` = `Err(e)`
 * `try { foo()?     }                   ` = `foo()`
 * `try { foo()      } catch e {     e  }` = `foo()`
 * `try { throw e    } catch e {     e  }` = `e`
 * `try { Ok(foo()?) } catch e { Err(e) }` = `foo()`

## Misc

 * Our current lint for unused results could be replaced by one which warns for
   any unused result of a type which implements `Carrier`.

 * If there is ever ambiguity due to the carrier type being underdetermined
   (experience should reveal whether this is a problem in practice), we could
   resolve it by defaulting to `Result`. (This would presumably involve making
   `Result` a lang item.)

 * Translating between different carrier types with the same `Normal` and
   `Exception` types *should*, but may not necessarily *currently* be, a no-op
   most of the time.

   We should make it so that:

     * repr(`Option<T>`) = repr(`Result<T, ()>`)
     * repr(`bool`) = repr(`Option<()>`) = repr(`Result<(), ()>`)

   If these hold, then `translate` between these types could in theory be
   compiled down to just a `transmute`. (Whether LLVM is smart enough to do
   this, I don't know.)

 * The `translate()` function smells to me like a natural transformation between
   functors, but I'm not category theorist enough for it to be obvious.


# Drawbacks

 * Adds new constructs to the language.

 * Some people have a philosophical objection to "there's more than one way to
   do it".

 * Relative to first-class checked exceptions, our implementation options are
   constrained: while actual checked exceptions could be implemented in a
   similar way to this proposal, they could also be implemented using unwinding,
   should we choose to do so, and we do not realistically have that option here.


# Alternatives

 * Do nothing.

 * Only add the `?` operator, but not any of the other constructs.

 * Instead of a general `Carrier` trait, define everything directly in terms of
   `Result`. This has precedent in that, for example, the `if`..`else` construct
   is also defined directly in terms of `bool`. (However, this would likely also
   lead to removing `Option` from the standard library in favor of
   `Result<_, ()>`.)

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


# Unresolved questions

 * What should the precedence of the `?` operator be?

 * Should we add `throw` and/or `throws`?

 * Should we have `impl Carrier for bool`?

 * Should we also add the "early return from any block" feature along with this
   proposal, or should that be considered separately? (If we add it: should we
   do it by generalizing `break` or `return`?)


# Appendices

## Alternative formulations of the `Carrier` trait

All of these have the form:

    trait Carrier {
        type Normal;
        type Exception;
        ...methods...
    }

and differ only in the methods, which will be given.

### Explicit isomorphism with `Result`

    fn from_result(Result<Normal, Exception>) -> Self;
    fn to_result(Self) -> Result<Normal, Exception>;

This is, of course, the simplest possible formulation.

The drawbacks are that it, in some sense, privileges `Result` over other
potentially equivalent types, and that it may be less efficient for those types:
for any non-`Result` type, every operation requires two method calls (one into
`Result`, and one out), whereas with the `Carrier` trait in the main text, they
only require one.

Laws:

  * For all `x`, `from_result(to_result(x))` = `x`.
  * For all `x`, `to_result(from_result(x))` = `x`.

Laws for the remaining formulations below are left as an exercise for the
reader.

### Avoid privileging `Result`, most naive version

    fn embed_normal(Normal) -> Self;
    fn embed_exception(Exception) -> Self;
    fn is_normal(&Self) -> bool;
    fn is_exception(&Self) -> bool;
    fn assert_normal(Self) -> Normal;
    fn assert_exception(Self) -> Exception;

Of course this is horrible.

### Destructuring with HOFs (a.k.a. Church/Scott-encoding)

    fn embed_normal(Normal) -> Self;
    fn embed_exception(Exception) -> Self;
    fn match_carrier<T>(Self, FnOnce(Normal) -> T, FnOnce(Exception) -> T) -> T;

This is probably the right approach for Haskell, but not for Rust.

With this formulation, because they each take ownership of them, the two
closures may not even close over the same variables!

### Destructuring with HOFs, round 2

    trait BiOnceFn {
        type ArgA;
        type ArgB;
        type Ret;
        fn callA(Self, ArgA) -> Ret;
        fn callB(Self, ArgB) -> Ret;
    }

    trait Carrier {
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
