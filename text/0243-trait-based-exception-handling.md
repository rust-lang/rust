- Feature-gates: `question_mark`, `try_catch`
- Start Date: 2014-09-16
- RFC PR #: [rust-lang/rfcs#243](https://github.com/rust-lang/rfcs/pull/243)
- Rust Issue #: [rust-lang/rust#31436](https://github.com/rust-lang/rust/issues/31436)


# Summary

Add syntactic sugar for working with the `Result` type which models common
exception handling constructs.

The new constructs are:

 * An `?` operator for explicitly propagating "exceptions".

 * A `catch { ... }` expression for conveniently catching and handling
   "exceptions".

The idea for the `?` operator originates from [RFC PR 204][204] by
[@aturon](https://github.com/aturon).

[204]: https://github.com/rust-lang/rfcs/pull/204


# Motivation and overview

Rust currently uses the `enum Result` type for error handling. This solution is
simple, well-behaved, and easy to understand, but often gnarly and inconvenient
to work with. We would like to solve the latter problem while retaining the
other nice properties and avoiding duplication of functionality.

We can accomplish this by adding constructs which mimic the exception-handling
constructs of other languages in both appearance and behavior, while improving
upon them in typically Rustic fashion. Their meaning can be specified by a
straightforward source-to-source translation into existing language constructs,
plus a very simple and obvious new one. (They may also, but need not
necessarily, be implemented in this way.)

These constructs are strict additions to the existing language, and apart from
the issue of keywords, the legality and behavior of all currently existing Rust
programs is entirely unaffected.

The most important additions are a postfix `?` operator for
propagating "exceptions" and a `catch {..}` expression for catching
them. By an "exception", for now, we essentially just mean the `Err`
variant of a `Result`, though the Unresolved Questions includes some
discussion of extending to other types.

## `?` operator

The postfix `?` operator can be applied to `Result` values and is equivalent to
the current `try!()` macro. It either returns the `Ok` value directly, or
performs an early exit and propagates the `Err` value further out. (So given
`my_result: Result<Foo, Bar>`, we have `my_result?: Foo`.) This allows it to be
used for e.g. conveniently chaining method calls which may each "throw an
exception":

    foo()?.bar()?.baz()

Naturally, in this case the types of the "exceptions thrown by" `foo()` and
`bar()` must unify. Like the current `try!()` macro, the `?` operator will also
perform an implicit "upcast" on the exception type.

When used outside of a `catch` block, the `?` operator propagates the exception to
the caller of the current function, just like the current `try!` macro does. (If
the return type of the function isn't a `Result`, then this is a type error.)
When used inside a `catch` block, it propagates the exception up to the innermost
`catch` block, as one would expect.

Requiring an explicit `?` operator to propagate exceptions strikes a very
pleasing balance between completely automatic exception propagation, which most
languages have, and completely manual propagation, which we'd have apart from
the `try!` macro. It means that function calls remain simply function calls
which return a result to their caller, with no magic going on behind the scenes;
and this also *increases* flexibility, because one gets to choose between
propagation with `?` or consuming the returned `Result` directly.

The `?` operator itself is suggestive, syntactically lightweight enough to not
be bothersome, and lets the reader determine at a glance where an exception may
or may not be thrown. It also means that if the signature of a function changes
with respect to exceptions, it will lead to type errors rather than silent
behavior changes, which is a good thing. Finally, because exceptions are tracked
in the type system, and there is no silent propagation of exceptions, and all
points where an exception may be thrown are readily apparent visually, this also
means that we do not have to worry very much about "exception safety".

### Exception type upcasting

In a language with checked exceptions and subtyping, it is clear that if a
function is declared as throwing a particular type, its body should also be able
to throw any of its subtypes. Similarly, in a language with structural sum types
(a.k.a. anonymous `enum`s, polymorphic variants), one should be able to throw a
type with fewer cases in a function declaring that it may throw a superset of
those cases. This is essentially what is achieved by the common Rust practice of
declaring a custom error `enum` with `From` `impl`s for each of the upstream
error types which may be propagated:

    enum MyError {
        IoError(io::Error),
        JsonError(json::Error),
        OtherError(...)
    }

    impl From<io::Error> for MyError { ... }
    impl From<json::Error> for MyError { ... }

Here `io::Error` and `json::Error` can be thought of as subtypes of `MyError`,
with a clear and direct embedding into the supertype.

The `?` operator should therefore perform such an implicit conversion, in the
nature of a subtype-to-supertype coercion. The present RFC uses the
`std::convert::Into` trait for this purpose (which has a blanket `impl`
forwarding from `From`). The precise requirements for a conversion to be "like"
a subtyping coercion are an open question; see the "Unresolved questions"
section.

## `catch` expressions

This RFC also introduces an expression form `catch {..}`, which serves
to "scope" the `?` operator. The `catch` operator executes its
associated block. If no exception is thrown, then the result is
`Ok(v)` where `v` is the value of the block. Otherwise, if an
exception is thrown, then the result is `Err(e)`. Note that unlike
other languages, a `catch` block always catches all errors, and they
must all be coercable to a single type, as a `Result` only has a
single `Err` type. This dramatically simplifies thinking about the
behavior of exception-handling code.

Note that `catch { foo()? }` is essentially equivalent to `foo()`.
`catch` can be useful if you want to coalesce *multiple* potential
exceptions -- `catch { foo()?.bar()?.baz()? }` -- into a single
`Result`, which you wish to then e.g. pass on as-is to another
function, rather than analyze yourself. (The last example could also
be expressed using a series of `and_then` calls.)

# Detailed design

The meaning of the constructs will be specified by a source-to-source
translation. We make use of an "early exit from any block" feature
which doesn't currently exist in the language, generalizes the current
`break` and `return` constructs, and is independently useful.

## Early exit from any block

The capability can be exposed either by generalizing `break` to take an optional
value argument and break out of any block (not just loops), or by generalizing
`return` to take an optional lifetime argument and return from any block, not
just the outermost block of the function. This feature is only used in this RFC
as an explanatory device, and implementing the RFC does not require exposing it,
so I am going to arbitrarily choose the `break` syntax for the following and
won't discuss the question further.

So we are extending `break` with an optional value argument: `break 'a EXPR`.
This is an expression of type `!` which causes an early return from the
enclosing block specified by `'a`, which then evaluates to the value `EXPR` (of
course, the type of `EXPR` must unify with the type of the last expression in
that block). This works for any block, not only loops.

A completely artificial example:

    'a: {
        let my_thing = if have_thing() {
            get_thing()
        } else {
            break 'a None
        };
        println!("found thing: {}", my_thing);
        Some(my_thing)
    }

Here if we don't have a thing, we escape from the block early with `None`.

If no value is specified, it defaults to `()`: in other words, the current
behavior. We can also imagine there is a magical lifetime `'fn` which refers to
the lifetime of the whole function: in this case, `break 'fn` is equivalent to
`return`.

Again, this RFC does not propose generalizing `break` in this way at this time:
it is only used as a way to explain the meaning of the constructs it does
propose.


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
            Err(e) => break 'here Err(e.into())
        }

   Where `'here` refers to the innermost enclosing `catch` block, or to `'fn` if
   there is none.

   The `?` operator has the same precedence as `.`.

 * Construct:

        catch {
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
                Err(e) => break 'here Err(e.into())
            }.bar())
        }

The fully expanded translations get quite gnarly, but that is why it's good that
you don't have to write them!

In general, the types of the defined constructs should be the same as the types
of their definitions.

(As noted earlier, while the behavior of the constructs can be *specified* using
a source-to-source translation in this manner, they need not necessarily be
*implemented* this way.)

As a result of this RFC, both `Into` and `Result` would have to become lang
items.


## Laws

Without any attempt at completeness, here are some things which should be true:

 * `catch { foo()          }                      ` = `Ok(foo())`
 * `catch { Err(e)?        }                      ` = `Err(e.into())`
 * `catch { try_foo()?     }                      ` = `try_foo().map_err(Into::into)`

(In the above, `foo()` is a function returning any type, and `try_foo()` is a
function returning a `Result`.)

## Feature gates

The two major features here, the `?` syntax and `catch` expressions,
will be tracked by independent feature gates. Each of the features has
a distinct motivation, and we should evaluate them independently.

# Unresolved questions

These questions should be satisfactorally resolved before stabilizing the
relevant features, at the latest.

## Optional `match` sugar

Originally, the RFC included the ability to `match` the errors caught
by a `catch` by writing `catch { .. } match { .. }`, which could be translated
as follows:

 * Construct:

        catch {
            foo()?.bar()
        } match {
            A(a) => baz(a),
            B(b) => quux(b)
        }

  Shallow:

        match (catch {
            foo()?.bar()
        }) {
            Ok(a) => a,
            Err(e) => match e {
                A(a) => baz(a),
                B(b) => quux(b)
            }
        }

   Deep:

        match ('here: {
            Ok(match foo() {
                Ok(a) => a,
                Err(e) => break 'here Err(e.into())
            }.bar())
        }) {
            Ok(a) => a,
            Err(e) => match e {
                A(a) => baz(a),
                B(b) => quux(b)
            }
        }

However, it was removed for the following reasons:

- The `catch` (originally: `try`) keyword adds the real expressive "step up" here, the `match` (originally: `catch`) was just sugar for `unwrap_or`.
- It would be easy to add further sugar in the future, once we see how `catch` is used (or not used) in practice.
- There was some concern about potential user confusion about two aspects:
  - `catch { }` yields a `Result<T,E>` but `catch { } match { }` yields just `T`;
  - `catch { } match { }` handles all kinds of errors, unlike `try/catch` in other languages which let you pick and choose.
 
It may be worth adding such a sugar in the future, or perhaps a
variant that binds irrefutably and does not immediately lead into a
`match` block.
 
## Choice of keywords

The RFC to this point uses the keyword `catch`, but there are a number
of other possibilities, each with different advantages and drawbacks:

 * `try { ... } catch { ... }`

 * `try { ... } match { ... }`

 * `try { ... } handle { ... }`

 * `catch { ... } match { ... }`

 * `catch { ... } handle { ... }`

 * `catch ...` (without braces or a second clause)

Among the considerations:

 * Simplicity. Brevity.

 * Following precedent from existing, popular languages, and familiarity with
   respect to their analogous constructs.

 * Fidelity to the constructs' actual behavior. For instance, the first clause
   always catches the "exception"; the second only branches on it.

 * Consistency with the existing `try!()` macro. If the first clause is called
   `try`, then `try { }` and `try!()` would have essentially inverse meanings.

 * Language-level backwards compatibility when adding new keywords. I'm not sure
   how this could or should be handled.

## Semantics for "upcasting"

What should the contract for a `From`/`Into` `impl` be? Are these even the right
`trait`s to use for this feature?

Two obvious, minimal requirements are:

 * It should be pure: no side effects, and no observation of side effects. (The
   result should depend *only* on the argument.)

 * It should be total: no panics or other divergence, except perhaps in the case
   of resource exhaustion (OOM, stack overflow).

The other requirements for an implicit conversion to be well-behaved in the
context of this feature should be thought through with care.

Some further thoughts and possibilities on this matter, only as brainstorming:

 * It should be "like a coercion from subtype to supertype", as described
   earlier. The precise meaning of this is not obvious.

 * A common condition on subtyping coercions is coherence: if you can
   compound-coerce to go from `A` to `Z` indirectly along multiple different
   paths, they should all have the same end result.

 * It should be lossless, or in other words, injective: it should map each
   observably-different element of the input type to observably-different
   elements of the output type. (Observably-different means that it is possible
   to write a program which behaves differently depending on which one it gets,
   modulo things that "shouldn't count" like observing execution time or
   resource usage.)

 * It should be unambiguous, or preserve the meaning of the input:
   `impl From<u8> for u32` as `x as u32` feels right; as `(x as u32) * 12345`
   feels wrong, even though this is perfectly pure, total, and injective. What
   this means precisely in the general case is unclear.

 * The types converted between should the "same kind of thing": for instance,
   the *existing* `impl From<u32> for Ipv4Addr` feels suspect on this count.
   (This perhaps ties into the subtyping angle: `Ipv4Addr` is clearly not a
   supertype of `u32`.)

## Forwards-compatibility

If we later want to generalize this feature to other types such as `Option`, as
described below, will we be able to do so while maintaining backwards-compatibility?

## Monadic do notation

There have been many comparisons drawn between this syntax and monadic
do notation. Before stabilizing, we should determine whether we plan
to make changes to better align this feature with a possible `do`
notation (for example, by removing the implicit `Ok` at the end of a
`catch` block). Note that such a notation would have to extend the
standard monadic bind to accommodate rich control flow like `break`,
`continue`, and `return`.

# Drawbacks

 * Increases the syntactic surface area of the language.

 * No expressivity is added, only convenience. Some object to "there's more than
   one way to do it" on principle.

 * If at some future point we were to add higher-kinded types and syntactic
   sugar for monads, a la Haskell's `do` or Scala's `for`, their functionality
   may overlap and result in redundancy. However, a number of challenges would
   have to be overcome for a generic monadic sugar to be able to fully supplant
   these features: the integration of higher-kinded types into Rust's type
   system in the first place, the shape of a `Monad` `trait` in a language with
   lifetimes and move semantics, interaction between the monadic control flow
   and Rust's native control flow (the "ambient monad"), automatic upcasting of
   exception types via `Into` (the exception (`Either`, `Result`) monad normally
   does not do this, and it's not clear whether it can), and potentially others.


# Alternatives

 * Don't.

 * Only add the `?` operator, but not `catch` expressions.

 * Instead of a built-in `catch` construct, attempt to define one using
   macros. However, this is likely to be awkward because, at least, macros may
   only have their contents as a single block, rather than two. Furthermore,
   macros are excellent as a "safety net" for features which we forget to add
   to the language itself, or which only have specialized use cases; but
   generally useful control flow constructs still work better as language
   features.

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

 * Wait (and hope) for HKTs and generic monad sugar.

[notes]: https://github.com/glaebhoerl/rust-notes/blob/268266e8fbbbfd91098d3bea784098e918b42322/my_rfcs/Exceptions.txt


# Future possibilities

## Expose a generalized form of `break` or `return` as described

This RFC doesn't propose doing so at this time, but as it would be an independently useful feature, it could be added as well.

## `throw` and `throws`

It is possible to carry the exception handling analogy further and also add
`throw` and `throws` constructs.

`throw` is very simple: `throw EXPR` is essentially the same thing as
`Err(EXPR)?`; in other words it throws the exception `EXPR` to the innermost
`catch` block, or to the function's caller if there is none.

A `throws` clause on a function:

    fn foo(arg: Foo) -> Bar throws Baz { ... }

would mean that instead of writing `return Ok(foo)` and `return Err(bar)` in the
body of the function, one would write `return foo` and `throw bar`, and these
are implicitly turned into `Ok` or `Err` for the caller. This removes syntactic
overhead from both "normal" and "throwing" code paths and (apart from `?` to
propagate exceptions) matches what code might look like in a language with
native exceptions.

## Generalize over `Result`, `Option`, and other result-carrying types

`Option<T>` is completely equivalent to `Result<T, ()>` modulo names, and many
common APIs use the `Option` type, so it would be useful to extend all of the
above syntax to `Option`, and other (potentially user-defined)
equivalent-to-`Result` types, as well.

This can be done by specifying a trait for types which can be used to "carry"
either a normal result or an exception. There are several different, equivalent
ways to formulate it, which differ in the set of methods provided, but the
meaning in any case is essentially just that you can choose some types `Normal`
and `Exception` such that `Self` is isomorphic to `Result<Normal, Exception>`.

Here is one way:

    #[lang(result_carrier)]
    trait ResultCarrier {
        type Normal;
        type Exception;
        fn embed_normal(from: Normal) -> Self;
        fn embed_exception(from: Exception) -> Self;
        fn translate<Other: ResultCarrier<Normal=Normal, Exception=Exception>>(from: Self) -> Other;
    }

For greater clarity on how these methods work, see the section on `impl`s below.
(For a simpler formulation of the trait using `Result` directly, see further
below.)

The `translate` method says that it should be possible to translate to any
*other* `ResultCarrier` type which has the same `Normal` and `Exception` types.
This may not appear to be very useful, but in fact, this is what can be used to
inspect the result, by translating it to a concrete, known type such as
`Result<Normal, Exception>` and then, for example, pattern matching on it.

Laws:

 1. For all `x`,       `translate(embed_normal(x): A): B      ` = `embed_normal(x): B`.
 2. For all `x`,       `translate(embed_exception(x): A): B   ` = `embed_exception(x): B`.
 3. For all `carrier`, `translate(translate(carrier: A): B): A` = `carrier: A`.

Here I've used explicit type ascription syntax to make it clear that e.g. the
types of `embed_` on the left and right hand sides are different.

The first two laws say that embedding a result `x` into one result-carrying type
and then translating it to a second result-carrying type should be the same as
embedding it into the second type directly.

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
translate to a different result-carrying type and then back without losing
information.

The `bool` impl may be surprising, or not useful, but it *is* well-behaved:
`bool` is, after all, isomorphic to `Result<(), ()>`.

### Other miscellaneous notes about `ResultCarrier`

 * Our current lint for unused results could be replaced by one which warns for
   any unused result of a type which implements `ResultCarrier`.

 * If there is ever ambiguity due to the result-carrying type being
   underdetermined (experience should reveal whether this is a problem in
   practice), we could resolve it by defaulting to `Result`.

 * Translating between different result-carrying types with the same `Normal`
   and `Exception` types *should*, but may not necessarily *currently* be, a
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
`Result`, and one out), whereas with the `ResultCarrier` trait in the main text,
they only require one.

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
