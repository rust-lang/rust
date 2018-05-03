- Feature Name: `try_expr`
- Start Date: 2018-04-04
- RFC PR: [rust-lang/rfcs#2388](https://github.com/rust-lang/rfcs/pull/2388)
- Rust Issue: [rust-lang/rust#50412](https://github.com/rust-lang/rust/issues/50412)

# Summary
[summary]: #summary

[RFC 243]: https://github.com/rust-lang/rfcs/blob/master/text/0243-trait-based-exception-handling.md#choice-of-keywords

[RFC 243] left the choice of keyword for `catch { .. }` expressions unresolved.
This RFC settles the choice of keyword. Namely, it:

1. reserves `try` as a keyword in edition 2018.
2. replaces `do catch { .. }` with `try { .. }`
3. does **not** reserve `catch` as a keyword.

# Motivation
[motivation]: #motivation

[catch_rfc]: https://github.com/rust-lang/rfcs/blob/master/text/0243-trait-based-exception-handling.md

[catch_rfc_motivation]: https://github.com/rust-lang/rfcs/blob/master/text/0243-trait-based-exception-handling.md#catch-expressions

This RFC does not motivate `catch { .. }` or `try { .. }` expressions.
To read the motivation for that, please consult [the original `catch` RFC][catch_rfc_motivation].

## For reserving a keyword

Whatever keyword is chosen, it can't be contextual.

As with `catch { .. }`, the syntactic form `<word> { .. }` where `<word>`
is replaced with any possible keyword would conflict with a struct named
`<word>` as seen in this perfectly legal snippet in Rust 2015,
where `<word>` has been substituted for `try`:

```rust
struct try;
fn main() {
    try {
    };
}
```

### Aside note:

The snippet above emits the following warning:

```
warning: type `try` should have a camel case name such as `Try`
```

which is also the case for `catch`.
This warning decreases the risk that someone has defined a type named `try`
anywhere in the ecosystem which happens to be beneficial to us.

## For reserving `try` specifically

This is discussed in the [rationale for `try`][rationale for try].

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

The keyword `try` will be reserved.
This will allow you to write expressions such as:

```rust
try {
    let x = foo?;
    let y = bar?;
    // Note: OK-wrapping is assumed here, but it is not the goal of this RFC
    // to decide either in favor or against OK-wrapping.
    x + y
}
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

[list of keywords]: https://doc.rust-lang.org/book/second-edition/appendix-01-keywords.html#keywords-currently-in-use

The word `try` is reserved as a keyword in the [list of keywords]
in Rust edition 2018 and later editions.

The keyword `try` is used in "try expressions" of the form `try { .. }`.

# Drawbacks
[drawbacks]: #drawbacks

There are two main drawbacks to the `try` keyword.

## Association with exception handling - Both a pro and con

> I think that there is a belief – one that I have shared from time to time – that it is not helpful to use familiar keywords unless the semantics are a perfect match, the concern being that they will setup an intuition that will lead people astray. I think that is a danger, but it works both ways: those intuitions also help people to understand, particularly in the early days. So it’s a question of “how far along will you get before the differences start to matter” and “how damaging is it if you misunderstand for a while”.
>
> [..]
>
> Rust has a lot of concepts to learn. If we are going to succeed, it’s essential that people can learn them a bit at a time, and that we not throw everything at you at once. I think we should always be on the lookout for places where we can build on intuitions from other languages; it doesn’t have to be a 100% match to be useful.

\- [Niko Matsakis](https://internals.rust-lang.org/t/bikeshed-rename-catch-blocks-to-fallible-blocks/7121/4)

For some people, the association to `try { .. } catch { .. }` in languages such
as Java, and others in the [prior-art] section, is unhelpful wrt. teachability
because they see the explicit, reified, and manually propagated exceptions in
Rust as something very different than the much more implicit exception handling
stories in Java et al.

[`ExceptT`]: https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Except.html#t:ExceptT

However, we make the case that other languages which do have these explicit and
reified exceptions as in Rust also use an exception vocabulary.
Notably, Haskell calls the monad-transformer for adding exceptions [`ExceptT`].

We also argue that even tho we are propagating exceptions manually,
we are following tradition in that other languages have very different
formulations of the exception idea.

The benefit of familiarity, even if not a perfect match, as Niko puts it,
helps in learning, particularly because Rust is not a language in lack of
concepts to learn.

[`try!`]: https://doc.rust-lang.org/nightly/std/macro.try.html

## Breakage of the [`try!`] macro

One possible result of introducing `try` as a keyword be that the old `try!`
macro would break. This could potentially be avoided but with great technical
challenges.

With the prospect of breaking [`try!`], a few notes are in order:

1. `?` was stabilized in 1.13, November 2016, which is roughly 1.4 years since
   the date this RFC was started.
2. `try!` has been "deprecated" since then since:
   > The `?` operator was added to replace `try!` and should be used instead.
3. `try!(expr)` can in virtually all instances be automatically `rustfix`ed
   automatically to `expr?`.
4. There are very few questions on Stack Overflow that mention `try!`.
5. ["The Rust Programming Language", 2nd edition](https://doc.rust-lang.org/book/second-edition/) (book) and "Rust by Example"
   have both already removed all mentions of `try!`.

> So overall I think it’s feasible to reduce the `try!` macro to a historical curiosity to the point it won’t be actively confusing to newbies coming to Rust.

\- [kornel](https://internals.rust-lang.org/t/bikeshed-rename-catch-blocks-to-fallible-blocks/7121/49)

However,

1. There are still plenty of materials out there which mention `try!`.
2. `try!` is essentially the inverse of `try { .. }`.

> Purging from the “collective memories of Rustaceans and Rust materials” is not something that easy.

\- [Manish Goregaokar](https://internals.rust-lang.org/t/bikeshed-rename-catch-blocks-to-fallible-blocks/7121/50)

In the RFC author's opinion however, the sum total benefits of `try { .. }`
seem to outweigh the drawbacks of the difficulty with purging [`try!`] from
our collective memory.

## Inverse semantics of `?`

The `?` postfix operator is sometimes referred to as the "try operator",
and can be seen as having the inverse semantics as `try { .. }`.

To many, this is a drawback. To others, this makes the `?` and `try { .. }`
expression forms more closely related and therefore makes them more findable
in relation to each other.

There is currently some ongoing debate about renaming the `?` operator to
something other than the "try operator". This could help in mitigating the
effects of picking `try` as the keyword.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

## Review considerations

Among the considerations when picking a keyword are, ordered by importance:

1. Fidelity to the construct's actual behavior.

2. Precedent from existing languages
    1. Popularity of the languages.
    2. Fidelity to behavior in those languages.
    3. Familiarity with respect to their analogous constructs.

   See the [prior art][prior-art] the [rationale for try]
   for more discussion on precedent.

2. Brevity.

[`Try`]: https://doc.rust-lang.org/nightly/std/ops/trait.Try.html

4. Consistency with related standard library function conventions.

5. Consistency with the naming of the trait used for `?` (the [`Try`] trait).
   Since the `Try` trait is unstable and the naming of the `?` operator in
   communication is still unsettled, this is not regarded as very important.

6. Degree / Risk of breakage.

7. Consistency with old learning material.

    1. Inversely: The extent of the old learning material

   That is, (in)consistency with `?` and the `try!()` macro.
   If the first clause is called `try`,
   then `try { }` and `try!()` would have essentially inverse meanings.

## Rationale for `try`
[rationale for try]: #rationale-for-try

1. **Fidelity to the construct's actual behavior:** Very high
2. **Precedent from existing languages:** A lot, see [prior-art]
    1. **Popularity of the languages:** Massive accumulated dominance
    2. **Fidelity to behavior in those languages:** Very high
    3. **Familiarity with respect to their analogous constructs:** Very high
3. **Brevity / Length:** 3
4. **Consistency with related libstd fn conventions:** Consistent
5. **Consistency with the naming of the trait used for `?`:** Consistent
6. **Risk of breakage:** High (if we assume `try!` will break, otherwise: Low)
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=try), `std::try!`, but it is technically possible to not break this macro. (unstable: `std::intrinsics::try` so irrelevant)
    - **Used as crate?** [*Yes*](https://crates.io/crates/try). No reverse dependencies. Described as: *"Deprecation warning resistant try macro"*
    - **Usage (sourcegraph):** **27** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+try\s+=|(fn|impl|mod|struct|enum|union|trait)\s+try)\b
    ```
7. **Consistency with old learning material:** Inconsistent ([`try!`])

### Review

This is our choice of keyword, because it:

1. has a massive dominance in both popular and less known languages and is
   sufficiently semantically faithful to what `try` means in those languages.
   Thus, we can leverage people's intuitions and not spend too much of our
   complexity budget.
2. is consistent with the standard library wrt. `Try` and `try_` prefixed methods.
3. it is brief. 
4. it has high fidelity wrt. the concepts it attempts to communicate
   (exception boundary for `?`). This high fidelity is from the perspective of
   a programmers intent, i.e: "I want to try a bunch of stuff in this block".
5. it can be further extended with `catch { .. }` handlers if we wish.

## Alternative: reserving `catch`

1. **Fidelity to the construct's actual behavior:** High
2. **Precedent from existing languages:** Erlang and Tcl, see [prior-art]
3. **Brevity / Length:** 6
4. **Consistency with related libstd fn conventions:** Somewhat consistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=catch)
    - **Used as crate?** [*No*](https://crates.io/crates/catch).
    - **Usage (sourcegraph):** **21** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+catch\s+=|(fn|impl|mod|struct|enum|union|trait)\s+catch)\b
    ```
7. **Consistency with old learning material:** Untaught

### Review

We believe `catch` to be a poor choice of keyword, because it:

1. is used in few in other languages to demarcate the body which can result in
   an exceptional path. Instead, it is almost exclusively used for exception
   handlers of the form: `catch(pat) { recover_expr }`.
4. extending `catch` with handlers will require a different word such as
   `handler` to get `catch { .. } handler(e) { .. }` semantics if we want.
   This inversion compared to a lot of other languages will only harm
   teachability of the language and steal a lot of our strangeness budget.
2. it is less brief than `try`.
3. the consistency wrt. methods in the standard library is low -
   there's only `catch_unwind`, but that has to do with panics,
   not `Try` style exceptions.

However, `catch` has high fidelity wrt. the operational semantics of "catching"
any exceptions in the `try { .. }` block.

## Alternative: keeping `do catch { .. }`

1. **Fidelity to the construct's actual behavior:** Middle
2. **Precedent from existing languages:**
    + `do`: Haskell, Idris
    + `catch`: Erlang and Tcl, see [prior-art]
3. **Brevity / Length:** 8
4. **Consistency with related libstd fn conventions:** Tiny bit consistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Impossible (already reserved keyword)
    - **Used in std:** *No*, the form `$ident $ident` is not a legal identifier.
    - **Used as crate?** *No*, as above.
    - **Usage (sourcegraph):** **0** regex: N/A
7. **Consistency with old learning material:** Untaught

An alternative would be to simply use the `do catch { ... }` syntax we have
in the nightly compiler. However, this syntax was not in the accepted `catch`
RFC and was only a temporarly fix around `catch { .. }` not working.

## Alternative: `do try { .. }`

1. **Fidelity to the construct's actual behavior:** High
2. **Precedent from existing languages:**
    + `do`: Haskell, Idris
    + `try`: A lot, see [prior-art]
    1. **Popularity of the languages:** Massive accumulated dominance
    2. **Fidelity to behavior in those languages:** High
    3. **Familiarity with respect to their analogous constructs:** High
3. **Brevity / Length:** 6 (including space)
4. **Consistency with related libstd fn conventions:** Moderately consistent
5. **Consistency with the naming of the trait used for `?`:** Moderately consistent
6. **Risk of breakage:** Impossible (already reserved keyword)
    - **Used in std:** *No*, the form `$ident $ident` is not a legal identifier.
    - **Used as crate?** *No*, as above.
    - **Usage (sourcegraph):** **0** regex: N/A
7. **Consistency with old learning material:** Untaught

### Review

We could in fact decide to keep the `do`-prefix but change the suffix to `try`.
The benefit here would be two-fold:

+ No keyword `try` would need to be introduced as `do` already is a keyword.
  Therefore, the `try!` macro would not break.

+ An association with monads due to `do`. This can be considered a benfit since
  `try` can be seen as sugar for the family of error monads
  (modulo kinks wrt. imperative flow), and thus,
  the `do` prefix leads to a path of generality if more monads are introduced.

The drawbacks would be:

+ The wider association with monads can be seen as a drawback for those not
  familiar with monads.

+ `do try { .. }` over `try { .. }` adds a small degree of ergonomics overhead
  but not much (3 characters including the space). However, the frequency with
  which the `try { .. }` construct might be used can make the small overhead
  accumulate to a significant overhead when a large codebase is considered.

Other than this, the argument for `do try` over `do catch` boils down to an
argument of `try` over `catch`.

## Alternative: using `do { .. }`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** Haskell, Idris
    1. **Popularity of the languages:** Haskell: Tiobe #42, PYPL #22
    2. **Fidelity to behavior in those languages:** Good
    3. **Familiarity with respect to their analogous constructs:** Poor
3. **Brevity / Length:** 2
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Impossible (already reserved keyword)
7. **Consistency with old learning material:** Untaught

### Review

The keyword `do` was probably originally reserved for two use cases:

1. `do while { .. }`

2. Monadic `do`-notation a la Haskell:

   ```haskell
   stuff = do
       x <- actionX
       y <- actionY x
       z <- actionZ
       sideEffect
       finalAction x y z
   ```

   The which would be translated into the following pseudo-Rust:

   ```rust
   let stuff = do {
       x <- actionX;
       y <- actionY(x);
       z <- actionZ;
       sideEffect;
       finalAction(x, y, z);
   };
   ```

   Or particularly for the `try { .. }` case:

   ```rust
   let stuff = try {
       let x = actionX?;
       let y = actionY(x)?;
       let z = actionZ?;
       sideEffect?;
       finalAction(x, y, z)
   };
   ```

   The Haskell version is syntactic sugar for:

   ```haskell
   stuff =  actionX   >>=
      \x -> actionY x >>=
      \y -> actionZ   >>=
      \z -> sideEffect >>
      finalAction x y z
   ```

   or in Rust:

   ```rust
   let stuff =
       actionX.flat_map(|x| // or .and_then(..)
           actionY(x).flat_map(|y|
               actionZ.flat_map(|z|
                   sideEffect.flat_map(|_|
                       finalAction(x, y, z)
                   )
               )
           )
       );
   ```

   In the Haskell version, `>>=` is defined in the `Monad` typeclass (trait):

   ```haskell
   {-# LANGUAGE KindSignatures #-}

   class Applicative m => Monad (m :: * -> *) where
       return :: a -> m a
       (>>=)  :: m a -> (a -> m b) -> m b

       (>>)   :: m a -> m b -> m b
       (>>) = \ma mb -> ma >>= \_ -> mb
   ```

   And some instances (impls) of `Monad` are:

   ```haskell
   -- | Same as Option<T>
   data Maybe a = Nothing | Just a

   instance Monad Maybe where
       return = Just
       (Just a) >>= f = f a
       _        >>= _ = Nothing

   -- | `struct Norm<T> { value: T, normalized: bool }`
   data Norm a = Norm a Bool

   instance Monad Norm where
       return a = Norm a False
       (Norm a u) >>= f = let Norm b w = f a in  Norm b (u || w)
   ```

[`MonadError`]: http://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Error-Class.html#t:MonadError

Considering the latter case of do-notation,
we saw how `try { .. }` and `do { .. }` relate.
In fact, `try { .. }` is special to the [`Try`] ([`MonadError`]) monads.
There are also more forms of monads which you might want to use `do { .. }` for.
Among these are: Futures, Iterators
Due to having more monads than [`Try`]-based ones,
using the `do { .. }` syntax directly as a replacement for `try { .. }` becomes
problematic as it:

1. confuses everyone familiar with do-notation and monads.
2. is in the way of use for monads in general.
3. `do` is generic and unclear wrt. semantics.

## Alternative: reserving `trap`

1. **Fidelity to the construct's actual behavior:** Good
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 4
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=trap)
    - **Used as crate?** [*No*](https://crates.io/crates/trap).
    - **Usage (sourcegraph):** **4** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+trap\s+=|(fn|impl|mod|struct|enum|union|trait)\s+trap)\b
    ```
7. **Consistency with old learning material:** Untaught

### Review

Arguably, this candidate keyword is a somewhat a good choice.

To `trap` an error is sufficently clear on the "exception boundary" semantics
we wish to communicate.

However, `trap` is used as an error handler in at least one langauge.

It also does not have the familiarity that `try` does have and is entirely
inconsistent wrt. naming in the standard library.

## Alternative: reserving `wrap`

1. **Fidelity to the construct's actual behavior:** Somewhat good
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 4
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=wrap)
    - **Used as crate?** [*Yes*](https://crates.io/crates/wrap), no reverse dependencies.
    - **Usage (sourcegraph):** **37+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+wrap\s+=|(fn|impl|mod|struct|enum|union|trait)\s+wrap)\b
    ```
7. **Consistency with old learning material:** Untaught

### Review

With `wrap { .. }` we can say that it "wraps" the result of the block as a
`Result` / `Option`, etc. and it is logically related to `.unwrap()`,
which is however a partial function, wherefore the connotation might be bad.

Also, `wrap` could be considered too generic as with `do` in that it could
fit for any monad.

## Alternative: reserving `result`

1. **Fidelity to the construct's actual behavior:** Somewhat good
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 6
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very high
    - **Used in std:** [*Yes*](https://doc.rust-lang.org/nightly/std/?search=result) for the `{std, core}::result` modules.
    - **Used as crate?** [*Yes*](https://crates.io/crates/result). 6 reverse dependencies (transitive closure).
    - **Usage (sourcegraph):** **43+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+result\s+=|(fn|impl|mod|struct|enum|union|trait)\s+result)\b
    ```
7. **Consistency with old learning material:** Untaught

## Review

[final encoding]: http://okmij.org/ftp/tagless-final/course/lecture.pdf

The fidelity of `result` is somewhat good due to the association with the
`Result` type as well as `Try` being a [final encoding] of `Result`.

However, when you consider `Option`, the association is less direct,
and thus it does not fit `Option` and other types well.

The breakage of the `result` module is however quite problematic,
making this particular choice of keyword more or less a non-starter.

## Alternative: a smattering of other possible keywords

There are a host of other keywords which have been suggested.

### `fallible`

On an [internals thread](https://internals.rust-lang.org/t/bikeshed-rename-catch-blocks-to-fallible-blocks/7121/), `fallible` was suggested. However, this keyword lacks the verb-form that
is the convention in Rust. Breaking with this convention should only be done
if there are significant reasons to do so, which do not seem to exist in this
case. It is also considerably longer than `try` (+5 character) which matters
for constructions which are oft used.

1. **Fidelity to the construct's actual behavior:**  High
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 8
4. **Consistency with related libstd fn conventions:** Highly inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=fallible)
    - **Used as crate?** [*Yes*](https://crates.io/crates/fallible), some reverse dependencies (all by the same author).
    - **Usage (sourcegraph)** [*None*](https://sourcegraph.com/search?q=repogroup:crates+case:yes++%5Cb%28%28let%7Cconst%7Ctype%7C%29%5Cs%2Bfallible%5Cs%2B%3D%7C%28fn%7Cimpl%7Cmod%7Cstruct%7Cenum%7Cunion%7Ctrait%29%5Cs%2Bfallible%29%5Cb+max:400)
7. **Consistency with old learning material:** Untaught

### Synonyms of `catch`:

Some synonyms of `catch` [have been suggested](https://internals.rust-lang.org/t/bikeshed-rename-catch-blocks-to-fallible-blocks/7121/2):

#### `accept`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 6
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Medium
    - **Used in std:** [*Yes*](https://doc.rust-lang.org/nightly/std/?search=accept)
    - **Used as crate?** [*No*](https://crates.io/crates/accept).
    - **Usage (sourcegraph):** **79+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+accept\s+=|(fn|impl|mod|struct|enum|union|trait)\s+accept)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `capture`

1. **Fidelity to the construct's actual behavior:** Good.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 7
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=capture)
    - **Used as crate?** [*Yes*](https://crates.io/crates/capture), no reverse dependencies.
    - **Usage (sourcegraph):** **6+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+capture\s+=|(fn|impl|mod|struct|enum|union|trait)\s+capture)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `collect`

1. **Fidelity to the construct's actual behavior:** Very much not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 7
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very high
    - **Used in std:** [*Yes*](https://doc.rust-lang.org/nightly/std/?search=collect) (`Iterator::collect`)
    - **Used as crate?** [*Yes*](https://crates.io/crates/collect), no reverse dependencies.
    - **Usage (sourcegraph):** **35+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+collect\s+=|(fn|impl|mod|struct|enum|union|trait)\s+collect)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `recover`

1. **Fidelity to the construct's actual behavior:** Good
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 7
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=recover)
    - **Used as crate?** [*No*](https://crates.io/crates/recover)
    - **Usage (sourcegraph):** **4+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+recover\s+=|(fn|impl|mod|struct|enum|union|trait)\s+recover)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `resolve`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 7
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Low to medium
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=resolve)
    - **Used as crate?** [*Yes*](https://crates.io/crates/resolve), 3 reverse dependencies
    - **Usage (sourcegraph):** **50+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+resolve\s+=|(fn|impl|mod|struct|enum|union|trait)\s+resolve)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `take`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 4
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Huge
    - **Used in std:** [*Yes*](https://doc.rust-lang.org/nightly/std/?search=take), `{Cell, HashSet, Read, Iterator, Option}::take`.
    - **Used as crate?** [*Yes*](https://crates.io/crates/resolve), a lot of reverse dependency (transitive closure).
    - **Usage (sourcegraph):** **62+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+take\s+=|(fn|impl|mod|struct|enum|union|trait)\s+take)\b
    ```
7. **Consistency with old learning material:** Untaught

#### Review

Of these, only `recover` and `capture` seem reasonable semantically.
But `recover` is even more problematic than `catch` because it enhances
the feeling of exception-handling instead of exception-boundaries.
However, `capture` is reasonable as a substitute for `try`,
but it seems obscure and lacks familiarity, which is counted as a strong downside.

### [and some other keywords:](https://internals.rust-lang.org/t/bikeshed-rename-catch-blocks-to-fallible-blocks/7121/13)

#### `coalesce`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 8
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Medium (itertools)
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=coalesce).
    - **Used as crate?** [*Yes*](https://crates.io/crates/coalesce), one reverse dependency.
    - **Usage (sourcegraph):** **3+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+coalesce\s+=|(fn|impl|mod|struct|enum|union|trait)\s+coalesce)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `fuse`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 4
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Medium (libstd)
    - **Used in std:** [*Yes*](https://doc.rust-lang.org/nightly/std/?search=fuse), `Iterator::fuse`.
    - **Used as crate?** [*Yes*](https://crates.io/crates/fuse), 8 reverse dependencies (transitive closure).
    - **Usage (sourcegraph):** **8+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+fuse\s+=|(fn|impl|mod|struct|enum|union|trait)\s+fuse)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `unite`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 5
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=unite).
    - **Used as crate?** [*No*](https://crates.io/crates/unite).
    - **Usage (sourcegraph):** **0+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+unite\s+=|(fn|impl|mod|struct|enum|union|trait)\s+unite)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `cohere`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 6
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=cohere).
    - **Used as crate?** [*No*](https://crates.io/crates/cohere).
    - **Usage (sourcegraph):** **0+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+cohere\s+=|(fn|impl|mod|struct|enum|union|trait)\s+cohere)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `consolidate`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 11
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=consolidate).
    - **Used as crate?** [*No*](https://crates.io/crates/consolidate).
    - **Usage (sourcegraph):** **0+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+consolidate\s+=|(fn|impl|mod|struct|enum|union|trait)\s+consolidate)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `unify`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 5
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=unify).
    - **Used as crate?** [*Yes*](https://crates.io/crates/unify), no dependencies
    - **Usage (sourcegraph):** **1** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+take\s+=|(fn|impl|mod|struct|enum|union|trait)\s+take)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `combine`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 7
4. **Consistency with related libstd fn conventions:** Inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Medium
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=combine).
    - **Used as crate?** [*Yes*](https://crates.io/crates/combine), 17 (direct dependencies)
    - **Usage (sourcegraph):** **6+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+combine\s+=|(fn|impl|mod|struct|enum|union|trait)\s+combine)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `resultof`

1. **Fidelity to the construct's actual behavior:** Somewhat
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 8
4. **Consistency with related libstd fn conventions:** Very inconsistent (not verb)
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:** Very low
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=resultof).
    - **Used as crate?** [*No*](https://crates.io/crates/resultof).
    - **Usage (sourcegraph):** **0+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+resultof\s+=|(fn|impl|mod|struct|enum|union|trait)\s+resultof)\b
    ```
7. **Consistency with old learning material:** Untaught

#### `returned`

1. **Fidelity to the construct's actual behavior:**  Not at all.
2. **Precedent from existing languages:** None
3. **Brevity / Length:** 8
4. **Consistency with related libstd fn conventions:** Very inconsistent
5. **Consistency with the naming of the trait used for `?`:** Inconsistent
6. **Risk of breakage:**
    - **Used in std:** [*No*](https://doc.rust-lang.org/nightly/std/?search=returned).
    - **Used as crate?** [*No*](https://crates.io/crates/returned).
    - **Usage (sourcegraph):** **0+** regex:
    ```
    repogroup:crates case:yes max:400
    \b((let|const|type|)\s+returned\s+=|(fn|impl|mod|struct|enum|union|trait)\s+returned)\b
    ```
7. **Consistency with old learning material:** Untaught

#### Review

Of these, only `resultof` seems to be semantically descriptive and has some support. However, it has three major drawbacks:

+ Length: Compared to `try`, it is 5 characters longer (see reasoning for `fallible`).

+ Not a word: `resultof` is in fact a concatenation of `result` and `of`.
  This does not feel like a natural fit for Rust, as we tend to use a `_` separator.
  Furthermore, there are no current keywords in use that are concatenations of two word.

+ `Result<T, E>` oriented: `resultof` is too tied to `Result<T, E>` and fits poorly with `Option<T>` or other types that implement `Try`.

# Prior art
[prior-art]: #prior-art

All of the languages listed below have a `try { .. } <handler_kw> { .. }` concept
(modulo layout syntax / braces) where `<handler_kw>` is one of:
`catch`, `with`, `except`, `trap`, `rescue`.

In total, these are 29 languages and they have massive ~80% dominance according
to the [TIOBE index](https://www.tiobe.com/tiobe-index/)
and roughly the same with the [PYPL index](http://pypl.github.io/PYPL.html).

+ [C++](http://en.cppreference.com/w/cpp/language/try_catch)
+ [D](https://tour.dlang.org/tour/en/basics/exceptions)
+ [C#](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/try-catch)
+ [Java](https://docs.oracle.com/javase/tutorial/essential/exceptions/try.html)
+ [Scala](https://stackoverflow.com/questions/18685573/try-catch-finally-return-value)
+ [Kotlin](https://kotlinlang.org/docs/reference/exceptions.html)
+ [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/try...catch)
+ [TypeScript](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-2-5.html)
+ [ActionScript](https://help.adobe.com/en_US/ActionScript/3.0_ProgrammingAS3/WS5b3ccc516d4fbf351e63e3d118a9b90204-7ed1.html#WS5b3ccc516d4fbf351e63e3d118a9b90204-7ec5)
+ [Dart](https://www.dartlang.org/resources/dart-tips/dart-tips-ep-9)
+ [Python](https://docs.python.org/3/tutorial/errors.html)
+ [PHP](http://php.net/manual/en/language.exceptions.php)
+ [Matlab](https://se.mathworks.com/help/matlab/ref/try.html)
+ [Visual Basic](https://docs.microsoft.com/en-us/dotnet/visual-basic/language-reference/statements/try-catch-finally-statement)
+ [OCaml](https://ocaml.org/learn/tutorials/error_handling.html)
+ [F#](https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/exception-handling/the-try-with-expression)
+ [Objective C](https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/ProgrammingWithObjectiveC/ErrorHandling/ErrorHandling.html#//apple_ref/doc/uid/TP40011210-CH9-SW3)
+ [Swift](https://developer.apple.com/library/content/documentation/Swift/Conceptual/Swift_Programming_Language/ErrorHandling.html)
+ [Delphi](https://stackoverflow.com/questions/6601147/how-to-correctly-write-try-finally-except-statements)
+ [Julia](https://docs.julialang.org/en/stable/manual/control-flow/#The-try/catch-statement-1)
+ [Elixir](https://elixir-lang.org/getting-started/try-catch-and-rescue.html)
+ [Erlang](http://erlang.org/doc/reference_manual/expressions.html#try)
+ [Clojure](https://clojuredocs.org/clojure.core/try)
+ [R](https://www.rdocumentation.org/packages/base/versions/3.0.3/topics/conditions), modulo minor syntactic difference.
+ [Powershell](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_try_catch_finally?view=powershell-6)
+ [Tcl](http://wiki.tcl.tk/8293)
+ [Apex](https://developer.salesforce.com/page/An_Introduction_to_Exception_Handling)
+ [RPG](http://devnet.asna.com/documentation/Help102/AVR/_HTML/TRYCATCHFINALLY.htm)
+ [ABAP](https://help.sap.com/doc/abapdocu_751_index_htm/7.51/en-US/abaptry.htm)

The syntactic form `catch { .. }` seems quite rare and is,
together with `trap`, `rescue`, `except`, only used for handlers.
However, the `<kw> { .. }` expression we want to introduce is not a handler,
but rather the body of expression we wish to `try`.

There are however a few languages where `catch { .. }` is used for the fallible
part and not for the handler, these languages are:
+ [Erlang](http://erlang.org/doc/reference_manual/expressions.html#catch)
+ [Tcl](https://www.tcl.tk/man/tcl/TclCmd/catch.htm)

However, the combined popularity of these langauges are not significant as
compared to that for `try { .. }`.

# Unresolved questions
[unresolved]: #unresolved-questions

None as of yet.
