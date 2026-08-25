# Lowering to logic

The key observation here is that the Rust trait system is basically a
kind of logic, and it can be mapped onto standard logical inference
rules. We can then look for solutions to those inference rules in a
very similar fashion to how e.g. a [Prolog] solver works. It turns out
that we can't *quite* use Prolog rules (also called Horn clauses) but
rather need a somewhat more expressive variant.

[Prolog]: https://en.wikipedia.org/wiki/Prolog

## Rust traits and logic

One of the first observations is that the Rust trait system is
basically a kind of logic. As such, we can map our struct, trait, and
impl declarations into logical inference rules. For the most part,
these are basically Horn clauses, though we'll see that to capture the
full richness of Rust – and in particular to support generic
programming – we have to go a bit further than standard Horn clauses.

To see how this mapping works, let's start with an example. Imagine
we declare a trait and a few impls, like so:

```rust
trait Clone { }
impl Clone for usize { }
impl<T> Clone for Vec<T> where T: Clone { }
```

We could map these declarations to some Horn clauses, written in a
Prolog-like notation, as follows:

```text
Clone(usize).
Clone(Vec<?T>) :- Clone(?T).

// The notation `A :- B` means "A is true if B is true".
// Or, put another way, B implies A.
```

In Prolog terms, we might say that `Clone(Foo)` – where `Foo` is some
Rust type – is a *predicate* that represents the idea that the type
`Foo` implements `Clone`. These rules are **program clauses**; they
state the conditions under which that predicate can be proven (i.e.,
considered true). So the first rule just says "Clone is implemented
for `usize`". The next rule says "for any type `?T`, Clone is
implemented for `Vec<?T>` if clone is implemented for `?T`". So
e.g. if we wanted to prove that `Clone(Vec<Vec<usize>>)`, we would do
so by applying the rules recursively:

- `Clone(Vec<Vec<usize>>)` is provable if:
  - `Clone(Vec<usize>)` is provable if:
    - `Clone(usize)` is provable. (Which it is, so we're all good.)

But now suppose we tried to prove that `Clone(Vec<Bar>)`. This would
fail (after all, I didn't give an impl of `Clone` for `Bar`):

- `Clone(Vec<Bar>)` is provable if:
  - `Clone(Bar)` is provable. (But it is not, as there are no applicable rules.)

We can easily extend the example above to cover generic traits with
more than one input type. So imagine the `Eq<T>` trait, which declares
that `Self` is equatable with a value of type `T`:

```rust,ignore
trait Eq<T> { ... }
impl Eq<usize> for usize { }
impl<T: Eq<U>> Eq<Vec<U>> for Vec<T> { }
```

That could be mapped as follows:

```text
Eq(usize, usize).
Eq(Vec<?T>, Vec<?U>) :- Eq(?T, ?U).
```

So far so good.

## Type-checking normal functions

OK, now that we have defined some logical rules that are able to
express when traits are implemented and to handle associated types,
let's turn our focus a bit towards **type-checking**. Type-checking is
interesting because it is what gives us the goals that we need to
prove. That is, everything we've seen so far has been about how we
derive the rules by which we can prove goals from the traits and impls
in the program; but we are also interested in how to derive the goals
that we need to prove, and those come from type-checking.

Consider type-checking the function `foo()` here:

```rust,ignore
fn foo() { bar::<usize>() }
fn bar<U: Eq<U>>() { }
```

This function is very simple, of course: all it does is to call
`bar::<usize>()`. Now, looking at the definition of `bar()`, we can see
that it has one where-clause `U: Eq<U>`. So, that means that `foo()` will
have to prove that `usize: Eq<usize>` in order to show that it can call `bar()`
with `usize` as the type argument.

If we wanted, we could write a Prolog predicate that defines the
conditions under which `bar()` can be called. We'll say that those
conditions are called being "well-formed":

```text
barWellFormed(?U) :- Eq(?U, ?U).
```

Then we can say that `foo()` type-checks if the reference to
`bar::<usize>` (that is, `bar()` applied to the type `usize`) is
well-formed:

```text
fooTypeChecks :- barWellFormed(usize).
```

If we try to prove the goal `fooTypeChecks`, it will succeed:

- `fooTypeChecks` is provable if:
  - `barWellFormed(usize)`, which is provable if:
    - `Eq(usize, usize)`, which is provable because of an impl.

Ok, so far so good. Let's move on to type-checking a more complex function.

## Type-checking generic functions: beyond Horn clauses

In the last section, we used standard Prolog horn-clauses (augmented with Rust's
notion of type equality) to type-check some simple Rust functions. But that only
works when we are type-checking non-generic functions. If we want to type-check
a generic function, it turns out we need a stronger notion of goal than what Prolog
can provide. To see what I'm talking about, let's revamp our previous
example to make `foo` generic:

```rust,ignore
fn foo<T: Eq<T>>() { bar::<T>() }
fn bar<U: Eq<U>>() { }
```

To type-check the body of `foo`, we need to be able to hold the type
`T` "abstract".  That is, we need to check that the body of `foo` is
type-safe *for all types `T`*, not just for some specific type. We might express
this like so:

```text
fooTypeChecks :-
  // for all types T...
  forall<T> {
    // ...if we assume that Eq(T, T) is provable...
    if (Eq(T, T)) {
      // ...then we can prove that `barWellFormed(T)` holds.
      barWellFormed(T)
    }
  }.
```

This notation I'm using here is the notation I've been using in my
prototype implementation; it's similar to standard mathematical
notation but a bit Rustified. Anyway, the problem is that standard
Horn clauses don't allow universal quantification (`forall`) or
implication (`if`) in goals (though many Prolog engines do support
them, as an extension). For this reason, we need to accept something
called "first-order hereditary harrop" (FOHH) clauses – this long
name basically means "standard Horn clauses with `forall` and `if` in
the body". But it's nice to know the proper name, because there is a
lot of work describing how to efficiently handle FOHH clauses; see for
example Gopalan Nadathur's excellent
["A Proof Procedure for the Logic of Hereditary Harrop Formulas"][pphhf]
in [the bibliography of Chalk Book][bibliography].

[bibliography]: https://rust-lang.github.io/chalk/book/bibliography.html
[pphhf]: https://rust-lang.github.io/chalk/book/bibliography.html#pphhf

It turns out that supporting FOHH is not really all that hard. And
once we are able to do that, we can easily describe the type-checking
rule for generic functions like `foo` in our logic.

## Source

This page is a lightly adapted version of a
[blog post by Nicholas Matsakis][lrtl].

[lrtl]: http://smallcultfollowing.com/babysteps/blog/2017/01/26/lowering-rust-traits-to-logic/
