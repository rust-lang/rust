# Trait resolution (old-style)

<!-- toc -->

This chapter describes the general process of _trait resolution_ and points out
some non-obvious things.

**Note:** This chapter (and its subchapters) describe how the trait
solver **currently** works. However, we are in the process of
designing a new trait solver. If you'd prefer to read about *that*,
see [*this* subchapter](./chalk.html).

## Major concepts

Trait resolution is the process of pairing up an impl with each
reference to a trait. So, for example, if there is a generic function like:

```rust,ignore
fn clone_slice<T:Clone>(x: &[T]) -> Vec<T> { ... }
```

and then a call to that function:

```rust,ignore
let v: Vec<isize> = clone_slice(&[1, 2, 3])
```

it is the job of trait resolution to figure out whether there exists an impl of
(in this case) `isize : Clone`.

Note that in some cases, like generic functions, we may not be able to
find a specific impl, but we can figure out that the caller must
provide an impl. For example, consider the body of `clone_slice`:

```rust,ignore
fn clone_slice<T:Clone>(x: &[T]) -> Vec<T> {
    let mut v = Vec::new();
    for e in &x {
        v.push((*e).clone()); // (*)
    }
}
```

The line marked `(*)` is only legal if `T` (the type of `*e`)
implements the `Clone` trait. Naturally, since we don't know what `T`
is, we can't find the specific impl; but based on the bound `T:Clone`,
we can say that there exists an impl which the caller must provide.

We use the term *obligation* to refer to a trait reference in need of
an impl. Basically, the trait resolution system resolves an obligation
by proving that an appropriate impl does exist.

During type checking, we do not store the results of trait selection.
We simply wish to verify that trait selection will succeed. Then
later, at codegen time, when we have all concrete types available, we
can repeat the trait selection to choose an actual implementation, which
will then be generated in the output binary.

## Overview

Trait resolution consists of three major parts:

- **Selection**: Deciding how to resolve a specific obligation. For
  example, selection might decide that a specific obligation can be
  resolved by employing an impl which matches the `Self` type, or by using a
  parameter bound (e.g. `T: Trait`). In the case of an impl, selecting one
  obligation can create *nested obligations* because of where clauses
  on the impl itself. It may also require evaluating those nested
  obligations to resolve ambiguities.

- **Fulfillment**: The fulfillment code is what tracks that obligations
  are completely fulfilled. Basically it is a worklist of obligations
  to be selected: once selection is successful, the obligation is
  removed from the worklist and any nested obligations are enqueued.
  Fulfillment constrains inference variables.

- **Evaluation**: Checks whether obligations holds without constraining
  any inference variables. Used by selection.

## Selection

Selection is the process of deciding whether an obligation can be
resolved and, if so, how it is to be resolved (via impl, where clause, etc).
The main interface is the `select()` function, which takes an obligation
and returns a `SelectionResult`. There are three possible outcomes:

- `Ok(Some(selection))` – yes, the obligation can be resolved, and
  `selection` indicates how. If the impl was resolved via an impl,
  then `selection` may also indicate nested obligations that are required
  by the impl.

- `Ok(None)` – we are not yet sure whether the obligation can be
  resolved or not. This happens most commonly when the obligation
  contains unbound type variables.

- `Err(err)` – the obligation definitely cannot be resolved due to a
  type error or because there are no impls that could possibly apply.

The basic algorithm for selection is broken into two big phases:
candidate assembly and confirmation.

Note that because of how lifetime inference works, it is not possible to
give back immediate feedback as to whether a unification or subtype
relationship between lifetimes holds or not. Therefore, lifetime
matching is *not* considered during selection. This is reflected in
the fact that subregion assignment is infallible. This may yield
lifetime constraints that will later be found to be in error (in
contrast, the non-lifetime-constraints have already been checked
during selection and can never cause an error, though naturally they
may lead to other errors downstream).

### Candidate assembly

**TODO**: Talk about _why_ we have different candidates, and why it needs to happen in a probe.

Searches for impls/where-clauses/etc that might
possibly be used to satisfy the obligation. Each of those is called
a candidate. To avoid ambiguity, we want to find exactly one
candidate that is definitively applicable. In some cases, we may not
know whether an impl/where-clause applies or not – this occurs when
the obligation contains unbound inference variables.

The subroutines that decide whether a particular impl/where-clause/etc applies
to a particular obligation are collectively referred to as the process of
_matching_. For `impl` candidates <!-- date-check: Oct 2022 -->,
this amounts to unifying the impl header (the `Self` type and the trait arguments)
while ignoring  nested obligations. If matching succeeds then we add it
to a set of candidates. There are other rules when assembling candidates for
built-in traits such as `Copy`, `Sized`, and `CoerceUnsized`.

Once this first pass is done, we can examine the set of candidates. If
it is a singleton set, then we are done: this is the only impl in
scope that could possibly apply. Otherwise, we can **winnow** down the set
of candidates by using where clauses and other conditions. Winnowing uses
`evaluate_candidate` to check whether the nested obligations may apply. 
If this still leaves more than 1 candidate, we use ` fn candidate_should_be_dropped_in_favor_of` 
to prefer some candidates over others. 


If this reduced set yields a single, unambiguous entry, we're good to go,
otherwise the result is considered ambiguous.

#### Winnowing: Resolving ambiguities

But what happens if there are multiple impls where all the types
unify? Consider this example:

```rust,ignore
trait Get {
    fn get(&self) -> Self;
}

impl<T: Copy> Get for T {
    fn get(&self) -> T {
        *self
    }
}

impl<T: Get> Get for Box<T> {
    fn get(&self) -> Box<T> {
        Box::new(<T>::get(self))
    }
}
```

What happens when we invoke `get(&Box::new(1_u16))`, for example? In this
case, the `Self` type is `Box<u16>` – that unifies with both impls,
because the first applies to all types `T`, and the second to all
`Box<T>`. In order for this to be unambiguous, the compiler does a *winnowing*
pass that considers `where` clauses
and attempts to remove candidates. In this case, the first impl only
applies if `Box<u16> : Copy`, which doesn't hold. After winnowing,
then, we are left with just one candidate, so we can proceed.

#### `where` clauses

Besides an impl, the other major way to resolve an obligation is via a
where clause. The selection process is always given a [parameter
environment] which contains a list of where clauses, which are
basically obligations that we can assume are satisfiable. We will iterate
over that list and check whether our current obligation can be found
in that list. If so, it is considered satisfied. More precisely, we
want to check whether there is a where-clause obligation that is for
the same trait (or some subtrait) and which can match against the obligation.

[parameter environment]: ../typing_parameter_envs.html

Consider this simple example:

```rust,ignore
trait A1 {
    fn do_a1(&self);
}
trait A2 : A1 { ... }

trait B {
    fn do_b(&self);
}

fn foo<X:A2+B>(x: X) {
    x.do_a1(); // (*)
    x.do_b();  // (#)
}
```

In the body of `foo`, clearly we can use methods of `A1`, `A2`, or `B`
on variable `x`. The line marked `(*)` will incur an obligation `X: A1`,
while the line marked `(#)` will incur an obligation `X: B`. Meanwhile,
the parameter environment will contain two where-clauses: `X : A2` and `X : B`.
For each obligation, then, we search this list of where-clauses. The
obligation `X: B` trivially matches against the where-clause `X: B`.
To resolve an obligation `X:A1`, we would note that `X:A2` implies that `X:A1`.

### Confirmation

_Confirmation_ unifies the output type parameters of the trait with the
values found in the obligation, possibly yielding a type error.

Suppose we have the following variation of the `Convert` example in the
previous section:

```rust,ignore
trait Convert<Target> {
    fn convert(&self) -> Target;
}

impl Convert<usize> for isize { ... } // isize -> usize
impl Convert<isize> for usize { ... } // usize -> isize

let x: isize = ...;
let y: char = x.convert(); // NOTE: `y: char` now!
```

Confirmation is where an error would be reported because the impl specified
that `Target` would be `usize`, but the obligation reported `char`. Hence the
result of selection would be an error.

Note that the candidate impl is chosen based on the `Self` type, but
confirmation is done based on (in this case) the `Target` type parameter.

### Selection during codegen

As mentioned above, during type checking, we do not store the results of trait
selection. At codegen time, we repeat the trait selection to choose a particular
impl for each method call. This is done using `fn codegen_select_candidate`. 
In this second selection, we do not consider any where-clauses to be in scope 
because we know that each resolution will resolve to a particular impl.

One interesting twist has to do with nested obligations. In general, in codegen,
we only need to figure out which candidate applies, and we do not care about nested obligations,
as these are already assumed to be true. Nonetheless, we *do* currently fulfill all of them.
That is because it can sometimes inform the results of type inference.
That is, we do not have the full substitutions in terms of the type variables
of the impl available to us, so we must run trait selection to figure
everything out.
