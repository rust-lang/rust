# Canonicalization

> **NOTE**: FIXME: The content of this chapter has some overlap with
> [Next-gen trait solving Canonicalization chapter](../solve/canonicalization.html).
> It is suggested to reorganize these contents in the future.

Canonicalization is the process of **isolating** an inference value
from its context. It is a key part of implementing
[canonical queries][cq], and you may wish to read the parent chapter
to get more context.

Canonicalization is really based on a very simple concept: every
[inference variable](../type-inference.html#vars) is always in one of
two states: either it is **unbound**, in which case we don't know yet
what type it is, or it is **bound**, in which case we do. So to
isolate some data-structure T that contains types/regions from its
environment, we just walk down and find the unbound variables that
appear in T; those variables get replaced with "canonical variables",
starting from zero and numbered in a fixed order (left to right, for
the most part, but really it doesn't matter as long as it is
consistent).

[cq]: ./canonical-queries.html

So, for example, if we have the type `X = (?T, ?U)`, where `?T` and
`?U` are distinct, unbound inference variables, then the canonical
form of `X` would be `(?0, ?1)`, where `?0` and `?1` represent these
**canonical placeholders**. Note that the type `Y = (?U, ?T)` also
canonicalizes to `(?0, ?1)`. But the type `Z = (?T, ?T)` would
canonicalize to `(?0, ?0)` (as would `(?U, ?U)`). In other words, the
exact identity of the inference variables is not important – unless
they are repeated.

We use this to improve caching as well as to detect cycles and other
things during trait resolution. Roughly speaking, the idea is that if
two trait queries have the same canonical form, then they will get
the same answer. That answer will be expressed in terms of the
canonical variables (`?0`, `?1`), which we can then map back to the
original variables (`?T`, `?U`).

## Canonicalizing the query

To see how it works, imagine that we are asking to solve the following
trait query: `?A: Foo<'static, ?B>`, where `?A` and `?B` are unbound.
This query contains two unbound variables, but it also contains the
lifetime `'static`. The trait system generally ignores all lifetimes
and treats them equally, so when canonicalizing, we will *also*
replace any [free lifetime](../appendix/background.html#free-vs-bound) with a
canonical variable (Note that `'static` is actually a _free_ lifetime
variable here. We are not considering it in the typing context of the whole
program but only in the context of this trait reference. Mathematically, we
are not quantifying over the whole program, but only this obligation).
Therefore, we get the following result:

```text
?0: Foo<'?1, ?2>
```

Sometimes we write this differently, like so:

```text
for<T,L,T> { ?0: Foo<'?1, ?2> }
```

This `for<>` gives some information about each of the canonical
variables within.  In this case, each `T` indicates a type variable,
so `?0` and `?2` are types; the `L` indicates a lifetime variable, so
`?1` is a lifetime. The `canonicalize` method *also* gives back a
`CanonicalVarValues` array OV with the "original values" for each
canonicalized variable:

```text
[?A, 'static, ?B]
```

We'll need this vector OV later, when we process the query response.

## Executing the query

Once we've constructed the canonical query, we can try to solve it.
To do so, we will wind up creating a fresh inference context and
**instantiating** the canonical query in that context. The idea is that
we create a substitution S from the canonical form containing a fresh
inference variable (of suitable kind) for each canonical variable.
So, for our example query:

```text
for<T,L,T> { ?0: Foo<'?1, ?2> }
```

the substitution S might be:

```text
S = [?A, '?B, ?C]
```

We can then replace the bound canonical variables (`?0`, etc) with
these inference variables, yielding the following fully instantiated
query:

```text
?A: Foo<'?B, ?C>
```

Remember that substitution S though! We're going to need it later.

OK, now that we have a fresh inference context and an instantiated
query, we can go ahead and try to solve it. The trait solver itself is
explained in more detail in [another section](../solve/the-solver.md), but
suffice to say that it will compute a [certainty value][cqqr] (`Proven` or
`Ambiguous`) and have side-effects on the inference variables we've
created. For example, if there were only one impl of `Foo`, like so:

[cqqr]: ./canonical-queries.html#query-response

```rust,ignore
impl<'a, X> Foo<'a, X> for Vec<X>
where X: 'a
{ ... }
```

then we might wind up with a certainty value of `Proven`, as well as
creating fresh inference variables `'?D` and `?E` (to represent the
parameters on the impl) and unifying as follows:

- `'?B = '?D`
- `?A = Vec<?E>`
- `?C = ?E`

We would also accumulate the region constraint `?E: '?D`, due to the
where clause.

In order to create our final query result, we have to "lift" these
values out of the query's inference context and into something that
can be reapplied in our original inference context. We do that by
**re-applying canonicalization**, but to the **query result**.

## Canonicalizing the query result

As discussed in [the parent section][cqqr], most trait queries wind up
with a result that brings together a "certainty value" `certainty`, a
result substitution `var_values`, and some region constraints. To
create this, we wind up re-using the substitution S that we created
when first instantiating our query. To refresh your memory, we had a query

```text
for<T,L,T> { ?0: Foo<'?1, ?2> }
```

for which we made a substutition S:

```text
S = [?A, '?B, ?C]
```

We then did some work which unified some of those variables with other things.
If we "refresh" S with the latest results, we get:

```text
S = [Vec<?E>, '?D, ?E]
```

These are precisely the new values for the three input variables from
our original query. Note though that they include some new variables
(like `?E`). We can make those go away by canonicalizing again! We don't
just canonicalize S, though, we canonicalize the whole query response QR:

```text
QR = {
    certainty: Proven,             // or whatever
    var_values: [Vec<?E>, '?D, ?E] // this is S
    region_constraints: [?E: '?D], // from the impl
    value: (),                     // for our purposes, just (), but
                                   // in some cases this might have
                                   // a type or other info
}
```

The result would be as follows:

```text
Canonical(QR) = for<T, L> {
    certainty: Proven,
    var_values: [Vec<?0>, '?1, ?0]
    region_constraints: [?0: '?1],
    value: (),
}
```

(One subtle point: when we canonicalize the query **result**, we do not
use any special treatment for free lifetimes. Note that both
references to `'?D`, for example, were converted into the same
canonical variable (`?1`). This is in contrast to the original query,
where we canonicalized every free lifetime into a fresh canonical
variable.)

Now, this result must be reapplied in each context where needed.

## Processing the canonicalized query result

In the previous section we produced a canonical query result. We now have
to apply that result in our original context. If you recall, way back in the
beginning, we were trying to prove this query:

```text
?A: Foo<'static, ?B>
```

We canonicalized that into this:

```text
for<T,L,T> { ?0: Foo<'?1, ?2> }
```

and now we got back a canonical response:

```text
for<T, L> {
    certainty: Proven,
    var_values: [Vec<?0>, '?1, ?0]
    region_constraints: [?0: '?1],
    value: (),
}
```

We now want to apply that response to our context. Conceptually, how
we do that is to (a) instantiate each of the canonical variables in
the result with a fresh inference variable, (b) unify the values in
the result with the original values, and then (c) record the region
constraints for later. Doing step (a) would yield a result of

```text
{
      certainty: Proven,
      var_values: [Vec<?C>, '?D, ?C]
                       ^^   ^^^ fresh inference variables
      region_constraints: [?C: '?D],
      value: (),
}
```

Step (b) would then unify:

```text
?A with Vec<?C>
'static with '?D
?B with ?C
```

And finally the region constraint of `?C: 'static` would be recorded
for later verification.

(What we *actually* do is a mildly optimized variant of that: Rather
than eagerly instantiating all of the canonical values in the result
with variables, we instead walk the vector of values, looking for
cases where the value is just a canonical variable. In our example,
`values[2]` is `?C`, so that means we can deduce that `?C := ?B` and
`'?D := 'static`. This gives us a partial set of values. Anything for
which we do not find a value, we create an inference variable.)

