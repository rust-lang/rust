# Member constraints

A member constraint `'m member of ['c_1..'c_N]` expresses that the
region `'m` must be *equal* to some **choice regions** `'c_i` (for
some `i`). These constraints cannot be expressed by users, but they
arise from `impl Trait` due to its lifetime capture rules. Consider a
function such as the following:

```rust,ignore
fn make(a: &'a u32, b: &'b u32) -> impl Trait<'a, 'b> { .. }
```

Here, the true return type (often called the "hidden type") is only
permitted to capture the lifetimes `'a` or `'b`. You can kind of see
this more clearly by desugaring that `impl Trait` return type into its
more explicit form:

```rust,ignore
type MakeReturn<'x, 'y> = impl Trait<'x, 'y>;
fn make(a: &'a u32, b: &'b u32) -> MakeReturn<'a, 'b> { .. }
```

Here, the idea is that the hidden type must be some type that could
have been written in place of the `impl Trait<'x, 'y>` -- but clearly
such a type can only reference the regions `'x` or `'y` (or
`'static`!), as those are the only names in scope. This limitation is
then translated into a restriction to only access `'a` or `'b` because
we are returning `MakeReturn<'a, 'b>`, where `'x` and `'y` have been
replaced with `'a` and `'b` respectively.

## Detailed example

To help us explain member constraints in more detail, let's spell out
the `make` example in a bit more detail. First off, let's assume that
you have some dummy trait:

```rust,ignore
trait Trait<'a, 'b> { }
impl<T> Trait<'_, '_> for T { }
```

and this is the `make` function (in desugared form):

```rust,ignore
type MakeReturn<'x, 'y> = impl Trait<'x, 'y>;
fn make(a: &'a u32, b: &'b u32) -> MakeReturn<'a, 'b> {
  (a, b)
}
```

What happens in this case is that the return type will be `(&'0 u32, &'1 u32)`,
where `'0` and `'1` are fresh region variables. We will have the following
region constraints:

```txt
'0 live at {L}
'1 live at {L}
'a: '0
'b: '1
'0 member of ['a, 'b, 'static]
'1 member of ['a, 'b, 'static]
```

Here the "liveness set" `{L}` corresponds to that subset of the body
where `'0` and `'1` are live -- basically the point from where the
return tuple is constructed to where it is returned (in fact, `'0` and
`'1` might have slightly different liveness sets, but that's not very
interesting to the point we are illustrating here).

The `'a: '0` and `'b: '1` constraints arise from subtyping. When we
construct the `(a, b)` value, it will be assigned type `(&'0 u32, &'1
u32)` -- the region variables reflect that the lifetimes of these
references could be made smaller. For this value to be created from
`a` and `b`, however, we do require that:

```txt
(&'a u32, &'b u32) <: (&'0 u32, &'1 u32)
```

which means in turn that `&'a u32 <: &'0 u32` and hence that `'a: '0`
(and similarly that `&'b u32 <: &'1 u32`, `'b: '1`).

Note that if we ignore member constraints, the value of `'0` would be
inferred to some subset of the function body (from the liveness
constraints, which we did not write explicitly). It would never become
`'a`, because there is no need for it too -- we have a constraint that
`'a: '0`, but that just puts a "cap" on how *large* `'0` can grow to
become. Since we compute the *minimal* value that we can, we are happy
to leave `'0` as being just equal to the liveness set. This is where
member constraints come in.

## Choices are always lifetime parameters

At present, the "choice" regions from a member constraint are always lifetime
parameters from the current function. As of <!-- date-check --> October 2021,
this falls out from the placement of impl Trait, though in the future it may not
be the case. We take some advantage of this fact, as it simplifies the current
code. In particular, we don't have to consider a case like `'0 member of ['1,
'static]`, in which the value of both `'0` and `'1` are being inferred and hence
changing. See [rust-lang/rust#61773][#61773] for more information.

[#61773]: https://github.com/rust-lang/rust/issues/61773

## Applying member constraints

Member constraints are a bit more complex than other forms of
constraints. This is because they have a "or" quality to them -- that
is, they describe multiple choices that we must select from. E.g., in
our example constraint `'0 member of ['a, 'b, 'static]`, it might be
that `'0` is equal to `'a`, `'b`, *or* `'static`. How can we pick the
correct one?  What we currently do is to look for a *minimal choice*
-- if we find one, then we will grow `'0` to be equal to that minimal
choice. To find that minimal choice, we take two factors into
consideration: lower and upper bounds.

### Lower bounds

The *lower bounds* are those lifetimes that `'0` *must outlive* --
i.e., that `'0` must be larger than. In fact, when it comes time to
apply member constraints, we've already *computed* the lower bounds of
`'0` because we computed its minimal value (or at least, the lower
bounds considering everything but member constraints).

Let `LB` be the current value of `'0`. We know then that `'0: LB` must
hold, whatever the final value of `'0` is. Therefore, we can rule out
any choice `'choice` where `'choice: LB` does not hold.

Unfortunately, in our example, this is not very helpful. The lower
bound for `'0` will just be the liveness set `{L}`, and we know that
all the lifetime parameters outlive that set. So we are left with the
same set of choices here. (But in other examples, particularly those
with different variance, lower bound constraints may be relevant.)

### Upper bounds

The *upper bounds* are those lifetimes that *must outlive* `'0` --
i.e., that `'0` must be *smaller* than. In our example, this would be
`'a`, because we have the constraint that `'a: '0`. In more complex
examples, the chain may be more indirect.

We can use upper bounds to rule out members in a very similar way to
lower bounds. If UB is some upper bound, then we know that `UB:
'0` must hold, so we can rule out any choice `'choice` where `UB:
'choice` does not hold.

In our example, we would be able to reduce our choice set from `['a,
'b, 'static]` to just `['a]`. This is because `'0` has an upper bound
of `'a`, and neither `'a: 'b` nor `'a: 'static` is known to hold.

(For notes on how we collect upper bounds in the implementation, see
[the section below](#collecting).)

### Minimal choice

After applying lower and upper bounds, we can still sometimes have
multiple possibilities. For example, imagine a variant of our example
using types with the opposite variance. In that case, we would have
the constraint `'0: 'a` instead of `'a: '0`. Hence the current value
of `'0` would be `{L, 'a}`. Using this as a lower bound, we would be
able to narrow down the member choices to `['a, 'static]` because `'b:
'a` is not known to hold (but `'a: 'a` and `'static: 'a` do hold). We
would not have any upper bounds, so that would be our final set of choices.

In that case, we apply the **minimal choice** rule -- basically, if
one of our choices if smaller than the others, we can use that. In
this case, we would opt for `'a` (and not `'static`).

This choice is consistent with the general 'flow' of region
propagation, which always aims to compute a minimal value for the
region being inferred. However, it is somewhat arbitrary.

<a id="collecting"></a>

### Collecting upper bounds in the implementation

In practice, computing upper bounds is a bit inconvenient, because our
data structures are setup for the opposite. What we do is to compute
the **reverse SCC graph** (we do this lazily and cache the result) --
that is, a graph where `'a: 'b` induces an edge `SCC('b) ->
SCC('a)`. Like the normal SCC graph, this is a DAG. We can then do a
depth-first search starting from `SCC('0)` in this graph. This will
take us to all the SCCs that must outlive `'0`.

One wrinkle is that, as we walk the "upper bound" SCCs, their values
will not yet have been fully computed. However, we **have** already
applied their liveness constraints, so we have some information about
their value. In particular, for any regions representing lifetime
parameters, their value will contain themselves (i.e., the initial
value for `'a` includes `'a` and the value for `'b` contains `'b`). So
we can collect all of the lifetime parameters that are reachable,
which is precisely what we are interested in.
