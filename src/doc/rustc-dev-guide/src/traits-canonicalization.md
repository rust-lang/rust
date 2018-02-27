# Canonicalization

Canonicalization is the process of **isolating** an inference value
from its context. It is really based on a very simple concept: every
[inference variable](./type-inference.html#vars) is always in one of two
states: either it is **unbound**, in which case we don't know yet what
type it is, or it is **bound**, in which case we do. So to isolate
some thing T from its environment, we just walk down and find the
unbound variables that appear in T; those variables get renumbered in
a canonical order (left to right, for the most part, but really it
doesn't matter as long as it is consistent).

So, for example, if we have the type `X = (?T, ?U)`, where `?T` and
`?U` are distinct, unbound inference variables, then the canonical
form of `X` would be `(?0, ?1)`, where `?0` and `?1` represent these
**canonical placeholders**. Note that the type `Y = (?U, ?T)` also
canonicalizes to `(?0, ?1)`. But the type `Z = (?T, ?T)` would
canonicalize to `(?0, ?0)` (as would `(?U, ?U)`). In other words, the
exact identity of the inference variables is not important -- unless
they are repeated.

We use this to improve caching as well as to detect cycles and other
things during trait resolution. Roughly speaking, the idea is that if
two trait queries have the same canonicalize form, then they will get
the same answer -- modulo the precise identities of the variables
involved.

To see how it works, imagine that we are asking to solve the following
trait query: `?A: Foo<'static, ?B>`, where `?A` and `?B` are unbound.
This query contains two unbound variables, but it also contains the
lifetime `'static`. The trait system generally ignores all lifetimes
and treats them equally, so when canonicalizing, we will *also*
replace any [free lifetime](./background.html#free-vs-bound) with a
canonical variable. Therefore, we get the following result: 

    ?0: Foo<'?1, ?2>
    
Sometimes we write this differently, like so:    

    for<T,L,T> { ?0: Foo<'?1, ?2> }
    
This `for<>` gives some information about each of the canonical
variables within.  In this case, I am saying that `?0` is a type
(`T`), `?1` is a lifetime (`L`), and `?2` is also a type (`T`). The
`canonicalize` method *also* gives back a `CanonicalVarValues` array
with the "original values" for each canonicalized variable:

    [?A, 'static, ?B]

Now we do the query and get back some result R. As part of that
result, we'll have an array of values for the canonical inputs. For
example, the canonical result might be:

```
for<2> {
    values = [ Vec<?0>, '1, ?0 ]
                   ^^   ^^  ^^ these are variables in the result!
    ...
}
```

Note that this result is itself canonical and may include some
variables (in this case, `?0`).

What we want to do conceptually is to (a) instantiate each of the
canonical variables in the result with a fresh inference variable
and then (b) unify the values in the result with the original values.
Doing step (a) would yield a result of

```
{
    values = [ Vec<?C>, '?X, ?C ]
                   ^^   ^^^ fresh inference variables in `self`
    ..
}
```

Step (b) would then unify:

```
?A with Vec<?C>
'static with '?X
?B with ?C
```

(What we actually do is a mildly optimized variant of that: Rather
than eagerly instantiating all of the canonical values in the result
with variables, we instead walk the vector of values, looking for
cases where the value is just a canonical variable. In our example,
`values[2]` is `?C`, so that we means we can deduce that `?C := ?B and
`'?X := 'static`. This gives us a partial set of values. Anything for
which we do not find a value, we create an inference variable.)

