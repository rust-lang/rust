# Canonicalization

Canonicalization is the process of *isolating* a value from its context and is necessary
for global caching of goals which include inference variables.

The idea is that given the goals `u32: Trait<?x>` and `u32: Trait<?y>`, where `?x` and `?y`
are two different currently unconstrained inference variables, we should get the same result
for both goals. We can therefore prove *the canonical query* `exists<T> u32: Trait<T>` once
and reuse the result.

Let's first go over the way canonical queries work and then dive into the specifics of
how canonicalization works.

## A walkthrough of canonical queries

To make this a bit easier, let's use the trait goal `u32: Trait<?x>` as an example with the
assumption that the only relevant impl is `impl<T> Trait<Vec<T>> for u32`.

### Canonicalizing the input

We start by *canonicalizing* the goal, replacing inference variables with existential and
placeholders with universal bound variables. This would result in the *canonical goal*
`exists<T> u32: Trait<T>`.

We remember the original values of all bound variables in the original context. Here this would
map `T` back to `?x`. These original values are used later on when dealing with the query
response.

We now call the canonical query with the canonical goal.

### Instantiating the canonical goal inside of the query

To actually try to prove the canonical goal we start by instantiating the bound variables with
inference variables and placeholders again.

This happens inside of the query in a completely separate `InferCtxt`. Inside of the query we
now have a goal `u32: Trait<?0>`. We also remember which value we've used to instantiate the bound
variables in the canonical goal, which maps `T` to `?0`.

We now compute the goal `u32: Trait<?0>` and figure out that this holds, but we've constrained
`?0` to `Vec<?1>`. We finally convert this result to something useful to the caller.

### Canonicalizing the query response

We have to return to the caller both whether the goal holds, and the inference constraints
from inside of the query.

To return the inference results to the caller we canonicalize the mapping from bound variables
to the instantiated values in the query. This means that the query response is `Certainty::Yes`
and a mapping from `T` to `exists<U> Vec<U>`.

### Instantiating the query response

The caller now has to apply the constraints returned by the query. For this they first
instantiate the bound variables of the canonical response with inference variables and
placeholders again, so the mapping in the response is now from `T` to `Vec<?z>`.

It now equates the original value of `T` (`?x`) with the value for `T` in the
response (`Vec<?z>`), which correctly constrains `?x` to `Vec<?z>`.

## `ExternalConstraints`

Computing a trait goal may not only constrain inference variables, it can also add region
obligations, e.g. given a goal `(): AOutlivesB<'a, 'b>` we would like to return the fact that
`'a: 'b` has to hold.

This is done by not only returning the mapping from bound variables to the instantiated values
from the query but also extracting additional `ExternalConstraints` from the `InferCtxt` context
while building the response.

## How exactly does canonicalization work

TODO: link to code once the PR lands and elaborate

- types and consts: infer to existentially bound var, placeholder to universally bound var,
    considering universes
- generic parameters in the input get treated as placeholders in the root universe
- all regions in the input get all mapped to existentially bound vars and we "uniquify" them.
    `&'a (): Trait<'a>` gets canonicalized to `exists<'0, '1> &'0 (): Trait<'1>`. We do not care
    about their universes and simply put all regions into the highest universe of the input.
- in the output everything in a universe of the caller gets put into the root universe and only
    gets its correct universe when we unify the var values with the orig values of the caller
- we do not uniquify regions in the response and don't canonicalize `'static`