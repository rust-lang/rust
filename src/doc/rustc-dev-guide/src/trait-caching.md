# Caching and subtle considerations therewith

In general we attempt to cache the results of trait selection.  This
is a somewhat complex process. Part of the reason for this is that we
want to be able to cache results even when all the types in the trait
reference are not fully known. In that case, it may happen that the
trait selection process is also influencing type variables, so we have
to be able to not only cache the *result* of the selection process,
but *replay* its effects on the type variables.

## An example

The high-level idea of how the cache works is that we first replace
all unbound inference variables with skolemized versions. Therefore,
if we had a trait reference `usize : Foo<$1>`, where `$n` is an unbound
inference variable, we might replace it with `usize : Foo<%0>`, where
`%n` is a skolemized type. We would then look this up in the cache.
If we found a hit, the hit would tell us the immediate next step to
take in the selection process: i.e. apply impl #22, or apply where
clause `X : Foo<Y>`. Let's say in this case there is no hit.
Therefore, we search through impls and where clauses and so forth, and
we come to the conclusion that the only possible impl is this one,
with def-id 22:

```rust
impl Foo<isize> for usize { ... } // Impl #22
```

We would then record in the cache `usize : Foo<%0> ==>
ImplCandidate(22)`. Next we would confirm `ImplCandidate(22)`, which
would (as a side-effect) unify `$1` with `isize`.

Now, at some later time, we might come along and see a `usize :
Foo<$3>`.  When skolemized, this would yield `usize : Foo<%0>`, just as
before, and hence the cache lookup would succeed, yielding
`ImplCandidate(22)`. We would confirm `ImplCandidate(22)` which would
(as a side-effect) unify `$3` with `isize`.

## Where clauses and the local vs global cache

One subtle interaction is that the results of trait lookup will vary
depending on what where clauses are in scope. Therefore, we actually
have *two* caches, a local and a global cache. The local cache is
attached to the `ParamEnv` and the global cache attached to the
`tcx`. We use the local cache whenever the result might depend on the
where clauses that are in scope. The determination of which cache to
use is done by the method `pick_candidate_cache` in `select.rs`. At
the moment, we use a very simple, conservative rule: if there are any
where-clauses in scope, then we use the local cache.  We used to try
and draw finer-grained distinctions, but that led to a serious of
annoying and weird bugs like #22019 and #18290. This simple rule seems
to be pretty clearly safe and also still retains a very high hit rate
(~95% when compiling rustc).

