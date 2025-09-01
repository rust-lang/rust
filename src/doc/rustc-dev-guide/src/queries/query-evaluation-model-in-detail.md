# The Query Evaluation Model in detail

This chapter provides a deeper dive into the abstract model queries are built on.
It does not go into implementation details but tries to explain
the underlying logic. The examples here, therefore, have been stripped down and
simplified and don't directly reflect the compilers internal APIs.

## What is a query?

Abstractly we view the compiler's knowledge about a given crate as a "database"
and queries are the way of asking the compiler questions about it, i.e.
we "query" the compiler's "database" for facts.

However, there's something special to this compiler database: It starts out empty
and is filled on-demand when queries are executed. Consequently, a query must
know how to compute its result if the database does not contain it yet. For
doing so, it can access other queries and certain input values that the database
is pre-filled with on creation.

A query thus consists of the following things:

 - A name that identifies the query
 - A "key" that specifies what we want to look up
 - A result type that specifies what kind of result it yields
 - A "provider" which is a function that specifies how the result is to be
   computed if it isn't already present in the database.

As an example, the name of the `type_of` query is `type_of`, its query key is a
`DefId` identifying the item we want to know the type of, the result type is
`Ty<'tcx>`, and the provider is a function that, given the query key and access
to the rest of the database, can compute the type of the item identified by the
key.

So in some sense a query is just a function that maps the query key to the
corresponding result. However, we have to apply some restrictions in order for
this to be sound:

 - The key and result must be immutable values.
 - The provider function must be a pure function in the sense that for the same
   key it must always yield the same result.
 - The only parameters a provider function takes are the key and a reference to
   the "query context" (which provides access to the rest of the "database").

The database is built up lazily by invoking queries. The query providers will
invoke other queries, for which the result is either already cached or computed
by calling another query provider. These query provider invocations
conceptually form a directed acyclic graph (DAG) at the leaves of which are
input values that are already known when the query context is created.



## Caching/Memoization

Results of query invocations are "memoized" which means that the query context
will cache the result in an internal table and, when the query is invoked with
the same query key again, will return the result from the cache instead of
running the provider again.

This caching is crucial for making the query engine efficient. Without
memoization the system would still be sound (that is, it would yield the same
results) but the same computations would be done over and over again.

Memoization is one of the main reasons why query providers have to be pure
functions. If calling a provider function could yield different results for
each invocation (because it accesses some global mutable state) then we could
not memoize the result.



## Input data

When the query context is created, it is still empty: No queries have been
executed, no results are cached. But the context already provides access to
"input" data, i.e. pieces of immutable data that were computed before the
context was created and that queries can access to do their computations.

As of <!-- date-check --> January 2021, this input data consists mainly of
the HIR map, upstream crate metadata, and the command-line options the compiler
was invoked with; but in the future inputs will just consist of command-line
options and a list of source files -- the HIR map will itself be provided by a
query which processes these source files.

Without inputs, queries would live in a void without anything to compute their
result from (remember, query providers only have access to other queries and
the context but not any other outside state or information).

For a query provider, input data and results of other queries look exactly the
same: It just tells the context "give me the value of X". Because input data
is immutable, the provider can rely on it being the same across
different query invocations, just as is the case for query results.



## An example execution trace of some queries

How does this DAG of query invocations come into existence? At some point
the compiler driver will create the, as yet empty, query context. It will then,
from outside of the query system, invoke the queries it needs to perform its
task. This looks something like the following:

```rust,ignore
fn compile_crate() {
    let cli_options = ...;
    let hir_map = ...;

    // Create the query context `tcx`
    let tcx = TyCtxt::new(cli_options, hir_map);

    // Do type checking by invoking the type check query
    tcx.type_check_crate();
}
```

The `type_check_crate` query provider would look something like the following:

```rust,ignore
fn type_check_crate_provider(tcx, _key: ()) {
    let list_of_hir_items = tcx.hir_map.list_of_items();

    for item_def_id in list_of_hir_items {
        tcx.type_check_item(item_def_id);
    }
}
```

We see that the `type_check_crate` query accesses input data
(`tcx.hir_map.list_of_items()`) and invokes other queries
(`type_check_item`). The `type_check_item`
invocations will themselves access input data and/or invoke other queries,
so that in the end the DAG of query invocations will be built up backwards
from the node that was initially executed:

```ignore
         (2)                                                 (1)
  list_of_all_hir_items <----------------------------- type_check_crate()
                                                               |
    (5)             (4)                  (3)                   |
  Hir(foo) <--- type_of(foo) <--- type_check_item(foo) <-------+
                                      |                        |
                    +-----------------+                        |
                    |                                          |
    (7)             v  (6)                  (8)                |
  Hir(bar) <--- type_of(bar) <--- type_check_item(bar) <-------+

// (x) denotes invocation order
```

We also see that often a query result can be read from the cache:
`type_of(bar)` was computed for `type_check_item(foo)` so when
`type_check_item(bar)` needs it, it is already in the cache.

Query results stay cached in the query context as long as the context lives.
So if the compiler driver invoked another query later on, the above graph
would still exist and already executed queries would not have to be re-done.



## Cycles

Earlier we stated that query invocations form a DAG. However, it would be easy
to form a cyclic graph by, for example, having a query provider like the
following:

```rust,ignore
fn cyclic_query_provider(tcx, key) -> u32 {
  // Invoke the same query with the same key again
  tcx.cyclic_query(key)
}
```

Since query providers are regular functions, this would behave much as expected:
Evaluation would get stuck in an infinite recursion. A query like this would not
be very useful either. However, sometimes certain kinds of invalid user input
can result in queries being called in a cyclic way. The query engine includes
a check for cyclic invocations of queries with the same input arguments. 
And, because cycles are an irrecoverable error, will abort execution with a 
"cycle error" message that tries to be human readable.

At some point the compiler had a notion of "cycle recovery", that is, one could
"try" to execute a query and if it ended up causing a cycle, proceed in some
other fashion. However, this was later removed because it is not entirely
clear what the theoretical consequences of this are, especially regarding
incremental compilation.


## "Steal" Queries

Some queries have their result wrapped in a `Steal<T>` struct. These queries
behave exactly the same as regular with one exception: Their result is expected
to be "stolen" out of the cache at some point, meaning some other part of the
program is taking ownership of it and the result cannot be accessed anymore.

This stealing mechanism exists purely as a performance optimization because some
result values are too costly to clone (e.g. the MIR of a function). It seems
like result stealing would violate the condition that query results must be
immutable (after all we are moving the result value out of the cache) but it is
OK as long as the mutation is not observable. This is achieved by two things:

- Before a result is stolen, we make sure to eagerly run all queries that
  might ever need to read that result. This has to be done manually by calling
  those queries.
- Whenever a query tries to access a stolen result, we make an ICE
  (Internal Compiler Error) so that such a condition cannot go unnoticed.

This is not an ideal setup because of the manual intervention needed, so it
should be used sparingly and only when it is well known which queries might
access a given result. In practice, however, stealing has not turned out to be
much of a maintenance burden.

To summarize: "Steal queries" break some of the rules in a controlled way.
There are checks in place that make sure that nothing can go silently wrong.
