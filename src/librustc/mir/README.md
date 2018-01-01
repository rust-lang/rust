# MIR definition and pass system

This file contains the definition of the MIR datatypes along with the
various types for the "MIR Pass" system, which lets you easily
register and define new MIR transformations and analyses.

Most of the code that operates on MIR can be found in the
`librustc_mir` crate or other crates. The code found here in
`librustc` is just the datatype definitions, along with the functions
which operate on MIR to be placed everywhere else.

## MIR Data Types and visitor

The main MIR data type is `rustc::mir::Mir`, defined in `mod.rs`.
There is also the MIR visitor (in `visit.rs`) which allows you to walk
the MIR and override what actions will be taken at various points (you
can visit in either shared or mutable mode; the latter allows changing
the MIR in place). Finally `traverse.rs` contains various traversal
routines for visiting the MIR CFG in [different standard orders][traversal]
(e.g. pre-order, reverse post-order, and so forth).

[traversal]: https://en.wikipedia.org/wiki/Tree_traversal

## MIR pass suites and their integration into the query system

As a MIR *consumer*, you are expected to use one of the queries that
returns a "final MIR". As of the time of this writing, there is only
one: `optimized_mir(def_id)`, but more are expected to come in the
future. For foreign def-ids, we simply read the MIR from the other
crate's metadata. But for local def-ids, the query will construct the
MIR and then iteratively optimize it by putting it through various
pipeline stages. This section describes those pipeline stages and how
you can extend them.

To produce the `optimized_mir(D)` for a given def-id `D`, the MIR
passes through several suites of optimizations, each represented by a
query. Each suite consists of multiple optimizations and
transformations. These suites represent useful intermediate points
where we want to access the MIR for type checking or other purposes:

- `mir_build(D)` -- not a query, but this constructs the initial MIR
- `mir_const(D)` -- applies some simple transformations to make MIR ready for constant evaluation;
- `mir_validated(D)` -- applies some more transformations, making MIR ready for borrow checking;
- `optimized_mir(D)` -- the final state, after all optimizations have been performed.

### Stealing

The intermediate queries `mir_const()` and `mir_validated()` yield up
a `&'tcx Steal<Mir<'tcx>>`, allocated using
`tcx.alloc_steal_mir()`. This indicates that the result may be
**stolen** by the next suite of optimizations -- this is an
optimization to avoid cloning the MIR. Attempting to use a stolen
result will cause a panic in the compiler. Therefore, it is important
that you do not read directly from these intermediate queries except as
part of the MIR processing pipeline.

Because of this stealing mechanism, some care must also be taken to
ensure that, before the MIR at a particular phase in the processing
pipeline is stolen, anyone who may want to read from it has already
done so. Concretely, this means that if you have some query `foo(D)`
that wants to access the result of `mir_const(D)` or
`mir_validated(D)`, you need to have the successor pass "force"
`foo(D)` using `ty::queries::foo::force(...)`. This will force a query
to execute even though you don't directly require its result.

As an example, consider MIR const qualification. It wants to read the
result produced by the `mir_const()` suite. However, that result will
be **stolen** by the `mir_validated()` suite. If nothing was done,
then `mir_const_qualif(D)` would succeed if it came before
`mir_validated(D)`, but fail otherwise. Therefore, `mir_validated(D)`
will **force** `mir_const_qualif` before it actually steals, thus
ensuring that the reads have already happened:

```
mir_const(D) --read-by--> mir_const_qualif(D)
     |                       ^
  stolen-by                  |
     |                    (forces)
     v                       |
mir_validated(D) ------------+
```

### Implementing and registering a pass

To create a new MIR pass, you simply implement the `MirPass` trait for
some fresh singleton type `Foo`. Once you have implemented a trait for
your type `Foo`, you then have to insert `Foo` into one of the suites;
this is done in `librustc_driver/driver.rs` by invoking `push_pass(S,
Foo)` with the appropriate suite substituted for `S`.

