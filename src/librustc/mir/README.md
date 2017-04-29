# MIR definition and pass system

This file contains the definition of the MIR datatypes along with the
various types for the "MIR Pass" system, which lets you easily
register and define new MIR transformations and analyses.

Most of the code that operates on MIR can be found in the
`librustc_mir` crate or other crates. The code found here in
`librustc` is just the datatype definitions, alonging the functions
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
crate's metadata. But for local query, this query will construct the
MIR and then iteratively optimize it by putting it through various
pipeline stages. This section describes those pipeline stages and how
you can extend them.

Here is a diagram showing the various MIR queries involved in producing
the final `optimized_mir()` for a single def-id `D`. The arrows here
indicate how data flows from query to query.

```
mir_build(D)
  -> mir_pass((0,0,D))              ---+ each suite consists of many passes
    -> ...                             |
      -> mir_pass((0,N,D))             |
        -> mir_suite((0,D))         ---+ ---+ there are several suites
          -> ...                            |
            -> mir_suite((M,D))          ---+
              -> mir_optimized(D)
```

The MIR transformation pipeline is organized into **suites**.  When
you ask for `mir_optimized(D)`, it will turn around and request the
result from the final **suite** of MIR passes
(`mir_suite((M,D))`). This will in turn (eventually) trigger the MIR
to be build and then passes through each of the optimization suites.
Each suite internally triggers one query for each of its passes
(`mir_pass(...)`).

The reason for the suites is that they represent points in the MIR
transformation pipeline where other bits of code are interested in
observing. For example, the `MIR_CONST` suite defines the point where
analysis for constant rvalues and expressions can take
place. `MIR_OPTIMIZED` naturally represents the point where we
actually generate machine code. Nobody should ever request the result
of an individual *pass*, at least outside of the transformation
pipeline: this allows us to add passes into the appropriate suite
without having to modify anything else in the compiler.

### Stealing

Each of these intermediate queries yields up a `&'tcx
Steal<Mir<'tcx>>`, allocated using `tcx.alloc_steal_mir()`. This
indicates that the result may be **stolen** by the next pass -- this
is an optimization to avoid cloning the MIR. Attempting to use a
stolen result will cause a panic in the compiler. Therefore, it is
important that you not read directly from these intermediate queries
except as part of the MIR processing pipeline.

Because of this stealing mechanism, some care must also be taken to
ensure that, before the MIR at a particular phase in the processing
pipeline is stolen, anyone who may want to read from it has already
done so. Sometimes this requires **forcing** queries
(`ty::queries::foo::force(...)`) during an optimization pass -- this
will force a query to execute even though you don't directly require
its result. The query can then read the MIR it needs, and -- once it
is complete -- you can steal it.

As an example, consider MIR const qualification. It wants to read the
result produced by the `MIR_CONST` suite. However, that result will be
**stolen** by the first pass in the next suite (that pass performs
const promotion):

```
mir_suite((MIR_CONST,D)) --read-by--> mir_const_qualif(D)
            |
        stolen-by
            |
            v
mir_pass((MIR_VALIDATED,0,D))
```

Therefore, the const promotion pass (the `mir_pass()` in the diagram)
will **force** `mir_const_qualif` before it actually steals, thus
ensuring that the reads have already happened (and the final result is
cached).

### Implementing and registering a pass

To create a new MIR pass, you have to implement one of the MIR pass
traits. There are several traits, and you want to pick the most
specific one that applies to your pass. They are described here in
order of preference. Once you have implemented a trait for your type
`Foo`, you then have to insert `Foo` into one of the suites; this is
done in `librustc_driver/driver.rs` by invoking `push_pass()` with the
appropriate suite.

**The `MirPass` trait.** For the most part, a MIR pass works by taking
as input the MIR for a single function and mutating it imperatively to
perform an optimization. To write such a pass, you can implement the
`MirPass` trait, which has a single callback that takes an `&mut Mir`.

**The `DefIdPass` trait.** When a `MirPass` trait is executed, the
system will automatically steal the result of the previous pass and
supply it to you. (See the section on queries and stealing below.)
Sometimes you don't want to steal the result of the previous pass
right away. In such cases, you can define a `DefIdPass`, which simply
gets a callback and lets you decide when to steal the previous result.

**The `Pass` trait.** The most primitive but flexible trait is `Pass`.
Unlike the other pass types, it returns a `Multi` result, which means
it scan be used for interprocedural passes which mutate more than one
MIR at a time (e.g., `inline`).

### The MIR Context

All of the passes when invoked take a `MirCtxt` object. This contains
various methods to find out (e.g.) the current pass suite and pass
index, the def-id you are operating on, and so forth. You can also
access the MIR for the current def-id using `read_previous_mir()`; the
"previous" refers to the fact that this will be the MIR that was
output by the previous pass. Finally, you can `steal_previous_mir()`
to steal the output of the current pass (in which case you get
ownership of the MIR).
