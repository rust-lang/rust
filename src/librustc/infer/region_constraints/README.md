# Region constraint collection

## Terminology

Note that we use the terms region and lifetime interchangeably.

## Introduction

As described in the [inference README](../README.md), and unlike
normal type inference, which is similar in spirit to H-M and thus
works progressively, the region type inference works by accumulating
constraints over the course of a function.  Finally, at the end of
processing a function, we process and solve the constraints all at
once.

The constraints are always of one of three possible forms:

- `ConstrainVarSubVar(Ri, Rj)` states that region variable Ri must be
  a subregion of Rj
- `ConstrainRegSubVar(R, Ri)` states that the concrete region R (which
  must not be a variable) must be a subregion of the variable Ri
- `ConstrainVarSubReg(Ri, R)` states the variable Ri shoudl be less
  than the concrete region R. This is kind of deprecated and ought to
  be replaced with a verify (they essentially play the same role).

In addition to constraints, we also gather up a set of "verifys"
(what, you don't think Verify is a noun? Get used to it my
friend!). These represent relations that must hold but which don't
influence inference proper. These take the form of:

- `VerifyRegSubReg(Ri, Rj)` indicates that Ri <= Rj must hold,
  where Rj is not an inference variable (and Ri may or may not contain
  one). This doesn't influence inference because we will already have
  inferred Ri to be as small as possible, so then we just test whether
  that result was less than Rj or not.
- `VerifyGenericBound(R, Vb)` is a more complex expression which tests
  that the region R must satisfy the bound `Vb`. The bounds themselves
  may have structure like "must outlive one of the following regions"
  or "must outlive ALL of the following regions. These bounds arise
  from constraints like `T: 'a` -- if we know that `T: 'b` and `T: 'c`
  (say, from where clauses), then we can conclude that `T: 'a` if `'b:
  'a` *or* `'c: 'a`.

## Building up the constraints

Variables and constraints are created using the following methods:

- `new_region_var()` creates a new, unconstrained region variable;
- `make_subregion(Ri, Rj)` states that Ri is a subregion of Rj
- `lub_regions(Ri, Rj) -> Rk` returns a region Rk which is
  the smallest region that is greater than both Ri and Rj
- `glb_regions(Ri, Rj) -> Rk` returns a region Rk which is
  the greatest region that is smaller than both Ri and Rj

The actual region resolution algorithm is not entirely
obvious, though it is also not overly complex.

## Snapshotting

It is also permitted to try (and rollback) changes to the graph.  This
is done by invoking `start_snapshot()`, which returns a value.  Then
later you can call `rollback_to()` which undoes the work.
Alternatively, you can call `commit()` which ends all snapshots.
Snapshots can be recursive---so you can start a snapshot when another
is in progress, but only the root snapshot can "commit".

## Skolemization

For a discussion on skolemization and higher-ranked subtyping, please
see the module `middle::infer::higher_ranked::doc`.
