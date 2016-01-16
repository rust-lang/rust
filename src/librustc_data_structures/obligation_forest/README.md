The `ObligationForest` is a utility data structure used in trait
matching to track the set of outstanding obligations (those not yet
resolved to success or error). It also tracks the "backtrace" of each
pending obligation (why we are trying to figure this out in the first
place).

### External view

`ObligationForest` supports two main public operations (there are a
few others not discussed here):

1. Add a new root obligation (`push_root`).
2. Process the pending obligations (`process_obligations`).

When a new obligation `N` is added, it becomes the root of an
obligation tree. This tree is a singleton to start, so `N` is both the
root and the only leaf. Each time the `process_obligations` method is
called, it will invoke its callback with every pending obligation (so
that will include `N`, the first time). The callback shoud process the
obligation `O` that it is given and return one of three results:

- `Ok(None)` -> ambiguous result. Obligation was neither a success
  nor a failure. It is assumed that further attempts to process the
  obligation will yield the same result unless something in the
  surrounding environment changes.
- `Ok(Some(C))` - the obligation was *shallowly successful*. The
  vector `C` is a list of subobligations. The meaning of this is that
  `O` was successful on the assumption that all the obligations in `C`
  are also successful. Therefore, `O` is only considered a "true"
  success if `C` is empty. Otherwise, `O` is put into a suspended
  state and the obligations in `C` become the new pending
  obligations. They will be processed the next time you call
  `process_obligations`.
- `Err(E)` -> obligation failed with error `E`. We will collect this
  error and return it from `process_obligations`, along with the
  "backtrace" of obligations (that is, the list of obligations up to
  and including the root of the failed obligation). No further
  obligations from that same tree will be processed, since the tree is
  now considered to be in error.

When the call to `process_obligations` completes, you get back an `Outcome`,
which includes three bits of information:

- `completed`: a list of obligations where processing was fully
  completed without error (meaning that all transitive subobligations
  have also been completed). So, for example, if the callback from
  `process_obligations` returns `Ok(Some(C))` for some obligation `O`,
  then `O` will be considered completed right away if `C` is the
  empty vector. Otherwise it will only be considered completed once
  all the obligations in `C` have been found completed.
- `errors`: a list of errors that occurred and associated backtraces
  at the time of error, which can be used to give context to the user.
- `stalled`: if true, then none of the existing obligations were
  *shallowly successful* (that is, no callback returned `Ok(Some(_))`).
  This implies that all obligations were either errors or returned an
  ambiguous result, which means that any further calls to
  `process_obligations` would simply yield back further ambiguous
  results. This is used by the `FulfillmentContext` to decide when it
  has reached a steady state.
  
#### Snapshots

The `ObligationForest` supports a limited form of snapshots; see
`start_snapshot`; `commit_snapshot`; and `rollback_snapshot`. In
particular, you can use a snapshot to roll back new root
obligations. However, it is an error to attempt to
`process_obligations` during a snapshot.

### Implementation details

For the most part, comments specific to the implementation are in the
code.  This file only contains a very high-level overview. Basically,
the forest is stored in a vector. Each element of the vector is a node
in some tree. Each node in the vector has the index of an (optional)
parent and (for convenience) its root (which may be itself). It also
has a current state, described by `NodeState`. After each
processing step, we compress the vector to remove completed and error
nodes, which aren't needed anymore.

  
