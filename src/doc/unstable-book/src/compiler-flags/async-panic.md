# `async-panic`

Part of project goal [async statemachine optimisation](https://github.com/rust-lang/rust-project-goals/issues/623).

--------------------

When set to `true` (default), async blocks and functions will generate a statemachine that will emit a panic when the statemachine is polled after completion.

When set to `false`, a poll after completion will yield a `Poll::Pending` value. This makes the statemachine simpler and saves on binary size. The downside is that buggy code may poll a future that will never yield `Ready(_)` anymore which can lead to getting stuck in infinite loops or general inefficiencies.
