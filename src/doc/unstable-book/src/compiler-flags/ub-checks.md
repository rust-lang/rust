# `ub-checks`

The tracking issue for this feature is: [#123499](https://github.com/rust-lang/rust/issues/123499).

--------------------

The `-Zub-checks` compiler flag enables additional runtime checks that detect some causes of Undefined Behavior at runtime.
By default, `-Zub-checks` flag inherits the value of `-Cdebug-assertions`.

All checks are generated on a best-effort basis; even if we have a check implemented for some cause of Undefined Behavior, it may be possible for the check to not fire.
If a dependency is compiled with `-Zub-checks=no` but the final binary or library is compiled with `-Zub-checks=yes`, UB checks reached by the dependency are likely to be optimized out.

When `-Zub-checks` detects UB, a non-unwinding panic is produced.
That means that we will not unwind the stack and will not call any `Drop` impls, but we will execute the configured panic hook.
We expect that unsafe code has been written which relies on code not unwinding which may have UB checks inserted.
Ergo, an unwinding panic could easily turn works-as-intended UB into a much bigger problem.
Calling the panic hook theoretically has the same implications, but we expect that the standard library panic hook will be stateless enough to be always called, and that if a user has configured a panic hook that the hook may be very helpful to debugging the detected UB.
