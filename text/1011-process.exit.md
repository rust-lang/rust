- Feature Name: exit
- Start Date: 2015-03-24
- RFC PR: https://github.com/rust-lang/rfcs/pull/1011
- Rust Issue: (leave this empty)

# Summary

Add a function to the `std::process` module to exit the process immediately with
a specified exit code.

# Motivation

Currently there is no stable method to exit a program in Rust with a nonzero
exit code without panicking. The current unstable method for doing so is by
using the `exit_status` feature with the `std::env::set_exit_status` function.

This function has not been stabilized as it diverges from the system APIs (there
is no equivalent) and it represents an odd piece of global state for a Rust
program to have. One example of odd behavior that may arise is that if a library
calls `env::set_exit_status`, then the process is not guaranteed to exit with
that status (e.g. Rust was called from C).

The purpose of this RFC is to provide at least one method on the path to
stabilization which will provide a method to exit a process with an arbitrary
exit code.

# Detailed design

The following function will be added to the `std::process` module:

```rust
/// Terminates the current process with the specified exit code.
///
/// This function will never return and will immediately terminate the current
/// process. The exit code is passed through to the underlying OS and will be
/// available for consumption by another process.
///
/// Note that because this function never returns, and that it terminates the
/// process, no destructors on the current stack or any other thread's stack
/// will be run. If a clean shutdown is needed it is recommended to only call
/// this function at a known point where there are no more destructors left
/// to run.
pub fn exit(code: i32) -> !;
```

Implementation-wise this will correspond to the [`exit` function][unix] on unix
and the [`ExitProcess` function][win] on windows.

[unix]: http://pubs.opengroup.org/onlinepubs/000095399/functions/exit.html
[win]: https://msdn.microsoft.com/en-us/library/windows/desktop/ms682658%28v=vs.85%29.aspx

This function is also not marked `unsafe`, despite the risk of leaking
allocated resources (e.g. destructors may not be run). It is already possible
to safely create memory leaks in Rust, however, (with `Rc` + `RefCell`), so
this is not considered a strong enough threshold to mark the function as
`unsafe`.

# Drawbacks

* This API does not solve all use cases of exiting with a nonzero exit status.
  It is sometimes more convenient to simply return a code from the `main`
  function instead of having to call a separate function in the standard
  library.

# Alternatives

* One alternative would be to stabilize `set_exit_status` as-is today. The
  semantics of the function would be clearly documented to prevent against
  surprises, but it would arguably not prevent all surprises from arising. Some
  reasons for not pursuing this route, however, have been outlined in the
  motivation.

* The `main` function of binary programs could be altered to require an
  `i32` return value. This would greatly lessen the need to stabilize this
  function as-is today as it would be possible to exit with a nonzero code by
  returning a nonzero value from `main`. This is a backwards-incompatible
  change, however.

* The `main` function of binary programs could optionally be typed as `fn() ->
  i32` instead of just `fn()`. This would be a backwards-compatible change, but
  does somewhat add complexity. It may strike some as odd to be able to define
  the `main` function with two different signatures in Rust. Additionally, it's
  likely that the `exit` functionality proposed will be desired regardless of
  whether the main function can return a code or not.

# Unresolved questions

* To what degree should the documentation imply that `rt::at_exit` handlers are
  run? Implementation-wise their execution is guaranteed, but we may not wish
  for this to always be so.
