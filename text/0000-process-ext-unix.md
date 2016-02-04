- Feature Name: `process_exec`
- Start Date: 2015-11-09
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add two methods to the `std::os::unix::process::CommandExt` trait to provide
more control over how processes are spawned on Unix, specifically:

```rust
fn exec(&mut self) -> io::Error;
fn before_exec<F>(&mut self, f: F) -> &mut Self
    where F: FnOnce() -> io::Result<()> + Send + Sync + 'static;
```

# Motivation
[motivation]: #motivation

Although the standard library's implementation of spawning processes on Unix is
relatively complex, it unfortunately doesn't provide the same flexibility as
calling `fork` and `exec` manually. For example, these sorts of use cases are
not possible with the `Command` API:

* The `exec` function cannot be called without `fork`. It's often useful on Unix
  in doing this to avoid spawning processes or improve debuggability if the
  pre-`exec` code was some form of shim.
* Execute other flavorful functions between the fork/exec if necessary. For
  example some proposed extensions to the standard library are [dealing with the
  controlling tty][tty] or dealing with [session leaders][session]. In theory
  any sort of arbitrary code can be run between these two syscalls, and it may
  not always be the case the standard library can provide a suitable
  abstraction.

[tty]: https://github.com/rust-lang/rust/pull/28982
[session]: https://github.com/rust-lang/rust/pull/26470

Note that neither of these pieces of functionality are possible on Windows as
there is no equivalent of the `fork` or `exec` syscalls in the standard APIs, so
these are specifically proposed as methods on the Unix extension trait.

# Detailed design
[design]: #detailed-design

The following two methods will be added to the
`std::os::unix::process::CommandExt` trait:

```rust
/// Performs all the required setup by this `Command`, followed by calling the
/// `execvp` syscall.
///
/// On success this function will not return, and otherwise it will return an
/// error indicating why the exec (or another part of the setup of the
/// `Command`) failed.
///
/// Note that the process may be in a "broken state" if this function returns in
/// error. For example the working directory, environment variables, signal
/// handling settings, various user/group information, or aspects of stdio
/// file descriptors may have changed. If a "transactional spawn" is required to
/// gracefully handle errors it is recommended to use the cross-platform `spawn`
/// instead.
fn exec(&mut self) -> io::Error;

/// Schedules a closure to be run just before the `exec` function is invoked.
///
/// This closure will be run in the context of the child process after the
/// `fork` and other aspects such as the stdio file descriptors and working
/// directory have successfully been changed. Note that this is often a very
/// constrained environment where normal operations like `malloc` or acquiring a
/// mutex are not guaranteed to work (due to other threads perhaps still running
/// when the `fork` was run).
///
/// The closure is allowed to return an I/O error whose OS error code will be
/// communicated back to the parent and returned as an error from when the spawn
/// was requested.
///
/// Multiple closures can be registered and they will be called in order of
/// their registration. If a closure returns `Err` then no further closures will
/// be called and the spawn operation will immediately return with a failure.
fn before_exec<F>(&mut self, f: F) -> &mut Self
    where F: FnOnce() -> io::Result<()> + Send + Sync + 'static;
```

The `exec` function is relatively straightforward as basically the entire spawn
operation minus the `fork`. The stdio handles will be inherited by default if
not otherwise configured. Note that a configuration of `piped` will likely just
end up with a broken half of a pipe on one of the file descriptors.

The `before_exec` function has extra-restrictive bounds to preserve the same
qualities that the `Command` type has (notably `Send`, `Sync`, and `'static`).
This also happens after all other configuration has happened to ensure that
libraries can take advantage of the other operations on `Command` without having
to reimplement them manually in some circumstances.

# Drawbacks
[drawbacks]: #drawbacks

This change is possible to be a breaking change to `Command` as it will no
longer implement all marker traits by default (due to it containing closure
trait objects). While the common marker traits are handled here, it's possible
that there are some traits in the wild in use which this could break.

Much of the functionality which may initially get funneled through `before_exec`
may actually be best implemented as functions in the standard library itself.
It's likely that many operations are well known across unixes and aren't niche
enough to stay outside the standard library.

# Alternatives
[alternatives]: #alternatives

Instead of souping up `Command` the type could instead provide accessors to all
of the configuration that it contains. This would enable this sort of
functionality to be built on crates.io first instead of requiring it to be built
into the standard library to start out with. Note that this may want to end up
in the standard library regardless, however.

# Unresolved questions
[unresolved]: #unresolved-questions

* Is it appropriate to run callbacks just before the `exec`? Should they instead
  be run before any standard configuration like stdio has run?
* Is it possible to provide "transactional semantics" to the `exec` function
  such that it is safe to recover from? Perhaps it's worthwhile to provide
  partial transactional semantics in the form of "this can be recovered from so
  long as all stdio is inherited".
