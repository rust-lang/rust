- Feature Name: panic_handler
- Start Date: 2015-10-08
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

When a thread panics in Rust, the unwinding runtime currently prints a message
to standard error containing the panic argument as well as the filename and
line number corresponding to the location from which the panic originated.
This RFC proposes a mechanism to allow user code to replace this logic with
custom handlers that will run before unwinding begins.

# Motivation

The default behavior is not always ideal for all programs:

* Programs with command line interfaces do not want their output polluted by
  random panic messages.
* Programs using a logging framework may want panic messages to be routed into
  that system so that they can be processed like other events.
* Programs with graphical user interfaces may not have standard error attached
  at all and want to be notified of thread panics to potentially display an
  internal error dialog to the user.

The standard library [previously
supported](https://doc.rust-lang.org/1.3.0/std/rt/unwind/fn.register.html) (in
unstable code) the registration of a set of panic handlers. This API had
several issues:

* The system supported a fixed but unspecified number of handlers, and a
  handler could never be unregistered once added.
* The callbacks were raw function pointers rather than closures.
* Handlers would be invoked on nested panics, which would result in a stack
  overflow if a handler itself panicked.
* The callbacks were specified to take the panic message, file name and line
  number directly. This would prevent us from adding more functionality in
  the future, such as access to backtrace information. In addition, the
  presence of file names and line numbers for all panics causes some amount of
  binary bloat and we may want to add some avenue to allow for the omission of
  those values in the future.

# Detailed design

A new module, `std::panic`, will be created with a panic handling API:

```rust
/// Unregisters the current panic handler, returning it.
///
/// If no custom handler is registered, the default handler will be returned.
///
/// # Panics
///
/// Panics if called from a panicking thread. Note that this will be a nested
/// panic and therefore abort the process.
pub fn take_handler() -> Box<Fn(&PanicInfo) + 'static + Sync + Send> { ... }

/// Registers a custom panic handler, replacing any that was previously
/// registered.
///
/// # Panics
///
/// Panics if called from a panicking thread. Note that this will be a nested
/// panic and therefore abort the process.
pub fn set_handler<F>(handler: F) where F: Fn(&PanicInfo) + 'static + Sync + Send { ... }

/// A struct providing information about a panic.
pub struct PanicInfo { ... }

impl PanicInfo {
    /// Returns the payload associated with the panic.
    ///
    /// This will commonly, but not always, be a `&'static str` or `String`.
    pub fn payload(&self) -> &Any + Send { ... }

    /// Returns information about the location from which the panic originated,
    /// if available.
    pub fn location(&self) -> Option<Location> { ... }
}

/// A struct containing information about the location of a panic.
pub struct Location<'a> { ... }

impl<'a> Location<'a> {
    /// Returns the name of the source file from which the panic originated.
    pub fn file(&self) -> &str { ... }

    /// Returns the line number from which the panic originated.
    pub fn line(&self) -> u32 { ... }
}
```

When a panic occurs, but before unwinding begins, the runtime will call the
registered panic handler. After the handler returns, the runtime will then
unwind the thread. If a thread panics while panicking (a "double panic"), the
panic handler will *not* be invoked and the process will abort. Note that the
thread is considered to be panicking while the panic handler is running, so a
panic originating from the panic handler will result in a double panic.

The `take_handler` method exists to allow for handlers to "chain" by closing
over the previous handler and calling into it:

```rust
let old_handler = panic::take_handler();
panic::set_handler(move |info| {
    println!("uh oh!");
    old_handler(info);
});
```

This is obviously a racy operation, but as a single global resource, the global
panic handler should only be adjusted by applications rather than libraries,
most likely early in the startup process.

The implementation of `set_handler` and `take_handler` will have to be
carefully synchronized to ensure that a handler is not replaced while executing
in another thread. This can be accomplished in a manner similar to [that used
by the `log`
crate](https://github.com/rust-lang-nursery/log/blob/aa8618c840dd88b27c487c9fc9571d89751583f3/src/lib.rs).
`take_handler` and `set_handler` will wait until no other threads are currently
running the panic handler, at which point they will atomically swap the handler
out as appropriate.

Note that `location` will always return `Some` in the current implementation.
It returns an `Option` to hedge against possible future changes to the panic
system that would allow a crate to be compiled with location metadata removed
to minimize binary size.

## Prior Art

C++ has a
[`std::set_terminate`](http://www.cplusplus.com/reference/exception/set_terminate/)
function which registers a handler for uncaught exceptions, returning the old
one. The handler takes no arguments.

Python passes uncaught exceptions to the global handler
[`sys.excepthook`](https://docs.python.org/2/library/sys.html#sys.excepthook)
which can be set by user code.

In Java, uncaught exceptions [can be
handled](http://docs.oracle.com/javase/7/docs/api/java/lang/Thread.html#setUncaughtExceptionHandler(java.lang.Thread.UncaughtExceptionHandler))
by handlers registered on an individual `Thread`, by the `Thread`'s,
`ThreadGroup`, and by a handler registered globally. The handlers are provided
with the `Throwable` that triggered the handler.

# Drawbacks

The more infrastructure we add to interact with panics, the more attractive it
becomes to use them as a more normal part of control flow.

# Alternatives

Panic handlers could be run after a panicking thread has unwound rather than
before. This is perhaps a more intuitive arrangement, and allows `catch_panic`
to prevent panic handlers from running. However, running handlers before
unwinding allows them access to more context, for example, the ability to take
a stack trace.

`PanicInfo::location` could be split into `PanicInfo::file` and
`PanicInfo::line` to cut down on the API size, though that would require
handlers to deal with weird cases like a line number but no file being
available.

[RFC 1100](https://github.com/rust-lang/rfcs/pull/1100) proposed an API based
around thread-local handlers. While there are reasonable use cases for the
registration of custom handlers on a per-thread basis, most of the common uses
for custom handlers want to have a single set of behavior cover all threads in
the process. Being forced to remember to register a handler in every thread
spawned in a program is tedious and error prone, and not even possible in many
cases for threads spawned in libraries the author has no control over.

While out of scope for this RFC, a future extension could add thread-local
handlers on top of the global one proposed here in a straightforward manner.

The implementation could be simplified by altering the API to store, and
`take_logger` to return, an `Arc<Fn(&PanicInfo) + 'static + Sync + Send>` or
a bare function pointer. This seems like a somewhat weirder API, however, and
the implementation proposed above should not end up complex enough to justify
the change.

# Unresolved questions

None at the moment.
