% Destructors

Unlike constructors, destructors in Rust have a special status: they are added
by implementing `Drop` for a type, and they are automatically invoked as values
go out of scope.

> **[FIXME]** This section needs to be expanded.

### Destructors should not fail. [FIXME: needs RFC]

Destructors are executed on thread failure, and in that context a failing
destructor causes the program to abort.

Instead of failing in a destructor, provide a separate method for checking for
clean teardown, e.g. a `close` method, that returns a `Result` to signal
problems.

### Destructors should not block. [FIXME: needs RFC]

Similarly, destructors should not invoke blocking operations, which can make
debugging much more difficult. Again, consider providing a separate method for
preparing for an infallible, nonblocking teardown.
