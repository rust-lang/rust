Interfaces for working with Errors.

# Error Handling In Rust

The Rust language provides two complementary systems for constructing /
representing, reporting, propagating, reacting to, and discarding errors.
These responsibilities are collectively known as "error handling." The
components of the first system, the panic runtime and interfaces, are most
commonly used to represent bugs that have been detected in your program. The
components of the second system, `Result`, the error traits, and user
defined types, are used to represent anticipated runtime failure modes of
your program.

## The Panic Interfaces

The following are the primary interfaces of the panic system and the
responsibilities they cover:

* [`panic!`] and [`panic_any`] (Constructing, Propagated automatically)
* [`PanicInfo`] (Reporting)
* [`set_hook`], [`take_hook`], and [`#[panic_handler]`][panic-handler] (Reporting)
* [`catch_unwind`] and [`resume_unwind`] (Discarding, Propagating)

The following are the primary interfaces of the error system and the
responsibilities they cover:

* [`Result`] (Propagating, Reacting)
* The [`Error`] trait (Reporting)
* User defined types (Constructing / Representing)
* [`match`] and [`downcast`] (Reacting)
* The question mark operator ([`?`]) (Propagating)
* The partially stable [`Try`] traits (Propagating, Constructing)
* [`Termination`] (Reporting)

## Converting Errors into Panics

The panic and error systems are not entirely distinct. Often times errors
that are anticipated runtime failures in an API might instead represent bugs
to a caller. For these situations the standard library provides APIs for
constructing panics with an `Error` as it's source.

* [`Result::unwrap`]
* [`Result::expect`]

These functions are equivalent, they either return the inner value if the
`Result` is `Ok` or panic if the `Result` is `Err` printing the inner error
as the source. The only difference between them is that with `expect` you
provide a panic error message to be printed alongside the source, whereas
`unwrap` has a default message indicating only that you unwraped an `Err`.

Of the two, `expect` is generally preferred since its `msg` field allows you
to convey your intent and assumptions which makes tracking down the source
of a panic easier. `unwrap` on the other hand can still be a good fit in
situations where you can trivially show that a piece of code will never
panic, such as `"127.0.0.1".parse::<std::net::IpAddr>().unwrap()` or early
prototyping.

# Common Message Styles

There are two common styles for how people word `expect` messages. Using
the message to present information to users encountering a panic
("expect as error message") or using the message to present information
to developers debugging the panic ("expect as precondition").

In the former case the expect message is used to describe the error that
has occurred which is considered a bug. Consider the following example:

```should_panic
// Read environment variable, panic if it is not present
let path = std::env::var("IMPORTANT_PATH").unwrap();
```

In the "expect as error message" style we would use expect to describe
that the environment variable was not set when it should have been:

```should_panic
let path = std::env::var("IMPORTANT_PATH")
    .expect("env variable `IMPORTANT_PATH` is not set");
```

In the "expect as precondition" style, we would instead describe the
reason we _expect_ the `Result` should be `Ok`. With this style we would
prefer to write:

```should_panic
let path = std::env::var("IMPORTANT_PATH")
    .expect("env variable `IMPORTANT_PATH` should be set by `wrapper_script.sh`");
```

The "expect as error message" style does not work as well with the
default output of the std panic hooks, and often ends up repeating
information that is already communicated by the source error being
unwrapped:

```text
thread 'main' panicked at 'env variable `IMPORTANT_PATH` is not set: NotPresent', src/main.rs:4:6
```

In this example we end up mentioning that an env variable is not set,
followed by our source message that says the env is not present, the
only additional information we're communicating is the name of the
environment variable being checked.

The "expect as precondition" style instead focuses on source code
readability, making it easier to understand what must have gone wrong in
situations where panics are being used to represent bugs exclusively.
Also, by framing our expect in terms of what "SHOULD" have happened to
prevent the source error, we end up introducing new information that is
independent from our source error.

```text
thread 'main' panicked at 'env variable `IMPORTANT_PATH` should be set by `wrapper_script.sh`: NotPresent', src/main.rs:4:6
```

In this example we are communicating not only the name of the
environment variable that should have been set, but also an explanation
for why it should have been set, and we let the source error display as
a clear contradiction to our expectation.

**Hint**: If you're having trouble remembering how to phrase
expect-as-precondition style error messages remember to focus on the word
"should" as in "env variable should be set by blah" or "the given binary
should be available and executable by the current user".

[`panic_any`]: ../../std/panic/fn.panic_any.html
[`PanicInfo`]: crate::panic::PanicInfo
[`catch_unwind`]: ../../std/panic/fn.catch_unwind.html
[`resume_unwind`]: ../../std/panic/fn.resume_unwind.html
[`downcast`]: crate::error::Error
[`Termination`]: ../../std/process/trait.Termination.html
[`Try`]: crate::ops::Try
[panic hook]: ../../std/panic/fn.set_hook.html
[`set_hook`]: ../../std/panic/fn.set_hook.html
[`take_hook`]: ../../std/panic/fn.take_hook.html
[panic-handler]: <https://doc.rust-lang.org/nomicon/panic-handler.html>
[`match`]: ../../std/keyword.match.html
[`?`]: ../../std/result/index.html#the-question-mark-operator-
