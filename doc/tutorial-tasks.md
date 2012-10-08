% Rust Tasks and Communication Tutorial

# Introduction

The Rust language is designed from the ground up to support pervasive
and safe concurrency through lightweight, memory-isolated tasks and
message passing.

Rust tasks are not the same as traditional threads - they are what are
often referred to as _green threads_, cooperatively scheduled by the
Rust runtime onto a small number of operating system threads.  Being
significantly cheaper to create than traditional threads, Rust can
create hundreds of thousands of concurrent tasks on a typical 32-bit
system.

Tasks provide failure isolation and recovery. When an exception occurs
in rust code (either by calling `fail` explicitly or by otherwise performing
an invalid operation) the entire task is destroyed - there is no way
to `catch` an exception as in other languages. Instead tasks may monitor
each other to detect when failure has occurred.

Rust tasks have dynamically sized stacks. When a task is first created
it starts off with a small amount of stack (currently in the low
thousands of bytes, depending on platform) and more stack is acquired as
needed. A Rust task will never run off the end of the stack as is
possible in many other languages, but they do have a stack budget, and
if a Rust task exceeds its stack budget then it will fail safely.

Tasks make use of Rust's type system to provide strong memory safety
guarantees, disallowing shared mutable state. Communication between
tasks is facilitated by the transfer of _owned_ data through the
global _exchange heap_.

This tutorial will explain the basics of tasks and communication in Rust,
explore some typical patterns in concurrent Rust code, and finally
discuss some of the more exotic synchronization types in the standard
library.

> ***Warning:*** This tutorial is incomplete

## A note about the libraries

While Rust's type system provides the building blocks needed for safe
and efficient tasks, all of the task functionality itself is implemented
in the core and standard libraries, which are still under development
and do not always present a consistent interface.

In particular, there are currently two independent modules that provide
a message passing interface to Rust code: `core::comm` and `core::pipes`.
`core::comm` is an older, less efficient system that is being phased out
in favor of `pipes`. At some point the existing `core::comm` API will
be removed and the user-facing portions of `core::pipes` will be moved
to `core::comm`. In this tutorial we will discuss `pipes` and ignore
the `comm` API.

For your reference, these are the standard modules involved in Rust
concurrency at the moment.

* [`core::task`] - All code relating to tasks and task scheduling
* [`core::comm`] - The deprecated message passing API
* [`core::pipes`] - The new message passing infrastructure and API
* [`std::comm`] - Higher level messaging types based on `core::pipes`
* [`std::sync`] - More exotic synchronization tools, including locks
* [`std::arc`] - The ARC type, for safely sharing immutable data
* [`std::par`] - Some basic tools for implementing parallel algorithms

[`core::task`]: core/task.html
[`core::comm`]: core/comm.html
[`core::pipes`]: core/pipes.html
[`std::comm`]: std/comm.html
[`std::sync`]: std/sync.html
[`std::arc`]: std/arc.html
[`std::par`]: std/par.html

# Basics

The programming interface for creating and managing tasks is contained
in the `task` module of the `core` library, making it available to all
Rust code by default. At it's simplest, creating a task is a matter of
calling the `spawn` function, passing a closure to run in the new
task.

~~~~
# use io::println;
use task::spawn;

// Print something profound in a different task using a named function
fn print_message() { println("I am running in a different task!"); }
spawn(print_message);

// Print something more profound in a different task using a lambda expression
spawn( || println("I am also running in a different task!") );

// The canonical way to spawn is using `do` notation
do spawn {
    println("I too am running in a different task!");
}
~~~~

In Rust, there is nothing special about creating tasks - the language
itself doesn't know what a 'task' is. Instead, Rust provides in the
type system all the tools necessary to implement safe concurrency,
_owned types_ in particular, and leaves the dirty work up to the
core library.

The `spawn` function has a very simple type signature: `fn spawn(f:
~fn())`. Because it accepts only owned closures, and owned closures
contained only owned data, `spawn` can safely move the entire closure
and all its associated state into an entirely different task for
execution. Like any closure, the function passed to spawn may capture
an environment that it carries across tasks.

~~~
# use io::println;
# use task::spawn;
# fn generate_task_number() -> int { 0 }
// Generate some state locally
let child_task_number = generate_task_number();

do spawn {
   // Capture it in the remote task
   println(fmt!("I am child number %d", child_task_number));
}
~~~

By default tasks will be multiplexed across the available cores, running
in parallel, thus on a multicore machine, running the following code
should interleave the output in vaguely random order.

~~~
# use io::print;
# use task::spawn;

for int::range(0, 20) |child_task_number| {
    do spawn {
       print(fmt!("I am child number %d\n", child_task_number));
    }
}
~~~

## Communication

Now that we have spawned a new task, it would be nice if we could
communicate with it. Recall that Rust does not have shared mutable
state, so one task may not manipulate variables owned by another task.
Instead we use *pipes*.

Pipes are simply a pair of endpoints, with one for sending messages
and another for receiving messages. Pipes are low-level communication
building-blocks and so come in a variety of forms, appropriate for
different use cases, but there are just a few varieties that are most
commonly used, which we will cover presently.

The simplest way to create a pipe is to use the `pipes::stream`
function to create a `(Chan, Port)` pair. In Rust parlance a 'channel'
is a sending endpoint of a pipe, and a 'port' is the receiving
endpoint. Consider the following example of performing two calculations
concurrently.

~~~~
use task::spawn;
use pipes::{stream, Port, Chan};

let (chan, port): (Chan<int>, Port<int>) = stream();

do spawn {
    let result = some_expensive_computation();
    chan.send(result);
}

some_other_expensive_computation();
let result = port.recv();
# fn some_expensive_computation() -> int { 42 }
# fn some_other_expensive_computation() {}
~~~~

Let's examine this example in detail. The `let` statement first creates a
stream for sending and receiving integers (recall that `let` can be
used for destructuring patterns, in this case separating a tuple into
its component parts).

~~~~
# use pipes::{stream, Chan, Port};
let (chan, port): (Chan<int>, Port<int>) = stream();
~~~~

The channel will be used by the child task to send data to the parent task,
which will wait to receive the data on the port. The next statement
spawns the child task.

~~~~
# use task::{spawn};
# use task::spawn;
# use pipes::{stream, Port, Chan};
# fn some_expensive_computation() -> int { 42 }
# let (chan, port) = stream();
do spawn {
    let result = some_expensive_computation();
    chan.send(result);
}
~~~~

Notice that `chan` was transferred to the child task implicitly by
capturing it in the task closure. Both `Chan` and `Port` are sendable
types and may be captured into tasks or otherwise transferred between
them. In the example, the child task performs an expensive computation
then sends the result over the captured channel.

Finally, the parent continues by performing some other expensive
computation and then waiting for the child's result to arrive on the
port:

~~~~
# use pipes::{stream, Port, Chan};
# fn some_other_expensive_computation() {}
# let (chan, port) = stream::<int>();
# chan.send(0);
some_other_expensive_computation();
let result = port.recv();
~~~~

The `Port` and `Chan` pair created by `stream` enable efficient
communication between a single sender and a single receiver, but
multiple senders cannot use a single `Chan`, nor can multiple
receivers use a single `Port`.  What if our example needed to perform
multiple computations across a number of tasks? The following cannot
be written:

~~~ {.xfail-test}
# use task::{spawn};
# use pipes::{stream, Port, Chan};
# fn some_expensive_computation() -> int { 42 }
let (chan, port) = stream();

do spawn {
    chan.send(some_expensive_computation());
}

// ERROR! The previous spawn statement already owns the channel,
// so the compiler will not allow it to be captured again
do spawn {
    chan.send(some_expensive_computation());
}
~~~

Instead we can use a `SharedChan`, a type that allows a single
`Chan` to be shared by multiple senders.

~~~
# use task::spawn;
use pipes::{stream, SharedChan};

let (chan, port) = stream();
let chan = SharedChan(move chan);

for uint::range(0, 3) |init_val| {
    // Create a new channel handle to distribute to the child task
    let child_chan = chan.clone();
    do spawn {
        child_chan.send(some_expensive_computation(init_val));
    }
}

let result = port.recv() + port.recv() + port.recv();
# fn some_expensive_computation(_i: uint) -> int { 42 }
~~~

Here we transfer ownership of the channel into a new `SharedChan`
value.  Like `Chan`, `SharedChan` is a non-copyable, owned type
(sometimes also referred to as an 'affine' or 'linear' type). Unlike
`Chan` though, `SharedChan` may be duplicated with the `clone()`
method.  A cloned `SharedChan` produces a new handle to the same
channel, allowing multiple tasks to send data to a single port.
Between `spawn`, `stream` and `SharedChan` we have enough tools
to implement many useful concurrency patterns.

Note that the above `SharedChan` example is somewhat contrived since
you could also simply use three `stream` pairs, but it serves to
illustrate the point. For reference, written with multiple streams it
might look like the example below.

~~~
# use task::spawn;
# use pipes::{stream, Port, Chan};

// Create a vector of ports, one for each child task
let ports = do vec::from_fn(3) |init_val| {
    let (chan, port) = stream();
    do spawn {
        chan.send(some_expensive_computation(init_val));
    }
    port
};

// Wait on each port, accumulating the results
let result = ports.foldl(0, |accum, port| *accum + port.recv() );
# fn some_expensive_computation(_i: uint) -> int { 42 }
~~~

# Handling task failure

Rust has a built-in mechanism for raising exceptions, written `fail`
(or `fail ~"reason"`, or sometimes `assert expr`), and it causes the
task to unwind its stack, running destructors and freeing memory along
the way, and then exit itself. Unlike C++, exceptions in Rust are
unrecoverable within a single task - once a task fails there is no way
to "catch" the exception.

All tasks are, by default, _linked_ to each other, meaning their fate
is intertwined, and if one fails so do all of them.

~~~
# use task::spawn;
# fn do_some_work() { loop { task::yield() } }
# do task::try {
// Create a child task that fails
do spawn { fail }

// This will also fail because the task we spawned failed
do_some_work();
# };
~~~

While it isn't possible for a task to recover from failure,
tasks may be notified when _other_ tasks fail. The simplest way
of handling task failure is with the `try` function, which is
similar to spawn, but immediately blocks waiting for the child
task to finish.

~~~
# fn some_condition() -> bool { false }
# fn calculate_result() -> int { 0 }
let result: Result<int, ()> = do task::try {
    if some_condition() {
        calculate_result()
    } else {
        fail ~"oops!";
    }
};
assert result.is_err();
~~~

Unlike `spawn`, the function spawned using `try` may return a value,
which `try` will dutifully propagate back to the caller in a [`Result`]
enum. If the child task terminates successfully, `try` will
return an `Ok` result; if the child task fails, `try` will return
an `Error` result.

[`Result`]: core/result.html

> ***Note:*** A failed task does not currently produce a useful error
> value (all error results from `try` are equal to `Err(())`). In the
> future it may be possible for tasks to intercept the value passed to
> `fail`.

TODO: Need discussion of `future_result` in order to make failure
modes useful.

But not all failure is created equal. In some cases you might need to
abort the entire program (perhaps you're writing an assert which, if
it trips, indicates an unrecoverable logic error); in other cases you
might want to contain the failure at a certain boundary (perhaps a
small piece of input from the outside world, which you happen to be
processing in parallel, is malformed and its processing task can't
proceed). Hence the need for different _linked failure modes_.

## Failure modes

By default, task failure is _bidirectionally linked_, which means if
either task dies, it kills the other one.

~~~
# fn sleep_forever() { loop { task::yield() } }
# do task::try {
do task::spawn {
    do task::spawn {
        fail;  // All three tasks will die.
    }
    sleep_forever();  // Will get woken up by force, then fail
}
sleep_forever();  // Will get woken up by force, then fail
# };
~~~

If you want parent tasks to kill their children, but not for a child
task's failure to kill the parent, you can call
`task::spawn_supervised` for _unidirectionally linked_ failure. The
function `task::try`, which we saw previously, uses `spawn_supervised`
internally, with additional logic to wait for the child task to finish
before returning. Hence:

~~~
# use pipes::{stream, Chan, Port};
# use task::{spawn, try};
# fn sleep_forever() { loop { task::yield() } }
# do task::try {
let (sender, receiver): (Chan<int>, Port<int>) = stream();
do spawn {  // Bidirectionally linked
    // Wait for the supervised child task to exist.
    let message = receiver.recv();
    // Kill both it and the parent task.
    assert message != 42;
}
do try {  // Unidirectionally linked
    sender.send(42);
    sleep_forever();  // Will get woken up by force
}
// Flow never reaches here -- parent task was killed too.
# };
~~~

Supervised failure is useful in any situation where one task manages
multiple fallible child tasks, and the parent task can recover
if any child files. On the other hand, if the _parent_ (supervisor) fails
then there is nothing the children can do to recover, so they should
also fail.

Supervised task failure propagates across multiple generations even if
an intermediate generation has already exited:

~~~
# fn sleep_forever() { loop { task::yield() } }
# fn wait_for_a_while() { for 1000.times { task::yield() } }
# do task::try::<int> {
do task::spawn_supervised {
    do task::spawn_supervised {
        sleep_forever();  // Will get woken up by force, then fail
    }
    // Intermediate task immediately exits
}
wait_for_a_while();
fail;  // Will kill grandchild even if child has already exited
# };
~~~

Finally, tasks can be configured to not propagate failure to each
other at all, using `task::spawn_unlinked` for _isolated failure_.

~~~
# fn random() -> int { 100 }
# fn sleep_for(i: int) { for i.times { task::yield() } }
# do task::try::<()> {
let (time1, time2) = (random(), random());
do task::spawn_unlinked {
    sleep_for(time2);  // Won't get forced awake
    fail;
}
sleep_for(time1);  // Won't get forced awake
fail;
// It will take MAX(time1,time2) for the program to finish.
# };
~~~

## Creating a task with a bi-directional communication path

A very common thing to do is to spawn a child task where the parent
and child both need to exchange messages with each other. The
function `std::comm::DuplexStream()` supports this pattern.  We'll
look briefly at how it is used.

To see how `spawn_conversation()` works, we will create a child task
that receives `uint` messages, converts them to a string, and sends
the string in response.  The child terminates when `0` is received.
Here is the function that implements the child task:

~~~~
# use std::comm::DuplexStream;
# use pipes::{Port, Chan};
fn stringifier(channel: &DuplexStream<~str, uint>) {
    let mut value: uint;
    loop {
        value = channel.recv();
        channel.send(uint::to_str(value, 10u));
        if value == 0u { break; }
    }
}
~~~~

The implementation of `DuplexStream` supports both sending and
receiving. The `stringifier` function takes a `DuplexStream` that can
send strings (the first type parameter) and receive `uint` messages
(the second type parameter). The body itself simply loops, reading
from the channel and then sending its response back.  The actual
response itself is simply the strified version of the received value,
`uint::to_str(value)`.

Here is the code for the parent task:

~~~~
# use std::comm::DuplexStream;
# use pipes::{Port, Chan};
# use task::spawn;
# fn stringifier(channel: &DuplexStream<~str, uint>) {
#     let mut value: uint;
#     loop {
#         value = channel.recv();
#         channel.send(uint::to_str(value, 10u));
#         if value == 0u { break; }
#     }
# }
# fn main() {

let (from_child, to_child) = DuplexStream();

do spawn || {
    stringifier(&to_child);
};

from_child.send(22u);
assert from_child.recv() == ~"22";

from_child.send(23u);
from_child.send(0u);

assert from_child.recv() == ~"23";
assert from_child.recv() == ~"0";

# }
~~~~

The parent task first calls `DuplexStream` to create a pair of bidirectional endpoints. It then uses `task::spawn` to create the child task, which captures one end of the communication channel.  As a result, both parent
and child can send and receive data to and from the other.

