% Rust Tasks and Communication Tutorial

# Introduction

Rust supports concurrency and parallelism through lightweight tasks.
Rust tasks are significantly cheaper to create than traditional
threads, with a typical 32-bit system able to run hundreds of
thousands simultaneously. Tasks in Rust are what are often referred to
as _green threads_, cooperatively scheduled by the Rust runtime onto a
small number of operating system threads.

Tasks provide failure isolation and recovery. When an exception occurs
in rust code (either by calling `fail` explicitly or by otherwise performing
an invalid operation) the entire task is destroyed - there is no way
to `catch` an exception as in other languages. Instead tasks may monitor
each other to detect when failure has occurred.

Rust tasks have dynamically sized stacks. When a task is first created
it starts off with a small amount of stack (in the hundreds to
low thousands of bytes, depending on plattform), and more stack is
added as needed. A Rust task will never run off the end of the stack as
is possible in many other languages, but they do have a stack budget,
and if a Rust task exceeds its stack budget then it will fail safely.

Tasks make use of Rust's type system to provide strong memory safety
guarantees, disallowing shared mutable state. Communication between
tasks is facilitated by the transfer of _owned_ data through the
global _exchange heap_.

This tutorial will explain the basics of tasks and communication in Rust,
explore some typical patterns in concurrent Rust code, and finally
discuss some of the more exotic synchronization types in the standard
library.

# A note about the libraries

While Rust's type system provides the building blocks needed for safe
and efficient tasks, all of the task functionality itself is implemented
in the core and standard libraries, which are still under development
and do not always present a nice programming interface.

In particular, there are currently two independent modules that provide
a message passing interface to Rust code: `core::comm` and `core::pipes`.
`core::comm` is an older, less efficient system that is being phased out
in favor of `pipes`. At some point the existing `core::comm` API will
be romoved and the user-facing portions of `core::pipes` will be moved
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

[`core::task`]: core/task.html
[`core::comm`]: core/comm.html
[`core::pipes`]: core/pipes.html
[`std::comm`]: std/comm.html
[`std::sync`]: std/sync.html
[`std::arc`]: std/arc.html


# Spawning a task

Spawning a task is done using the various spawn functions in the
module `task`.  Let's begin with the simplest one, `task::spawn()`:

~~~~
use task::spawn;
use io::println;

let some_value = 22;

do spawn {
    println(~"This executes in the child task.");
    println(fmt!("%d", some_value));
}
~~~~

The argument to `task::spawn()` is a [unique
closure](#unique-closures) of type `fn~()`, meaning that it takes no
arguments and generates no return value. The effect of `task::spawn()`
is to fire up a child task that will execute the closure in parallel
with the creator.

# Communication

Now that we have spawned a child task, it would be nice if we could
communicate with it. This is done using *pipes*. Pipes are simply a
pair of endpoints, with one for sending messages and another for
receiving messages. The easiest way to create a pipe is to use
`pipes::stream`.  Imagine we wish to perform two expensive
computations in parallel.  We might write something like:

~~~~
use task::spawn;
use pipes::{stream, Port, Chan};

let (chan, port) = stream();

do spawn {
    let result = some_expensive_computation();
    chan.send(result);
}

some_other_expensive_computation();
let result = port.recv();

# fn some_expensive_computation() -> int { 42 }
# fn some_other_expensive_computation() {}
~~~~

Let's walk through this code line-by-line.  The first line creates a
stream for sending and receiving integers:

~~~~ {.ignore}
# use pipes::stream;
let (chan, port) = stream();
~~~~

This port is where we will receive the message from the child task
once it is complete.  The channel will be used by the child to send a
message to the port.  The next statement actually spawns the child:

~~~~
# use task::{spawn};
# use comm::{Port, Chan};
# fn some_expensive_computation() -> int { 42 }
# let port = Port();
# let chan = port.chan();
do spawn {
    let result = some_expensive_computation();
    chan.send(result);
}
~~~~

This child will perform the expensive computation send the result
over the channel.  (Under the hood, `chan` was captured by the
closure that forms the body of the child task.  This capture is
allowed because channels are sendable.)

Finally, the parent continues by performing
some other expensive computation and then waiting for the child's result
to arrive on the port:

~~~~
# use pipes::{stream, Port, Chan};
# fn some_other_expensive_computation() {}
# let (chan, port) = stream::<int>();
# chan.send(0);
some_other_expensive_computation();
let result = port.recv();
~~~~

# Creating a task with a bi-directional communication path

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
