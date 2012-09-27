% Rust Tasks and Communication Tutorial

# Introduction

Rust supports a system of lightweight tasks, similar to what is found
in Erlang or other actor systems. Rust tasks communicate via messages
and do not share data. However, it is possible to send data without
copying it by making use of [the exchange heap](#unique-boxes), which
allow the sending task to release ownership of a value, so that the
receiving task can keep on using it.

> ***Note:*** As Rust evolves, we expect the task API to grow and
> change somewhat.  The tutorial documents the API as it exists today.

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
