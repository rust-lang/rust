# Tasks

Rust supports a system of lightweight tasks, similar to what is found
in Erlang or other actor systems.  Rust tasks communicate via messages
and do not share data.  However, it is possible to send data without
copying it by making use of [unique boxes][uniques] (still, the data
is owned by only one task at a time).

[uniques]: data.html#unique-box

NOTE: As Rust evolves, we expect the Task API to grow and change
somewhat.  The tutorial documents the API as it exists today.

## Spawning a task

Spawning a task is done using the various spawn functions in the
module task.  Let's begin with the simplest one, `task::spawn()`, and
later move on to the others:

    let some_value = 22;
    let child_task = task::spawn {||
        std::io::println("This executes in the child task.");
        std::io::println(#fmt("%d", some_value));
    };

The argument to `task::spawn()` is a [unique closure](func) of type
`fn~()`, meaning that it takes no arguments and generates no return
value.  The effect of `task::spawn()` is to fire up a child task that
will execute the closure in parallel with the creator.  The result is
a task id, here stored into the variable `child_task`.

## Ports and channels

Now that we have spawned a child task, it would be nice if we could
communicate with it.  This is done by creating a *port* with an
associated *channel*.  A port is simply a location to receive messages
of a particular type.  A channel is used to send messages to a port.
For example, imagine we wish to perform two expensive computations
in parallel.  We might write something like:

    let port = comm::port::<int>();
    let chan = comm::chan::<int>(port);
    let child_task = task::spawn {||
        let result = some_expensive_computation();
        comm::send(chan, result);
    };
    some_other_expensive_computation();
    let result = comm::recv(port);

Let's walk through this code line-by-line.  The first line creates a
port for receiving integers:

    let port = comm::port::<int>();
    
This port is where we will receive the message from the child task
once it is complete.  The second line creates a channel for sending
integers to the port `port`:

    let chan = comm::chan::<int>(port);

The channel will be used by the child to send a message to the port.
The next statement actually spawns the child:

    let child_task = task::spawn {||
        let result = some_expensive_computation();
        comm::send(chan, result);
    };

This child will perform the expensive computation send the result
over the channel.  Finally, the parent continues by performing
some other expensive computation and then waiting for the child's result
to arrive on the port:

    some_other_expensive_computation();
    let result = comm::recv(port);

