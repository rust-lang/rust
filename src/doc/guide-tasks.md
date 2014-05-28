% The Rust Tasks and Communication Guide

# Introduction

Rust provides safe concurrency through a combination
of lightweight, memory-isolated tasks and message passing.
This guide will describe the concurrency model in Rust, how it
relates to the Rust type system, and introduce
the fundamental library abstractions for constructing concurrent programs.

Rust tasks are not the same as traditional threads: rather,
they are considered _green threads_, lightweight units of execution that the Rust
runtime schedules cooperatively onto a small number of operating system threads.
On a multi-core system Rust tasks will be scheduled in parallel by default.
Because tasks are significantly
cheaper to create than traditional threads, Rust can create hundreds of
thousands of concurrent tasks on a typical 32-bit system.
In general, all Rust code executes inside a task, including the `main` function.

In order to make efficient use of memory Rust tasks have dynamically sized stacks.
A task begins its life with a small
amount of stack space (currently in the low thousands of bytes, depending on
platform), and acquires more stack as needed.
Unlike in languages such as C, a Rust task cannot accidentally write to
memory beyond the end of the stack, causing crashes or worse.

Tasks provide failure isolation and recovery. When a fatal error occurs in Rust
code as a result of an explicit call to `fail!()`, an assertion failure, or
another invalid operation, the runtime system destroys the entire
task. Unlike in languages such as Java and C++, there is no way to `catch` an
exception. Instead, tasks may monitor each other for failure.

Tasks use Rust's type system to provide strong memory safety guarantees. In
particular, the type system guarantees that tasks cannot share mutable state
with each other. Tasks communicate with each other by transferring _owned_
data through the global _exchange heap_.

## A note about the libraries

While Rust's type system provides the building blocks needed for safe
and efficient tasks, all of the task functionality itself is implemented
in the standard and sync libraries, which are still under development
and do not always present a consistent or complete interface.

For your reference, these are the standard modules involved in Rust
concurrency at this writing:

* [`std::task`] - All code relating to tasks and task scheduling,
* [`std::comm`] - The message passing interface,
* [`sync::DuplexStream`] - An extension of `pipes::stream` that allows both sending and receiving,
* [`sync::Arc`] - The Arc (atomically reference counted) type, for safely sharing immutable data,
* [`sync::Semaphore`] - A counting, blocking, bounded-waiting semaphore,
* [`sync::Mutex`] - A blocking, bounded-waiting, mutual exclusion lock with an associated
    FIFO condition variable,
* [`sync::RWLock`] - A blocking, no-starvation, reader-writer lock with an associated condvar,
* [`sync::Barrier`] - A barrier enables multiple tasks to synchronize the beginning
    of some computation,
* [`sync::TaskPool`] - A task pool abstraction,
* [`sync::Future`] - A type encapsulating the result of a computation which may not be complete,
* [`sync::one`] - A "once initialization" primitive
* [`sync::mutex`] - A proper mutex implementation regardless of the "flavor of task" which is
    acquiring the lock.

[`std::task`]: std/task/index.html
[`std::comm`]: std/comm/index.html
[`sync::DuplexStream`]: sync/struct.DuplexStream.html
[`sync::Arc`]: sync/struct.Arc.html
[`sync::Semaphore`]: sync/raw/struct.Semaphore.html
[`sync::Mutex`]: sync/struct.Mutex.html
[`sync::RWLock`]: sync/struct.RWLock.html
[`sync::Barrier`]: sync/struct.Barrier.html
[`sync::TaskPool`]: sync/struct.TaskPool.html
[`sync::Future`]: sync/struct.Future.html
[`sync::one`]: sync/one/index.html
[`sync::mutex`]: sync/mutex/index.html

# Basics

The programming interface for creating and managing tasks lives
in the `task` module of the `std` library, and is thus available to all
Rust code by default. At its simplest, creating a task is a matter of
calling the `spawn` function with a closure argument. `spawn` executes the
closure in the new task.

~~~~
# use std::task::spawn;

// Print something profound in a different task using a named function
fn print_message() { println!("I am running in a different task!"); }
spawn(print_message);

// Print something profound in a different task using a `proc` expression
// The `proc` expression evaluates to an (unnamed) owned closure.
// That closure will call `println!(...)` when the spawned task runs.

spawn(proc() println!("I am also running in a different task!") );
~~~~

In Rust, there is nothing special about creating tasks: a task is not a
concept that appears in the language semantics. Instead, Rust's type system
provides all the tools necessary to implement safe concurrency: particularly,
_owned types_. The language leaves the implementation details to the standard
library.

The `spawn` function has a very simple type signature: `fn spawn(f:
proc())`. Because it accepts only owned closures, and owned closures
contain only owned data, `spawn` can safely move the entire closure
and all its associated state into an entirely different task for
execution. Like any closure, the function passed to `spawn` may capture
an environment that it carries across tasks.

~~~
# use std::task::spawn;
# fn generate_task_number() -> int { 0 }
// Generate some state locally
let child_task_number = generate_task_number();

spawn(proc() {
    // Capture it in the remote task
    println!("I am child number {}", child_task_number);
});
~~~

## Communication

Now that we have spawned a new task, it would be nice if we could
communicate with it. Recall that Rust does not have shared mutable
state, so one task may not manipulate variables owned by another task.
Instead we use *pipes*.

A pipe is simply a pair of endpoints: one for sending messages and another for
receiving messages. Pipes are low-level communication building-blocks and so
come in a variety of forms, each one appropriate for a different use case. In
what follows, we cover the most commonly used varieties.

The simplest way to create a pipe is to use the `channel`
function to create a `(Sender, Receiver)` pair. In Rust parlance, a *sender*
is a sending endpoint of a pipe, and a *receiver* is the receiving
endpoint. Consider the following example of calculating two results
concurrently:

~~~~
# use std::task::spawn;

let (tx, rx): (Sender<int>, Receiver<int>) = channel();

spawn(proc() {
    let result = some_expensive_computation();
    tx.send(result);
});

some_other_expensive_computation();
let result = rx.recv();
# fn some_expensive_computation() -> int { 42 }
# fn some_other_expensive_computation() {}
~~~~

Let's examine this example in detail. First, the `let` statement creates a
stream for sending and receiving integers (the left-hand side of the `let`,
`(tx, rx)`, is an example of a *destructuring let*: the pattern separates
a tuple into its component parts).

~~~~
let (tx, rx): (Sender<int>, Receiver<int>) = channel();
~~~~

The child task will use the sender to send data to the parent task,
which will wait to receive the data on the receiver. The next statement
spawns the child task.

~~~~
# use std::task::spawn;
# fn some_expensive_computation() -> int { 42 }
# let (tx, rx) = channel();
spawn(proc() {
    let result = some_expensive_computation();
    tx.send(result);
});
~~~~

Notice that the creation of the task closure transfers `tx` to the child
task implicitly: the closure captures `tx` in its environment. Both `Sender`
and `Receiver` are sendable types and may be captured into tasks or otherwise
transferred between them. In the example, the child task runs an expensive
computation, then sends the result over the captured channel.

Finally, the parent continues with some other expensive
computation, then waits for the child's result to arrive on the
receiver:

~~~~
# fn some_other_expensive_computation() {}
# let (tx, rx) = channel::<int>();
# tx.send(0);
some_other_expensive_computation();
let result = rx.recv();
~~~~

The `Sender` and `Receiver` pair created by `channel` enables efficient
communication between a single sender and a single receiver, but multiple
senders cannot use a single `Sender` value, and multiple receivers cannot use a
single `Receiver` value.  What if our example needed to compute multiple
results across a number of tasks? The following program is ill-typed:

~~~ {.ignore}
# fn some_expensive_computation() -> int { 42 }
let (tx, rx) = channel();

spawn(proc() {
    tx.send(some_expensive_computation());
});

// ERROR! The previous spawn statement already owns the sender,
// so the compiler will not allow it to be captured again
spawn(proc() {
    tx.send(some_expensive_computation());
});
~~~

Instead we can clone the `tx`, which allows for multiple senders.

~~~
let (tx, rx) = channel();

for init_val in range(0u, 3) {
    // Create a new channel handle to distribute to the child task
    let child_tx = tx.clone();
    spawn(proc() {
        child_tx.send(some_expensive_computation(init_val));
    });
}

let result = rx.recv() + rx.recv() + rx.recv();
# fn some_expensive_computation(_i: uint) -> int { 42 }
~~~

Cloning a `Sender` produces a new handle to the same channel, allowing multiple
tasks to send data to a single receiver. It upgrades the channel internally in
order to allow this functionality, which means that channels that are not
cloned can avoid the overhead required to handle multiple senders. But this
fact has no bearing on the channel's usage: the upgrade is transparent.

Note that the above cloning example is somewhat contrived since
you could also simply use three `Sender` pairs, but it serves to
illustrate the point. For reference, written with multiple streams, it
might look like the example below.

~~~
# use std::task::spawn;

// Create a vector of ports, one for each child task
let rxs = Vec::from_fn(3, |init_val| {
    let (tx, rx) = channel();
    spawn(proc() {
        tx.send(some_expensive_computation(init_val));
    });
    rx
});

// Wait on each port, accumulating the results
let result = rxs.iter().fold(0, |accum, rx| accum + rx.recv() );
# fn some_expensive_computation(_i: uint) -> int { 42 }
~~~

## Backgrounding computations: Futures
With `sync::Future`, rust has a mechanism for requesting a computation and getting the result
later.

The basic example below illustrates this.

~~~
extern crate sync;

# fn main() {
# fn make_a_sandwich() {};
fn fib(n: u64) -> u64 {
    // lengthy computation returning an uint
    12586269025
}

let mut delayed_fib = sync::Future::spawn(proc() fib(50));
make_a_sandwich();
println!("fib(50) = {:?}", delayed_fib.get())
# }
~~~

The call to `future::spawn` returns immediately a `future` object regardless of how long it
takes to run `fib(50)`. You can then make yourself a sandwich while the computation of `fib` is
running. The result of the execution of the method is obtained by calling `get` on the future.
This call will block until the value is available (*i.e.* the computation is complete). Note that
the future needs to be mutable so that it can save the result for next time `get` is called.

Here is another example showing how futures allow you to background computations. The workload will
be distributed on the available cores.

~~~
# extern crate sync;
fn partial_sum(start: uint) -> f64 {
    let mut local_sum = 0f64;
    for num in range(start*100000, (start+1)*100000) {
        local_sum += (num as f64 + 1.0).powf(-2.0);
    }
    local_sum
}

fn main() {
    let mut futures = Vec::from_fn(1000, |ind| sync::Future::spawn( proc() { partial_sum(ind) }));

    let mut final_res = 0f64;
    for ft in futures.mut_iter()  {
        final_res += ft.get();
    }
    println!("π^2/6 is not far from : {}", final_res);
}
~~~

## Sharing immutable data without copy: Arc

To share immutable data between tasks, a first approach would be to only use pipes as we have seen
previously. A copy of the data to share would then be made for each task. In some cases, this would
add up to a significant amount of wasted memory and would require copying the same data more than
necessary.

To tackle this issue, one can use an Atomically Reference Counted wrapper (`Arc`) as implemented in
the `sync` library of Rust. With an Arc, the data will no longer be copied for each task. The Arc
acts as a reference to the shared data and only this reference is shared and cloned.

Here is a small example showing how to use Arcs. We wish to run concurrently several computations on
a single large vector of floats. Each task needs the full vector to perform its duty.

~~~
extern crate rand;
extern crate sync;

use sync::Arc;

fn pnorm(nums: &[f64], p: uint) -> f64 {
    nums.iter().fold(0.0, |a, b| a + b.powf(p as f64)).powf(1.0 / (p as f64))
}

fn main() {
    let numbers = Vec::from_fn(1000000, |_| rand::random::<f64>());
    let numbers_arc = Arc::new(numbers);

    for num in range(1u, 10) {
        let task_numbers = numbers_arc.clone();

        spawn(proc() {
            println!("{}-norm = {}", num, pnorm(task_numbers.as_slice(), num));
        });
    }
}
~~~

The function `pnorm` performs a simple computation on the vector (it computes the sum of its items
at the power given as argument and takes the inverse power of this value). The Arc on the vector is
created by the line

~~~
# extern crate sync;
# extern crate rand;
# use sync::Arc;
# fn main() {
# let numbers = Vec::from_fn(1000000, |_| rand::random::<f64>());
let numbers_arc=Arc::new(numbers);
# }
~~~

and a unique clone is captured for each task via a procedure. This only copies the wrapper and not
it's contents. Within the task's procedure, the captured Arc reference can be used as an immutable
reference to the underlying vector as if it were local.

~~~
# extern crate sync;
# extern crate rand;
# use sync::Arc;
# fn pnorm(nums: &[f64], p: uint) -> f64 { 4.0 }
# fn main() {
# let numbers=Vec::from_fn(1000000, |_| rand::random::<f64>());
# let numbers_arc = Arc::new(numbers);
# let num = 4;
let task_numbers = numbers_arc.clone();
spawn(proc() {
    // Capture task_numbers and use it as if it was the underlying vector
    println!("{}-norm = {}", num, pnorm(task_numbers.as_slice(), num));
});
# }
~~~

The `arc` module also implements Arcs around mutable data that are not covered here.

# Handling task failure

Rust has a built-in mechanism for raising exceptions. The `fail!()` macro
(which can also be written with an error string as an argument: `fail!(
~reason)`) and the `assert!` construct (which effectively calls `fail!()`
if a boolean expression is false) are both ways to raise exceptions. When a
task raises an exception the task unwinds its stack---running destructors and
freeing memory along the way---and then exits. Unlike exceptions in C++,
exceptions in Rust are unrecoverable within a single task: once a task fails,
there is no way to "catch" the exception.

While it isn't possible for a task to recover from failure, tasks may notify
each other of failure. The simplest way of handling task failure is with the
`try` function, which is similar to `spawn`, but immediately blocks waiting
for the child task to finish. `try` returns a value of type `Result<T,
()>`. `Result` is an `enum` type with two variants: `Ok` and `Err`. In this
case, because the type arguments to `Result` are `int` and `()`, callers can
pattern-match on a result to check whether it's an `Ok` result with an `int`
field (representing a successful result) or an `Err` result (representing
termination with an error).

~~~{.ignore .linked-failure}
# use std::task;
# fn some_condition() -> bool { false }
# fn calculate_result() -> int { 0 }
let result: Result<int, ()> = task::try(proc() {
    if some_condition() {
        calculate_result()
    } else {
        fail!("oops!");
    }
});
assert!(result.is_err());
~~~

Unlike `spawn`, the function spawned using `try` may return a value,
which `try` will dutifully propagate back to the caller in a [`Result`]
enum. If the child task terminates successfully, `try` will
return an `Ok` result; if the child task fails, `try` will return
an `Error` result.

[`Result`]: std/result/index.html

> *Note:* A failed task does not currently produce a useful error
> value (`try` always returns `Err(())`). In the
> future, it may be possible for tasks to intercept the value passed to
> `fail!()`.

TODO: Need discussion of `future_result` in order to make failure
modes useful.

But not all failures are created equal. In some cases you might need to
abort the entire program (perhaps you're writing an assert which, if
it trips, indicates an unrecoverable logic error); in other cases you
might want to contain the failure at a certain boundary (perhaps a
small piece of input from the outside world, which you happen to be
processing in parallel, is malformed and its processing task can't
proceed).

## Creating a task with a bi-directional communication path

A very common thing to do is to spawn a child task where the parent
and child both need to exchange messages with each other. The
function `sync::comm::duplex` supports this pattern.  We'll
look briefly at how to use it.

To see how `duplex` works, we will create a child task
that repeatedly receives a `uint` message, converts it to a string, and sends
the string in response.  The child terminates when it receives `0`.
Here is the function that implements the child task:

~~~
extern crate sync;
# fn main() {
fn stringifier(channel: &sync::DuplexStream<String, uint>) {
    let mut value: uint;
    loop {
        value = channel.recv();
        channel.send(value.to_str().to_string());
        if value == 0 { break; }
    }
}
# }
~~~

The implementation of `DuplexStream` supports both sending and
receiving. The `stringifier` function takes a `DuplexStream` that can
send strings (the first type parameter) and receive `uint` messages
(the second type parameter). The body itself simply loops, reading
from the channel and then sending its response back.  The actual
response itself is simply the stringified version of the received value,
`uint::to_str(value)`.

Here is the code for the parent task:

~~~
extern crate sync;
# use std::task::spawn;
# use sync::DuplexStream;
# fn stringifier(channel: &sync::DuplexStream<String, uint>) {
#     let mut value: uint;
#     loop {
#         value = channel.recv();
#         channel.send(value.to_str().to_string());
#         if value == 0u { break; }
#     }
# }
# fn main() {

let (from_child, to_child) = sync::duplex();

spawn(proc() {
    stringifier(&to_child);
});

from_child.send(22);
assert!(from_child.recv().as_slice() == "22");

from_child.send(23);
from_child.send(0);

assert!(from_child.recv().as_slice() == "23");
assert!(from_child.recv().as_slice() == "0");

# }
~~~

The parent task first calls `DuplexStream` to create a pair of bidirectional
endpoints. It then uses `task::spawn` to create the child task, which captures
one end of the communication channel.  As a result, both parent and child can
send and receive data to and from the other.
