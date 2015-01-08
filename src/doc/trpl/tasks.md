% The Rust Threads and Communication Guide

**NOTE** This guide is badly out of date and needs to be rewritten.

# Introduction

Rust provides safe concurrent abstractions through a number of core library
primitives. This guide will describe the concurrency model in Rust, how it
relates to the Rust type system, and introduce the fundamental library
abstractions for constructing concurrent programs.

Threads provide failure isolation and recovery. When a fatal error occurs in Rust
code as a result of an explicit call to `panic!()`, an assertion failure, or
another invalid operation, the runtime system destroys the entire thread. Unlike
in languages such as Java and C++, there is no way to `catch` an exception.
Instead, threads may monitor each other to see if they panic.

Threads use Rust's type system to provide strong memory safety guarantees.  In
particular, the type system guarantees that threads cannot induce a data race
from shared mutable state.

# Basics

At its simplest, creating a thread is a matter of calling the `spawn` function
with a closure argument. `spawn` executes the closure in the new thread.

```{rust,ignore}
# use std::thread::spawn;

// Print something profound in a different thread using a named function
fn print_message() { println!("I am running in a different thread!"); }
spawn(print_message);

// Alternatively, use a `move ||` expression instead of a named function.
// `||` expressions evaluate to an unnamed closure. The `move` keyword
// indicates that the closure should take ownership of any variables it
// touches.
spawn(move || println!("I am also running in a different thread!"));
```

In Rust, a thread is not a concept that appears in the language semantics.
Instead, Rust's type system provides all the tools necessary to implement safe
concurrency: particularly, ownership. The language leaves the implementation
details to the standard library.

The `spawn` function has the type signature: `fn
spawn<F:FnOnce()+Send>(f: F)`.  This indicates that it takes as
argument a closure (of type `F`) that it will run exactly once. This
closure is limited to capturing `Send`-able data from its environment
(that is, data which is deeply owned). Limiting the closure to `Send`
ensures that `spawn` can safely move the entire closure and all its
associated state into an entirely different thread for execution.

```{rust,ignore}
# use std::thread::spawn;
# fn generate_thread_number() -> int { 0 }
// Generate some state locally
let child_thread_number = generate_thread_number();

spawn(move || {
    // Capture it in the remote thread. The `move` keyword indicates
    // that this closure should move `child_thread_number` into its
    // environment, rather than capturing a reference into the
    // enclosing stack frame.
    println!("I am child number {}", child_thread_number);
});
```

## Communication

Now that we have spawned a new thread, it would be nice if we could communicate
with it. For this, we use *channels*. A channel is simply a pair of endpoints:
one for sending messages and another for receiving messages.

The simplest way to create a channel is to use the `channel` function to create a
`(Sender, Receiver)` pair. In Rust parlance, a **sender** is a sending endpoint
of a channel, and a **receiver** is the receiving endpoint. Consider the following
example of calculating two results concurrently:

```{rust,ignore}
# use std::thread::spawn;

let (tx, rx): (Sender<int>, Receiver<int>) = channel();

spawn(move || {
    let result = some_expensive_computation();
    tx.send(result);
});

some_other_expensive_computation();
let result = rx.recv();
# fn some_expensive_computation() -> int { 42 }
# fn some_other_expensive_computation() {}
```

Let's examine this example in detail. First, the `let` statement creates a
stream for sending and receiving integers (the left-hand side of the `let`,
`(tx, rx)`, is an example of a destructuring let: the pattern separates a tuple
into its component parts).

```{rust,ignore}
let (tx, rx): (Sender<int>, Receiver<int>) = channel();
```

The child thread will use the sender to send data to the parent thread, which will
wait to receive the data on the receiver. The next statement spawns the child
thread.

```{rust,ignore}
# use std::thread::spawn;
# fn some_expensive_computation() -> int { 42 }
# let (tx, rx) = channel();
spawn(move || {
    let result = some_expensive_computation();
    tx.send(result);
});
```

Notice that the creation of the thread closure transfers `tx` to the child thread
implicitly: the closure captures `tx` in its environment. Both `Sender` and
`Receiver` are sendable types and may be captured into threads or otherwise
transferred between them. In the example, the child thread runs an expensive
computation, then sends the result over the captured channel.

Finally, the parent continues with some other expensive computation, then waits
for the child's result to arrive on the receiver:

```{rust,ignore}
# fn some_other_expensive_computation() {}
# let (tx, rx) = channel::<int>();
# tx.send(0);
some_other_expensive_computation();
let result = rx.recv();
```

The `Sender` and `Receiver` pair created by `channel` enables efficient
communication between a single sender and a single receiver, but multiple
senders cannot use a single `Sender` value, and multiple receivers cannot use a
single `Receiver` value.  What if our example needed to compute multiple
results across a number of threads? The following program is ill-typed:

```{rust,ignore}
# fn some_expensive_computation() -> int { 42 }
let (tx, rx) = channel();

spawn(move || {
    tx.send(some_expensive_computation());
});

// ERROR! The previous spawn statement already owns the sender,
// so the compiler will not allow it to be captured again
spawn(move || {
    tx.send(some_expensive_computation());
});
```

Instead we can clone the `tx`, which allows for multiple senders.

```{rust,ignore}
let (tx, rx) = channel();

for init_val in range(0u, 3) {
    // Create a new channel handle to distribute to the child thread
    let child_tx = tx.clone();
    spawn(move || {
        child_tx.send(some_expensive_computation(init_val));
    });
}

let result = rx.recv() + rx.recv() + rx.recv();
# fn some_expensive_computation(_i: uint) -> int { 42 }
```

Cloning a `Sender` produces a new handle to the same channel, allowing multiple
threads to send data to a single receiver. It upgrades the channel internally in
order to allow this functionality, which means that channels that are not
cloned can avoid the overhead required to handle multiple senders. But this
fact has no bearing on the channel's usage: the upgrade is transparent.

Note that the above cloning example is somewhat contrived since you could also
simply use three `Sender` pairs, but it serves to illustrate the point. For
reference, written with multiple streams, it might look like the example below.

```{rust,ignore}
# use std::thread::spawn;

// Create a vector of ports, one for each child thread
let rxs = Vec::from_fn(3, |init_val| {
    let (tx, rx) = channel();
    spawn(move || {
        tx.send(some_expensive_computation(init_val));
    });
    rx
});

// Wait on each port, accumulating the results
let result = rxs.iter().fold(0, |accum, rx| accum + rx.recv() );
# fn some_expensive_computation(_i: uint) -> int { 42 }
```

## Backgrounding computations: Futures

With `sync::Future`, rust has a mechanism for requesting a computation and
getting the result later.

The basic example below illustrates this.

```{rust,ignore}
# #![allow(deprecated)]
use std::sync::Future;

# fn main() {
# fn make_a_sandwich() {};
fn fib(n: u64) -> u64 {
    // lengthy computation returning an uint
    12586269025
}

let mut delayed_fib = Future::spawn(move || fib(50));
make_a_sandwich();
println!("fib(50) = {}", delayed_fib.get())
# }
```

The call to `future::spawn` immediately returns a `future` object regardless of
how long it takes to run `fib(50)`. You can then make yourself a sandwich while
the computation of `fib` is running. The result of the execution of the method
is obtained by calling `get` on the future. This call will block until the
value is available (*i.e.* the computation is complete). Note that the future
needs to be mutable so that it can save the result for next time `get` is
called.

Here is another example showing how futures allow you to background
computations. The workload will be distributed on the available cores.

```{rust,ignore}
# #![allow(deprecated)]
# use std::num::Float;
# use std::sync::Future;
fn partial_sum(start: uint) -> f64 {
    let mut local_sum = 0f64;
    for num in range(start*100000, (start+1)*100000) {
        local_sum += (num as f64 + 1.0).powf(-2.0);
    }
    local_sum
}

fn main() {
    let mut futures = Vec::from_fn(200, |ind| Future::spawn(move || partial_sum(ind)));

    let mut final_res = 0f64;
    for ft in futures.iter_mut()  {
        final_res += ft.get();
    }
    println!("π^2/6 is not far from : {}", final_res);
}
```

## Sharing without copying: Arc

To share data between threads, a first approach would be to only use channel as
we have seen previously. A copy of the data to share would then be made for
each thread. In some cases, this would add up to a significant amount of wasted
memory and would require copying the same data more than necessary.

To tackle this issue, one can use an Atomically Reference Counted wrapper
(`Arc`) as implemented in the `sync` library of Rust. With an Arc, the data
will no longer be copied for each thread. The Arc acts as a reference to the
shared data and only this reference is shared and cloned.

Here is a small example showing how to use Arcs. We wish to run concurrently
several computations on a single large vector of floats. Each thread needs the
full vector to perform its duty.

```{rust,ignore}
use std::num::Float;
use std::rand;
use std::sync::Arc;

fn pnorm(nums: &[f64], p: uint) -> f64 {
    nums.iter().fold(0.0, |a, b| a + b.powf(p as f64)).powf(1.0 / (p as f64))
}

fn main() {
    let numbers = Vec::from_fn(1000000, |_| rand::random::<f64>());
    let numbers_arc = Arc::new(numbers);

    for num in range(1u, 10) {
        let thread_numbers = numbers_arc.clone();

        spawn(move || {
            println!("{}-norm = {}", num, pnorm(thread_numbers.as_slice(), num));
        });
    }
}
```

The function `pnorm` performs a simple computation on the vector (it computes
the sum of its items at the power given as argument and takes the inverse power
of this value). The Arc on the vector is created by the line:

```{rust,ignore}
# use std::rand;
# use std::sync::Arc;
# fn main() {
# let numbers = Vec::from_fn(1000000, |_| rand::random::<f64>());
let numbers_arc = Arc::new(numbers);
# }
```

and a clone is captured for each thread via a procedure. This only copies
the wrapper and not its contents. Within the thread's procedure, the captured
Arc reference can be used as a shared reference to the underlying vector as
if it were local.

```{rust,ignore}
# use std::rand;
# use std::sync::Arc;
# fn pnorm(nums: &[f64], p: uint) -> f64 { 4.0 }
# fn main() {
# let numbers=Vec::from_fn(1000000, |_| rand::random::<f64>());
# let numbers_arc = Arc::new(numbers);
# let num = 4;
let thread_numbers = numbers_arc.clone();
spawn(move || {
    // Capture thread_numbers and use it as if it was the underlying vector
    println!("{}-norm = {}", num, pnorm(thread_numbers.as_slice(), num));
});
# }
```

# Handling thread panics

Rust has a built-in mechanism for raising exceptions. The `panic!()` macro
(which can also be written with an error string as an argument: `panic!(
~reason)`) and the `assert!` construct (which effectively calls `panic!()` if a
boolean expression is false) are both ways to raise exceptions. When a thread
raises an exception, the thread unwinds its stack—running destructors and
freeing memory along the way—and then exits. Unlike exceptions in C++,
exceptions in Rust are unrecoverable within a single thread: once a thread panics,
there is no way to "catch" the exception.

While it isn't possible for a thread to recover from panicking, threads may notify
each other if they panic. The simplest way of handling a panic is with the
`try` function, which is similar to `spawn`, but immediately blocks and waits
for the child thread to finish. `try` returns a value of type
`Result<T, Box<Any + Send>>`. `Result` is an `enum` type with two variants:
`Ok` and `Err`. In this case, because the type arguments to `Result` are `int`
and `()`, callers can pattern-match on a result to check whether it's an `Ok`
result with an `int` field (representing a successful result) or an `Err` result
(representing termination with an error).

```{rust,ignore}
# use std::thread::Thread;
# fn some_condition() -> bool { false }
# fn calculate_result() -> int { 0 }
let result: Result<int, Box<std::any::Any + Send>> = Thread::spawn(move || {
    if some_condition() {
        calculate_result()
    } else {
        panic!("oops!");
    }
}).join();
assert!(result.is_err());
```

Unlike `spawn`, the function spawned using `try` may return a value, which
`try` will dutifully propagate back to the caller in a [`Result`] enum. If the
child thread terminates successfully, `try` will return an `Ok` result; if the
child thread panics, `try` will return an `Error` result.

[`Result`]: ../std/result/index.html

> *Note:* A panicked thread does not currently produce a useful error
> value (`try` always returns `Err(())`). In the
> future, it may be possible for threads to intercept the value passed to
> `panic!()`.

But not all panics are created equal. In some cases you might need to abort
the entire program (perhaps you're writing an assert which, if it trips,
indicates an unrecoverable logic error); in other cases you might want to
contain the panic at a certain boundary (perhaps a small piece of input from
the outside world, which you happen to be processing in parallel, is malformed
such that the processing thread cannot proceed).
