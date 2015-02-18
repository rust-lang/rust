% Concurrency

Concurrency and parallelism are incredibly important topics in computer
science, and are also a hot topic in industry today. Computers are gaining more
and more cores, yet many programmers aren't prepared to fully utilize them.

Rust's memory safety features also apply to its concurrency story too. Even
concurrent Rust programs must be memory safe, having no data races. Rust's type
system is up to the task, and gives you powerful ways to reason about
concurrent code at compile time.

Before we talk about the concurrency features that come with Rust, it's important
to understand something: Rust is low-level enough that all of this is provided
by the standard library, not by the language. This means that if you don't like
some aspect of the way Rust handles concurrency, you can implement an alternative
way of doing things. [mio](https://github.com/carllerche/mio) is a real-world
example of this principle in action.

## Background: `Send` and `Sync`

Concurrency is difficult to reason about. In Rust, we have a strong, static
type system to help us reason about our code. As such, Rust gives us two traits
to help us make sense of code that can possibly be concurrent.

### `Send`

The first trait we're going to talk about is
[`Send`](../std/marker/trait.Send.html). When a type `T` implements `Send`, it indicates
to the compiler that something of this type is able to have ownership transferred
safely between threads.

This is important to enforce certain restrictions. For example, if we have a
channel connecting two threads, we would want to be able to send some data
down the channel and to the other thread. Therefore, we'd ensure that `Send` was
implemented for that type.

In the opposite way, if we were wrapping a library with FFI that isn't
threadsafe, we wouldn't want to implement `Send`, and so the compiler will help
us enforce that it can't leave the current thread.

### `Sync`

The second of these two trait is called [`Sync`](../std/marker/trait.Sync.html).
When a type `T` implements `Sync`, it indicates to the compiler that something
of this type has no possibility of introducing memory unsafety when used from
multiple threads concurrently.

For example, sharing immutable data with an atomic reference count is
threadsafe. Rust provides a type like this, `Arc<T>`, and it implements `Sync`,
so that it could be safely shared between threads.

These two traits allow you to use the type system to make strong guarantees
about the properties of your code under concurrency. Before we demonstrate
why, we need to learn how to create a concurrent Rust program in the first
place!

## Threads

Rust's standard library provides a library for 'threads', which allow you to
run Rust code in parallel. Here's a basic example of using `std::thread`:

```
use std::thread;

fn main() {
    thread::scoped(|| {
        println!("Hello from a thread!");
    });
}
```

The `Thread::scoped()` method accepts a closure, which is executed in a new
thread. It's called `scoped` because this thread returns a join guard:

```
use std::thread;

fn main() {
    let guard = thread::scoped(|| {
        println!("Hello from a thread!");
    });

    // guard goes out of scope here
}
```

When `guard` goes out of scope, it will block execution until the thread is
finished. If we didn't want this behaviour, we could use `thread::spawn()`:

```
use std::thread;
use std::old_io::timer;
use std::time::Duration;

fn main() {
    thread::spawn(|| {
        println!("Hello from a thread!");
    });

    timer::sleep(Duration::milliseconds(50));
}
```

We need to `sleep` here because when `main()` ends, it kills all of the
running threads.

[`scoped`](std/thread/struct.Builder.html#method.scoped) has an interesting
type signature:

```text
fn scoped<'a, T, F>(self, f: F) -> JoinGuard<'a, T>
    where T: Send + 'a,
          F: FnOnce() -> T,
          F: Send + 'a
```

Specifically, `F`, the closure that we pass to execute in the new thread. It
has two restrictions: It must be a `FnOnce` from `()` to `T`. Using `FnOnce`
allows the closure to take ownership of any data it mentions from the parent
thread. The other restriction is that `F` must be `Send`. We aren't allowed to
transfer this ownership unless the type thinks that's okay.

Many languages have the ability to execute threads, but it's wildly unsafe.
There are entire books about how to prevent errors that occur from shared
mutable state. Rust helps out with its type system here as well, by preventing
data races at compile time. Let's talk about how you actually share things
between threads.

## Safe Shared Mutable State

Due to Rust's type system, we have a concept that sounds like a lie: "safe
shared mutable state." Many programmers agree that shared mutable state is
very, very bad.

Someone once said this:

> Shared mutable state is the root of all evil. Most languages attempt to deal
> with this problem through the 'mutable' part, but Rust deals with it by
> solving the 'shared' part.

The same [ownership system](ownership.html) that helps prevent using pointers
incorrectly also helps rule out data races, one of the worst kinds of
concurrency bugs.

As an example, here is a Rust program that would have a data race in many
languages. It will not compile:

```ignore
use std::thread;
use std::old_io::timer;
use std::time::Duration;

fn main() {
    let mut data = vec![1u32, 2, 3];

    for i in 0..2 {
        thread::spawn(move || {
            data[i] += 1;
        });
    }

    timer::sleep(Duration::milliseconds(50));
}
```

This gives us an error:

```text
12:17 error: capture of moved value: `data`
        data[i] += 1;
        ^~~~
```

In this case, we know that our code _should_ be safe, but Rust isn't sure. And
it's actually not safe: if we had a reference to `data` in each thread, and the
thread takes ownership of the reference, we have three owners! That's bad. We
can fix this by using the `Arc<T>` type, which is an atomic reference counted
pointer. The 'atomic' part means that it's safe to share across threads.

`Arc<T>` assumes one more property about its contents to ensure that it is safe
to share across threads: it assumes its contents are `Sync`. But in our
case, we want to be able to mutate the value. We need a type that can ensure
only one person at a time can mutate what's inside. For that, we can use the
`Mutex<T>` type. Here's the second version of our code. It still doesn't work,
but for a different reason:

```ignore
use std::thread;
use std::old_io::timer;
use std::time::Duration;
use std::sync::Mutex;

fn main() {
    let mut data = Mutex::new(vec![1u32, 2, 3]);

    for i in 0..2 {
        let data = data.lock().unwrap();
        thread::spawn(move || {
            data[i] += 1;
        });
    }

    timer::sleep(Duration::milliseconds(50));
}
```

Here's the error:

```text
<anon>:11:9: 11:22 error: the trait `core::marker::Send` is not implemented for the type `std::sync::mutex::MutexGuard<'_, collections::vec::Vec<u32>>` [E0277]
<anon>:11         Thread::spawn(move || {
                  ^~~~~~~~~~~~~
<anon>:11:9: 11:22 note: `std::sync::mutex::MutexGuard<'_, collections::vec::Vec<u32>>` cannot be sent between threads safely
<anon>:11         Thread::spawn(move || {
                  ^~~~~~~~~~~~~
```

You see, [`Mutex`](std/sync/struct.Mutex.html) has a
[`lock`](http://doc.rust-lang.org/nightly/std/sync/struct.Mutex.html#method.lock)
method which has this signature:

```ignore
fn lock(&self) -> LockResult<MutexGuard<T>>
```

If we [look at the code for MutexGuard](https://github.com/rust-lang/rust/blob/ca4b9674c26c1de07a2042cb68e6a062d7184cef/src/libstd/sync/mutex.rs#L172), we'll see
this:

```ignore
__marker: marker::NoSend,
```

Because our guard is `NoSend`, it's not `Send`. Which means we can't actually
transfer the guard across thread boundaries, which gives us our error.

We can use `Arc<T>` to fix this. Here's the working version:

```
use std::sync::{Arc, Mutex};
use std::thread;
use std::old_io::timer;
use std::time::Duration;

fn main() {
    let data = Arc::new(Mutex::new(vec![1u32, 2, 3]));

    for i in 0..2 {
        let data = data.clone();
        thread::spawn(move || {
            let mut data = data.lock().unwrap();
            data[i] += 1;
        });
    }

    timer::sleep(Duration::milliseconds(50));
}
```

We now call `clone()` on our `Arc`, which increases the internal count. This
handle is then moved into the new thread. Let's examine the body of the
thread more closely:

```
# use std::sync::{Arc, Mutex};
# use std::thread;
# use std::old_io::timer;
# use std::time::Duration;
# fn main() {
#     let data = Arc::new(Mutex::new(vec![1u32, 2, 3]));
#     for i in 0..2 {
#         let data = data.clone();
thread::spawn(move || {
    let mut data = data.lock().unwrap();
    data[i] += 1;
});
#     }
# }
```

First, we call `lock()`, which acquires the mutex's lock. Because this may fail,
it returns an `Result<T, E>`, and because this is just an example, we `unwrap()`
it to get a reference to the data. Real code would have more robust error handling
here. We're then free to mutate it, since we have the lock.

This timer bit is a bit awkward, however. We have picked a reasonable amount of
time to wait, but it's entirely possible that we've picked too high, and that
we could be taking less time. It's also possible that we've picked too low,
and that we aren't actually finishing this computation.

Rust's standard library provides a few more mechanisms for two threads to
synchronize with each other. Let's talk about one: channels.

## Channels

Here's a version of our code that uses channels for synchronization, rather
than waiting for a specific time:

```
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc;

fn main() {
    let data = Arc::new(Mutex::new(0u32));

    let (tx, rx) = mpsc::channel();

    for _ in 0..10 {
        let (data, tx) = (data.clone(), tx.clone());

        thread::spawn(move || {
            let mut data = data.lock().unwrap();
            *data += 1;

            tx.send(());
        });
    }

    for _ in 0..10 {
        rx.recv();
    }
}
```

We use the `mpsc::channel()` method to construct a new channel. We just `send`
a simple `()` down the channel, and then wait for ten of them to come back.

While this channel is just sending a generic signal, we can send any data that
is `Send` over the channel!

```
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc;

fn main() {
    let (tx, rx) = mpsc::channel();

    for _ in 0..10 {
        let tx = tx.clone();

        thread::spawn(move || {
            let answer = 42u32;

            tx.send(answer);
        });
    }

   rx.recv().ok().expect("Could not recieve answer");
}
```

A `u32` is `Send` because we can make a copy. So we create a thread, ask it to calculate
the answer, and then it `send()`s us the answer over the channel.


## Panics

A `panic!` will crash the currently executing thread. You can use Rust's
threads as a simple isolation mechanism:

```
use std::thread;

let result = thread::spawn(move || {
    panic!("oops!");
}).join();

assert!(result.is_err());
```

Our `Thread` gives us a `Result` back, which allows us to check if the thread
has panicked or not.
