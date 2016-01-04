% Concurrency

The Rust project was initiated to solve two thorny problems:

* How do you do safe systems programming?
* How do you make concurrency painless?

Initially these problems seemed orthogonal, but to our amazement, the
solution turned out to be identical: **the same tools that make Rust
safe also help you tackle concurrency head-on**.

Memory safety bugs and concurrency bugs often come down to code
accessing data when it shouldn't. Rust's secret weapon is *ownership*,
a discipline for access control that systems programmers try to
follow, but that Rust's compiler checks statically for you.

For memory safety, this means you can program without a garbage
collector *and* without fear of segfaults, because Rust will catch
your mistakes.

For concurrency, this means you can choose from a wide variety of
paradigms (message passing, shared state, lock-free, purely
functional), and Rust will help you avoid common pitfalls.

Here's a taste of concurrency in Rust:

* A [channel][mpsc] transfers ownership of the messages sent along it,
  so you can send a pointer from one thread to another without fear of
  the threads later racing for access through that pointer. **Rust's
  channels enforce thread isolation.**

* A [lock][mutex] knows what data it protects, and Rust guarantees
  that the data can only be accessed when the lock is held. State is
  never accidentally shared. **"Lock data, not code" is enforced in
  Rust.**

* Every data type knows whether it can safely be [sent][send] between
  or [accessed][sync] by multiple threads, and Rust enforces this safe
  usage; there are no data races, even for lock-free data structures.
  **Thread safety isn't just documentation; it's law.**

All of these benefits come out of Rust's ownership model, and in fact
locks, channels, lock-free data structures and so on are defined in
libraries, not the core language. That means that Rust's approach to
concurrency is *open ended*: new libraries can embrace new paradigms
and catch new bugs, just by adding APIs that use Rust's ownership
features.

The goal of this post is to give you some idea of how that's done.

## Message passing

Now that we've covered the basic ownership story in Rust, let's see
what it means for concurrency.

Concurrent programming comes in many styles, but a particularly simple
one is message passing, where threads or actors communicate by sending
each other messages.  Proponents of the style emphasize the way that
it ties together sharing and communication:

> Do not communicate by sharing memory; instead, share memory by
> communicating.
>
> --[Effective Go](http://golang.org/doc/effective_go.html)

**Rust's ownership makes it easy to turn that advice into a
compiler-checked rule**. Consider the following channel API
([channels in Rust's standard library][mpsc] are a bit different):

~~~~rust,ignore
fn send<T: Send>(chan: &Channel<T>, t: T);
fn recv<T: Send>(chan: &Channel<T>) -> T;
~~~~

Channels are generic over the type of data they transmit (the `<T:
Send>` part of the API). The `Send` part means that `T` must be
considered safe to send between threads; we'll come back to that later
in the post, but for now it's enough to know that `Vec<i32>` is
`Send`.

As always in Rust, passing in a `T` to the `send` function means
transferring ownership of it. This fact has profound consequences: it
means that code like the following will generate a compiler error.

~~~~rust,ignore
// Suppose chan: Channel<Vec<i32>>

let mut vec = Vec::new();
// do some computation
send(&chan, vec);
print_vec(&vec);
~~~~

Here, the thread creates a vector, sends it to another thread, and
then continues using it. The thread receiving the vector could mutate
it as this thread continues running, so the call to `print_vec` could
lead to race condition or, for that matter, a use-after-free bug.

Instead, the Rust compiler will produce an error message on the call
to `print_vec`:

~~~~text
Error: use of moved value `vec`
~~~~

Disaster averted.

### Locks

Another way to deal with concurrency is by having threads communicate
through passive, shared state.

Shared-state concurrency has a bad rap. It's easy to forget to acquire
a lock, or otherwise mutate the wrong data at the wrong time, with
disastrous results -- so easy that many eschew the style altogether.

Rust's take is that:

1. Shared-state concurrency is nevertheless a fundamental programming
style, needed for systems code, for maximal performance, and for
implementing other styles of concurrency.

2. The problem is really about *accidentally* shared state.

Rust aims to give you the tools to conquer shared-state concurrency
directly, whether you're using locking or lock-free techniques.

In Rust, threads are "isolated" from each other automatically, due to
ownership. Writes can only happen when the thread has mutable access,
either by owning the data, or by having a mutable borrow of it. Either
way, **the thread is guaranteed to be the only one with access at the
time**.  To see how this plays out, let's look at locks.

Remember that mutable borrows cannot occur simultaneously with other
borrows. Locks provide the same guarantee ("mutual exclusion") through
synchronization at runtime. That leads to a locking API that hooks
directly into Rust's ownership system.

Here is a simplified version (the [standard library's][mutex]
is more ergonomic):

~~~~rust,ignore
// create a new mutex
fn mutex<T: Send>(t: T) -> Mutex<T>;

// acquire the lock
fn lock<T: Send>(mutex: &Mutex<T>) -> MutexGuard<T>;

// access the data protected by the lock
fn access<T: Send>(guard: &mut MutexGuard<T>) -> &mut T;
~~~~

This lock API is unusual in several respects.

First, the `Mutex` type is generic over a type `T` of **the data
protected by the lock**. When you create a `Mutex`, you transfer
ownership of that data *into* the mutex, immediately giving up access
to it. (Locks are unlocked when they are first created.)

Later, you can `lock` to block the thread until the lock is
acquired. This function, too, is unusual in providing a return value,
`MutexGuard<T>`. The `MutexGuard` automatically releases the lock when
it is destroyed; there is no separate `unlock` function.

The only way to access the lock is through the `access` function,
which turns a mutable borrow of the guard into a mutable borrow of the
data (with a shorter lease):

~~~~rust,ignore
fn use_lock(mutex: &Mutex<Vec<i32>>) {
    // acquire the lock, taking ownership of a guard;
    // the lock is held for the rest of the scope
    let mut guard = lock(mutex);

    // access the data by mutably borrowing the guard
    let vec = access(&mut guard);

    // vec has type `&mut Vec<i32>`
    vec.push(3);

    // lock automatically released here, when `guard` is destroyed
}
~~~~

There are two key ingredients here:

* The mutable reference returned by `access` cannot outlive the
  `MutexGuard` it is borrowing from.

* The lock is only released when the `MutexGuard` is destroyed.

The result is that **Rust enforces locking discipline: it will not let
you access lock-protected data except when holding the lock**. Any
attempt to do otherwise will generate a compiler error. For example,
consider the following buggy "refactoring":

~~~~rust,ignore
fn use_lock(mutex: &Mutex<Vec<i32>>) {
    let vec = {
        // acquire the lock
        let mut guard = lock(mutex);

        // attempt to return a borrow of the data
        access(&mut guard)

        // guard is destroyed here, releasing the lock
    };

    // attempt to access the data outside of the lock.
    vec.push(3);
}
~~~~

Rust will generate an error pinpointing the problem:

~~~~text
error: `guard` does not live long enough
access(&mut guard)
            ^~~~~
~~~~

Disaster averted.

### Thread safety and "Send"

It's typical to distinguish some data types as "thread safe" and
others not. Thread safe data structures use enough internal
synchronization to be safely used by multiple threads concurrently.

For example, Rust ships with two kinds of "smart pointers" for
reference counting:

* `Rc<T>` provides reference counting via normal reads/writes. It is
  not thread safe.

* `Arc<T>` provides reference counting via *atomic* operations. It is
  thread safe.

The hardware atomic operations used by `Arc` are more expensive than
the vanilla operations used by `Rc`, so it's advantageous to use `Rc`
rather than `Arc`. On the other hand, it's critical that an `Rc<T>`
never migrate from one thread to another, because that could lead to
race conditions that corrupt the count.

Usually, the only recourse is careful documentation; most languages
make no *semantic* distinction between thread-safe and thread-unsafe
types.

In Rust, the world is divided into two kinds of data types: those that
are [`Send`][send], meaning they can be safely moved from one thread to
another, and those that are `!Send`, meaning that it may not be safe
to do so. If all of a type's components are `Send`, so is that type --
which covers most types. Certain base types are not inherently
thread-safe, though, so it's also possible to explicitly mark a type
like `Arc` as `Send`, saying to the compiler: "Trust me; I've verified
the necessary synchronization here."

Naturally, `Arc` is `Send`, and `Rc` is not.

We already saw that the `Channel` and `Mutex` APIs work only with
`Send` data. Since they are the point at which data crosses thread
boundaries, they are also the point of enforcement for `Send`.

Putting this all together, Rust programmers can reap the benefits of
`Rc` and other thread-*unsafe* types with confidence, knowing that if
they ever do accidentally try to send one to another thread, the Rust
compiler will say:

~~~~text
`Rc<Vec<i32>>` cannot be sent between threads safely
~~~~

Disaster averted.

### Data races

At this point, we've seen enough to venture a strong statement about
Rust's approach to concurrency: **the compiler prevents all *data races*.**

> A data race is any unsynchronized, concurrent access to data
> involving a write.

Synchronization here includes things as low-level as atomic
instructions. Essentially, this is a way of saying that you cannot
accidentally "share state" between threads; all (mutating) access to
state has to be mediated by *some* form of synchronization.

Data races are just one (very important) kind of race condition, but
by preventing them, Rust often helps you prevent other, more subtle
races as well. For example, it's often important that updates to
different locations appear to take place *atomically*: other threads
see either all of the updates, or none of them. In Rust, having `&mut`
access to the relevant locations at the same time **guarantees
atomicity of updates to them**, since no other thread could possibly
have concurrent read access.

It's worth pausing for a moment to think about this guarantee in the
broader landscape of languages. Many languages provide memory safety
through garbage collection. But garbage collection doesn't give you
any help in preventing data races.

Rust instead uses ownership and borrowing to provide its two key value
propositions:

* Memory safety without garbage collection.
* Concurrency without data races.

### The future

When Rust first began, it baked channels directly into the language,
taking a very opinionated stance on concurrency.

In today's Rust, concurrency is *entirely* a library affair;
everything described in this post, including `Send`, is defined in the
standard library, and could be defined in an external library instead.

And that's very exciting, because it means that Rust's concurrency
story can endlessly evolve, growing to encompass new paradigms and
catch new classes of bugs.

[mpsc]: ../std/sync/mpsc/index.html
[mutex]: ../std/sync/struct.Mutex.html
[send]: ../std/marker/trait.Send.html
[sync]: ../std/marker/trait.Sync.html
