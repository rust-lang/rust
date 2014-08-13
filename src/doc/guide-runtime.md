% A Guide to the Rust Runtime

Rust includes two runtime libraries in the standard distribution, which provide
a unified interface to primitives such as I/O, but the language itself does not
require a runtime. The compiler is capable of generating code that works in all
environments, even kernel environments. Neither does the Rust language need a
runtime to provide memory safety; the type system itself is sufficient to write
safe code, verified statically at compile time. The runtime merely uses the
safety features of the language to build a number of convenient and safe
high-level abstractions.

That being said, code without a runtime is often very limited in what it can do.
As a result, Rust's standard libraries supply a set of functionality that is
normally considered the Rust runtime.  This guide will discuss Rust's user-space
runtime, how to use it, and what it can do.

# What is the runtime?

The Rust runtime can be viewed as a collection of code which enables services
like I/O, task spawning, TLS, etc. It's essentially an ephemeral collection of
objects which enable programs to perform common tasks more easily. The actual
implementation of the runtime itself is mostly a sparse set of opt-in primitives
that are all self-contained and avoid leaking their abstractions into libraries.

The current runtime is the engine behind these features (not a comprehensive
list):

* I/O
* Task spawning
* Message passing
* Task synchronization
* Task-local storage
* Logging
* Local heaps (GC heaps)
* Task unwinding

## What is the runtime accomplishing?

The runtime is designed with a few goals in mind:

* Rust libraries should work in a number of environments without having to worry
  about the exact details of the environment itself. Two commonly referred to
  environments are the M:N and 1:1 environments. Since the Rust runtime was
  first designed, it has supported M:N threading, and it has since gained 1:1
  support as well.

* The runtime should not enforce separate "modes of compilation" in order to
  work in multiple circumstances. It is an explicit goal that you compile a Rust
  library once and use it forever (in all environments).

* The runtime should be fast. There should be no architectural design barrier
  which is preventing programs from running at optimal speeds. It is not a goal
  for the runtime to be written "as fast as can be" at every moment in time. For
  example, no claims will be made that the current implementation of the runtime
  is the fastest it will ever be. This goal is simply to prevent any
  architectural roadblock from hindering performance.

* The runtime should be nearly invisible. The design of the runtime should not
  encourage direct interaction with it, and using the runtime should be
  essentially transparent to libraries. This does not mean it should be
  impossible to query the runtime, but rather it should be unconventional.

# Architecture of the runtime

This section explains the current architecture of the Rust runtime. It has
evolved over the development of Rust through many iterations, and this is simply
the documentation of the current iteration.

## A local task

The core abstraction of the Rust runtime is the task. A task represents a
"thread" of execution of Rust code, but it does not necessarily correspond to an
OS thread. Most runtime services are accessed through the local task, allowing
for runtime policy decisions to be made on a per-task basis.

A consequence of this decision is to require all Rust code using the standard
library to have a local `Task` structure available to them. This `Task` is
stored in the OS's thread local storage (OS TLS) to allow for efficient access
to it.

It has also been decided that the presence or non-presence of a local `Task` is
essentially the *only* assumption that the runtime can make. Almost all runtime
services are routed through this local structure.

This requirement of a local task is a core assumption on behalf of *all* code
using the standard library, hence it is defined in the standard library itself.

## I/O

When dealing with I/O in general, there are a few flavors by which it can be
dealt with, and not all flavors are right for all situations. I/O is also a
tricky topic that is nearly impossible to get consistent across all
environments. As a result, a Rust task is not guaranteed to have access to I/O,
and it is not even guaranteed what the implementation of the I/O will be.

This conclusion implies that I/O *cannot* be defined in the standard library.
The standard library does, however, provide the interface to I/O that all Rust
tasks are able to consume.

This interface is implemented differently for various flavors of tasks, and is
designed with a focus around synchronous I/O calls. This architecture does not
fundamentally prevent other forms of I/O from being defined, but it is not done
at this time.

The I/O interface that the runtime must provide can be found in the
[std::rt::rtio](std/rt/rtio/trait.IoFactory.html) module. Note that this
interface is *unstable*, and likely always will be.

## Task Spawning

A frequent operation performed by tasks is to spawn a child task to perform some
work. This is the means by which parallelism is enabled in Rust. This decision
of how to spawn a task is not a general decision, and is hence a local decision
to the task (not defined in the standard library).

Task spawning is interpreted as "spawning a sibling" and is enabled through the
high level interface in `std::task`. The child task can be configured
accordingly, and runtime implementations must respect these options when
spawning a new task.

Another local task operation is dealing with the runnable state of the task
itself.  This frequently comes up when the question is "how do I block a task?"
or "how do I wake up a task?". These decisions are inherently local to the task
itself, yet again implying that they are not defined in the standard library.

## The `Runtime` trait and the `Task` structure

The full complement of runtime features is defined by the [`Runtime`
trait](std/rt/trait.Runtime.html) and the [`Task`
struct](std/rt/task/struct.Task.html). A `Task` is constant among all runtime
implementations, but each runtime has its own implementation of the
`Runtime` trait.

The local `Task` stores the runtime value inside of itself, and then ownership
dances ensue to invoke methods on the runtime.

# Implementations of the runtime

The Rust distribution provides two implementations of the runtime. These two
implementations are generally known as 1:1 threading and M:N threading.

As with many problems in computer science, there is no right answer in this
question of which implementation of the runtime to choose. Each implementation
has its benefits and each has its drawbacks. The descriptions below are meant to
inform programmers about what the implementation provides and what it doesn't
provide in order to make an informed decision about which to choose.

## 1:1 - using `libnative`

The library `libnative` is an implementation of the runtime built upon native OS
threads plus libc blocking I/O calls. This is called 1:1 threading because each
user-space thread corresponds to exactly one kernel thread.

In this model, each Rust task corresponds to one OS thread, and each I/O object
essentially corresponds to a file descriptor (or the equivalent of the platform
you're running on).

Some benefits to using libnative are:

* Guaranteed interop with FFI bindings. If a C library you are using blocks the
  thread to do I/O (such as a database driver), then this will not interfere
  with other Rust tasks (because only the OS thread will be blocked).
* Less I/O overhead as opposed to M:N in some cases. Not all M:N I/O is
  guaranteed to be "as fast as can be", and some things (like filesystem APIs)
  are not truly asynchronous on all platforms, meaning that the M:N
  implementation may incur more overhead than a 1:1 implementation.

## M:N - using `libgreen`

The library `libgreen` implements the runtime with "green threads" on top of the
asynchronous I/O framework [libuv][libuv]. The M in M:N threading is the number
of OS threads that a process has, and the N is the number of Rust tasks. In this
model, N Rust tasks are multiplexed among M OS threads, and context switching is
implemented in user-space.

The primary concern of an M:N runtime is that a Rust task cannot block itself in
a syscall. If this happens, then the entire OS thread is frozen and unavailable
for running more Rust tasks, making this a (M-1):N runtime (and you can see how
this can reach 0/deadlock). By using asynchronous I/O under the hood (all I/O
still looks synchronous in terms of code), OS threads are never blocked until
the appropriate time comes.

Upon reading `libgreen`, you may notice that there is no I/O implementation
inside of the library, but rather just the infrastructure for maintaining a set
of green schedulers which switch among Rust tasks. The actual I/O implementation
is found in `librustuv` which are the Rust bindings to libuv. This distinction
is made to allow for other I/O implementations not built on libuv (but none
exist at this time).

Some benefits of using libgreen are:

* Fast task spawning. When using M:N threading, spawning a new task can avoid
  executing a syscall entirely, which can lead to more efficient task spawning
  times.
* Fast task switching. Because context switching is implemented in user-space,
  all task contention operations (mutexes, channels, etc) never execute
  syscalls, leading to much faster implementations and runtimes. An efficient
  context switch also leads to higher throughput servers than 1:1 threading
  because tasks can be switched out much more efficiently.

### Pools of Schedulers

M:N threading is built upon the concept of a pool of M OS threads (which
libgreen refers to as schedulers), able to run N Rust tasks. This abstraction is
encompassed in libgreen's [`SchedPool`](green/struct.SchedPool.html) type. This type allows for
fine-grained control over the pool of schedulers which will be used to run Rust
tasks.

In addition the `SchedPool` type is the *only* way through which a new M:N task
can be spawned. Sibling tasks to Rust tasks themselves (created through
`std::task::spawn`) will be spawned into the same pool of schedulers that the
original task was home to. New tasks must previously have some form of handle
into the pool of schedulers in order to spawn a new task.

## Which to choose?

With two implementations of the runtime available, a choice obviously needs to
be made to see which will be used. The compiler itself will always by-default
link to one of these runtimes.

Having a default decision made in the compiler is done out of necessity and
convenience. The compiler's decision of runtime to link to is *not* an
endorsement of one over the other. As always, this decision can be overridden.

For example, this program will be linked to "the default runtime". The current
default runtime is to use libnative.

~~~{.rust}
fn main() {}
~~~

### Force booting with libgreen

In this example, the `main` function will be booted with I/O support powered by
libuv. This is done by linking to the `rustuv` crate and specifying the
`rustuv::event_loop` function as the event loop factory.

To create a pool of green tasks which have no I/O support, you may shed the
`rustuv` dependency and use the `green::basic::event_loop` function instead of
`rustuv::event_loop`. All tasks will have no I/O support, but they will still be
able to deschedule/reschedule (use channels, locks, etc).

~~~{.rust}
extern crate green;
extern crate rustuv;

#[start]
fn start(argc: int, argv: *const *const u8) -> int {
    green::start(argc, argv, rustuv::event_loop, main)
}

fn main() {}
~~~

### Force booting with libnative

This program's `main` function will always be booted with libnative, running
inside of an OS thread.

~~~{.rust}
extern crate native;

#[start]
fn start(argc: int, argv: *const *const u8) -> int {
    native::start(argc, argv, main)
}

fn main() {}
~~~

# Finding the runtime

The actual code for the runtime is spread out among a few locations:

* [std::rt][stdrt]
* [libnative][libnative]
* [libgreen][libgreen]
* [librustuv][librustuv]

[libuv]: https://github.com/joyent/libuv/
[stdrt]: std/rt/index.html
[libnative]: native/index.html
[libgreen]: green/index.html
[librustuv]: rustuv/index.html
