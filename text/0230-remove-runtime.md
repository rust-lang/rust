- Start Date: 2014-09-16
- RFC PR: https://github.com/rust-lang/rfcs/pull/230
- Rust Issue: https://github.com/rust-lang/rust/issues/17325

# Summary

This RFC proposes to remove the *runtime system* that is currently part of the
standard library, which currently allows the standard library to support both
native and green threading. In particular:

* The `libgreen` crate and associated support will be moved out of tree, into a
  separate Cargo package.

* The `librustrt` (the runtime) crate will be removed entirely.

* The `std::io` implementation will be directly welded to native threads and
  system calls.

* The `std::io` module will remain completely cross-platform, though *separate*
  platform-specific modules may be added at a later time.

# Motivation

## Background: thread/task models and I/O

Many languages/libraries offer some notion of "task" as a unit of concurrent
execution, possibly distinct from native OS threads. The characteristics of
tasks vary along several important dimensions:

* *1:1 vs M:N*. The most fundamental question is whether a "task" always
  corresponds to an OS-level thread (the 1:1 model), or whether there is some
  userspace scheduler that maps tasks onto worker threads (the M:N model).  Some
  kernels -- notably, Windows -- support a 1:1 model where the scheduling is
  performed in userspace, which combines some of the advantages of the two
  models.

  In the M:N model, there are various choices about whether and when blocked
  tasks can migrate between worker threads. One basic downside of the model,
  however, is that if a task takes a page fault, the entire worker thread is
  essentially blocked until the fault is serviced. Choosing the optimal number
  of worker threads is difficult, and some frameworks attempt to do so
  dynamically, which has costs of its own.

* *Stack management*. In the 1:1 model, tasks are threads and therefore must be
  equipped with their own stacks. In M:N models, tasks may or may not need their
  own stack, but there are important tradeoffs:

  * Techniques like *segmented stacks* allow stack size to grow over time,
    meaning that tasks can be equipped with their own stack but still be
    lightweight. Unfortunately, segmented stacks come with
    [a significant performance and complexity cost](https://mail.mozilla.org/pipermail/rust-dev/2013-November/006314.html).

  * On the other hand, if tasks are not equipped with their own stack, they
    either cannot be migrated between underlying worker threads (the case for
    frameworks like Java's
    [fork/join](http://gee.cs.oswego.edu/dl/papers/fj.pdf)), or else must be
    implemented using *continuation-passing style (CPS)*, where each blocking
    operation takes a closure representing the work left to do. (CPS essentially
    moves the needed parts of the stack into the continuation closure.) The
    upside is that such tasks can be extremely lightweight -- essentially just
    the size of a closure.

* *Blocking and I/O support*. In the 1:1 model, a task can block freely without
  any risk for other tasks, since each task is an OS thread. In the M:N model,
  however, blocking in the OS sense means blocking the worker thread. (The same
  applies to long-running loops or page faults.)

  M:N models can deal with blocking in a couple of ways. The approach taken in
  Java's [fork/join](http://gee.cs.oswego.edu/dl/papers/fj.pdf) framework, for
  example, is to dynamically spin up/down worker threads. Alternatively, special
  task-aware blocking operations (including I/O) can be provided, which are
  mapped under the hood to nonblocking operations, allowing the worker thread to
  continue. Unfortunately, this latter approach helps only with explicit
  blocking; it does nothing for loops, page faults and the like.

### Where Rust is now

Rust has gradually migrated from a "green" threading model toward a native
threading model:

* In Rust's green threading, tasks are scheduled M:N and are equipped with their
  own stack. Initially, Rust used segmented stacks to allow growth over time,
  but that
  [was removed](https://mail.mozilla.org/pipermail/rust-dev/2013-November/006314.html)
  in favor of pre-allocated stacks, which means Rust's green threads are not
  "lightweight". The treatment of blocking is described below.

* In Rust's native threading model, tasks are 1:1 with OS threads.

Initially, Rust supported only the green threading model. Later, native
threading was added and ultimately became the default.

In today's Rust, there is a single I/O API -- `std::io` -- that provides
blocking operations only and works with both threading models.
Rust is somewhat unusual in allowing programs to mix native and green threading,
and furthermore allowing *some* degree of interoperation between the two. This
feat is achieved through the runtime system -- `librustrt` -- which exposes:

* The `Runtime` trait, which abstracts over the scheduler (via methods like
  `deschedule` and `spawn_sibling`) as well as the entire I/O API (via
  `local_io`).

* The `rtio` module, which provides a number of traits that define the standard I/O
  abstraction.

* The `Task` struct, which includes a `Runtime` trait object as the dynamic entry point
  into the runtime.

In this setup, `libstd` works directly against the runtime interface. When
invoking an I/O or scheduling operation, it first finds the current `Task`, and
then extracts the `Runtime` trait object to actually perform the operation.

On native tasks, blocking operations simply block. On green tasks, blocking
operations are routed through the green scheduler and/or underlying event loop
and nonblocking I/O.

The actual scheduler and I/O implementations -- `libgreen` and `libnative` --
then live as crates "above" `libstd`.

## The problems

While the situation described above may sound good in principle, there are
several problems in practice.

**Forced co-evolution.** With today's design, the green and native
  threading models must provide the same I/O API at all times. But
  there is functionality that is only appropriate or efficient in one
  of the threading models.

  For example, the lightest-weight M:N task models are essentially just
  collections of closures, and do not provide any special I/O support. This
  style of lightweight tasks is used in Servo, but also shows up in
  [java.util.concurrent's exectors](http://docs.oracle.com/javase/7/docs/api/java/util/concurrent/Executors.html)
  and [Haskell's par monad](https://hackage.haskell.org/package/monad-par),
  among many others. These lighter weight models do not fit into the current
  runtime system.

  On the other hand, green threading systems designed explicitly to support I/O
  may also want to provide low-level access to the underlying event loop -- an
  API surface that doesn't make sense for the native threading model.

  Under the native model we want to provide direct non-blocking and/or
  asynchronous I/O support -- as a systems language, Rust should be able to work
  directly with what the OS provides without imposing global abstraction
  costs. These APIs may involve some platform-specific abstractions (`epoll`,
  `kqueue`, IOCP) for maximal performance. But integrating them cleanly with a
  green threading model may be difficult or impossible -- and at the very least,
  makes it difficult to add them quickly and seamlessly to the current I/O
  system.

  In short, the current design couples threading and I/O models together, and
  thus forces the green and native models to supply a common I/O interface --
  despite the fact that they are pulling in different directions.

**Overhead.** The current Rust model allows runtime mixtures of the green and
  native models. The implementation achieves this flexibility by using trait
  objects to model the entire I/O API. Unfortunately, this flexibility has
  several downsides:

- *Binary sizes*. A significant overhead caused by the trait object design is that
  the entire I/O system is included in any binary that statically links to
  `libstd`. See
  [this comment](https://github.com/rust-lang/rust/issues/10740#issuecomment-31475987)
  for more details.

- *Task-local storage*. The current implementation of task-local storage is
  designed to work seamlessly across native and green threads, and its performs
  substantially suffers as a result. While it is feasible to provide a more
  efficient form of "hybrid" TLS that works across models, doing so is *far*
  more difficult than simply using native thread-local storage.

- *Allocation and dynamic dispatch*. With the current design, any invocation of
  I/O involves at least dynamic dispatch, and in many cases allocation, due to
  the use of trait objects. However, in most cases these costs are trivial when
  compared to the cost of actually doing the I/O (or even simply making a
  syscall), so they are not strong arguments against the current design.

**Problematic I/O interactions.** As the
  [documentation for libgreen](http://doc.rust-lang.org/green/#considerations-when-using-libgreen)
  explains, only some I/O and synchronization methods work seamlessly across
  native and green tasks. For example, any invocation of native code that calls
  blocking I/O has the potential to block the worker thread running the green
  scheduler. In particular, `std::io` objects created on a native task cannot
  safely be used within a green task. Thus, even though `std::io` presents a
  unified I/O API for green and native tasks, it is not fully interoperable.

**Embedding Rust.** When embedding Rust code into other contexts -- whether
  calling from C code or embedding in high-level languages -- there is a fair
  amount of setup needed to provide the "runtime" infrastructure that `libstd`
  relies on. If `libstd` was instead bound to the native threading and I/O
  system, the embedding setup would be much simpler.

**Maintenance burden.** Finally, `libstd` is made somewhat more complex by
  providing such a flexible threading model. As this RFC will explain, moving to
  a strictly native threading model will allow substantial simplification and
  reorganization of the structure of Rust's libraries.

# Detailed design

To mitigate the above problems, this RFC proposes to tie `std::io` directly to
the native threading model, while moving `libgreen` and its supporting
infrastructure into an external Cargo package with its own I/O API.

## The near-term plan
### `std::io` and native threading

The plan is to entirely remove `librustrt`, including all of the traits.
The abstraction layers will then become:

- Highest level: `libstd`, providing cross-platform, high-level I/O and
  scheduling abstractions.  The crate will depend on `libnative` (the opposite
  of today's situation).

- Mid-level: `libnative`, providing a cross-platform Rust interface for I/O and
  scheduling. The API will be relatively low-level, compared to `libstd`. The
  crate will depend on `libsys`.

- Low-level: `libsys` (renamed from `liblibc`), providing platform-specific Rust
  bindings to system C APIs.

In this scheme, the actual API of `libstd` will not change significantly. But
its implementation will invoke functions in `libnative` directly, rather than
going through a trait object.

A goal of this work is to minimize the complexity of embedding Rust code in
other contexts. It is not yet clear what the final embedding API will look like.

### Green threading

Despite tying `libstd` to native threading, however, `libgreen` will still be
supported -- at least initially. The infrastructure in `libgreen` and friends will
move into its own Cargo package.

Initially, the green threading package will support essentially the same
interface it does today; there are no immediate plans to change its API, since
the focus will be on first improving the native threading API. Note, however,
that the I/O API will be exposed separately within `libgreen`, as opposed to the
current exposure through `std::io`.

## The long-term plan

Ultimately, a large motivation for the proposed refactoring is to allow the APIs
for native I/O to grow.

In particular, over time we should expose more of the underlying system
capabilities under the native threading model. Whenever possible, these
capabilities should be provided at the `libstd` level -- the highest level of
cross-platform abstraction. However, an important goal is also to provide
nonblocking and/or asynchronous I/O, for which system APIs differ greatly. It
may be necessary to provide additional, platform-specific crates to expose this
functionality. Ideally, these crates would interoperate smoothly with `libstd`,
so that for example a `libposix` crate would allow using an `poll` operation
directly against a `std::io::fs::File` value, for example.

We also wish to expose "lowering" operations in `libstd` -- APIs that allow
you to get at the file descriptor underlying a `std::io::fs::File`, for example.

On the other hand, we very much want to explore and support truly lightweight
M:N task models (that do not require per-task stacks) -- supporting efficient
data parallelism with work stealing for CPU-bound computations. These
lightweight models will not provide any special support for I/O. But they may
benefit from a notion of "task-local storage" and interfacing with the task
scheduler when explicitly synchronizing between tasks (via channels, for
example).

All of the above long-term plans will require substantial new design and
implementation work, and the specifics are out of scope for this RFC. The main
point, though, is that the refactoring proposed by this RFC will make it much
more plausible to carry out such work.

Finally, a guiding principle for the above work is *uncompromising support* for
native system APIs, in terms of both functionality and performance. For example,
it must be possible to use thread-local storage without significant overhead,
which is very much not the case today. Any abstractions to support M:N threading
models -- including the now-external `libgreen` package -- must respect this
constraint.

# Drawbacks

The main drawback of this proposal is that green I/O will be provided by a
forked interface of `std::io`. This change makes green threading
"second class", and means there's more to learn when using both models
together.

This setup also somewhat increases the risk of invoking native blocking I/O on a
green thread -- though of course that risk is very much present today. One way
of mitigating this risk in general is the Java executor approach, where the
native "worker" threads that are executing the green thread scheduler are
monitored for blocking, and new worker threads are spun up as needed.

# Unresolved questions

There are may unresolved questions about the exact details of the refactoring,
but these are considered implementation details since the `libstd` interface
itself will not substantially change as part of this RFC.
