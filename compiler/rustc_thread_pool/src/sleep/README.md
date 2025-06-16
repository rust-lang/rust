# Introduction: the sleep module

The code in this module governs when worker threads should go to
sleep. The system used in this code was introduced in [Rayon RFC #5].
There is also a [video walkthrough] available. Both of those may be
valuable resources to understanding the code, though naturally they
will also grow stale over time. The comments in this file are
extracted from the RFC and meant to be kept up to date.

[Rayon RFC #5]: https://github.com/rayon-rs/rfcs/pull/5
[video walkthrough]: https://youtu.be/HvmQsE5M4cY

# The `Sleep` struct

The `Sleep` struct is embedded into each registry. It performs several functions:

* It tracks when workers are awake or asleep.
* It decides how long a worker should look for work before it goes to sleep,
  via a callback that is invoked periodically from the worker's search loop.
* It is notified when latches are set, jobs are published, or other
  events occur, and it will go and wake the appropriate threads if
  they are sleeping.

# Thread states

There are three main thread states:

* An **active** thread is one that is actively executing a job.
* An **idle** thread is one that is searching for work to do. It will be
  trying to steal work or pop work from the global injector queue.
* A **sleeping** thread is one that is blocked on a condition variable,
  waiting to be awoken.

We sometimes refer to the final two states collectively as **inactive**.
Threads begin as idle but transition to idle and finally sleeping when
they're unable to find work to do.

## Sleepy threads

There is one other special state worth mentioning. During the idle state,
threads can get **sleepy**. A sleepy thread is still idle, in that it is still
searching for work, but it is *about* to go to sleep after it does one more
search (or some other number, potentially). When a thread enters the sleepy
state, it signals (via the **jobs event counter**, described below) that it is
about to go to sleep. If new work is published, this will lead to the counter
being adjusted. When the thread actually goes to sleep, it will (hopefully, but
not guaranteed) see that the counter has changed and elect not to sleep, but
instead to search again. See the section on the **jobs event counter** for more
details.

# The counters

One of the key structs in the sleep module is `AtomicCounters`, found in
`counters.rs`. It packs three counters into one atomically managed value:

* Two **thread counters**, which track the number of threads in a particular state.
* The **jobs event counter**, which is used to signal when new work is available.
  It (sort of) tracks the number of jobs posted, but not quite, and it can rollover.

## Thread counters

There are two thread counters, one that tracks **inactive** threads and one that
tracks **sleeping** threads. From this, one can deduce the number of threads
that are idle by subtracting sleeping threads from inactive threads. We track
the counters in this way because it permits simpler atomic operations. One can
increment the number of sleeping threads (and thus decrease the number of idle
threads) simply by doing one atomic increment, for example. Similarly, one can
decrease the number of sleeping threads (and increase the number of idle
threads) through one atomic decrement.

These counters are adjusted as follows:

* When a thread enters the idle state: increment the inactive thread counter.
* When a thread enters the sleeping state: increment the sleeping thread counter.
* When a thread awakens a sleeping thread: decrement the sleeping thread counter.
  * Subtle point: the thread that *awakens* the sleeping thread decrements the
    counter, not the thread that is *sleeping*. This is because there is a delay
    between signaling a thread to wake and the thread actually waking:
    decrementing the counter when awakening the thread means that other threads
    that may be posting work will see the up-to-date value that much faster.
* When a thread finds work, exiting the idle state: decrement the inactive
  thread counter.

## Jobs event counter

The final counter is the **jobs event counter**. The role of this counter is to
help sleepy threads detect when new work is posted in a lightweight fashion. In
its simplest form, we would simply have a counter that gets incremented each
time a new job is posted. This way, when a thread gets sleepy, it could read the
counter, and then compare to see if the value has changed before it actually
goes to sleep. But this [turns out to be too expensive] in practice, so we use a
somewhat more complex scheme.

[turns out to be too expensive]: https://github.com/rayon-rs/rayon/pull/746#issuecomment-624802747

The idea is that the counter toggles between two states, depending on whether
its value is even or odd (or, equivalently, on the value of its low bit):

* Even -- If the low bit is zero, then it means that there has been no new work
  since the last thread got sleepy.
* Odd -- If the low bit is one, then it means that new work was posted since
  the last thread got sleepy.

### New work is posted

When new work is posted, we check the value of the counter: if it is even,
then we increment it by one, so that it becomes odd.

### Worker thread gets sleepy

When a worker thread gets sleepy, it will read the value of the counter. If the
counter is odd, it will increment the counter so that it is even. Either way, it
remembers the final value of the counter. The final value will be used later,
when the thread is going to sleep. If at that time the counter has not changed,
then we can assume no new jobs have been posted (though note the remote
possibility of rollover, discussed in detail below).

# Protocol for a worker thread to post work

The full protocol for a thread to post work is as follows

* If the work is posted into the injection queue, then execute a seq-cst fence (see below).
* Load the counters, incrementing the JEC if it is even so that it is odd.
* Check if there are idle threads available to handle this new job. If not,
  and there are sleeping threads, then wake one or more threads.

# Protocol for a worker thread to fall asleep

The full protocol for a thread to fall asleep is as follows:

* After completing all its jobs, the worker goes idle and begins to
  search for work. As it searches, it counts "rounds". In each round,
  it searches all other work threads' queues, plus the 'injector queue' for
  work injected from the outside. If work is found in this search, the thread
  becomes active again and hence restarts this protocol from the top.
* After a certain number of rounds, the thread "gets sleepy" and executes `get_sleepy`
  above, remembering the `final_value` of the JEC. It does one more search for work.
* If no work is found, the thread atomically:
  * Checks the JEC to see that it has not changed from `final_value`.
    * If it has, then the thread goes back to searching for work. We reset to
      just before we got sleepy, so that we will do one more search
      before attempting to sleep again (rather than searching for many rounds).
  * Increments the number of sleeping threads by 1.
* The thread then executes a seq-cst fence operation (see below).
* The thread then does one final check for injected jobs (see below). If any
  are available, it returns to the 'pre-sleepy' state as if the JEC had changed.
* The thread waits to be signaled. Once signaled, it returns to the idle state.

# The jobs event counter and deadlock

As described in the section on the JEC, the main concern around going to sleep
is avoiding a race condition wherein:

* Thread A looks for work, finds none.
* Thread B posts work but sees no sleeping threads.
* Thread A goes to sleep.

The JEC protocol largely prevents this, but due to rollover, this prevention is
not complete. It is possible -- if unlikely -- that enough activity occurs for
Thread A to observe the same JEC value that it saw when getting sleepy. If the
new work being published came from *inside* the thread-pool, then this race
condition isn't too harmful. It means that we have fewer workers processing the
work then we should, but we won't deadlock. This seems like an acceptable risk
given that this is unlikely in practice.

However, if the work was posted as an *external* job, that is a problem. In that
case, it's possible that all of our workers could go to sleep, and the external
job would never get processed. To prevent that, the sleeping protocol includes
one final check to see if the injector queue is empty before fully falling
asleep. Note that this final check occurs **after** the number of sleeping
threads has been incremented. We are not concerned therefore with races against
injections that occur after that increment, only before.

Unfortunately, there is one rather subtle point concerning this final check:
we wish to avoid the possibility that:

* work is pushed into the injection queue by an outside thread X,
* the sleepy thread S sees the JEC but it has rolled over and is equal
* the sleepy thread S reads the injection queue but does not see the work posted by X.

This is possible because the C++ memory model typically offers guarantees of the
form "if you see the access A, then you must see those other accesses" -- but it
doesn't guarantee that you will see the access A (i.e., if you think of
processors with independent caches, you may be operating on very out of date
cache state).

## Using seq-cst fences to prevent deadlock

To overcome this problem, we have inserted two sequentially consistent fence
operations into the protocols above:

* One fence occurs after work is posted into the injection queue, but before the
  counters are read (including the number of sleeping threads).
  * Note that no fence is needed for work posted to internal queues, since it is ok
    to overlook work in that case.
* One fence occurs after the number of sleeping threads is incremented, but
  before the injection queue is read.

### Proof sketch

What follows is a "proof sketch" that the protocol is deadlock free. We model
two relevant bits of memory, the job injector queue J and the atomic counters C.

Consider the actions of the injecting thread:

* PushJob: Job is injected, which can be modeled as an atomic write to J with release semantics.
* PushFence: A sequentially consistent fence is executed.
* ReadSleepers: The counters C are read (they may also be incremented, but we just consider the read that comes first).

Meanwhile, the sleepy thread does the following:

* IncSleepers: The number of sleeping threads is incremented, which is atomic exchange to C.
* SleepFence: A sequentially consistent fence is executed.
* ReadJob: We look to see if the queue is empty, which is a read of J with acquire semantics.

Either PushFence or SleepFence must come first:

* If PushFence comes first, then PushJob must be visible to ReadJob.
* If SleepFence comes first, then IncSleepers is visible to ReadSleepers.

# Deadlock detection

This module tracks a number of variables in order to detect deadlocks due to user code blocking.
These variables are stored in the `SleepData` struct which itself is kept behind a mutex.
It contains the following fields:
- `worker_count` - The number of threads in the thread pool.
- `active_threads` - The number of threads in the thread pool which are running
                     and aren't blocked in user code or sleeping.
- `blocked_threads` - The number of threads which are blocked in user code.
                      This doesn't include threads blocked by Rayon.

User code can indicate blocking by calling `mark_blocked` before blocking and
calling `mark_unblocked` before unblocking a thread.
This will adjust `active_threads` and `blocked_threads` accordingly.

When we tickle the thread pool in `Sleep::tickle_cold`, we set `active_threads` to
`worker_count` - `blocked_threads` since we wake up all Rayon threads, but not thread blocked
by user code.

A deadlock is detected by checking if `active_threads` is 0 and `blocked_threads` is above 0.
If we ignored `blocked_threads` we would have a deadlock
immediately when creating the thread pool.
We would also deadlock once the thread pool ran out of work.
It is not possible for Rayon itself to deadlock.
Deadlocks can only be caused by user code blocking, so this condition doesn't miss any deadlocks.

We check for the deadlock condition when
threads fall asleep in `mark_unblocked` and in `Sleep::sleep`.
If there's a deadlock detected we call the user provided deadlock handler while we hold the
lock to `SleepData`. This means the deadlock handler cannot call `mark_blocked` and
`mark_unblocked`. The user is expected to handle the deadlock in some non-Rayon thread.
Once the deadlock handler returns, the thread which called the deadlock handler will go to sleep.
