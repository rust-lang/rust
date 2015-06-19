% Concurrency and Paralellism



# Data Races and Race Conditions

Safe Rust guarantees an absence of data races, which are defined as:

* two or more threads concurrently accessing a location of memory
* one of them is a write
* one of them is unsynchronized

A data race has Undefined Behaviour, and is therefore impossible to perform
in Safe Rust. Data races are *mostly* prevented through rust's ownership system:
it's impossible to alias a mutable reference, so it's impossible to perform a
data race. Interior mutability makes this more complicated, which is largely why
we have the Send and Sync traits (see below).

However Rust *does not* prevent general race conditions. This is
pretty fundamentally impossible, and probably honestly undesirable. Your hardware
is racy, your OS is racy, the other programs on your computer are racy, and the
world this all runs in is racy. Any system that could genuinely claim to prevent
*all* race conditions would be pretty awful to use, if not just incorrect.

So it's perfectly "fine" for a Safe Rust program to get deadlocked or do
something incredibly stupid with incorrect synchronization. Obviously such a
program isn't very good, but Rust can only hold your hand so far. Still, a
race condition can't violate memory safety in a Rust program on
its own. Only in conjunction with some other unsafe code can a race condition
actually violate memory safety. For instance:

```rust
use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

let data = vec![1, 2, 3, 4];
// Arc so that the memory the AtomicUsize is stored in still exists for
// the other thread to increment, even if we completely finish executing
// before it. Rust won't compile the program without it, because of the
// lifetime requirements of thread::spawn!
let idx = Arc::new(AtomicUsize::new(0));
let other_idx = idx.clone();

// `move` captures other_idx by-value, moving it into this thread
thread::spawn(move || {
    // It's ok to mutate idx because this value
    // is an atomic, so it can't cause a Data Race.
    other_idx.fetch_add(10, Ordering::SeqCst);
});

// Index with the value loaded from the atomic. This is safe because we
// read the atomic memory only once, and then pass a *copy* of that value
// to the Vec's indexing implementation. This indexing will be correctly
// bounds checked, and there's no chance of the value getting changed
// in the middle. However our program may panic if the thread we spawned
// managed to increment before this ran. A race condition because correct
// program execution (panicing is rarely correct) depends on order of
// thread execution.
println!("{}", data[idx.load(Ordering::SeqCst)]);

if idx.load(Ordering::SeqCst) < data.len() {
    unsafe {
        // Incorrectly loading the idx *after* we did the bounds check.
        // It could have changed. This is a race condition, *and dangerous*
        // because we decided to do `get_unchecked`, which is `unsafe`.
        println!("{}", data.get_unchecked(idx.load(Ordering::SeqCst)));
    }
}
```




# Send and Sync

Not everything obeys inherited mutability, though. Some types allow you to multiply
alias a location in memory while mutating it. Unless these types use synchronization
to manage this access, they are absolutely not thread safe. Rust captures this with
through the `Send` and `Sync` traits.

* A type is Send if it is safe to send it to another thread.
* A type is Sync if it is safe to share between threads (`&T` is Send).

Send and Sync are *very* fundamental to Rust's concurrency story. As such, a
substantial amount of special tooling exists to make them work right. First and
foremost, they're *unsafe traits*. This means that they are unsafe *to implement*,
and other unsafe code can *trust* that they are correctly implemented. Since
they're *marker traits* (they have no associated items like methods), correctly
implemented simply means that they have the intrinsic properties an implementor
should have. Incorrectly implementing Send or Sync can cause Undefined Behaviour.

Send and Sync are also what Rust calls *opt-in builtin traits*.
This means that, unlike every other trait, they are *automatically* derived:
if a type is composed entirely of Send or Sync types, then it is Send or Sync.
Almost all primitives are Send and Sync, and as a consequence pretty much
all types you'll ever interact with are Send and Sync.

Major exceptions include:
* raw pointers are neither Send nor Sync (because they have no safety guards)
* `UnsafeCell` isn't Sync (and therefore `Cell` and `RefCell` aren't)
* `Rc` isn't Send or Sync (because the refcount is shared and unsynchronized)

`Rc` and `UnsafeCell` are very fundamentally not thread-safe: they enable
unsynchronized shared mutable state. However raw pointers are, strictly speaking,
marked as thread-unsafe as more of a *lint*. Doing anything useful
with a raw pointer requires dereferencing it, which is already unsafe. In that
sense, one could argue that it would be "fine" for them to be marked as thread safe.

However it's important that they aren't thread safe to prevent types that
*contain them* from being automatically marked as thread safe. These types have
non-trivial untracked ownership, and it's unlikely that their author was
necessarily thinking hard about thread safety. In the case of Rc, we have a nice
example of a type that contains a `*mut` that is *definitely* not thread safe.

Types that aren't automatically derived can *opt-in* to Send and Sync by simply
implementing them:

```rust
struct MyBox(*mut u8);

unsafe impl Send for MyBox {}
unsafe impl Sync for MyBox {}
```

In the *incredibly rare* case that a type is *inappropriately* automatically
derived to be Send or Sync, then one can also *unimplement* Send and Sync:

```rust
struct SpecialThreadToken(u8);

impl !Send for SpecialThreadToken {}
impl !Sync for SpecialThreadToken {}
```

Note that *in and of itself* it is impossible to incorrectly derive Send and Sync.
Only types that are ascribed special meaning by other unsafe code can possible cause
trouble by being incorrectly Send or Sync.

Most uses of raw pointers should be encapsulated behind a sufficient abstraction
that Send and Sync can be derived. For instance all of Rust's standard
collections are Send and Sync (when they contain Send and Sync types)
in spite of their pervasive use raw pointers to
manage allocations and complex ownership. Similarly, most iterators into these
collections are Send and Sync because they largely behave like an `&` or `&mut`
into the collection.

TODO: better explain what can or can't be Send or Sync. Sufficient to appeal
only to data races?




# Atomics

Rust pretty blatantly just inherits LLVM's model for atomics, which in turn is
largely based off of the C11 model for atomics. This is not due these models
being particularly excellent or easy to understand. Indeed, these models are
quite complex and are known to have several flaws. Rather, it is a pragmatic
concession to the fact that *everyone* is pretty bad at modeling atomics. At very
least, we can benefit from existing tooling and research around C's model.

Trying to fully explain these models is fairly hopeless, so we're just going to
drop that problem in LLVM's lap.




# Actually Doing Things Concurrently

Rust as a language doesn't *really* have an opinion on how to do concurrency or
parallelism. The standard library exposes OS threads and blocking sys-calls
because *everyone* has those and they're uniform enough that you can provide
an abstraction over them in a relatively uncontroversial way. Message passing,
green threads, and async APIs are all diverse enough that any abstraction over
them tends to involve trade-offs that we weren't willing to commit to for 1.0.

However Rust's current design is setup so that you can set up your own
concurrent paradigm or library as you see fit. Just require the right
lifetimes and Send and Sync where appropriate and everything should Just Work
with everyone else's stuff.




[llvm-conc]: http://llvm.org/docs/Atomics.html
[trpl-conc]: https://doc.rust-lang.org/book/concurrency.html
