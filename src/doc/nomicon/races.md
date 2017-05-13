% Data Races and Race Conditions

Safe Rust guarantees an absence of data races, which are defined as:

* two or more threads concurrently accessing a location of memory
* one of them is a write
* one of them is unsynchronized

A data race has Undefined Behavior, and is therefore impossible to perform
in Safe Rust. Data races are *mostly* prevented through rust's ownership system:
it's impossible to alias a mutable reference, so it's impossible to perform a
data race. Interior mutability makes this more complicated, which is largely why
we have the Send and Sync traits (see below).

**However Rust does not prevent general race conditions.**

This is pretty fundamentally impossible, and probably honestly undesirable. Your
hardware is racy, your OS is racy, the other programs on your computer are racy,
and the world this all runs in is racy. Any system that could genuinely claim to
prevent *all* race conditions would be pretty awful to use, if not just
incorrect.

So it's perfectly "fine" for a Safe Rust program to get deadlocked or do
something nonsensical with incorrect synchronization. Obviously such a program
isn't very good, but Rust can only hold your hand so far. Still, a race
condition can't violate memory safety in a Rust program on its own. Only in
conjunction with some other unsafe code can a race condition actually violate
memory safety. For instance:

```rust,no_run
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
// read the atomic memory only once, and then pass a copy of that value
// to the Vec's indexing implementation. This indexing will be correctly
// bounds checked, and there's no chance of the value getting changed
// in the middle. However our program may panic if the thread we spawned
// managed to increment before this ran. A race condition because correct
// program execution (panicking is rarely correct) depends on order of
// thread execution.
println!("{}", data[idx.load(Ordering::SeqCst)]);
```

```rust,no_run
use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

let data = vec![1, 2, 3, 4];

let idx = Arc::new(AtomicUsize::new(0));
let other_idx = idx.clone();

// `move` captures other_idx by-value, moving it into this thread
thread::spawn(move || {
    // It's ok to mutate idx because this value
    // is an atomic, so it can't cause a Data Race.
    other_idx.fetch_add(10, Ordering::SeqCst);
});

if idx.load(Ordering::SeqCst) < data.len() {
    unsafe {
        // Incorrectly loading the idx after we did the bounds check.
        // It could have changed. This is a race condition, *and dangerous*
        // because we decided to do `get_unchecked`, which is `unsafe`.
        println!("{}", data.get_unchecked(idx.load(Ordering::SeqCst)));
    }
}
```
