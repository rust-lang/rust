# `hint_core_should_pause`

The tracking issue for this feature is: [#41196]

[#41196]: https://github.com/rust-lang/rust/issues/41196

------------------------

Many programs have spin loops like the following:

```rust,no_run
use std::sync::atomic::{AtomicBool,Ordering};

fn spin_loop(value: &AtomicBool) {
    loop {
        if value.load(Ordering::Acquire) {
             break;
        }
    }
}
```

These programs can be improved in performance like so:

```rust,no_run
#![feature(hint_core_should_pause)]
use std::sync::atomic;
use std::sync::atomic::{AtomicBool,Ordering};

fn spin_loop(value: &AtomicBool) {
    loop {
        if value.load(Ordering::Acquire) {
             break;
        }
        atomic::hint_core_should_pause();
    }
}
```

Further improvements could combine `hint_core_should_pause` with
exponential backoff or `std::thread::yield_now`.
