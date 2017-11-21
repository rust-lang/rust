# `spin_loop_hint`

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
#![feature(spin_loop_hint)]
use std::sync::atomic;
use std::sync::atomic::{AtomicBool,Ordering};

fn spin_loop(value: &AtomicBool) {
    loop {
        if value.load(Ordering::Acquire) {
             break;
        }
        atomic::spin_loop_hint();
    }
}
```

Further improvements could combine `spin_loop_hint` with
exponential backoff or `std::thread::yield_now`.
