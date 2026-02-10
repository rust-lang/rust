//! Test for <https://github.com/rust-lang/rust/issues/145739>: width and precision arguments
//! implicitly captured by `format_args!` should behave as if they were written individually.
//@ run-pass
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};

static DEREF_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct LogDeref;
impl Deref for LogDeref {
    type Target = usize;
    fn deref(&self) -> &usize {
        if DEREF_COUNTER.fetch_add(1, Ordering::Relaxed).is_multiple_of(2) {
            &2
        } else {
            &3
        }
    }
}

fn main() {
    assert_eq!(DEREF_COUNTER.load(Ordering::Relaxed), 0);

    let x = 0.0;

    let _ = format_args!("{x:LogDeref$} {x:LogDeref$}");
    // TODO: Increased by 2, as `&LogDeref` is coerced to a `&usize` twice.
    assert_eq!(DEREF_COUNTER.load(Ordering::Relaxed), 1);

    let _ = format_args!("{x:.LogDeref$} {x:.LogDeref$}");
    // TODO: Increased by 2, as `&LogDeref` is coerced to a `&usize` twice.
    assert_eq!(DEREF_COUNTER.load(Ordering::Relaxed), 2);

    let _ = format_args!("{x:LogDeref$} {x:.LogDeref$}");
    // TODO: Increased by 2, as `&LogDeref` is coerced to a `&usize` twice.
    assert_eq!(DEREF_COUNTER.load(Ordering::Relaxed), 3);
}
