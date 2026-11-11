//! Regression test for <https://github.com/rust-lang/rust/issues/25343>.
//! Ensure we're not wrongly producing shadowing label warning.
//! More cases added from issue <https://github.com/rust-lang/rust/issues/31754>.
//@ run-pass

#[allow(unused)]
fn main() {
    || {
        'label: loop {
        }
    };

    'label2: loop {
        break;
    }

    let closure = || {
        'label2: loop {}
    };

    fn inner_fn() {
        'label2: loop {}
    }
}
