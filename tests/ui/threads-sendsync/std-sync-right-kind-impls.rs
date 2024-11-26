//@ run-pass

use std::sync;

fn assert_both<T: Sync + Send>() {}

fn main() {
    assert_both::<sync::Mutex<()>>();
    assert_both::<sync::Condvar>();
    assert_both::<sync::RwLock<()>>();
    assert_both::<sync::Barrier>();
    assert_both::<sync::Arc<()>>();
    assert_both::<sync::Weak<()>>();
    assert_both::<sync::Once>();
}
