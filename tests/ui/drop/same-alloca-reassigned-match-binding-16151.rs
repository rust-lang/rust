//! Regression test for https://github.com/rust-lang/rust/issues/16151

//@ run-pass

use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};

static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

struct Fragment;

impl Drop for Fragment {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    {
        let mut fragments = vec![Fragment, Fragment, Fragment];
        let _new_fragments: Vec<Fragment> = mem::replace(&mut fragments, vec![])
            .into_iter()
            .skip_while(|_fragment| {
                true
            }).collect();
    }
    assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 3);
}
