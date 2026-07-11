//! Regression test for https://github.com/rust-lang/rust/issues/16151

//@ run-pass

use std::mem;

static mut DROP_COUNT: usize = 0;

fn increment_drop_count() {
    unsafe {
        let drop_count = &raw mut DROP_COUNT;
        drop_count.write(drop_count.read() + 1);
    }
}

fn drop_count() -> usize {
    unsafe { (&raw const DROP_COUNT).read() }
}

struct Fragment;

impl Drop for Fragment {
    fn drop(&mut self) {
        increment_drop_count();
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
    assert_eq!(drop_count(), 3);
}
