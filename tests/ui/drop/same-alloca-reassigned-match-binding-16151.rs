//! Regression test for https://github.com/rust-lang/rust/issues/16151

//@ run-pass

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use std::mem;

static mut DROP_COUNT: usize = 0;

struct Fragment;

impl Drop for Fragment {
    fn drop(&mut self) {
        unsafe {
            DROP_COUNT += 1;
        }
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
    unsafe {
        assert_eq!(DROP_COUNT, 3);
    }
}
