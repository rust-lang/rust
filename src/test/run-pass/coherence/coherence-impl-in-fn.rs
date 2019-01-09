// run-pass
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![allow(dead_code)]
#![allow(non_camel_case_types)]

pub fn main() {
    #[derive(Copy, Clone)]
    enum x { foo }
    impl ::std::cmp::PartialEq for x {
        fn eq(&self, other: &x) -> bool {
            (*self) as isize == (*other) as isize
        }
        fn ne(&self, other: &x) -> bool { !(*self).eq(other) }
    }
}
