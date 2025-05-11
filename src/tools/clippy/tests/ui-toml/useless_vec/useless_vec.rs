//@compile-flags: --test
#![warn(clippy::useless_vec)]
#![allow(clippy::unnecessary_operation, clippy::no_effect)]

fn foo(_: &[u32]) {}

fn main() {
    foo(&vec![1_u32]);
    //~^ useless_vec
}

#[test]
pub fn in_test() {
    foo(&vec![2_u32]);
}

#[cfg(test)]
fn in_cfg_test() {
    foo(&vec![3_u32]);
}

#[cfg(test)]
mod mod1 {
    fn in_cfg_test_mod() {
        super::foo(&vec![4_u32]);
    }
}
