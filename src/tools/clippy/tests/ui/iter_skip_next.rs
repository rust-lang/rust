//@aux-build:option_helpers.rs

#![warn(clippy::iter_skip_next)]
#![allow(clippy::disallowed_names)]
#![allow(clippy::iter_nth)]
#![allow(clippy::useless_vec)]
#![allow(clippy::iter_out_of_bounds)]
#![allow(unused_mut, dead_code)]

extern crate option_helpers;

use option_helpers::IteratorFalsePositives;

/// Checks implementation of `ITER_SKIP_NEXT` lint
fn main() {
    let some_vec = vec![0, 1, 2, 3];
    let _ = some_vec.iter().skip(42).next();
    //~^ iter_skip_next
    let _ = some_vec.iter().cycle().skip(42).next();
    //~^ iter_skip_next
    let _ = (1..10).skip(10).next();
    //~^ iter_skip_next
    let _ = &some_vec[..].iter().skip(3).next();
    //~^ iter_skip_next
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.skip(42).next();
    let _ = foo.filter().skip(42).next();

    // fix #8128
    let test_string = "1|1 2";
    let mut sp = test_string.split('|').map(|s| s.trim());
    let _: Vec<&str> = sp.skip(1).next().unwrap().split(' ').collect();
    //~^ iter_skip_next
    if let Some(mut s) = Some(test_string.split('|').map(|s| s.trim())) {
        let _: Vec<&str> = s.skip(1).next().unwrap().split(' ').collect();
        //~^ iter_skip_next
    };
    fn check<T>(mut s: T)
    where
        T: Iterator<Item = String>,
    {
        let _: Vec<&str> = s.skip(1).next().unwrap().split(' ').collect();
        //~^ iter_skip_next
    }
}
