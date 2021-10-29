// run-rustfix

#![feature(custom_inner_attributes)]
#![warn(clippy::manual_split_once)]
#![allow(clippy::iter_skip_next, clippy::iter_nth_zero, clippy::needless_splitn)]

extern crate itertools;

#[allow(unused_imports)]
use itertools::Itertools;

fn main() {
    let _ = "key=value".splitn(2, '=').next();
    let _ = "key=value".splitn(2, '=').nth(2);
    let _ = "key=value".splitn(2, '=').next().unwrap();
    let _ = "key=value".splitn(2, '=').nth(0).unwrap();
    let _ = "key=value".splitn(2, '=').nth(1).unwrap();
    let _ = "key=value".splitn(2, '=').skip(1).next().unwrap();
    let (_, _) = "key=value".splitn(2, '=').next_tuple().unwrap();

    let s = String::from("key=value");
    let _ = s.splitn(2, '=').next().unwrap();

    let s = Box::<str>::from("key=value");
    let _ = s.splitn(2, '=').nth(0).unwrap();

    let s = &"key=value";
    let _ = s.splitn(2, '=').skip(0).next().unwrap();

    fn _f(s: &str) -> Option<&str> {
        let _ = s.splitn(2, "key=value").next()?;
        let _ = s.splitn(2, "key=value").nth(1)?;
        let _ = s.splitn(2, "key=value").skip(1).next()?;
        None
    }

    // Don't lint, slices don't have `split_once`
    let _ = [0, 1, 2].splitn(2, |&x| x == 1).nth(1).unwrap();

    // `rsplitn` gives the results in the reverse order of `rsplit_once`
    let _ = "key=value".rsplitn(2, '=').next().unwrap();
    let _ = "key=value".rsplitn(2, '=').nth(1).unwrap();
    let _ = "key=value".rsplitn(2, '=').nth(0);
    let (_, _) = "key=value".rsplitn(2, '=').next_tuple().unwrap();
}

fn _msrv_1_51() {
    #![clippy::msrv = "1.51"]
    // `str::split_once` was stabilized in 1.16. Do not lint this
    let _ = "key=value".splitn(2, '=').nth(1).unwrap();
}

fn _msrv_1_52() {
    #![clippy::msrv = "1.52"]
    let _ = "key=value".splitn(2, '=').nth(1).unwrap();
}
