//@edition:2018

#![warn(clippy::needless_splitn)]
#![allow(clippy::iter_skip_next, clippy::iter_nth_zero, clippy::manual_split_once)]

extern crate itertools;

#[allow(unused_imports)]
use itertools::Itertools;

fn main() {
    let str = "key=value=end";
    let _ = str.splitn(2, '=').next();
    //~^ needless_splitn
    let _ = str.splitn(2, '=').nth(0);
    //~^ needless_splitn
    let _ = str.splitn(2, '=').nth(1);
    let (_, _) = str.splitn(2, '=').next_tuple().unwrap();
    let (_, _) = str.splitn(3, '=').next_tuple().unwrap();
    //~^ needless_splitn
    let _: Vec<&str> = str.splitn(3, '=').collect();

    let _ = str.rsplitn(2, '=').next();
    //~^ needless_splitn
    let _ = str.rsplitn(2, '=').nth(0);
    //~^ needless_splitn
    let _ = str.rsplitn(2, '=').nth(1);
    let (_, _) = str.rsplitn(2, '=').next_tuple().unwrap();
    let (_, _) = str.rsplitn(3, '=').next_tuple().unwrap();
    //~^ needless_splitn

    let _ = str.splitn(5, '=').next();
    //~^ needless_splitn
    let _ = str.splitn(5, '=').nth(3);
    //~^ needless_splitn
    let _ = str.splitn(5, '=').nth(4);
    let _ = str.splitn(5, '=').nth(5);
}

fn _question_mark(s: &str) -> Option<()> {
    let _ = s.splitn(2, '=').next()?;
    //~^ needless_splitn
    let _ = s.splitn(2, '=').nth(0)?;
    //~^ needless_splitn
    let _ = s.rsplitn(2, '=').next()?;
    //~^ needless_splitn
    let _ = s.rsplitn(2, '=').nth(0)?;
    //~^ needless_splitn

    Some(())
}

#[clippy::msrv = "1.51"]
fn _test_msrv() {
    // `manual_split_once` MSRV shouldn't apply to `needless_splitn`
    let _ = "key=value".splitn(2, '=').nth(0).unwrap();
    //~^ needless_splitn
}
