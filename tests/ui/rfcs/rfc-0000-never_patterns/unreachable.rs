// revisions: normal exh_pats
//[normal] check-pass
#![feature(never_patterns)]
#![allow(incomplete_features)]
#![cfg_attr(exh_pats, feature(min_exhaustive_patterns))]
#![allow(dead_code, unreachable_code)]
#![deny(unreachable_patterns)]

#[derive(Copy, Clone)]
enum Void {}

fn main() {
    let res_void: Result<bool, Void> = Ok(true);

    match res_void {
        Ok(_x) => {}
        Err(!),
        //[exh_pats]~^ ERROR unreachable
    }
    let (Ok(_x) | Err(!)) = res_void;
    //[exh_pats]~^ ERROR unreachable
    if let Err(!) = res_void {}
    //[exh_pats]~^ ERROR unreachable
    if let (Ok(true) | Err(!)) = res_void {}
    //[exh_pats]~^ ERROR unreachable
    for (Ok(mut _x) | Err(!)) in [res_void] {}
    //[exh_pats]~^ ERROR unreachable
}

fn foo((Ok(_x) | Err(!)): Result<bool, Void>) {}
//[exh_pats]~^ ERROR unreachable
