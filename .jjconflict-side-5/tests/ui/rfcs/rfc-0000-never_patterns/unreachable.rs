#![feature(exhaustive_patterns)]
#![feature(never_patterns)]
#![allow(incomplete_features)]
#![allow(dead_code, unreachable_code)]
#![deny(unreachable_patterns)]

#[derive(Copy, Clone)]
enum Void {}

fn main() {
    let res_void: Result<bool, Void> = Ok(true);

    match res_void {
        Ok(_x) => {}
        Err(!),
        //~^ ERROR unreachable
    }
    let (Ok(_x) | Err(!)) = res_void;
    //~^ ERROR unreachable
    if let Err(!) = res_void {}
    //~^ ERROR unreachable
    if let (Ok(true) | Err(!)) = res_void {}
    //~^ ERROR unreachable
    for (Ok(mut _x) | Err(!)) in [res_void] {}
    //~^ ERROR unreachable
}

fn foo((Ok(_x) | Err(!)): Result<bool, Void>) {}
//~^ ERROR unreachable
