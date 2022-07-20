// run-rustfix

#![allow(unused)]

fn a(i: i32) -> i32 {}
//~^ ERROR mismatched types

fn b(opt_str: Option<String>) {
    let s: String = if let Some(s) = opt_str {
        //~^ ERROR mismatched types
    } else {
        String::new()
    };
}

fn c() -> Option<i32> {
    //~^ ERROR mismatched types
    let x = Some(1);
}

fn main() {}
