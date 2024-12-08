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

fn d(opt_str: Option<String>) {
    let s: String = if let Some(s) = opt_str {
        //~^ ERROR mismatched types
    } else {
        String::new()
    };
}

fn d2(opt_str: Option<String>) {
    let s = if let Some(s) = opt_str {
    } else {
        String::new()
        //~^ ERROR `if` and `else` have incompatible types
    };
}

fn e(opt_str: Option<String>) {
    let s: String = match opt_str {
        Some(s) => {}
        //~^ ERROR mismatched types
        None => String::new(),
    };
}

fn e2(opt_str: Option<String>) {
    let s = match opt_str {
        Some(s) => {}
        None => String::new(),
        //~^ ERROR `match` arms have incompatible types
    };
}

fn main() {}
