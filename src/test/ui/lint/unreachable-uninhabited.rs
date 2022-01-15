// Regression test for #89779.

#![forbid(unreachable_code)]
//~^ NOTE: the lint level is defined here

fn stop() -> Result<std::convert::Infallible, u64> {
    Err(5)
}

fn _foo1() {
    if false {
        stop().unwrap();
    }

    println!("Hello, world!");
    // no warnings here
}

fn _foo2() {
    if false {
    //~^ NOTE: any code following this expression is unreachable
    //~| NOTE: both branches of this `if` expression diverge
        stop().unwrap();
    } else {
        stop().unwrap();
    }

    println!("Hello, world!");
    //~^ ERROR: unreachable expression
}

fn _foo3() {
    match stop() {
    //~^ NOTE: any code following this expression is unreachable
    //~| NOTE: all arms of this `match` expression diverge
        Ok(x) => stop().unwrap(),
        Err(_) => stop().unwrap(),
    };

    println!("Hello, world!");
    //~^ ERROR: unreachable expression
}

fn bar() -> Result<std::convert::Infallible, u64> {
    Err(5)
}

fn main() {
    match bar().expect("should") {}
    // no warnings here
}
