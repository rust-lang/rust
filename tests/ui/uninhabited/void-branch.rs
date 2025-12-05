#![deny(unreachable_code)]
#![allow(deprecated, invalid_value)]

enum Void {}

fn with_void() {
    if false {
        unsafe {
            std::mem::uninitialized::<Void>();
            println!();
            //~^ ERROR unreachable expression
        }
    }

    println!();
}

fn infallible() -> std::convert::Infallible {
    loop {}
}

fn with_infallible() {
    if false {
        infallible();
        println!()
        //~^ ERROR unreachable expression
    }

    println!()
}

fn main() {}
