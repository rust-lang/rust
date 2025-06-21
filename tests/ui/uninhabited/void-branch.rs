#![deny(unreachable_code)]
#![allow(deprecated, invalid_value)]

enum Void {}

fn with_void() {
    if false {
        unsafe {
            //~^ ERROR unreachable expression
            std::mem::uninitialized::<Void>();
        }
    }

    println!();
}

fn infallible() -> std::convert::Infallible {
    loop {}
}

fn with_infallible() {
    if false {
        //~^ ERROR unreachable expression
        infallible();
    }

    println!()
}

fn main() {}
