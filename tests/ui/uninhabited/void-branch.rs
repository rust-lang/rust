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

fn infallible_with_arg<T>(x: T) -> (T, std::convert::Infallible) {
    (x, loop {})
    //~^ ERROR unreachable expression
}

fn in_if_else(x: String) -> Result<String, (String, std::convert::Infallible)> {
    if x.len() > 0 { Err(infallible_with_arg(x)) } else { Ok(x) }
    //~^ ERROR unreachable expression
}

fn main() {}
