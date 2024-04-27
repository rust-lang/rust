#![feature(if_let_guard)]
#![allow(irrefutable_let_patterns)]

fn match_option(x: Option<u32>) {
    match x {
        //~^ ERROR non-exhaustive patterns: `None` not covered
        Some(_) => {}
        None if let y = x => {}
    }
}

fn main() {
    let x = ();
    match x {
        //~^ ERROR non-exhaustive patterns: `()` not covered
        y if let z = y => {}
    }
}
