// run-rustfix
// edition:2018
// check-pass
#![warn(rust_2021_compatibility)]

macro_rules! m {
    (@ $body:expr) => {{
        let f = || $body;
        //~^ WARNING: drop order
        f();
    }};
    ($body:block) => {{
        m!(@ $body);
    }};
}

fn main() {
    let a = (1.to_string(), 2.to_string());
    m!({
        //~^ HELP: add a dummy
        let x = a.0;
        println!("{}", x);
    });
}
