#![feature(nll)]

fn test<'a>() {
    let _:fn(&()) = |_:&'a ()| {};
}

fn main() {
    test();
}
