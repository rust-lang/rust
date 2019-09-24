// This issue reproduces an ICE on compile (E.g. fails on 2018-12-19 nightly).
// "cannot relate bound region: ReLateBound(DebruijnIndex(1), BrAnon(1)) <= '_#1r"
// run-pass
// edition:2018
#![feature(generators,generator_trait)]
use std::ops::Generator;

fn with<F>(f: F) -> impl Generator<Yield=(), Return=()>
where F: Fn() -> ()
{
    move || {
        loop {
            match f() {
                _ => yield,
            }
        }
    }
}

fn main() {
    let data = &vec![1];
    || {
        let _to_pin = with(move || println!("{:p}", data));
        loop {
            yield
        }
    };
}
