// This issue reproduces an ICE on compile (E.g. fails on 2018-12-19 nightly).
// "cannot relate bound region: ReBound(DebruijnIndex(1), BrAnon(1)) <= '?1"
//@ run-pass
//@ edition:2018
#![feature(coroutines,coroutine_trait)]
use std::ops::Coroutine;

fn with<F>(f: F) -> impl Coroutine<Yield=(), Return=()>
where F: Fn() -> ()
{
    #[coroutine] move || {
        loop {
            match f() {
                _ => yield,
            }
        }
    }
}

fn main() {
    let data = &vec![1];
    #[coroutine] || { //~ WARN unused coroutine that must be used
        let _to_pin = with(move || println!("{:p}", data));
        loop {
            yield
        }
    };
}
