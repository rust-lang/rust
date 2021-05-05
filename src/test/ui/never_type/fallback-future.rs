// check-pass
// edition:2018

use std::future::Future;

fn foo() {
    let ticker = loopify(async move { loop {} });

    match ticker {
        Ok(v) => v,
        Err(()) => return,
    };
}

fn loopify<F>(_: F) -> Result<F::Output, ()>
where
    F: Future,
{
    loop {}
}

fn main() {}
