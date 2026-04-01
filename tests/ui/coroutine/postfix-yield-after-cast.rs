// Regression test for <https://github.com/rust-lang/rust/issues/144527>.

#![feature(yield_expr, coroutines)]

fn main() {
    #[coroutine] || {
        0 as u8.yield
        //~^ ERROR cast cannot be followed by `.yield`
    };
}
