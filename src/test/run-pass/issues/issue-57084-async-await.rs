// This issue reproduces an ICE on compile (E.g. fails on 2018-12-19 nightly).
// run-pass
// edition:2018
#![feature(async_await,futures_api,await_macro,generators)]

pub struct Foo;

impl Foo {
    async fn with<'a, F, R>(&'a self, f: F) -> R
    where F: Fn() -> R + 'a,
    {
        loop {
            match f() {
                _ => yield,
            }
        }
    }

    pub async fn run<'a>(&'a self, data: &'a [u8])
    {
        await!(self.with(move || {
            println!("{:p}", data);
        }))
    }
}

fn main() {}
