//@ edition:2021
// issue: rust-lang/rust#127555

pub trait Foo {
    fn bar<F>(&mut self, func: F) -> impl std::future::Future<Output = ()> + Send
    where
        F: FnMut();
}

struct Baz {}

impl Foo for Baz {
    async fn bar<F>(&mut self, _func: F) -> ()
    where
        F: FnMut() + Send,
        //~^ ERROR impl has stricter requirements than trait
    {
        ()
    }
}

fn main() {}
