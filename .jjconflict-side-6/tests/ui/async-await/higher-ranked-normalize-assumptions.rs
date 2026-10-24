//@ revisions: stock hr
//@[hr] compile-flags: -Zhigher-ranked-assumptions
//@ edition: 2024
//@ check-pass

// Test that we don't normalize the higher-ranked assumptions of an auto trait goal
// unless we have `-Zhigher-ranked-assumptions`, since obligations that result from
// this normalization may lead to higher-ranked lifetime errors when the flag is not
// enabled.

// Regression test for <https://github.com/rust-lang/rust/issues/147285>.

pub trait Service {
    type Response;
}

impl<T, R> Service for T
where
    T: FnMut() -> R,
    R: 'static,
{
    type Response = R;
}

async fn serve<C>(_: C)
where
    C: Service,
    C::Response: 'static,
{
    connect::<C>().await;
}

async fn connect<C>()
where
    C: Service,
    C::Response: 'static,
{
}

fn repro() -> impl Send {
    async {
        let server = || do_something();
        serve(server).await;
    }
}

fn do_something() -> Box<dyn std::error::Error> {
    unimplemented!()
}

fn main() {}
