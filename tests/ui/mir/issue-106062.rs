//@ edition:2018

use std::{future::Future, marker::PhantomData};

fn spawn<T>(future: T) -> PhantomData<T::Output>
where
    T: Future,
{
    loop {}
}

#[derive(Debug)]
struct IncomingServer {}
impl IncomingServer {
    async fn connection_handler(handler: impl Sized) -> Result<Ok, std::io::Error> {
        //~^ ERROR expected type, found variant `Ok` [E0573]
        loop {}
    }
    async fn spawn(&self, request_handler: impl Sized) {
        async move {
            spawn(Self::connection_handler(&request_handler));
        };
    }
}

fn main() {}
