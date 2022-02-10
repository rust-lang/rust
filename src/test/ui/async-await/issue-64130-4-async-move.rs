// edition:2018
use std::any::Any;
use std::future::Future;

struct Client(Box<dyn Any + Send>);

impl Client {
    fn status(&self) -> u16 {
        200
    }
}

async fn get() {}

pub fn foo() -> impl Future + Send {
    //~^ ERROR future cannot be sent between threads safely
    async {
        match Client(Box::new(true)).status() {
            200 => {
                let _x = get().await;
            }
            _ => (),
        }
    }
}

fn main() {}
