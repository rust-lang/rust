#![warn(clippy::future_not_send)]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

async fn private_future(rc: Rc<[u8]>, cell: &Cell<usize>) -> bool {
    //~^ ERROR: future cannot be sent between threads safely
    async { true }.await
}

pub async fn public_future(rc: Rc<[u8]>) {
    //~^ ERROR: future cannot be sent between threads safely
    async { true }.await;
}

pub async fn public_send(arc: Arc<[u8]>) -> bool {
    async { false }.await
}

async fn private_future2(rc: Rc<[u8]>, cell: &Cell<usize>) -> bool {
    //~^ ERROR: future cannot be sent between threads safely
    true
}

pub async fn public_future2(rc: Rc<[u8]>) {}
//~^ ERROR: future cannot be sent between threads safely

pub async fn public_send2(arc: Arc<[u8]>) -> bool {
    false
}

struct Dummy {
    rc: Rc<[u8]>,
}

impl Dummy {
    async fn private_future(&self) -> usize {
        //~^ ERROR: future cannot be sent between threads safely
        async { true }.await;
        self.rc.len()
    }

    pub async fn public_future(&self) {
        //~^ ERROR: future cannot be sent between threads safely
        self.private_future().await;
    }

    #[allow(clippy::manual_async_fn)]
    pub fn public_send(&self) -> impl std::future::Future<Output = bool> {
        async { false }
    }
}

async fn generic_future<T>(t: T) -> T
//~^ ERROR: future cannot be sent between threads safely
where
    T: Send,
{
    let rt = &t;
    async { true }.await;
    let _ = rt;
    t
}

async fn generic_future_send<T>(t: T)
where
    T: Send,
{
    async { true }.await;
}

async fn unclear_future<T>(t: T) {}
//~^ ERROR: future cannot be sent between threads safely

fn main() {
    let rc = Rc::new([1, 2, 3]);
    private_future(rc.clone(), &Cell::new(42));
    public_future(rc.clone());
    let arc = Arc::new([4, 5, 6]);
    public_send(arc);
    generic_future(42);
    generic_future_send(42);

    let dummy = Dummy { rc };
    dummy.public_future();
    dummy.public_send();
}
