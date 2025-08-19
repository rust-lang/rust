//! Verify that we do not ICE when a coroutine body is malformed.
//@ compile-flags: -Zmir-enable-passes=+GVN
//@ edition: 2018

pub enum Request {
    TestSome(T),
    //~^ ERROR cannot find type `T` in this scope [E0412]
}

pub async fn handle_event(event: Request) {
    async move {
        static instance: Request = Request { bar: 17 };
        //~^ ERROR expected struct, variant or union type, found enum `Request` [E0574]
        &instance
    }
    .await;
}

fn main() {}
