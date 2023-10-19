#![feature(coroutines)]
#![warn(clippy::large_futures)]
#![allow(clippy::never_loop)]
#![allow(clippy::future_not_send)]
#![allow(clippy::manual_async_fn)]

async fn big_fut(_arg: [u8; 1024 * 16]) {}

async fn wait() {
    let f = async {
        big_fut([0u8; 1024 * 16]).await;
        //~^ ERROR: large future with a size of 16385 bytes
        //~| NOTE: `-D clippy::large-futures` implied by `-D warnings`
    };
    f.await
    //~^ ERROR: large future with a size of 16386 bytes
}
async fn calls_fut(fut: impl std::future::Future<Output = ()>) {
    loop {
        wait().await;
        //~^ ERROR: large future with a size of 16387 bytes
        if true {
            return fut.await;
        } else {
            wait().await;
            //~^ ERROR: large future with a size of 16387 bytes
        }
    }
}

pub async fn test() {
    let fut = big_fut([0u8; 1024 * 16]);
    foo().await;
    //~^ ERROR: large future with a size of 65540 bytes
    calls_fut(fut).await;
    //~^ ERROR: large future with a size of 49159 bytes
}

pub fn foo() -> impl std::future::Future<Output = ()> {
    async {
        let x = [0i32; 1024 * 16];
        async {}.await;
        dbg!(x);
    }
}

pub async fn lines() {
    async {
        //~^ ERROR: large future with a size of 65540 bytes
        let x = [0i32; 1024 * 16];
        async {}.await;
        println!("{:?}", x);
    }
    .await;
}

pub async fn macro_expn() {
    macro_rules! macro_ {
        () => {
            async {
                let x = [0i32; 1024 * 16];
                async {}.await;
                println!("macro: {:?}", x);
            }
        };
    }
    macro_!().await
}

fn main() {}
