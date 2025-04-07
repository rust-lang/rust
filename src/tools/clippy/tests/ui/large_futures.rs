#![warn(clippy::large_futures)]
#![allow(clippy::never_loop)]
#![allow(clippy::future_not_send)]
#![allow(clippy::manual_async_fn)]

async fn big_fut(_arg: [u8; 1024 * 16]) {}

async fn wait() {
    let f = async {
        big_fut([0u8; 1024 * 16]).await;
        //~^ large_futures
    };
    f.await
    //~^ large_futures
}
async fn calls_fut(fut: impl std::future::Future<Output = ()>) {
    loop {
        wait().await;
        //~^ large_futures

        if true {
            return fut.await;
        } else {
            wait().await;
            //~^ large_futures
        }
    }
}

pub async fn test() {
    let fut = big_fut([0u8; 1024 * 16]);
    foo().await;
    //~^ large_futures

    calls_fut(fut).await;
    //~^ large_futures
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
        //~^ large_futures

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
                //~^ large_futures
                let x = [0i32; 1024 * 16];
                async {}.await;
                println!("macro: {:?}", x);
            }
        };
    }
    macro_!().await
}

fn main() {}
