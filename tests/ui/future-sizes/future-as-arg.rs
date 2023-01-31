// edition: 2021
// run-pass

async fn test(_arg: [u8; 8192]) {}

async fn use_future(fut: impl std::future::Future<Output = ()>) {
    fut.await
}

fn main() {
    let n = std::mem::size_of_val(&use_future(use_future(use_future(use_future(use_future(
            use_future(use_future(use_future(use_future(use_future(test(
                [0; 8192]
            ))))))
        ))))));
    println!("size is: {n}");
    // Not putting an exact number in case it slightly changes over different commits
    assert!(n > 8000000);
}
