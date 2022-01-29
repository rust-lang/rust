// check-pass
// edition:2021

macro_rules! m {
    () => {
        async {}.await
    };
}

async fn with_await() {
    println!("{} {:?}", "", async {}.await);
}

async fn with_macro_call() {
    println!("{} {:?}", "", m!());
}

fn assert_send(_: impl Send) {}

fn main() {
    assert_send(with_await());
    assert_send(with_macro_call());
}
