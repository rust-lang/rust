// edition:2018

type AsyncFnPtr = Box<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>>;

async fn test() {}

#[allow(unused_must_use)]
fn main() {
    Box::new(test) as AsyncFnPtr;
    //~^ ERROR expected `fn() -> impl Future<Output = ()> {test}` to be a fn item that returns `Pin<Box<(dyn Future<Output = ()> + 'static)>>`, but it returns `impl Future<Output = ()>`
}
