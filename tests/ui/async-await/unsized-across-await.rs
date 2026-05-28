//@ edition: 2021

async fn f() {}

async fn g(x: Box<dyn std::fmt::Display>) {
    let _x = *x;
    //~^ ERROR the size for values of type `dyn std::fmt::Display` cannot be known at compilation time
    f().await;
}

fn main() {
    let _a = g(Box::new(5));
}
