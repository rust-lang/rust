// edition: 2021
// revisions: no_drop_tracking drop_tracking_mir
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

#![feature(unsized_locals)]
//~^ WARN the feature `unsized_locals` is incomplete

async fn f() {}

async fn g(x: Box<dyn std::fmt::Display>) {
    let _x = *x;
    //~^ ERROR the size for values of type `dyn std::fmt::Display` cannot be known at compilation time
    f().await;
}

fn main() {
    let _a = g(Box::new(5));
}
