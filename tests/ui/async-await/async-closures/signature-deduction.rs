//@ build-pass
//@ edition: 2021

async fn foo(x: impl AsyncFn(&str) -> &str) {}

fn main() {
    foo(async |x| x);
}
