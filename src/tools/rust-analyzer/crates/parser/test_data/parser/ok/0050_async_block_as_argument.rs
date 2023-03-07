fn foo(x: impl std::future::Future<Output = i32>) {}

fn main() {
    foo(async move { 12 })
}
