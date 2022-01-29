fn test_ref(x: &u32) -> impl std::future::Future<Output = u32> + '_ {
    *x //~^ ERROR `u32` is not a future
}

fn main() {
    let _ = test_ref & u; //~ ERROR cannot find value `u` in this scope
}
