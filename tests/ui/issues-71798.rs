fn test_ref(x: &u32) -> impl std::future::Future<Output = u32> + '_ {
    //~^ ERROR `u32` is not a future
    *x
}

fn main() {
    let _ = test_ref & u; //~ ERROR cannot find value `u` in this scope
}
