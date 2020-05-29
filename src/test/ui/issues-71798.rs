fn test_ref(x: &u32) -> impl std::future::Future<Output = u32> + '_ {
    *x //~^ ERROR the trait bound `u32: std::future::Future` is not satisfied
}

fn main() {
    let _ = test_ref & u; //~ ERROR cannot find value `u` in this scope
}
