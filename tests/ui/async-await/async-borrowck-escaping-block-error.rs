//@ edition:2018
//@ run-rustfix

fn test_boxed() -> Box<impl std::future::Future<Output = u32>> {
    let x = 0u32;
    Box::new(async { x } )
    //~^ ERROR E0373
}

fn test_ref(x: &u32) -> impl std::future::Future<Output = u32> + '_ {
    async { *x }
    //~^ ERROR E0373
}

fn main() {
    let _ = test_boxed();
    let _ = test_ref(&0u32);
}
