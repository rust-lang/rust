// edition:2018

#![feature(async_closure)]
fn foo() -> Box<dyn std::future::Future<Output = u32>> {
    let x = 0u32;
    Box::new((async || x)())
    //~^ ERROR closure may outlive the current function, but it borrows `x`, which is owned by the current function
}

fn main() {
}
