//@ edition:2018

fn foo() -> Box<dyn std::future::Future<Output = u32>> {
    let x = 0u32;
    Box::new((async || x)())
    //~^ ERROR cannot return value referencing local variable `x`
}

fn main() {
}
