//@ edition:2021

// Test that we do not suggest `.await` when it doesn't make sense.

struct A;

impl A {
    fn test(&self) -> i32 {
        1
    }
}

async fn foo() -> A {
    A
}

async fn async_main() {
    let x: u32 = foo().test();
    //~^ ERROR no method named `test` found for opaque type `impl Future<Output = A>` in the current scope
}

fn main() {
    let _ = async_main();
}
