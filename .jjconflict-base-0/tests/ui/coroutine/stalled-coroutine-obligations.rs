//@ edition: 2021

// Regression tests for #137916 and #138274
// We now check stalled coroutine obligations eagerly at the start of `mir_borrowck`.
// So these unsatisfied bounds are caught before causing ICEs.
use std::ptr::null;

async fn a() -> Box<dyn Send> {
    Box::new(async {
        //~^ ERROR: future cannot be sent between threads safely
        let non_send = null::<()>();
        &non_send;
        async {}.await
    })
}


trait Trait {}
fn foo() -> Box<dyn Trait> { todo!() }

fn fetch() {
    async {
        let fut = async {
            let _x = foo();
            async {}.await;
        };
        let _: Box<dyn Send> = Box::new(fut);
        //~^ ERROR: future cannot be sent between threads safely
    };
}

fn main() {}
