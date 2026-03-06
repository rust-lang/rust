//@ edition: 2021
#![allow(todo_macro_uses)]

struct S {
    field: (),
}

async fn foo() -> S { todo!() }

fn main() -> Result<(), ()> {
    foo().field;
    //~^ ERROR no field `field` on type `impl Future<Output = S>`
    Ok(())
}
