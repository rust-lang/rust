//@ edition: 2021

async fn a() -> i32 {
    0
}

fn foo() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
    //~| NOTE not allowed in type signatures
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl Future<Output = i32>
    a()
}

fn bar() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
    //~| NOTE not allowed in type signatures
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl Future<Output = i32>
    async { a().await }
}

fn main() {}
