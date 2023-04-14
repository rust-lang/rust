fn foo() -> [i32] {
    //~^ ERROR the size for values of type `[i32]` cannot be known at compilation time
    todo!()
}

fn main() {
    let x = foo();
    //~^ ERROR the size for values of type `[i32]` cannot be known at compilation time
}
