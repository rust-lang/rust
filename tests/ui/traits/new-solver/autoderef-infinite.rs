// compile-flags: -Ztrait-solver=next

fn main() {
    let y = [Default::default()];
    y[0].method();
    //~^ ERROR type annotations needed
    //~| ERROR no method named `method` found
}
