fn foo() -> &'static u32 {
    let x = 0;
    &x
    //~^ ERROR cannot return reference to local variable `x` [E0515]
}

fn main() { }
