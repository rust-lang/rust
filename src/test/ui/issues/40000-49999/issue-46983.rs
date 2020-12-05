fn foo(x: &u32) -> &'static u32 {
    &*x
    //~^ ERROR explicit lifetime required in the type of `x` [E0621]
}

fn main() {}
