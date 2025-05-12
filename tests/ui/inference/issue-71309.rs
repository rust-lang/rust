fn foo(x: Result<i32, ()>) -> Result<(), ()> {
    let y: u32 = x?;
    //~^ ERROR: `?` operator has incompatible types
    Ok(())
}

fn main() {}
