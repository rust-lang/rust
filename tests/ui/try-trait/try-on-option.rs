fn main() {}

fn foo() -> Result<u32, ()> {
    let x: Option<u32> = None;
    x?; //~ ERROR the `?` operator
    Ok(22)
}

fn bar() -> u32 {
    let x: Option<u32> = None;
    x?; //~ ERROR the `?` operator
    22
}
