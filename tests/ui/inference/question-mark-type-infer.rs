// Test that type inference fails where there are multiple possible return types
// for the `?` operator.

fn f(x: &i32) -> Result<i32, ()> {
    Ok(*x)
}

fn g() -> Result<Vec<i32>, ()> {
    let l = [1, 2, 3, 4];
    l.iter().map(f).collect()?
    //~^ ERROR type annotations needed
}
fn h() -> Result<(), ()> {
    let l = [1, 2, 3, 4];
    // The resulting binding doesn't have a type, so we need to guess the `Ok` type too.
    let x = l.iter().map(f).collect()?;
    //~^ ERROR type annotations needed
    Ok(())
}
fn i() -> Result<(), ()> {
    let l = [1, 2, 3, 4];
    // The resulting binding already has a type, so we don't need to specify the `Ok` type.
    let x: Vec<i32> = l.iter().map(f).collect()?;
    //~^ ERROR type annotations needed
    Ok(())
}

fn main() {
    g();
}
