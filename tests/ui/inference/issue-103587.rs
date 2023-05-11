fn main() {
    let x = Some(123);

    if let Some(_) == x {}
    //~^ ERROR expected `=`, found `==`

    if Some(_) = x {}
    //~^ ERROR mismatched types

    if None = x { }
    //~^ ERROR mismatched types
}
