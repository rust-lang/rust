fn main() {
    let foo = Some(0);
    let bar = None;
    if Some(x) = foo {} //~ ERROR cannot find value `x` in this scope
    //~^ ERROR mismatched types
    if Some(foo) = bar {} //~ ERROR mismatched types
    if 3 = foo {} //~ ERROR mismatched types
    if Some(3) = foo {} //~ ERROR mismatched types
    //~^ ERROR invalid left-hand side of assignment
    if x = 5 {}  //~ ERROR cannot find value `x` in this scope
}
