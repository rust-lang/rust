fn main() {
    let foo = Some(0);
    let bar = None;
    while Some(x) = foo {} //~ ERROR cannot find value `x` in this scope
    while Some(foo) = bar {}
    while 3 = foo {} //~ ERROR mismatched types
    while Some(3) = foo {} //~ ERROR invalid left-hand side of assignment
    while x = 5 {} //~ ERROR cannot find value `x` in this scope
}
