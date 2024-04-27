fn digit() -> str {
    return {};
    //~^ ERROR: mismatched types [E0308]
}
fn main() {
    let [_y..] = [Box::new(1), Box::new(2)];
    //~^ ERROR: cannot find value `_y` in this scope [E0425]
    //~| ERROR: `X..` patterns in slices are experimental [E0658]
    //~| ERROR: pattern requires 1 element but array has 2 [E0527]
}
