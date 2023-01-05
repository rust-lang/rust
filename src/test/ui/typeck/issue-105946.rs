fn digit() -> str {
  return {};
  //~^ ERROR: mismatched types [E0308]
}
fn main() {
    let [_y..] = [box 1, box 2];
    //~^ ERROR: cannot find value `_y` in this scope [E0425]
    //~| ERROR: `X..` patterns in slices are experimental [E0658]
    //~| ERROR: box expression syntax is experimental; you can call `Box::new` instead [E0658]
    //~| ERROR: box expression syntax is experimental; you can call `Box::new` instead [E0658]
    //~| ERROR: pattern requires 1 element but array has 2 [E0527]
}
