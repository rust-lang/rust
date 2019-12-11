// Test that ! errors when used in illegal positions with feature(never_type) disabled

trait Foo {
    type Wub;
}

type Ma = (u32, !, i32); //~ ERROR type is experimental
type Meeshka = Vec<!>; //~ ERROR type is experimental
type Mow = &'static fn(!) -> !; //~ ERROR type is experimental
type Skwoz = &'static mut !; //~ ERROR type is experimental

impl Foo for Meeshka {
    type Wub = !; //~ ERROR type is experimental
}

fn main() {
}
