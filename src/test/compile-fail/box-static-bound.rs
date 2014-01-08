#[feature(managed_boxes)];

fn f<T>(x: T) -> @T {
    @x  //~ ERROR value may contain references
}

fn g<T:'static>(x: T) -> @T {
    @x  // ok
}

fn main() {}

