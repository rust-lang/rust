fn f<T>(x: T) -> @T {
    @x  //~ ERROR value may contain borrowed pointers
}

fn g<T:'static>(x: T) -> @T {
    @x  // ok
}

fn main() {}

