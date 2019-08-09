fn id<T>(x: T) -> T { x }

fn f(_x: &isize) -> &isize {
    return &id(3); //~ ERROR cannot return reference to temporary value
}

fn main() {
}
