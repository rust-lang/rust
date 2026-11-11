struct Bar<T> { x: T } where T: Copy //~ ERROR expected item, found keyword `where`

fn main() {}
