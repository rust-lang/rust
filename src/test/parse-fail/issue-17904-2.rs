// compile-flags: -Z parse-only -Z continue-parse-after-error

struct Bar<T> { x: T } where T: Copy //~ ERROR expected item, found `where`

fn main() {}
