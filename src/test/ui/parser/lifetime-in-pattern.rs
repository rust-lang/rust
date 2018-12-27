// compile-flags: -Z parse-only

fn test(&'a str) {
    //~^ ERROR unexpected lifetime `'a` in pattern
}

fn main() {
}
