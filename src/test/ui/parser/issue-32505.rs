// compile-flags: -Z continue-parse-after-error

pub fn test() {
    foo(|_|) //~ ERROR expected expression, found `)`
}

fn main() { }
