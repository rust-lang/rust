pub fn test() {
    foo(|_|) //~ ERROR expected expression, found `)`
}

fn main() { }
