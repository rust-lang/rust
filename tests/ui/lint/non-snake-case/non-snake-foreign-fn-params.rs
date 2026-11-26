//@ check-pass

#![deny(non_snake_case)]

extern "C" {
    fn test(X: usize); // OK
}

fn main() {}
