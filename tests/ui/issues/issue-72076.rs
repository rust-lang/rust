trait X {
    type S;
    fn f() -> Self::S {} //~ ERROR mismatched types
}

fn main() {}
