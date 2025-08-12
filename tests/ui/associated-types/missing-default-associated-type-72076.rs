// https://github.com/rust-lang/rust/issues/72076
trait X {
    type S;
    fn f() -> Self::S {} //~ ERROR mismatched types
}

fn main() {}
