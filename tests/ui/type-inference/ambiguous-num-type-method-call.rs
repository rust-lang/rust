//! regression test for <https://github.com/rust-lang/rust/issues/51874>

fn main() {
    let a = (1.0).pow(1.0); //~ ERROR can't call method `pow` on ambiguous numeric type
}
