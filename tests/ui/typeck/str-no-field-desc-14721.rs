//! Regression test for https://github.com/rust-lang/rust/issues/14721

fn main() {
    let foo = "str";
    println!("{}", foo.desc); //~ ERROR no field `desc` on type `&str`
}
