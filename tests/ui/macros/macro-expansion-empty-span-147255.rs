//! Regression test for <https://github.com/rust-lang/rust/issues/147255>

fn main() {
    let mut x = 4;
    let x_str = {
        format!("{}", x);
        //()
    };
    println!("{}", x_str); //~ ERROR `()` doesn't implement `std::fmt::Display`
}
