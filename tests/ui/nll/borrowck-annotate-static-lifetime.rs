//! regression test for issue <https://github.com/rust-lang/rust/issues/47184>
fn main() {
    let _vec: Vec<&'static String> = vec![&String::new()];
    //~^ ERROR temporary value dropped while borrowed [E0716]
}
