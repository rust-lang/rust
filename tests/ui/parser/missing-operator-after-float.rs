//! regression test for issue <https://github.com/rust-lang/rust/issues/45965>
fn main() {
    let a = |r: f64| if r != 0.0(r != 0.0) { 1.0 } else { 0.0 };
    //~^ ERROR expected function, found `{float}`
}
