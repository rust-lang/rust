// https://github.com/rust-lang/rust/issues/64559
fn main() {
    let orig = vec![true];
    for _val in orig {}
    let _closure = || orig;
    //~^ ERROR use of moved value: `orig`
}
