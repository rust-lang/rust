// Issue #52544
// run-rustfix

fn main() {
    let i: &i64 = &1;
    if i < 0 {}
    //~^ ERROR mismatched types [E0308]
}
