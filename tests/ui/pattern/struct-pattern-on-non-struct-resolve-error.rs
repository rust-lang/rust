// Regression test for #135209.
// We ensure that we don't try to access fields on a non-struct pattern type.
fn main() {
    if let <Vec<()> as Iterator>::Item { .. } = 1 {
        //~^ ERROR E0658
        //~| ERROR E0071
        //~| ERROR E0277
        x //~ ERROR E0425
    }
}
