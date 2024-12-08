// Regression test for issue #120615.

fn main() {
    [(); loop {}].field; //~ ERROR constant evaluation is taking a long time
}
