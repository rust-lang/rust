// Documents the current repeated-stop behavior when multiple MIR locations map
// to the same source breakpoint line.
// This may look trivial, but a bunch of code runs in std before
// `main` is called, so we are ensuring that that all works.
fn main() {
    let _value = 0;
}
