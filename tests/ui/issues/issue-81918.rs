// check-pass
// dont-check-compiler-stdout
// compile-flags: -Z unpretty=mir-cfg

// This checks that unpretty=mir-cfg does not panic. See #81918.

const TAG: &'static str = "ABCD";

fn main() {
    if TAG == "" {}
}
