// compile-flags: -Z continue-parse-after-error

fn main() {
    let x = "\x80"; //~ ERROR may only be used
    let y = "\xff"; //~ ERROR may only be used
    let z = "\xe2"; //~ ERROR may only be used
    let a = b"\x00e2";  // ok because byte literal
}
