//@ compile-flags: -Z unpretty=mir

fn main() {
    let x: () = 0; //~ ERROR: mismatched types
}
