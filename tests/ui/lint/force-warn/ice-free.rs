//@ compile-flags: --force-warn unused_variables
//@ check-pass

fn main() {
    let x = 10; //~ WARN unused variable: `x`
}
