// check-pass
#![feature(inline_const)]
fn main() {
    match true {
        true => const {}
        false => ()
    }
    const {}
    ()
}
