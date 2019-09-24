// build-pass (FIXME(62277): could be check-pass?)

#![allow(const_err)]

fn main() {
    let x: &'static _ = &|| { let z = 3; z };
}
