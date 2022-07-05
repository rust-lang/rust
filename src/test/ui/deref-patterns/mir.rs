// compile-flags: -Z unpretty=mir
// build-pass
#![feature(deref_patterns)]

fn main() {
    let s = Some(String::new());
    let a;
    match s {
        Some("a") => a = 1234,
        s => a = 4321,
    }
}
