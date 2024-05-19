#![deny(unused_qualifications)]
//@ check-pass
fn bar() {
    match Option::<Option<()>>::None {
        Some(v) => {}
        None => {}
    }
}

fn main() {}
