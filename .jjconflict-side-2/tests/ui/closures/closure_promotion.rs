//@ check-pass

fn main() {
    let x: &'static _ = &|| { let z = 3; z };
}
