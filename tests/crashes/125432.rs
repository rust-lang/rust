//@ known-bug: rust-lang/rust#125432

fn separate_arms() {
    // Here both arms perform assignments, but only one is illegal.

    let mut x = None;
    match x {
        None => {
            // It is ok to reassign x here, because there is in
            // fact no outstanding loan of x!
            x = Some(0);
        }
        Some(right) => consume(right),
    }
}

fn main() {}
