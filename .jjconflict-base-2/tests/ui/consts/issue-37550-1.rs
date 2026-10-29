//@ check-pass

const fn x() {
    let t = true;
    let x = || t;
}

fn main() {}
