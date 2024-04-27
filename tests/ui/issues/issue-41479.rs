//@ run-pass
fn split<A, B>(pair: (A, B)) {
    let _a = pair.0;
    let _b = pair.1;
}

fn main() {
    split(((), ((), ())));
}
