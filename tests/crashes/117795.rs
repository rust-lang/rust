//@ known-bug: #117795
const fn f() -> usize {
    5
}

fn main() {
    let _ = [0; FnMut::call_mut(&mut f, ())];
}
