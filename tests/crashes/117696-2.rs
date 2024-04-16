//@ known-bug: #117696
//@ compile-flags: -Copt-level=0
fn main() {
    rec(&mut None::<()>.into_iter());
}

fn rec<T: Iterator>(mut it: T) {
    if true {
        it.next();
    } else {
        rec(&mut it);
    }
}
