//@ build-fail
//@ compile-flags: -Copt-level=0
fn main() {
    rec(&mut None::<()>.into_iter());
}

fn rec<T: Iterator>(mut it: T) {
    if true {
        it.next();
    } else {
        rec(&mut it);
        //~^ ERROR: reached the recursion limit while instantiating
    }
}
