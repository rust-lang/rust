//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
// FIXME(#61117): Respect debuginfo-level-tests, do not force debuginfo=0
//@[opt] compile-flags: -C debuginfo=0
//@[opt] compile-flags: -O
//! Make sure we error on erroneous consts even if they get promoted.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation panicked: explicit panic
    //[opt]~^ ERROR evaluation panicked: explicit panic
    // (Not sure why optimizations lead to this being emitted twice, but as long as compilation
    // fails either way it's fine.)
}

#[inline(never)]
fn f<T>() {
    if false {
        // If promotion moved `C` from our required_consts to its own, without adding itself to
        // our required_consts, then we'd miss the const-eval failure here.
        let _val = &Fail::<T>::C;
    }
}

fn main() {
    f::<i32>();
}
