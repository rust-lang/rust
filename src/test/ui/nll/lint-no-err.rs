// check-pass

// mir borrowck previously incorrectly set `tainted_by_errors`
// when buffering lints, which resulted in ICE later on,
// see #94502.

// Errors with `nll` which is already tested in enough other tests,
// so we ignore it here.
//
// ignore-compare-mode-nll

struct Repro;
impl Repro {
    fn get(&self) -> &i32 {
        &3
    }

    fn insert(&mut self, _: i32) {}
}

fn main() {
    let x = &0;
    let mut conflict = Repro;
    let prev = conflict.get();
    conflict.insert(*prev + *x);
    //~^ WARN cannot borrow `conflict` as mutable because it is also borrowed as immutable
    //~| WARN this borrowing pattern was not meant to be accepted
}
