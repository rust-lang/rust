//@ check-pass

// mir borrowck previously incorrectly set `tainted_by_errors`
// when buffering lints, which resulted in ICE later on,
// see #94502.

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
}
