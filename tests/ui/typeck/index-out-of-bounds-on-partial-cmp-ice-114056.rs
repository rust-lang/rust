//! Regression test for https://github.com/rust-lang/rust/issues/114056

struct P<Q>(Q);

impl<Q> P<Q> {
    fn foo(&self) {
        self.partial_cmp(())
        //~^ ERROR `&P<Q>` is not an iterator
    }
}

fn main() {}
