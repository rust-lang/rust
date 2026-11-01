//! Regression test for https://github.com/rust-lang/rust/issues/114056

struct P<Q>(Q);

impl<Q> P<Q> {
    fn foo(&self) {
        self.partial_cmp(())
        //~^ ERROR the method `partial_cmp` exists for reference `&P<Q>`, but its trait bounds were not satisfied
    }
}

fn main() {}
