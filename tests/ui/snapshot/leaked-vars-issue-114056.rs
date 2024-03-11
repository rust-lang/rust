// Regression test for #114056. Fixed by #111516.
struct P<Q>(Q);
impl<Q> P<Q> {
    fn foo(&self) {
        self.partial_cmp(())
        //~^ ERROR the method `partial_cmp` exists for reference `&P<Q>`
    }
}

fn main() {}
