//! regression test for <https://github.com/rust-lang/rust/issues/16683>
trait T<'a> {
    fn a(&'a self) -> &'a bool;
    fn b(&self) {
        self.a();
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
