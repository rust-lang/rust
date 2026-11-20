//! Regression test for issue #22077
//! lifetime parameters must be constrained in associated type definitions

trait Fun {
    type Output;
    fn call<'x>(&'x self) -> Self::Output;
}

struct Holder {
    x: String,
}

impl<'a> Fun for Holder {
    //~^ ERROR E0207
    type Output = &'a str;
    fn call<'b>(&'b self) -> &'b str {
        &self.x[..]
    }
}

fn main() {}
