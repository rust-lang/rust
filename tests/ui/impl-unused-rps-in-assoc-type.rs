// Test that lifetime parameters must be constrained if they appear in
// an associated type def'n. Issue #22077.

trait Fun {
    type Output;
    fn call<'x>(&'x self) -> Self::Output;
}

struct Holder { x: String }

impl<'a> Fun for Holder { //~ ERROR E0207
    type Output = &'a str;
    fn call<'b>(&'b self) -> &'b str {
        &self.x[..]
    }
}

fn main() { }
