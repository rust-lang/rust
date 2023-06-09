// run-pass
// Regression test for issue #21422, which was related to failing to
// add inference constraints that the operands of a binary operator
// should outlive the binary operation itself.

// pretty-expanded FIXME #23616

pub struct P<'a> {
    _ptr: *const &'a u8,
}

impl <'a> PartialEq for P<'a> {
    fn eq(&self, other: &P<'a>) -> bool {
        (self as *const _) == (other as *const _)
    }
}

fn main() {}
