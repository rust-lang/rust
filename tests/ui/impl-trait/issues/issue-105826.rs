//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

use std::io::Write;

struct A(Vec<u8>);

struct B<'a> {
    one: &'a mut A,
    two: &'a mut Vec<u8>,
    three: Vec<u8>,
}

impl<'a> B<'a> {
    fn one(&mut self) -> &mut impl Write {
        &mut self.one.0
    }
    fn two(&mut self) -> &mut impl Write {
        &mut *self.two
    }
    fn three(&mut self) -> &mut impl Write {
        &mut self.three
    }
}

struct C<'a>(B<'a>);

impl<'a> C<'a> {
    fn one(&mut self) -> &mut impl Write {
        self.0.one()
    }
    fn two(&mut self) -> &mut impl Write {
        self.0.two()
    }
    fn three(&mut self) -> &mut impl Write {
        self.0.three()
    }
}

fn main() {}
