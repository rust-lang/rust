// Check that when we test the supertrait we ensure consistent use of
// lifetime parameters. In this case, implementing T2<'a,'b> requires
// an impl of T1<'a>, but we have an impl of T1<'b>.

trait T1<'x> {
    fn x(&self) -> &'x isize;
}

trait T2<'x, 'y> : T1<'x> {
    fn y(&self) -> &'y isize;
}

struct S<'a, 'b> {
    a: &'a isize,
    b: &'b isize
}

impl<'a,'b> T1<'b> for S<'a, 'b> {
    fn x(&self) -> &'b isize {
        self.b
    }
}

impl<'a,'b> T2<'a, 'b> for S<'a, 'b> { //~ ERROR cannot infer an appropriate lifetime
    fn y(&self) -> &'b isize {
        self.b
    }
}

pub fn main() {
}
