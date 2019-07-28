// run-pass
#![allow(unused_variables)]
// Test that we correctly handle projection bounds appearing in the
// supertrait list (and in conjunction with overloaded operators). In
// this case, the `Result=Self` binding in the supertrait listing of
// `Int` was being ignored.

trait Not {
    type Result;

    fn not(self) -> Self::Result;
}

trait Int: Not<Result=Self> + Sized {
    fn count_ones(self) -> usize;
    fn count_zeros(self) -> usize {
        // neither works
        let x: Self = self.not();
        0
    }
}

fn main() { }
