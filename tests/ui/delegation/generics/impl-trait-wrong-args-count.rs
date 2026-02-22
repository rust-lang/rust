#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn bar<'a: 'a, 'b: 'b, A, B>(x: &super::XX) {}
    pub fn bar1(x: &super::XX) {}
    pub fn bar2<A, B, C, D, E, F, const X: usize, const Y: bool>(x: &super::XX) {}
}

trait Trait<'a, 'b, 'c, A, B, const N: usize>: Sized {
    fn bar<'x: 'x, 'y: 'y, AA, BB, const NN: usize>(&self) {}
    fn bar1<'x: 'x, 'y: 'y, AA, BB, const NN: usize>(&self) {}
    fn bar2(&self) {}
    fn bar3(&self) {}
    fn bar4<X, Y, Z>(&self) {}
}

struct X<'x1, 'x2, 'x3, 'x4, X1, X2, const X3: usize>(
    &'x1 X1, &'x2 X2, &'x3 X1, &'x4 [usize; X3]);
type XX = X::<'static, 'static, 'static, 'static, i32, i32, 3>;

impl<'a, 'b, 'c, A, B, const N: usize> Trait<'a, 'b, 'c, A, B, N> for XX {
    reuse to_reuse::bar;
    //~^ ERROR: type annotations needed

    reuse to_reuse::bar1;

    reuse to_reuse::bar2;
    //~^ ERROR: type annotations needed
    //~| ERROR: type annotations needed

    reuse to_reuse::bar2::<i32, i32, i32, i32, i32, i32, 123, true> as bar3;

    reuse to_reuse::bar2::<i32, i32, i32, i32, i32, i32, 123, true> as bar4;
}

fn main() {
}
