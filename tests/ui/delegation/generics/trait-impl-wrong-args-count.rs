#![feature(fn_delegation)]

// Testing delegation from trait impl to free functions.
mod test_1 {
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
        //~^ ERROR: function takes at most 2 generic arguments but 3 generic arguments were supplied

        reuse to_reuse::bar1;
        //~^ ERROR: function takes 0 generic arguments but 3 generic arguments were supplied

        reuse to_reuse::bar2;
        //~^ ERROR: type annotations needed
        //~| ERROR: type annotations needed

        reuse to_reuse::bar2::<i32, i32, i32, i32, i32, i32, 123, true> as bar3;

        reuse to_reuse::bar2::<i32, i32, i32, i32, i32, i32, 123, true> as bar4;
    }
}

// Testing delegations of trait impl to other different trait
// with errors in Trait1 generics count.
mod test_2 {
    trait Trait<A, const N: usize> {
        fn bar<'x: 'x, 'y: 'y, AA, BB, const NN: usize>() {}
        fn bar1<'x: 'x, 'y: 'y, AA, BB, const NN: usize>() {}
        fn bar2() {}
        fn bar3() {}
        fn bar4<X, Y, Z>() {}
    }

    trait Trait1<A, B> {
        fn bar<'x: 'x, AA, BB, const NN: usize>() {}
    }

    struct X;

    impl<A, B> Trait1<A, B> for X {}

    impl Trait<String, 1> for X {
        reuse <X as Trait1>::bar;
        //~^ ERROR: missing generics for trait

        reuse <X as Trait1::<bool, bool>>::bar as bar1;

        reuse <X as Trait1::<bool, bool>>::bar::<'static, u32, u32, 1> as bar2;

        reuse <X as Trait1>::bar::<'static, u32, u32, 1> as bar3;
        //~^ ERROR: missing generics for trait

        reuse <X as Trait1>::bar as bar4;
        //~^ ERROR: missing generics for trait
    }
}

// Testing delegations of trait impl to other different trait
// with Trait1::bar and Trait1::foo wrong generics count.
mod test_3 {
    trait Trait<A, const N: usize> {
        fn bar<'x: 'x, 'y: 'y, AA, BB, const NN: usize>() {}
        fn bar1<'x: 'x, 'y: 'y, AA, BB, const NN: usize>() {}
        fn bar2() {}
        fn bar3<X, Y, Z>() {}
    }

    trait Trait1<A, B> {
        fn bar() {}
        fn foo<X, Y>() {}
    }

    struct X;

    impl<A, B> Trait1<A, B> for X {}

    impl Trait<String, 1> for X {
        reuse <X as Trait1::<(), ()>>::bar;
        //~^ ERROR: associated function takes 0 generic arguments but 3 generic arguments were supplied

        reuse <X as Trait1::<(), ()>>::bar as bar1;
        //~^ ERROR: associated function takes 0 generic arguments but 3 generic arguments were supplied

        reuse <X as Trait1::<(), ()>>::foo as bar2;
        //~^ ERROR: type annotations needed

        reuse <X as Trait1::<(), ()>>::foo as bar3;
        //~^ ERROR: associated function takes at most 2 generic arguments but 3 generic arguments were supplied
    }
}

fn main() {
}
