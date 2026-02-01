//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(warnings)]

mod free_to_trait {
    mod test_1 {
        trait Trait<'b, 'c, 'a, T, const N: usize>: Sized {
            fn foo<'d: 'd, U, const M: bool>(self) {}
        }

        impl Trait<'static, 'static, 'static, i32, 1> for u8 {}

        pub fn check() {
            fn no_ctx() {
                reuse Trait::foo as bar;
                bar::<'static, 'static, 'static, 'static, u8, i32, 1, String, true>(123);
            }

            fn with_ctx<'a, 'b, 'c, A, B, C, const N: usize, const M: bool>() {
                reuse Trait::foo as bar;
                bar::<'static, 'static, 'static, 'a, u8, i32, 1, A, M>(123);
            }

            no_ctx();
            with_ctx::<i32, i32, i32, 1, true>();
        }
    }

    mod test_2 {
        trait Trait<'a, T, const N: usize>: Sized {
            fn foo<'b: 'b, U, const M: bool>(self) {}
        }

        impl Trait<'static, i32, 1> for u8 {}

        pub fn check() {
            reuse Trait::<'static, i32, 1>::foo as bar;

            bar::<'static, u8, String, true>(123);
        }
    }

    mod test_3 {
        trait Trait<'a, T, const N: usize>: Sized {
            fn foo<'b: 'b, U, const M: bool>(self) {}
        }

        impl Trait<'static, String, 1> for u8 {}

        pub fn check() {
            reuse Trait::foo::<'static, i32, true> as bar;

            bar::<'static, u8, String, 1>(123);
        }
    }

    mod test_4 {
        trait Trait<T, const N: usize>: Sized {
            fn foo<'b: 'b, U, const M: bool>(self) {}
        }

        impl Trait<String, 1> for u8 {}

        pub fn check() {
            reuse Trait::foo::<'static, i32, true> as bar;

            bar::<u8, String, 1>(123);
        }
    }

    mod test_5 {
        trait Trait<const N: usize>: Sized {
            fn foo<'b: 'b, U, const M: bool>(self) {}
        }

        impl Trait<1> for u8 {}

        pub fn check() {
            reuse Trait::foo::<'static, i32, true> as bar;

            bar::<u8, 1>(123);
        }
    }

    mod test_6 {
        trait Trait: Sized {
            fn foo<'b: 'b, U, const M: bool>(self) {}
        }

        impl Trait for u8 {}

        pub fn check() {
            reuse Trait::foo::<'static, i32, true> as bar;

            bar::<u8>(123);
        }
    }

    mod test_7 {
        trait Trait<'a, T, const N: usize>: Sized {
            fn foo<U, const M: bool>(self) {}
        }

        impl Trait<'static, i32, 1> for u8 {}

        pub fn check() {
            reuse Trait::<'static, i32, 1>::foo as bar;

            bar::<u8, String, true>(123);
        }
    }

    mod test_8 {
        trait Trait<'a, T, const N: usize>: Sized {
            fn foo<const M: bool>(self) {}
        }

        impl Trait<'static, i32, 1> for u8 {}

        pub fn check() {
            reuse Trait::<'static, i32, 1>::foo as bar;

            bar::<u8, true>(123);
        }
    }

    mod test_9 {
        trait Trait<'a, T, const N: usize>: Sized {
            fn foo(self) {}
        }

        impl Trait<'static, i32, 1> for u8 {}

        pub fn check() {
            reuse Trait::<'static, i32, 1>::foo as bar;

            bar::<u8>(123);
        }
    }

    mod test_10 {
        trait Trait<'b, 'c, T> {
            fn foo<'d: 'd, U, const M: bool>() {}
        }

        impl<'b, 'c, T> Trait<'b, 'c, T> for u8 {}

        pub fn check() {
            fn with_ctx<'a, 'b, 'c, A, B, C, const N: usize, const M: bool>() {
                reuse <u8 as Trait>::foo as bar;
                bar::<'a, 'b, 'c, u8, C, A, M>();
                bar::<'static, 'static, 'static, u8, i32, i32, false>();

                reuse <u8 as Trait::<'static, 'static, i32>>::foo as bar1;
                bar1::<'static, u8, i32, true>();

                reuse <u8 as Trait::<'static, 'static, i32>>::foo::<'static, u32, true> as bar2;
                bar2::<u8>();
            }

            with_ctx::<i32, i32, i32, 1, true>();
        }
    }

    mod test_11 {
        trait Bound0 {}
        trait Bound1 {}
        trait Bound2<T, U> {}

        trait Trait<'a: 'static, T, P>
        where
            Self: Sized,
            T: Bound0,
            P: Bound2<P, Vec<Vec<Vec<T>>>>,
        {
            fn foo<'d: 'd, U: Bound1, const M: bool>() {}
        }

        impl Bound0 for u32 {}
        impl Bound1 for String {}
        impl<'a: 'static, T: Bound0, P: Bound2<P, Vec<Vec<Vec<T>>>>> Trait<'a, T, P> for usize {}

        struct Struct;
        impl Bound2<Struct, Vec<Vec<Vec<u32>>>> for Struct {}

        pub fn check<'b>() {
            reuse <usize as Trait>::foo;
            foo::<'static, 'b, usize, u32, Struct, String, false>();
        }
    }

    mod test_12 {
        trait Trait<'a, T = usize, const N: usize = 123>: Sized {
            fn foo(self) {}
        }

        impl Trait<'static, i32, 1> for u8 {}

        pub fn check() {
            reuse Trait::foo as bar;

            bar::<u8, i32, 1>(123);
        }
    }

    pub fn check() {
        test_1::check();
        test_2::check();
        test_3::check();
        test_4::check();
        test_5::check();
        test_6::check();
        test_7::check();
        test_8::check();
        test_9::check();
        test_10::check();
        test_11::check();
        test_12::check();
    }
}

fn main() {
    free_to_trait::check();
}
