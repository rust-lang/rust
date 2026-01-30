//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(warnings)]

mod free_to_free {
    mod test_1 {
        fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

        pub fn check() {
            reuse foo as bar;
            bar::<i32, i32, 1>();
        }
    }

    mod test_2 {
        fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

        pub fn check<T: Clone, U: Clone>() {
            reuse foo as bar;
            bar::<T, U, 1>();
        }
    }

    mod test_3 {
        fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

        pub fn check<T: Clone, U: Clone>() {
            reuse foo as bar;
            bar::<u64, i32, 1>();
        }
    }

    mod test_4 {
        fn foo<'a, 'b, T: Clone, U: Clone, const N: usize>(_t: &'a T, _u: &'b U) {}

        pub fn check<T: Clone, U: Clone>() {
            reuse foo as bar;
            bar::<u64, i32, 1>(&1, &2);
        }
    }

    mod test_5 {
        fn foo<'a, 'b, T: Clone, const N: usize, U: Clone>(_t: &'a T, _u: &'b U) {}

        pub fn check<T: Clone, U: Clone>() {
            reuse foo as bar;
            bar::<u64, 1, u32>(&1, &2);
        }
    }

    mod test_6 {
        fn foo<'a, 'b, T: Clone, const N: usize, U: Clone>(_t: &'a T, _u: &'b U) {}

        pub fn check<T: Clone, U: Clone>() {
            reuse foo::<String, 1, String> as bar;
            bar(&"".to_string(), &"".to_string());
        }
    }

    // FIXME(fn_delegation): Uncomment this test when impl Traits in function params are supported

    // mod test_7 {
    //     fn foo<T, U>(f: impl FnOnce(T, U) -> U) {

    //     }

    //     pub fn check() {
    //         reuse foo as bar;
    //         bar::<i32, i32>(|x, y| y);
    //     }
    // }

    mod test_8 {
        pub fn check<T: Clone, U: Clone>() {
            fn foo<'a, 'b, T: Clone, const N: usize, U: Clone>(_t: &'a T, _u: &'b U) {}

            reuse foo::<String, 1, String> as bar;
            bar(&"".to_string(), &"".to_string());
        }
    }

    mod test_9 {
        pub fn check<T: Clone, U: Clone>() {
            let closure = || {
                fn foo<'a, 'b, T: Clone, const N: usize, U: Clone>(_t: &'a T, _u: &'b U) {}

                reuse foo::<String, 1, String> as bar;
                bar(&"".to_string(), &"".to_string());
            };

            closure();
        }
    }

    pub fn check() {
        test_1::check();
        test_2::check::<i32, String>();
        test_3::check::<i32, String>();
        test_4::check::<i32, String>();
        test_5::check::<i32, String>();
        test_6::check::<i32, String>();
        //test_7::check();
        test_8::check::<i32, String>();
        test_9::check::<String, i32>();
    }
}

fn main() {
    free_to_free::check();
}
