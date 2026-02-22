#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(late_bound_lifetime_arguments)]

//! This is one of the mapping tests, which tests mapping of delegee parent and child
//! generic params, whose main goal is to create cases with
//! different number of lifetimes/types/consts in delegee child and parent; and in
//! delegation parent if applicable. At some tests predicates are
//! added. At some tests user-specified args are specified in reuse statement.

// Testing lifetimes + types + consts, reusing without
// user args, checking predicates inheritance
mod test_1 {
    fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

    pub fn check() {
        reuse foo as bar;
        //~^ ERROR: type annotations needed [E0284]
        bar::<i32, i32, 1>();
    }
}

// Testing lifetimes + types + consts, reusing without user args,
// providing delegation parent args in invocation
mod test_2 {
    fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

    pub fn check<T: Clone, U: Clone>() {
        reuse foo as bar;
        //~^ ERROR: type annotations needed [E0284]
        bar::<T, U, 1>();
    }
}

// Testing lifetimes + types + consts, reusing without user args,
// providing random types with delegation parent generics specified
mod test_3 {
    fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

    pub fn check<T: Clone, U: Clone>() {
        reuse foo as bar;
        //~^ ERROR: type annotations needed [E0284]
        bar::<u64, i32, 1>();
    }
}

// Testing late-bound lifetimes + types + consts, reusing without user args,
// providing random types with delegation parent generics specified,
// checking signature inheritance
mod test_4 {
    fn foo<'a, 'b, T: Clone, U: Clone, const N: usize>(_t: &'a T, _u: &'b U) {}

    pub fn check<T: Clone, U: Clone>() {
        reuse foo as bar;
        //~^ ERROR: type annotations needed [E0284]
        bar::<u64, i32, 1>(&1, &2);
    }
}

// Testing late-bound lifetimes + types + consts, reusing without user args,
// providing random types with delegation parent generics specified,
// checking signature inheritance, testing mixed order of types and consts
mod test_5 {
    fn foo<'a, 'b, T: Clone, const N: usize, U: Clone>(_t: &'a T, _u: &'b U) {}

    pub fn check<T: Clone, U: Clone>() {
        reuse foo as bar;
        //~^ ERROR: type annotations needed [E0284]
        bar::<u64, 1, u32>(&1, &2);
    }
}

// Testing late-bound lifetimes + types + consts, reusing with user args,
// checking signature inheritance, testing mixed order of types and consts
mod test_6 {
    fn foo<'a, 'b, T: Clone, const N: usize, U: Clone>(_t: &'a T, _u: &'b U) {}

    pub fn check<T: Clone, U: Clone>() {
        reuse foo::<String, 1, String> as bar;
        //~^ ERROR: arguments to this function are incorrect [E0308]
        bar(&"".to_string(), &"".to_string());
        //~^ ERROR: type annotations needed [E0284]
    }
}

// FIXME(fn_delegation): Uncomment this test when impl Traits in function params are supported

// mod test_7 {
//     fn foo<T, U>(t: T, u: U, f: impl FnOnce(T, U) -> U) -> U {
//         f(t, u)
//     }

//     pub fn check() {
//         reuse foo as bar;
//         assert_eq!(bar::<i32, i32>(1, 2, |x, y| y), 2);
//     }
// }

// Testing reuse of local fn with delegation parent generic params specified,
// late-bound lifetimes + types + consts, reusing with user args,
// checking signature inheritance, mixed consts and types ordering
mod test_8 {
    pub fn check<T: Clone, U: Clone>() {
        fn foo<'a, 'b, const N: usize, T: Clone, U: Clone>(_t: &'a T, _u: &'b U) {}

        reuse foo::<1, String, String> as bar;
        //~^ ERROR: arguments to this function are incorrect [E0308]
        bar(&"".to_string(), &"".to_string());
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing reuse of local fn inside closure,
// late-bound lifetimes + types + consts, reusing with user args,
// checking signature inheritance, mixed consts and types ordering
mod test_9 {
    pub fn check<T: Clone, U: Clone>() {
        let closure = || {
            fn foo<'a, 'b, T: Clone, const N: usize, U: Clone>(_t: &'a T, _u: &'b U) {}

            reuse foo::<String, 1, String> as bar;
            //~^ ERROR: arguments to this function are incorrect [E0308]
            bar(&"".to_string(), &"".to_string());
            //~^ ERROR: type annotations needed [E0284]
        };

        closure();
    }
}

pub fn main() {
    test_1::check();
    test_2::check::<i32, String>();
    test_3::check::<i32, String>();
    test_4::check::<i32, String>();
    test_5::check::<i32, String>();
    test_6::check::<i32, String>();
    // test_7::check();
    test_8::check::<i32, String>();
    test_9::check::<String, i32>();
}
