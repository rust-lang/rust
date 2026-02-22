#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(late_bound_lifetime_arguments)]

//! This is one of the mapping tests, which tests mapping of delegee parent and child
//! generic params, whose main goal is to create cases with
//! different number of lifetimes/types/consts in delegee child and parent; and in
//! delegation parent if applicable. At some tests predicates are
//! added. At some tests user-specified args are specified in reuse statement.

// Testing lifetimes + types/consts in child, lifetimes + types/consts in delegation parent,
// with(out) user-specified args
mod test_1 {
    fn foo<'a: 'a, 'b: 'b, T: Clone + ToString, U: Clone, const N: usize>() {}

    trait Trait<'a, A, B, C, const N: usize> {
        reuse foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse foo::<'static, 'static, i32, String, 1> as bar;
    }

    impl Trait<'static, i32, i32, i32, 1> for u32 {}
    pub fn check() {
        <u32 as Trait<'static, i32, i32, i32, 1>>::foo::<'static, 'static, i32, String, 1>();
        <u32 as Trait<'static, i32, i32, i32, 1>>::bar();
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing types/consts in child, lifetimes + types/consts in delegation parent,
// with(out) user-specified args
mod test_2 {
    fn foo<T: Clone, U: Clone, const N: usize>() {}

    trait Trait<'a, A, B, C, const N: usize> {
        reuse foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse foo::<i32, String, 1> as bar;
    }

    impl Trait<'static, i32, i32, i32, 1> for u32 {}
    pub fn check() {
        <u32 as Trait<'static, i32, i32, i32, 1>>::foo::<i32, String, 1>();
        <u32 as Trait<'static, i32, i32, i32, 1>>::bar();
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing none in child, lifetimes + types/consts in delegation parent,
// with(out) user-specified args
mod test_3 {
    fn foo() {}

    trait Trait<'a, A, B, C, const N: usize> {
        reuse foo;
    }

    impl Trait<'static, i32, i32, i32, 1> for u32 {}
    pub fn check() {
        <u32 as Trait<'static, i32, i32, i32, 1>>::foo();
    }
}

// Testing lifetimes + types/consts in child, types/consts in delegation parent,
// with(out) user-specified args
mod test_4 {
    fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

    trait Trait<A, B, C, const N: usize> {
        reuse foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse foo::<'static, 'static, i32, String, 1> as bar;
    }

    impl Trait<i32, i32, i32, 1> for u32 {}
    pub fn check() {
        <u32 as Trait<i32, i32, i32, 1>>::foo::<'static, 'static, i32, String, 1>();
        <u32 as Trait<i32, i32, i32, 1>>::bar();
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing lifetimes + types/consts in child, none in delegation parent,
// with(out) user-specified args
mod test_5 {
    fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

    trait Trait {
        reuse foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse foo::<'static, 'static, i32, String, 1> as bar;
    }

    impl Trait for u32 {}
    pub fn check() {
        <u32 as Trait>::foo::<'static, 'static, i32, String, 1>();
        <u32 as Trait>::bar();
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing types/consts in child, none in delegation parent, with(out) user-specified args
mod test_6 {
    fn foo<T: Clone, U: Clone, const N: usize>() {}

    trait Trait {
        reuse foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse foo::<i32, String, 1> as bar;
    }

    impl Trait for u32 {}
    pub fn check() {
        <u32 as Trait>::foo::<i32, String, 1>();
        <u32 as Trait>::bar();
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing none in child, none in delegation parent, with(out) user-specified args
mod test_7 {
    fn foo() {}

    trait Trait {
        reuse foo;
    }

    impl Trait for u32 {}
    pub fn check() {
        <u32 as Trait>::foo();
    }
}

pub fn main() {
    test_1::check();
    test_2::check();
    test_3::check();
    test_4::check();
    test_5::check();
    test_6::check();
    test_7::check();
}
