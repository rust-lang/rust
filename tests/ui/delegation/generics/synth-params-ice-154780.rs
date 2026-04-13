//@ run-pass

#![feature(fn_delegation)]

// Almost original ICE with recursive delegation.
mod test_1 {
    pub fn check() {
        fn foo<const N: usize, T, U>(f: impl FnOnce() -> usize) -> usize {
            f()
        }

        reuse foo::<1, String, String> as bar;

        reuse bar as bar2;

        assert_eq!(bar(|| 123), 123);
        assert_eq!(bar2(|| 123), 123);
    }
}

// Test recursive delegations through trait.
mod test_2 {
    fn foo<'a, const B: bool, T, U>(_x: impl Trait<'a, T, B>, f: impl FnOnce() -> usize) -> usize {
        f()
    }

    trait Trait<'a, A, const B: bool> {
        reuse foo;
        reuse foo::<'a, false, (), ()> as bar;
    }

    struct X;
    impl<'a, A, const B: bool> Trait<'a, A, B> for X {}

    reuse <X as Trait>::foo as foo2;
    reuse <X as Trait>::bar as bar2;

    pub fn check() {
        assert_eq!(foo2::<'static, 'static, X, (), true, false, (), ()>(X, || 123), 123);
        assert_eq!(bar2::<'static, X, (), true>(X, || 123), 123);
    }
}

// Testing impl Traits with SelfAndUserSpecified case.
mod test_3 {
    trait Trait<'a, A, const B: bool> {
        fn foo<'b, const B2: bool, T, U>(&self, f: impl FnOnce() -> usize) -> usize {
            f()
        }
    }

    struct X;
    impl<'a, A, const B: bool> Trait<'a, A, B> for X {}

    reuse Trait::foo;
    reuse Trait::<'static, (), true>::foo::<true, (), ()> as bar;

    pub fn check() {
        assert_eq!(foo::<'static, X, (), true, false, (), ()>(&X, || 123), 123);
        assert_eq!(bar::<X>(&X, || 123), 123);
        assert_eq!(bar(&X, || 123), 123);
    }
}

// FIXME(fn_delegation): rename Self generic param in recursive delegations
// mod test_4 {
//     trait Trait<'a, A, const B: bool> {
//         fn foo<'b, const B2: bool, T, U>(&self, f: impl FnOnce() -> usize) -> usize {
//             f()
//         }
//     }

//     struct X;
//     impl<'a, A, const B: bool> Trait<'a, A, B> for X {}

//     reuse Trait::foo;
//     reuse Trait::<'static, (), true>::foo::<true, (), ()> as bar;

//     trait Trait2 {
//         reuse foo;
//         reuse bar;
//     }

//     reuse Trait2::foo as foo2;
//     reuse Trait2::foo::<'static, X, (), true, false, (), ()> as foo3;
//     reuse Trait2::bar as bar2;
//     reuse Trait2::bar::<X> as bar3;

//     pub fn check() {
//         assert_eq!(foo::<'static, X, (), true, false, (), ()>(&X, || 123), 123);
//         assert_eq!(bar::<X>(&X, || 123), 123);
//         assert_eq!(bar(&X, || 123), 123);
//     }
// }

fn main() {
    test_1::check();
    test_2::check();
    test_3::check();
    // test_4::check();
}
