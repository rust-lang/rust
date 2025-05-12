//@ edition: 2024
//@ revisions: with_impl without_impl
//@[with_impl] run-pass
//! Sanity check that experimental new pattern typing rules work as expected with trait selection

fn main() {
    fn generic<R: Ref>() -> (R, bool) {
        R::meow()
    }

    trait Ref: Sized {
        fn meow() -> (Self, bool);
    }

    #[cfg(with_impl)]
    impl Ref for &'static [(); 0] {
        fn meow() -> (Self, bool) {
            (&[], false)
        }
    }

    impl Ref for &'static mut [(); 0] {
        fn meow() -> (Self, bool) {
            (&mut [], true)
        }
    }

    let (&_, b) = generic(); //[without_impl]~ ERROR: the trait bound `&_: main::Ref` is not satisfied [E0277]
    assert!(!b);
}
