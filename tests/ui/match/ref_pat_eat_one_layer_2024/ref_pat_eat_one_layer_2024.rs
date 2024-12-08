//@ run-pass
//@ edition: 2024
//@ revisions: classic structural both
#![allow(incomplete_features)]
#![cfg_attr(any(classic, both), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural, both), feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let Some(Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&x)) = &Some(Some(&0)) {
        let _: &u32 = x;
    }
    if let Some(Some(&&x)) = &Some(Some(&0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &Some(Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&mut x)) = &mut Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &mut Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &mut Some(&Some(0)) {
        let _: &u32 = x;
    }
    if let Some(&Some(&mut ref x)) = Some(&Some(&mut 0)) {
        let _: &u32 = x;
    }
    if let &Some(Some(x)) = &Some(&mut Some(0)) {
        let _: &u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(Some(&x))) = &Some(Some(&mut Some(0))) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = Some(&Some(&mut 0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &mut Some(Some(0)) {
        let _: u32 = x;
    }
    #[cfg(any(classic, both))]
    if let Some(&mut x) = &mut Some(&0) {
        let _: &u32 = x;
    }
    #[cfg(any(structural, both))]
    if let Some(&mut x) = &Some(&mut 0) {
        let _: &u32 = x;
    }

    fn generic<R: Ref>() -> (R, bool) {
        R::meow()
    }

    trait Ref: Sized {
        fn meow() -> (Self, bool);
    }

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

    let (&_, b) = generic();
    assert!(!b);
}
