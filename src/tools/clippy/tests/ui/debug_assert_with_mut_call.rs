#![feature(custom_inner_attributes)]
#![rustfmt::skip]
#![warn(clippy::debug_assert_with_mut_call)]
#![allow(clippy::redundant_closure_call, clippy::get_first)]


struct S;

impl S {
    fn bool_self_ref(&self) -> bool { false }
    fn bool_self_mut(&mut self) -> bool { false }
    fn bool_self_ref_arg_ref(&self, _: &u32) -> bool { false }
    fn bool_self_ref_arg_mut(&self, _: &mut u32) -> bool { false }
    fn bool_self_mut_arg_ref(&mut self, _: &u32) -> bool { false }
    fn bool_self_mut_arg_mut(&mut self, _: &mut u32) -> bool { false }

    fn u32_self_ref(&self) -> u32 { 0 }
    fn u32_self_mut(&mut self) -> u32 { 0 }
    fn u32_self_ref_arg_ref(&self, _: &u32) -> u32 { 0 }
    fn u32_self_ref_arg_mut(&self, _: &mut u32) -> u32 { 0 }
    fn u32_self_mut_arg_ref(&mut self, _: &u32) -> u32 { 0 }
    fn u32_self_mut_arg_mut(&mut self, _: &mut u32) -> u32 { 0 }
}

fn bool_ref(_: &u32) -> bool { false }
fn bool_mut(_: &mut u32) -> bool { false }
fn u32_ref(_: &u32) -> u32 { 0 }
fn u32_mut(_: &mut u32) -> u32 { 0 }

fn func_non_mutable() {
    debug_assert!(bool_ref(&3));
    debug_assert!(!bool_ref(&3));

    debug_assert_eq!(0, u32_ref(&3));
    debug_assert_eq!(u32_ref(&3), 0);

    debug_assert_ne!(1, u32_ref(&3));
    debug_assert_ne!(u32_ref(&3), 1);
}

fn func_mutable() {
    debug_assert!(bool_mut(&mut 3));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`
    //~| NOTE: `-D clippy::debug-assert-with-mut-call` implied by `-D warnings`
    debug_assert!(!bool_mut(&mut 3));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`

    debug_assert_eq!(0, u32_mut(&mut 3));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_eq!`
    debug_assert_eq!(u32_mut(&mut 3), 0);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_eq!`

    debug_assert_ne!(1, u32_mut(&mut 3));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_ne!`
    debug_assert_ne!(u32_mut(&mut 3), 1);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_ne!`
}

fn method_non_mutable() {
    debug_assert!(S.bool_self_ref());
    debug_assert!(S.bool_self_ref_arg_ref(&3));

    debug_assert_eq!(S.u32_self_ref(), 0);
    debug_assert_eq!(S.u32_self_ref_arg_ref(&3), 0);

    debug_assert_ne!(S.u32_self_ref(), 1);
    debug_assert_ne!(S.u32_self_ref_arg_ref(&3), 1);
}

fn method_mutable() {
    debug_assert!(S.bool_self_mut());
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`
    debug_assert!(!S.bool_self_mut());
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`
    debug_assert!(S.bool_self_ref_arg_mut(&mut 3));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`
    debug_assert!(S.bool_self_mut_arg_ref(&3));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`
    debug_assert!(S.bool_self_mut_arg_mut(&mut 3));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`

    debug_assert_eq!(S.u32_self_mut(), 0);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_eq!`
    debug_assert_eq!(S.u32_self_mut_arg_ref(&3), 0);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_eq!`
    debug_assert_eq!(S.u32_self_ref_arg_mut(&mut 3), 0);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_eq!`
    debug_assert_eq!(S.u32_self_mut_arg_mut(&mut 3), 0);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_eq!`

    debug_assert_ne!(S.u32_self_mut(), 1);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_ne!`
    debug_assert_ne!(S.u32_self_mut_arg_ref(&3), 1);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_ne!`
    debug_assert_ne!(S.u32_self_ref_arg_mut(&mut 3), 1);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_ne!`
    debug_assert_ne!(S.u32_self_mut_arg_mut(&mut 3), 1);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_ne!`
}

fn misc() {
    // with variable
    let mut v: Vec<u32> = vec![1, 2, 3, 4];
    debug_assert_eq!(v.get(0), Some(&1));
    debug_assert_ne!(v[0], 2);
    debug_assert_eq!(v.pop(), Some(1));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_eq!`
    debug_assert_ne!(Some(3), v.pop());
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_ne!`

    let a = &mut 3;
    debug_assert!(bool_mut(a));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`

    // nested
    debug_assert!(!(bool_ref(&u32_mut(&mut 3))));
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`

    // chained
    debug_assert_eq!(v.pop().unwrap(), 3);
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert_eq!`

    // format args
    debug_assert!(bool_ref(&3), "w/o format");
    debug_assert!(bool_mut(&mut 3), "w/o format");
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`
    debug_assert!(bool_ref(&3), "{} format", "w/");
    debug_assert!(bool_mut(&mut 3), "{} format", "w/");
    //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!`

    // sub block
    let mut x = 42_u32;
    debug_assert!({
        bool_mut(&mut x);
        //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!
        x > 10
    });

    // closures
    debug_assert!((|| {
        let mut x = 42;
        bool_mut(&mut x);
        //~^ ERROR: do not call a function with mutable arguments inside of `debug_assert!
        x > 10
    })());
}

async fn debug_await() {
    debug_assert!(async {
        true
    }.await);
}

fn main() {
    func_non_mutable();
    func_mutable();
    method_non_mutable();
    method_mutable();

    misc();
    debug_await();
}
