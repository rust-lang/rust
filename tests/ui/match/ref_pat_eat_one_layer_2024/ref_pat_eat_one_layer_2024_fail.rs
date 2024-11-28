//@ edition: 2024
//@ revisions: classic structural both
#![allow(incomplete_features)]
#![cfg_attr(any(classic, both), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural, both), feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let Some(&mut Some(&_)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&mut _)) = &Some(&mut Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(x)) = &mut Some(&Some(0)) {
        let _: &mut u32 = x;
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&mut _)) = &mut Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(Some((&mut _)))) = &Some(Some(&mut Some(0))) {
        //~^ ERROR: mismatched types
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
    }

    let &mut _ = &&0;
    //~^ ERROR: mismatched types

    let &mut _ = &&&&&&&&&&&&&&&&&&&&&&&&&&&&0;
    //~^ ERROR: mismatched types

    if let Some(&mut Some(&_)) = &Some(&mut Some(0)) {
        //[classic]~^ ERROR: mismatched types
    }

    if let Some(Some(&mut x)) = &Some(Some(&mut 0)) {
        //[classic]~^ ERROR: mismatched types
    }

    let &mut _ = &&mut 0;
    //~^ ERROR: mismatched types

    let &mut _ = &&&&&&&&&&&&&&&&&&&&&&&&&&&&mut 0;
    //~^ ERROR: mismatched types

    let &mut &mut &mut &mut _ = &mut &&&&mut &&&mut &mut 0;
    //~^ ERROR: mismatched types

    if let Some(&mut _) = &mut Some(&0) {
        //[structural]~^ ERROR
    }

    struct Foo(u8);

    let Foo(mut a) = &Foo(0);
    //~^ ERROR: binding cannot be both mutable and by-reference
    a = &42;

    let Foo(mut a) = &mut Foo(0);
    //~^ ERROR: binding cannot be both mutable and by-reference
    a = &mut 42;

    fn generic<R: Ref>() -> R {
        R::meow()
    }

    trait Ref: Sized {
        fn meow() -> Self;
    }

    impl Ref for &'static mut [(); 0] {
        fn meow() -> Self {
            &mut []
        }
    }

    let &_ = generic(); //~ERROR: the trait bound `&_: main::Ref` is not satisfied [E0277]
}
