//@ edition: 2024
//@ compile-flags: -Zunstable-options
#![allow(incomplete_features)]
#![feature(ref_pat_eat_one_layer_2024)]

pub fn main() {
    if let Some(&mut Some(&_)) = &Some(&Some(0)) {
        //~^ ERROR: cannot match inherited `&` with `&mut` pattern
    }
    if let Some(&Some(&mut _)) = &Some(&mut Some(0)) {
        //~^ ERROR: cannot match inherited `&` with `&mut` pattern
    }
    if let Some(&Some(x)) = &mut Some(&Some(0)) {
        let _: &mut u32 = x;
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&mut _)) = &mut Some(&Some(0)) {
        //~^ ERROR: cannot match inherited `&` with `&mut` pattern
    }
    if let Some(&Some(Some((&mut _)))) = &Some(Some(&mut Some(0))) {
        //~^ ERROR: cannot match inherited `&` with `&mut` pattern
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: cannot match inherited `&` with `&mut` pattern
    }
    if let Some(&mut Some(x)) = &Some(Some(0)) {
        //~^ ERROR: cannot match inherited `&` with `&mut` pattern
    }

    let &mut _ = &&0;
    //~^ ERROR: mismatched types

    let &mut _ = &&&&&&&&&&&&&&&&&&&&&&&&&&&&0;
    //~^ ERROR: mismatched types

    if let Some(&mut Some(&_)) = &Some(&mut Some(0)) {
        //~^ ERROR: cannot match inherited `&` with `&mut` pattern
    }

    if let Some(Some(&mut x)) = &Some(Some(&mut 0)) {
        //~^ ERROR: cannot match inherited `&` with `&mut` pattern
    }

    let &mut _ = &&mut 0;
    //~^ ERROR: mismatched types

    let &mut _ = &&&&&&&&&&&&&&&&&&&&&&&&&&&&mut 0;
    //~^ ERROR: mismatched types

    let &mut &mut &mut &mut _ = &mut &&&&mut &&&mut &mut 0;
    //~^ ERROR: mismatched types

    struct Foo(u8);

    let Foo(mut a) = &Foo(0);
    //~^ ERROR: binding cannot be both mutable and by-reference
    a = &42;

    let Foo(mut a) = &mut Foo(0);
    //~^ ERROR: binding cannot be both mutable and by-reference
    a = &mut 42;
}
