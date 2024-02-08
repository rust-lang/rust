//@aux-build:proc_macro_derive.rs
#![warn(clippy::ignored_unit_patterns)]
#![allow(
    clippy::let_unit_value,
    clippy::redundant_pattern_matching,
    clippy::single_match,
    clippy::needless_borrow
)]

fn foo() -> Result<(), ()> {
    unimplemented!()
}

fn main() {
    match foo() {
        Ok(_) => {},  //~ ERROR: matching over `()` is more explicit
        Err(_) => {}, //~ ERROR: matching over `()` is more explicit
    }
    if let Ok(_) = foo() {}
    //~^ ERROR: matching over `()` is more explicit
    let _ = foo().map_err(|_| todo!());
    //~^ ERROR: matching over `()` is more explicit

    println!(
        "{:?}",
        match foo() {
            Ok(_) => {},
            //~^ ERROR: matching over `()` is more explicit
            Err(_) => {},
            //~^ ERROR: matching over `()` is more explicit
        }
    );
}

// ignored_unit_patterns in derive macro should be ok
#[derive(proc_macro_derive::StructIgnoredUnitPattern)]
pub struct B;

#[allow(unused)]
pub fn moo(_: ()) {
    let _ = foo().unwrap();
    //~^ ERROR: matching over `()` is more explicit
    let _: () = foo().unwrap();
    let _: () = ();
}

fn test_unit_ref_1() {
    let x: (usize, &&&&&()) = (1, &&&&&&());
    match x {
        (1, _) => unimplemented!(),
        //~^ ERROR: matching over `()` is more explicit
        _ => unimplemented!(),
    };
}

fn test_unit_ref_2(v: &[(usize, ())]) {
    for (x, _) in v {
        //~^ ERROR: matching over `()` is more explicit
        let _ = x;
    }
}
