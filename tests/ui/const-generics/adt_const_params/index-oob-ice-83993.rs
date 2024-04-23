// issue: rust-lang/rust/#83993

#![feature(adt_const_params)]
//~^ WARN the feature `adt_const_params` is incomplete and may not be safe to use and/or cause compiler crashes
fn bug<'a>()
where
    for<'b> [(); {
        let x: &'b ();
        //~^ ERROR generic parameters may not be used in const operations
        0
    }]:
{}

fn bad() where for<'b> [();{let _:&'b (); 0}]: Sized { }
//~^ ERROR generic parameters may not be used in const operations
fn good() where for<'b> [();{0}]: Sized { }

pub fn main() {}
