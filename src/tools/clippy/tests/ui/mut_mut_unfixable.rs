//@no-rustfix

#![warn(clippy::mut_mut)]
#![allow(unused)]
#![expect(clippy::no_effect)]

//! removing the extra `&mut`s will break the derefs

fn fun(x: &mut &mut u32) -> bool {
    //~^ mut_mut
    **x > 0
}

fn main() {
    let mut x = &mut &mut 1u32;
    //~^ mut_mut
    {
        let mut y = &mut x;
        //~^ mut_mut
        ***y + **x;
    }

    if fun(x) {
        let y = &mut &mut 2;
        //~^ mut_mut
        **y + **x;
    }

    if fun(x) {
        let y = &mut &mut &mut 2;
        //~^ mut_mut
        ***y + **x;
    }

    if fun(x) {
        // The lint will remove the extra `&mut`, but the result will still be a `&mut` of an expr
        // of type `&mut _` (x), so the lint will fire again. That's because we've decided that
        // doing both fixes in one run is not worth it, given how improbable code like this is.
        let y = &mut &mut x;
        //~^ mut_mut
    }
}
