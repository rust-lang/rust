// compile-pass
// FIXME(#54943) This test targets the scenario where proving the WF requirements of a user
// type annotation requires checking dead code. This test should actually fail to compile.

#![feature(nll)]
#![allow(warnings)]

fn foo<T: 'static>() { }

fn boo<'a>() {
    return;

    let x = foo::<&'a u32>();
    //~^ ERROR the type `&'a u32` does not fulfill the required lifetime [E0477]
}

fn main() {}
