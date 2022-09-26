// run-rustfix
#![allow(unused_variables, dead_code, non_upper_case_globals)]

fn main() {
    const Foo: [i32; 3] = [1, 2, 3];
    //~^ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    //~| ERROR using `_` for array lengths is unstable
    const REF_FOO: &[u8; 1] = &[1];
    //~^ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    //~| ERROR using `_` for array lengths is unstable
    let foo: [i32; 3] = [1, 2, 3];
    //~^ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    //~| ERROR using `_` for array lengths is unstable
    let bar: [i32; 3] = [0; 3];
    //~^ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    //~| ERROR using `_` for array lengths is unstable
    let ref_foo: &[i32; 3] = &[1, 2, 3];
    //~^ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    //~| ERROR using `_` for array lengths is unstable
    let ref_bar: &[i32; 3] = &[0; 3];
    //~^ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    //~| ERROR using `_` for array lengths is unstable
    let multiple_ref_foo: &&[i32; 3] = &&[1, 2, 3];
    //~^ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    //~| ERROR using `_` for array lengths is unstable
}
