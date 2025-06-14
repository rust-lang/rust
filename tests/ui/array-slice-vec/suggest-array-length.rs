//@ run-rustfix
#![allow(unused_variables, dead_code, non_upper_case_globals)]

fn main() {
    const Foo: [i32; _] = [1, 2, 3];
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
    const REF_FOO: &[u8; _] = &[1];
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
    static Statik: [i32; _] = [1, 2, 3];
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables
    static REF_STATIK: &[u8; _] = &[1];
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables
    let foo: [i32; _] = [1, 2, 3];
    let bar: [i32; _] = [0; 3];
    let ref_foo: &[i32; _] = &[1, 2, 3];
    let ref_bar: &[i32; _] = &[0; 3];
    let multiple_ref_foo: &&[i32; _] = &&[1, 2, 3];
}
