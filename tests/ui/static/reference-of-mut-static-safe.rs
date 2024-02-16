//@ revisions: e2021 e2024

//@ [e2021] edition:2021
//@ [e2024] compile-flags: --edition 2024 -Z unstable-options

fn main() {
    static mut X: i32 = 1;

    let _x = &X;
    //[e2024]~^ reference of mutable static is disallowed [E0796]
    //[e2021]~^^ use of mutable static is unsafe and requires unsafe function or block [E0133]
    //[e2021]~^^^ shared reference of mutable static is discouraged [static_mut_ref]
}
