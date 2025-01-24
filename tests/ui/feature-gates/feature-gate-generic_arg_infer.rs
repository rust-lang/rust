//@ [feature] run-pass
//@ revisions: normal feature

#![cfg_attr(feature, feature(generic_arg_infer))]

fn foo<const N: usize>(_: [u8; N]) -> [u8; N] {
    [0; N]
}

fn bar() {
    let _x: [u8; 3] = [0; _];
    //[normal]~^ ERROR: using `_` for array lengths is unstable
    let _y: [u8; _] = [0; 3];
    //[normal]~^ ERROR: using `_` for array lengths is unstable
}

fn main() {
    let _x = foo::<_>([1, 2]);
    //[normal]~^ ERROR: const arguments cannot yet be inferred with `_`
    bar();
}
