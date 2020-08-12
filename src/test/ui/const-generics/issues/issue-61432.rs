// run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete
#![cfg_attr(min, feature(min_const_generics))]

fn promote<const N: i32>() {
    // works:
    //
    // let n = N;
    // &n;

    &N;
}

fn main() {
    promote::<0>();
}
