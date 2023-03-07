// run-pass
#![feature(fn_traits)]

fn main() {
    let mut zero = || 0;
    let x = zero.call_mut(());
    assert_eq!(x, 0);
}
