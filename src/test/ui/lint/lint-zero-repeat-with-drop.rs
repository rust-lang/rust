#![feature(const_generics)]
#![allow(incomplete_features)]
#![deny(zero_repeat_with_drop)]

const ZERO: usize = 1 * 0;

const fn make_val<T>(val: T) -> T { val }

struct NoDropOrCopy;
struct WithDropGlue(String);

fn foo<T, V: Copy, F: Fn() -> T, const N: usize>(not_copy: F, copy: V) {
    // All of these should triger the lint
    let _ = [not_copy(); 0]; //~ ERROR used a type
    let _ = [not_copy(); 1 - 1]; //~ ERROR used a type
    let _ = [not_copy(); ZERO]; //~ ERROR used a type
    let _ = [Some(not_copy()); 0]; //~ ERROR used a type
    let _ = [None::<T>; 0]; //~ ERROR used a type
    let _ = [make_val(not_copy()); 0]; //~ ERROR used a type
    let _ = [String::new(); 0]; //~ ERROR used a type
    let _ = [WithDropGlue(String::new()); 0]; //~ ERROR used a type

    // None of these should trigger the lint
    let _ = [copy; 0];
    let _ = [Some(copy); 0];
    let _ = [NoDropOrCopy; 0];
    let _ = [not_copy(); 1];
    let _ = [copy; N];
}

fn allow_it() {
    #[allow(zero_repeat_with_drop)]
    let _ = [WithDropGlue(String::new()); 1 - 1];
}

fn main() {}
