//@ run-pass
// Checks that the Fn trait hierarchy rules permit
// FnMut or FnOnce to be used where FnMut is implemented.

#![feature(unboxed_closures, fn_traits)]

struct S;

impl FnMut<(i32,)> for S {
    extern "rust-call" fn call_mut(&mut self, (x,): (i32,)) -> i32 {
        x * x
    }
}

impl FnOnce<(i32,)> for S {
    type Output = i32;

    extern "rust-call" fn call_once(mut self, args: (i32,)) -> i32 { self.call_mut(args) }
}

fn call_it_mut<F:FnMut(i32)->i32>(f: &mut F, x: i32) -> i32 {
    f(x)
}

fn call_it_once<F:FnOnce(i32)->i32>(f: F, x: i32) -> i32 {
    f(x)
}

fn main() {
    let y = call_it_mut(&mut S, 22);
    let z = call_it_once(S, 22);
    assert_eq!(y, z);
}
