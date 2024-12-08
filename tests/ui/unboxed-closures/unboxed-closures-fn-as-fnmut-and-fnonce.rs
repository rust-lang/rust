//@ run-pass
// Checks that the Fn trait hierarchy rules permit
// any Fn trait to be used where Fn is implemented.

#![feature(unboxed_closures, fn_traits)]

struct S;

impl Fn<(i32,)> for S {
    extern "rust-call" fn call(&self, (x,): (i32,)) -> i32 {
        x * x
    }
}

impl FnMut<(i32,)> for S {
    extern "rust-call" fn call_mut(&mut self, args: (i32,)) -> i32 { self.call(args) }
}

impl FnOnce<(i32,)> for S {
    type Output = i32;
    extern "rust-call" fn call_once(self, args: (i32,)) -> i32 { self.call(args) }
}

fn call_it<F:Fn(i32)->i32>(f: &F, x: i32) -> i32 {
    f(x)
}

fn call_it_mut<F:FnMut(i32)->i32>(f: &mut F, x: i32) -> i32 {
    f(x)
}

fn call_it_once<F:FnOnce(i32)->i32>(f: F, x: i32) -> i32 {
    f(x)
}

fn main() {
    let x = call_it(&S, 22);
    let y = call_it_mut(&mut S, 22);
    let z = call_it_once(S, 22);
    assert_eq!(x, y);
    assert_eq!(y, z);
}
