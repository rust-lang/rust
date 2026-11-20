//@ run-pass

#![feature(unboxed_closures, fn_traits)]

struct S {
    x: i32,
    y: i32,
}

impl FnMut<()> for S {
    extern "rust-call" fn call_mut(&mut self, (): ()) -> i32 {
        self.x * self.y
    }
}

impl FnOnce<()> for S {
    type Output = i32;
    extern "rust-call" fn call_once(mut self, args: ()) -> i32 { self.call_mut(args) }
}

fn main() {
    let mut s = S {
        x: 3,
        y: 3,
    };
    let ans = s();
    assert_eq!(ans, 9);
}
