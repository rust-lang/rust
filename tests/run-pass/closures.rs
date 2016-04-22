#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn simple() -> i32 {
    let y = 10;
    let f = |x| x + y;
    f(2)
}

#[miri_run]
fn crazy_closure() -> (i32, i32, i32) {
    fn inner<T: Copy>(t: T) -> (i32, T, T) {
        struct NonCopy;
        let x = NonCopy;

        let a = 2;
        let b = 40;
        let f = move |y, z, asdf| {
            drop(x);
            (a + b + y + z, asdf, t)
        };
        f(a, b, t)
    }

    inner(10)
}

// #[miri_run]
// fn closure_arg_adjustment_problem() -> i64 {
//     fn once<F: FnOnce(i64)>(f: F) { f(2); }
//     let mut y = 1;
//     {
//         let f = |x| y += x;
//         once(f);
//     }
//     y
// }

fn main() {}
