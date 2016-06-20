fn simple() -> i32 {
    let y = 10;
    let f = |x| x + y;
    f(2)
}

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

// TODO(solson): Implement closure argument adjustment and uncomment this test.
// fn closure_arg_adjustment_problem() -> i64 {
//     fn once<F: FnOnce(i64)>(f: F) { f(2); }
//     let mut y = 1;
//     {
//         let f = |x| y += x;
//         once(f);
//     }
//     y
// }

fn main() {
    assert_eq!(simple(), 12);
    assert_eq!(crazy_closure(), (84, 10, 10));
}
