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

fn closure_arg_adjustment_problem() -> i64 {
    fn once<F: FnOnce(i64)>(f: F) { f(2); }
    let mut y = 1;
    {
        let f = |x| y += x;
        once(f);
    }
    y
}

fn fn_once_closure_with_multiple_args() -> i64 {
    fn once<F: FnOnce(i64, i64) -> i64>(f: F) -> i64 { f(2, 3) }
    let y = 1;
    {
        let f = |x, z| x + y + z;
        once(f)
    }
}

fn boxed(f: Box<dyn FnOnce() -> i32>) -> i32 {
    f()
}

fn main() {
    assert_eq!(simple(), 12);
    assert_eq!(crazy_closure(), (84, 10, 10));
    assert_eq!(closure_arg_adjustment_problem(), 3);
    assert_eq!(fn_once_closure_with_multiple_args(), 6);
    assert_eq!(boxed(Box::new({let x = 13; move || x})), 13);
}
