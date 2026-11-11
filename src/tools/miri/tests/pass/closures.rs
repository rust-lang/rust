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
    fn once<F: FnOnce(i64)>(f: F) {
        f(2);
    }
    let mut y = 1;
    {
        let f = |x| y += x;
        once(f);
    }
    y
}

fn fn_once_closure_with_multiple_args() -> i64 {
    fn once<F: FnOnce(i64, i64) -> i64>(f: F) -> i64 {
        f(2, 3)
    }
    let y = 1;
    {
        let f = |x, z| x + y + z;
        once(f)
    }
}

fn boxed_fn_once(f: Box<dyn FnOnce() -> i32>) -> i32 {
    f()
}

fn box_dyn() {
    let x: Box<dyn Fn(i32) -> i32> = Box::new(|x| x * 2);
    assert_eq!(x(21), 42);
    let mut i = 5;
    {
        let mut x: Box<dyn FnMut()> = Box::new(|| i *= 2);
        x();
        x();
    }
    assert_eq!(i, 20);
}

fn fn_item_as_closure_trait_object() {
    fn foo() {}
    let f: &dyn Fn() = &foo;
    f();
}

fn fn_item_with_args_as_closure_trait_object() {
    fn foo(i: i32) {
        assert_eq!(i, 42);
    }
    let f: &dyn Fn(i32) = &foo;
    f(42);
}

fn fn_item_with_multiple_args_as_closure_trait_object() {
    fn foo(i: i32, j: i32) {
        assert_eq!(i, 42);
        assert_eq!(j, 55);
    }

    fn bar(i: i32, j: i32, k: f32) {
        assert_eq!(i, 42);
        assert_eq!(j, 55);
        assert_eq!(k, 3.14159)
    }
    let f: &dyn Fn(i32, i32) = &foo;
    f(42, 55);
    let f: &dyn Fn(i32, i32, f32) = &bar;
    f(42, 55, 3.14159);
}

fn fn_ptr_as_closure_trait_object() {
    fn foo() {}
    fn bar(u: u32) {
        assert_eq!(u, 42);
    }
    fn baa(u: u32, f: f32) {
        assert_eq!(u, 42);
        assert_eq!(f, 3.141);
    }
    let f: &dyn Fn() = &(foo as fn());
    f();
    let f: &dyn Fn(u32) = &(bar as fn(u32));
    f(42);
    let f: &dyn Fn(u32, f32) = &(baa as fn(u32, f32));
    f(42, 3.141);
}

fn main() {
    assert_eq!(simple(), 12);
    assert_eq!(crazy_closure(), (84, 10, 10));
    assert_eq!(closure_arg_adjustment_problem(), 3);
    assert_eq!(fn_once_closure_with_multiple_args(), 6);
    assert_eq!(
        boxed_fn_once(Box::new({
            let x = 13;
            move || x
        })),
        13,
    );

    box_dyn();
    fn_item_as_closure_trait_object();
    fn_item_with_args_as_closure_trait_object();
    fn_item_with_multiple_args_as_closure_trait_object();
    fn_ptr_as_closure_trait_object();
}
