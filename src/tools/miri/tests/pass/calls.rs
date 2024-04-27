fn call() -> i32 {
    fn increment(x: i32) -> i32 {
        x + 1
    }
    increment(1)
}

fn factorial_recursive() -> i64 {
    fn fact(n: i64) -> i64 {
        if n == 0 { 1 } else { n * fact(n - 1) }
    }
    fact(10)
}

fn call_generic() -> (i16, bool) {
    fn id<T>(t: T) -> T {
        t
    }
    (id(42), id(true))
}

// Test calling a very simple function from the standard library.
fn cross_crate_fn_call() -> i64 {
    if 1i32.is_positive() { 1 } else { 0 }
}

const fn foo(i: i64) -> i64 {
    *&i + 1
}

fn const_fn_call() -> i64 {
    let x = 5 + foo(5);
    assert_eq!(x, 11);
    x
}

fn call_return_into_passed_reference() {
    pub fn func<T>(v: &mut T, f: fn(&T) -> T) {
        // MIR building will introduce a temporary, so this becomes
        // `let temp = f(v); *v = temp;`.
        // If this got optimized to `*v = f(v)` on the MIR level we'd have UB
        // since the return place may not be observed while the function runs!
        *v = f(v);
    }

    let mut x = 0;
    func(&mut x, |v| v + 1);
    assert_eq!(x, 1);
}

fn main() {
    assert_eq!(call(), 2);
    assert_eq!(factorial_recursive(), 3628800);
    assert_eq!(call_generic(), (42, true));
    assert_eq!(cross_crate_fn_call(), 1);
    assert_eq!(const_fn_call(), 11);

    call_return_into_passed_reference();
}
