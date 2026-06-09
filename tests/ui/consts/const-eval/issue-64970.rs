//@ run-pass

fn main() {
    foo(10);
}

fn foo(mut n: i32) {
    if false {
        n = 0i32;
    }

    if n > 0i32 {
        let _ = 1i32 / n;
    }
}
