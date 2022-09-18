// run-pass
// issue #101932

#![cfg_attr(bootstrap, feature(let_else))]

const fn foo(a: Option<i32>) -> i32 {
    let Some(a) = a else {
        return 42
    };

    a + 1
}

fn main() {
    const A: i32 = foo(None);
    const B: i32 = foo(Some(1));

    println!("{} {}", A, B);
}
